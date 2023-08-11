import argparse
import math
import random
import os
# import sys
# sys.path.append('..')
from torch.nn.utils import clip_grad_norm_
import os.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from math import exp
from torch.autograd import Variable

from models.stylegan2_seg import GeneratorSeg, Discriminator, MultiscaleDiscriminator, GANLoss
from dataloader.dataset import CelebAMaskDataset
# from models.PAR import PAR

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

import functools
from utils.inception_utils import sample_gema, prepare_inception_metrics
import cv2
import random

from models.vggNet import VGGFeatureExtractor
from models import lpips
from models.encoder_model import FPNEncoder, ResEncoder
from semanticGAN.losses import SoftmaxLoss, SoftBinaryCrossEntropyLoss, DiceLoss
from PIL import Image
from models.mask_generator_256 import BatchRandomMask
from models.pcp import PerceptualLoss
import torchvision
import matplotlib.pyplot as plt

def data_sampler(dataset, shuffle, distributed):
    if distributed:    
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .to('cpu')
            .numpy()
    )

def mask2rgb(args, mask):
    if args.seg_name == 'celeba-mask':
        color_table = torch.tensor(
            [[0, 0, 0],
             [0, 0, 205],
             # [132, 112, 255],
             # [25, 25, 112],
             # [187, 255, 255],
             # [102, 205, 170],
             # [227, 207, 87],
             # [142, 142, 56]
            ], dtype=torch.float)
    else:
        raise Exception('No such a dataloader!')

    rgb_tensor = F.embedding(mask, color_table).permute(0, 3, 1, 2)
    return rgb_tensor

def denormalize_img(imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:,0,:,:] = imgs[:,0,:,:] * std[0] + mean[0]
    _imgs[:,1,:,:] = imgs[:,1,:,:] * std[1] + mean[1]
    _imgs[:,2,:,:] = imgs[:,2,:,:] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    _imgs_norm = _imgs / 255.0

    return _imgs_norm

def compute_gram(x):
    if len(x.shape) == 4:
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2) # (b, w*h, ch)
        G = f.bmm(f_T) / (h * w * ch)
    else:
        b, wh, ch = x.size()
        f = x.transpose(1, 2)
        f_T = x
        G = f.bmm(f_T) / (wh * ch)
    return G

def calculate_style_loss(encoder, x, x_mask, y, y_mask):
    criterionIdt = torch.nn.L1Loss().to(x.device)
    x_encoder, y_encoder = encoder(x, x_mask), encoder(y, y_mask)

    style_loss = 0.0
    style_loss += criterionIdt(compute_gram(x_encoder['conv_list']), compute_gram(y_encoder['conv_list']))
    style_loss += criterionIdt(compute_gram(x_encoder['trans_0']), compute_gram(y_encoder['trans_0']))
    style_loss += criterionIdt(compute_gram(x_encoder['trans_1']), compute_gram(y_encoder['trans_1']))
    style_loss += criterionIdt(compute_gram(x_encoder['trans_2']), compute_gram(y_encoder['trans_2']))

    percept_loss = 0.0
    percept_loss += criterionIdt(x_encoder['conv_list'], y_encoder['conv_list'])
    percept_loss += criterionIdt(x_encoder['trans_0'], y_encoder['trans_0'])
    percept_loss += criterionIdt(x_encoder['trans_1'], y_encoder['trans_1'])
    percept_loss += criterionIdt(x_encoder['trans_2'], y_encoder['trans_2'])

    return style_loss, percept_loss

class Encoder_style(torch.nn.Module):
    def __init__(self, load_encoder):
        super(Encoder_style, self).__init__()

        self.encoder = load_encoder
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, x_mask):
        feature_list = self.encoder(x, x_mask)

        return feature_list


def batch_overlay(args, img_tensor, mask_tensor, alpha=0.3):
    b = img_tensor.shape[0]
    overlays = []
    imgs_np = make_image(img_tensor)
    if args.seg_dim == 1:
        idx = np.nonzero(mask_tensor.detach().cpu().numpy()[:, 0, :, :])
        masks_np = np.zeros((mask_tensor.shape[0], mask_tensor.shape[2], mask_tensor.shape[3], 3), dtype=np.uint8)
        masks_np[idx] = (0, 255, 0)
    else:
        masks_np = mask_tensor.detach().cpu().permute(0, 2, 3, 1).type(torch.uint8).numpy()

    for i in range(b):
        img_pil = Image.fromarray(imgs_np[i][0]).convert('RGBA')
        mask_pil = Image.fromarray(masks_np[i]).convert('RGBA')

        overlay_pil = Image.blend(img_pil, mask_pil, alpha)
        overlay_tensor = transforms.functional.to_tensor(overlay_pil)
        overlays.append(overlay_tensor)
    overlays = torch.stack(overlays, dim=0)

    return overlays

def batch_pix_accuracy(output, target):
    _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target) * (target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union(output, target, num_class):
    _, predict = torch.max(output, 1)
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


def eval_metrics(output, target, num_classes, ignore_index):
    target = target.clone()
    target[target == ignore_index] = -1
    correct, labeled = batch_pix_accuracy(output.data, target)
    inter, union = batch_intersection_union(output.data, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]


def sample_val_viz_imgs(args, seg_val_loader, generator):
    with torch.no_grad():
        generator.eval()
        val_count = 0
        recon_imgs = []
        recon_segs = []
        real_imgs = []
        real_segs = []
        real_overlays = []
        fake_overlays = []

        for i, data in enumerate(seg_val_loader):
            if val_count >= args.n_sample:
                break
            val_count += data['image'].shape[0]

            real_img, real_mask, real_imgs_r_mask  = data['image'].to(device), data['mask'].to(device), data['random_mask'].to(device)

            recon_img, recon_seg, feas = generator(real_img, real_imgs_r_mask)

            recon_img = recon_img.detach().cpu()
            recon_seg = recon_seg.detach().cpu()

            real_mask = (real_mask + 1.0) / 2.0
            real_img = real_img.detach().cpu()
            real_mask = real_mask.detach().cpu()

            if args.seg_dim == 1:
                sample_seg = torch.sigmoid(recon_seg)
                sample_mask = torch.zeros_like(sample_seg)
                sample_mask[sample_seg > 0.5] = 1.0

            else:
                sample_seg = torch.softmax(recon_seg, dim=1)
                sample_mask = torch.argmax(sample_seg, dim=1)
                sample_mask = mask2rgb(args, sample_mask)

                real_mask = torch.argmax(real_mask, dim=1)
                real_mask = mask2rgb(args, real_mask)

            real_overlay = batch_overlay(args, real_img, real_mask)
            fake_overlay = batch_overlay(args, real_img, sample_mask)

            recon_imgs.append(recon_img)
            recon_segs.append(sample_mask)

            real_imgs.append(real_img)
            real_segs.append(real_mask)

            real_overlays.append(real_overlay)
            fake_overlays.append(fake_overlay)

        recon_imgs = torch.cat(recon_imgs, dim=0)
        recon_segs = torch.cat(recon_segs, dim=0)
        real_imgs = torch.cat(real_imgs, dim=0)
        real_segs = torch.cat(real_segs, dim=0)
        recon_imgs = torch.cat([real_imgs, recon_imgs], dim=0)
        recon_segs = torch.cat([real_segs, recon_segs], dim=0)

        real_overlays = torch.cat(real_overlays, dim=0)
        fake_overlays = torch.cat(fake_overlays, dim=0)
        overlay = torch.cat([real_overlays, fake_overlays], dim=0)

        return (recon_imgs, recon_segs, overlay)


def sample_unlabel_viz_imgs(args, unlabel_n_sample, unlabel_loader, generator):
    with torch.no_grad():

        generator.eval()
        val_count = 0
        real_imgs = []
        recon_imgs = []
        recon_segs = []
        fake_overlays = []

        for i, data in enumerate(unlabel_loader):
            if val_count >= unlabel_n_sample:
                break
            if args.seg_name == 'CXR' or args.seg_name == 'CXR-single':
                val_count += data.shape[0]
                real_img = data.to(device)
            else:
                val_count += data['image'].shape[0]
                real_img = data['image'].to(device)
                real_img_r_mask = data['random_mask'].to(device)

            recon_img, recon_seg, feas = generator(real_img, real_img_r_mask)

            recon_img = recon_img.detach().cpu()
            recon_seg = recon_seg.detach().cpu()
            real_img = real_img.detach().cpu()

            if args.seg_dim == 1:
                sample_seg = torch.sigmoid(recon_seg)
                sample_mask = torch.zeros_like(sample_seg)
                sample_mask[sample_seg > 0.5] = 1.0
            else:
                sample_seg = torch.softmax(recon_seg, dim=1)
                sample_mask = torch.argmax(sample_seg, dim=1)
                sample_mask = mask2rgb(args, sample_mask)

            fake_overlay = batch_overlay(args, real_img, sample_mask)

            real_imgs.append(real_img)
            recon_imgs.append(recon_img)
            recon_segs.append(sample_mask)
            fake_overlays.append(fake_overlay)

        real_imgs = torch.cat(real_imgs, dim=0)
        recon_imgs = torch.cat(recon_imgs, dim=0)
        recon_segs = torch.cat(recon_segs, dim=0)
        recon_imgs = torch.cat([real_imgs, recon_imgs], dim=0)
        fake_overlays = torch.cat(fake_overlays, dim=0)

        return (recon_imgs, recon_segs, fake_overlays)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)  
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):  
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):  #正则化项

    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size)).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class TV_Loss(torch.nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IF):
        r = IA - IF
        h = r.shape[2]
        w = r.shape[3]
        tv1 = torch.pow((r[:, :, 1:, :] - r[:, :, :h - 1, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:] - r[:, :, :, :w - 1]), 2).mean()
        return tv1 + tv2


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:   
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def validate(args, d_img, d_seg, generator, val_loader, device, writer, step):
    with torch.no_grad():
        generator.eval()

        d_img_val_scores = []
        d_seg_val_scores = []

        total_inter, total_union = 0.0, 0.0
        total_correct, total_label = 0.0, 0.0
        for i, data in enumerate(val_loader):
            img, mask, r_mask = data['image'].to(device), data['mask'].to(device), data['random_mask'].to(device)
            
            d_img_val_score = d_img(img)
            d_seg_val_score = d_seg(prep_dseg_input(args, img, mask, is_real=True))
            d_seg_val_score = torch.tensor([feat[-1].mean() for feat in d_seg_val_score])

            d_img_val_scores.append(d_img_val_score.mean().item())
            d_seg_val_scores.append(d_seg_val_score.mean().item())

            # shift mask to 0 - 1   为下面的计算miou做准备
            mask = (mask + 1.0) / 2.0

            recon_img, recon_seg, feas = generator(img, r_mask)

            if args.seg_dim == 1:
                label_pred = torch.sigmoid(recon_seg)
                bg_pred = 1.0 - label_pred
                mask_pred = torch.cat([bg_pred, label_pred], dim=1)
                true_mask = mask.squeeze(1)
                n_class = 2
            else:
                mask_pred = torch.softmax(recon_seg, dim=1)
                true_mask = torch.argmax(mask, dim=1)
                n_class = args.seg_dim

            correct, labeled, inter, union = eval_metrics(mask_pred, true_mask, n_class, -100)

            # from matplotlib import  pyplot as plt
            # fig, ax = plt.subplots(2, 1)
            # tmp = np.argmax(mask_pred.data.cpu().numpy()[0, ...], axis=0)
            # ax[0].imshow(tmp)
            # ax[1].imshow(true_mask.data.cpu().numpy()[0, ...])
            # plt.show()

            total_inter, total_union = total_inter + inter, total_union + union
            total_correct, total_label = total_correct + correct, total_label + labeled

        d_img_val_scores = np.array(d_img_val_scores).mean()
        d_seg_val_scores = np.array(d_seg_val_scores).mean()

        print("d_img val scores: {0:.4f}, d_seg val scores: {1:.4f}".format(d_img_val_scores, d_seg_val_scores))

        writer.add_scalar('scores/d_img_val', d_img_val_scores, global_step=step)
        writer.add_scalar('scores/d_seg_val', d_seg_val_scores, global_step=step)

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()

        print("===========val miou scores: {0:.4f}, pixel acc: {1:.4f} ========================".format(mIoU, pixAcc))
        writer.add_scalar('scores/miou', mIoU, global_step=step)
        writer.add_scalar('scores/pixel_acc', pixAcc, global_step=step)
        for i in range(IoU.shape[0]):
            print("===========val {0} miou scores: {1:.4f} ========================".format(i, IoU[i]))
            writer.add_scalar('scores/{} val_miou'.format(i), IoU[i], global_step=step)


def prep_dseg_input(args, img, mask, is_real):
    dseg_in = torch.cat([img, mask], dim=1)

    return dseg_in

def prep_dseg_output(args, pred, use_feat=False):
    if use_feat:
        return pred
    else:
        for i in range(len(pred)):
            for j in range(len(pred[i])-1):
                pred[i][j] = pred[i][j].detach()
        return pred

def create_heatmap(mask_tensor):
    mask_np = mask_tensor.detach().cpu().numpy()
    batch_size = mask_tensor.shape[0]
    heatmap_tensors = []
    for i in range(batch_size):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_np[i][0]), cv2.COLORMAP_JET)
        # convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_tensor = torch.tensor(heatmap, dtype=torch.float)
        heatmap_tensor = heatmap_tensor.permute(2,0,1)
        heatmap_tensors.append(heatmap_tensor)
    heatmap_tensors = torch.stack(heatmap_tensors, dim=0)
    return heatmap_tensors

def train(args, ckpt_dir, img_loader, seg_loader, seg_val_loader, generator, discriminator_img,
          discriminator_seg, g_label_optim, g_unlabel_optim, d_img_optim, d_seg_optim, percept,
          g_ema, device, writer, style_encode):

    # d_seg gan loss
    seg_gan_loss = GANLoss(gan_mode='hinge', tensor=torch.cuda.FloatTensor)    # original: 'hinge'

    img_loader = sample_data(img_loader)
    seg_loader = sample_data(seg_loader)

    if args.seg_dim == 1:
        ce_loss_func = SoftBinaryCrossEntropyLoss(tau=0.3)
        dice_loss_func = DiceLoss(sigmoid_tau=0.3, include_bg=True)
    else:
        ce_loss_func = SoftmaxLoss(tau=0.1)
        dice_loss_func = DiceLoss(sigmoid_tau=0.3)

    pbar = range(args.iter)

    mean_path_length = 0

    d_loss_val = 0
    r1_img_loss = torch.tensor(0.0, device=device)
    r1_seg_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss_label = torch.tensor(0.0, device=device)
    path_loss_unlabel = torch.tensor(0.0, device=device)
    path_lengths_label = torch.tensor(0.0, device=device)
    path_lengths_unlabel = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_img_module = discriminator_img.module
        d_seg_module = discriminator_seg.module
    else:
        g_module = generator
        d_img_module = discriminator_img
        d_seg_module = discriminator_seg
        
    accum = 0.5 ** (32 / (10 * 1000))

    get_inception_metrics = prepare_inception_metrics(args.inception, False)
    seg_img_val_loader = sample_data(seg_val_loader)
    sample_fn = functools.partial(sample_gema, g_ema=g_ema, img=next(seg_img_val_loader)['image'].to(device), img_mask=next(seg_img_val_loader)['random_mask'].to(device), device=device,
                                  truncation=1.0, mean_latent=None, batch_size=args.batch)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break

        real_data = next(img_loader)
        real_img, real_img_r_mask = real_data['image'], real_data['random_mask']

        real_img = real_img.to(device)
        real_img_r_mask = real_img_r_mask.to(device)
       
        seg_data = next(seg_loader)
        seg_img, seg_mask, seg_img_r_mask = seg_data['image'], seg_data['mask'], seg_data['random_mask']
        seg_img, seg_mask, seg_img_r_mask = seg_img.to(device), seg_mask.to(device), seg_img_r_mask.to(device)


        # =============================== Step1: train the d_img ===================================
        requires_grad(generator, False)
        requires_grad(discriminator_img, True)
        requires_grad(discriminator_seg, False)

        fake_img, fake_seg = generator(real_img, real_img_r_mask)

        # detach fake seg
        fake_seg = fake_seg.detach()

        fake_img_pred = discriminator_img(fake_img)

        real_img_pred = discriminator_img(real_img)

        d_img_loss = d_logistic_loss(real_img_pred, fake_img_pred)

        loss_dict['d_img'] = d_img_loss
        loss_dict['d_img_real_score'] = real_img_pred.mean()
        loss_dict['d_img_fake_score'] = fake_img_pred.mean()

        discriminator_img.zero_grad()
        d_img_loss.backward()
        d_img_optim.step()
        
        # =============================== Step2: train the d_seg ===================================
        requires_grad(generator, False)
        requires_grad(discriminator_img, False)
        requires_grad(discriminator_seg, True)

        fake_img, fake_seg = generator(real_img, real_img_r_mask)

        real_seg_pred = discriminator_seg(prep_dseg_input(args, seg_img, seg_mask, is_real=True))
        fake_seg_pred = discriminator_seg(prep_dseg_input(args, fake_img, fake_seg, is_real=False))

        # prepare output
        fake_seg_pred = prep_dseg_output(args, fake_seg_pred, use_feat=False)
        real_seg_pred = prep_dseg_output(args, real_seg_pred, use_feat=False)

        d_seg_loss = (seg_gan_loss(fake_seg_pred, False, for_discriminator=True).mean() + seg_gan_loss(real_seg_pred, True, for_discriminator=True).mean()) / 2.0

        loss_dict['d_seg'] = d_seg_loss
        loss_dict['d_seg_real_score'] = (real_seg_pred[0][-1].mean()+real_seg_pred[1][-1].mean()+real_seg_pred[2][-1].mean()) / 3.0
        loss_dict['d_seg_fake_score'] = (fake_seg_pred[0][-1].mean()+fake_seg_pred[1][-1].mean()+fake_seg_pred[2][-1].mean()) / 3.0

        discriminator_seg.zero_grad()
        d_seg_loss.backward()
        d_seg_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator_img(real_img)
            r1_img_loss = d_r1_loss(real_pred, real_img)

            discriminator_img.zero_grad()
            (args.r1 / 2 * r1_img_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_img_optim.step()

            # seg discriminator regulate
            real_img_seg = prep_dseg_input(args, seg_img, seg_mask, is_real=True)
            real_img_seg.requires_grad = True

            real_pred = discriminator_seg(real_img_seg)
            real_pred = prep_dseg_output(args, real_pred, use_feat=False)

            # select three D
            real_pred = real_pred[0][-1].mean() + real_pred[1][-1].mean() + real_pred[2][-1].mean()

            r1_seg_loss = d_r1_loss(real_pred, real_img_seg)

            discriminator_seg.zero_grad()
            (args.r1 / 2 * r1_seg_loss * args.d_reg_every + 0 * real_pred).backward()

            d_seg_optim.step()

        loss_dict['r1_img'] = r1_img_loss
        loss_dict['r1_seg'] = r1_seg_loss

        # =============================== Step3: train the generator ===================================
        requires_grad(generator, True)
        requires_grad(discriminator_img, False)
        requires_grad(discriminator_seg, False)

        # =====================train with unlabel data========================
        fake_img, fake_seg = generator(real_img, real_img_r_mask)

        fake_img_pred = discriminator_img(fake_img)
 
        # stop gradient from d_seg to g_img
        fake_seg_pred = discriminator_seg(prep_dseg_input(args, fake_img.detach(), fake_seg, is_real=False))   # prep_dseg_input是concat的功能
        real_seg_pred = discriminator_seg(prep_dseg_input(args, seg_img, seg_mask, is_real=True))

        # prepare output
        fake_seg_pred = prep_dseg_output(args, fake_seg_pred, use_feat=True)    # 转换成数组形式
        real_seg_pred = prep_dseg_output(args, real_seg_pred, use_feat=False)

        g_unlabel_img_loss = g_nonsaturating_loss(fake_img_pred)

        # g seg adv loss
        g_unlabel_seg_adv_loss = seg_gan_loss(fake_seg_pred, True, for_discriminator=False).mean()

        # g seg feat loss
        g_unlabel_seg_feat_loss = 0.0
        feat_weights = 1.0
        D_weights = 1.0 / 3.0

        for D_i in range(len(fake_seg_pred)):
            for D_j in range(len(fake_seg_pred[D_i])-1):
                g_unlabel_seg_feat_loss += D_weights * feat_weights * \
                    F.l1_loss(fake_seg_pred[D_i][D_j], real_seg_pred[D_i][D_j].detach()) * args.lambda_dseg_feat

        # detach fake seg
        fake_seg = fake_seg.detach()

        g_unlabel_mse_loss = torch.sum(F.l1_loss(real_img, fake_img) * (1 - real_img_r_mask)) / torch.sum(1 - real_img_r_mask)
        fake_img_mask = torch.ones((fake_img.shape[0], 1, args.size, args.size)).to(fake_img.device)
        g_unlabel_style_loss, g_unlabel_lpips_loss = calculate_style_loss(style_encode, real_img, real_img_r_mask, fake_img, fake_img_mask)

        g_unlabel_loss = g_unlabel_img_loss + (g_unlabel_mse_loss * args.lambda_unlabel_mse + g_unlabel_lpips_loss * args.lambda_unlabel_lpips +
                         g_unlabel_seg_adv_loss + g_unlabel_seg_feat_loss + g_unlabel_style_loss * 250) # +  args.lamda_tv * g_unlabel_tv_loss     #
        # + args.lamda_ssim * g_unlabel_ssim_loss
  
        loss_dict['g_unlabel_img'] = g_unlabel_img_loss
        loss_dict['g_unlabel_seg_adv'] = g_unlabel_seg_adv_loss
        loss_dict['g_unlabel_seg_feat'] = g_unlabel_seg_feat_loss
        loss_dict['g_unlabel_mse'] = g_unlabel_mse_loss
        loss_dict['g_unlabel_lpips'] = g_unlabel_lpips_loss
        loss_dict['g_unlabel_style'] = g_unlabel_style_loss
        loss_dict['g_unlabel'] = g_unlabel_loss

        generator.zero_grad()
        g_unlabel_loss.backward()

        if args.profile_grad_norm == True:
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in g_module.parameters()]), 2.0)
            print("e_unlabel_average grad norm: {0:.6f}".format(total_norm))

        if args.no_grad_clip != True:
            clip_grad_norm_(g_module.parameters(), args.unlabel_grad_clip)

        g_unlabel_optim.step()
        torch.cuda.empty_cache()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:

            fake_img, ws_latents, feature = generator(real_img, real_img_r_mask, return_latents=True, phase='train')

            path_loss_unlabel, mean_path_length, path_lengths_unlabel = g_path_regularize(
                fake_img, ws_latents, mean_path_length
            )

            generator.zero_grad()

            weighted_path_loss_unlabel = args.path_regularize * args.g_reg_every * path_loss_unlabel
            if args.path_batch_shrink:
                weighted_path_loss_unlabel += 0 * fake_img[0, 0, 0, 0]
            weighted_path_loss_unlabel.backward()

            g_unlabel_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )
        loss_dict['path_unlabel'] = path_loss_unlabel
        loss_dict['path_length_unlabel'] = path_lengths_unlabel.mean()

        update_learning_rate(args, i, g_unlabel_optim)

        #=================== train with label data ========================
        fake_img, fake_seg, feature_label = generator(seg_img, seg_img_r_mask)

        fake_img_pred = discriminator_img(fake_img)

        # stop gradient from d_seg to g_img
        fake_seg_pred = discriminator_seg(
            prep_dseg_input(args, fake_img.detach(), fake_seg, is_real=False))  
        real_seg_pred = discriminator_seg(prep_dseg_input(args, seg_img, seg_mask, is_real=True))

        # prepare output
        fake_seg_pred = prep_dseg_output(args, fake_seg_pred, use_feat=True) 
        real_seg_pred = prep_dseg_output(args, real_seg_pred, use_feat=False)

        g_label_img_loss = g_nonsaturating_loss(fake_img_pred)

        # g seg adv loss
        g_label_seg_adv_loss = seg_gan_loss(fake_seg_pred, True, for_discriminator=False).mean()

        # g seg feat loss
        g_label_seg_feat_loss = 0.0
        feat_weights = 1.0
        D_weights = 1.0 / 3.0

        for D_i in range(len(fake_seg_pred)):
            for D_j in range(len(fake_seg_pred[D_i]) - 1):
                g_label_seg_feat_loss += D_weights * feat_weights * \
                                           F.l1_loss(fake_seg_pred[D_i][D_j],
                                                     real_seg_pred[D_i][D_j].detach()) * args.lambda_dseg_feat

        if args.seg_dim == 1:
            # shift to 0-1
            seg_mask = (seg_mask + 1.0) / 2.0
            g_label_ce_loss = ce_loss_func(fake_seg, seg_mask)
            g_label_dice_loss = dice_loss_func(fake_seg, seg_mask)
        else:
            # make seg mask to label
            seg_mask_ce = torch.argmax(seg_mask, dim=1)
            seg_mask_dice = (seg_mask + 1.0) / 2.0
            g_label_ce_loss = ce_loss_func(fake_seg, seg_mask_ce)
            g_label_dice_loss = dice_loss_func(fake_seg, seg_mask_dice)


        g_label_mse_loss = torch.sum(F.l1_loss(seg_img, fake_img) * (1 - seg_img_r_mask)) / torch.sum(1 - seg_img_r_mask)
        g_label_style_loss, g_label_lpips_loss = calculate_style_loss(style_encode, seg_img, seg_img_r_mask, fake_img, fake_img_mask)
        # g_label_lpips_loss = percept(fake_img, seg_img).mean()
        # g_label_lpips_loss, _  = percept(fake_img, seg_img)
        # g_label_ssim_loss = (1 - SSIM_loss(fake_img, seg_img))
        # g_label_tv_loss = TV_loss(seg_img, fake_img)

        g_label_loss = g_label_img_loss + g_label_seg_adv_loss + g_label_seg_feat_loss \
            + (g_label_mse_loss * args.lambda_label_mse +
                        g_label_lpips_loss * args.lambda_label_lpips + g_label_style_loss * 250) + \
        (g_label_ce_loss * args.lambda_label_ce +
                           g_label_dice_loss * args.lambda_label_dice)
        #+ args.lamda_tv * g_label_tv_loss
                # + args.lamda_ssim * g_label_ssim_loss

        loss_dict['g_label_img'] = g_label_img_loss
        loss_dict['g_label_seg_adv'] = g_label_seg_adv_loss
        loss_dict['g_label_seg_feat'] = g_label_seg_feat_loss
        loss_dict['g_label_mse'] = g_label_mse_loss
        loss_dict['g_label_style'] = g_label_style_loss
        loss_dict['g_label_lpips'] = g_label_lpips_loss
        loss_dict['g_label_ce'] = g_label_ce_loss
        loss_dict['g_label_dice'] = g_label_dice_loss
        # loss_dict['g_label_tv'] = g_label_tv_loss
        # loss_dict['g_label_ssim'] = g_label_ssim_loss
        loss_dict['g_label'] = g_label_loss

        generator.zero_grad()
        g_label_loss.backward()

        if args.profile_grad_norm == True:
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in g_module.parameters()]), 2.0)
            print("e_label_average grad norm: {0:.6f}".format(total_norm))

        if args.no_grad_clip != True:
            # gradient clipping
            clip_grad_norm_(g_module.parameters(), args.label_grad_clip)

        g_label_optim.step()

        g_regularize = i % args.g_reg_every == 0
        if g_regularize:

            fake_img, ws_latents, features = generator(seg_img, seg_img_r_mask, return_latents=True)

            path_loss_label, mean_path_length, path_lengths_label = g_path_regularize(
                fake_img, ws_latents, mean_path_length
            )

            generator.zero_grad()

            weighted_path_loss_label = args.path_regularize * args.g_reg_every * path_loss_label
            if args.path_batch_shrink:
                weighted_path_loss_label += 0 * fake_img[0, 0, 0, 0]
            weighted_path_loss_label.backward()

            g_label_optim.step()

            mean_path_length_avg = (
                    reduce_sum(mean_path_length).item() / get_world_size()
            )
        loss_dict['path_label'] = path_loss_label
        loss_dict['path_length_label'] = path_lengths_label.mean()

        update_learning_rate(args, i, g_label_optim)

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        g_unlabel_mse_loss_val = loss_reduced['g_unlabel_mse'].mean().item()
        g_unlabel_style_loss_val = loss_reduced['g_unlabel_style'].mean().item()
        g_unlabel_lpips_loss_val = loss_reduced['g_unlabel_lpips'].mean().item()
        # g_unlabel_tv_loss_val = loss_reduced['g_unlabel_tv'].mean().item()
        # g_unlabel_ssim_loss_val = loss_reduced['g_unlabel_ssim'].mean().item()

        g_unlabel_img_loss_val = loss_reduced['g_unlabel_img'].mean().item()
        g_unlabel_seg_adv_loss_val = loss_reduced['g_unlabel_seg_adv'].mean().item()
        g_unlabel_seg_feat_loss_val = loss_reduced['g_unlabel_seg_feat'].mean().item()
        g_unlabel_loss_val = loss_reduced['g_unlabel'].mean().item()

        g_label_mse_loss_val = loss_reduced['g_label_mse'].mean().item()
        g_label_ce_loss_val = loss_reduced['g_label_ce'].mean().item()
        g_label_dice_loss_val = loss_reduced['g_label_dice'].mean().item()
        g_label_style_loss_val = loss_reduced['g_label_style'].mean().item()
        g_label_lpips_loss_val = loss_reduced['g_label_lpips'].mean().item()
        # g_label_tv_loss_val = loss_reduced['g_label_tv'].mean().item()
        # g_label_ssim_loss_val = loss_reduced['g_label_ssim'].mean().item()
        g_label_img_loss_val = loss_reduced['g_label_img'].mean().item()
        g_label_seg_adv_loss_val = loss_reduced['g_label_seg_adv'].mean().item()
        g_label_seg_feat_loss_val = loss_reduced['g_label_seg_feat'].mean().item()

        g_label_loss_val = loss_reduced['g_label'].mean().item()

        d_img_loss_val = loss_reduced['d_img'].mean().item()
        d_seg_loss_val = loss_reduced['d_seg'].mean().item()

        r1_img_val = loss_reduced['r1_img'].mean().item()
        r1_seg_val = loss_reduced['r1_seg'].mean().item()

        d_img_real_score_val = loss_reduced['d_img_real_score'].mean().item()
        d_img_fake_score_val = loss_reduced['d_img_fake_score'].mean().item()
        d_seg_real_score_val = loss_reduced['d_seg_real_score'].mean().item()
        d_seg_fake_score_val = loss_reduced['d_seg_fake_score'].mean().item()

        path_loss_val_unlabel = loss_reduced['path_unlabel'].mean().item()
        path_length_val_unlabel = loss_reduced['path_length_unlabel'].mean().item()
        path_loss_val_label = loss_reduced['path_label'].mean().item()
        path_length_val_label = loss_reduced['path_length_label'].mean().item()

        if get_rank() == 0:
            # write to tensorboard
            writer.add_scalars('scores/d_img', {'real_score': d_img_real_score_val,
                                                'fake_score': d_img_fake_score_val
                                                }, global_step=i)

            writer.add_scalars('scores/d_seg', {'real_score': d_seg_real_score_val,
                                                'fake_score': d_seg_fake_score_val
                                                }, global_step=i)

            writer.add_scalar('r1/d_img', r1_img_val, global_step=i)
            writer.add_scalar('r1/d_seg', r1_seg_val, global_step=i)

            writer.add_scalar('g_unlabel/mse_loss', g_unlabel_mse_loss_val, global_step=i)
            writer.add_scalar('g_unlabel/style_loss', g_unlabel_style_loss_val, global_step=i)
            writer.add_scalar('g_unlabel/lpips_loss', g_unlabel_lpips_loss_val, global_step=i)

            writer.add_scalar('g_unlabel/img_loss', g_unlabel_img_loss_val, global_step=i)
            writer.add_scalar('g_unlabel/seg_adv_loss', g_unlabel_seg_adv_loss_val, global_step=i)
            writer.add_scalar('g_unlabel/seg_feat_loss', g_unlabel_seg_feat_loss_val, global_step=i)
            writer.add_scalar('g_unlabel/total_loss', g_unlabel_loss_val, global_step=i)

            writer.add_scalar('g_label/mse_loss', g_label_mse_loss_val, global_step=i)
            writer.add_scalar('g_label/style_loss', g_label_style_loss_val, global_step=i)
            writer.add_scalar('g_label/lpips_loss', g_label_lpips_loss_val, global_step=i)
            writer.add_scalar('g_label/ce_loss', g_label_ce_loss_val, global_step=i)
            writer.add_scalar('g_label/dice_loss', g_label_dice_loss_val, global_step=i)

            writer.add_scalar('g_label/img_loss', g_label_img_loss_val, global_step=i)
            writer.add_scalar('g_label/seg_adv_loss', g_label_seg_adv_loss_val, global_step=i)
            writer.add_scalar('g_label/seg_feat_loss', g_label_seg_feat_loss_val, global_step=i)
            writer.add_scalar('g_label/total_loss', g_label_loss_val, global_step=i)

            # learning rate
            writer.add_scalar('lr/g_label_lr', g_label_optim.param_groups[0]['lr'], global_step=i)
            writer.add_scalar('lr/g_unlabel_lr', g_unlabel_optim.param_groups[0]['lr'], global_step=i)

            writer.add_scalar('path/path_loss_label', path_loss_val_label, global_step=i)
            writer.add_scalar('path/path_length_label', path_length_val_label, global_step=i)
            writer.add_scalar('path/path_loss_unlabel', path_loss_val_unlabel, global_step=i)
            writer.add_scalar('path/path_length_unlabel', path_length_val_unlabel, global_step=i)

            writer.add_scalar('loss/g_label', g_label_loss_val, global_step=i)
            writer.add_scalar('loss/g_unlabel', g_unlabel_loss_val, global_step=i)
            writer.add_scalar('loss/d_img', d_img_loss_val, global_step=i)
            writer.add_scalar('loss/d_seg', d_seg_loss_val, global_step=i)

            if i % args.viz_every == 0:
                with torch.no_grad():
                    val_recon_img, val_recon_seg, _, = sample_val_viz_imgs(args, seg_val_loader,
                                                                           generator)
                    val_n_sample = min(len(seg_val_loader), args.n_sample * 2)

                    unlabel_n_sample = 32
                    unlabel_recon_img, unlabel_recon_seg, unlabel_fake_overlays = sample_unlabel_viz_imgs(args,
                                                                                                          unlabel_n_sample,
                                                                                                          img_loader,
                                                                                                          generator,
                                                                                                          )

                    g_ema.eval()
                    for j, data in enumerate(seg_val_loader):
                        if j == 1:
                            sample_img, sample_seg, feas = g_ema(
                                data['image'].to(device), data['random_mask'].to(device)) 
                    sample_img = sample_img.detach().cpu()
                    sample_seg = sample_seg.detach().cpu()

                    if args.seg_name == 'celeba-mask':
                        sample_seg = torch.argmax(sample_seg, dim=1)
                        color_map = seg_val_loader.dataset.color_map
                        sample_mask = torch.zeros((sample_seg.shape[0], sample_seg.shape[1], sample_seg.shape[2], 3),
                                                  dtype=torch.float)
                        for key in color_map:
                            sample_mask[sample_seg == key] = torch.tensor(color_map[key], dtype=torch.float)
                        sample_mask = sample_mask.permute(0, 3, 1, 2)

                    else:
                        raise Exception('No such a dataloader!')

                    os.makedirs(os.path.join(ckpt_dir, 'sample'), exist_ok=True)

                    utils.save_image(
                        sample_img,
                        os.path.join(ckpt_dir, f'sample/img_{str(i).zfill(6)}.png'),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        sample_mask,
                        os.path.join(ckpt_dir, f'sample/mask_{str(i).zfill(6)}.png'),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                    )

                    # 直接做重建得到的图保存
                    utils.save_image(
                        val_recon_img,
                        os.path.join(ckpt_dir, f'sample/val_recon_img_{str(i).zfill(6)}.png'),
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        val_recon_seg,
                        os.path.join(ckpt_dir, f'sample/val_recon_seg_{str(i).zfill(6)}.png'),
                        nrow=8,
                        normalize=True,
                    )

                    utils.save_image(
                        unlabel_recon_img,
                        os.path.join(ckpt_dir, f'sample/unlabel_recon_img_{str(i).zfill(6)}.png'),
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        unlabel_recon_seg,
                        os.path.join(ckpt_dir, f'sample/unlabel_recon_seg_{str(i).zfill(6)}.png'),
                        nrow=8,
                        normalize=True,
                    )

            if i %  args.eval_every == 0:  #
                print("==================Start calculating validation scores==================")
                validate(args, discriminator_img, discriminator_seg, g_ema, seg_val_loader, device, writer, i)
                
            if i % args.save_every == 0:
                print("==================Start calculating FID==================")
                IS_mean, IS_std, FID = get_inception_metrics(sample_fn, num_inception_images=100, use_torch=False)
                print("iteration {0:08d}: FID: {1:.4f}, IS_mean: {2:.4f}, IS_std: {3:.4f}".format(i, FID, IS_mean, IS_std))

                writer.add_scalar('metrics/FID', FID, global_step=i)
                writer.add_scalar('metrics/IS_mean', IS_mean, global_step=i)
                writer.add_scalar('metrics/IS_std', IS_std, global_step=i)

                writer.add_text('metrics/FID', 'FID is {0:.4f}'.format(FID), global_step=i)
                
                os.makedirs(os.path.join(ckpt_dir, 'ckpt'), exist_ok=True)
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd_img': d_img_module.state_dict(),
                        'd_seg': d_seg_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'args': args,
                        'g_unlabel_optim': g_unlabel_optim.state_dict(),
                        'g_label_optim': g_label_optim.state_dict(),
                        'd_img_optim': d_img_optim.state_dict(),
                        'd_seg_optim': d_seg_optim.state_dict(),
                    },
                    os.path.join(ckpt_dir, f'ckpt/{str(i).zfill(6)}.pt'),
                )

def get_seg_dataset(args, phase='train'):
    if args.seg_name == 'celeba-mask':
        seg_dataset = CelebAMaskDataset(args, args.seg_dataset, is_label=True, phase=phase,
                                            limit_size=args.limit_data, aug=args.seg_aug, resolution=args.size)
    else:
        raise Exception('No such a dataloader!')
    
    return seg_dataset

def update_learning_rate(args, i, optimizer):
    if i < args.lr_decay_iter_start:
        pass
    elif i < args.lr_decay_iter_end:
        lr_max = args.lr
        lr_min = args.lr_decay
        t_max = args.lr_decay_iter_end - args.lr_decay_iter_start
        t_cur = i - args.lr_decay_iter_start

        optimizer.param_groups[0]['lr'] = lr_min + 0.5 * (lr_max - lr_min) * (
                    1 + math.cos(t_cur * 1.0 / t_max * math.pi))
    else:
        pass

def get_transformation(args):
    if args.seg_name == 'celeba-mask':
        transform = transforms.Compose(
                    [
                        # transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # transforms.Normalize((0.5), (0.5), inplace=True)
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
                    ]
                )
    else:
        raise Exception('No such a dataloader!')
    
    return transform

if __name__ == '__main__':

    torch.cuda.empty_cache()
    device = 'cuda'
    torch.cuda.set_device(0)  
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dataset', default='', type=str)
    parser.add_argument('--seg_dataset', default='', type=str)
    parser.add_argument('--inception', default='./camus/inception.txt', type=str, help='inception pkl')
    parser.add_argument('--inception', type=str, default='', help='inception pkl')

    parser.add_argument('--seg_name', type=str, help='segmentation dataloader name[celeba-mask]', default='celeba-mask')
    parser.add_argument('--iter', type=int, default=60000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--val_batch', type=int, default=2)
    parser.add_argument('--n_sample', type=int, default=1)    
    parser.add_argument('--size', type=int, default=256)

    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--d_use_seg_every', type=int, help='frequency mixing seg image with real image', default=-1)
    parser.add_argument('--viz_every', type=int, default=2000)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)

    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--lambda_dseg_feat', type=float, default=2.0)
    parser.add_argument('--ckpt_encoder', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.0002)   # original 0.0002
    parser.add_argument('--lr_decay', type=float, default=0.00002)
    parser.add_argument('--lambda_lr', type=float, default=0.999)
    parser.add_argument('--lr_decay_iter_start', type=int, default=30000)
    parser.add_argument('--lr_decay_iter_end', type=int, default=100000)
    parser.add_argument('--channel_multiplier', type=int, default=2)

    parser.add_argument('--lambda_label_lpips', type=float, default=10.0)   #original 10.0
    parser.add_argument('--lambda_label_mse', type=float, default=10.0)   # original 1.0
    parser.add_argument('--lambda_label_ce', type=float, default=1.0)    # original 1.0
    parser.add_argument('--lambda_label_depth', type=float, default=1.0)
    parser.add_argument('--lambda_label_dice', type=float, default=1.0)    # original 1.0
    parser.add_argument('--lambda_label_latent', type=float, default=0.0)
    parser.add_argument('--lambda_label_adv', type=float, default=0.1)

    parser.add_argument('--lambda_unlabel_lpips', type=float, default=10.0)
    parser.add_argument('--lambda_unlabel_mse', type=float, default=10.0)     # original 10.0
    parser.add_argument('--lambda_unlabel_adv', type=float, default=0.1)

    parser.add_argument('--lamda_ssim', type=float, default=1, help='weight of the SSIM loss')  #original 1.0
    parser.add_argument('--limit_data', type=str, default=None, help='number of limited label data point to use')
    parser.add_argument('--unlabel_limit_data', type=str, default=None, help='number of limited unlabel data point to use')

    parser.add_argument('--image_mode', type=str, default='RGB', help='Image mode RGB|L')
    parser.add_argument('--seg_dim', type=int, default=2)
    parser.add_argument('--seg_aug', default=False, action='store_true', help='seg augmentation')

    parser.add_argument('--no_grad_clip', action='store_true', help='if use gradient clipping')
    parser.add_argument('--profile_grad_norm', action='store_true', help='if profile average grad norm')
    parser.add_argument('--label_grad_clip', type=float, help='gradient clip norm value for labeled dataloader',
                        default=5.0)
    parser.add_argument('--unlabel_grad_clip', type=float, help='gradient clip norm value for unlabeled dataloader',
                        default=2.0)

    parser.add_argument('--enc_backbone', type=str, help='encoder backbone[res|fpn]', default='fpn')
    parser.add_argument('--optimizer', type=str, help='encoder backbone[adam|ranger]', default='ranger')

    parser.add_argument('--checkpoint_dir', type=str, default='..\\checkpoint\\TN-SCUI\\gan')

    parser.add_argument('--local_rank', type=int, default=0)

    # 关于Swin_ tranformer的相关设置
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    args = parser.parse_args()

    # build checkpoint dir
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    ckpt_dir = os.path.join(args.checkpoint_dir, 'run-'+current_time)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.n_gpu = n_gpu

    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = GeneratorSeg(
        args, args.size, args.latent, args.n_mlp, seg_dim=args.seg_dim,
        image_mode=args.image_mode, channel_multiplier=args.channel_multiplier
    ).to(device)

    if args.image_mode == 'RGB':
        d_input_dim = 3
    else:
        d_input_dim = 1

    d_seg_input_dim = d_input_dim + args.seg_dim

    discriminator_img = Discriminator(
        args.size, input_dim=d_input_dim, channel_multiplier=args.channel_multiplier
    ).to(device)

    discriminator_seg = MultiscaleDiscriminator(input_nc=d_seg_input_dim, getIntermFeat=True).to(device)
 
    g_ema = GeneratorSeg(
        args, args.size, args.latent, args.n_mlp, seg_dim=args.seg_dim,
        image_mode=args.image_mode, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    # percep
    percept = PerceptualLoss(layer_weights=dict(conv4_4=1 / 4, conv5_4=1 / 2)).to(device)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_unlabel_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    g_label_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_img_optim = optim.Adam(
        discriminator_img.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    d_seg_optim = optim.Adam(
        discriminator_seg.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )


    if args.ckpt is not None:
        print('load model:', args.ckpt)
        
        ckpt = torch.load(args.ckpt, map_location={'cuda:0': 'cpu'})

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            
        except ValueError:
            pass
            
        generator.load_state_dict(ckpt['g'], strict=False)
        discriminator_img.load_state_dict(ckpt['d_img'])
        discriminator_seg.load_state_dict(ckpt['d_seg'])
        g_ema.load_state_dict(ckpt['g_ema'], strict=False)


        g_label_optim.load_state_dict(ckpt['g_label_optim'])        
        g_unlabel_optim.load_state_dict(ckpt['g_unlabel_optim'])
        d_img_optim.load_state_dict(ckpt['d_img_optim'])
        d_seg_optim.load_state_dict(ckpt['d_seg_optim'])

    from models.mat import FirstStage_encoder
    ckpt_encoder = torch.load(args.ckpt_encoder, map_location={'cuda:0': 'cpu'})
    style_encoder = FirstStage_encoder().to(device)
    style_encoder_dict = style_encoder.state_dict()
    load_weights_dict = {k: v for k, v in ckpt_encoder['g'].items() if (
        k if not k.startswith('swin_net.synthesis.first_stage.') else k[
                                                                      len('swin_net.synthesis.first_stage.'):]) in style_encoder_dict.keys()}
    style_encoder.load_state_dict(load_weights_dict, strict=False)
    style_encode = Encoder_style(style_encoder)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator_img = nn.parallel.DistributedDataParallel(
            discriminator_img,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator_seg = nn.parallel.DistributedDataParallel(
            discriminator_seg,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        g_ema = nn.parallel.DistributedDataParallel(
            g_ema,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        percept = nn.parallel.DistributedDataParallel(
            percept,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        style_encode = nn.parallel.DistributedDataParallel(
            style_encode,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    if args.seg_name == 'celeba-mask':
        transform = get_transformation(args)
        img_dataset = CelebAMaskDataset(args, args.img_dataset, unlabel_transform=transform, unlabel_limit_size=args.unlabel_limit_data,
                                                is_label=False, resolution=args.size)
    else:
        raise Exception('No such a dataloader!')

    print("Loading unlabel dataloader with size ", img_dataset.data_size)

    img_loader = data.DataLoader(
        img_dataset,
        batch_size=args.batch,
        sampler=data_sampler(img_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    seg_dataset = get_seg_dataset(args, phase='train')

    print("Loading train dataloader with size ", seg_dataset.data_size)

    seg_loader = data.DataLoader(
        seg_dataset,
        batch_size=args.batch,
        sampler=data_sampler(seg_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    seg_val_dataset = get_seg_dataset(args, phase='val')

    print("Loading val dataloader with size ", seg_val_dataset.data_size)

    seg_val_loader = data.DataLoader(
        seg_val_dataset,
        batch_size=args.val_batch,
        shuffle=False,
        drop_last=True,
    )

    print("local rank: {}, Start training!".format(args.local_rank))

    # setup benchmark
    torch.backends.cudnn.benchmark = True
    
    train(args, ckpt_dir, img_loader, seg_loader, seg_val_loader, generator, discriminator_img, discriminator_seg,
                    g_label_optim, g_unlabel_optim, d_img_optim, d_seg_optim, percept, g_ema, device, writer, style_encode)
