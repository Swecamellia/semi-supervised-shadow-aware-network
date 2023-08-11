import argparse
import math
import random
import os
from re import I
from sklearn.exceptions import NonBLASDotWarning
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
from models.PAR import PAR

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
# from models.encoder_model import FPNEncoder, ResEncoder
from semanticGAN.losses import SoftmaxLoss, SoftBinaryCrossEntropyLoss, DiceLoss
from PIL import Image
from models.mask_generator_256 import BatchRandomMask
from models.pcp import PerceptualLoss
import torchvision
import matplotlib.pyplot as plt

import segmentation_models_pytorch as smp

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
             [132, 112, 255],
             [25, 25, 112],
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

def get_mask_by_radius(h=16, w=16, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1
    return mask

def get_seg_loss(pred, label, ignore_index=255):

    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5

def seg_to_affinity_label(seg_mask, mask=None, ignore_index=255):
    seg_label = torch.argmax(seg_mask, dim=1)
    b, h, w = seg_label.shape
    seg_label_resized = F.interpolate(seg_label.unsqueeze(1).type(torch.float32), size=[h // 4, w // 4],
                                      mode="nearest")
    _seg_label = seg_label_resized.reshape(b, 1, -1)
    _seg_label_rep = _seg_label.repeat([1, _seg_label.shape[-1], 1])
    _seg_label_rep_t = _seg_label_rep.permute(0, 2, 1)
    aff_label = (_seg_label_rep == _seg_label_rep_t).type(torch.long)

    for i in range(b):
        if mask is not None:
            aff_label[i, mask == 0] = ignore_index
        aff_label[i, :, _seg_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _seg_label_rep[i, 0, :] == ignore_index, :] = ignore_index

    return aff_label

def mask_to_affinity_label(seg_mask):
    seg_label = torch.argmax(seg_mask, dim=1)
    b, h, w = seg_label.shape
    seg_label_resized = F.interpolate(seg_label.unsqueeze(1).type(torch.float32), size=[h // 4, w // 4],
                                      mode="nearest")
    _seg_label = seg_label_resized.reshape(b, 1, -1)
    _seg_label_rep = _seg_label.repeat([1, _seg_label.shape[-1], 1])
    _seg_label_rep_t = _seg_label_rep.permute(0, 2, 1)
    aff_label = (_seg_label_rep == _seg_label_rep_t).type(torch.long)

    return aff_label


def propagte_aff_seg_with_bkg(fea_maps, aff=None, mask=None, bkg_score=None):

    b,_,h,w = fea_maps.shape
    bkg = torch.ones(size=(b,1,h,w)) * bkg_score
    bkg = bkg.to(fea_maps.device)

    feas_with_bkg = torch.cat((bkg, fea_maps), dim=1)
    feas_rw = torch.zeros_like(feas_with_bkg)

    b, c, h, w = feas_with_bkg.shape
    n_pow = 2
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            aff[i, mask==0] = 0

    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-1) ## avoid nan

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _feas = feas_with_bkg[i].reshape(c, -1)
        _feas = F.softmax(_feas, dim=0)
        _aff = aff[i]
        _feas_rw = torch.matmul(_feas, _aff)
        feas_rw[i] = _feas_rw.reshape(-1, feas_rw.shape[2], feas_rw.shape[3])

    return feas_rw


class Active_Contour_Loss(torch.nn.Module):
    def __init__(self):
        super(Active_Contour_Loss, self).__init__()

    def forward(self, y_pred, y_true):
        """
            lenth term
            """

        x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal and vertical directions
        y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

        delta_x = x[:, :, 1:, :-2] ** 2
        delta_y = y[:, :, :-2, 1:] ** 2
        delta_u = torch.abs(delta_x + delta_y)

        lenth = torch.mean(torch.sqrt(delta_u + 0.00000001))  # equ.(11) in the paper

        """
        region term
        """

        C_1 = torch.ones((256, 256)).to(y_pred.device)
        C_2 = torch.zeros((256, 256)).to(y_pred.device)

        region_in = torch.abs(
            torch.mean(y_pred[:, 0, :, :] * ((y_true[:, 0, :, :] - C_1) ** 2)))  # equ.(12) in the paper
        region_out = torch.abs(
            torch.mean((1 - y_pred[:, 0, :, :]) * ((y_true[:, 0, :, :] - C_2) ** 2)))  # equ.(12) in the paper

        lambdaP = 1  # lambda parameter could be various.
        mu = 1  # mu parameter could be various.

        return lenth + lambdaP * (mu * region_in + region_out)


def propagte_aff_seg(fea_maps, aff=None, mask=None):
    b, c, h, w = fea_maps.shape
    n_pow = 2
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            aff[i, mask==0] = 0

    fea_rw = fea_maps.clone()
    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-4)

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _feas = fea_maps[i].reshape(c, -1)
        _aff = aff[i]
        _fea_rw = torch.matmul(_feas, _aff)
        fea_rw[i] = _fea_rw.reshape(fea_rw[i].shape)

    return fea_rw


def refine_feas_with_bkg_v2(ref_mod=None, images=None, feas=None, down_scale=2, seg_prob_lower=0.35, seg_prob_higher=0.55, ignore_index=255):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * seg_prob_higher
    bkg_h = bkg_h.to(feas.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * seg_prob_lower
    bkg_l = bkg_l.to(feas.device)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(feas.device)

    feas_ = F.interpolate(feas, size=[h, w], mode="bilinear",
                                     align_corners=False)
    feas_with_bkg_h = torch.cat((bkg_h, feas_), dim=1)
    _feas_with_bkg_h = F.interpolate(feas_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    feas_with_bkg_l = torch.cat((bkg_l, feas_), dim=1)
    _feas_with_bkg_l = F.interpolate(feas_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    valid_feas_h = _feas_with_bkg_h.softmax(dim=1)
    valid_feas_l = _feas_with_bkg_l.softmax(dim=1)

    _refined_label_h = _refine_feas(ref_mod=ref_mod, images=_images, feas=valid_feas_h,
                                    orig_size=(h, w))
    _refined_label_l = _refine_feas(ref_mod=ref_mod, images=_images, feas=valid_feas_l,
                                    orig_size=(h, w))

    refined_label = _refined_label_h.clone()
    # refined_label[_refined_label_h == 0] = ignore_index
    refined_label[(_refined_label_h + _refined_label_l) == 0] = 0

    return refined_label


def _refine_feas(ref_mod, images, feas, orig_size):
    refined_feas = ref_mod(images, feas)
    refined_feas = F.interpolate(refined_feas, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_feas.argmax(dim=1)
    return refined_label


def refine_feas(ref_mod=None, images=None, feas=None):

    _, _, h, w = images.shape
    _images_ = F.interpolate(images, size=[h // 2, w // 2], mode="bilinear", align_corners=False)
    _refined_feas = ref_mod(_images_, feas)
    _refined_feas = F.interpolate(_refined_feas, size=images.shape[2:], mode="bilinear", align_corners=False)

    refined_feas = _refined_feas

    return refined_feas

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

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=0)
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=0)
    return iou

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

            real_img, real_mask, real_rmask = data['image'].to(device), data['mask'].to(device), data['random_mask'].to(device)

            recon_img, recon_seg, feas = generator(real_img, None, phase='val')

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


def tensorboard_attn(attns=None, size=[256, 256], n_pix=0, n_row=4):
    n = len(attns)
    imgs = []
    for idx, attn in enumerate(attns):

        b, hw, _ = attn.shape
        h = w = int(np.sqrt(hw))

        attn_ = attn.clone()  # - attn.min()
        # attn_ = attn_ / attn_.max()
        _n_pix = int(h * n_pix) * (w + 1)
        attn_ = attn_[:, _n_pix, :].reshape(b, 1, h, w)

        attn_ = F.interpolate(attn_, size=size, mode='bilinear', align_corners=True)

        attn_ = attn_.cpu()[:, 0, :, :]

        def minmax_norm(x):
            for i in range(x.shape[0]):
                x[i, ...] = x[i, ...] - x[i, ...].min()
                x[i, ...] = x[i, ...] / x[i, ...].max()
            return x

        attn_ = minmax_norm(attn_)

        attn_heatmap = plt.get_cmap('viridis')(attn_.numpy())[:, :, :, 0:3] * 255
        attn_heatmap = torch.from_numpy(attn_heatmap).permute([0, 3, 2, 1])
        imgs.append(attn_heatmap)
    attn_img = torch.cat(imgs, dim=0)

    grid_attn = torchvision.utils.make_grid(tensor=attn_img.type(torch.float32), nrow=n_row).permute(0, 2, 1)
    grid_attn = grid_attn / 255.0 * 2 - 1.0

    return grid_attn

def tensorboard_attn1(attns=None, size=[256, 256], n_pix=0, n_row=4):
    n = len(attns)
    imgs = []

    b, hw, _ = attns.shape
    h = w = int(np.sqrt(hw))

    attn_ = attns.clone()  # - attn.min()
    # attn_ = attn_ / attn_.max()
    _n_pix = int(h * n_pix) * (w + 1)
    attn_ = attn_[:, _n_pix, :].reshape(b, 1, h, w)

    attn_ = F.interpolate(attn_, size=size, mode='bilinear', align_corners=True)

    attn_ = attn_.cpu()[:, 0, :, :]

    def minmax_norm(x):
        for i in range(x.shape[0]):
            x[i, ...] = x[i, ...] - x[i, ...].min()
            x[i, ...] = x[i, ...] / x[i, ...].max()
        return x

    attn_ = minmax_norm(attn_)

    attn_heatmap = plt.get_cmap('viridis')(attn_.numpy())[:, :, :, 0:4] * 255
    attn_heatmap = torch.from_numpy(attn_heatmap).permute([0, 3, 2, 1])

    grid_attn = torchvision.utils.make_grid(tensor=attn_heatmap.type(torch.float32), nrow=n_row).permute(0, 2, 1)
    grid_attn = grid_attn / 255.0 * 2 - 1.0

    return grid_attn


def tensorboard_attn2(attns=None, size=[256, 256], n_pixs=[0.0, 0.3, 0.6, 0.9], n_row=4, with_attn_pred=True):
    n = len(attns)
    attns_top_layers = []
    attns_last_layer = []
    grid_attns = []
    if with_attn_pred:
        _attns_top_layers = attns[:-3]
        _attns_last_layer = attns[-3:-1]
    else:
        _attns_top_layers = attns[:-2]
        _attns_last_layer = attns[-2:]

    attns_top_layers = [_attns_top_layers[i][:, 0, ...] for i in range(len(_attns_top_layers))]
    if with_attn_pred:
        attns_top_layers.append(attns[-1])
    grid_attn_top_case0 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[0], n_row=n_row)
    grid_attn_top_case1 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[1], n_row=n_row)
    grid_attn_top_case2 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[2], n_row=n_row)
    grid_attn_top_case3 = tensorboard_attn(attns_top_layers, n_pix=n_pixs[3], n_row=n_row)
    grid_attns.append(grid_attn_top_case0)
    grid_attns.append(grid_attn_top_case1)
    grid_attns.append(grid_attn_top_case2)
    grid_attns.append(grid_attn_top_case3)

    for attn in _attns_last_layer:
        for i in range(attn.shape[1]):
            attns_last_layer.append(attn[:, i, :, :])
    grid_attn_last_case0 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[0], n_row=2 * n_row)
    grid_attn_last_case1 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[1], n_row=2 * n_row)
    grid_attn_last_case2 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[2], n_row=2 * n_row)
    grid_attn_last_case3 = tensorboard_attn(attns_last_layer, n_pix=n_pixs[3], n_row=2 * n_row)
    grid_attns.append(grid_attn_last_case0)
    grid_attns.append(grid_attn_last_case1)
    grid_attns.append(grid_attn_last_case2)
    grid_attns.append(grid_attn_last_case3)

    return grid_attns

def sample_unlabel_viz_imgs(args, unlabel_n_sample, unlabel_loader, generator, par, now_iter):
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
                real_img, real_rmask = data['image'].to(device), data['random_mask'].to(device)

            recon_img, recon_seg, feas = generator(real_img, None, phase='test')

            recon_img = recon_img.detach().cpu()
            recon_seg = recon_seg.detach().cpu()
            real_img = real_img.detach().cpu()
            # refine_aff_seg = refined_aff_label.detach().cpu()

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

def g_label_get_aff_loss(inputs, targets):

    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):  
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

    # 若需要n_noise > 1，则返回n_noise个(batch, latent_dim, device=device)
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

def validate(args, generator, val_loader, device, writer, step):
    with torch.no_grad():
        generator.eval()

        val_scores = []
        for i, data in enumerate(val_loader):
            img, mask, rmask = data['image'].to(device), data['mask'].to(device), data['random_mask'].to(device)
            mask = (mask + 1.0) / 2.0

            recon_img, recon_seg, feas = generator(img, None, phase='val')

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

            val_dice = dice_coef(mask, mask_pred).cpu().detach().numpy()
            val_jaccard = iou_coef(mask, mask_pred).cpu().detach().numpy()
            val_scores.append([val_dice, val_jaccard])

        val_scores  = np.mean(val_scores, axis=0)
        mIoU = np.mean(val_scores[1])
        mdice = np.mean(val_scores[0])

        print("===========val miou scores: {0:.4f}, mdice: {1:.4f} ========================".format(mIoU, mdice))
        writer.add_scalar('scores/miou', mIoU, global_step=step)
        writer.add_scalar('scores/mdice', mdice, global_step=step)
        for i in range(val_scores.shape[1]):
            print("===========val {0} miou scores: {1:.4f} ========================".format(i, val_scores[1][i]))
            writer.add_scalar('scores/{} val_miou'.format(i), val_scores[1][i], global_step=step)
            print("===========val {0} mdice scores: {1:.4f} ========================".format(i, val_scores[0][i]))
            writer.add_scalar('scores/{} val_mdice'.format(i), val_scores[0][i], global_step=step)


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

def train(args, ckpt_dir, img_loader, seg_loader, seg_val_loader, generator, g_ema, percept, g_label_optim, device, writer):    # discriminator_mask,

    par = PAR(num_iter=15, dilations=[1, 2, 4, 8, 12, 24])

    seg_loader = sample_data(seg_loader)

    if args.seg_dim == 1:
        ce_loss_func = SoftBinaryCrossEntropyLoss(tau=0.3)
        dice_loss_func = DiceLoss(sigmoid_tau=0.3, include_bg=True)
    else:
        ce_loss_func = SoftmaxLoss(tau=1.0)
        dice_loss_func = DiceLoss(sigmoid_tau=1.0)

    pbar = range(args.iter)

    loss_dict = {}

    if args.distributed:
        g_module = generator.module
    else:
        g_module = generator
        
    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break
        
       
        seg_data = next(seg_loader)
        seg_img, seg_mask = seg_data['image'], seg_data['mask']
        seg_img, seg_mask = seg_img.to(device), seg_mask.to(device)
        g_module.train()

        fake_img, fake_seg, feature_label = g_module(seg_img, None)

        if args.seg_dim == 1:
            g_label_ce_loss = ce_loss_func(fake_seg, seg_mask)
        else:
            # make seg mask to label
            seg_mask_ce = torch.argmax(seg_mask, dim=1)
            seg_mask_dice = (seg_mask + 1.0) / 2.0
            g_label_ce_loss = ce_loss_func(fake_seg, seg_mask_ce)
            g_label_dice_loss = dice_loss_func(fake_seg, seg_mask_dice)


        g_label_loss = g_label_ce_loss+g_label_dice_loss

        loss_dict['g_label_ce'] = g_label_ce_loss
        loss_dict['g_label_dice'] = g_label_dice_loss
        loss_dict['g_label'] = g_label_loss

        g_module.zero_grad()
        g_label_loss.backward()

        if args.profile_grad_norm == True:
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in g_module.parameters()]), 2.0)
            print("e_label_average grad norm: {0:.6f}".format(total_norm))

        if args.no_grad_clip != True:
            # gradient clipping
            clip_grad_norm_(g_module.parameters(), args.label_grad_clip)

        g_label_optim.step()
        update_learning_rate(args, i, g_label_optim)

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        g_label_ce_loss_val = loss_reduced['g_label_ce'].mean().item()
        g_label_dice_loss_val = loss_reduced['g_label_dice'].mean().item()
        g_label_loss_val = loss_reduced['g_label'].mean().item()

        if get_rank() == 0:
            # write to tensorboard
            writer.add_scalar('g_labe/dice_loss', g_label_dice_loss_val, global_step=i)
            writer.add_scalar('g_label/ce_loss', g_label_ce_loss_val, global_step=i)

            writer.add_scalar('g_label/total_loss', g_label_loss_val, global_step=i)

            # learning rate
            writer.add_scalar('lr/g_label_lr', g_label_optim.param_groups[0]['lr'], global_step=i)

            writer.add_scalar('loss/g_label', g_label_loss_val, global_step=i)
            if i % args.viz_every == 0:
                with torch.no_grad():
            
                    g_module.eval()
                    for j, data in enumerate(seg_val_loader):
                            sample_img, sample_seg, feas = g_module(data['image'].to(device),None) 
                            break
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
                        sample_mask,
                        os.path.join(ckpt_dir, f'sample/mask_{str(i).zfill(6)}.png'),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,

                    )

        

            if i %  args.eval_every == 0:  #
                print("==================Start calculating validation scores==================")
                validate(args,  g_module, seg_val_loader, device, writer, i)
                
            if i % args.save_every == 0:
                
                
                os.makedirs(os.path.join(ckpt_dir, 'ckpt'), exist_ok=True)
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'args': args,
                        'g_label_optim': g_label_optim.state_dict(),
                    },
                    os.path.join(ckpt_dir, f'ckpt/{str(i).zfill(6)}.pt'),
                )

def get_seg_dataset(args, phase='train'):
    if args.seg_name == 'celeba-mask':
        seg_dataset = CelebAMaskDataset(args, args.seg_dataset, is_label=True, phase=phase,
                                            limit_size=args.limit_data, aug=args.seg_aug if phase=='train' else False, resolution=args.size)
   
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
    torch.cuda.set_device(1)  
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dataset', default='data', type=str)
    parser.add_argument('--seg_dataset', default='data', type=str)
    parser.add_argument('--inception', type=str, default='data/inception.txt', help='inception pkl')


    parser.add_argument('--seg_name', type=str, help='segmentation dataloader name[celeba-mask]', default='celeba-mask')
    parser.add_argument('--iter', type=int, default=15000)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--val_batch', type=int, default=8)
    parser.add_argument('--n_sample', type=int, default=1)     
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--radius', type=int, default=8)       
    parser.add_argument('--ignore_index', type=int, default=255)  
    parser.add_argument('--seg_prob_lower', type=int, default=0.35)
    parser.add_argument('--seg_prob_higher', type=int, default=0.55)

    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--d_use_seg_every', type=int, help='frequency mixing seg image with real image', default=-1)
    parser.add_argument('--viz_every', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=5000)

    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--lambda_dseg_feat', type=float, default=2.0)
    parser.add_argument('--ckpt', type=str, default='')
    # parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', type=float, default=0.00002)
    parser.add_argument('--lambda_lr', type=float, default=0.999)
    parser.add_argument('--lr_decay_iter_start', type=int, default=30000)
    parser.add_argument('--lr_decay_iter_end', type=int, default=100000)
    parser.add_argument('--channel_multiplier', type=int, default=2)

    parser.add_argument('--lambda_label_lpips', type=float, default=1.0)   #original 10.0
    parser.add_argument('--lambda_label_mse', type=float, default=1.0)   # original 1.0
    parser.add_argument('--lambda_label_ce', type=float, default=1.0)    # original 1.0
    parser.add_argument('--lambda_label_depth', type=float, default=1.0)
    parser.add_argument('--lambda_label_dice', type=float, default=1.0)    # original 1.0
    parser.add_argument('--lambda_label_latent', type=float, default=0.0)
    parser.add_argument('--lambda_label_adv', type=float, default=0.1)

    parser.add_argument('--lambda_unlabel_lpips', type=float, default=1.0)
    parser.add_argument('--lambda_unlabel_mse', type=float, default=1.0)
    parser.add_argument('--lambda_unlabel_adv', type=float, default=0.1)

    parser.add_argument('--lamda_ssim', type=float, default=10, help='weight of the SSIM loss')  #original 1.0
    parser.add_argument('--lamda_tv', type=float, default=20, help='weight of the tv loss')
    
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

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/TNUS/gan')

    parser.add_argument('--local_rank', type=int, default=0)

    # 关于Swin_ tranformer的相关设置
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    
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


    # percep
    percept = PerceptualLoss(layer_weights=dict(conv4_4=1 / 4, conv5_4=1 / 2)).to(device)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)

    g_label_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)
        
        ckpt = torch.load(args.ckpt, map_location={'cuda:1': 'cpu'})
            
        generator.load_state_dict(ckpt['g'], strict=False)
       
    
    g_ema = GeneratorSeg(
        args, args.size, args.latent, args.n_mlp, seg_dim=args.seg_dim,
        image_mode=args.image_mode, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)


    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
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
    
    train(args, ckpt_dir, img_loader, seg_loader, seg_val_loader, generator, g_ema, percept, g_label_optim, device, writer)
