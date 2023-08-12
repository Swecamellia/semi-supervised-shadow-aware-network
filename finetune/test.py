import os.path

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.modules import distance
from hausdorff import hausdorff_distance
import surface_distance as surfdist
from torch import mode
from torch import imag
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from config import CFG
from collections import OrderedDict
# from utils.display import plot_batch, create_animation
from dataset.dataset_segfix import prepare_test_loaders
from torch.utils.data import DataLoader
from utils import losses, metrics, ramps
# from util.metric import dice_coef, iou_coef
from torch.nn.modules.loss import CrossEntropyLoss
from PIL import Image
from models.model_segfix import GeneratorSeg
import matplotlib.pyplot as plt

def batch_pix_accuracy(output, target):
    _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target) * (target > 0)).sum()
    assert pixel_correct <= pixel_labeled, 
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

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)
        print("dic")
        print(self.sum / self.count)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :] * targets[:, class_index, :, :])
            union = torch.sum(logits[:, class_index, :, :]) + torch.sum(targets[:, class_index, :, :])
            dice = (2. * inter + 1) / (union + 1 )#
            dices.append(dice.item())
        return np.asarray(dices)

class JaccAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = JaccAverage.get_iou(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)
        print("jac")
        print(self.sum / self.count)

    @staticmethod
    def get_iou(logits, targets):
        jaccard = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :] * targets[:, class_index, :, :])
            union = torch.sum(logits[:, class_index, :, :]) + torch.sum(targets[:, class_index, :, :])
            iou = (inter + 1) / (union - inter + 1)
            jaccard.append(iou.item())
        return np.asarray(jaccard)

class AccAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([[0, 0, 0] for _ in range(self.class_num)], dtype='float64')
        self.avg = np.asarray([[0, 0, 0] for _ in range(self.class_num)], dtype='float64')
        self.sum = np.asarray([[0, 0, 0] for _ in range(self.class_num)], dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = AccAverage.get_acc(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)
        print("acc")
        print(self.sum / self.count)

    @staticmethod
    def get_acc(logits, targets):
        output = logits
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        value = []
        for i in range(targets.size()[1]):
            output_ind = output[:, i, ...]
            targets_ind = targets[:, i, ...]
            for class_index in range(1, 2):
                true_positives = torch.tensor([((output_ind == class_index) * (targets_ind == class_index)).sum()]).float()
                true_negatives = torch.tensor([((output_ind != class_index) * (targets_ind != class_index)).sum()]).float()
                false_positives = torch.tensor([((output_ind == class_index) * (targets_ind != class_index)).sum()]).float()
                false_negatives = torch.tensor([((output_ind != class_index) * (targets_ind == class_index)).sum()]).float()

                precision = (true_positives + 1) / (true_positives + false_positives + 1)
                recall = (true_positives + 1) / (true_positives + false_negatives + 1)
                classwise_f1 = 2 * (precision * recall) / (precision + recall)
                acc = (true_positives + true_negatives + 1) / (true_positives + true_negatives + false_positives + false_negatives + 1)

                value.append([acc.item(), precision.item(), recall.item()])

        return np.asarray(value)

class HDAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()


    def reset(self):
        self.value = np.asarray([[0, 0] for _ in range(self.class_num)], dtype='float64')
        self.avg = np.asarray([[0, 0] for _ in range(self.class_num)], dtype='float64')
        self.sum = np.asarray([[0, 0] for _ in range(self.class_num)], dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = HDAverage.get_hd(logits, targets)
        if self.value != []:
            self.sum += self.value
            self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # return self.avg
        # print(self.value)
        print("hd5")
        print(self.sum / self.count)

    @staticmethod
    def get_hd(logits, targets):
        output = torch.argmax(logits, dim=1).cpu().numpy()
        value = []
        for i in range(output.shape[0]):
            for j in range(targets.size()[1]):
                targets_idx = targets[:, j, :, :].cpu().numpy()
                t = np.zeros((targets_idx.shape[1]+2, targets_idx.shape[2]+2))
                t[1:-1, 1:-1] = targets_idx[i, :, :]
                t = np.array(t, dtype=bool)
                o = np.zeros((output.shape[1]+2, output.shape[2]+2))
                o[1:-1, 1:-1] = output[i, :, :]
                o = np.array(o, dtype=bool)
                if np.sum(o) != 0 and np.sum(t) != 0:
                    surface_distances = surfdist.compute_surface_distances(t, o, spacing_mm=(1.0, 1.0))
                    asd = surfdist.compute_average_surface_distance(surface_distances)
                    hd_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
                    value.append([hd_95, np.mean(np.array(asd))])
        value = np.asarray(value)

        return value

def to_one_hot(mask, num_class):
    y_one_hot = torch.zeros((mask.shape[0], num_class, mask.shape[2], mask.shape[3]))
    y_one_hot = y_one_hot.scatter(1, mask.long(), 1).long()
    return y_one_hot
        
def val(model, val_loader, loss_func, n_labels, patch_size=[256, 256], test_save_path=None):
    # model.eval()

    val_loss1 = LossAverage()
    val_loss2 = LossAverage()
    val_dice = DiceAverage(n_labels)
    val_jacc = JaccAverage(n_labels)
    val_acc = AccAverage(n_labels)
    val_hd = HDAverage(n_labels)

    total_inter, total_union = 0.0, 0.0
    total_correct, total_label = 0.0, 0.0

    with torch.no_grad():
        for idx, sample in tqdm(enumerate(val_loader),total=len(val_loader)):
            data = sample[0]
            target = sample[1]
            case_list = sample[4][0].split('/')[-3:]
            case_list[-1] = case_list[-1].split('.')[0]
            case = '_'.join(case_list)

            tar = target.cuda()
            target = to_one_hot(target, n_labels)
            data, target = data.cuda(), target.cuda()

            y_pred, bdr_pred, dir_pred = model(data, None)
            if 0:
                ax, fig = plt.subplots(2, 3)
                fig[0, 0].imshow(tar[0].squeeze(0).cpu().numpy())
                fig[0, 1].imshow(torch.argmax(torch.softmax(y_pred, dim=1), dim=1).squeeze(0).cpu().numpy())
                fig[0, 2].imshow(data[0][:1, ...].squeeze(0).cpu().numpy()*255)
                plt.show()
            offset = model.get_offset(bdr_pred, dir_pred)
            output = model.refiner(y_pred, offset)

            loss=loss_func[0](output, tar.squeeze(0).long())
            loss2 = 5 * loss_func[1](torch.softmax(output, dim=1), tar)
            
            if 0:
                ax, fig = plt.subplots(2, 3)
                fig[0, 0].imshow(tar[0].squeeze(0).cpu().numpy())
                fig[0, 1].imshow(torch.argmax(torch.softmax(output[0], dim=0), dim=0).cpu().numpy())
                plt.show()

            predict = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()*50
            pred_img = Image.fromarray(predict.astype(np.uint8))

            val_loss1.update(loss.item(),data.size(0))
            val_loss2.update(loss2.item(),data.size(0))
            val_dice.update(torch.softmax(output, dim=1), target)
            val_jacc.update(torch.softmax(output, dim=1), target)
            val_acc.update(torch.softmax(output, dim=1), target)
            val_hd.update(torch.softmax(output, dim=1), target)

            correct, labeled, inter, union = eval_metrics(torch.softmax(output, dim=1), torch.argmax(target, dim=1), n_labels, -100)
            total_inter, total_union = total_inter + inter, total_union + union
            total_correct, total_label = total_correct + correct, total_label + labeled

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        DSC = (2.0 * total_inter + np.spacing(1)) / (total_union + total_inter + np.spacing(1))
        mIoU = IoU.mean()
        mDSC = DSC.mean()

    val_log = OrderedDict({'Val_Loss1': val_loss1.avg, 'Val_Loss2': val_loss2.avg, 'Val_dice': val_dice.avg[1], 'Val_Jacc': val_jacc.avg[1], 'Val_acc': val_acc.avg[0],
                    'Val_prec': val_acc.avg[1], 'Val_recal': val_acc.avg[0], 'Val_hd95': val_hd.avg[0], 'Val_Asd': val_hd.avg[1]})
    print(val_log)

    print(
        "===========val miou scores: {0:.4f}, pixel acc: {1:.4f}, val mdsc scores: {2:.4f} ========================".format(
            mIoU, pixAcc, mDSC))
    for i in range(IoU.shape[0]):
        print("===========val {0} miou scores: {1:.4f} ========================".format(i, IoU[i]))
        print("===========val {0} mdsc scores: {1:.4f} ========================".format(i, DSC[i]))

    if n_labels==3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log

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

from scipy.ndimage import zoom
@torch.no_grad()
def test_epoch(model, dataloader, device, display=False, patch_size=[256, 256]):
    model.eval()
    
    val_scores = []
    imgs, msks, preds = [], [], []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='test ')
    for step, sample in pbar:
        images = sample[0]
        masks = sample[1]
        if display:
            imgs.append(images)
            msks.append(masks)

        masks = to_one_hot(masks, CFG.num_classes) 
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        
        y_pred, bdr_pred, dir_pred = model(images, None)
        offset = model.get_offset(bdr_pred, dir_pred)
        y_pred = model.refiner(y_pred, offset)

        if display:
            preds.append(y_pred)
        
        y_pred = torch.softmax(y_pred, dim=1)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
    val_scores  = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    print(f"dice_coef : {val_scores[0]},  iou_coef : {val_scores[1]}")
    if display:
        imgs = torch.cat(imgs, dim=0).permute(0, 2, 3, 1).cpu().detach().numpy()
        msks = torch.cat(msks,dim=0).permute(0, 2, 3, 1).cpu().detach().numpy()
        preds = torch.cat(preds, dim=0).permute(0, 2, 3, 1).cpu().detach().numpy()
        # create_animation(imgs, msks, preds)

if __name__ == '__main__':
    preds = []
    for fold in range(1):
        checkpoint = f""
        model = GeneratorSeg(3, 2)

        model.to(CFG.device)
        d = torch.load(checkpoint, map_location={'cuda:0': 'cpu'})
        model.load_state_dict(d)
        
        test_loader = prepare_test_loaders(cfg=CFG)
        save_dir = ''
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        loss_func = [CrossEntropyLoss(), losses.DiceLoss(CFG.num_classes)]
        val(model, test_loader, loss_func, CFG.num_classes, test_save_path=save_dir)

