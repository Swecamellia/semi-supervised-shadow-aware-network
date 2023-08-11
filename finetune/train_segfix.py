from tkinter.messagebox import NO
from utils.loss import SegFixLoss
import numpy as np
import pandas as pd
from glob import glob
import os, shutil
from torch import norm, tensor
from tqdm import tqdm
tqdm.pandas()
import time
import copy
from collections import defaultdict
# from IPython import display as ipd
import wandb

# PyTorch 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp

import segmentation_models_pytorch as smp
from config import CFG
from utils.utils import set_seed, fetch_scheduler, to_one_hot
from utils.metric import dice_coef, iou_coef
from dataset.dataset_segfix import prepare_train_loaders, BuildDataset
from semanticGAN.losses import SoftmaxLoss, SoftBinaryCrossEntropyLoss, DiceLoss
from models.model_segfix import GeneratorSeg
from models.segfix import segfix

from matplotlib import pyplot as plt

# For colored terminal text
from colorama import Fore, Back, Style
c_  = Fore.GREEN
sr_ = Style.RESET_ALL

torch.cuda.set_device(0)

def train_one_epoch(model, optimizer, scheduler, criterion, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    dataset_size = 0
    running_loss = 0.0
    a = 1
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train')
    for step, (images, masks, boundary, direction) in pbar:
        masks_one_hot = to_one_hot(masks, CFG.num_classes) 
        masks = masks[:, 0, ...]
        images = images.to(device, dtype=torch.float)
        random_mask = torch.ones_like(images[:, :1, ...]).to(device, dtype=torch.float)
        masks  = masks.long().to(device)
        masks_one_hot = masks_one_hot.to(device, dtype=torch.float)
        boundary, direction = boundary.to(device, dtype=torch.float), direction.to(device, dtype=torch.float)
        batch_size = images.size(0)
        
        # with amp.autocast(enabled=True):
        y_pred, bdr_pred, dir_pred = model(images, None)
        loss1  = criterion[0](y_pred, masks)
        loss2  = criterion[1](y_pred, masks_one_hot)
        loss3  = criterion[2]((bdr_pred, dir_pred), (masks, boundary, direction))

        offset = model.get_offset(bdr_pred, dir_pred)

        refiner = model.refiner(y_pred, offset)
        refiner = torch.softmax(refiner, dim=1)
        y_mask = torch.argmax(refiner, dim=1)


        loss4  = criterion[0](refiner, masks)
        loss5  = criterion[1](refiner, masks_one_hot)

        loss = 0.5*(loss1+loss2)+loss3+1*(loss4+loss5)

        loss  = loss / CFG.n_accumulate
            
        scaler.scale(loss).backward()
    
        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, criterion, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    val_scores = []
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid')
    for step, (images, masks, boundary, direction) in pbar: 
        # plot_batch(images, masks)   
        masks_one_hot = to_one_hot(masks, CFG.num_classes) 
        masks = masks[:, 0, ...]
        images = images.to(device, dtype=torch.float)
        random_mask = torch.ones_like(images[:, :1, ...]).to(device, dtype=torch.float)
        masks  = masks.long().to(device)
        masks_one_not = masks_one_hot.long().to(device)
        boundary, direction = boundary.to(device, dtype=torch.float), direction.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        y_pred, bdr_pred, dir_pred = model(images, None)
        
        loss1 = criterion[0](y_pred, masks)
        loss2 = criterion[1](y_pred, masks_one_not)
        loss3 = criterion[2]((bdr_pred, dir_pred), (masks, boundary, direction))
        loss = 1*(loss1 + loss2) + loss3

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        offset = model.get_offset(bdr_pred, dir_pred)

        refiner = model.refiner(y_pred, offset)
        refiner = torch.softmax(refiner, dim=1)

       
        val_dice = dice_coef(masks_one_not, refiner).cpu().detach().numpy()
        val_jaccard = iou_coef(masks_one_not, refiner).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                        lr=f'{current_lr:0.5f}',
                        gpu_memory=f'{mem:0.2f} GB')
    val_scores  = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    return epoch_loss, val_scores

def run_training(model, optimizer, scheduler, criterion, train_loader, valid_loader,
                run, device, num_epochs):
    # To automatically log gradients
    if CFG.wandb_log:
        wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_dice = np.inf
    best_dice = 0.0
    best_epoch = -1
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        print(f'Epoch {epoch}/{num_epochs}')
        train_loss = train_one_epoch(model, optimizer, scheduler, criterion,
                                           dataloader=train_loader,
                                           device=CFG.device, epoch=epoch)
        # train_loss = 0.0
        val_loss, val_scores = valid_one_epoch(model, criterion, valid_loader, 
                                                 device=CFG.device, 
                                                 epoch=epoch)
        
               
        val_dice, val_jaccard = val_scores
    
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        
        # Log the metrics
        if CFG.wandb_log:
            wandb.log({"Train Loss": train_loss, +
                    "Valid Loss": val_loss,
                    "Valid Dice": val_dice,
                    "Valid Jaccard": val_jaccard,
                    "LR":scheduler.get_last_lr()[0]})
            
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        if not os.path.exists(os.path.join(CFG.save_path, CFG.model_name)):
            os.makedirs(os.path.join(CFG.save_path, CFG.model_name))
        save_path = os.path.join(CFG.save_path, CFG.model_name)
        # deep copy the model
        if val_dice > best_dice:
            print(f"{c_}Vallogistsid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice    = val_dice
            # best_dice = val_loss
            best_jaccard = val_jaccard
            best_epoch   = epoch

            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"./{save_path}/best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            if CFG.wandb_log:
                run.summary["Best Dice"]    = best_dice
                run.summary["Best Jaccard"] = best_jaccard
                run.summary["Best Epoch"]   = best_epoch
                wandb.save(f"best_epoch-{fold:02d}.bin")
            print(f"Model Saved{sr_}")
            
        
        PATH = f"./{save_path}/last_epoch-{fold:02d}.bin"
        torch.save(model.state_dict(), PATH)
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_dice))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history
 
if __name__ == '__main__':
    set_seed(CFG.seed)
    for fold in range(1):
        print(f'#'*15)
        print(f'### Fold: {fold}')
        print(f'#'*15)

        if CFG.wandb_log:
            os.environ['WANDB_MODE'] = "offline"
            run = wandb.init(project="uncertainty_segmentation", entity="xieyt",
                            config={k:v for k, v in dict(vars(CFG)).items() if '__' not in k},
                            name=f"fold-{fold}|dim-{CFG.img_size[0]}x{CFG.img_size[1]}|model-{CFG.model_name}",
                            group=CFG.comment
                            )
        else: run = None
        train_loader, valid_loader = prepare_train_loaders(fold=fold, cfg=CFG)

        model = GeneratorSeg(3, 2)
      
        model.to(CFG.device)    
        if CFG.pretrain:
            with torch.no_grad():
                
                d = torch.load(CFG.pretrain, map_location={'cuda:1': 'cpu'})
               
                model.load_state_dict(d)
                
        optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=CFG.lr, weight_decay=CFG.wd)
        scheduler = fetch_scheduler(optimizer)
        loss_func = [SoftmaxLoss(tau=1.0), DiceLoss(sigmoid_tau=1.0), SegFixLoss()]

        model, history = run_training(model, optimizer, scheduler, loss_func,
                                    train_loader, valid_loader,
                                    run, device=CFG.device,
                                    num_epochs=CFG.epochs)
        if CFG.wandb_log:
            run.finish()
