from resnet1d import Resnet34
from resnet_zoo import *
from dataset import Dataset_per_file, Dataset_PPG, Dataset_ori, Dataset_PPG_limited_labeled
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.autograd import Variable
import os
import numpy as np
from tqdm import tqdm
import argparse
import random
from sklearn.metrics import f1_score, accuracy_score
import sys


'''PREFLIGHT SETUP'''
from functools import partial
print_flush = partial(print, flush=True)
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
'''PREFLIGHT SETUP'''


'''HYPER PARAMS'''
BATCH_SIZE = 5000
NUM_EPOCHS = 30
device = 'cuda'
PPG_LR = 1e-4
LAMBDA = 0.9
subset = 0
LABEL_PERC = 0.05
COMMENT = ''
MODEL_FOLDER = f'corrected_res34_epoch_{NUM_EPOCHS}_ppglr_{PPG_LR}_lambda_{LAMBDA}_LABEL_PERC_{LABEL_PERC}_{COMMENT}'
os.mkdir(f'saved_models/'+MODEL_FOLDER)

print_flush('BATCH_SIZE', BATCH_SIZE)
print_flush('NUM_EPOCHS', NUM_EPOCHS)
print_flush('device', device)
print_flush('PPG_LR', PPG_LR)
print_flush('LAMBDA', LAMBDA)
print_flush('subset', subset)
print_flush('COMMENT', COMMENT)
print_flush('LABEL_PERC', LABEL_PERC)
print_flush('MODEL_FOLDER', MODEL_FOLDER)


def tell_time(tdelta):
    # minutes, seconds = divmod(tdelta.seconds, 60)
    # hours, minutes = divmod(minutes, 60)
    # # millis = round(tdelta.microseconds/1000, 0)
    # return f"{hours}:{minutes:02}:{seconds:02}"
    return tdelta


def train_epoch(epoch_idx, PPG_model, ce_loss_fn, PPG_optimizer, train_loader, lambda_):

    train_loss = 0

    PPG_f1s = 0
    
    tstart = datetime.now()
    for batch_idx, data in enumerate(train_loader):
        batch_tstart = datetime.now()

        PPG, target = data
        # PPG = PPG.to(device)
        # target = target.to(device)
        PPG = PPG.to(device).float()
        target = target.to(device).long()

        if torch.isinf(PPG).any():
            print('invalid PPG detected at iteration ', epoch_idx, batch_idx)
            # continue

        PPG_feature, PPG_out = PPG_model(PPG)

        PPG_loss = ce_loss_fn(PPG_out, target)
        PPG_optimizer.zero_grad()
        PPG_loss.backward()
        PPG_optimizer.step()


        total_loss = PPG_loss
        train_loss += total_loss.item()

        PPG_predicted = PPG_out.argmax(1)

        PPG_f1 = f1_score(target.detach().cpu().numpy(), PPG_predicted.detach().cpu().numpy())

        PPG_f1s += PPG_f1

        batch_tend = datetime.now()
        # print_flush(batch_tend - batch_tstart)

        if batch_idx % 100 == 0:
            print_flush(f'\t[TRAIN] Epoch {epoch_idx} Batch {batch_idx}/{len(train_loader)} Loss: {train_loss / (batch_idx + 1)}, \tPPG F1: {PPG_f1s / (batch_idx + 1)}, \tBatch Avg-T: {(batch_tend - tstart) / (batch_idx + 1)}')

    # f1_score(y_true, y_pred
    print_flush(f'[TRAIN] Epoch {epoch_idx} Loss: {train_loss / len(train_loader)}, \
            \tPPG F1: {PPG_f1s / len(train_loader)}')

    tend = datetime.now()

    print_flush(f'Time - {tell_time(tend - tstart)}')

    return train_loss / (batch_idx + 1)

def eval_epoch(epoch_idx, PPG_model, ce_loss_fn, val_loader, lambda_):
    with torch.no_grad():
       
        val_loss = 0

        PPG_preds = None
        all_targets = None
        
        PPG_model.eval()
        tstart = datetime.now()

        for batch_idx, data in enumerate(val_loader):
            PPG, target = data
            # PPG = PPG.to(device)
            # target = target.to(device)

            PPG = PPG.to(device).float()
            target = target.to(device).long()

            PPG_feature, PPG_out = PPG_model(PPG)

            PPG_loss = ce_loss_fn(PPG_out, target)

            total_loss = PPG_loss
            
            val_loss += total_loss.item()
            PPG_predicted = PPG_out.argmax(1)

            if PPG_preds == None:
                PPG_preds = PPG_predicted
                all_targets = target
            else:
                PPG_preds = torch.cat((PPG_preds, PPG_predicted))
                all_targets = torch.cat((all_targets, target))
        tend = datetime.now()
        print_flush(f'[VAL] Epoch {epoch_idx} Loss: {val_loss / (batch_idx + 1)}, \tPPG F1: {f1_score(all_targets.detach().cpu().numpy(), PPG_preds.detach().cpu().numpy())}')
    return val_loss / (batch_idx + 1)


def train(num_epochs, PPG_model, ce_loss_fn, PPG_optimizer, train_loader, val_loader, lambda_=1):

    best_val_loss = 99999999999999

    for epoch_idx in range(num_epochs):
        print_flush(f'Epoch {epoch_idx} training...')
        tstart = datetime.now()
        train_loss = train_epoch(epoch_idx, PPG_model, ce_loss_fn, PPG_optimizer, train_loader, lambda_)
        val_loss = eval_epoch(epoch_idx, PPG_model, ce_loss_fn, val_loader, lambda_)
        tend = datetime.now()
        print_flush(f'Epoch {epoch_idx} finished. t = {tell_time(tend-tstart)}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving...")
            torch.save(PPG_model.state_dict(), f"saved_models/{MODEL_FOLDER}/PPG_best_{epoch_idx}.pt")

        print_flush('\n')
    

if __name__=='__main__':

    '''DATALOADERS'''
    print_flush('Creating datasets')
    train_dataset = Dataset_PPG_limited_labeled('/labs/hulab/stark_stuff/ppg_ecg_project/data/', split='train', labeled_perc=LABEL_PERC)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataset = Dataset_PPG('/labs/hulab/stark_stuff/ppg_ecg_project/data/', split='val', subset=subset)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print_flush('Dataset finished')

    PPG_model = Resnet34()
    # PPG_model = ResNet50()
    PPG_model = nn.DataParallel(PPG_model)
    PPG_model.to(device)

    ce_loss_fn = nn.CrossEntropyLoss()

    PPG_optimizer = optim.Adam(PPG_model.parameters(), lr=PPG_LR)

    train(NUM_EPOCHS, PPG_model, ce_loss_fn, PPG_optimizer, train_loader, val_loader, lambda_=LAMBDA)