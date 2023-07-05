from resnet1d import Res34SimSiam, Res34SimSiamSplitHeads, Res50SimSiam
from dataset import Dataset_per_file, Dataset_whole_limited_labeled
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
import os


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
ECG_LR = 1e-4
LAMBDA = 1.0
subset = 0
DIM1, DIM2 = 512, 128
PREDICTOR = True
LABEL_PERC=0.05
comment = ''
MODEL_FOLDER = f'corrected_resnet_34_epoch_{NUM_EPOCHS}_ecglr_{ECG_LR}_ppglr_{PPG_LR}_lambda_{LAMBDA}_dim1_{DIM1}_dim2_{DIM2}_pred_{PREDICTOR}_labelperc_{LABEL_PERC}_{comment}'
os.mkdir(f'saved_models/'+MODEL_FOLDER)


print_flush('BATCH_SIZE', BATCH_SIZE)
print_flush('NUM_EPOCHS', NUM_EPOCHS)
print_flush('device', device)
print_flush('PPG_LR', PPG_LR)
print_flush('ECG_LR', ECG_LR)
print_flush('LAMBDA', LAMBDA)
print_flush('subset', subset)
print_flush('DIM1', DIM1)
print_flush('DIM2', DIM2)
print_flush('LABEL_PERC', LABEL_PERC)
print_flush('comment', comment)
print_flush('MODEL_FOLDER', MODEL_FOLDER)



def tell_time(tdelta):
    # minutes, seconds = divmod(tdelta.seconds, 60)
    # hours, minutes = divmod(minutes, 60)
    # # millis = round(tdelta.microseconds/1000, 0)
    # return f"{hours}:{minutes:02}:{seconds:02}"
    return tdelta


def train_epoch(epoch_idx, model, cos_loss, ce_loss_fn, optimizer, train_loader, lambda_):
    with torch.autograd.set_detect_anomaly(True):
        train_loss = 0
        simsam_losses = 0
        ce_losses = 0
        ECG_f1s = 0
        PPG_f1s = 0

        tstart = datetime.now()
        for batch_idx, data in enumerate(train_loader):
            batch_tstart = datetime.now()

            ECG, PPG, ECG_classification_head, PPG_classification_head, target = data
            ECG = ECG.cuda()
            PPG = PPG.cuda()
            ECG_classification_head = ECG_classification_head.cuda()
            PPG_classification_head = PPG_classification_head.cuda()
            target = target.cuda()

            p1, p2, z1, z2, ECG_pred, PPG_pred = model(ECG, PPG, ECG_classification_head, PPG_classification_head)
            
            simsam_loss = cos_loss(p1, z2) / 2 + cos_loss(p2, z1) / 2
            ecg_ce_loss = ce_loss_fn(ECG_pred, target)
            ppg_ce_loss = ce_loss_fn(PPG_pred, target)
            
            # print_flush(torch.min(ECG_pred).item(), torch.min(PPG_pred).item(), ecg_ce_loss.item(), ppg_ce_loss.item())

            total_loss = lambda_*simsam_loss + ecg_ce_loss + ppg_ce_loss
            train_loss += total_loss.item()
            simsam_losses += simsam_loss.item()
            ce_losses += (ecg_ce_loss.item()+ppg_ce_loss.item())
            
            model.zero_grad()
            
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            ECG_f1 = f1_score(target.detach().cpu().numpy(), ECG_pred.argmax(1).detach().cpu().numpy())
            PPG_f1 = f1_score(target.detach().cpu().numpy(), PPG_pred.argmax(1).detach().cpu().numpy())

            ECG_f1s += ECG_f1
            PPG_f1s += PPG_f1

            batch_tend = datetime.now()

            if batch_idx % 100 == 0:
                print_flush(f'\t[TRAIN] Epoch {epoch_idx} Batch {batch_idx}/{len(train_loader)} Loss: {train_loss / (batch_idx + 1)}|loss atm:{simsam_loss.item()}+{ecg_ce_loss.item()+ppg_ce_loss.item()}, \tECG F1: {ECG_f1s / (batch_idx + 1)}, PPG F1: {PPG_f1s / (batch_idx + 1)}, \tBatch Avg-T: {(batch_tend - tstart) / (batch_idx + 1)}')

        return {train_loss / (batch_idx + 1)}


def eval_epoch(epoch_idx, model, cos_loss, ce_loss_fn, val_loader, lambda_):

    with torch.no_grad():
       
        val_loss = 0
        simsam_losses = 0
        ce_losses = 0

        PPG_preds = None
        ECG_preds = None
        all_targets = None
        
        model.eval()
        tstart = datetime.now()

        for batch_idx, data in enumerate(val_loader):

            ECG, PPG, ECG_classification_head, PPG_classification_head, target = data
            ECG = ECG.cuda()
            PPG = PPG.cuda()
            ECG_classification_head = ECG_classification_head.cuda()
            PPG_classification_head = PPG_classification_head.cuda()
            target = target.cuda()

            p1, p2, z1, z2, ECG_pred, PPG_pred = model(ECG, PPG, ECG_classification_head, PPG_classification_head)
            
            simsam_loss = cos_loss(p1, z2) / 2 + cos_loss(p2, z1) / 2
            ecg_ce_loss = ce_loss_fn(ECG_pred, target)
            ppg_ce_loss = ce_loss_fn(PPG_pred, target)
            
            total_loss = lambda_*simsam_loss + ecg_ce_loss + ppg_ce_loss
            val_loss += total_loss.item()
            simsam_losses += simsam_loss.item()
            ce_losses += (ecg_ce_loss.item()+ppg_ce_loss.item())
            
            PPG_predicted = PPG_pred.argmax(1)
            ECG_predicted = ECG_pred.argmax(1)

            if PPG_preds == None:
                PPG_preds = PPG_predicted
                ECG_preds = ECG_predicted
                all_targets = target
            else:
                PPG_preds = torch.cat((PPG_preds, PPG_predicted))
                ECG_preds = torch.cat((ECG_preds, ECG_predicted))
                all_targets = torch.cat((all_targets, target))
        tend = datetime.now()
        print_flush(f'[VAL] Epoch {epoch_idx} Loss: {val_loss / (batch_idx + 1)} simsam loss : {simsam_losses / (batch_idx + 1)}, ce loss : {ce_losses / (batch_idx + 1)}, \tPPG F1: {f1_score(all_targets.detach().cpu().numpy(), PPG_preds.detach().cpu().numpy())}, ECG F1: {f1_score(all_targets.detach().cpu().numpy(), ECG_preds.detach().cpu().numpy())}')
    return val_loss / (batch_idx + 1)


def train(num_epochs, model, cos_loss, ce_loss_fn, optimizer, train_loader, val_loader, lambda_=1):

    best_val_loss = 99999999999999

    for epoch_idx in range(num_epochs):
        print_flush(f'Epoch {epoch_idx} training...')
        tstart = datetime.now()
        train_loss = train_epoch(epoch_idx, model, cos_loss, ce_loss_fn, optimizer, train_loader, lambda_)
        val_loss = eval_epoch(epoch_idx, model, cos_loss, ce_loss_fn, val_loader, lambda_)
        tend = datetime.now()
        print_flush(f'Epoch {epoch_idx} finished. t = {tell_time(tend-tstart )}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("Saving...")
            torch.save(model.state_dict(), f"saved_models/{MODEL_FOLDER}/model_{epoch_idx}.pt")

        print_flush('\n')
    

def cos_loss(p, z):
    p = F.normalize(p, dim=1) # l2-normalize 
    z = F.normalize(z, dim=1) # l2-normalize 
    return -(p*z).sum(dim=1).mean()


if __name__=='__main__':

    '''DATALOADERS'''
    print_flush('Creating datasets')
    train_dataset = Dataset_whole_limited_labeled('/labs/hulab/stark_stuff/ppg_ecg_project/data/', split='train', labeled_perc=LABEL_PERC)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dataset = Dataset_whole_limited_labeled('/labs/hulab/stark_stuff/ppg_ecg_project/data/', split='val', labeled_perc=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print_flush('Dataset finished')
    
    model = Res34SimSiam(DIM1, DIM2, predictor=PREDICTOR)
    # model = Res34SimSiamSplitHeads(DIM1, DIM2)
    # model = Res50SimSiam(DIM1, DIM2, predictor=PREDICTOR)
    model = nn.DataParallel(model)
    model.to(device)

    ce_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    train(NUM_EPOCHS, model, cos_loss, ce_loss_fn, optimizer, train_loader, val_loader, lambda_=LAMBDA)