import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import Dataset_ori
import pickle as pkl
import torch
import random
import gc
from datetime import datetime
import psutil
from resnet1d import Resnet34
from sklearn.metrics import f1_score, accuracy_score

from functools import partial
print_flush = partial(print, flush=True)


device = 'cuda'

def test_epoch(PPG_model, test_loader):
    with torch.no_grad():
       

        PPG_preds = None
        all_targets = None
        
        PPG_model.eval()
        tstart = datetime.now()

        for batch_idx, (PPG, target) in enumerate(test_loader):
            print(PPG.shape)
            PPG = PPG.to(device).float()
            target = target.to(device).long()

            PPG_feature, PPG_out = PPG_model(PPG)
            
            PPG_predicted = PPG_out.argmax(1)

            if PPG_preds == None:
                PPG_preds = PPG_predicted
                all_targets = target
            else:
                PPG_preds = torch.cat((PPG_preds, PPG_predicted))
                all_targets = torch.cat((all_targets, target))
        tend = datetime.now()
        print_flush(f'[TEST] \tPPG F1: {f1_score(all_targets.detach().cpu().numpy(), PPG_preds.detach().cpu().numpy())}')



test_dataset = Dataset_ori('/labs/hulab/stark_stuff/ppg_ecg_project/data/x_test_2400.npy', '/labs/hulab/stark_stuff/ppg_ecg_project/data/y_test.npy')
# test_dataset = Dataset_ori('/labs/hulab/stark_stuff/ppg_ecg_project/data/x_test.npy', '/labs/hulab/stark_stuff/ppg_ecg_project/data/y_test.npy')
testloader = DataLoader(test_dataset, batch_size=2500, shuffle=False, num_workers=2)


PPG_model = Resnet34().cuda()


state_dict = torch.load('/home/zguo30/ppg_ecg_proj/saved_models/epoch_40_ecglr_0.0001_ppglr_0.0001_lambda_0.9/PPG_best_5.pt') 
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] #remove 'module'
    new_state_dict[name] = v
    
state_dict = new_state_dict

# state_dict = torch.load('/home/zguo30/ppg_ecg_proj/ori_ppg_only_baseline/saved_models/PPG_best.pt')

PPG_model.load_state_dict(state_dict)

test_epoch(PPG_model, testloader)