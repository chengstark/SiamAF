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
import os
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
from functools import partial
import torch.nn.functional as F
print_flush = partial(print, flush=True)


device = 'cuda'
MODEL_PATH = '/home/zguo30/ppg_ecg_proj/ppg_only_baseline/saved_models/epoch_40_ppglr_0.0001_lambda_0.9/PPG_best_2.pt'
PPG_model = Resnet34().cuda()
state_dict = torch.load(MODEL_PATH) 

from collections import OrderedDict

# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] #remove 'module'
#     new_state_dict[name] = v
    
# state_dict = new_state_dict

PPG_model.load_state_dict(state_dict)



def test_epoch(PPG_model, test_loader):
    with torch.no_grad():
       

        PPG_preds = None
        PPG_pred_probs = None
        all_targets = None
        
        PPG_model.eval()
        tstart = datetime.now()

        for batch_idx, (PPG, target) in enumerate(test_loader):

            PPG = PPG.to(device).float()
            target = target.to(device).long()
            
            PPG_feature, PPG_out = PPG_model(PPG)

            PPG_predicted = PPG_out.argmax(1)
            PPG_predicted_prob = F.softmax(PPG_out, dim=1)[:, 1]

            if PPG_preds == None:
                PPG_pred_probs = PPG_predicted_prob
                PPG_preds = PPG_predicted
                all_targets = target
            else:
                PPG_preds = torch.cat((PPG_preds, PPG_predicted))
                PPG_pred_probs = torch.cat((PPG_pred_probs, PPG_predicted_prob))
                all_targets = torch.cat((all_targets, target))
        tend = datetime.now()


        precision, recall, thresholds = precision_recall_curve(all_targets.detach().cpu().numpy(), PPG_pred_probs.detach().cpu().numpy())
        pr_auc = auc(recall, precision)

        print_flush(f'[TEST] \tPPG      F1: {f1_score(all_targets.detach().cpu().numpy(), PPG_preds.detach().cpu().numpy())}')
        print_flush(f'[TEST] \tPPG ROC AUC: {roc_auc_score(all_targets.detach().cpu().numpy(), PPG_pred_probs.detach().cpu().numpy())}')
        print_flush(f'[TEST] \tPPG PR  AUC: {pr_auc}')


test_data_folder = '/labs/hulab/Robust_learning_TESTDATA/'
for f in sorted(os.listdir(test_data_folder)):
    if '2400' in f:
        
        x_path = test_data_folder + f
        y_path = test_data_folder + 'label_'+'_'.join(f[:-4].split('_')[1:-1]) + '.npy'

        print(f[:-4])
        print(x_path)
        print(y_path)
        test_dataset = Dataset_ori(x_path, y_path)
        testloader = DataLoader(test_dataset, batch_size=2500, shuffle=False, num_workers=0)
        
        test_epoch(PPG_model, testloader)

        print()