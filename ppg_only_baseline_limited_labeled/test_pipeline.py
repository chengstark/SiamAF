# %%
import torch
import torch.nn as nn
import torch.nn.functional as F 
from resnet1d import Resnet34
# from resnet_zoo import Resnet34
from dataset import Dataset_per_file, Dataset_whole, Dataset_ori
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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
import sys
import os

from collections import OrderedDict
import pickle as pkl

# %%
def test_epoch(PPG_model, test_loader, test_ds_name, baseline=False, patient_info_path=None):
    with torch.no_grad():
       

        PPG_preds = None
        PPG_pred_probs = None
        all_targets = None
        
        PPG_model.eval()
        tstart = datetime.now()

        for batch_idx, (PPG, target) in enumerate(test_loader):

            PPG = PPG.cuda().float()
            target = target.cuda().long()
            
            _, PPG_out = PPG_model(PPG)

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

        print(f'[TEST] \tPPG      F1: {round(f1_score(all_targets.detach().cpu().numpy(), PPG_preds.detach().cpu().numpy()), 4)}')
        print(f'[TEST] \tPPG ROC AUC: {round(roc_auc_score(all_targets.detach().cpu().numpy(), PPG_pred_probs.detach().cpu().numpy()), 4)}')
        print(f'[TEST] \tPPG PR  AUC: {round(pr_auc, 4)}')
        rounding = 3
        if patient_info_path is not None:
            print(patient_info_path)
            patient_idx_dict = pkl.load(open(patient_info_path, 'rb'))
            unique_patients = list(patient_idx_dict.keys())
            all_bt_aurocs = []
            all_bt_auprcs = []
            all_bt_pred_probs = np.asarray([])
            all_bt_targets = np.asarray([])
            for i in tqdm(range(1000)):
                np.random.seed(i)
                random_sample_patients = np.random.choice(unique_patients, len(unique_patients), replace=True)
                sample_idx = []
                random_subsample_idx = []
                for patient in random_sample_patients:
                    random_subsample_idx += patient_idx_dict[patient]
                random_subsample_idx = np.asarray(random_subsample_idx)
                auroc = roc_auc_score(all_targets[random_subsample_idx].detach().cpu().numpy(), PPG_pred_probs[random_subsample_idx].detach().cpu().numpy())
                precision, recall, thresholds = precision_recall_curve(all_targets[random_subsample_idx].detach().cpu().numpy(), PPG_pred_probs[random_subsample_idx].detach().cpu().numpy())
                auprc = auc(recall, precision)

                all_bt_pred_probs = np.concatenate((PPG_pred_probs[random_subsample_idx].detach().cpu().numpy().flatten(), all_bt_pred_probs))
                all_bt_targets = np.concatenate((all_targets[random_subsample_idx].detach().cpu().numpy().flatten(), all_bt_targets))
                all_bt_aurocs.append(auroc)
                all_bt_auprcs.append(auprc)
            all_bt_aurocs = np.asarray(all_bt_aurocs)
            all_bt_auprcs = np.asarray(all_bt_auprcs)
                        
            np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_ppgonlylimitedlabel_aurocs.npy', all_bt_aurocs)
            np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_ppgonlylimitedlabel_auprcs.npy', all_bt_auprcs)
            np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_ppgonlylimitedlabel_targets.npy', all_bt_targets)
            np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_ppgonlylimitedlabel_pred_probs.npy', all_bt_pred_probs)

            print(f'\t AUROC {round(np.mean(all_bt_aurocs), rounding)} [{round(np.mean(all_bt_aurocs) - 1.96 *  (np.std(all_bt_aurocs) / np.sqrt(len(all_bt_aurocs))) , rounding)} {round(np.mean(all_bt_aurocs) + 1.96 *  (np.std(all_bt_aurocs) / np.sqrt(len(all_bt_aurocs))) , rounding)}]')
            print(f'\t AUPRC {round(np.mean(all_bt_auprcs), rounding)} [{round(np.mean(all_bt_auprcs) - 1.96 *  (np.std(all_bt_auprcs) / np.sqrt(len(all_bt_auprcs))) , rounding)} {round(np.mean(all_bt_auprcs) + 1.96 *  (np.std(all_bt_auprcs) / np.sqrt(len(all_bt_auprcs))) , rounding)}]')


# %% [markdown]
# # 0.1

# %%

# %% [markdown]
# # 0.3

# %%

# %% [markdown]
# # 0.5

# %%

# %% [markdown]
# # 0.01

# %%
MODEL_PATH = '/home/zguo30/ppg_ecg_proj/ppg_only_baseline_limited_labeled/saved_models/corrected_res34_epoch_30_ppglr_0.0001_lambda_0.9_LABEL_PERC_0.01_/PPG_best_12.pt'
state_dict = torch.load(MODEL_PATH) 

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] #remove 'module'
    new_state_dict[name] = v
    
state_dict = new_state_dict

model = Resnet34().cuda()
model.load_state_dict(state_dict)

testing_data_label_pairs = [
    ['data_simband_ecg_2400.npy', 'label_simband.npy', '/labs/hulab/Robust_learning_TESTDATA/simband_patient_idx_dict.pkl'],
    ['data_simband_ppg_2400.npy', 'label_simband.npy', '/labs/hulab/Robust_learning_TESTDATA/simband_patient_idx_dict.pkl'],
    ['data_ucla_ecg_2400.npy', 'label_ucla_ecg.npy', '/labs/hulab/Robust_learning_TESTDATA/UCLA_patient_idx_dict.pkl'],
    ['data_ucla_ppg_2400.npy', 'label_ucla_ppg.npy', '/labs/hulab/Robust_learning_TESTDATA/UCLA_patient_idx_dict.pkl'],
    ['data_staford_2400.npy', 'label_staford.npy', '/labs/hulab/Robust_learning_TESTDATA/stanford_patient_idx_dict.pkl'],
#     ['data_staford_goodquality_2400.npy', 'label_staford_goodquality.npy', '/labs/hulab/Robust_learning_TESTDATA/stanford_patient_idx_dict.pkl'],
#     ['data_staford_badquality_2400.npy', 'label_staford_badquality.npy', '/labs/hulab/Robust_learning_TESTDATA/stanford_patient_idx_dict.pkl']
]

test_data_folder = '/labs/hulab/Robust_learning_TESTDATA/'
for pair in testing_data_label_pairs:
    if 'ppg' in pair[0] or 'staford' in pair[0]:

        print(pair[0].split('.')[0].split('/')[-1])

        x_path = test_data_folder + pair[0]
        y_path = test_data_folder + pair[1]

        print(x_path)
        print(y_path)
        test_dataset = Dataset_ori(x_path, y_path)
    #     x = np.load(x_path)
    #     y = np.load(y_path)
    #     print(np.max(x), np.min(x))

    #     print(x.shape, y.shape)

        testloader = DataLoader(test_dataset, batch_size=2500, shuffle=False, num_workers=0)

        test_epoch(model, testloader, pair[0].split('.')[0],  patient_info_path=pair[2] if len(pair)==3 else None)

        print()

# %%



