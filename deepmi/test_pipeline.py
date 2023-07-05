# %%
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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
import torch.nn.functional as F
from tqdm import tqdm

import os
device = 'cuda'

import pickle as pkl

# %%
# test_dataset = Dataset_ori('/labs/hulab/stark_stuff/ppg_ecg_project/data/x_test_2400.npy', '/labs/hulab/stark_stuff/ppg_ecg_project/data/y_test.npy')
# testloader = DataLoader(test_dataset, batch_size=2500, shuffle=False, num_workers=2)

# %%
def test_epoch(PPG_model, test_loader, test_ds_name, patient_info_path=None):
    with torch.no_grad():
       

        PPG_preds = None
        all_targets = None
        PPG_pred_probs = None

        PPG_model.eval()
        tstart = datetime.now()

        for batch_idx, (PPG, target) in enumerate(test_loader):

            PPG = PPG.to(device).float()
            target = target.to(device).long()
            
            PPG_feature, PPG_out = PPG_model(PPG)
#             print(PPG_feature.shape, PPG_out.shape)
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
                        
            np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_deepmi_aurocs.npy', all_bt_aurocs)
            np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_deepmi_auprcs.npy', all_bt_auprcs)
            np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_deepmi_targets.npy', all_bt_targets)
            np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_deepmi_pred_probs.npy', all_bt_pred_probs)

            print(f'\t AUROC {round(np.mean(all_bt_aurocs), rounding)} [{round(np.mean(all_bt_aurocs) - 1.96 *  (np.std(all_bt_aurocs) / np.sqrt(len(all_bt_aurocs))) , rounding)} {round(np.mean(all_bt_aurocs) + 1.96 *  (np.std(all_bt_aurocs) / np.sqrt(len(all_bt_aurocs))) , rounding)}]')
            print(f'\t AUPRC {round(np.mean(all_bt_auprcs), rounding)} [{round(np.mean(all_bt_auprcs) - 1.96 *  (np.std(all_bt_auprcs) / np.sqrt(len(all_bt_auprcs))) , rounding)} {round(np.mean(all_bt_auprcs) + 1.96 *  (np.std(all_bt_auprcs) / np.sqrt(len(all_bt_auprcs))) , rounding)}]')


# %%
PPG_model = Resnet34().cuda()

# %%
state_dict = torch.load('/home/zguo30/ppg_ecg_proj/deepmi/saved_models/corrected_epoch_40_ecglr_0.0001_ppglr_0.0001_lambda_0.9/PPG_best_5.pt') 
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] #remove 'module'
    new_state_dict[name] = v
    
state_dict = new_state_dict

PPG_model.load_state_dict(state_dict)

# %%
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
    
    print(pair[0].split('.')[0].split('/')[-1])
    
    x_path = test_data_folder + pair[0]
    y_path = test_data_folder + pair[1]
    
    if 'ppg' in pair[0] or 'staford' in pair[0]:

        print(x_path)
        print(y_path)
        test_dataset = Dataset_ori(x_path, y_path)
        testloader = DataLoader(test_dataset, batch_size=2500, shuffle=False, num_workers=0)

        test_epoch(PPG_model, testloader, pair[0].split('.')[0], patient_info_path=pair[2] if len(pair)==3 else None)

        print()

# %%
ECG_model = Resnet34().cuda()

# %%
state_dict = torch.load('/home/zguo30/ppg_ecg_proj/deepmi/saved_models/corrected_epoch_40_ecglr_0.0001_ppglr_0.0001_lambda_0.9/ECG_best_5.pt') 
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] #remove 'module'
    new_state_dict[name] = v
    
state_dict = new_state_dict

ECG_model.load_state_dict(state_dict)

# %%
# testing_data_label_pairs = [
#     ['data_simband_ecg_2400.npy', 'label_simband.npy'],
#     ['data_simband_ppg_2400.npy', 'label_simband.npy'],
#     ['data_ucla_ecg_2400.npy', 'label_ucla_ecg.npy'],
#     ['data_ucla_ppg_2400.npy', 'label_ucla_ppg.npy'],
#     ['data_staford_2400.npy', 'label_staford.npy'],
#     ['data_staford_goodquality_2400.npy', 'label_staford_goodquality.npy'],
#     ['data_staford_badquality_2400.npy', 'label_staford_badquality.npy']
# ]

test_data_folder = '/labs/hulab/Robust_learning_TESTDATA/'
for pair in testing_data_label_pairs:
    
    print(pair[0].split('.')[0].split('/')[-1])
    
    x_path = test_data_folder + pair[0]
    y_path = test_data_folder + pair[1]
    
    if 'ecg' in pair[0]:

        print(x_path)
        print(y_path)
        test_dataset = Dataset_ori(x_path, y_path)
        testloader = DataLoader(test_dataset, batch_size=2500, shuffle=False, num_workers=0)

        test_epoch(ECG_model, testloader, pair[0].split('.')[0], patient_info_path=pair[2] if len(pair)==3 else None)

        print()

# %%



