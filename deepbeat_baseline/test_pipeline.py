# %%
# import warnings
# warnings.filterwarnings('ignore',category=FutureWarning)
# import warnings
# warnings.filterwarnings("ignore")

import sys
#import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py as h5
import sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_recall_curve, accuracy_score, confusion_matrix  
from sklearn.metrics import average_precision_score
import pickle
import keras
from keras.models import load_model
import tensorflow as tf
# tf.enable_eager_execution()
print(tf.__version__)
# import util

print("Python version:\n{}\n".format(sys.version))
print("matplotlib version: {}".format(matplotlib.__version__))
print("pandas version: {}".format(pd.__version__))
print("numpy version: {}".format(np.__version__))
print("sklearn version: {}".format(sklearn.__version__))
print("keras version: {}".format(keras.__version__))
import pickle as pkl

# %%
model = load_model('deepbeat.h5')

# %%
from scipy.signal import resample
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
from scipy.special import softmax

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

# test_data_folder = '/labs/hulab/Robust_learning_TESTDATA/'
# for pair in testing_data_label_pairs:
    
#     print(pair[0].split('.')[0].split('/')[-1])
    
#     x_path = test_data_folder + pair[0]
#     y_path = test_data_folder + pair[1]
#     x_id = pair[0].split('.')[0][:-5]

#     x = np.load(x_path)
# #     y = np.load(y)
#     x_resampled = []

#     for x in tqdm(x):
#         resampled = resample(x, 800)
#         resampled = (resampled - np.min(resampled) / np.max(resampled) - np.min(resampled))
#         x_resampled.append(resampled)
        
#     x_resampled = np.asarray(x_resampled)
#     x_resampled = x_resampled.reshape(x_resampled.shape[0], x_resampled.shape[1], 1)

#     print(x_resampled.shape)
#     np.save(test_data_folder+x_id+'_800.npy', x_resampled)
        
        

#     print(x_path)
#     print(y_path)

# %%
testing_data_label_pairs = [
    ['data_simband_ppg_800.npy', 'label_simband.npy', '/labs/hulab/Robust_learning_TESTDATA/simband_patient_idx_dict.pkl'],
    ['data_ucla_ppg_800.npy', 'label_ucla_ppg.npy', '/labs/hulab/Robust_learning_TESTDATA/UCLA_patient_idx_dict.pkl'],
    ['data_staford_800.npy', 'label_staford.npy', '/labs/hulab/Robust_learning_TESTDATA/stanford_patient_idx_dict.pkl'],
#     ['data_staford_goodquality_800.npy', 'label_staford_goodquality.npy', '/labs/hulab/Robust_learning_TESTDATA/stanford_patient_idx_dict.pkl'],
#     ['data_staford_badquality_800.npy', 'label_staford_badquality.npy', '/labs/hulab/Robust_learning_TESTDATA/stanford_patient_idx_dict.pkl']
]
test_data_folder = '/labs/hulab/Robust_learning_TESTDATA/'
for pair in testing_data_label_pairs:
    
    print(pair[0].split('.')[0].split('/')[-1])
    
    x_path = test_data_folder + pair[0]
    y_path = test_data_folder + pair[1]
    patient_info_path = pair[2]
    test_ds_name = pair[0].split('.')[0]
    print(x_path)
    print(y_path)

    x = np.load(x_path)
    y = np.load(y_path)
    
    PPG_out = model.predict(x)[1]
    PPG_preds = PPG_out.argmax(1)
    PPG_pred_probs = softmax(PPG_out, axis=1)[:, 1]
        
    all_targets = y
    
    precision, recall, thresholds = precision_recall_curve(all_targets, PPG_pred_probs)
    pr_auc = auc(recall, precision)

    print(f'[TEST] \tPPG      F1: {round(f1_score(all_targets, PPG_preds), 4)}')
    print(f'[TEST] \tPPG ROC AUC: {round(roc_auc_score(all_targets, PPG_pred_probs), 4)}')
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

            auroc = roc_auc_score(all_targets[random_subsample_idx], PPG_pred_probs[random_subsample_idx])
            precision, recall, thresholds = precision_recall_curve(all_targets[random_subsample_idx], PPG_pred_probs[random_subsample_idx])
            auprc = auc(recall, precision)

            all_bt_pred_probs = np.concatenate((PPG_pred_probs[random_subsample_idx], all_bt_pred_probs))
            all_bt_targets = np.concatenate((all_targets[random_subsample_idx], all_bt_targets))
            all_bt_aurocs.append(auroc)
            all_bt_auprcs.append(auprc)

        all_bt_aurocs = np.asarray(all_bt_aurocs)
        all_bt_auprcs = np.asarray(all_bt_auprcs)
        
        print(all_bt_targets.shape, all_bt_pred_probs.shape)
        
        np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_deepbeat_aurocs.npy', all_bt_aurocs)
        np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_deepbeat_auprcs.npy', all_bt_auprcs)
        np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_deepbeat_targets.npy', all_bt_targets)
        np.save(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{test_ds_name}_deepbeat_pred_probs.npy', all_bt_pred_probs)

        print(f'\t AUROC {round(np.mean(all_bt_aurocs), rounding)} [{round(np.mean(all_bt_aurocs) - 1.96 *  (np.std(all_bt_aurocs) / np.sqrt(len(all_bt_aurocs))) , rounding)} {round(np.mean(all_bt_aurocs) + 1.96 *  (np.std(all_bt_aurocs) / np.sqrt(len(all_bt_aurocs))) , rounding)}]')
        print(f'\t AUPRC {round(np.mean(all_bt_auprcs), rounding)} [{round(np.mean(all_bt_auprcs) - 1.96 *  (np.std(all_bt_auprcs) / np.sqrt(len(all_bt_auprcs))) , rounding)} {round(np.mean(all_bt_auprcs) + 1.96 *  (np.std(all_bt_auprcs) / np.sqrt(len(all_bt_auprcs))) , rounding)}]')

    print()
    
#     break

# %%
# mv /labs/hulab/Robust_learning_TESTDATA/data_ucl_800.npy /labs/hulab/Robust_learning_TESTDATA/trash/data_ucl_800.npy

# %%



