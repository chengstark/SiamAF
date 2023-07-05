import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain, combinations



#!/usr/bin/env python

# Copyright 2016-2020 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adopted from https://github.com/yandexdataschool/roc_comparison."""

import numpy as np
import scipy.stats


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/

def compute_midrank(x):
    """Computes midranks.

    Args:
       x - a 1D numpy array
    Returns:
       array of midranks

    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.

    Args:
       x - a 1D numpy array
    Returns:
       array of midranks

    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong test computation.

    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }

    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.

    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)

    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    log10pval = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
    return np.power(10, log10pval)


def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions):
    """Computes ROC AUC variance for a single set of predictions.

    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1

    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """Computes log(p-value) for hypothesis that two ROC AUCs are different.

    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1

    """
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def calc_delong(y_true, y_p, alpha=0.95):
    
#     alpha = .95

    y_true = y_true.reshape(-1)
    y_p = y_p.reshape(-1)

    auc, auc_cov = delong_roc_variance(
        y_true,
        y_p)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1

    print('AUC:', auc)
    print('AUC COV:', auc_cov)
    print('95% AUC CI:', ci)
    
    return auc, auc_cov, ci


testing_data_label_pairs = [
    ['data_simband_ecg_2400.npy', 'label_simband.npy', '/labs/hulab/Robust_learning_TESTDATA/simband_patient_idx_dict.pkl'],
    ['data_simband_ppg_2400.npy', 'label_simband.npy', '/labs/hulab/Robust_learning_TESTDATA/simband_patient_idx_dict.pkl'],
    ['data_ucla_ecg_2400.npy', 'label_ucla_ecg.npy', '/labs/hulab/Robust_learning_TESTDATA/UCLA_patient_idx_dict.pkl'],
    ['data_ucla_ppg_2400.npy', 'label_ucla_ppg.npy', '/labs/hulab/Robust_learning_TESTDATA/UCLA_patient_idx_dict.pkl'],
    ['data_staford_2400.npy', 'label_staford.npy', '/labs/hulab/Robust_learning_TESTDATA/stanford_patient_idx_dict.pkl'],
]

# # PPG FULL DATASET
# print('PPG FULL DATASET', flush=True)
# folders = ['simsiam', 'deepmi', 'deepbeat_baseline', 'ppg_only_baseline']
# model_names = ['simsiam', 'deepmi', 'deepbeat', 'ppgonly']
# dataset_lengths = ['2400', '2400', '800', '2400']
# combs = list(combinations(folders, 2))
# print(combs, flush=True)

# for pair in testing_data_label_pairs:
        
#     dataset = pair[0].split('.')[0]
#     print('---------------------------------------------------------------', flush=True)
#     if 'ppg' in dataset or 'staford' in dataset:
#         print(dataset, flush=True)
#         for model_pair in combs:

#             dataset_1 = dataset.replace('2400', dataset_lengths[folders.index(model_pair[0])])
#             dataset_2 = dataset.replace('2400', dataset_lengths[folders.index(model_pair[1])])
            
#             targets = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_targets.npy')
#             pred_probs_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_pred_probs.npy')
#             pred_probs_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_pred_probs.npy')
#             auroc_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_aurocs.npy')       
#             auroc_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_aurocs.npy')
#             auprc_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_auprcs.npy')            
#             auprc_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_auprcs.npy')
# #             print(targets.shape, pred_probs_1.shape, pred_probs_2.shape, auroc_1.shape, auroc_2.shape, auprc_1.shape, auprc_2.shape)
#             p_auroc = delong_roc_test(targets, pred_probs_1, pred_probs_2)[0][0]
# #             p_auroc = stats.ttest_rel(auroc_1, auroc_2).pvalue
#             p_auprc = stats.ttest_rel(auprc_1, auprc_2).pvalue
            
#             print(f'{model_pair}: p_auroc={p_auroc} , p_auprc={p_auprc}', flush=True)


# # ECG DATASETS FULL
# print('ECG DATASETS FULL', flush=True)
# folders = ['simsiam', 'deepmi', 'ecg_only_baseline']
# model_names = ['simsiam', 'deepmi', 'ecgonly']
# dataset_lengths = ['2400', '2400', '2400']
# combs = list(combinations(folders, 2))
# print(combs, flush=True)


# for pair in testing_data_label_pairs:
        
#     dataset = pair[0].split('.')[0]
#     print('---------------------------------------------------------------', flush=True)
#     if 'ecg' in dataset:
#         print(dataset, flush=True)
#         for model_pair in combs:
#             print(model_pair, flush=True)
            
#             dataset_1 = dataset.replace('2400', dataset_lengths[folders.index(model_pair[0])])
#             dataset_2 = dataset.replace('2400', dataset_lengths[folders.index(model_pair[1])])
            
#             targets = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_targets.npy')
#             pred_probs_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_pred_probs.npy')
#             pred_probs_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_pred_probs.npy')
#             auroc_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_aurocs.npy')       
#             auroc_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_aurocs.npy')
#             auprc_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_auprcs.npy')            
#             auprc_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_auprcs.npy')
# #             print(targets.shape, pred_probs_1.shape, pred_probs_2.shape, auroc_1.shape, auroc_2.shape, auprc_1.shape, auprc_2.shape)
#             p_auroc = delong_roc_test(targets, pred_probs_1, pred_probs_2)[0][0]
# #             p_auroc = stats.ttest_rel(auroc_1, auroc_2).pvalue
#             p_auprc = stats.ttest_rel(auprc_1, auprc_2).pvalue
            
#             print(f'{model_pair}: p_auroc={p_auroc} , p_auprc={p_auprc}', flush=True)


# # PPG LIMITED
# print('PPG LIMITED', flush=True)
# folders = ['simsiam_limited_labeled', 'deepmi_limited_labels', 'ppg_only_baseline_limited_labeled']
# model_names = ['simsiamlimitedlabels', 'deepmilimitedlabels', 'ppgonlylimitedlabel']
# dataset_lengths = ['2400', '2400', '2400']
# combs = list(combinations(folders, 2))
# print(combs, flush=True)

# for pair in testing_data_label_pairs:

#     dataset = pair[0].split('.')[0]
#     print('---------------------------------------------------------------', flush=True)
#     if 'ppg' in dataset or 'staford' in dataset:
#         print(dataset, flush=True)
#         for model_pair in combs:

#             dataset_1 = dataset.replace('2400', dataset_lengths[folders.index(model_pair[0])])
#             dataset_2 = dataset.replace('2400', dataset_lengths[folders.index(model_pair[1])])
            
#             targets = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_targets.npy')
#             pred_probs_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_pred_probs.npy')
#             pred_probs_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_pred_probs.npy')
#             auroc_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_aurocs.npy')       
#             auroc_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_aurocs.npy')
#             auprc_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_auprcs.npy')            
#             auprc_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_auprcs.npy')
#             # print(targets.shape, pred_probs_1.shape, pred_probs_2.shape, auroc_1.shape, auroc_2.shape, auprc_1.shape, auprc_2.shape)
#             p_auroc = delong_roc_test(targets, pred_probs_1, pred_probs_2)[0][0]
# #             p_auroc = stats.ttest_rel(auroc_1, auroc_2).pvalue
#             p_auprc = stats.ttest_rel(auprc_1, auprc_2).pvalue
            
#             print(f'{model_pair}: p_auroc={p_auroc} , p_auprc={p_auprc}', flush=True)


# ECG LIMITED
print('ECG LIMITED', flush=True)
folders = ['simsiam_limited_labeled', 'deepmi_limited_labels', 'ecg_only_baseline_limited_labeled']
model_names = ['simsiamlimitedlabels', 'deepmilimitedlabels', 'ecgonlylimitedlabel']
dataset_lengths = ['2400', '2400', '2400']
combs = list(combinations(folders, 2))
print(combs, flush=True)

for pair in testing_data_label_pairs:

    dataset = pair[0].split('.')[0]
    print('---------------------------------------------------------------', flush=True)
    if 'ecg' in dataset:
        print(dataset, flush=True)
        for model_pair in combs:

            dataset_1 = dataset.replace('2400', dataset_lengths[folders.index(model_pair[0])])
            dataset_2 = dataset.replace('2400', dataset_lengths[folders.index(model_pair[1])])
            
            targets = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_targets.npy')
            pred_probs_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_pred_probs.npy')
            pred_probs_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_pred_probs.npy')
            auroc_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_aurocs.npy')       
            auroc_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_aurocs.npy')
            auprc_1 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_1}_{model_names[folders.index(model_pair[0])]}_auprcs.npy')            
            auprc_2 = np.load(f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{dataset_2}_{model_names[folders.index(model_pair[1])]}_auprcs.npy')
            # print(targets.shape, pred_probs_1.shape, pred_probs_2.shape, auroc_1.shape, auroc_2.shape, auprc_1.shape, auprc_2.shape)
            p_auroc = delong_roc_test(targets, pred_probs_1, pred_probs_2)[0][0]
#             p_auroc = stats.ttest_rel(auroc_1, auroc_2).pvalue
            p_auprc = stats.ttest_rel(auprc_1, auprc_2).pvalue
            
            print(f'{model_pair}: p_auroc={p_auroc} , p_auprc={p_auprc}', flush=True)

