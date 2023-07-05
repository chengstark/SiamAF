import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import multiprocessing
import numpy as np
import shutil
import pickle as pkl
import gc

from scipy.signal import butter, lfilter, resample

import torch

import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    if self._state != mpp.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


data_path = '/labs/hulab/stark_stuff/ppg_ecg_project/data/'

def load_sigs(path, placeholder):
    sig = np.load(path)
    ecg = sig[:, 0]
    ppg = sig[:, 1]
    return ecg, ppg

def load_sigs_resample(path, placeholder):
    sig = np.load(path)
    ecg = sig[:, 0]
    ppg = sig[:, 1]
    ecg = resample(ecg, 2400)
    ppg = resample(ppg, 2400)
    ecg = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg))
    ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))
    
    return ecg, ppg


def save_sig_npys(data_path, folder, split):
    paths = pkl.load(open(f'{data_path}/{folder}/split_{split}.pkl', 'rb'))
    ecgs = []
    ppgs = []
    pool_args = [[p, None] for p in paths]

    with Pool(80) as pool:
        for ret in tqdm(pool.istarmap(load_sigs, pool_args), total=len(pool_args)):
            ecgs.append(ret[0])
            ppgs.append(ret[1])
    
    ecgs = np.asarray(ecgs)
    ppgs = np.asarray(ppgs)
    print(ecgs.shape, ppgs.shape)
    
    print(f'{data_path}/{folder}/{split}_ECG.npy')
    print(f'{data_path}/{folder}/{split}_PPG.npy')
    
    np.save(f'{data_path}/{folder}/{split}_ECG.npy', ecgs)
    np.save(f'{data_path}/{folder}/{split}_PPG.npy', ppgs)
    
    del ecgs, ppgs, paths
    gc.collect()
    

def save_resample_sig_npys(data_path, folder, split):
    paths = pkl.load(open(f'{data_path}/{folder}/split_{split}.pkl', 'rb'))
    ecgs = []
    ppgs = []
    pool_args = [[p, None] for p in paths]

    with Pool(80) as pool:
        for ret in tqdm(pool.istarmap(load_sigs_resample, pool_args), total=len(pool_args)):
            ecgs.append(ret[0])
            ppgs.append(ret[1])
    
    ecgs = np.asarray(ecgs)
    ppgs = np.asarray(ppgs)
    print(ecgs.shape, ppgs.shape)
    
    np.save(f'{data_path}/{folder}/{split}_ECG_resampled2400.npy', ecgs)
    np.save(f'{data_path}/{folder}/{split}_PPG_resampled2400.npy', ppgs)
    
    del ecgs, ppgs, paths
    gc.collect()
    
    
save_resample_sig_npys(data_path, 'NSR_v5', 'train')
save_resample_sig_npys(data_path, 'NSR_v5', 'val')