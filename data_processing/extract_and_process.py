import os
import bin2data
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import multiprocessing
import numpy as np

from scipy.signal import butter, lfilter, resample

import multiprocessing.pool as mpp

n_cpus = multiprocessing.cpu_count()
print(n_cpus)


np.seterr(all='raise')

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

print(multiprocessing.cpu_count(), flush=True)

NSR_path = '/labs/hulab/chengding_project_data/AF_alarm/NSR_alarm_signals/'
PVC_path = '/labs/hulab/chengding_project_data/AF_alarm/PVC_alarm_signals/'
AF_path = '/labs/hulab/chengding_project_data/AF_alarm/AF_alarm_signals/'

NSR_path2 = '/labs/hulab/chengding_project_data/AF_alarm/NSR_alarm_signals2/'
AF_path2 = '/labs/hulab/chengding_project_data/AF_alarm/AF_alarm_signals2/'

NSR_files = os.listdir(NSR_path)
PVC_files = os.listdir(PVC_path)
AF_files = os.listdir(AF_path)

NSR_files2 = os.listdir(NSR_path2)
AF_files2 = os.listdir(AF_path2)

print(len(AF_files))
print(len(AF_files2))
print(len(PVC_files))
print(len(NSR_files))
print(len(NSR_files2))

print(len(AF_files)+len(AF_files2))
print(len(PVC_files))
print(len(NSR_files)+len(NSR_files2))


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_and_proecss(file_path, file, signal_type, skip_exist):
    file_id = file.split('.')[0]

    try:
        save_dir = f'/labs/hulab/stark_stuff/ppg_ecg_project/data/{signal_type}_v6'
        # skip finished ones
        if skip_exist:
            if os.path.exists(f'{save_dir}/{file_id}.npy'):
                return
        
        ECG = np.asarray(bin2data.get_data_sample(file_path+file, 'I'))
        PPG = np.asarray(bin2data.get_data_sample(file_path+file, 'SPO2'))

        assert ECG.shape == (7200, )
        assert PPG.shape == (7200, )

        ECG = butter_bandpass_filter(ECG, 0.67, 40, 240, order=1)

        assert np.isnan(PPG).any() == False and np.isinf(PPG).any() == False and np.max(PPG) > np.min(PPG)
        assert np.isnan(ECG).any() == False and np.isinf(PPG).any() == False and np.max(ECG) > np.min(ECG)

        PPG = resample(PPG, 2400)
        ECG = resample(ECG, 2400)

        PPG = (PPG - np.min(PPG)) / (np.max(PPG) - np.min(PPG))
        ECG = (ECG - np.min(ECG)) / (np.max(ECG) - np.min(ECG))
        
        cat = np.concatenate((ECG.reshape(ECG.shape[0], 1), PPG.reshape(PPG.shape[0], 1)), axis=1)
        np.save(f"{save_dir}/{file_id}.npy", cat)
    
    except Exception as e:
        pass  

os.makedirs('/labs/hulab/stark_stuff/ppg_ecg_project/data/AF_v6', exist_ok=True)
os.makedirs('/labs/hulab/stark_stuff/ppg_ecg_project/data/NSR_v6', exist_ok=True)
os.makedirs('/labs/hulab/stark_stuff/ppg_ecg_project/data/PVC_v6', exist_ok=True)

pool_args = [[NSR_path, x, 'NSR', True] for x in NSR_files]
with Pool(n_cpus) as pool:
    for _ in tqdm(pool.istarmap(extract_and_proecss, pool_args), total=len(pool_args)):
        pass

pool_args = [[NSR_path2, x, 'NSR', True] for x in NSR_files2]
with Pool(n_cpus) as pool:
    for _ in tqdm(pool.istarmap(extract_and_proecss, pool_args), total=len(pool_args)):
        pass

pool_args = [[PVC_path, x, 'PVC', True] for x in PVC_files]
with Pool(n_cpus) as pool:
    for _ in tqdm(pool.istarmap(extract_and_proecss, pool_args), total=len(pool_args)):
        pass

pool_args = [[AF_path, x, 'AF', True] for x in AF_files]
with Pool(n_cpus) as pool:
    for _ in tqdm(pool.istarmap(extract_and_proecss, pool_args), total=len(pool_args)):
        pass

pool_args = [[AF_path2, x, 'AF', True] for x in AF_files2]
with Pool(n_cpus) as pool:
    for _ in tqdm(pool.istarmap(extract_and_proecss, pool_args), total=len(pool_args)):
        pass

