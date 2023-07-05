import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import torch
import random
import gc
from datetime import datetime
import psutil


'''PREFLIGHT SETUP'''
from functools import partial
print_flush = partial(print, flush=True)
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
'''PREFLIGHT SETUP'''


class Dataset_per_file(Dataset):
    def __init__(self, data_path, split, subset=0):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.dataset_file_paths,self.labels= self.build_dataset(subset=subset)
        self.length = len(self.dataset_file_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_path = self.dataset_file_paths[idx]
        x = np.load(file_path)
        # print(np.isnan(x).any(), np.isnan(x[:, 0]).any(), np.isnan(x[:, 1]).any(), '---------------------------------------')
        ECG, PPG = torch.from_numpy(x[:, 0][None]).float(), torch.from_numpy(x[:, 1][None]).float()
        target = self.labels[idx]
        return ECG, PPG, target

    def build_dataset(self, subset=0):
        print_flush(f'\tpkl loading paths splits {self.split}')
        AF_paths = pkl.load(open(f'{self.data_path}/AF_v5/split_{self.split}.pkl', 'rb'))
        NSR_paths = pkl.load(open(f'{self.data_path}/NSR_v5/split_{self.split}.pkl', 'rb'))
        PVC_paths = pkl.load(open(f'{self.data_path}/PVC_v5/split_{self.split}.pkl', 'rb'))
        print_flush('\tpkl loading paths splits finished')

        if subset > 0:
            AF_paths = random.sample(AF_paths, k=subset)
            NSR_paths = random.sample(NSR_paths, k=subset)
            PVC_paths = random.sample(PVC_paths, k=subset)
        # print(len(AF_paths), len(NSR_paths), len(PVC_paths))
        
        AF_paths += PVC_paths
        AF_labels = np.ones((len(AF_paths)))
        NSR_labels = np.zeros((len(NSR_paths)))

        print(f'dataset built AF counts {len(AF_labels)}, NSR counts {len(NSR_labels)}', flush=True)

        all_paths = AF_paths + NSR_paths
        all_labels = torch.from_numpy(np.concatenate((AF_labels, NSR_labels))).long()
        return all_paths, all_labels



class Dataset_whole(Dataset):
    def __init__(self, data_path, split, subset):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.build_dataset()
        self.length = len(self.all_labels)
        self.subset = subset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if idx < self.AF_ECG.shape[0]:
            return self.AF_ECG[idx][None, :], self.AF_PPG[idx][None, :], self.all_labels[idx]
        elif idx >= self.AF_ECG.shape[0] and idx < (self.AF_ECG.shape[0] + self.PVC_ECG.shape[0]):
            offset = self.AF_ECG.shape[0]
            return self.PVC_ECG[idx-offset][None, :], self.PVC_PPG[idx-offset][None, :], self.all_labels[idx]
        else:
            offset = self.AF_ECG.shape[0] + self.PVC_ECG.shape[0]
            return self.NSR_ECG[idx-offset][None, :], self.NSR_PPG[idx-offset][None, :], self.all_labels[idx]

    def build_dataset(self):
        tstart = datetime.now()
        print_flush(f'\tloading... {self.split}')
        
        self.AF_ECG = torch.from_numpy(np.load(f'{self.data_path}/AF_v5/{self.split}_ECG_resampled2400.npy')).float()
        self.AF_PPG = torch.from_numpy(np.load(f'{self.data_path}/AF_v5/{self.split}_PPG_resampled2400.npy')).float()
        print_flush(f'\tAF loaded {self.split}')
        self.PVC_ECG = torch.from_numpy(np.load(f'{self.data_path}/PVC_v5/{self.split}_ECG_resampled2400.npy')).float()
        self.PVC_PPG = torch.from_numpy(np.load(f'{self.data_path}/PVC_v5/{self.split}_PPG_resampled2400.npy')).float()
        print_flush(f'\tPVC loaded {self.split}')
        self.NSR_ECG = torch.from_numpy(np.load(f'{self.data_path}/NSR_v5/{self.split}_ECG_resampled2400.npy')).float()
        self.NSR_PPG = torch.from_numpy(np.load(f'{self.data_path}/NSR_v5/{self.split}_PPG_resampled2400.npy')).float()
        print_flush(f'\tNSR loaded {self.split}')

        assert self.AF_ECG.shape == self.AF_PPG.shape
        assert self.NSR_ECG.shape == self.NSR_PPG.shape
        assert self.PVC_ECG.shape == self.PVC_PPG.shape

        print_flush(f'\tloading {self.split} finished t={datetime.now() - tstart}, mem used={psutil.virtual_memory()[3]/1000000000}')
        
        AF_labels = np.ones((self.AF_ECG.shape[0]))
        NSR_labels = np.zeros((self.PVC_ECG.shape[0] + self.NSR_ECG.shape[0]))
        self.all_labels = torch.from_numpy(np.concatenate((AF_labels, NSR_labels), axis=0)).long()

        print(f'dataset built AF counts {len(AF_labels)}, NSR counts {len(NSR_labels)}, total counts {len(self.all_labels)}', flush=True)



class Dataset_PPG(Dataset):
    def __init__(self, data_path, split, subset):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.build_dataset()
        self.length = len(self.all_labels)
        self.subset = subset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if idx < self.AF_PPG.shape[0]:
            return self.AF_PPG[idx][None, :], self.all_labels[idx]
        elif idx >= self.AF_PPG.shape[0] and idx < (self.AF_PPG.shape[0] + self.PVC_PPG.shape[0]):
            offset = self.AF_PPG.shape[0]
            return self.PVC_PPG[idx-offset][None, :], self.all_labels[idx]
        else:
            offset = self.AF_PPG.shape[0] + self.PVC_PPG.shape[0]
            return self.NSR_PPG[idx-offset][None, :], self.all_labels[idx]

    def build_dataset(self):
        tstart = datetime.now()
        print_flush(f'\tloading... {self.split}')
        
        self.AF_PPG = torch.from_numpy(np.load(f'{self.data_path}/AF_v5/{self.split}_PPG_resampled2400.npy')).float()
        print_flush(f'\tAF loaded {self.split}')
        self.PVC_PPG = torch.from_numpy(np.load(f'{self.data_path}/PVC_v5/{self.split}_PPG_resampled2400.npy')).float()
        print_flush(f'\tPVC loaded {self.split}')
        self.NSR_PPG = torch.from_numpy(np.load(f'{self.data_path}/NSR_v5/{self.split}_PPG_resampled2400.npy')).float()
        print_flush(f'\tNSR loaded {self.split}')

        print_flush(f'\tloading {self.split} finished t={datetime.now() - tstart}, mem used={psutil.virtual_memory()[3]/1000000000}')
        
        AF_labels = np.ones((self.AF_PPG.shape[0]))
        NSR_labels = np.zeros((self.PVC_PPG.shape[0] + self.NSR_PPG.shape[0]))
        self.all_labels = torch.from_numpy(np.concatenate((AF_labels, NSR_labels), axis=0)).long()

        print(f'dataset built AF counts {len(AF_labels)}, NSR counts {len(NSR_labels)}, total counts {len(self.all_labels)}', flush=True)



class Dataset_ECG(Dataset):
    def __init__(self, data_path, split, subset):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.build_dataset()
        self.length = len(self.all_labels)
        self.subset = subset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        if idx < self.AF_ECG.shape[0]:
            return self.AF_ECG[idx][None, :], self.all_labels[idx]
        elif idx >= self.AF_ECG.shape[0] and idx < (self.AF_ECG.shape[0] + self.PVC_ECG.shape[0]):
            offset = self.AF_ECG.shape[0]
            return self.PVC_ECG[idx-offset][None, :], self.all_labels[idx]
        else:
            offset = self.AF_ECG.shape[0] + self.PVC_ECG.shape[0]
            return self.NSR_ECG[idx-offset][None, :], self.all_labels[idx]

    def build_dataset(self):
        tstart = datetime.now()
        print_flush(f'\tloading... {self.split}')
        
        self.AF_ECG = torch.from_numpy(np.load(f'{self.data_path}/AF_v5/{self.split}_ECG_resampled2400.npy')).float()
        print_flush(f'\tAF loaded {self.split}')
        self.PVC_ECG = torch.from_numpy(np.load(f'{self.data_path}/PVC_v5/{self.split}_ECG_resampled2400.npy')).float()
        print_flush(f'\tPVC loaded {self.split}')
        self.NSR_ECG = torch.from_numpy(np.load(f'{self.data_path}/NSR_v5/{self.split}_ECG_resampled2400.npy')).float()
        print_flush(f'\tNSR loaded {self.split}')

        print_flush(f'\tloading {self.split} finished t={datetime.now() - tstart}, mem used={psutil.virtual_memory()[3]/1000000000}')
        
        AF_labels = np.ones((self.AF_ECG.shape[0]))
        NSR_labels = np.zeros((self.PVC_ECG.shape[0] + self.NSR_ECG.shape[0]))
        self.all_labels = torch.from_numpy(np.concatenate((AF_labels, NSR_labels), axis=0)).long()

        print(f'dataset built AF counts {len(AF_labels)}, NSR counts {len(NSR_labels)}, total counts {len(self.all_labels)}', flush=True)


class Dataset_ori():
    def __init__(self,data_path,label_path):
        # self.root = root
        self.data_path = data_path
        self.label_path = label_path
        self.dataset,self.labelset= self.build_dataset()
        self.length = self.dataset.shape[0]
        # self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[idx,:]
        step = torch.unsqueeze(step, 0)
        # target = self.label[idx]
        target = self.labelset[idx]
        # target = torch.unsqueeze(target, 0)# only one class
        return step, target

    def build_dataset(self):
        '''get dataset of signal'''

        dataset = np.load(self.data_path)
        labelset = np.load(self.label_path)

        # dataset,labelset = shuffle(dataset,labelset)
        dataset = torch.from_numpy(dataset)
        labelset = torch.from_numpy(labelset)

        return dataset,labelset
    

class Dataset_ori_npy():
    def __init__(self,x,y):
        # self.root = root
        self.x = x
        self.y = y
        self.dataset,self.labelset= self.build_dataset()
        self.length = self.dataset.shape[0]
        # self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[idx,:]
        step = torch.unsqueeze(step, 0)
        # target = self.label[idx]
        target = self.labelset[idx]
        # target = torch.unsqueeze(target, 0)# only one class
        return step, target

    def build_dataset(self):
        '''get dataset of signal'''

        dataset = self.x
        labelset = self.y

        # dataset,labelset = shuffle(dataset,labelset)
        dataset = torch.from_numpy(dataset)
        labelset = torch.from_numpy(labelset)

        return dataset,labelset




class Dataset_valsubset():
    def __init__(self, ECG_path, PPG_path, label_path):
        # self.root = root
        self.ECG_path = ECG_path
        self.PPG_path = PPG_path

        self.label_path = label_path
        self.ECGs,self.PPGs, self.labels= self.build_dataset()
        self.length = self.ECGs.shape[0]
        # self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        return self.ECGs[idx], self.PPGs[idx], self.labels[idx]

    def build_dataset(self):
        '''get dataset of signal'''

        ECGs = np.load(self.ECG_path)
        PPGs = np.load(self.PPG_path)
        labels = np.load(self.label_path)
        # dataset,labelset = shuffle(dataset,labelset)
        ECGs = torch.from_numpy(ECGs).float()
        PPGs = torch.from_numpy(PPGs).float()

        labels = torch.from_numpy(labels).long()

        return ECGs, PPGs, labels
