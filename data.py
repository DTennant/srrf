import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from fastai.vision.all import *

class ErrAug(Callback):
    def __init__(self, max_noise=0.1, p=0.5):
        self.max_noise = max_noise
        self.p = p
        
    def before_batch(self):
        if np.random.rand() < self.p:
            # new_react = react + randn * react_err
            self.learn.yb[0]['react'] += torch.randn_like(self.learn.yb[0]['react']) * \
                                            self.learn.yb[0]['react_err'] * self.max_noise

class RNA_Dataset_Test(Dataset):
    def __init__(self, df, mask_only=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        df['L'] = df.sequence.apply(len)
        self.Lmax = df['L'].max()
        self.df = df
        self.mask_only = mask_only
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        id_min, id_max, seq = self.df.loc[idx, ['id_min','id_max','sequence']]
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        L = len(seq)
        mask[:L] = True
        if self.mask_only: return {'mask':mask},{}
        ids = np.arange(id_min,id_max+1)
        
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        seq = np.pad(seq,(0,self.Lmax-L))
        ids = np.pad(ids,(0,self.Lmax-L), constant_values=-1)
        
        return {'seq':torch.from_numpy(seq), 'mask':mask}, \
               {'ids':ids}


class RNA_Dataset(Dataset):
    def __init__(self, df, mode='train', seed=2023, fold=0, nfolds=4, 
                 mask_only=False, pretrain=False, add_noise=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.Lmax = 206
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']
        
        split = list(KFold(n_splits=nfolds, random_state=seed, 
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        
        if not pretrain:
            m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
            df_2A3 = df_2A3.loc[m].reset_index(drop=True)
            df_DMS = df_DMS.loc[m].reset_index(drop=True)
        else:
            print('Pretraining on all data')
            print(f'2A3: {len(df_2A3)}')
            print(f'DMS: {len(df_DMS)}')
        self.add_noise = add_noise
        print(f'add_noise: {self.add_noise}')
        
        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values
        
        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        self.mask_only = mask_only
        
    def __len__(self):
        return len(self.seq)  
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        if self.mask_only:
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[:len(seq)] = True
            return {'mask':mask},{'mask':mask}
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq,(0,self.Lmax-len(seq)))
        
        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]],-1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        sn = torch.FloatTensor([self.sn_2A3[idx],self.sn_DMS[idx]])

        # if self.add_noise:
        #     react += torch.randn_like(react) * react_err
        
        return {'seq':torch.from_numpy(seq), 'mask':mask}, \
               {'react':react, 'react_err':react_err,
                'sn':sn, 'mask':mask}
    

class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]
            if isinstance(s,tuple): L = s[0]["mask"].sum()
            else: L = s["mask"].sum()
            L = max(1,L // 16) 
            if len(buckets[L]) == 0:  buckets[L] = []
            buckets[L].append(idx)
            
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []
                
        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch
            
def dict_to(x, device='cuda'):
    return {k:x[k].to(device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)
            