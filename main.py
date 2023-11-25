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
from tqdm import tqdm

# Fix fastai bug to enable fp16 training with dictionaries

import torch
from fastai.vision.all import *
def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item

from torch.cuda.amp import GradScaler, autocast
@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10
    def __init__(self, **kwargs): self.kwargs = kwargs
    def before_fit(self): 
        self.autocast,self.learn.scaler,self.scales = autocast(),GradScaler(**self.kwargs),L()
    def before_batch(self): self.autocast.__enter__()
    def after_pred(self):
        if next(flatten(self.pred)).dtype==torch.float16: self.learn.pred = to_float(self.pred)
    def after_loss(self): self.autocast.__exit__(None, None, None)
    def before_backward(self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)
    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow. "
        self.skipped=True
        self.scaler.step(self)
        if self.skipped: raise CancelStepException()
        self.scales.append(self.scaler.get_scale())
    def after_step(self): self.learn.scaler.update()

    @property 
    def param_groups(self): 
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups
    def step(self, *args, **kwargs): 
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped=False
    def after_fit(self): self.autocast,self.learn.scaler,self.scales = None,None,None
        
import fastai
fastai.callback.fp16.MixedPrecision = MixedPrecision
import argparse
from data import RNA_Dataset, LenMatchBatchSampler, DeviceDataLoader, ErrAug, RNA_Dataset_Test
from model import RNA_Model, loss, combine_loss, RNA_Bert_Model
from utils import seed_everything, MAE


# TODO: 
# 1. pretrain on all data, and then finetune -- done
# 2. try augmentation with random applied error -- done
# 3. try augmentation with reverse sequence
# 4. try augmentation with random mask
# 5. pretrain with noise
# 6. try bert model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str, default='example0')
parser.add_argument('--path', type=str, default='stanford-ribonanza-rna-folding-converted/')
parser.add_argument('--out', type=str, default='output/baseline-larger/')
parser.add_argument('--bs', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nfolds', type=int, default=6)
parser.add_argument('--ep', type=int, default=32)

parser.add_argument('--use_fastai', type=str2bool, default='True')
parser.add_argument('--add_noise', action='store_true')
parser.add_argument('--max_noise', type=float, default=0.1)
parser.add_argument('--noise_p', type=float, default=0.3)

parser.add_argument('--use_combine_loss', action='store_true')

parser.add_argument('--debug', action='store_true')

parser.add_argument('--dim', type=int, default=192)
parser.add_argument('--depth', type=int, default=12)

parser.add_argument('--use_bert', action='store_true')
parser.add_argument('--num_bert_layers', type=int, default=2)

parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--pretrain_path', type=str, default='output/baseline-larger/')

parser.add_argument('--run_test', action='store_true')

args = parser.parse_args()
fname = args.fname
PATH = args.path
OUT = args.out
bs = args.bs
num_workers = args.num_workers
SEED = args.seed
nfolds = args.nfolds

seed_everything(SEED)
os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(os.path.join(PATH, 'train_data.parquet'))

if args.debug:
    __import__('ipdb').set_trace()
    
if args.run_test:
    df_test = pd.read_parquet(os.path.join(PATH, 'test_sequences.parquet'))
    ds = RNA_Dataset_Test(df_test)
    dl = DeviceDataLoader(torch.utils.data.DataLoader(ds, batch_size=bs, 
                shuffle=False, drop_last=False, num_workers=num_workers), device)
    del df_test
    gc.collect()
    from glob import glob
    # __import__("ipdb").set_trace()
    models = sorted(glob(os.path.join(OUT, '*.pth')))
    model = RNA_Model(dim=args.dim, depth=args.depth)
    model.cuda()
    for m in models:
        ids, preds = [], []
        model.load_state_dict(torch.load(m))
        model.eval()
        for x, y in tqdm(dl):
            with torch.no_grad(), torch.cuda.amp.autocast():
                pred = torch.nan_to_num(model(x)).clip(0, 1)
            for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), pred):
                ids.append(idx[mask])
                preds.append(pi[mask[:pi.shape[0]]].cpu())
        ids = torch.concat(ids)
        preds = torch.concat(preds)
        df = pd.DataFrame({'id':ids.numpy(), 'reactivity_DMS_MaP':preds[:,1].numpy(), 
                   'reactivity_2A3_MaP':preds[:,0].numpy()})
        df.to_parquet(os.path.join(OUT, f'{fname}_{m.split("/")[-1].split(".")[0]}.parquet'), index=False, float_format='%.4f')
    exit(0)

for fold in range(nfolds): # running multiple folds at kaggle may cause OOM
    ds_train = RNA_Dataset(df, mode='train', fold=fold, nfolds=nfolds,pretrain=args.pretrain)
    ds_train_len = RNA_Dataset(df, mode='train', fold=fold, 
                nfolds=nfolds, mask_only=True, pretrain=args.pretrain)
    sampler_train = torch.utils.data.RandomSampler(ds_train_len)
    len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs,
                drop_last=True)
    dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train, 
                batch_sampler=len_sampler_train, num_workers=num_workers,
                persistent_workers=True), device)

    ds_val = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds, pretrain=args.pretrain)
    ds_val_len = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds, 
               mask_only=True, pretrain=args.pretrain)
    sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
    len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=bs, 
               drop_last=False)
    dl_val= DeviceDataLoader(torch.utils.data.DataLoader(ds_val, 
               batch_sampler=len_sampler_val, num_workers=num_workers), device)
    gc.collect()

    data = DataLoaders(dl_train,dl_val)

    if not args.use_bert:
        model = RNA_Model(dim=args.dim, depth=args.depth)   
    else:
        model = RNA_Bert_Model(dim=args.dim, depth=args.depth, num_bert_layers=args.num_bert_layers)
    model = model.to(device)

    if args.use_fastai:
        cbs = []
        cbs.append(GradientClip(3.0))
        if args.add_noise:
            cbs.append(ErrAug(max_noise=args.max_noise, p=args.noise_p))
            
        if args.use_combine_loss:
            learn = Learner(data, model, loss_func=combine_loss,cbs=cbs,
                    metrics=[MAE()]).to_fp16()
        else:
            learn = Learner(data, model, loss_func=loss,cbs=cbs,
                        metrics=[MAE()]).to_fp16() 

        learn.fit_one_cycle(args.ep, lr_max=5e-4, wd=0.05, pct_start=0.02)
        torch.save(learn.model.state_dict(),os.path.join(OUT,f'{fname}_{fold}.pth'))
        gc.collect()

    else:
        # train
        for idx, batch in enumerate(tqdm(dl_train)):
            __import__("ipdb").set_trace()
            pred = model(batch[0])
            label = batch[1]
            loss_val = loss(pred, label)





