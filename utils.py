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




class MAE(Metric):
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        self.x,self.y = [],[]
        
    def accumulate(self, learn):
        x = learn.pred[learn.y['mask'][:,:learn.pred.shape[1]]]
        y = learn.y['react'][learn.y['mask']].clip(0,1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        x,y = torch.cat(self.x,0),torch.cat(self.y,0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss
    

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
