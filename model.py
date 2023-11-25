import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RNA_Model(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4,dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        self.proj_out = nn.Linear(dim,2)
    
    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:,:Lmax]
        x = x0['seq'][:,:Lmax]
        
        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos
        
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.proj_out(x)
        
        return x
    
from transformers import BertModel

class Bert_Layers(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask)[0]
        return x

# bert_model = BertModel.from_pretrained('bert-base-uncased')
class RNA_Bert_Model(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, num_bert_layers=2, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4,dim)
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        bert_dim = bert_model.config.hidden_size
        self.to_bert = nn.Linear(dim, bert_dim)
        self.from_bert = nn.Linear(bert_dim, dim)
        self.bert_layers = Bert_Layers(bert_model.encoder.layer[:num_bert_layers])
        self.pos_enc = SinusoidalPosEmb(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        self.proj_out = nn.Linear(dim,2)
        
        self.get_extended_attention_mask = bert_model.get_extended_attention_mask
    
    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:,:Lmax]
        x = x0['seq'][:,:Lmax]
        
        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos
        
        extend_attn_mask = self.get_extended_attention_mask(mask, mask.shape, x.device)
        x = self.to_bert(x)
        x = self.bert_layers(x, attn_mask=extend_attn_mask)
        x = self.from_bert(x)
        
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.proj_out(x)
        
        return x

def loss(pred, target):
    p = pred[target['mask'][:,:pred.shape[1]]]
    y = target['react'][target['mask']].clip(0,1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    
    return loss

def combine_loss(pred, target):
    p = pred[target['mask'][:,:pred.shape[1]]]
    y = target['react'][target['mask']].clip(0,1)
    
    # loss = l1 + l2
    loss = F.l1_loss(p, y, reduction='none') + F.mse_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    return loss
    