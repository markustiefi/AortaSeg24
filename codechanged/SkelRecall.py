# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:35:20 2024

@author: q117mt
"""
from typing import Callable

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import torch.nn as nn
import torch

class SkelRecall(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, do_bg: bool = False, smooth: float = 1.):
        super().__init__()
        self.do_bg = do_bg
        self.smooth = smooth
        self.apply_nonlin = apply_nonlin

    @staticmethod
    def TubedSkeletonization(y, device):
        from skimage.morphology import skeletonize_3d
        from scipy.ndimage import binary_dilation
        #Clone for batch, is there another way?
        y_bin = y.clone()
        y_bin[y_bin>0] = 1
        y_skel = torch.zeros_like(y_bin.clone())
        for i in range(y_skel.shape[0]):
            y_skel_tmp = skeletonize_3d(y_bin[i].numpy())
            y_skel_tmp = binary_dilation(y_skel_tmp, iterations = 1)
            y_skel[i] = torch.from_numpy(y_skel_tmp)
        y_mcskel = y*y_skel
        y_mcskel = y_mcskel.to(device)
        return y_mcskel
        
    def forward(self, x, y, loss_mask: float = None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2,x.ndim))
        y_mcskel = self.TubedSkeletonization(y.squeeze().cpu(), x.device)
        with torch.no_grad():
            if x.ndim != y_mcskel.ndim:
                y_mcskel = y_mcskel.view((y_mcskel.shape[0], 1, *y_mcskel.shape[1:]))
            y_mcskel_onehot = torch.zeros(x.shape, device = x.device, dtype = torch.bool)
            y_mcskel_onehot.scatter_(1, y_mcskel.long(),1)
            
            if not self.do_bg:
                y_mcskel_onehot = y_mcskel_onehot[:, 1:]
            sum_skel = y_mcskel_onehot.sum(axes) if loss_mask is None else (y_mcskel_onehot * loss_mask).sum(axes)
        if not self.do_bg:
            x = x[:, 1:]

        tp = (y_mcskel_onehot*x).sum(axes)
        #print(f'tp: {tp}')
        #print(f'sum_skel: {sum_skel}')
        Loss_mcskel = -tp/torch.clip(sum_skel.float()+self.smooth, min =1e-8)
        #print(f'Loss_mcskel: {Loss_mcskel}')
        LskelRecall = torch.mean(Loss_mcskel)
        #print(f'LskelRecall: {LskelRecall}')
        return LskelRecall