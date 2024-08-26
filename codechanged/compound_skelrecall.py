# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:35:20 2024

@author: q117mt
"""
from typing import Callable

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.training.loss.SkelRecall import SkelRecall
from nnunetv2.utilities.helpers import softmax_helper_dim1
import torch.nn as nn
import torch
    
class DC_and_CE_and_SkelRecall_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, skel_recall_kwargs, weight_ce=1, weight_dice=1, weight_skelrecall=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param skel_recal_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        :param weight_skelrecall:
        """
        super(DC_and_CE_and_SkelRecall_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
            

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_skelrecall = weight_skelrecall
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.sr = SkelRecall(apply_nonlin=softmax_helper_dim1, **skel_recall_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        sr_loss = self.sr(net_output, target_dice, loss_mask = mask) \
            if self.weight_skelrecall != 0 else 0
                

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_skelrecall * sr_loss
        return result

class DC_and_topk_and_SkelRecall_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, skel_recall_kwargs, weight_ce=1, weight_dice=1, weight_skelrecall=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param skel_recal_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        :param weight_skelrecall:
        """
        super(DC_and_topk_and_SkelRecall_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
            

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_skelrecall = weight_skelrecall
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.sr = SkelRecall(apply_nonlin=softmax_helper_dim1, **skel_recall_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        sr_loss = self.sr(net_output, target_dice, loss_mask = mask) \
            if self.weight_skelrecall != 0 else 0
                

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_skelrecall * sr_loss
        return result
