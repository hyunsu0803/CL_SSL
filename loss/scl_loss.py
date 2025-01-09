import torch
import numpy as np
# from torch.nn.modules.loss import _Loss


"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
# from __future__ import print_function

import torch
import torch.nn as nn


class Weighted_SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07,
                 weights=[1,1], step_size=0.9999, step_per_iter=1, last_weight=1, loss_resolution=None):
        
        super(Weighted_SupConLoss, self).__init__()
        
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
            
        self.azi_size=360
        self.degree_resolution=360/self.azi_size
        self.sigma = torch.tensor([2.0])
        self.p = torch.tensor([0.707106781])
        self.labelling = self.generate_weight()
        self.contrast_count = None
        self.device = None
        self.contrast_mode = 'all'
            
            
    def generate_weight(self,):
        
        distance=torch.arange(0, self.azi_size//2+1)            # (181)  0~180

        distance=torch.deg2rad(distance)                        # (181)
                          
        sigma=torch.deg2rad(self.sigma)
        kappa_d=torch.log(self.p)/(torch.cos(sigma)-1)          # (181)
        

        labelling=torch.exp(kappa_d*(torch.cos(distance)-1))    # (181)
        
        return labelling

        # self.sigma_update(iter_num, epoch)
        
    
    def generate_mask(self, labels):
        
        distance = torch.abs(labels - labels.T)     # (512, 512)
        distance = torch.where(distance>180, 360-distance, distance)
        
        self.labelling = self.labelling.to(distance.device)
        mask = self.labelling[distance]
        
        return mask
            

    def forward(self, features, labels):
        """Compute loss for model. 
        Args:
            features: hidden vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        self.device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        
        # labels: (B,)                  
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        mask = self.generate_mask(labels).to(self.device)         # (256, 256)
        # mask = torch.eq(labels, labels.T).float().to(self.device)   # (B, B)
        

        contrast_feature = features
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (256, 2, 128) -> (512, 128)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            # anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            # anchor_count = self.contrast_count       # 2
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # (512, 512)    # 수치적으로 안전하면서도 softmax의 계산 결과에는 영향을 주지 않음.

        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size, dtype=torch.float32).to(self.device)
        
        mask = mask * logits_mask       # i 샘플과 i 샘플은 비교하지 않음. false로 만들어줌. 아마 true인 경우만 분자에 올리려고 하나봄. 같은 class 아니어도 False.

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask        # (512, 512) # 대각선은 0이므로 exp(0) = 1, 자기 자신 제외 전체 샘플 비교한 것. 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (512, 512) - (512, 1) # 자기 자신 제외한 전체 샘플 - 각 샘플 별 log(exp(자기 자신 제외한 전체 샘플의 합))(이부분이 분모)
        # 지금 분자에도 모든 샘플에 대한 similarity가 있음. 밑에서 mask를 곱해주면 weighted pos pair만 남음.
         
        mask_pos_pairs = mask.sum(1)                                            # (512,)    # |P(i)|
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # (512,)    # pos pair가 없는 경우 1로 대체, division by zero 방지
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs           # (512,)    # 분자에 모든 pair의 similarity가 있었는데 mask를 곱해주면 pos pair만 남음.

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # 모든 i에 대해서 평균을 냄. (1/N)

        return loss
