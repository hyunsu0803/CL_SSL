import torch
import numpy as np
import matplotlib.pyplot as plt


"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""


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
        
        # print(labelling)
        # plt.figure(figsize=(8, 4))
        # plt.plot(torch.rad2deg(distance), labelling.numpy(), label="Labelling Curve")
        # plt.title("Contrast Weight")
        # plt.xlabel("Distance (degrees)")
        # plt.ylabel("Labelling Value")
        # plt.grid(True)
        # plt.legend()
        # plt.savefig("Labelling_Curve.png")
        # exit()
        
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
        
        # mask = self.generate_mask(labels).to(self.device)         # (256, 256)
        mask = torch.eq(labels, labels.T).float().to(self.device)   # (B, B)
        

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
        logits = anchor_dot_contrast - logits_max.detach()  # (512, 512)    # Numerically stable and does not affect the result of softmax computation.

        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size, dtype=torch.float32).to(self.device)

        mask = mask * logits_mask       # Ensures that the i-th sample is not compared with itself by setting it to False. Likely to include only True values in the numerator, even if the same class is False.

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask        # (512, 512) # The diagonal is 0, so exp(0) = 1; compares all samples except itself.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (512, 512) - (512, 1) # All samples except itself minus the log of the sum of exp(all samples except itself), which becomes the denominator.
        # Currently, the numerator contains the similarity for all samples. After applying the mask, only the weighted positive pairs remain.

        mask_pos_pairs = mask.sum(1)                                            # (512,)    # |P(i)|
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # (512,)    # Replace with 1 if there are no positive pairs to prevent division by zero.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs           # (512,)    # The numerator initially contained similarities for all pairs, but applying the mask leaves only positive pairs.

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # Computes the mean across all samples (1/N).


        return loss
