import torch
import torch.nn as nn


"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

class EMA_Weighted_SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        
        super(EMA_Weighted_SupConLoss, self).__init__()
        
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
            
        self.azi_size=360
        self.degree_resolution=360/self.azi_size
        self.sigma = None
        self.p = torch.tensor([0.707106781])
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

        
    
    def generate_mask(self, labels):

        distance = torch.abs(labels - labels.T)     # (512, 512)
        distance = torch.where(distance>180, 360-distance, distance)
        distance = torch.where(abs(distance)>180, 180, distance)
        
        self.labelling = self.labelling.to(distance.device)
        mask = self.labelling[distance]
        
        return mask
    
          

    def forward(self, feature_s, feature_t, labels, sigma):

        self.device = (torch.device('cuda:0')
                  if feature_s.is_cuda
                  else torch.device('cpu'))


        

        batch_size = feature_s.shape[0]
                         
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        


        if sigma == 0:
            mask = torch.eq(labels, labels.T).float().to(self.device)   # (B, B)
        else:
            self.sigma = torch.tensor(sigma)
            self.labelling = self.generate_weight().to(self.device)
            mask = self.generate_mask(labels).to(self.device)           # (B, B)



        anchor_feature = feature_s              # (B, 128)
        contrast_feature = feature_t            # (B, 128)

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)


        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # (512, 512)    # Numerically stable and does not affect the result of softmax computation.

        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size, dtype=torch.float32).to(self.device)

        mask = mask * logits_mask       # Ensures that the i-th sample is not compared with itself by setting it to False. Likely to include only True values in the numerator, even if the same class is False.



        exp_logits = torch.exp(logits) * logits_mask        # (512, 512) # The diagonal is 0, so exp(0) = 1; compares all samples except itself.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (512, 512) - (512, 1) # All samples except itself minus the log of the sum of exp(all samples except itself), which becomes the denominator.
        # Currently, the numerator contains the similarity for all samples. After applying the mask, only the weighted positive pairs remain.



        mask_pos_pairs = mask.sum(1)                                            # (512,)    # |P(i)|
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # (512,)    # Replace with 1 if there are no positive pairs to prevent division by zero.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs           # (512,)    # The numerator initially contained similarities for all pairs, but applying the mask leaves only positive pairs.

        
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # Computes the mean across all samples (1/N).


        return loss
