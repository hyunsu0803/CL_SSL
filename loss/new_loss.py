import torch
import torch.nn as nn
import torch.nn.functional as F


###########################################################

class SupCon_Loss(nn.Module):   
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):

        super(SupCon_Loss, self).__init__()

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
            

    def forward(self, features, labels, sigma):

        self.device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))


        

        batch_size = features.shape[0]
                         
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size: 
            raise ValueError('Num of labels does not match num of features')
        

        mask = torch.eq(labels, labels.T).float().to(self.device)


        contrast_feature = features         # (B, 128)
        anchor_feature = contrast_feature   # (B, 128)

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # Underflow 및 Overflow 방지, logits: 내적/temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # (512, 512)    # Numerically stable and does not affect the result of softmax computation.

        # 자기 자신과의 비교를 제외하고자
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask = mask * logits_mask       # Ensures that the i-th sample is not compared with itself by setting it to False. Likely to include only True values in the numerator, even if the same class is False.


        # 분모항 계산 (negative pair들과의 합)
        exp_logits = torch.exp(logits) * logits_mask        # (512, 512) # The diagonal is 0, so exp(0) = 1; compares all samples except itself.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (512, 512) - (512, 1) # All samples except itself minus the log of the sum of exp(all samples except itself), which becomes the denominator.
        # Currently, the numerator contains the similarity for all samples. After applying the mask, only the weighted positive pairs remain.


        # positive pair들의 평균 계산
        mask_pos_pairs = mask.sum(1)                                            # (512,)    # |P(i)|
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # (512,)    # Replace with 1 if there are no positive pairs to prevent division by zero.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs           # (512,)    # The numerator initially contained similarities for all pairs, but applying the mask leaves only positive pairs.

        # 최종 Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # Computes the mean across all samples (1/N).


        return loss

class Weighted_SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        
        super(Weighted_SupConLoss, self).__init__()
        
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
    
          

    def forward(self, features, labels, sigma):

        self.device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))


        

        batch_size = features.shape[0]
                         
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        


        if sigma == 0:
            mask = torch.eq(labels, labels.T).float().to(self.device)   # (B, B)
        else:
            self.sigma = torch.tensor(sigma)
            self.labelling = self.generate_weight().to(self.device)
            mask = self.generate_mask(labels).to(self.device)           # (B, B)



        contrast_feature = features         # (B, 128)
        anchor_feature = contrast_feature   # (B, 128)

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

# Positive pair: exp(cos(theta) - dynamic_margin) where dynamic = margin * exp(1-cos(theta))/lambda
# Negative pair: exp(cos(theta))
class DynamicMargin_Loss(nn.Module):   
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, margin = 0.3):
        
        super(DynamicMargin_Loss, self).__init__()
        
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.margin = margin
        self.lambda_val = 2.0  # lambda = 2

        self.azi_size=360
        self.degree_resolution=360/self.azi_size
        self.sigma = None
        self.p = torch.tensor([0.707106781])
        self.contrast_count = None
        self.device = None
        self.contrast_mode = 'all'
            

    def forward(self, features, labels, sigma):

        self.device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))



        batch_size = features.shape[0]
                         
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        

        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_feature = features         # (B, 128)
        anchor_feature = contrast_feature   # (B, 128)


        contrast_norm = F.normalize(contrast_feature, dim=1)  
        anchor_norm = F.normalize(anchor_feature, dim=1)      

        # cos(theta) 계산
        cos_theta = torch.matmul(anchor_norm, contrast_norm.T)  # (B, B)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
        
        # Dynamic margin 계산: margin * exp(1 - cos(theta)) / lambda
        dynamic_margin = self.margin * torch.exp(1.0 - cos_theta) / self.lambda_val  # (B, B)
        
        # Positive pair: exp(cos(theta) - dynamic_margin)
        # Negative pair: exp(cos(theta))
        pos_logits = cos_theta - dynamic_margin  # positive pair용
        neg_logits = cos_theta  # negative pair용
        
        # 마스크에 따라 positive/negative logits 적용
        logits = mask * pos_logits + (1.0 - mask) * neg_logits
        logits = logits / self.temperature

        # Underflow 및 Overflow 방지, logits: 내적/temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # (512, 512)    # Numerically stable and does not affect the result of softmax computation.

        # 자기 자신과의 비교를 제외하고자
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask = mask * logits_mask       # Ensures that the i-th sample is not compared with itself by setting it to False. Likely to include only True values in the numerator, even if the same class is False.


        # 분모항 계산 (negative pair들과의 합)
        exp_logits = torch.exp(logits) * logits_mask        # (512, 512) # The diagonal is 0, so exp(0) = 1; compares all samples except itself.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (512, 512) - (512, 1) # All samples except itself minus the log of the sum of exp(all samples except itself), which becomes the denominator.
        # Currently, the numerator contains the similarity for all samples. After applying the mask, only the weighted positive pairs remain.


        # positive pair들의 평균 계산
        mask_pos_pairs = mask.sum(1)                                            # (512,)    # |P(i)|
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # (512,)    # Replace with 1 if there are no positive pairs to prevent division by zero.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs           # (512,)    # The numerator initially contained similarities for all pairs, but applying the mask leaves only positive pairs.

        # 최종 Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # Computes the mean across all samples (1/N).


        return loss

class ArcCos_Loss(nn.Module):   
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, margin = 0.3):

        super(ArcCos_Loss, self).__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.margin = margin

        self.azi_size=360
        self.degree_resolution=360/self.azi_size
        self.sigma = None
        self.p = torch.tensor([0.707106781])
        self.contrast_count = None
        self.device = None
        self.contrast_mode = 'all'
            

    def forward(self, features, labels, sigma):

        self.device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))



        batch_size = features.shape[0]
                         
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        

        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_feature = features         # (B, 128)
        anchor_feature = contrast_feature   # (B, 128)


        contrast_norm = F.normalize(contrast_feature, dim=1)  
        anchor_norm = F.normalize(anchor_feature, dim=1)      

        # Cos(theta + m) = Cos(theta) * Cos(m) - Sin(theta) * Sin(m)
        cos_theta = torch.matmul(anchor_norm, contrast_norm.T)  # (B, B)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)  # (B, B)
        cos_m = torch.cos(torch.tensor(self.margin))
        sin_m = torch.sin(torch.tensor(self.margin))
    
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m  # (B, B)


        logits = cos_theta.clone()
        # Positive pair에는 cos(theta + m), Negative pair에는 cos(theta)
        logits = mask * cos_theta_m + (1.0 - mask) * logits
        logits = logits / self.temperature

        # Underflow 및 Overflow 방지, logits: 내적/temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # (512, 512)    # Numerically stable and does not affect the result of softmax computation.

        # 자기 자신과의 비교를 제외하고자
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask = mask * logits_mask       # Ensures that the i-th sample is not compared with itself by setting it to False. Likely to include only True values in the numerator, even if the same class is False.


        # 분모항 계산 (negative pair들과의 합)
        exp_logits = torch.exp(logits) * logits_mask        # (512, 512) # The diagonal is 0, so exp(0) = 1; compares all samples except itself.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (512, 512) - (512, 1) # All samples except itself minus the log of the sum of exp(all samples except itself), which becomes the denominator.
        # Currently, the numerator contains the similarity for all samples. After applying the mask, only the weighted positive pairs remain.


        # positive pair들의 평균 계산
        mask_pos_pairs = mask.sum(1)                                            # (512,)    # |P(i)|
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # (512,)    # Replace with 1 if there are no positive pairs to prevent division by zero.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs           # (512,)    # The numerator initially contained similarities for all pairs, but applying the mask leaves only positive pairs.

        # 최종 Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # Computes the mean across all samples (1/N).


        return loss

###########################################################

class NegDot_Loss(nn.Module):   
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        
        super(NegDot_Loss, self).__init__()
        
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
            

    def forward(self, features, labels, sigma):

        self.device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))


        

        batch_size = features.shape[0]
                         
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size: 
            raise ValueError('Num of labels does not match num of features')
        

        mask = torch.eq(labels, labels.T).float().to(self.device)

        # if sigma == 0:
        #     mask = torch.eq(labels, labels.T).float().to(self.device)   # (B, B)
        # else:
        #     self.sigma = torch.tensor(sigma)
        #     self.labelling = self.generate_weight().to(self.device)
        #     mask = self.generate_mask(labels).to(self.device)           # (B, B)


        contrast_feature = features         # (B, 128)
        anchor_feature = contrast_feature   # (B, 128)

        # cos(pi-theta)를 위해 -를 앞에 붙여주었음
        anchor_dot_contrast = -torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # Underflow 및 Overflow 방지, logits: 내적/temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # (512, 512)    # Numerically stable and does not affect the result of softmax computation.

        # 자기 자신과의 비교를 제외하고자
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask = mask * logits_mask       # Ensures that the i-th sample is not compared with itself by setting it to False. Likely to include only True values in the numerator, even if the same class is False.


        # 분모항 계산 (negative pair들과의 합)
        exp_logits = torch.exp(logits) * logits_mask        # (512, 512) # The diagonal is 0, so exp(0) = 1; compares all samples except itself.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (512, 512) - (512, 1) # All samples except itself minus the log of the sum of exp(all samples except itself), which becomes the denominator.
        # Currently, the numerator contains the similarity for all samples. After applying the mask, only the weighted positive pairs remain.


        # positive pair들의 평균 계산
        mask_pos_pairs = mask.sum(1)                                            # (512,)    # |P(i)|
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # (512,)    # Replace with 1 if there are no positive pairs to prevent division by zero.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs           # (512,)    # The numerator initially contained similarities for all pairs, but applying the mask leaves only positive pairs.

        # 최종 Loss
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # Computes the mean across all samples (1/N).


        return loss

class NegWeighted_SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        
        super(NegWeighted_SupConLoss, self).__init__()
        
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
    
          

    def forward(self, features, labels, sigma):

        self.device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))


        batch_size = features.shape[0]
                         
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        

        if sigma == 0:
            mask = torch.eq(labels, labels.T).float().to(self.device)   # (B, B)
        else:
            self.sigma = torch.tensor(sigma)
            self.labelling = self.generate_weight().to(self.device)
            mask = self.generate_mask(labels).to(self.device)           # (B, B)


        contrast_feature = features         # (B, 128)
        anchor_feature = contrast_feature   # (B, 128)

        anchor_dot_contrast = torch.div(
            -torch.matmul(anchor_feature, contrast_feature.T),
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

    
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # Computes the mean across all samples (1/N).


        return loss

class NegDynamicMargin_Loss(nn.Module):   

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, margin = 0.3):
        
        super(NegDynamicMargin_Loss, self).__init__()
        
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.margin = margin
        self.lambda_val = 2.0  # lambda = 2

        self.azi_size=360
        self.degree_resolution=360/self.azi_size
        self.sigma = None
        self.p = torch.tensor([0.707106781])
        self.contrast_count = None
        self.device = None
        self.contrast_mode = 'all'
            

    def forward(self, features, labels, sigma):

        self.device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))



        batch_size = features.shape[0]
                         
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        

        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_feature = features         # (B, 128)
        anchor_feature = contrast_feature   # (B, 128)


        contrast_norm = F.normalize(contrast_feature, dim=1)  
        anchor_norm = F.normalize(anchor_feature, dim=1)      

        # DynamicMargin_Loss
        # Positive: exp( (cos(theta) - dynamic_margin) / tau )
        # Negative: exp( cos(theta) / tau )
        # dynamic_margin = margin * exp(1 - cos(theta)) / lambda

        # NegativeDynamicMargin_Loss
        # Positive: exp( (cos(PI-theta) - dynamic_margin) / tau )
        # Negative: exp( cos(PI-theta) / tau )
        # dynamic_margin = margin * exp(1 - cos(PI-theta)) / lambda


        # cos(PI-theta) 계산
        cos_theta = -torch.matmul(anchor_norm, contrast_norm.T)  # (B, B)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
        
        # Dynamic margin 계산: margin * exp(1 - cos(theta)) / lambda
        dynamic_margin = self.margin * torch.exp(1.0 - cos_theta) / self.lambda_val  # (B, B)
        
        # Positive pair: exp(cos(theta) - dynamic_margin)
        # Negative pair: exp(cos(theta))
        pos_logits = cos_theta - dynamic_margin  # positive pair용
        neg_logits = cos_theta  # negative pair용
        
        # 마스크에 따라 positive/negative logits 적용
        logits = mask * pos_logits + (1.0 - mask) * neg_logits
        logits = logits / self.temperature

        # Underflow 및 Overflow 방지, logits: 내적/temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # (512, 512)    # Numerically stable and does not affect the result of softmax computation.

        # 자기 자신과의 비교를 제외하고자
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask = mask * logits_mask       # Ensures that the i-th sample is not compared with itself by setting it to False. Likely to include only True values in the numerator, even if the same class is False.


        # 분모항 계산 (negative pair들과의 합)
        exp_logits = torch.exp(logits) * logits_mask        # (512, 512) # The diagonal is 0, so exp(0) = 1; compares all samples except itself.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (512, 512) - (512, 1) # All samples except itself minus the log of the sum of exp(all samples except itself), which becomes the denominator.
        # Currently, the numerator contains the similarity for all samples. After applying the mask, only the weighted positive pairs remain.


        # positive pair들의 평균 계산
        mask_pos_pairs = mask.sum(1)                                            # (512,)    # |P(i)|
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # (512,)    # Replace with 1 if there are no positive pairs to prevent division by zero.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs           # (512,)    # The numerator initially contained similarities for all pairs, but applying the mask leaves only positive pairs.

        # 최종 Loss
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # Computes the mean across all samples (1/N).


        return loss

class NegArcCos_Loss(nn.Module):   
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, margin = 0.3):

        super(NegArcCos_Loss, self).__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.margin = margin

        self.azi_size=360
        self.degree_resolution=360/self.azi_size
        self.sigma = None
        self.p = torch.tensor([0.707106781])
        self.contrast_count = None
        self.device = None
        self.contrast_mode = 'all'
            

    def forward(self, features, labels, sigma):

        self.device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))



        batch_size = features.shape[0]
                         
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        

        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_feature = features         # (B, 128)
        anchor_feature = contrast_feature   # (B, 128)


        contrast_norm = F.normalize(contrast_feature, dim=1)  
        anchor_norm = F.normalize(anchor_feature, dim=1)      

        # Cos(theta + m) = Cos(theta) * Cos(m) - Sin(theta) * Sin(m)
        cos_theta = torch.matmul(anchor_norm, contrast_norm.T)  # (B, B)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)  # (B, B)
        cos_m = torch.cos(torch.tensor(self.margin))
        sin_m = torch.sin(torch.tensor(self.margin))
    
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m  # (B, B)
        

        logits = cos_theta.clone()
        # Positive pair에는 cos(theta + m), Negative pair에는 cos(theta)
        logits = mask * (-cos_theta_m) + (1.0 - mask) * (-logits)
        logits = logits / self.temperature

        # Underflow 및 Overflow 방지, logits: 내적/temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # (512, 512)    # Numerically stable and does not affect the result of softmax computation.

        # 자기 자신과의 비교를 제외하고자
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask = mask * logits_mask       # Ensures that the i-th sample is not compared with itself by setting it to False. Likely to include only True values in the numerator, even if the same class is False.


        # 분모항 계산 (negative pair들과의 합)
        exp_logits = torch.exp(logits) * logits_mask        # (512, 512) # The diagonal is 0, so exp(0) = 1; compares all samples except itself.
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (512, 512) - (512, 1) # All samples except itself minus the log of the sum of exp(all samples except itself), which becomes the denominator.
        # Currently, the numerator contains the similarity for all samples. After applying the mask, only the weighted positive pairs remain.


        # positive pair들의 평균 계산
        mask_pos_pairs = mask.sum(1)                                            # (512,)    # |P(i)|
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # (512,)    # Replace with 1 if there are no positive pairs to prevent division by zero.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs           # (512,)    # The numerator initially contained similarities for all pairs, but applying the mask leaves only positive pairs.

        # 최종 Loss
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # Computes the mean across all samples (1/N).


        return loss



###########################################################
# 여러가지 Loss들을 식 하나의 꼴로 통합
class NegWeightedArc_Loss(nn.Module):
    """Supervised Contrastive Learning with negative-weighted positives (your version)
       + ArcFace-style additive angular margin on POSITIVE pairs only.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, margin=0.30):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.margin = margin
        self.azi_size = 360
        self.p = torch.tensor([0.707106781])

    def generate_weight(self, sigma, device):
        distance = torch.arange(0, self.azi_size // 2 + 1, device=device)
        distance = torch.deg2rad(distance)
        sigma = torch.deg2rad(torch.tensor(sigma, device=device))
        kappa_d = torch.log(self.p.to(device)) / (torch.cos(sigma) - 1)
        labelling = torch.exp(kappa_d * (torch.cos(distance) - 1))
        return labelling

    def generate_mask(self, labels, labelling):
        distance = torch.abs(labels - labels.T)
        distance = torch.where(distance > 180, 360 - distance, distance)
        distance = torch.where(torch.abs(distance) > 180, 180, distance)
        mask = labelling[distance]
        return mask

    def forward(self, features, labels, sigma):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # autocast로 mixed precision 안전하게 적용
        with torch.cuda.amp.autocast(enabled=features.is_cuda):
            if sigma == 0:
                mask = torch.eq(labels, labels.T).float().to(device)
            else:
                labelling = self.generate_weight(sigma, device)
                mask = self.generate_mask(labels, labelling).to(device)

            contrast_feature = F.normalize(features, dim=1)
            cos_theta = torch.matmul(contrast_feature, contrast_feature.T)
            cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

            if self.margin == 0:
                s = cos_theta
            else:
                sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=1e-12))
                cos_m = torch.cos(torch.tensor(self.margin, device=device, dtype=cos_theta.dtype))
                sin_m = torch.sin(torch.tensor(self.margin, device=device, dtype=cos_theta.dtype))
                cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
                pos_mask_binary = (mask > 0).to(cos_theta.dtype)
                s = pos_mask_binary * cos_theta_m + (1.0 - pos_mask_binary) * cos_theta

            anchor_dot_contrast = -s / self.temperature
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            logits_mask = torch.ones_like(mask, device=device) - torch.eye(batch_size, dtype=torch.float32, device=device)
            mask = mask * logits_mask

            exp_logits = torch.exp(logits) * logits_mask
            denom = exp_logits.sum(1, keepdim=True).clamp(min=1e-12)
            log_prob = logits - torch.log(denom)

            mask_pos_pairs = mask.sum(1)
            mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

            loss = (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()

            # nan/inf 방지: loss가 nan/inf면 큰 값으로 대체
            if not torch.isfinite(loss):
                loss = torch.tensor(1e6, device=loss.device, dtype=loss.dtype)
        return loss

class WeightedArc_Loss(nn.Module):
    """Supervised Contrastive Learning with negative-weighted positives (your version)
       + ArcFace-style additive angular margin on POSITIVE pairs only.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, margin=0.30):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.margin = margin
        self.azi_size = 360
        self.p = torch.tensor([0.707106781])

    def generate_weight(self, sigma, device):
        distance = torch.arange(0, self.azi_size // 2 + 1, device=device)
        distance = torch.deg2rad(distance)
        sigma = torch.deg2rad(torch.tensor(sigma, device=device))
        kappa_d = torch.log(self.p.to(device)) / (torch.cos(sigma) - 1)
        labelling = torch.exp(kappa_d * (torch.cos(distance) - 1))
        return labelling

    def generate_mask(self, labels, labelling):
        distance = torch.abs(labels - labels.T)
        distance = torch.where(distance > 180, 360 - distance, distance)
        distance = torch.where(torch.abs(distance) > 180, 180, distance)
        mask = labelling[distance]
        return mask

    def forward(self, features, labels, sigma):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')

        # autocast로 mixed precision 안전하게 적용
        with torch.cuda.amp.autocast(enabled=features.is_cuda):
            if sigma == 0:
                mask = torch.eq(labels, labels.T).float().to(device)
            else:
                labelling = self.generate_weight(sigma, device)
                mask = self.generate_mask(labels, labelling).to(device)

            contrast_feature = F.normalize(features, dim=1)
            cos_theta = torch.matmul(contrast_feature, contrast_feature.T)
            cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

            if self.margin == 0:
                s = cos_theta
            else:
                sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=1e-12))
                cos_m = torch.cos(torch.tensor(self.margin, device=device, dtype=cos_theta.dtype))
                sin_m = torch.sin(torch.tensor(self.margin, device=device, dtype=cos_theta.dtype))
                cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
                pos_mask_binary = (mask > 0).to(cos_theta.dtype)
                s = pos_mask_binary * cos_theta_m + (1.0 - pos_mask_binary) * cos_theta

            anchor_dot_contrast = + s / self.temperature
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            logits_mask = torch.ones_like(mask, device=device) - torch.eye(batch_size, dtype=torch.float32, device=device)
            mask = mask * logits_mask

            exp_logits = torch.exp(logits) * logits_mask
            denom = exp_logits.sum(1, keepdim=True).clamp(min=1e-12)
            log_prob = logits - torch.log(denom)

            mask_pos_pairs = mask.sum(1)
            mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

            loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()

            # nan/inf 방지: loss가 nan/inf면 큰 값으로 대체
            if not torch.isfinite(loss):
                loss = torch.tensor(1e6, device=loss.device, dtype=loss.dtype)
        return loss

###########################################################
# Negative Pair에다가 mean을 취하는 방식들

class MeanNegDot_Loss(nn.Module):   


    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        
        super(MeanNegDot_Loss, self).__init__()
        
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
            

    def forward(self, features, labels, sigma):

        self.device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))


        

        batch_size = features.shape[0]
                         
        labels = labels.contiguous().view(-1, 1)    # (B, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        

        mask = torch.eq(labels, labels.T).float().to(self.device)
        contrast_feature = features         # (B, 128)
        anchor_feature = contrast_feature   # (B, 128)

        # cos(pi-theta)를 위해 -를 앞에 붙여주었음
        anchor_dot_contrast = -torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # Underflow 및 Overflow 방지, logits: 내적/temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # (512, 512)    # Numerically stable and does not affect the result of softmax computation.

        # 자기 자신과의 비교를 제외하고자
        logits_mask = torch.ones_like(mask).to(self.device) - torch.eye(batch_size, dtype=torch.float32).to(self.device)
        mask = mask * logits_mask       # Ensures that the i-th sample is not compared with itself by setting it to False. Likely to include only True values in the numerator, even if the same class is False.


        # 분모항 계산 (negative pair들의 평균 + positive pairs)
        exp_logits = torch.exp(logits) * logits_mask        # (512, 512) 
        
        # negative pairs만 따로 계산해서 평균 구하기
        negative_mask = logits_mask - mask  # negative pairs만 True
        exp_negatives = torch.exp(logits) * negative_mask   # (512, 512)
        negative_counts = negative_mask.sum(1, keepdim=True)  # (512, 1)
        mean_exp_negatives = exp_negatives.sum(1, keepdim=True) / (negative_counts + 1e-8)  # (512, 1)
        
        # positive pairs는 그대로 유지
        exp_positives = torch.exp(logits) * mask  # (512, 512)
        
        # 분모 = positive pairs + mean(negative pairs)
        denominator = exp_positives.sum(1, keepdim=True) + mean_exp_negatives  # (512, 1)
        
        log_prob = logits - torch.log(denominator)  # (512, 512) - (512, 1)
        # Currently, the numerator contains the similarity for all samples. After applying the mask, only the weighted positive pairs remain.


        # positive pair들의 평균 계산
        mask_pos_pairs = mask.sum(1)                                            # (512,)    # |P(i)|
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)  # (512,)    # Replace with 1 if there are no positive pairs to prevent division by zero.
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs           # (512,)    # The numerator initially contained similarities for all pairs, but applying the mask leaves only positive pairs.

        # 최종 Loss
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos # (512,)
        loss = loss.mean()                       # Computes the mean across all samples (1/N).


        return loss

# =========================
# MeanNegWeighted_SupConLoss
# =========================
# 얘 코드 이따가 다시 보기 Weight Negative에 반영 안된 것 같음

class MeanNegWeighted_SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(MeanNegWeighted_SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.azi_size = 360
        self.degree_resolution = 360 / self.azi_size
        self.sigma = None
        self.p = torch.tensor([0.707106781])
        self.contrast_count = None
        self.device = None
        self.contrast_mode = 'all'

    def generate_weight(self,):
        distance = torch.arange(0, self.azi_size//2 + 1)      # (181) 0~180
        distance = torch.deg2rad(distance)
        sigma = torch.deg2rad(self.sigma)
        kappa_d = torch.log(self.p) / (torch.cos(sigma) - 1)
        labelling = torch.exp(kappa_d * (torch.cos(distance) - 1))
        return labelling

    def generate_mask(self, labels):
        distance = torch.abs(labels - labels.T)
        distance = torch.where(distance > 180, 360 - distance, distance)
        distance = torch.where(torch.abs(distance) > 180, torch.tensor(180, device=distance.device, dtype=distance.dtype), distance)
        self.labelling = self.labelling.to(distance.device)
        mask = self.labelling[distance]
        return mask

    def forward(self, features, labels, sigma):
        self.device = (torch.device('cuda:0') if features.is_cuda else torch.device('cpu'))
        B = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != B:
            raise ValueError('Num of labels does not match num of features')

        if sigma == 0:
            mask = torch.eq(labels, labels.T).float().to(self.device)      # (B,B), 0/1
        else:
            self.sigma = torch.tensor(sigma)
            self.labelling = self.generate_weight().to(self.device)        # (181,)
            mask = self.generate_mask(labels).to(self.device)              # (B,B), weighted in [0,1]

        contrast = features
        anchor = contrast

        # logits (use cos(pi-theta) = -cos(theta) effect by putting minus)
        logits = -torch.matmul(anchor, contrast.T) / self.temperature       # (B,B)

        # numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # remove self-comparisons
        logits_mask = torch.ones_like(logits, device=self.device) - torch.eye(B, dtype=torch.float32, device=self.device)

        # positives: any position with mask>0 is considered positive position
        pos_mask_bin = (mask > 0).float() * logits_mask
        neg_mask = logits_mask - pos_mask_bin

        exp_logits = torch.exp(logits)

        # weighted positives (use the given continuous mask as weights)
        exp_pos = exp_logits * mask

        # negatives mean
        exp_neg = exp_logits * neg_mask
        neg_counts = neg_mask.sum(1, keepdim=True)                          # (B,1)
        mean_exp_neg = exp_neg.sum(1, keepdim=True) / (neg_counts + 1e-8)   # (B,1)

        # denominator = sum of positive terms + mean(negative terms)
        denominator = exp_pos.sum(1, keepdim=True) + mean_exp_neg

        log_prob = logits - torch.log(denominator)

        # average over positives for each anchor (use mask weights)
        pos_weights_sum = mask.sum(1)                                        # (B,)
        pos_weights_sum = torch.where(pos_weights_sum < 1e-6, torch.ones_like(pos_weights_sum), pos_weights_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_weights_sum

        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


# =========================
# MeanNegDynamicMargin_Loss
# =========================
class MeanNegDynamicMargin_Loss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, margin=0.3):
        super(MeanNegDynamicMargin_Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.margin = margin
        self.lambda_val = 2.0  # lambda = 2
        self.azi_size = 360
        self.degree_resolution = 360 / self.azi_size
        self.sigma = None
        self.p = torch.tensor([0.707106781])
        self.contrast_count = None
        self.device = None
        self.contrast_mode = 'all'

    def forward(self, features, labels, sigma):
        self.device = (torch.device('cuda:0') if features.is_cuda else torch.device('cpu'))
        B = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != B:
            raise ValueError('Num of labels does not match num of features')

        # binary positives
        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast = F.normalize(features, dim=1)
        anchor = contrast

        # use cos(pi - theta) = -cos(theta)
        cos_theta = -torch.matmul(anchor, contrast.T).clamp(-1 + 1e-7, 1 - 1e-7)   # (B,B)

        # dynamic margin (element-wise)
        dynamic_margin = self.margin * torch.exp(1.0 - cos_theta) / self.lambda_val # (B,B)

        # construct separate logits for pos/neg BEFORE stabilization
        logits_pos = (cos_theta - dynamic_margin) / self.temperature
        logits_neg = (cos_theta) / self.temperature

        # masks
        logits_mask = torch.ones_like(mask, device=self.device) - torch.eye(B, dtype=torch.float32, device=self.device)
        pos_mask_bin = mask * logits_mask
        neg_mask = logits_mask - pos_mask_bin

        # combine for row-wise max stabilization
        logits_combined = pos_mask_bin * logits_pos + neg_mask * logits_neg

        logits_max, _ = torch.max(logits_combined, dim=1, keepdim=True)
        logits_stable = logits_combined - logits_max.detach()

        exp_logits = torch.exp(logits_stable)

        exp_pos = exp_logits * pos_mask_bin
        exp_neg = exp_logits * neg_mask

        neg_counts = neg_mask.sum(1, keepdim=True)
        mean_exp_neg = exp_neg.sum(1, keepdim=True) / (neg_counts + 1e-8)

        denominator = exp_pos.sum(1, keepdim=True) + mean_exp_neg
        log_prob = logits_stable - torch.log(denominator)

        pos_counts = pos_mask_bin.sum(1)
        pos_counts = torch.where(pos_counts < 1e-6, torch.ones_like(pos_counts), pos_counts)
        mean_log_prob_pos = (pos_mask_bin * log_prob).sum(1) / pos_counts

        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


# =========================
# MeanNegArcCos_Loss
# =========================
class MeanNegArcCos_Loss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, margin=0.3):
        super(MeanNegArcCos_Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.margin = margin
        self.azi_size = 360
        self.degree_resolution = 360 / self.azi_size
        self.sigma = None
        self.p = torch.tensor([0.707106781])
        self.contrast_count = None
        self.device = None
        self.contrast_mode = 'all'

    def forward(self, features, labels, sigma):
        self.device = (torch.device('cuda:0') if features.is_cuda else torch.device('cpu'))
        B = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != B:
            raise ValueError('Num of labels does not match num of features')

        # binary positives
        mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast = F.normalize(features, dim=1)
        anchor = contrast

        cos_theta = torch.matmul(anchor, contrast.T).clamp(-1 + 1e-7, 1 - 1e-7)  # (B,B)
        sin_theta = torch.sqrt(1.0 - cos_theta**2)
        m = torch.tensor(self.margin, device=features.device, dtype=features.dtype)
        cos_m = torch.cos(m)
        sin_m = torch.sin(m)

        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m                    # ArcFace for positives

        # we use -cos(*) to keep the "cos(pi-theta)" style from your other losses
        logits_pos = (-cos_theta_m) / self.temperature
        logits_neg = (-cos_theta) / self.temperature

        # masks
        logits_mask = torch.ones_like(mask, device=self.device) - torch.eye(B, dtype=torch.float32, device=self.device)
        pos_mask_bin = mask * logits_mask
        neg_mask = logits_mask - pos_mask_bin

        # stabilization over both pos/neg
        logits_combined = pos_mask_bin * logits_pos + neg_mask * logits_neg
        logits_max, _ = torch.max(logits_combined, dim=1, keepdim=True)
        logits_stable = logits_combined - logits_max.detach()

        exp_logits = torch.exp(logits_stable)
        exp_pos = exp_logits * pos_mask_bin
        exp_neg = exp_logits * neg_mask

        neg_counts = neg_mask.sum(1, keepdim=True)
        mean_exp_neg = exp_neg.sum(1, keepdim=True) / (neg_counts + 1e-8)

        denominator = exp_pos.sum(1, keepdim=True) + mean_exp_neg
        log_prob = logits_stable - torch.log(denominator)

        pos_counts = pos_mask_bin.sum(1)
        pos_counts = torch.where(pos_counts < 1e-6, torch.ones_like(pos_counts), pos_counts)
        mean_log_prob_pos = (pos_mask_bin * log_prob).sum(1) / pos_counts

        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()

