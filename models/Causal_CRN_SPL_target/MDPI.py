from .FFT import ConvSTFT 
from torch import nn
import torch.nn.functional as F
import torch
from util import *
import numpy as np



class Causal_Conv2D_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Causal_Conv2D_Block, self).__init__()
        
        self.conv2d=nn.Conv2d(*args, **kwargs)

        self.norm=nn.BatchNorm2d(args[1])

        self.activation=nn.LeakyReLU(0.01)
        

    def forward(self, x, norm=True):
        original_frame_num=x.shape[-1]           
        
        x=self.conv2d(x)
        if norm:
            x=self.norm(x)
        x=self.activation(x)   
        
        x=x[...,:original_frame_num] 
        
        return x


class crn(nn.Module):
    def __init__(self, ):
        super(crn, self).__init__()

        kwargs = {'stride': 1, 'padding': 1, 'dilation': 1}
        ##############################
        # CNN layer
        ##############################
        # input : (B, 3, C, C)
        self.cnn=nn.ModuleList()
        args = [3, 128, [3, 3]]
        self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))       
        args = [128, 64, [3, 3]]
        self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))       
        args = [64, 32, [3, 3]]
        self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))       
        args = [32, 16, [3, 3]]
        self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))       
        
        

    def forward(self, x):
        
        # x: (B, 3, C, C)
        for layer in enumerate(self.cnn[:-1]):
            x = layer(x)
        x = self.cnn[-1](x, norm=False)

        # x: (B, 16, C, C)
        output = x.view(x.size(0), x.size(1), -1)  # (B, 16, 16)
        
        return output



class main_model_for_scl(nn.Module):
    def __init__(self, config):
        super(main_model_for_scl, self).__init__()
        self.config=config
        
        self.eps=np.finfo(np.float32).eps


    def forward(self, mixed, vad):
        
        feature, vad_frame = self.compressed_RTF(mixed, vad)    # (B, 2(C-1), F), (B, 1, T)
        # feature, vad_frame=self._get_gcc(mixed, vad)
        
        
        # model forward
        out, embedding = self.crn(feature)   # (B, 128), (B, 256)
        
        
        return out, embedding, vad_frame


