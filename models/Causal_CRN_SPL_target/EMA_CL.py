from .FFT import ConvSTFT 
from .input_process import Processing
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

        self.activation=nn.ELU()
        

    def forward(self, x):
        original_frame_num=x.shape[-1]           
        
        x=self.conv2d(x)
        x=self.norm(x)
        x=self.activation(x)   
        
        x=x[...,:original_frame_num] 
        
        return x

class Conv1D_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv1D_Block, self).__init__()
        
        self.conv1d=nn.Conv1d(*args, **kwargs)
        
        self.norm=nn.BatchNorm1d(args[1])
        
        self.activation=nn.ELU()


    def forward(self, x):
        
        x=self.conv1d(x)
        x=self.norm(x)
        x=self.activation(x)       

        return x


class CRN(nn.Module):
    def __init__(self, config):
        super(CRN, self).__init__()

        self.config=config

        
        self.cnn_num=self.config['CNN']['layer_num']
        self.kernel_size=self.config['CNN']['kernel_size']       # 3
        self.filter_size=self.config['CNN']['filter']            # 64

        self.max_pool_kernel=self.config['CNN']['max_pool']['kernel_size']
        self.max_pool_stride=self.config['CNN']['max_pool']['stride']
        
        args = [self.config['input_cnn_channel'],  self.filter_size,   self.kernel_size]     # in_channel, out_channel, kernel size
       
        kwargs = {'stride': 1, 'padding': [1, 2], 'dilation': 1}


        ##############################
        # CNN layer
        ##############################
        self.cnn=nn.ModuleList()
        self.pooling=nn.ModuleList()
        self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))       # (2*2*(C-1), 32, 3)
        self.pooling.append(nn.MaxPool2d(kernel_size=self.max_pool_kernel, stride=self.max_pool_stride))  # (2, 2)
        
        args[0]=self.config['CNN']['filter']                             # (64, 32, 3)   in_channel 변경
        for count in range(self.cnn_num-1):
            self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))   
            self.pooling.append(nn.MaxPool2d(kernel_size=self.max_pool_kernel, stride=self.max_pool_stride))  # (2, 2)
    

        ##############################
        # projection layer
        ##############################

        self.GRU_layer=nn.ModuleList()
        self.h0_layer=[]
        self.GRU_layer.append(nn.GRU(**config['GRU']))
        self.GRU_layer.append(nn.GRU(**config['GRU']))
        self.GRU_layer.append(nn.GRU(**config['GRU']))

        
        self.h0_layer.append(torch.nn.parameter.Parameter(torch.zeros(*config['GRU_init']['shape']), requires_grad=config['GRU_init']['learnable']))
        self.h0_layer.append(torch.nn.parameter.Parameter(torch.zeros(*config['GRU_init']['shape']), requires_grad=config['GRU_init']['learnable']))
        self.h0_layer.append(torch.nn.parameter.Parameter(torch.zeros(*config['GRU_init']['shape']), requires_grad=config['GRU_init']['learnable']))
        

        self.azi_mapping_conv_layer=nn.ModuleList()
        self.azi_mapping_final=nn.ModuleList()


        args = [256, 256, 1]
        kwargs['padding']=0

        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))       
        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))       
        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))       

        args[1] = 128
        self.azi_mapping_final.append(nn.Conv1d(*args, **kwargs))       
        self.azi_mapping_final.append(nn.Conv1d(*args, **kwargs))       
        self.azi_mapping_final.append(nn.Conv1d(*args, **kwargs))
        

    def forward(self, x):
        
        ##############################
        # CNN layer
        ##############################
        # x: (B, 6, 129, n) -> (B, 64, 64, n) -> (B, 64, 32, n) -> (B, 64, 16, n) -> (B, 64, 8, n)
        for cnn_layer, pooling_layer in zip(self.cnn, self.pooling):
            x=cnn_layer(x)
            x=pooling_layer(x)


        b, c, f, t=x.shape              # (B, 64, 8, n)
        x=x.view(b, -1, t)                  # (B, 512, n)
        embedding = F.normalize(x, dim=1)   # (B, 512, n)

        ##############################
        # projection layer
        ##############################

        outputs=[]

        for gru_layer, h0_layer, cnn_layer, final_layer in zip(self.GRU_layer, self.h0_layer, self.azi_mapping_conv_layer, self.azi_mapping_final):
            
            h0 = h0_layer.repeat_interleave(embedding.shape[0], dim=1).to(embedding.device)  # h0 : (num_layers, B, hidden_size) = (3, 512, 256)
            gru_layer.flatten_parameters()


            out, h =gru_layer(embedding.permute(0, 2, 1), h0)  # (B, n, 256)
            out = out.permute(0, 2, 1)  # (B, 256, n) 
            out = cnn_layer(out)
            out = final_layer(out)

            out = F.normalize(out, dim=1)  # (B, 128, n)
            outputs.append(out)
        
        return outputs, embedding



class main_model_for_scl(nn.Module):
    def __init__(self, config, doa=False):
        super(main_model_for_scl, self).__init__()
        
        self.config=config
        self.sigma=torch.tensor(self.config['sigma_start'])
        self.degree_resolution = self.config['degree_resolution']
        self.azi_size=360//self.degree_resolution

        
        self.stft_model=ConvSTFT(**self.config['FFT'])
        self.data_proc=Processing(self.stft_model)

        self.encoder_s = CRN(self.config['CRN'])
        self.encoder_t = CRN(self.config['CRN'])

        self.doa = doa
        if not doa:
            for param_s, param_t in zip(self.encoder_s.parameters(), self.encoder_t.parameters()):
                param_t.data.copy_(param_s.data)
                param_t.requires_grad = False

        self.m = self.config['momentum'] 

        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_s, param_t in zip(
            self.encoder_s.parameters(), self.encoder_t.parameters()
        ):
            param_t.data = param_t.data * self.m + param_s.data * (1.0 - self.m)

        
    def forward(self, mixed_s, mixed_t, vad, azi_list):

        mixed_s, mixed_t, vad, azi_list = self.data_proc.permute_data(mixed_s, mixed_t, vad, azi_list)
        block_stft_s, block_vad_frame = self.data_proc.make_block(mixed_s, vad)
        block_stft_t, block_vad_frame = self.data_proc.make_block(mixed_t, vad)
        ibRTF_s, vad_block = self.data_proc.ib_RTF(block_stft_s, block_vad_frame)      # (B, 2(C-1), F, n), (B, n)
        ibRTF_t, vad_block = self.data_proc.ib_RTF(block_stft_t, block_vad_frame)      # (B, 2(C-1), F, n), (B, n)

        outputs_s, embedding_s = self.encoder_s(ibRTF_s)    # (B, 128, n), (B, 256, n)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            outputs_t, embedding_t = self.encoder_t(ibRTF_t)    # (B, 128, n), (B, 256, n)


        outputs = [outputs_s, outputs_t]
        embedding = [embedding_s, embedding_t]
        
        return outputs, embedding, azi_list, vad_block
    

