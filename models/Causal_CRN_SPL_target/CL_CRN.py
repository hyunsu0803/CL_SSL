from .FFT import ConvSTFT 
from torch import nn
import torch.nn.functional as F
import torch
from util import *
import numpy as np
import math
import librosa
import torchaudio.transforms as T


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


class crn(nn.Module):
    def __init__(self, config, output_num, azi_size):
        super(crn, self).__init__()

        
        self.output_num=output_num  
        self.azi_size=azi_size

        
        self.cnn_num=config['CNN']['layer_num']
        self.kernel_size=config['CNN']['kernel_size']       # 3
        self.filter_size=config['CNN']['filter']            # 32

        self.max_pool_kernel=config['CNN']['max_pool']['kernel_size']
        self.max_pool_stride=config['CNN']['max_pool']['stride']
        
        args = [config['input_cnn_channel'],  self.filter_size,   self.kernel_size]     # in_channel, out_channel, kernel size
       
        kwargs = {'stride': 1, 'padding': self.kernel_size // 2, 'dilation': 1}


        ##############################
        # CNN layer
        ##############################
        self.cnn=nn.ModuleList()
        self.pooling=nn.ModuleList()
        self.cnn.append(Conv1D_Block(*args, **kwargs))       # (2*2*(C-1), 32, 3)
        self.pooling.append(nn.MaxPool1d(kernel_size=self.max_pool_kernel, stride=self.max_pool_stride))  # (2, 2)
        
        args[0]=config['CNN']['filter']                             # (64, 32, 3)   in_channel 변경
        for count in range(self.cnn_num-1):
            self.cnn.append(Conv1D_Block(*args, **kwargs))   
            self.pooling.append(nn.MaxPool1d(kernel_size=self.max_pool_kernel, stride=self.max_pool_stride))  # (2, 2)
    
        ##############################
        # projection layer
        ##############################
        # self.projection = nn.Linear(config['embedding_size'], config['projection_size'])  # (256, 128)

        self.embedding_layer=nn.ModuleList()
        self.azi_mapping_conv_layer=nn.ModuleList()
        self.azi_mapping_final=nn.ModuleList()


        args = [256, 256, 1]
        kwargs['padding']=0

        self.embedding_layer.append(Conv1D_Block(*args, **kwargs))
        self.embedding_layer.append(Conv1D_Block(*args, **kwargs))
        self.embedding_layer.append(Conv1D_Block(*args, **kwargs))

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
        # x: (B, 6, 129)
        for cnn_layer, pooling_layer in zip(self.cnn, self.pooling):
            x=cnn_layer(x)
            x=pooling_layer(x)
        # x: (B, 32, 8)
        x_flatten = x.view(x.size(0), -1, 1)  # (B, 256)
        embedding = F.normalize(x_flatten, dim=1)
        
        ##############################
        # projection layer
        ##############################
        # x = self.projection(embedding)  # (B, 128)
        # x = F.normalize(x, dim=1)

        outputs=[]

        for cnn_layer, final_layer in zip(self.azi_mapping_conv_layer, self.azi_mapping_final):
            out = cnn_layer(embedding)
            out = final_layer(out)

            out = out.squeeze(dim=-1)
            out = F.normalize(out, dim=-1)  # (B, 128)
            outputs.append(out)
        
        return outputs, embedding



class main_model_for_scl(nn.Module):
    def __init__(self, config):
        super(main_model_for_scl, self).__init__()
        self.config=config
        
        self.eps=np.finfo(np.float32).eps
        self.ref_ch=self.config['ref_ch']

        ###### sigma

        self.p=torch.tensor(self.config['p'])
        self.sigma=torch.tensor(self.config['sigma_start'])
        self.sigma_max=torch.tensor(self.config['sigma_end']['max'])
        self.sigma_min=torch.tensor(self.config['sigma_end']['min'])
        self.sigma_rate=torch.tensor(self.config['sigma_rate'])
        self.sigma_update_method=self.config['sigma_update_method']
        
        self.iteration_count=0        
        self.epoch_count=0
        self.now_epoch=0

       
        ######
       
        # self.max_spk=self.config['max_spk']
        self.degree_resolution = self.config['degree_resolution']
        self.azi_size=360//self.degree_resolution

        self.stft_model=ConvSTFT(**self.config['FFT'])
        self.crn=crn(self.config['CRN'], self.sigma.shape[0], self.azi_size)
    
    
    def pseudo_RTF(self, stft, vad_frame):
        
        time_indices = [len(torch.nonzero(vad_frame[b, 0] == 1, as_tuple=True)[0]) for b in range(vad_frame.shape[0])]
        time_indices = [ idx if idx > 0 else 1 for idx in time_indices ]
        time_len = torch.tensor(time_indices, device=stft.device)
    
        linear_spectra = stft.permute(0, 2, 3, 1)    # (B, F, T, C)
        
        cov_z = torch.einsum('bftc,bftd->bfcd', linear_spectra, linear_spectra.conj()) / time_len[:, None, None, None]    # (B, F, C, C)

        col0 = cov_z[:, :, :, self.ref_ch]                                # (B, F, C)        
        col00 = col0[:, :, self.ref_ch]                                   # (B, F)
        
        col0 = torch.cat((col0[:,:,self.ref_ch-1:self.ref_ch], col0[:,:,self.ref_ch+1:]), dim=-1)    # (B, F, C-1)
        col00 = torch.complex(col00.real.clamp(self.eps), col00.imag.clamp(self.eps))
        
        pRTF = col0 / col00[:, :, None]    # (B, F, C-1)
        pRTF = torch.cat((pRTF.real, pRTF.imag), dim=-1)    # (B, F, 2(C-1))

        pRTF = pRTF.permute(0, 2, 1)    # (B, 2(C-1), F)

        return pRTF


        
    def forward(self, mixed=None, vad=None, stft=None, vad_frame=None):

        if mixed is not None and vad is not None:   # for SCL training
            r, i, vad_frame =self.stft_model(mixed, vad, cplx=True)
            stft = torch.complex(r, i)

        pRTF = self.pseudo_RTF(stft, vad_frame)    # (B, 2(C-1), F), (B, 1, T)


        outputs, embedding = self.crn(pRTF)   # 3 * (B, 128), (B, 256)
        
        
        return outputs, embedding.squeeze(dim=-1)