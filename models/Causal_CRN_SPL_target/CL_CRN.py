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
        self.projection = nn.Linear(config['embedding_size'], config['projection_size'])  # (256, 128)
        

    def forward(self, x):
        
        ##############################
        # CNN layer
        ##############################
        # x: (B, 6, 129)
        for cnn_layer, pooling_layer in zip(self.cnn, self.pooling):
            x=cnn_layer(x)
            x=pooling_layer(x)
        # x: (B, 32, 8)
        x_flatten = x.view(x.size(0), -1)  # (B, 256)
        embedding = F.normalize(x_flatten, dim=1)
        
        ##############################
        # projection layer
        ##############################
        x = self.projection(embedding)  # (B, 128)
        x = F.normalize(x, dim=1)
        
        return x, embedding



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
    


    def sigma_update(self, iter_num, epoch):
        if iter_num%500==0:
                # print(self.sigma)
                pass
        if epoch<self.config['wait_epoch']:
            return
        def update():
            

            if self.sigma_update_method=='add':
                self.sigma+=self.sigma_rate
            elif self.sigma_update_method=='multiply':
                self.sigma*=self.sigma_rate
            else:
                "Not exist!!!"
                exit()

            self.sigma=torch.clamp(self.sigma, self.sigma_min, self.sigma_max)

       
        if self.training:

            if self.config['iter']['update']:
                if self.iteration_count!=self.config['iter']['update_period']:
                    self.iteration_count+=1
                
                else:
                    print('sigma_iter update')
                    update()
                    self.iteration_count=0
                    return
            
            if self.config['epoch']['update']:

                if self.now_epoch!=epoch:
                    self.now_epoch=epoch
                    self.epoch_count+=1
                
                if self.epoch_count==self.config['epoch']['update_period']:
                    print('sigma_epoch update')
                    update()
                    self.epoch_count=0
                    return 


    def irtf_feature(self, mixed, vad):  
        mixed_max = mixed.max()
        mixed_min = mixed.min()
        mixed = (mixed - mixed_min) / (mixed_max - mixed_min) * 2 - 1
        
        r, i, vad_frame =self.stft_model(mixed, vad, cplx=True)
        # B x C x F x T = (B, 4, 129, 501)
        comp = torch.complex(r, i)
        comp = comp[:, :, :26, :]
        
        comp_ref = comp[..., [self.ref_ch], :, :]
        comp_ref = torch.complex(
            comp_ref.real.clamp(min=1e-2), comp_ref.imag.clamp(min=1e-2)
        )


        comp=torch.cat(
        (comp[..., self.ref_ch-1:self.ref_ch, :, :], comp[..., self.ref_ch+1:, :, :]),
        dim=-3) / (comp_ref + 1e-2)

        feature=torch.cat((comp.real, comp.imag), dim=1)    # (B, 6, 129, 501)
        
        
        # (B, 2*(C-1), F, T), (B, F, T)
        # (B, 6, 129, 501)
        return feature, vad_frame
    
    
    def _get_gcc(self, mixed, vad):     # T x F x C
        self.device = mixed.device
        
        r, i, vad_frame =self.stft_model(mixed, vad, cplx=True)
        comp = torch.complex(r, i)  # B x C x F x T
        # comp = comp[:, :, :33, :]
        
        linear_spectra = comp.permute(0, 3, 2, 1)   # B x T x F x C = (B, 501, 129, 4)
        
        # self._nb_mel_bins = 64  
        
        gcc_feat = []
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = torch.conj(linear_spectra[:, :, :, m]) * linear_spectra[:, :, :, n]        
                cc = torch.fft.irfft(torch.exp(1.j*torch.angle(R)), dim=-1)      # (B, T, 2*(F-1)) (B, 501, 256)
                # cc.shape (B, 345, 256)
                gcc_feat.append(cc)
        
        gcc_feat = torch.stack(gcc_feat, dim=-1)      # (B, 501, 256, 6)
        gcc_feat = gcc_feat.permute(0, 3, 2, 1)         # (B, 6, 256, 501)

        return gcc_feat, vad_frame
    
    
    def compressed_RTF(self, mixed, vad):
        
        r, i, vad_frame =self.stft_model(mixed, vad, cplx=True)     # vad_frame : (B, 1, 501)
        stft = torch.complex(r, i)  # B x C x F x T
        
        num_select_time = 31
        
        time_indices = [len(torch.nonzero(vad_frame[b, 0] == 1, as_tuple=True)[0]) for b in range(vad_frame.shape[0])]
        time_indices = [ idx if idx > 0 else 1 for idx in time_indices ]
        time_len = torch.tensor(time_indices, device=stft.device)
    
        linear_spectra = stft.permute(0, 2, 3, 1)    # (B, F, T, C)
        
        cov_z = torch.einsum('bftc,bftd->bfcd', linear_spectra, linear_spectra.conj()) / time_len[:, None, None, None]    # (B, F, C, C)

        col0 = cov_z[:, :, :, self.ref_ch]                                # (B, F, C)        
        col00 = col0[:, :, self.ref_ch]                                   # (B, F)
        
        col0 = torch.cat((col0[:,:,self.ref_ch-1:self.ref_ch], col0[:,:,self.ref_ch+1:]), dim=-1)    # (B, F, C-1)
        col00 = torch.complex(col00.real.clamp(self.eps), col00.imag.clamp(self.eps))
        
        c_rtf = col0 / col00[:, :, None]    # (B, F, C-1)
        c_rtf = torch.cat((c_rtf.real, c_rtf.imag), dim=-1)    # (B, F, 2(C-1))

        c_rtf = c_rtf.permute(0, 2, 1)    # (B, 2(C-1), F)

        return c_rtf, vad_frame


        
    def forward(self, mixed, vad):
        
        feature, vad_frame = self.compressed_RTF(mixed, vad)    # (B, 2(C-1), F), (B, 1, T)

        out, embedding = self.crn(feature)   # (B, 128), (B, 256)
        
        
        return out, embedding, vad_frame


