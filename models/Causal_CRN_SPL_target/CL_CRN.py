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

        self.norm=nn.BatchNorm2d(args[1], eps=1e-5)
        # self.norm=nn.LayerNorm(args[1])

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
        # self.norm=nn.LayerNorm(args[1])
        
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
        self.kernel_size=config['CNN']['kernel_size']
        self.filter_size=config['CNN']['filter']            # 64

        self.max_pool_kernel=config['CNN']['max_pool']['kernel_size']   # [2,1]
        self.max_pool_stride=config['CNN']['max_pool']['stride']        # [2,1]

        args = [2*(config['input_audio_channel']-1),  self.filter_size,   self.kernel_size]     # in_channel, out_channel, kernel size
       
        kwargs = {'stride': 1, 'padding': (self.kernel_size[0] // 2, self.kernel_size[1] // 2), 'dilation': 1}

      

        ##############################
        # CNN layer
        ##############################
        self.cnn=nn.ModuleList()
        self.pooling=nn.ModuleList()
        self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))       # (2*(C-1), 64, [3,3])
        self.pooling.append(nn.MaxPool2d(self.max_pool_kernel, stride=self.max_pool_stride))

        args[0]=config['CNN']['filter']                             # (64, 64, [3,3])   in_channel 변경
        for count in range(self.cnn_num-1):
            self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))
            self.pooling.append(nn.MaxPool2d(self.max_pool_kernel, stride=self.max_pool_stride))
    
    
    
        ##############################
        # GRU layer
        ##############################
        # self.h0=torch.zeros(*config['GRU_init']['shape'])
        self.GRU_layer=nn.GRU(**config['GRU'])                      # bidirectional=True      
        
        num_layers = config['GRU_init']['shape'][0]  # 3
        hidden_size = config['GRU_init']['shape'][-1]  # 256

        # self.h0 = torch.zeros(num_layers * 2, 1, hidden_size)  # [6, 1, 256]
        self.h0 = torch.zeros(*config['GRU_init']['shape'])  # 
        self.h0=torch.nn.parameter.Parameter(self.h0, requires_grad=config['GRU_init']['learnable'])
        


        ##############################
        # projection layer
        ##############################
        self.time_comp_layer=nn.ModuleList()
        kwargs['padding']=1
        args = [501, 128, 3]  # in_channels, out_channels, kernel_size
        self.time_comp_layer.append(Conv1D_Block(*args, **kwargs))
        args = [128, 32, 3]
        self.time_comp_layer.append(Conv1D_Block(*args, **kwargs))
        

        self.channel_comp_layer=nn.ModuleList()
        args = [256, 128, 3]    
        self.channel_comp_layer.append(Conv1D_Block(*args, **kwargs))
        args = [128, 64, 3]     
        self.channel_comp_layer.append(Conv1D_Block(*args, **kwargs))
        
    
     

    def forward(self, x):
        
        ##############################
        # CNN layer
        ##############################
        # x: (B, 6, 256, 501)
        for cnn_layer, pooling_layer in zip(self.cnn, self.pooling):
            x=cnn_layer(x)[...,:x.shape[-1]]    # (B, 64, 64, 501)  (B, 64, 32, 501)    (B, 64, 6, 501)
            x=pooling_layer(x)                  # (B, 64, 32, 501)  (B, 64, 16, 501)    (B, 64, 3, 501)

        
        ##############################
        # GRU layer
        ##############################
        b, c, f, t = x.shape                  # (B, 64, 16, 501)
        x = x.view(b, -1, t).permute(0,2,1)   # (B, 501, 1024)

        h0 = self.h0.repeat_interleave(x.shape[0])  # h0 : (2*gru_num_layers, B, hidden_size) (3, B, 256)
        h0 = h0.view(self.h0.shape[0], x.shape[0], self.h0.shape[-1])  # (3, B, 256)
        self.GRU_layer.flatten_parameters()
        
        x, h=self.GRU_layer(x, h0)      # (B, 501, 256(hidden size))
        embedding = F.normalize(x.permute(0,2,1), dim=1)    # (B, 256, 501)

        
        ##############################
        # output layer
        ##############################
        for cnn_layer in self.time_comp_layer:
            x = cnn_layer(x)  # Reduce along the time axis (B, 128, 256), (B, 32, 256)
        x = x.permute(0, 2, 1)  # (B, 32, 256) -> (B, 256, 32)
        for cnn_layer in self.channel_comp_layer:
            x = cnn_layer(x) # Reduce along the channel axis (B, 128, 32), (B, 64, 32)



        x = x.view(x.size(0), -1)  # (B, 2048)
        x = F.normalize(x, dim=1)
        
        return x, embedding



class main_model_for_scl(nn.Module):
    def __init__(self, config):
        super(main_model_for_scl, self).__init__()
        self.config=config
        
        self.eps=np.finfo(np.float32).eps
        # self.eps=1e-3
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

    
    def make_weight(self, azi):
        
        azi_target=torch.div(azi, 360//self.azi_size, rounding_mode='floor').long()     # (B, 1)
        azi_range=torch.arange(0, self.azi_size).unsqueeze(0).to(azi_target.device)     # (1, 360)

        distance=azi_target.unsqueeze(-1)*self.degree_resolution-azi_range*self.degree_resolution   # (B, 1, 360) = (B, 1, 1) - (1, 360)
        
        distance_abs=torch.abs(distance)
        distance_abs=torch.stack((distance_abs, 360-distance_abs), dim=0)       # distance_abs : (2, B, 1, 360)
     
        distance=torch.min(distance_abs, dim=0).values                          # distance : (B, 1, 360)
        distance=torch.deg2rad(distance).unsqueeze(1)                           # distance : (B, 1, 1, 360)
        
        sigma=self.sigma.view(1,-1, 1,1).to(distance.device)                    # (1, 3, 1, 1)
        sigma=torch.deg2rad(sigma)
        kappa_d=torch.log(self.p)/(torch.cos(sigma)-1)                          # (1, 3, 1, 1)
        

        labelling=torch.exp(kappa_d*(torch.cos(distance)-1)).unsqueeze(-1)      # (B, 3, 1, 360, 1)  
        
        return labelling

        # self.sigma_update(iter_num, epoch)
       

    

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


    # def vad_framing(self, vad_batch):

    #     vad_output_th = vad_batch.mean(axis=2) > 2 / 3
        
    #     vad_output_th = vad_output_th[:, np.newaxis, :, np.newaxis, np.newaxis]
    #     vad_output_th = torch.from_numpy(vad_output_th.astype(float)).to(maps.device)
    #     repeat_factor = np.array(maps.shape)
    #     repeat_factor[:-2] = 1
    #     maps *= vad_output_th.float().repeat(repeat_factor.tolist())

    
    # def target_flip(self, target):

  

    #     target_flipped=torch.flip(target, dims=[2])
    #     target_flipped=torch.roll(target_flipped, dims=2, shifts=1)
    #     target_cat=torch.stack([target_flipped, target], dim=0)
    #     target=torch.max(target_cat, dim=0).values

 
    #     return target

        
    def forward(self, mixed, vad):
        ###### irtf feature extraction  (B, 6, 129, 501)
        # feature, vad_frame=self.irtf_feature(mixed, vad) 
        
        ###### gcc feature extraction  (B, 6, 256, 501)
        feature, vad_frame=self._get_gcc(mixed, vad)   
        
        # model forward
        out, embedding = self.crn(feature)   # (B, 2048)
        
        
        return out, embedding, vad_frame


