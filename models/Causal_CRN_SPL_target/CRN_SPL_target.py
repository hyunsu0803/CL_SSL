from .FFT import ConvSTFT 
from torch import nn
import torch
from util import *
import numpy as np
import math
import librosa
import torchaudio.transforms as T

    
    
class NonCausal_Conv2D_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NonCausal_Conv2D_Block, self).__init__()
        
        self.conv2d = nn.Conv2d(*args, **kwargs)

        self.norm = nn.BatchNorm2d(args[1])     # args[1] : [3, 3]

        self.activation = nn.ELU()

    def forward(self, x):
        original_frame_num = x.shape[-1]  # time 축 크기 저장
        
        x = self.conv2d(x)
        x = self.norm(x)
        x = self.activation(x)   
        
        x = x[..., :original_frame_num]  # 원래 time 축 크기로 자르기
        
        return x


class Causal_Conv2D_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Causal_Conv2D_Block, self).__init__()
        
        self.conv2d=nn.Conv2d(*args, **kwargs)

        self.norm=nn.BatchNorm2d(args[1])

        self.activation=nn.ELU()
        

    def forward(self, x):
        original_frame_num=x.shape[-1]      # 182
        # print("11111", x.shape)             # (B, 6, 129, 182)   #      
        
        x=self.conv2d(x)
        # print("33333", x.shape)             # (B, 64, 129, 184)
        
        x=self.norm(x)
        x=self.activation(x)   
        
        x=x[...,:original_frame_num] 
        # print("44444", x.shape)             # (B, 64, 129, 182)
        
        return x

class Conv1D_Block(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv1D_Block, self).__init__()
        
        self.conv1d=nn.Conv1d(*args, **kwargs)
        
        self.norm=nn.BatchNorm1d(args[1])
        
        self.activation=nn.ELU()


    def forward(self, x):
        # print("x.shape", x.shape)   # (B, 256, 690)
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
        self.filter_size=config['CNN']['filter']        

        self.max_pool_kernel=config['CNN']['max_pool']['kernel_size']
        self.max_pool_stride=config['CNN']['max_pool']['stride']

        # args = [2*(config['input_audio_channel']-1),  self.filter_size,   self.kernel_size]     # in_channel, out_channel, kernel size
        args = [6,  self.filter_size,   self.kernel_size]     # in_channel, out_channel, kernel size
       
        # kwargs={'stride': 1, 'padding': [1,2], 'dilation': 1}
        kwargs = {'stride': 1, 'padding': (self.kernel_size[0] // 2, self.kernel_size[1] // 2), 'dilation': 1}

      


        self.cnn=nn.ModuleList()
        self.pooling=nn.ModuleList()
        self.cnn.append(NonCausal_Conv2D_Block(*args, **kwargs))       # (2*(C-1), 64, [3,3])
        self.pooling.append(nn.MaxPool2d(self.max_pool_kernel, stride=self.max_pool_stride))

        args[0]=config['CNN']['filter']                             # (64, 64, [3,3])
        for count in range(self.cnn_num-1):
            self.cnn.append(NonCausal_Conv2D_Block(*args, **kwargs))
            self.pooling.append(nn.MaxPool2d(self.max_pool_kernel, stride=self.max_pool_stride))
    
        self.GRU_layer=nn.GRU(**config['GRU'])                      # bidirectional=True      
        # self.h0=torch.zeros(*config['GRU_init']['shape'])
        
        # Bidirectional GRU를 위한 h0 차원 수정
        num_layers = config['GRU_init']['shape'][0]  # 3
        hidden_size = config['GRU_init']['shape'][-1]  # 256

        # bidirectional을 고려하여 h0의 shape 변경
        self.h0 = torch.zeros(num_layers * 2, 1, hidden_size)  # [6, 1, 256]
        self.h0=torch.nn.parameter.Parameter(self.h0, requires_grad=config['GRU_init']['learnable'])
        

        self.azi_mapping_conv_layer=nn.ModuleList()
        self.azi_mapping_final=nn.ModuleList()

        # (B, 512, 690) => (B, 360)
        # args[0] = 690  # input channel
        # args[1] = 32    # output channel
        # args[2] = (3, 3)          # kernel size      
        # kwargs['padding'] = (0, 0)
        # self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))
        # self.azi_mapping_final.append(nn.Conv2d(args[1], self.azi_size, (1, 1)))
        
        # args[0]=32     #   input channel
        # args[1]=32     #   output channel
        # for _ in range(output_num-1):
        #     self.azi_mapping_conv_layer.append(Conv2D_Block(*args, **kwargs))
        #     self.azi_mapping_final.append(nn.Conv2d(args[1], self.azi_size, (1, 1)))
        
        kwargs['padding'] = 0
        args[0] = 690
        args[1] = 64
        args[2] = 1
        # args1 = [690, 64, 1]    # in_channel, out_channel, kernel size
        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))
        args[0] = 64
        args[1] = 1
        args[2] = 1
        # args2 = [64, 1, 1]
        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))
        self.azi_mapping_final.append(nn.Conv1d(in_channels=512, out_channels=self.azi_size, kernel_size=1, padding=0))
        
        args[0] = 512
        args[1] = 512
        args[2] = 1
        # args3 = [512, 512, 1]
        for _ in range(output_num-1):
            self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))
            self.azi_mapping_final.append(nn.Conv1d(in_channels=512, out_channels=self.azi_size, kernel_size=1, padding=0))
     
     

    def forward(self, x):
      
        for cnn_layer, pooling_layer in zip(self.cnn, self.pooling):
            x=cnn_layer(x)[...,:x.shape[-1]]
            x=pooling_layer(x)
 
        
        b, c, f, t=x.shape              # (B, 64, 3, 690)
        x=x.view(b, -1, t).permute(0,2,1)

        h0 = self.h0.repeat_interleave(x.shape[0], dim=1)  # h0 : (2*num_layers, B, hidden_size)
        
        self.GRU_layer.flatten_parameters()
        
        x, h=self.GRU_layer(x, h0)      # (B, t, 256(hidden size))
        # x=x.permute(0,2,1)              # (B, 512, 690)
        outputs=[]
        
        
        
        ### time axis compression
        #                        x.shape: (B, 690, 512)
        cnn_layer=self.azi_mapping_conv_layer[0]
        x=cnn_layer(x)                  # (B, 64, 512)
        cnn_layer=self.azi_mapping_conv_layer[1]
        x=cnn_layer(x)                  # (B, 1, 512)
        
        
        x=x.permute(0,2,1)              # (B, 512, 1)
        
        ### channel axis compression
        final_layer=self.azi_mapping_final[0]
        res_output=final_layer(x)       # (B, 360, 1)
        outputs.append(res_output.squeeze(-1))
        
        for cnn_layer, final_layer in zip(self.azi_mapping_conv_layer[2:], self.azi_mapping_final[1:]):
            x=cnn_layer(x)              # (B, 512, 1)           
            res_output=final_layer(x)   # (B, 360, 1)
            outputs.append(res_output.squeeze(-1))
            
        output=torch.stack(outputs).permute(1,0,2)  # (3, B, 360) -> (B, 3, 360)
        
        return output



class main_model(nn.Module):
    def __init__(self, config):
        super(main_model, self).__init__()
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

    
    def make_target(self, vad_frame, azi, iter_num, epoch):
        
        azi_target=torch.div(azi, 360//self.azi_size, rounding_mode='floor').long()        
        azi_range=torch.arange(0, self.azi_size).unsqueeze(0).to(azi_target.device)

        distance=azi_target.unsqueeze(-1)*self.degree_resolution-azi_range*self.degree_resolution
        
        distance_abs=torch.abs(distance)
        distance_abs=torch.stack((distance_abs, 360-distance_abs), dim=0)
     
        distance=torch.min(distance_abs, dim=0).values
        distance=torch.deg2rad(distance).unsqueeze(1)
        
        sigma=self.sigma.view(1,-1, 1,1).to(distance.device)
        sigma=torch.deg2rad(sigma)
        kappa_d=torch.log(self.p)/(torch.cos(sigma)-1)
        

        labelling=torch.exp(kappa_d*(torch.cos(distance)-1)).unsqueeze(-1) # batch, number of sigma, number of speakers, time, 1  
        

        vad_frame=vad_frame.unsqueeze(1).unsqueeze(-2)
        
        vad_frame=labelling*vad_frame
    
        vad_frame=torch.max(vad_frame, dim=2).values
       
        self.sigma_update(iter_num, epoch)
       
        return vad_frame # batch, sigma_num, degree, frame
    

    def irtf_feature(self, mixed, vad):  
        r, i, vad_frame =self.stft_model(mixed, vad, cplx=True)
        # B x C x F x T = (B, 4, 513, 345)
        comp = torch.complex(r, i)
        comp = comp[:, :, :26, :]
        # print("comp.shape", comp.shape)
        # exit()
        
        comp_ref = comp[..., [self.ref_ch], :, :]
        comp_ref = torch.complex(
            comp_ref.real.clamp(self.eps), comp_ref.imag.clamp(self.eps)
        )

        comp=torch.cat(
        (comp[..., self.ref_ch-1:self.ref_ch, :, :], comp[..., self.ref_ch+1:, :, :]),
        dim=-3) / comp_ref
        feature=torch.cat((comp.real, comp.imag), dim=1)
        
        
        # (B, 2*(C-1), F, T), (B, F, T)
        # (B, 6, 129, 501)
        return feature, vad_frame
    
    
    def _get_gcc(self, mixed, vad):     # T x F x C
        self.device = mixed.device
        
        r, i, vad_frame =self.stft_model(mixed, vad, cplx=True)
        comp = torch.complex(r, i)  # B x C x F x T
        comp = comp[:, :, :26, :]
        
        linear_spectra = comp.permute(0, 3, 2, 1)   # B x T x F x C
        
        self._nb_mel_bins = 64  
        
        gcc_feat = []
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = torch.conj(linear_spectra[:, :, :, m]) * linear_spectra[:, :, :, n]        
                cc = torch.fft.irfft(torch.exp(1.j*torch.angle(R)), dim=-1)      # (B, T, 2*(F-1)) (B, 345, 1024)
                # cc.shape (B, 345, 64)
                gcc_feat.append(cc)
        
        gcc_feat = torch.stack(gcc_feat, dim=-1)      # (B, 690, 50, 6)
        gcc_feat = gcc_feat.permute(0, 3, 2, 1)         # (B, 6, 50, 690)

        return gcc_feat, vad_frame


    def vad_framing(self, vad_batch):

        vad_output_th = vad_batch.mean(axis=2) > 2 / 3
        
        vad_output_th = vad_output_th[:, np.newaxis, :, np.newaxis, np.newaxis]
        vad_output_th = torch.from_numpy(vad_output_th.astype(float)).to(maps.device)
        repeat_factor = np.array(maps.shape)
        repeat_factor[:-2] = 1
        maps *= vad_output_th.float().repeat(repeat_factor.tolist())

    
    def target_flip(self, target):

  

        target_flipped=torch.flip(target, dims=[2])
        target_flipped=torch.roll(target_flipped, dims=2, shifts=1)
        target_cat=torch.stack([target_flipped, target], dim=0)
        target=torch.max(target_cat, dim=0).values

 
        return target

        
    def forward(self, mixed, vad, azi, iter_num, epoch, mic_type, LOCATA=False):
        ###### irtf feature extraction  (B, 6, 50, 690), (B, 1, 690)
        # feature, vad_frame=self.irtf_feature(mixed, vad)
        
        ###### gcc feature extraction  
        feature, vad_frame=self._get_gcc(mixed, vad)
       
        # model forward
        out=self.crn(feature)   # (B, 3, 360)
        
        if LOCATA:
            target=self.stft_model.azimuth_strided(vad_frame, azi).unsqueeze(0)
            azi=azi[...,0]
      
            vad_target_pic=self.make_target( vad_frame, azi, iter_num, epoch)
           

            return out, target, vad_frame,vad_target_pic
            
        else:
            target=self.make_target( vad_frame, azi, iter_num, epoch)
            target, _ = torch.max(target, dim=3)   # (B, 3, 360)
        
        # if mic_type=='linear':
        #     target=self.target_flip(target)
        
        return out, target, vad_frame


