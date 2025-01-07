from .FFT import ConvSTFT 
from torch import nn
import torch
from util import *
import numpy as np
import math
import librosa
import torchaudio.transforms as T
from .CL_CRN import main_model_for_scl
import importlib


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
        
        args[0]=config['CNN']['filter']                             # (64, 32, 3)   
        for count in range(self.cnn_num-1):
            self.cnn.append(Conv1D_Block(*args, **kwargs))   
            self.pooling.append(nn.MaxPool1d(kernel_size=self.max_pool_kernel, stride=self.max_pool_stride))  # (2, 2)
    

        ##############################
        # output layer
        ##############################
        self.azi_mapping_conv_layer=nn.ModuleList()
        self.azi_mapping_final=nn.ModuleList()
        
        # args = [1024, 512, 1]
        args = [512, 512, 1]
        kwargs['padding']=0
        
        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))       # (1024, 512, 1)
        args[0]=512
        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))       # (512, 512, 1)
        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))       # (512, 512, 1)
        
        args[1] = 360
        self.azi_mapping_final.append(nn.Conv1d(*args, **kwargs))       # (512, 360, 1)
        self.azi_mapping_final.append(nn.Conv1d(*args, **kwargs))       # (512, 360, 1)
        self.azi_mapping_final.append(nn.Conv1d(*args, **kwargs))       # (512, 360, 1)


    def forward(self, x):
        # x : (B, 1, 256)
        ##############################
        # CNN layer
        ##############################
        for cnn_layer, pooling_layer in zip(self.cnn, self.pooling):
            x=cnn_layer(x)[...,:x.shape[-1]]
            x=pooling_layer(x)
 
        # x : (B, 64, 16)
        x = x.reshape(x.shape[0], -1, 1)   # (B, 1024, 1)
        
        
        ##############################
        # output layer
        ##############################
        outputs=[]

        for cnn_layer, final_layer in zip(self.azi_mapping_conv_layer, self.azi_mapping_final):
            x=cnn_layer(x)
            res_output=final_layer(x)
            outputs.append(res_output)
        output=torch.stack(outputs).permute(1,0,2,3)    # (B, 3, 360, 1)
        
        return output.squeeze(dim=-1)



class main_model_for_doa(nn.Module):
    def __init__(self, config, config_scl=None, hyparam=None):
        super(main_model_for_doa, self).__init__()
        self.config=config
        self.use_scl = self.config['SCL']
        self.config_scl=config_scl
        self.hyparam=hyparam
        
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
        
        if self.use_scl:
            self.trained_scl_model_path = self.hyparam['trained_scl_model_path']
            self.model_select_for_scl_feature()     # self.scl_model
       
    

    def model_select_for_scl_feature(self):
        model_name=self.config_scl['name']
        model_import='models.'+model_name+'.main'

        model_dir=importlib.import_module(model_import)
        
        self.scl_model=model_dir.get_model_for_scl(self.config_scl)#.to(self.device)

        trained=torch.load(self.trained_scl_model_path)#, map_location=self.device)     
        self.scl_model.load_state_dict(trained['model_state_dict'], )                       
        self.scl_model=torch.nn.DataParallel(self.scl_model, self.hyparam['GPGPU']['device_ids'])       
        self.scl_model.eval()
        

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
        # comp = comp[:, :, :26, :]
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
    
    
    def _get_mel_spectrogram(self, linear_spectra):
        
        self.nb_mel_bins = 64
        self.n_fft =256
        self.mel_wts = torch.tensor(librosa.filters.mel(sr=16000, n_fft=self.n_fft, n_mels=self.nb_mel_bins).T, device='cuda', dtype=torch.float32)
        
        
        B, T, F, C = linear_spectra.shape
        mel_feat = torch.zeros((B, T, self.nb_mel_bins, C), device='cuda')

        for ch in range(C):
            mag_spectra = torch.abs(linear_spectra[:, :, :, ch]) ** 2  # (B, T, F)
            mel_spectra = torch.matmul(mag_spectra, self.mel_wts)  # (B, T, nb_mel_bins)
            log_mel_spectra = torch.log10(torch.clamp(mel_spectra, min=1e-10)) * 10  # librosa power_to_db equivalent
            mel_feat[:, :, :, ch] = log_mel_spectra     # (B, T, nb_mel_bins, C)

        mel_feat = mel_feat.permute(0, 3, 2, 1)         # (B, C, nb_mel_bins, T)
        return mel_feat     
    
    
    def _get_gcc(self, linear_spectra):     # T x F x C       
        
        
        B, T, F, C = linear_spectra.shape
        gcc_channels = (C * (C - 1)) // 2
        gcc_feat = torch.zeros((B, T, self.nb_mel_bins, gcc_channels), device='cuda')

        cnt = 0
        for m in range(C):
            for n in range(m + 1, C):
                R = torch.conj(linear_spectra[:, :, :, m]) * linear_spectra[:, :, :, n]  # (B, T, F)
                cc = torch.fft.irfft(torch.exp(1j * torch.angle(R)), n=self.n_fft)  # (B, T, n_fft)
                cc = torch.cat([cc[:, :, -self.nb_mel_bins // 2:], cc[:, :, :self.nb_mel_bins // 2]], dim=-1)  # Wrap GCC
                gcc_feat[:, :, :, cnt] = cc         # (B, T, nb_mel_bins, 6)
                cnt += 1

        gcc_feat = gcc_feat.permute(0, 3, 2, 1)     # (B, 6, nb_mel_bins, T)
        return gcc_feat     
    
    def compressed_RTF(self, mixed, vad):
        
        r, i, vad_frame =self.stft_model(mixed, vad, cplx=True)     # vad_frame : (B, 1, 501)
        stft = torch.complex(r, i)  # B x C x F x T
        
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

        
    def forward(self, mixed, vad, azi, iter_num, epoch, LOCATA=False):

        if self.use_scl:
            with torch.no_grad():
                _, feature, vad_frame = self.scl_model(mixed, vad)         # (B, 256)
                feature = feature.unsqueeze(dim=1)                # (B, 1, 256)
        else:  
            # r, i, vad_frame =self.stft_model(mixed, vad, cplx=True)
            # comp = torch.complex(r, i)  # B x C x F x T
            # linear_spectra = comp.permute(0, 3, 2, 1)   # B x T x F x C
            
            # mel_spect = self._get_mel_spectrogram(linear_spectra)
            # gcc = self._get_gcc(linear_spectra)
            
            # feature = torch.cat((mel_spect, gcc), dim=1)       # (B, 10, 64, 501)
            
            feature, vad_frame = self.compressed_RTF(mixed, vad)
            
        
        out=self.crn(feature)   # (B, 3, 360)
        
        if LOCATA:
            target=self.stft_model.azimuth_strided(vad_frame, azi).unsqueeze(0)
            azi=azi[...,0]
      
            vad_target_pic=self.make_target( vad_frame, azi, iter_num, epoch)
           

            return out, target, vad_frame,vad_target_pic
            
        else:
            target=self.make_target( vad_frame, azi, iter_num, epoch)
            target, _ = torch.max(target, dim=3)   # (B, 3, 360)
        
        
        return out, target


