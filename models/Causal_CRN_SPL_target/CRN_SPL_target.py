from .FFT import ConvSTFT 
from torch import nn
import torch
# from util import *
import numpy as np
import librosa
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
        

        args = [config['input_mapping_dim'], 512, 1]
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
        self.config_scl=config_scl
        self.hyparam=hyparam


        self.use_scl=self.hyparam['SCL']
        self.finetune=self.hyparam['finetune']
        self.config['CRN']['input_mapping_dim'] = 1024

        
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
       
        self.degree_resolution = self.config['degree_resolution']
        self.azi_size=360//self.degree_resolution

        self.stft_model=ConvSTFT(**self.config['FFT'])
        
        self.crn=crn(self.config['CRN'], self.sigma.shape[0], self.azi_size)
        
        self.model_select_for_scl_feature()     # self.scl_model
       
    

    def model_select_for_scl_feature(self):
        model_name=self.config_scl['name']
        model_import='models.'+model_name+'.main'

        model_dir=importlib.import_module(model_import)
        
        self.scl_model=model_dir.get_model_for_scl(self.config_scl)

        if self.use_scl and not self.finetune:

            self.trained_scl_model_path = self.hyparam['trained_scl_model_path']
            trained=torch.load(self.trained_scl_model_path) 
            self.scl_model.load_state_dict(trained['model_state_dict'], )   

        self.scl_model.train()
        

    def make_target(self, vad_block, azi):

        # vad_block : (B, 1, block_num)
        # azi : (B, 1)
        
        azi_target=torch.div(azi, 360//self.azi_size, rounding_mode='floor').long()        
        azi_range=torch.arange(0, self.azi_size).unsqueeze(0).to(azi_target.device)

        ang_diff=azi_target.unsqueeze(-1)*self.degree_resolution-azi_range*self.degree_resolution
        
        distance_abs=torch.abs(ang_diff)
        distance_abs=torch.stack((distance_abs, 360-distance_abs), dim=0)
     
        ang_diff=torch.min(distance_abs, dim=0).values
        ang_diff=torch.deg2rad(ang_diff).unsqueeze(1)
        
        sigma=self.sigma.view(1,-1, 1,1).to(ang_diff.device)
        sigma=torch.deg2rad(sigma)
        kappa_d=torch.log(self.p)/(torch.cos(sigma)-1)
        

        labelling=torch.exp(kappa_d*(torch.cos(ang_diff)-1)).unsqueeze(-1) # (B, 3, num_spk, 360, 1)  
        
        # (B, 1, block_num) -> (B, 1, 1, 1, block_num)
        vad_block=vad_block.unsqueeze(1).unsqueeze(-2)
        
        target = labelling*vad_block   # (B, 3, num_spk, 360, block_num)
       
        return target.squeeze(dim=2) # (B, 3, 360, block_num)

    
    def pseudo_RTF(self, mixed=None, vad=None, stft=None, vad_frame=None):
        
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
        
        pRTF = col0 / col00[:, :, None]    # (B, F, C-1)
        pRTF = torch.cat((pRTF.real, pRTF.imag), dim=-1)    # (B, F, 2(C-1))

        pRTF = pRTF.permute(0, 2, 1)    # (B, 2(C-1), F)

        return pRTF, vad_frame

        
    def forward(self, mixed, vad, azi):

        r, i, vad_frame =self.stft_model(mixed, vad, cplx=True)
        stft = torch.complex(r, i)      # B x C x F x T
        B, C, F, T = stft.shape


        ############ split into blocks
        frame_num = stft.shape[-1]
        block_size = 25
        block_num = frame_num // block_size
        if frame_num % block_size != 0:
            stft = stft[..., :block_size*block_num]
            vad_frame = vad_frame[..., :block_size*block_num]

        block_stft = stft.reshape(B, C, F, block_num, block_size)   # (B, C, F, block_num, block_size)
        block_stft = block_stft.permute(0, 3, 1, 2, 4).reshape(-1, C, F, block_size)    # (B*block_num, C, F, block_size)

        block_vad_frame = vad_frame.reshape(B, vad_frame.shape[1], block_num, block_size)  # (B, 1, block_num, block_size)
        block_vad_frame = block_vad_frame.permute(0, 2, 1, 3).reshape(-1, block_vad_frame.shape[1], block_size) # (B*block_num, 1, block_size)


        ############# extract embedding
        z, embedding = self.scl_model(stft=block_stft, vad_frame=block_vad_frame)   # (B*block_num, 128) (B*block_num, 256)
        embedding = embedding.unsqueeze(dim=1)     # (B*block_num, 1, 256)
        

        ############# DOA estimation
        out=self.crn(embedding)   # (B*block_num, 3, 360)
        out=out.reshape(B, block_num, 3, 360).permute(0, 2, 3, 1)   # (B, 3, 360, block_num)


        ############# make target
        vad_block = block_vad_frame.reshape(B, block_num, -1)   # (B, block_num, block_size)
        count_active_frame = torch.sum(vad_block==1, dim=-1)   # (B, block_num)
        vad_block = (count_active_frame > block_size//2).int()    # (B, block_num)
        vad_block = vad_block.unsqueeze(dim=1)   # (B, 1, block_num)

        # azi : (B, 1)
        target=self.make_target(vad_block, azi)   # (B, 3, 360, block_num)

        
        return out, target, vad_block


