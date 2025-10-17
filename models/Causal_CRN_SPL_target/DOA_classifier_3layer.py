from .FFT import ConvSTFT 
from .input_process import Processing
from torch import nn
import torch
import numpy as np
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
    def __init__(self, config):
        super(crn, self).__init__()

        self.config=config
        
        self.cnn_num=self.config['CNN']['layer_num']
        self.kernel_size=self.config['CNN']['kernel_size']       # 3
        self.filter_size=self.config['CNN']['filter']            # 32

        self.max_pool_kernel=self.config['CNN']['max_pool']['kernel_size']
        self.max_pool_stride=self.config['CNN']['max_pool']['stride']


        ##############################
        # CNN layer
        ##############################
        kwargs = {'stride': 1, 'padding': [1, 2], 'dilation': 1}
        args = [self.config['input_cnn_channel'],  self.filter_size,   self.kernel_size]

        self.cnn=nn.ModuleList()
        self.pooling=nn.ModuleList()
        self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))       
        self.pooling.append(nn.MaxPool2d(kernel_size=self.max_pool_kernel, stride=self.max_pool_stride))  # (2, 2)
        
        args[0]=self.config['CNN']['filter']                               
        for count in range(self.cnn_num-1):
            self.cnn.append(Causal_Conv2D_Block(*args, **kwargs))   
            self.pooling.append(nn.MaxPool2d(kernel_size=self.max_pool_kernel, stride=self.max_pool_stride))  # (2, 2)
    

        self.GRU_layer=nn.GRU(**config['GRU'])
        self.h0=torch.zeros(*config['GRU_init']['shape'])
        self.h0=torch.nn.parameter.Parameter(self.h0, requires_grad=config['GRU_init']['learnable'])


        ##############################
        # output layer
        ##############################
        self.azi_mapping_conv_layer=nn.ModuleList()
        self.azi_mapping_final=nn.ModuleList()
        

        args = [256, 256, 1]
        kwargs['padding']=0
        
        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))       # (256, 256, 1)
        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))       # (256, 256, 1)
        self.azi_mapping_conv_layer.append(Conv1D_Block(*args, **kwargs))       # (256, 256, 1)
        
        args[1] = 360
        self.azi_mapping_final.append(nn.Conv1d(*args, **kwargs))       # (256, 360, 1)
        self.azi_mapping_final.append(nn.Conv1d(*args, **kwargs))       # (256, 360, 1)
        self.azi_mapping_final.append(nn.Conv1d(*args, **kwargs))       # (256, 360, 1)


    def forward(self, x):

        # x : (B, 1, 256, n) -> (B, 64, 128, n) -> (B, 64, 64, n) -> (B, 64, 32, n) -> (B, 64, 16, n)
        for cnn_layer, pooling_layer in zip(self.cnn, self.pooling):
            x=cnn_layer(x)[...,:x.shape[-1]]
            x=pooling_layer(x)


        b, c, f, n=x.shape              # (B, 64, 16, n)
        x=x.view(b, -1, n).permute(0,2,1)   # (B, n, 64*16=1024)
        
        h0 = self.h0.repeat_interleave(x.shape[0], dim=1)  
        self.GRU_layer.flatten_parameters()
        
        x, h=self.GRU_layer(x, h0)      # (B, n, 256(hidden size))

        x = x.permute(0,2,1)    # (B, 256, n)
        


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

        if self.use_scl:
            self.config['CRN']['input_cnn_channel'] = 1
            self.config['CRN']['GRU']['input_size'] = 2048
        else:
            self.config['CRN']['input_cnn_channel'] = 6
            self.config['CRN']['GRU']['input_size'] = 512
        
        self.eps=np.finfo(np.float32).eps
        self.ref_ch=self.config['ref_ch']

        ###### sigma

        self.p=torch.tensor(self.config['p'])
        self.sigma=torch.tensor(self.config['sigma_start'])
        self.degree_resolution = self.config['degree_resolution']
        self.azi_size=360//self.degree_resolution

        self.stft_model=ConvSTFT(**self.config['FFT'])
        self.data_proc=Processing(self.stft_model)
        self.crn=crn(self.config['CRN'])
        
        self.model_select_for_scl_feature()     # self.scl_model
       
    

    def model_select_for_scl_feature(self):
        
        model_name=self.config_scl['name']
        model_import='models.'+model_name+'.main'

        model_dir=importlib.import_module(model_import)
        
        self.scl_model=model_dir.get_model_for_scl(self.config_scl, doa=True)

        if self.use_scl and not self.finetune:

            self.trained_scl_model_path = self.hyparam['trained_scl_model_path']
            trained=torch.load(self.trained_scl_model_path) 
            self.scl_model.load_state_dict(trained['model_state_dict'], )   

        self.scl_model.train()
        # for param in self.scl_model.encoder_t.parameters():
        #     param.requires_grad = True
        

    def make_target(self, vad_block, azi):

        # vad_block : (B, num_spk, n)
        # azi : (B, num_spk)
        
        azi_range=torch.arange(0, self.azi_size).unsqueeze(0).to(azi.device)     # (1, 360)

        ang_diff=azi.unsqueeze(-1)*self.degree_resolution - azi_range*self.degree_resolution   # (B, num_spk, 360)
        
        distance_abs=torch.abs(ang_diff)    # (B, num_spk, 360)
        distance_abs=torch.stack((distance_abs, 360-distance_abs), dim=0)   # (2, B, num_spk, 360)
     
        ang_diff=torch.min(distance_abs, dim=0).values  # (B, num_spk, 360)
        ang_diff=torch.deg2rad(abs(ang_diff)).unsqueeze(1)   # (B, 1, num_spk, 360)
        
        sigma=self.sigma.view(1,-1, 1,1).to(ang_diff.device)    # (1, 3, 1, 1)
        sigma=torch.deg2rad(sigma)                              # (1, 3, 1, 1)
        kappa_d=torch.log(self.p)/(torch.cos(sigma)-1)          # (1, 3, 1, 1)
        

        labelling=torch.exp(kappa_d*(torch.cos(ang_diff)-1)).unsqueeze(-1) # (B, 3, num_spk, 360, 1)  
        
        # (B, num_spk, n) -> (B, 1, num_spk, 1, n)
        vad_block=vad_block[:, None, :, None, :]
        
        target = labelling*vad_block   # (B, 3, num_spk, 360, block_num)

        target=torch.max(target, dim=2).values   # (B, 3, 360, block_num)
       
        return target   

        
    def forward(self, mixed, vad, azi_list):

        block_stft, block_vad_frame = self.data_proc.make_block(mixed, vad)
        ibRTF, vad_block = self.data_proc.ib_RTF(block_stft, block_vad_frame)      # (B, 2(C-1), F, n), (B, num_spk, n)

        if self.use_scl:
            z, embedding, azi_list, vad_block = self.scl_model(mixed, vad, azi_list)
            embedding_s, embedding_t = embedding, embedding
            embedding_s = embedding_s.unsqueeze(1)   # (B, 1, 256, n)
            embedding_t = embedding_t.unsqueeze(1)
        else:
            embedding = ibRTF   # (B, 2(C-1), F, n)


        out=self.crn(embedding) # (B, 3, 360, n)

        target=self.make_target(vad_block, azi_list)   # (B, 3, 360, n)

        
        return out, target, vad_block


