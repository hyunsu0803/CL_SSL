
from .datamake import datamake
import os
import pickle
from natsort import natsorted
import pandas as pd
import soundfile as sf
from scipy.signal import resample
import torch
import numpy as np
import random
from glob import glob


class synth_data_loader(datamake):
    
    def __init__(self, args):    
        super(synth_data_loader, self).__init__()   
        
        self.args=args
        
        self.ans_azi=self.args['ans_azi']
        self.degree_resolution=self.args['degree_resolution']  
        
        self.pkl_dir = self.args['pkl_dir']
        self.pkl_list=natsorted(os.listdir(self.pkl_dir))
        
        
    def __len__(self):
        return len(self.pkl_list)
    
    
    def  __getitem__(self, idx):
        
        pkl_idx = self.pkl_list[idx]
        data_dir = self.pkl_dir + pkl_idx
        
        pkl_file = open(data_dir, 'rb')
        data_dict = pickle.load(pkl_file)   # torch tensors
        pkl_file.close()
        
        mixed = data_dict['mixed'].numpy()
        vad = data_dict['vad'].numpy()
        azi_list = data_dict['azi'].tolist()
        white_snr = data_dict['white_snr']
        coherent_snr = data_dict['coherent_snr']
        rt60 = data_dict['rt60']
        
        
        # does nothing when 'ans_azi'== 0
        # become torch.tensor
        vad, azi_list = self.multi_ans(vad, azi_list, self.ans_azi, self.degree_resolution)   
        
        return torch.from_numpy(mixed), vad, azi_list, white_snr, coherent_snr, rt60
    
    
class real_data_loader(datamake):
    
    def __init__(self, args):    
        super(real_data_loader, self).__init__()   
        
        self.args=args
        
        self.pkl_list = glob('/root/clssl/STARSS23/mic_dev_pkl/*/*.pkl')
        # self.pkl_list = glob('/root/clssl/STARSS23/mic_dev_pkl/dev-test-tau/*.pkl') + glob('/root/clssl/STARSS23/mic_dev_pkl/dev-train-tau/*.pkl')

        
    def __len__(self):
        return len(self.pkl_list)
    
    
    def  __getitem__(self, idx):
        
        pkl_file = self.pkl_list[idx]

        pkl_file = open(pkl_file, 'rb')
        data_dict = pickle.load(pkl_file)   # torch tensors
        pkl_file.close()
        
        mixed = data_dict['mixed'].T      # (n_channels, duration)
        vad = data_dict['vad']
        target = data_dict['azi']

        mixed=mixed.astype('float32')
        vad=vad.astype('float32')
        target=target.astype('int64')

        
        return torch.from_numpy(mixed), torch.from_numpy(vad), torch.tensor(target), 0, 0, 0


        