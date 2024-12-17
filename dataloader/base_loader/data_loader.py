
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
        
        mixed = data_dict['noisy'].numpy()
        vad = data_dict['vad'].numpy()
        # vad = np.ones_like(vad)
        azi_list = data_dict['azi'].tolist()
        
        
        # does nothing when 'ans_azi'== 0
        # become torch.tensor
        vad, azi_list = self.multi_ans(vad, azi_list, self.ans_azi, self.degree_resolution)   
        
        return torch.from_numpy(mixed), vad, azi_list
    
    
class real_data_loader(datamake):
    
    def __init__(self, args):    
        super(real_data_loader, self).__init__()   
        
        self.args=args
        
        self.ans_azi=self.args['ans_azi']
        self.degree_resolution=self.args['degree_resolution']  
        self.num_spk = self.args['num_spk']
        
        self.data_dir = self.args['data_dir']        
        self.data_csv = pd.read_csv(self.args['data_csv'], index_col=0)
        self.vad_dir = self.args['vad_dir']
        
        
    def __len__(self):
        return len(self.data_csv)
    
    
    def  __getitem__(self, idx):
        
        data_idx = self.data_csv.loc[idx]
        azi_list = [int(data_idx['azimuth'])]       
        data_name = str(data_idx['filename'])
        
        mixed, original_fs = sf.read(self.data_dir+data_name, dtype='float32')

        vad_name = data_name.replace('.wav', '.npy')
        vad = np.load(self.vad_dir + vad_name)
        
        if mixed.ndim == 1:
            mixed = np.expand_dims(mixed, 0)
        elif mixed.shape[0] > mixed.shape[1]:
            mixed = mixed.T     # (4, ...)
        
        if vad.ndim == 1:
            vad = np.expand_dims(vad, 0)
        elif vad.shape[0] > vad.shape[1]:
            vad = vad.T
            
        
        mixed=mixed.astype('float32')
        vad=vad.astype('float32')
        
        samples = 8 * 44100
        if mixed.shape[1] < samples:
                pad = samples - mixed.shape[1]
                
                mixed = np.pad(mixed, ((0,0), (0, pad)), mode='constant')
                vad = np.pad(vad, ((0,0), (0, pad)), mode='constant')
        else:
            start = random.randint(0, mixed.shape[1]-samples)
            mixed = mixed[:, start:start+samples]
            vad = vad[:, start:start+samples]
        
        
        # does nothing but changes them into torch tensor when 'ans_azi'== 0
        vad, azi_list = self.multi_ans(vad, azi_list, self.ans_azi, self.degree_resolution)   
        
        return torch.from_numpy(mixed), vad, azi_list, self.num_spk, data_name


        