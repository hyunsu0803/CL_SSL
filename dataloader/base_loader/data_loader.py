
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
        white_snr = data_dict['white_snr_list']
        rt60 = data_dict['rt60_list']
        
        
        # does nothing when 'ans_azi'== 0
        # become torch.tensor
        vad, azi_list = self.multi_ans(vad, azi_list, self.ans_azi, self.degree_resolution)   
        
        return torch.from_numpy(mixed), vad, azi_list, white_snr, rt60
    
    
class real_data_loader(datamake):
    
    def __init__(self, args):    
        super(real_data_loader, self).__init__()   
        
        self.args=args
        
        self.csv_dir = 'STARSS23/metadata_dev/'
        self.wav_dir = 'STARSS23/mic_dev/'
        self.vad_dir = 'STARSS23/mic_dev_vad/'
        self.label_dir = 'STARSS23/mic_dev_label/'
        
        self.csv_list = glob('STARSS23/metadata_dev/*/*.csv')


        
        
    def __len__(self):
        return len(self.csv_list)
    
    
    def  __getitem__(self, idx):
        
        csv_file = self.csv_list[idx]

        wav_file = csv_file.replace(self.csv_dir, self.wav_dir).replace('.csv', '.wav')
        vad_file = csv_file.replace(self.csv_dir, self.vad_dir).replace('.csv', '.npy')
        azi_file = csv_file.replace(self.csv_dir, self.label_dir).replace('.csv', '.npy')
        
        mixed, fs = sf.read(wav_file, dtype='float32')
        vad = np.load(vad_file)
        azi = np.load(azi_file)
        
            
        mixed=mixed.astype('float32')
        vad=vad.astype('float32')
        azi=azi.astype('float32')
        
        
        return torch.from_numpy(mixed), torch.from_numpy(vad), torch.from_numpy(azi)


        