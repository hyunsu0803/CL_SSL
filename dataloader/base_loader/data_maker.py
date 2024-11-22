from .datamake import datamake
from ..random_gpu_rir_generator import gpu_rir_gen
import os
import pandas as pd
import soundfile as sf
import numpy as np
import random
import torch
import pickle


        
class base_data_maker(datamake):
    def __init__(self, args):

        super(base_data_maker, self).__init__() 
        
        self.args=args
        self.noise_dir= self.args['noise_dir']
        self.speech_dir= self.args['speech_dir']
        self.vad_dir=self.args['vad_dir']
        self.metadata_dir=self.args['metadata_dir']
        
        self.noise_csv=pd.read_csv(self.metadata_dir+self.args['noise_csv'], index_col=0)
        self.speech_csv=pd.read_csv(self.metadata_dir+self.args['speech_csv'], index_col=0) 
        
        # hyperparameter
        self.duration=self.args['duration']
        self.least_chunk_size=self.args['speech_least_chunk_size']
        
        # random
        self.max_num_people=self.args['max_spk']
        self.normalize_factor_bound=self.args['normalize_factor']        
        self.white_noise_snr=self.args['white_noise_snr'] 
        
        self.rir_maker = gpu_rir_gen.acoustic_simulator_on_the_fly(self.args)
        
        #### only train on the fly maker
        self.ans_azi=self.args['ans_azi']
        self.degree_resolution=self.args['degree_resolution']
    
    # both
    def noise_load(self):
        
        noise_info=self.noise_csv.sample(n=1)
        noise_total_duration=noise_info['length'].iloc[0]
        
        noise_start_sample=0
        if noise_total_duration>self.duration:
            noise_start_sample=random.randrange(0, noise_total_duration-self.duration)
            padding_size=0            
            get_duration=self.duration
        
        else:
            padding_size=self.duration-noise_total_duration
            get_duration=-1
        
        
        noise_path = str(noise_info['noise_directory'].iloc[0])
        # only train
        if noise_path.split('/')[0] == 'from_train_set':
            noise_path = noise_path.replace('from_train_set', 'noise_train')
        elif noise_path.split('/')[0] == 'from_test_set':
            noise_path = noise_path.replace('from_test_set', 'noise_test')
        
        noise_wav, _ = sf.read(self.noise_dir + noise_path, dtype='float32', start=noise_start_sample, frames=get_duration)
        
        if padding_size>0:
            front_padding=np.random.randint(0, padding_size)
            noise_wav=np.pad(noise_wav, (front_padding, padding_size-front_padding))

        noise_wav=np.expand_dims(noise_wav, 0)
        noise_wav=self.remove_dc(noise_wav)
       
        return noise_wav

    # both
    def select_different_speakers(self, speech_info, num_spk):
        temp_total_df=self.speech_csv
        
        for spk in range(num_spk-1): # selecting num_spk-1 more speakers, not overlapped
            last_speaker=speech_info.iloc[-1]['speaker']
            temp_total_df=temp_total_df.drop(temp_total_df[temp_total_df['speaker']==last_speaker].index)
            temp_df=temp_total_df.sample(1)            
            speech_info=pd.concat((speech_info, temp_df))
      
        return speech_info

    # both
    def speech_get_wav(self, wav_file, vad):

        """
        This is data maker. 'speech_wav' must be 1D array.
        """
        
        speech_wav, fs = sf.read(wav_file, dtype='float32')
        if speech_wav.ndim>1:
            speech_wav=speech_wav[:,0]
        
        speech_total_duration=speech_wav.shape[0]
        
        ######## select postion
        if speech_total_duration<self.least_chunk_size:
            pos='mid'
        elif speech_total_duration>self.duration:
            pos=random.choice(['full', 'back', 'front'])
        else:
            pos=random.choice(['front', 'back', 'mid'])

        ###### get chunk
        speech_start_sample=0
        if pos=='full':
            speech_start_sample=random.randrange(0, speech_total_duration-self.duration)        
            speech_wav=speech_wav[speech_start_sample:speech_start_sample+self.duration,]
            vad=vad[speech_start_sample:speech_start_sample+self.duration,]
            start_point=0

        elif pos=='front': 
            new_duration=random.randrange(self.least_chunk_size, self.duration+1)
            speech_wav=speech_wav[speech_start_sample:speech_start_sample+new_duration,]
            vad=vad[speech_start_sample:speech_start_sample+new_duration,]
            start_point=0

        elif pos=='back':
            new_duration=random.randrange(self.least_chunk_size, self.duration+1)
           
            speech_wav=speech_wav[-new_duration:,]
            vad=vad[-new_duration:,]
            start_point=self.duration-new_duration
   
        elif pos=='mid':
    
            start_point=random.randint(0, self.duration-speech_total_duration)

        speech_wav=self.remove_dc(speech_wav)
        
        return speech_wav, pos, start_point, vad, fs

    # both
    def get_speech_start_point(self, speech_wav, rir_peak, pos, vad, ):
        
        vad_len=vad.shape[-1]
        rired_len=speech_wav.shape[-1]
        if pos=='full':
            start_point=0
            speech_wav=speech_wav[:, rir_peak:rir_peak+self.duration]

        elif pos=='mid':
            
            if rired_len>self.duration:
                speech_wav=speech_wav[:, rir_peak:self.duration+rir_peak]
            rired_len=speech_wav.shape[-1]
            start_point=random.randint(0, self.duration-rired_len)

        
            vad=np.pad(vad, ( 0, rired_len-vad_len))
            
        elif pos=='front':
            back_cut=rired_len-vad_len-rir_peak
            speech_wav=speech_wav[:, :-back_cut]
            rired_len=speech_wav.shape[-1]

            if rired_len>self.duration:
                start_point=0
                speech_wav=speech_wav[:, -self.duration]
                vad=np.pad(vad, ( self.duration-vad_len, 0))

            else:
                start_point=self.duration-rired_len
                vad=np.pad(vad, ( rir_peak, 0))

        elif pos=='back':
            front_cut=rir_peak
            speech_wav=speech_wav[:, front_cut:]
            rired_len=speech_wav.shape[-1]

            if rired_len>self.duration:
                start_point=0
                speech_wav=speech_wav[:, :self.duration]
                vad=np.pad(vad, (0, self.duration-vad_len))
            else:
                start_point=0
                vad=np.pad(vad, ( 0, rired_len-vad_len))


        return start_point, speech_wav, vad
    

    def spk_mixer(self, rired_speech_list):
        ref_wav=rired_speech_list[0]
        
        for spk_num in range(len(rired_speech_list[0:])):
            spk_snr=random.uniform(*self.args['spk_SNR'])
            other_spk=self.snr_mix(ref_wav, rired_speech_list[spk_num], spk_snr)
            rired_speech_list[spk_num]=other_spk
     
        return rired_speech_list
    

    def main_speech_load(self, speech_info):
        
        wav_path = self.speech_dir + speech_info['filename']

        ftype = '.' + speech_info['filename'].split('.')[-1]
        vad_name = self.vad_dir + speech_info['filename'].replace(ftype, '.npy')

        vad=np.load(vad_name)

        return self.speech_get_wav(wav_path, vad)


    def __len__(self):
        return len(self.speech_csv)


    def make_data(self, idx, with_coherent_noise=True):

        num_spk=random.randint(1, self.max_num_people) 
        
        rirs, mic_pos, azi_list, linear_azi_list = self.rir_maker.create_rir(num_spk=num_spk, 
                                                                             with_coherent_noise=with_coherent_noise, 
                                                                             mic_type=self.args['mic_type'], 
                                                                             mic_num=self.args['mic_num'])
 
        coherent_noise_snr=None
        rired_noise_wav=None
        
        ####### coherent noise
        if with_coherent_noise:
            noise_rir=rirs[-1]
            self.noise_rir_peak=self.rir_peak_find(noise_rir)
            noise_wav=self.noise_load()
            
            rired_noise_wav=self.gpu_convolve(noise_wav, noise_rir)[:,self.noise_rir_peak:self.duration+self.noise_rir_peak]
            coherent_noise_snr=np.random.uniform(*self.args['SNR'])
        
        speech_rirs=rirs[:num_spk]
        azi_list=azi_list[:num_spk]
      
      
        ##### speech
        
        rired_speech_list=[]
        vad_list=[]
        speech_start_point_list=[]
        
        speech_info=self.select_different_speakers(self.speech_csv.iloc[idx:idx+1], num_spk)    # num_spk=1

        # only 1 iterration
        for spk_num, spk_info in enumerate(speech_info.iterrows()):
            
            spk_info = spk_info[1]          
            speech_wav, pos, speech_start_point, vad_out, fs = self.main_speech_load(spk_info)
            s_clean = speech_wav * vad_out

            speech_rir = speech_rirs[spk_num]       
            self.speech_rir_peak = self.rir_peak_find(speech_rir)
            
            # print(s_clean.shape, speech_rir.shape)      # (34416,) (4, 5040)
            if s_clean.ndim == 1:
                s_clean = np.expand_dims(s_clean, 0)
            
            # RIR convolution
            rired_speech = self.gpu_convolve(s_clean, speech_rir)
            
            start_point, rired_speech, vad_out=self.get_speech_start_point(rired_speech, self.speech_rir_peak, pos, vad_out, )

            rired_speech_list.append(rired_speech)
            vad_list.append(vad_out)
            speech_start_point_list.append(start_point)
        
        
        white_noise_snr, normalize_factor=self.get_random_snr(self.white_noise_snr, self.normalize_factor_bound)

        ####### spk mixer, normalizing speech
        # rired_speech_list=self.spk_mixer(rired_speech_list)
        
        ########### get vad     (1, 64000)
        vad=self.get_vad(self.duration, vad_list, speech_start_point_list, self.max_num_people)
        # vad=np.ones_like(vad)
        
        ######## speech & noise   
        mixed=self.make_noisy(self.duration, rired_speech_list, white_noise_snr, normalize_factor, 
                              speech_start_point_list, with_coherent_noise, coherent_noise_snr, rired_noise_wav)
        mixed=self.clipping(mixed)
      

        for i in range(self.max_num_people-num_spk):
            azi_list.append(0)
        mixed=mixed.astype('float32')
        vad=vad.astype('float32')
        # mixed : (4, 64000)
        # vad : (1, 64000)
        # speech_azi : (1,)
        # num_spk : 1
        
    
        # vad & azi_list == torch.tensor
        vad, azi_list=self.multi_ans(vad, azi_list, self.ans_azi, self.degree_resolution)
        
        return torch.from_numpy(mixed), vad, azi_list, num_spk, fs
    
    
    def __getitem__(self, idx):
        return self.make_data(idx)

    
    
class train_data_maker(base_data_maker):
    def __init__(self, args):
        super(train_data_maker, self).__init__(args)
    
    
class speech_data_maker(base_data_maker):
    def __init__(self, args):
        super(speech_data_maker, self).__init__(args)
        
        print('speech_csv', self.args['speech_csv'])
        print('noise_csv', self.args['noise_csv'])
        
        self.pkl_dir=self.args['pkl_dir']
        os.makedirs(self.pkl_dir, exist_ok=True)
        
        
    def save_data(self, idx):
        mixed, vad, azi_list, num_spk, fs = self.make_data(idx)
        save_dict={}
        save_dict['noisy']=mixed
        save_dict['vad']=vad
        save_dict['azi']=azi_list
        save_dict['num_spk']=num_spk
        save_dict['fs']=fs

        pkl_name=self.pkl_dir+str(idx)+'.pkl'
        output=open(pkl_name, 'wb')
        pickle.dump(save_dict, output)
        output.close()
        
        return 1, 2, 3, 4
        
        
    def __getitem__(self, idx):
        return self.save_data(idx)


# pkl making
class gunshot_data_maker(base_data_maker):
    def __init__(self, args):
        super(gunshot_data_maker, self).__init__(args)
        
        print('speech_csv', self.args['speech_csv'])
        print('noise_csv', self.args['noise_csv'])
        
        self.pkl_dir=self.args['pkl_dir']
        os.makedirs(self.pkl_dir, exist_ok=True)
               

    def save_data(self, idx):
            mixed, vad, azi_list, num_spk, fs= self.make_data(idx, with_coherent_noise=True)
            save_dict={}
            save_dict['noisy']=mixed
            save_dict['vad']=vad
            save_dict['azi']=azi_list
            save_dict['num_spk']=num_spk
            save_dict['fs']=fs

            pkl_name=self.pkl_dir+str(idx)+'.pkl'
            output=open(pkl_name, 'wb')
            pickle.dump(save_dict, output)
            output.close()
            
            return 1, 2, 3, 4
            
        
    def __getitem__(self, idx):
        return self.save_data(idx)
    