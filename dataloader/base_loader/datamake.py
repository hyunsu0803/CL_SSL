import numpy as np
import torch
import cupyx
import cupyx.scipy.signal 
import cupy
import math
from scipy.signal import resample


class datamake():
    def __init__(self,):

        self.eps = np.finfo(np.float32).eps

    
    def multi_ans(self, vad, azi_list, num_ans, degree_resolution):
        vad=torch.from_numpy(vad)

        if num_ans==0:
            return vad, torch.tensor(azi_list)

        vad=torch.repeat_interleave(vad, 2*num_ans+1, dim=0)
        
        new_azi=[]

        for azi in azi_list:
            temp_azi=torch.arange(azi-num_ans*degree_resolution, azi+(num_ans+1)*degree_resolution, degree_resolution)
            
            temp_azi=360+temp_azi
            temp_azi=torch.remainder(temp_azi, 360)
            new_azi.append(temp_azi)
        new_azi=torch.concat(new_azi)
        
        return vad, new_azi


    def gpu_convolve(self, audio, rir):
        audio=cupy.asarray(audio)
        rir=cupy.asarray(rir)

        audio=cupyx.scipy.signal.convolve(audio, rir, mode='full', method='fft')     
        audio=cupy.asnumpy(audio)           
      
        return audio


    def rms(self, data):
        """
        calc rms of wav
        """
        energy = data ** 2
        max_e = np.max(energy)
        low_thres = max_e*(10**(-50/10)) # to filter lower than 50dB 
        rms = np.mean(energy[energy>=low_thres])
        # rms=np.sqrt(rms)
        #rms = np.mean(energy)
        return rms

    def remove_dc(self, data):
        '''
        data: 1, T
        '''
        data_mean=data.mean()
        data=data-data_mean
        return data
     

    def snr_mix(self, clean, noise, snr):
        '''
        mix clean and noise according to snr
        '''
        
        clean_rms = self.rms(clean)
        clean_rms = np.maximum(clean_rms, self.eps)
        noise_rms = self.rms(noise)
        noise_rms = np.maximum(noise_rms, self.eps)
        k = math.sqrt(clean_rms / (10**(snr/10) * noise_rms))
        new_noise = noise * k
        return new_noise

    def scaling(self, data, normalize_factor):
        max_amp = np.max(np.abs(data))
        max_amp = np.maximum(max_amp, self.eps)
        scale=1. / max_amp * normalize_factor
        return scale

    def clipping(self, data, min=-1.0, max=1.0):

        return np.clip(data, min, max)

    def get_random_snr(self, white_noise_snr, normalize_factor):
        
        white_noise_snr=np.random.uniform(*white_noise_snr)
        normalize_factor=np.random.uniform(*normalize_factor)

        return white_noise_snr, normalize_factor
    
        
    def make_noisy(self,duration,  rired_speech_list,  white_noise_snr, normalize_factor, 
                   speech_start_point_list,  with_coherent_noise, coherent_noise_snr, rired_noise_wav):

        noisy_wav=np.zeros((rired_speech_list[0].shape[0], duration))

        for wav, start_point in zip(rired_speech_list, speech_start_point_list):
            noisy_wav[:,start_point:start_point+wav.shape[-1]]+=wav
        
        ##### white noise
        
        white_noise=np.random.normal(0, 1, noisy_wav.shape, ).astype('float32')
        white_noise=self.remove_dc(white_noise)
        noise=self.snr_mix(rired_speech_list[0], white_noise, white_noise_snr)

        if with_coherent_noise:
            
            rired_noise_wav=self.snr_mix(rired_speech_list[0], rired_noise_wav, coherent_noise_snr)

            noise=noise+rired_noise_wav

            noise=self.snr_mix(rired_speech_list[0], noise, coherent_noise_snr)
        
        # noise = np.zeros_like(noise)
        noisy_wav=noisy_wav+noise
        noisy_wav*=self.scaling(noisy_wav, normalize_factor)
        return noisy_wav
        
     
    def get_vad(self, duration, vad_list, speech_start_point_list, max_spk):
        vad=np.zeros((max_spk, duration))   # (1, 64000)
        for num, tmp in enumerate(vad_list):
            vad[num, speech_start_point_list[num]:speech_start_point_list[num]+tmp.shape[0]]=tmp
        return vad
    
    
    def resample(self, mixed, vad, original_fs, new_fs=16000):
        # mixed : (4, 64000)
        # vad : (1, 64000)
        mixed = mixed.astype(np.float32)
        vad = vad.astype(np.float32)
        
        dim1 = False
        if mixed.ndim == 1:
            mixed = mixed[np.newaxis, :]
            vad = vad[np.newaxis, :]
            dim1 = True
            
        num_samples_mixed = round(mixed.shape[1] * float(new_fs) / original_fs)
        mixed_resampled = np.zeros((mixed.shape[0], num_samples_mixed))
        for i in range(mixed.shape[0]):
            mixed_resampled[i, :] = resample(mixed[i, :], num_samples_mixed)
        mixed = mixed_resampled
        
        num_samples_vad = round(vad.shape[1] * float(new_fs) / original_fs)
        vad_resampled = np.zeros((vad.shape[0], num_samples_vad))
        for i in range(vad.shape[0]):
            vad_resampled[i, :] = resample(vad[i, :], num_samples_vad)
        vad = vad_resampled
        
        if dim1:
            mixed = mixed.squeeze(0)
            vad = vad.squeeze(0)

        return mixed, vad
    

    def rir_peak_find(self,rir):
        rir_peak=np.argmax(np.abs(rir[0]))
        return rir_peak
    

    def fit_max_mic(self, data, max_mic):
        data=np.pad(data, ((0, max_mic-data.shape[0]), (0,0)))
        return data
