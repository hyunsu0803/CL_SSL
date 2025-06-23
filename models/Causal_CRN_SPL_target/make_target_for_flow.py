from FFT import ConvSTFT 
import pickle
from glob import glob
import numpy as np
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import soundfile as sf



fft_config = {
    'win_len': 256,
    'win_inc': 128,
    'fft_len': 718,
    # 'fft_len': 256,
    'vad_threshold': 0.6666,
}
stft_model=ConvSTFT(**fft_config).to('cuda' if torch.cuda.is_available() else 'cpu')


def irtf_feature(mixed, vad):  

    ref_ch = 0
    eps = np.finfo(np.float32).eps

    mixed = mixed.unsqueeze(0)  # (1, C, T)
    vad = vad.unsqueeze(0)      # (1, T)

    r, i, vad_frame =stft_model(mixed, vad, cplx=True)
    # B x C x F x T = (B, 4, 513, 345)
    comp = torch.complex(r, i)

    
    comp_ref = comp[..., [ref_ch], :, :]
    comp=torch.cat((comp[..., ref_ch-1:ref_ch, :, :], comp[..., ref_ch+1:, :, :]), dim=-3)

    comp_norm = comp / (comp.abs() + eps)
    comp_ref_norm = comp_ref / (comp_ref.abs() + eps)
    
    irtf = comp_norm / comp_ref_norm

    feature=torch.cat((irtf.real, irtf.imag), dim=1)
    
    
    # (B, 2*(C-1), F, T), (B, F, T)
    # (B, 6, 129, 501)
    return feature, vad_frame


def make_target(vad_frame, azi):
    
    azi_target=torch.div(azi, 360//360, rounding_mode='floor').long()        
    azi_range=torch.arange(0, 360).unsqueeze(0).to(azi_target.device)

    distance=azi_target.unsqueeze(-1)-azi_range
    
    distance_abs=torch.abs(distance)
    distance_abs=torch.stack((distance_abs, 360-distance_abs), dim=0)
    
    distance=torch.min(distance_abs, dim=0).values
    distance=torch.deg2rad(distance).unsqueeze(1)
    
    sigma=torch.tensor([16.0, 6.0, 2.5]).view(1,-1, 1,1).to(distance.device)
    sigma=torch.deg2rad(sigma).to(distance.device)  # (1, 3, 1, 1)
    p = torch.tensor([0.707106781]).to(distance.device)  # (1, 1, 1, 1)
    kappa_d=torch.log(p)/(torch.cos(sigma)-1)
    

    labelling=torch.exp(kappa_d*(torch.cos(distance)-1)).unsqueeze(-1) # batch, number of sigma, number of speakers, time, 1  
    

    vad_frame=vad_frame.unsqueeze(1).unsqueeze(-2)
    
    target=labelling*vad_frame

    target=torch.max(target, dim=2).values
    
    
    return target


if __name__ == '__main__':

    pkl_list = glob("SSL_src/prepared/pkl/doa/*.pkl")

    for pkl_name in tqdm(pkl_list):

        pkl_file = open(pkl_name, 'rb')
        data_dict = pickle.load(pkl_file)   # torch tensors
        pkl_file.close()

        mixed = data_dict['mixed'][0]      # (n_channels, duration)
        vad = data_dict['vad']
        azi = data_dict['azi']

        mixed = mixed.to('cuda' if torch.cuda.is_available() else 'cpu')
        vad = vad.to('cuda' if torch.cuda.is_available() else 'cpu')
        azi = azi.to('cuda' if torch.cuda.is_available() else 'cpu')

        irtf, vad_frame = irtf_feature(mixed, vad)
        target = make_target(vad_frame, azi.unsqueeze(0))  


        target = target.cpu().numpy()
        target_name = pkl_name.split('/')[-1].replace('.pkl', '.npy')
        target_path = os.path.join("SSL_src/prepared", 'clean', target_name)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        np.save(target_path, target[0,2])

        irtf = irtf.cpu().numpy()
        irtf_name = pkl_name.split('/')[-1].replace('.pkl', '.npy')
        irtf_path = os.path.join("SSL_src/prepared", 'noisy', irtf_name)
        os.makedirs(os.path.dirname(irtf_path), exist_ok=True)
        np.save(irtf_path, irtf)


        ##### save target as png
        png_name = pkl_name.split('/')[-1].replace('.pkl', '.png')
        png_path = os.path.join("SSL_src/prepared", 'target_pngs', png_name)
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.figure()
        plt.imshow(target[0,1], aspect='auto', vmin=0.0, vmax=1.0, interpolation='nearest')
        plt.xlabel('Time frame')
        plt.ylabel('Source angle')
        plt.title('Target DOA spatial spectrum')
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()

