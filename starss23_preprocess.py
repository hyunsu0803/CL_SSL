from sklearn.model_selection import train_test_split
import pandas as pd
import os
from glob import glob
import librosa
import soundfile as sf
import numpy as np
import pickle
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
  


def dcase_csv_column_name():

    csv_list = glob('/root/clssl/STARSS23/metadata_dev/**/*.csv')

    for csv_file in csv_list:
        df = pd.read_csv(csv_file)
        df.columns = ['event_frame', 'event', 'index', 'azimuth', 'elevation', 'distance']
        df.to_csv(csv_file, index=False)


def delete_csv():
    csv_list = glob('/root/clssl/STARSS23/metadata_dev/*/*.csv')

    for csv_file in csv_list:

        df = pd.read_csv(csv_file)

        has_0 = (df['event'] == 0).any()
        has_1 = (df['event'] == 1).any()

        if not has_0 and not has_1:
            print(f"Deleting: {csv_file}")
            os.remove(csv_file)
            os.remove(csv_file.replace('metadata_dev', 'mic_dev_downsampled').replace('.csv', '.wav'))
        


def downsampling():

    target_dir = 'dev-train-tau/'

    meta_dir = 'metadata_dev/'
    wav_dir = 'mic_dev/'
    downsampled_dir = 'mic_dev_downsampled/'

    csv_list = glob(os.path.join(meta_dir, target_dir, '*.csv'))


    target_sr = 16000

    for csv_file in csv_list:

        input_wav = csv_file.replace('.csv', '.wav').replace(meta_dir, wav_dir)
        output_wav = input_wav.replace(wav_dir, downsampled_dir)
        

        y, sr = librosa.load(input_wav, sr=24000, mono=False)   
        print("Original sampling rate:", sr)

        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr, axis=-1)
        if y_resampled.ndim == 2:
            y_resampled = y_resampled.T

        os.makedirs(os.path.join(downsampled_dir, target_dir), exist_ok=True)
        sf.write(output_wav, y_resampled, target_sr)

    
def plot_out_target(out, target, pkl_idx):

    ## save as png
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(out, aspect='auto', vmin=0.0, vmax=1.0, interpolation='nearest')
    plt.xlabel('Time frame')
    plt.ylabel('Source angle')
    plt.title('Estimated DOA spatial spectrum')
    plt.subplot(2,1,2)
    plt.imshow(target, aspect='auto', vmin=0.0, vmax=1.0, interpolation='nearest')
    plt.xlabel('Time frame')
    plt.ylabel('Source angle')
    plt.title('Target DOA spatial spectrum')
    os.makedirs('STARSS23/pngs/', exist_ok=True)
    plt.tight_layout()
    plt.savefig('STARSS23/pngs/' + pkl_idx.split('/')[-1].replace('.pkl', '.png'), dpi=600)
    plt.close()


def generate_weight(azi):
        
    p = torch.tensor(0.707106781)
    sigma = torch.tensor([16.0, 6.0, 2.5])
    
    distance = torch.abs(azi - torch.arange(0, 360))            
    distance = torch.where(distance>180, 360-distance, distance)
    distance = torch.deg2rad(distance)                        
                        
    sigma=torch.deg2rad(sigma)
    kappa_d=torch.log(p)/(torch.cos(sigma)-1)          
    
    kappa_d.unsqueeze_(-1)   # (3, 1)
    distance.unsqueeze_(0) # (1, 360)

    labelling=torch.exp(kappa_d*(torch.cos(distance)-1))    # (3, 360)

    return labelling.numpy()


def make_vad_and_label():

    meta_dir = '/root/clssl/STARSS23/metadata_dev/'
    wav_dir = '/root/clssl/STARSS23/mic_dev_downsampled/'
    pkl_dir = '/root/clssl/STARSS23/mic_dev_pkl/'

    csv_list = glob('/root/clssl/STARSS23/metadata_dev/*/*.csv')

    for csv_file in tqdm(csv_list, total=len(csv_list)):

        input_wav = csv_file.replace('.csv', '.wav').replace(meta_dir, wav_dir)
        pkl_name = csv_file.replace('.csv', '.pkl').replace(meta_dir, pkl_dir)

        mixed, sr = sf.read(input_wav)      # y : (duration, n_channels)
        if sr != 16000:
            raise ValueError("Invalid sampling rate")

        voice_list = [0, 1, 4]

        duration, n_channels = mixed.shape
        duration_sec = duration / sr
        num_event_frame = int(duration_sec / 0.1) + 1  # 0.1 sec blocks
        pseudo_target = np.zeros((3, 360, num_event_frame))

        
        df_event = pd.read_csv(csv_file)
        for i, row in df_event.iterrows():

            if int(row['event']) not in voice_list:
                continue


            event_frame = int(row['event_frame'])
            azimuth = int(row['azimuth'])

            labelling = generate_weight(azimuth)    # (3, 360)
            event_frame_array = pseudo_target[..., event_frame]
            event_frame_array = np.maximum(event_frame_array, labelling)

            pseudo_target[..., event_frame] = event_frame_array

        if pseudo_target.shape[-1] % 2 != 0:
            pseudo_target = pseudo_target[..., :-1]   # remove last frame
        else:
            pseudo_target = pseudo_target[..., 1:-1]   # remove first and last frame
        n_layers, azi_size, num_event_frame = pseudo_target.shape
        pseudo_target = pseudo_target.reshape(n_layers, azi_size, -1, 2)
        pseudo_target = pseudo_target.max(axis=-1)   # (3, 360, num_event_frame/2)

        # plot_out_target(pseudo_target[0], pseudo_target[1], '22')


        os.makedirs(os.path.dirname(pkl_name), exist_ok=True)
        with open(pkl_name, 'wb') as f:
            pickle.dump({'mixed': mixed, 'vad': np.zeros_like(mixed[None, :,0]), 'azi': pseudo_target}, f)


def test():
    pkl_file = '/root/clssl/STARSS23/mic_dev_pkl/dev-test-sony/fold4_room24_mix011.pkl'
        
    pkl_file = open(pkl_file, 'rb')
    data_dict = pickle.load(pkl_file)   # torch tensors
    pkl_file.close()
    
    mixed = data_dict['mixed']      # (duration, n_channels)
    vad_6 = data_dict['vad']
    azi_list_6 = data_dict['azi']

    vad = []
    azi_list = []

    for i in range(6):
        if azi_list_6[i] is not None:
            vad.append(vad_6[:, i])
            azi_list.append(azi_list_6[i])
    
    vad = np.stack(vad, axis=0)
    mixed = mixed.T
            


# dcase_csv_column_name()
# delete_csv()
# downsampling()
make_vad_and_label()
# test()
