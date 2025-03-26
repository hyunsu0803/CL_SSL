from sklearn.model_selection import train_test_split
import pandas as pd
import os
from glob import glob
import librosa
import soundfile as sf
import numpy as np
import pickle
  


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
            # os.remove(csv_file.replace('metadata_dev', 'mic_dev_downsampled').replace('.csv', '.wav'))
        


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
        

        y, sr = librosa.load(input_wav, sr=24000, mono=False)   # 
        print("Original sampling rate:", sr)

        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr, axis=-1)
        if y_resampled.ndim == 2:
            y_resampled = y_resampled.T

        os.makedirs(os.path.join(downsampled_dir, target_dir), exist_ok=True)
        sf.write(output_wav, y_resampled, target_sr)


def make_vad_and_label():

    meta_dir = '/root/clssl/STARSS23/metadata_dev/'
    wav_dir = '/root/clssl/STARSS23/mic_dev_downsampled/'
    pkl_dir = '/root/clssl/STARSS23/mic_dev_pkl/'

    csv_list = glob('/root/clssl/STARSS23/metadata_dev/*/*.csv')

    for csv_file in csv_list:

        input_wav = csv_file.replace('.csv', '.wav').replace(meta_dir, wav_dir)
        pkl_name = csv_file.replace('.csv', '.pkl').replace(meta_dir, pkl_dir)

        mixed, sr = sf.read(input_wav)      # y : (duration, n_channels)
        if sr != 16000:
            raise ValueError("Invalid sampling rate")

        vad = np.zeros((mixed.shape[0], 6))
        azimuth_list = [None] * 6

        unit_samples = 1600
        voice_list = [0, 1, 4]


        df_event = pd.read_csv(csv_file)
        for i, row in df_event.iterrows():
            if int(row['event']) not in voice_list:
                continue
            event_frame = int(row['event_frame'])
            azimuth = int(row['azimuth'])
            speaker_index = int(row['index'])

            azimuth_list[speaker_index] = azimuth
            vad[event_frame*unit_samples:(event_frame+1)*unit_samples, speaker_index] = 1


        os.makedirs(os.path.dirname(pkl_name), exist_ok=True)
        with open(pkl_name, 'wb') as f:
            pickle.dump({'mixed': mixed, 'vad': vad, 'azi': azimuth_list}, f)


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
# make_vad_and_label()
test()
