import webrtcvad
import pathlib
from tqdm import tqdm
import soundfile as sf
import numpy as np
import os 
import pandas as pd
from glob import glob
import ast


def _cleanSilences(s, vad_tool, fs, aggressiveness, return_vad=False):
    vad_tool.set_mode(aggressiveness)

    vad_out = np.zeros_like(s)
    vad_frame_len = int(10e-3 * fs)
    n_vad_frames = len(s) // vad_frame_len
    for frame_idx in range(n_vad_frames):
        frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
        frame_bytes = (frame * 32767).astype('int16').tobytes()
        vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = vad_tool.is_speech(frame_bytes, fs)
    
    s_clean = s * vad_out
    
    return (s_clean, vad_out) if return_vad else s_clean

def _clean_gunshot_locations(location_str):
    try:
        # Remove extra commas and spaces
        cleaned_str = location_str.replace(' ', ',').replace(',,', ',')
        # Remove leading and trailing commas
        cleaned_str = cleaned_str.strip(',')
        # Convert to list
        return ast.literal_eval(cleaned_str)
    except (SyntaxError, ValueError):
        # Handle invalid entries
        return []



def make_speech_vad(vad_tool, wav_folder, vad_folder):
    key_list = ['train-clean-100']

    for key in key_list:
        print("check 1", key)
        vad_dir=vad_folder[key]
        data_dir=wav_folder[key]
        print("vad dir", vad_dir)
        print("data dir", data_dir)

            
        for audio_name in tqdm(pathlib.Path(data_dir).rglob('*.flac')):
            audio_name=str(audio_name)
            print(audio_name)
            
            audio_file, fs= sf.read(audio_name)
            s_clean, vad_out=_cleanSilences(audio_file, vad_tool, fs, 3, return_vad=True)

            if np.count_nonzero(s_clean) < len(audio_file) * 0.66:
                s_clean, vad_out = _cleanSilences(audio_file, vad_tool, fs, 2, return_vad=True)
            if np.count_nonzero(s_clean) < len(audio_file) * 0.66:
                s_clean, vad_out = _cleanSilences(audio_file, vad_tool, fs, 1, return_vad=True)

                
            vad_out=vad_out.astype(bool)

            vad_name=audio_name.replace('.flac', '.npy')
            vad_name=vad_name.replace(data_dir, vad_dir)
            os.makedirs(os.path.dirname(vad_name), exist_ok=True)
            
            np.save(vad_name, vad_out)
            
            
def make_online_gunshot_vad(vad_tool, wav_folder, vad_folder):
    key_list = ['gunshot']

    for key in key_list:
        
        vad_dir=vad_folder[key]
        data_dir=wav_folder[key]
        
        print("key     ", key)
        print("vad dir ", vad_dir)
        print("data dir", data_dir)


        # 1989 + 159 = 214
        label_csv = '/root/mydir/miyungpa/metadata/gunshot_online_whole_labeled.csv'
        df = pd.read_csv(label_csv)
        
        # Convert string representation of lists to actual lists
        # df['gunshot_location_in_seconds'] = df['gunshot_location_in_seconds'].apply(_clean_gunshot_locations)

        audio_list=glob(data_dir+'/**/*.wav', recursive=True)
            
        drop_idx_list = []
        for idx, audio_path in tqdm(enumerate(audio_list), total=len(audio_list)):
            audio_path=str(audio_path)
            
            filename = audio_path.split('/')[-1].strip('.wav')
            
            audio_file, fs= sf.read(audio_path)
            vad_out = np.zeros(audio_file.shape[0])
            
            data_row = df[df['filename'] == filename]
            if data_row.empty:
                drop_idx_list.append(idx)
                print("empty row", filename, idx)
                continue
            
            num_gunshots = data_row.iloc[0]['num_gunshots']
            
            # gunshot_locations = data_row.iloc[0]['gunshot_location_in_seconds']
            gunshot_locations = []
            for n in range(num_gunshots):
                gunshot_locations.append(data_row.iloc[0, 2+n])

            for t in gunshot_locations:
                bang = t * fs
                start = max(0, round(bang - 0.1*fs))
                end = min(len(vad_out), round(bang + 0.4*fs))
                vad_out[start:end] = 1
            
            
            vad_name=audio_path.replace('.wav', '.npy')
            vad_name=vad_name.replace(data_dir, vad_dir)
            os.makedirs(os.path.dirname(vad_name), exist_ok=True)
            
            np.save(vad_name, vad_out)
            
        df.drop(df.index[drop_idx_list], inplace=True)
        df.to_csv(label_csv, index=False)


def make_real_gunshot_vad(vad_tool, wav_folder, vad_folder):
    key_list = ['gunshot_real']
    wav_folder['gunshot_real'] = '/root/mydir/miyungpa/gunshot_real/val/'
    vad_folder['gunshot_real'] = '/root/mydir/miyungpa/prepared/vad/gunshot_real/val/'

    for key in key_list:
        
        vad_dir=vad_folder[key]
        data_dir=wav_folder[key]
        
        print("key     ", key)
        print("vad dir ", vad_dir)
        print("data dir", data_dir)
        
        label_csv = '/root/mydir/miyungpa/metadata/ours_val.tsv'
        df = pd.read_csv(label_csv, sep='\t')
        
        audio_list=glob(data_dir+'/*.wav', recursive=True)
    
              
        for _, audio_path in tqdm(enumerate(audio_list), total=len(audio_list)):
            audio_path=str(audio_path)
            
            filename = audio_path.split('/')[-1]
            
            audio_file, fs= sf.read(audio_path)
            vad_out = np.zeros(audio_file.shape[0])
            
            data_rows = df[df['filename'] == filename]
            
            if data_rows.empty:
                print("empty")
                continue
            
            for _, row in data_rows.iterrows():
                onset = row['onset']
                # offset = row['offset']
                offset = onset + 0.4
                
                # Calculate start and end in samples
                start = max(0, round(onset * fs))
                end = min(len(vad_out), round(offset * fs))
                vad_out[start:end] = 1
            
            vad_name = audio_path.replace('.wav', '.npy')
            vad_name = vad_name.replace(data_dir, vad_dir)
            os.makedirs(os.path.dirname(vad_name), exist_ok=True)
    
            np.save(vad_name, vad_out)
             
        
            
if __name__=='__main__':
    
    vad_tool=webrtcvad.Vad()

    wav_folder=dict()
    wav_folder['train-clean-100'] = "/root/mydir/LibriSpeech/train-clean-100/"
    wav_folder['The_Terror_Live'] = '/root/mydir/miyungpa/speech/'
    wav_folder['gunshot_online'] = '/root/mydir/miyungpa/gunshot_online/edge-collected-gunshot-audio/'
    

    vad_folder=dict()
    vad_folder['train-clean-100'] = "/root/mydir/SSL_src/prepared/vad/train/"
    vad_folder['The_Terror_Live'] = '/root/mydir/miyungpa/prepared/vad/'
    vad_folder['gunshot_online'] = '/root/mydir/miyungpa/prepared/vad/gunshot/'
    
    
    # make_speech_vad(vad_tool, wav_folder, vad_folder)
    # make_online_gunshot_vad(vad_tool, wav_folder, vad_folder)
    make_real_gunshot_vad(vad_tool, wav_folder, vad_folder)
    
    