import webrtcvad
import soundfile as sf
import numpy as np
import pandas as pd
import ast
import os
import tqdm


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


def make_online_gunshot_vad():

    audio_dir='/root/mydir/miyungpa/gunshot_real_old/'
    vad_dir='/root/mydir/miyungpa/prepared/vad/gunshot_real_old_111/'
    
    
    df = pd.read_csv('/root/mydir/miyungpa/metadata/gunshot_real_whole_labeled.csv', header=0)
    df.columns = df.columns.str.strip()
    
    for idx, data_row in df.iterrows():
        # print(idx)
        audio_name = data_row['filename']
        audio_path = audio_dir + audio_name
        # print("######", audio_name)

        num_gunshots = data_row['num_gunshots']
        
        gunshot_locations = []
        for n in range(num_gunshots):
            if 3+n < 10:
                gunshot_locations.append(data_row.iloc[3+n])

        
        audio_file, fs= sf.read(audio_path)
        # print(audio_file.shape)   :    (165375, )
        
        vad_out = np.ones_like(audio_file)
        # for t in gunshot_locations:
        #     bang = t * fs
        #     start = max(0, round(bang - 0.1*fs))
        #     end = min(len(vad_out), round(bang + 0.4*fs))
        #     vad_out[start:end] = 1
            
        vad_name=audio_path.replace('.wav', '.npy')
        vad_name=vad_name.replace(audio_dir, vad_dir)
        
      
        os.makedirs(os.path.dirname(vad_name), exist_ok=True)
        
        np.save(vad_name, vad_out)
     
       
# def make_real_gunshot_vad -> wevrtcvad.py

    
    
if __name__=='__main__':
    
    vad_tool=webrtcvad.Vad()

    wav_folder=dict()
    wav_folder['train-clean-100'] = "/root/mydir/LibriSpeech/train-clean-100/"
    wav_folder['The_Terror_Live'] = '/root/mydir/miyungpa/speech/'
    
    make_online_gunshot_vad()