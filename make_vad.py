import webrtcvad
import pathlib
from tqdm import tqdm
import soundfile as sf
import numpy as np
import os 


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


def make_speech_vad(vad_tool, wav_folder, vad_folder):
    key_list = ['train-clean-100', 'test-clean']

    for key in key_list:

        print(key)

        vad_dir=vad_folder[key]
        data_dir=wav_folder[key]
        print("vad dir", vad_dir)
        print("data dir", data_dir)

            
        for audio_name in tqdm(pathlib.Path(data_dir).rglob('*.flac')):
            audio_name=str(audio_name)
            
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
        
            
if __name__=='__main__':
    
    vad_tool=webrtcvad.Vad()

    wav_folder=dict()
    wav_folder['train-clean-100'] = "/root/clssl/LibriSpeech/train-clean-100/"
    wav_folder['test-clean'] = "/root/clssl/LibriSpeech/test-clean/"
    

    vad_folder=dict()
    vad_folder['train-clean-100'] = "/root/clssl/SSL_src/prepared/vad/train-clean-100/"
    vad_folder['test-clean'] = "/root/clssl/SSL_src/prepared/vad/test-clean/"
    
    
    make_speech_vad(vad_tool, wav_folder, vad_folder)
