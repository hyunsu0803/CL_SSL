import torch
import numpy as np


class Prepro():

    def __init__(self, stft_model):

        self.stft_model = stft_model
        self.ref_ch = 0
        self.eps = np.finfo(np.float32).eps


    def permute_data(self, mixed, vad, speech_azi):
        
        mixed = mixed.reshape(-1, mixed.shape[-2], mixed.shape[-1])
        vad = vad.reshape(-1, vad.shape[-2], vad.shape[-1])
        speech_azi = speech_azi.reshape(-1, speech_azi.shape[-1])
            
        perm = torch.randperm(mixed.size(0)).to(mixed.device) # (size : 256)
        
        mixed = mixed.index_select(0, perm)  
        vad = vad.index_select(0, perm)  
        speech_azi = speech_azi.index_select(0, perm)
        
        return mixed, vad, speech_azi
    

    def make_block(self, mixed, vad):

        r, i, vad_frame =self.stft_model(mixed, vad, cplx=True)
        stft = torch.complex(r, i)

        B, C, F, T = stft.shape


        frame_num = stft.shape[-1]
        block_size = 24
        block_num = frame_num // block_size
        if frame_num % block_size != 0:
            stft = stft[..., :block_size*block_num]
            vad_frame = vad_frame[..., :block_size*block_num]


        block_stft = stft.reshape(B, C, F, block_num, block_size)   # (B, C, F, n, s)
        block_vad_frame = vad_frame.reshape(B, vad_frame.shape[1], block_num, block_size)  # (B, 1, n, s)
        

        return block_stft, block_vad_frame
    

    def ib_RTF(self, block_stft, block_vad_frame):  # (B, C, F, n, s) (B, num_spk, n, s)

        B, C, F, n, s = block_stft.shape
        num_spk = block_vad_frame.shape[1]

        block_stft = block_stft.permute(0, 3, 1, 2, 4).reshape(-1, C, F, s)    # (B*block_num, C, F, block_size)
        block_vad_frame = block_vad_frame.permute(0, 2, 1, 3).reshape(-1, block_vad_frame.shape[1], s) # (B*block_num, num_spk, block_size)
    
        linear_spectra = block_stft.permute(0, 2, 3, 1)    # (B*n, F, s, C)
        
        cov_z = torch.einsum('bftc,bftd->bfcd', linear_spectra, linear_spectra.conj())

        col0 = cov_z[:, :, :, self.ref_ch]  # (B*n, F, C)        
        col00 = col0[:, :, self.ref_ch]     # (B*n, F)
        
        col0 = torch.cat((col0[:,:,self.ref_ch-1:self.ref_ch], col0[:,:,self.ref_ch+1:]), dim=-1)    # (B*n, F, C-1)
        col00 = torch.complex(col00.real.clamp(self.eps), col00.imag.clamp(self.eps))
        
        ibRTF = col0 / col00[:, :, None]    # (B*n, F, C-1)
        ibRTF = torch.cat((ibRTF.real, ibRTF.imag), dim=-1)    # (B*n, F, 2(C-1))

        ibRTF = ibRTF.permute(0, 2, 1)    # (B*n, 2(C-1), F)
        ibRTF = ibRTF.reshape(B, n, 2*(C-1), F).permute(0, 2, 3, 1)    # (B, 2(C-1), F, n)

        active_block_list = []
        for spk in range(num_spk):
            active_frames = [len(torch.nonzero(block_vad_frame[b, spk] == 1, as_tuple=True)[0]) for b in range(block_vad_frame.shape[0])]
            active_frames = [ 1 if idx > 5 else 0 for idx in active_frames ]    # (B*n)
            active_block = torch.tensor(active_frames, device=block_stft.device)   # (B*n)
            active_block_list.append(active_block)

        vad_block = torch.stack(active_block_list).reshape(num_spk, B, -1).permute(1, 0, 2)    # (B, num_spk, n)

        return ibRTF, vad_block     # (B, 2(C-1), F, n), (B, num_spk, n)