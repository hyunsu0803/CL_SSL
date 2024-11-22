import sys, os
import util
import torch
import numpy as np
import random
import importlib
from tqdm import tqdm
from dataloader.wrap_dataload import Synth_dataload, Real_dataload
import matplotlib.pyplot as plt
import metric
import soundfile as sf
from scipy.signal import spectrogram, resample

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

from multiprocessing import Pool
import tempfile
import imageio
from PIL import Image
from io import BytesIO


class Hyparam_set():
    
    def __init__(self, args):
        self.args=args

    def set_torch_method(self,):
        try:
            torch.multiprocessing.set_start_method(self.args['hyparam']['torch_start_method'], force=False) # spawn
        except:
            torch.multiprocessing.set_start_method(self.args['hyparam']['torch_start_method'], force=True) # spawn
        

    def randomseed_init(self,):
        np.random.seed(self.args['hyparam']['randomseed'])
        random.seed(self.args['hyparam']['randomseed'])
        torch.manual_seed(self.args['hyparam']['randomseed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args['hyparam']['randomseed'])

            device_primary_num=self.args['hyparam']['GPGPU']['device_ids'][0]
            device= 'cuda'+':'+str(device_primary_num)
        else:
            device= 'cpu'
        self.args['hyparam']['GPGPU']['device']=device

        return device
    
    def set_on(self):
        self.set_torch_method()
        self.device=self.randomseed_init()
       
        return self.args


class Learner_config():
    def __init__(self, args) -> None:
        self.args=args
    
    def memory_delete(self, *args):
        for a in args:
            del a

    def model_select(self):
        model_name=self.args['model']['name']
        model_import='models.'+model_name+'.main'

        model_dir=importlib.import_module(model_import)
        
        self.model=model_dir.get_model(self.args['model']).to(self.device)

        trained=torch.load(self.args['hyparam']['model'], map_location=self.device)     # only for infer
        self.model.load_state_dict(trained['model_state_dict'], )                       # only for infer
        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])       
        
        
    def model_select_for_evaluation(self, epoch):
        model_name=self.args['model']['name']
        model_import='models.'+model_name+'.main'

        model_dir=importlib.import_module(model_import)
        
        self.model=model_dir.get_model(self.args['model']).to(self.device)

        trained=torch.load(self.args['hyparam']['model_dir']+f'{epoch}_model.tar', map_location=self.device)
        self.model.load_state_dict(trained['model_state_dict'], )                       # only for infer
        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])
        

    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']
        # self.model_select_for_evaluation(0)
        self.model_select()
        # self.init_loss_func()
        
        return self.args


class Logger_config():
    def __init__(self, args) -> None:
        self.args=args
        self.result_folder=self.args['hyparam']['result_folder']
        self.room_type=self.result_folder['room_type']
        

    def save_output(self, epoch):
        
        squared_error_sum = self.save_config_dict['squared_error_sum']
        number_of_degrees = self.save_config_dict['number_of_degrees']
        rmsae = (squared_error_sum/number_of_degrees)**0.5
        print(f"총성 방향 추정 RMSAE : {rmsae:.2f}\n\n\n\n\n")
        os.makedirs(self.result_folder['inference_folder']+ self.room_type[0], exist_ok=True)
        with open(self.result_folder['inference_folder']+ self.room_type[0]+f'/{epoch}_result_{rmsae:.1f}.txt', 'w') as f:

            f.write('\nargmax_doa_error\n')
            f.write(str(rmsae)+'\n')

    
    def error_update(self, argmax_doa_error):
        
        self.save_config_dict['squared_error_sum'] += argmax_doa_error**2
        self.save_config_dict['number_of_degrees'] += 1

  
    def config(self,):
        
        self.save_config_dict=dict()

        self.save_config_dict['squared_error_sum']=0
        self.save_config_dict['number_of_degrees']=0

        return self.args

   
class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args

    
    def config(self):
        self.test_loader=Synth_dataload(self.args['dataloader']['test']['loader'])
        # self.test_loader=Real_dataload(self.args['dataloader']['test']['loader'])
       
        return self.args
    
    
class Utils_for_demo():
    def __init__(self) -> None:
        pass
    
    def polar_histogram(self, out, wave_path=None, alpha=1):
        # out (B, 3, 360)
        # out_sum = out[0,2].sum(dim=1)
        # out_sum = out[:,2].sum(dim=2)
        # out_sum = out_sum.sum(dim=0)
        out_sum = out[0,2]
        normed_out = out_sum / out_sum.max()
        estimation = out_sum.argmax()
        
        N = 360
        bottom = 0.08
        
        azi, ele, dist = self.wave_path_parsing(wave_path)

        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        width = (2*np.pi) / N 

        fig, ax = plt.subplots(subplot_kw={'polar': True})
        bars = ax.bar(theta, normed_out, width=width, zorder=3, bottom=bottom)
        

        # Use custom colors and opacity
        cmap = plt.cm.jet
        norm = Normalize(vmin=min(normed_out), vmax=max(normed_out))
        for r, bar in zip(normed_out, bars):
            # bar.set_facecolor(cmap(norm(r)))
            bar.set_facecolor(cmap(r))
            bar.set_alpha(alpha)

        # Add light purple bar at azi direction
        if azi != -1:
            azi_rad = np.deg2rad(azi)
            ax.bar(azi_rad, max(normed_out), width=width, color='purple', alpha=1, zorder=4)

            # Add "GT" text outside the polar plot at azi direction
            ax.text(azi_rad, max(normed_out) + 0.2, 'GT', color='purple', fontsize=15, ha='center', va='center')
            
            
            plt.title('Sniper azimuth : '+str(azi), pad=20)
            plt.subplots_adjust(top=0.8)
            
        # Remove the radial tick labels (2, 4, 6, 8, 10, 12)
        ax.set_yticklabels([])
        ax.grid(True, zorder=0)
        
        # Add colorbar
        sm = ScalarMappable(cmap=cmap)#, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Probability', labelpad=10)
        cbar.ax.set_position([0.85, 0.15, 0.05, 0.7])
        
        # Add a circle patch in the center of the plot
        circle = Circle((0.5, 0.5), 0.08, transform=ax.transAxes, color='white', alpha=1, zorder=10)
        ax.add_patch(circle)
        # Add answer text in the center of the plot
        ax.annotate(str(estimation.item())+'', xy=(0.5, 0.5), xycoords='axes fraction',
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, color='red', 
                zorder=11, alpha=alpha)
        
        return fig
    
        plt.savefig('/root/mydir/SSL_src/demo.png', dpi=300)
        plt.close()         
        exit()
        
    
    def wave_path_parsing(self, wave_path):
        if wave_path == None:
            return -1, -1, -1
        
        wave_path=wave_path.split('/')[-1].strip('.wav')
        wave_path=wave_path.split('_')
        
        azi, ele, dis = -1, -1, -1
        
        for i, item in enumerate(wave_path):
            if 'azi' in item:
                azi = int(wave_path[i+1])
            elif 'ele' in item:
                ele = int(wave_path[i+1])
            elif 'dis' in item:
                dis = int(wave_path[i+1])
        
        return azi, ele, dis     
        
    
    def figure_to_array(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        return np.array(img)
    
    
    def get_gun_fired_frames(self, audio, fps):
        # audio = audio.cpu().numpy()
        
        max_val = audio.max()
        audio = audio / max_val
        
        fired = (audio > 0.6).astype(np.int32)

        fired_frame = np.where(fired == 1)[0] / 44100 * fps + 1
        fired_frame = list(set([int(f) for f in fired_frame]))
        
        
        for idx, frame in enumerate(fired_frame):
            if idx == 0:
                continue
            if frame - fired_frame[idx-1] < 2:
                fired_frame.pop(idx)

        
        return fired_frame
    
    def resample(self, mixed, vad, original_fs, new_fs=44100):
        # mixed : (4, 64000)
        # vad : (1, 64000)
        mixed = mixed.astype(np.float32)
        vad = vad.astype(np.float32)
        
        is_dim1 = False
        if mixed.ndim == 1:
            mixed = mixed[np.newaxis, :]
            vad = vad[np.newaxis, :]
            is_dim1 = True
            
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
        
        if is_dim1:
            mixed = mixed.squeeze(0)
            vad = vad.squeeze(0)

        return mixed, vad   
        

class Tester():

    def __init__(self, args):

        self.args=args

        self.hyperparameter=Hyparam_set(self.args)
        self.args=self.hyperparameter.set_on()

        self.learner=Learner_config(self.args)
        self.args=self.learner.config()
        self.model=self.learner.model


        self.dataloader=Dataloader_config(self.args)
        self.args=self.dataloader.config()

        self.logger=Logger_config(self.args)
        self.args=self.logger.config()
        
        self.utils=Utils_for_demo()
        
    
    def run(self, ):
      
        size = len(os.listdir(self.args['hyparam']['model_dir']))
        self.test(0)
        # for epoch in range(0, size-1):
        # # for epoch in [50]:
        #     print('epoch :', epoch)
        #     self.learner.model_select_for_evaluation(epoch)
        #     self.model=self.learner.model
        #     self.test(epoch)


    def demo(self, wave_path):
        self.model.eval()
        
        with torch.no_grad():
            
            mixed, original_fs = sf.read(wave_path, dtype='float32')
            print("Audio inserted")
            
            if mixed.ndim == 1:
                mixed = np.expand_dims(mixed, 0)
            elif mixed.shape[0] > mixed.shape[1]:
                mixed = mixed.T
                
            if mixed.shape[0] > 4:
                mixed = mixed[::2]
                mixed = mixed[:4]    
            
            vad = np.zeros_like(mixed)
            
            if original_fs != 44100:
                mixed, vad = self.utils.resample(mixed, vad, original_fs)
                print("audio resampled")
            original_mixed = mixed.copy()
                
            mixed=mixed.astype('float32')
            vad=vad.astype('float32')    
                
            samples = 8 * 44100
            if mixed.shape[1] < samples:
                    pad = samples - mixed.shape[1]
                    
                    mixed = np.pad(mixed, ((0,0), (0, pad)), mode='constant')
                    vad = np.pad(vad, ((0,0), (0, pad)), mode='constant')
            else:
                start = 0
                mixed = mixed[:, start:start+samples]
                vad = vad[:, start:start+samples]
            
            mixed = np.expand_dims(mixed, 0)
            vad = np.expand_dims(vad, 0)
            
            mixed=torch.from_numpy(mixed.astype(np.float32))
            vad=torch.from_numpy(vad.astype(np.float32))
            mixed=mixed.to(self.hyperparameter.device)
            vad=vad.to(self.hyperparameter.device)
            speech_azi=torch.tensor([0]).to(self.hyperparameter.device)
            iter_num=0



            # inference
            out, target, vad=self.model(mixed, vad, speech_azi, iter_num, epoch=0, mic_type='miyungpa')
            print("estimation done")
            
            out=out.sigmoid().detach().cpu()    # (B, 3, 360, 501)
            target=target.cpu()
            
        
            # make video
            print("making video")
            fps = 3
            empty_histogram = self.utils.polar_histogram(out, wave_path, 0)       # <class 'matplotlib.figure.Figure'>
            empty_histogram = self.utils.figure_to_array(empty_histogram)
            
            # making 2 seconds of block
            print("1...")            
            full_histogram = self.utils.polar_histogram(out, wave_path, 1)
            full_histogram = self.utils.figure_to_array(full_histogram)
            
            decrease = [self.utils.polar_histogram(out, wave_path, 0.9),
                        self.utils.polar_histogram(out, wave_path, 0.8),
                        self.utils.polar_histogram(out, wave_path, 0.6),
                        self.utils.polar_histogram(out, wave_path, 0.3)]
            decrease = [self.utils.figure_to_array(fig) for fig in decrease]
            
            two_sec = [full_histogram] * 2 + decrease
                
            
            # calculate number of frames
            print("2...")
            audio_duration = len(original_mixed[0]) / 44100                # in seconds
            num_video_frames = int(audio_duration * fps)           # 3 frames per second
            gun_fired = self.utils.get_gun_fired_frames(original_mixed[0], fps)  # frame numbers when gun fired                
            print("gun fired at", gun_fired)
            
            # make images
            print("3...")
            whole_frames = [empty_histogram] * num_video_frames
            for f in gun_fired:
                if f+6 < num_video_frames:
                    whole_frames[f:f+6] = two_sec
                else:
                    whole_frames[f:] = two_sec[:num_video_frames-f]


            # convert images to array
            print("converting...")
                            
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', dir='/root/mydir/', prefix='doa_1_', delete=False)
            
            print("Saving")
            imageio.mimsave(temp_file.name, whole_frames, fps=fps)
            
            print("SAVED")
            print(temp_file.name)
            
            return temp_file.name, (44100, mixed[0,0].cpu().numpy())


    def demo_old(self, wave_path):
        self.model.eval()
        
        with torch.no_grad():
            
            mixed, original_fs = sf.read(wave_path, dtype='float32')
            print("Audio inserted")
            
            if mixed.ndim == 1:
                mixed = np.expand_dims(mixed, 0)
            elif mixed.shape[0] > mixed.shape[1]:
                mixed = mixed.T
                
            if mixed.shape[0] > 4:
                mixed = mixed[::2]
                mixed = mixed[:4]    
                
            vad = np.zeros_like(mixed)
            
            if original_fs != 44100:
                mixed, vad = self.utils.resample(mixed, vad, original_fs)
                print("audio resampled")
            
            mixed = np.expand_dims(mixed, 0)
            vad = np.expand_dims(vad, 0)
            
            mixed=torch.from_numpy(mixed.astype(np.float32))
            vad=torch.from_numpy(vad.astype(np.float32))
            mixed=mixed.to(self.hyperparameter.device)
            vad=vad.to(self.hyperparameter.device)
            speech_azi=torch.tensor([0]).to(self.hyperparameter.device)
            iter_num=0

            # inference
            out, target, vad=self.model(mixed, vad, speech_azi, iter_num, epoch=0, mic_type='miyungpa')
            print("estimation done")
            
            out=out.sigmoid().detach().cpu()    # (B, 3, 360, 501)
            target=target.cpu()
            
            
            print("generating results")
            histogram = self.utils.polar_histogram(out, wave_path)
            print("done")
            
            return histogram 
        

    def test(self, epoch):
        self.model.eval()

        for room_type in self.args['hyparam']['result_folder']['room_type']:
            room_type=str(room_type)    
            self.dataloader.test_loader.dataset.room_type=str(room_type)

            with torch.no_grad():
                
                # mixed : (1, 4, 64000)
                # speech_azi : (1, 1)
                # num_spk : (1)
                # vad : (1, 1, 64000)
                for iter_num, (mixed, vad, speech_azi, num_spk, pkl_idx) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader))):
                    
                    mixed=mixed.to(self.hyperparameter.device)
                    vad=vad.to(self.hyperparameter.device)
                    speech_azi=speech_azi.to(self.hyperparameter.device)
    

                    out, target, vad=self.model(mixed, vad, speech_azi, iter_num, epoch=0, mic_type='miyungpa')

                    out=out.sigmoid().detach().cpu().numpy()    # (B, 3, 360, 501) for speech, (B, 3, 360) for gunshot                    
                    target=target.cpu().numpy()                 # (B, 3, 360, 501) for speech, (B, 3, 360) for gunshot
                    vad=vad.cpu().numpy()
                    speech_azi=speech_azi.cpu().numpy()         # (B, 1)
                    mixed=mixed.cpu().numpy()                   # (B, 4, 64000)

                    ans_azi_output = out[0,2].argmax()
                    ans_azi_target = target[0,2].argmax()
                    error = abs(ans_azi_output - ans_azi_target)
                    error = min(error, 360-error)
                    
                    pkl_idx=pkl_idx[0]
                    
                    # ### save as polar histogram
                    # duration = int(mixed.shape[2] / 44100 * 10) / 10 # in seconds
                    
                    # histogram = self.utils.polar_histogram(out)
                    # plt.title('Sniper azimuth : '+str(speech_azi.item()) + ' / Duration : ' + str(duration) + 's')  
                    # os.makedirs('/root/mydir/results/pngs/', exist_ok=True)
                    # plt.savefig('/root/mydir/results/pngs/' + pkl_idx.split('.')[0]+ '.png', dpi=300)
                    # plt.close()         
                    # # exit()
                    
                    
                    ### save as png
                    # plt.figure()
                    # plt.subplot(2,1,1)
                    # plt.imshow(out[0,2], aspect='auto', vmin=0.0, vmax=1.0)
                    # plt.xlabel('Time frame')
                    # plt.ylabel('Source angle')
                    # plt.title('Estimated DOA spatial spectrum')
                    # plt.subplot(2,1,2)
                    # plt.imshow(target[0,2], aspect='auto', vmin=0.0, vmax=1.0)
                    # plt.xlabel('Time frame')
                    # plt.ylabel('Source angle')
                    # plt.title('Target DOA spatial spectrum')
                    # os.makedirs('/root/mydir/results/pngs/', exist_ok=True)
                    # plt.tight_layout()
                    # plt.savefig('/root/mydir/results/pngs/' + pkl_idx.split('.')[0]+ '.png', dpi=600)
                    # plt.close()
                    
                    ### save as wav
                    # os.makedirs('/root/mydir/results/wavs/', exist_ok=True)
                    # sf.write('/root/mydir/results/wavs/' + pkl_idx.split('.')[0]+ '.wav', mixed[0,0].numpy(), 44100)
                    
                    ### save as spectrogram
                    # plt.figure()
                    # freq, times, sxx = spectrogram(mixed[0,0].numpy(), fs=44100, nperseg=1024, noverlap=512, nfft=1024)
                    # plt.pcolormesh(times, freq, 10*np.log10(sxx), shading='gouraud')
                    # plt.xlabel('Time (s)')
                    # plt.ylabel('Frequency (Hz)')
                    # plt.title('Mixed spectrogram')
                    # plt.tight_layout()
                    # os.makedirs('/root/mydir/results/spectrograms/', exist_ok=True)
                    # plt.savefig('/root/mydir/results/spectrograms/' + pkl_idx.split('.')[0]+ '.png')
                    # plt.close()
                  

                    # total_argmax_acc, \
                    # total_softmax_acc, \
                    # total_half_softmax_acc, \
                    # total_argmax_doa_error, \
                    # total_softmax_doa_error,\
                    # total_half_softmax_doa_error, \
                    # number_of_degrees_to_estimate   =   metric.mae.calc_rmsae(out, target, vad, num_spk, speech_azi,\
                    #     calc_layer=self.args['learner']['loss']['option']['train_map_num'],\
                    #         acc_threshold=self.args['hyparam']['acc_threshold'],\
                    #             local_maximum_distance=self.args['hyparam']['local_maximum_distance'])

                    # total_argmax_doa_error=(total_argmax_doa_error/number_of_degrees_to_estimate)**0.5
                    # self.logger.error_update(room_type, 
                    #                          total_argmax_acc, 
                    #                          total_softmax_acc,
                    #                          total_half_softmax_acc, 
                    #                          total_argmax_doa_error, 
                    #                          total_softmax_doa_error, 
                    #                          total_half_softmax_doa_error,
                    #                          number_of_degrees_to_estimate)

                    self.logger.error_update(error)
                    
                    self.learner.memory_delete([mixed, vad, speech_azi, out, target,])
                  
                self.logger.save_output(epoch)
                self.logger.config()

            break




if __name__=='__main__':
    args=sys.argv[1:]

    args = ['model /root/mydir/SSL_src/models/Causal_CRN_SPL_target/model.yaml', 
            'dataloader /root/mydir/SSL_src/dataloader/data_loader.yaml', 
            'hyparam /root/mydir/SSL_src/hyparam/test.yaml', 
            'learner /root/mydir/SSL_src/hyparam/learner.yaml', 
            'logger /root/mydir/SSL_src/hyparam/logger.yaml']
    args=util.util.get_yaml_args(args)    
    
    t=Tester(args)
    
    t.run()