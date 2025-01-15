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
        
        if self.args['hyparam']['SCL']:
            self.args['model']['CRN']['input_cnn_channel'] = 1
            self.model=model_dir.get_model_for_doa(self.args['model'], self.args['model_scl'], self.args['hyparam']).to(self.device)
            
        else:
            self.args['model']['CRN']['input_cnn_channel'] = 6
            self.model=model_dir.get_model_for_doa(self.args['model']).to(self.device)
            
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
        
        total_error_sum = self.save_config_dict['total_error_sum']
        number_of_degrees = self.save_config_dict['number_of_degrees']

        MAE = total_error_sum/number_of_degrees
        acc_180 = self.save_config_dict['acc_180']/number_of_degrees * 100
        acc_20 = self.save_config_dict['acc_20']/number_of_degrees  * 100
        acc_10 = self.save_config_dict['acc_10']/number_of_degrees  * 100
        acc_5 = self.save_config_dict['acc_5']/number_of_degrees    * 100
        acc_3 = self.save_config_dict['acc_3']/number_of_degrees    * 100
        acc_1 = self.save_config_dict['acc_1']/number_of_degrees    * 100

        print(f"MAE : {MAE:.2f}\n")
        print(f"acc_180 : {acc_180:.2f}\n")
        print(f"acc_20 : {acc_20:.2f}\n")
        print(f"acc_10 : {acc_10:.2f}\n")
        print(f"acc_5 : {acc_5:.2f}\n")
        print(f"acc_3 : {acc_3:.2f}\n")
        print(f"acc_1 : {acc_1:.2f}\n")

        os.makedirs(self.result_folder['inference_folder']+ self.room_type[0], exist_ok=True)
        with open(self.result_folder['inference_folder']+ self.room_type[0]+f'/{epoch}_result_{MAE:.1f}_acc.txt', 'w') as f:

            f.write('\nargmax_doa_error\n')
            f.write(str(MAE)+'\n\n')

            f.write('\nacc_180\n')
            f.write(str(acc_180)+'\n')
            f.write('\nacc_20\n')
            f.write(str(acc_20)+'\n')
            f.write('\nacc_10\n')
            f.write(str(acc_10)+'\n')
            f.write('\nacc_5\n')
            f.write(str(acc_5)+'\n')
            f.write('\nacc_3\n')
            f.write(str(acc_3)+'\n')
            f.write('\nacc_1\n')
            f.write(str(acc_1)+'\n')

    
    def error_update(self, argmax_doa_error):
        
        self.save_config_dict['total_error_sum'] += argmax_doa_error
        self.save_config_dict['number_of_degrees'] += 1

        if argmax_doa_error <= 180:
            self.save_config_dict['acc_180'] += 1
        if argmax_doa_error <= 20:
            self.save_config_dict['acc_20'] += 1
        if argmax_doa_error <= 10:
            self.save_config_dict['acc_10'] += 1
        if argmax_doa_error <= 5:
            self.save_config_dict['acc_5'] += 1
        if argmax_doa_error <= 3:
            self.save_config_dict['acc_3'] += 1
        if argmax_doa_error <= 1:
            self.save_config_dict['acc_1'] += 1

  
    def config(self,):
        
        self.save_config_dict=dict()

        self.save_config_dict['acc_1']=0
        self.save_config_dict['acc_3']=0
        self.save_config_dict['acc_5']=0
        self.save_config_dict['acc_10']=0
        self.save_config_dict['acc_20']=0
        self.save_config_dict['acc_180']=0
        self.save_config_dict['total_error_sum']=0
        self.save_config_dict['number_of_degrees']=0

        return self.args

   
class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args

    
    def config(self):
        self.test_loader=Synth_dataload(self.args['dataloader']['test']['loader'])
        # self.test_loader=Real_dataload(self.args['dataloader']['test']['loader'])
       
        return self.args
    

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
        
        # self.utils=Utils_for_demo()
        
    
    def run(self, ):
      
        self.test(0)


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
                for iter_num, (mixed, vad, speech_azi) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader))):
                    
                    mixed=mixed.to(self.hyperparameter.device)
                    vad=vad.to(self.hyperparameter.device)
                    speech_azi=speech_azi.to(self.hyperparameter.device)
    

                    out, target=self.model(mixed, vad, speech_azi, iter_num, epoch=0)

                    out=out.sigmoid().detach().cpu().numpy()    # (B, 3, 360, 501) for speech, (B, 3, 360) for gunshot                    
                    target=target.cpu().numpy()                 # (B, 3, 360, 501) for speech, (B, 3, 360) for gunshot
                    vad=vad.cpu().numpy()
                    speech_azi=speech_azi.cpu().numpy()         # (B, 1)
                    mixed=mixed.cpu().numpy()                   # (B, 4, 64000)


                    output_azi = out[0,2].argmax()
                    ans_azi = speech_azi[0,0]
                    
                    error = abs(output_azi - ans_azi)
                    error = min(error, 360-error)
                    
                    # pkl_idx=pkl_idx[0]
                    pkl_idx = self.dataloader.test_loader.dataset.pkl_list[iter_num]
                    
                    # ### save as polar histogram
                    # duration = int(mixed.shape[2] / 44100 * 10) / 10 # in seconds
                    
                    # histogram = self.utils.polar_histogram(out)
                    # plt.title('Sniper azimuth : '+str(speech_azi.item()) + ' / Duration : ' + str(duration) + 's')  
                    # os.makedirs('/root/mydir/results/pngs/', exist_ok=True)
                    # plt.savefig('/root/mydir/results/pngs/' + pkl_idx.split('.')[0]+ '.png', dpi=300)
                    # plt.close()         
                    # # exit()
                    
                    
                    # ## save as png
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
                    # os.makedirs('/root/clssl/results/pngs/', exist_ok=True)
                    # plt.tight_layout()
                    # plt.savefig('/root/clssl/results/pngs/' + pkl_idx.split('.')[0]+ '.png', dpi=600)
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

                    self.logger.error_update(error)
                    
                    self.learner.memory_delete([mixed, vad, speech_azi, out, target,])
                  
                self.logger.save_output(epoch)
                # self.logger.config()

            break




if __name__=='__main__':
    args=sys.argv[1:]

    args = ['model /root/clssl/SSL_src/models/Causal_CRN_SPL_target/model_doa.yaml', 
            'model_scl /root/clssl/SSL_src/models/Causal_CRN_SPL_target/model_scl.yaml',
            'dataloader /root/clssl/SSL_src/dataloader/data_loader.yaml', 
            'hyparam /root/clssl/SSL_src/hyparam/test.yaml', 
            'learner /root/clssl/SSL_src/hyparam/learner.yaml', 
            'logger /root/clssl/SSL_src/hyparam/logger.yaml']
    args=util.util.get_yaml_args(args)    
    
    t=Tester(args)
    
    t.run()