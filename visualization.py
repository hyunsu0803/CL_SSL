import sys, os
import util
import torch
import numpy as np
import random
import importlib
from tqdm import tqdm
from dataloader.wrap_dataload import Synth_dataload, Real_dataload
import matplotlib.pyplot as plt


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
        
    
    def permute_n_augment(self, mixed, vad, speech_azi):
        
        mixed = mixed.reshape(-1, mixed.shape[-2], mixed.shape[-1])
        vad = vad.reshape(-1, vad.shape[-2], vad.shape[-1])
        speech_azi = speech_azi.reshape(-1, speech_azi.shape[-1])
            
        perm = torch.randperm(mixed.size(0))  # 무작위 인덱스 생성 (size : 256)
        
        mixed = mixed.index_select(0, perm)  
        vad = vad.index_select(0, perm)  
        speech_azi = speech_azi.index_select(0, perm)
        
        return mixed, vad, speech_azi
        
    
    def run(self, ):
      
        size = len(os.listdir(self.args['hyparam']['model_dir']))
        self.test(0)


    def test(self, epoch):
        self.model.eval()

        for room_type in self.args['hyparam']['result_folder']['room_type']:
            room_type=str(room_type)    
            self.dataloader.test_loader.dataset.room_type=str(room_type)

            with torch.no_grad():
                
                # mixed : (1, 4, 4, 64000)
                # speech_azi : (1, 4, 1)
                # num_spk : (1)
                # vad : (1, 4, 1, 64000)
                for iter_num, (mixed, vad, speech_azi) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader))):
                    if iter_num % 10 != 0:
                        continue
                    mixed, vad, speech_azi = self.permute_n_augment(mixed, vad, speech_azi)
                    
                    mixed=mixed.to(self.hyperparameter.device)
                    vad=vad.to(self.hyperparameter.device)
                    speech_azi=speech_azi.to(self.hyperparameter.device)
    

                    out = self.model(mixed, vad, speech_azi, iter_num, epoch=0, mic_type='miyungpa')

                    out=out.sigmoid().detach().cpu().numpy()        # (4, 2048)
                    out=out.reshape(out.shape[0], 64, 32)           # (4, 64, 32)
                    
                    speech_azi=speech_azi.cpu().numpy()             # (4, 1)
                    
                    
                    
                    
                    ## save as png
                    plt.figure()
                    plt.subplot(2,2,1)
                    plt.imshow(out[0], aspect='auto', vmin=0.0, vmax=1.0)
                    plt.subplot(2,2,2)
                    plt.imshow(out[1], aspect='auto', vmin=0.0, vmax=1.0)
                    plt.subplot(2,2,3)
                    plt.imshow(out[2], aspect='auto', vmin=0.0, vmax=1.0)
                    plt.subplot(2,2,4)
                    plt.imshow(out[3], aspect='auto', vmin=0.0, vmax=1.0)
                    
                    
                    os.makedirs('/root/clssl/results/pngs/', exist_ok=True)
                    plt.tight_layout()
                    plt.savefig('/root/clssl/results/pngs/' + str(speech_azi[0, 0])+ '.png', dpi=600)
                    plt.close()
            
                    
                    self.learner.memory_delete([mixed, vad, speech_azi, out])
                  
                # self.logger.save_output(epoch)
                self.logger.config()

            break




if __name__=='__main__':
    args=sys.argv[1:]

    args = ['model /root/clssl/SSL_src/models/Causal_CRN_SPL_target/model.yaml', 
            'dataloader /root/clssl/SSL_src/dataloader/data_loader.yaml', 
            'hyparam /root/clssl/SSL_src/hyparam/test.yaml', 
            'learner /root/clssl/SSL_src/hyparam/learner.yaml', 
            'logger /root/clssl/SSL_src/hyparam/logger.yaml']
    args=util.util.get_yaml_args(args)    
    
    t=Tester(args)
    
    t.run()