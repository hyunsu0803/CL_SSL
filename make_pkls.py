import sys
import util
import torch
import numpy as np
import random
from tqdm import tqdm
from dataloader.wrap_dataload import Speech_datamake_for_scl, Speech_datamake_for_doa


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
            print("device : cuda")
            torch.cuda.manual_seed(self.args['hyparam']['randomseed'])

            device_primary_num=self.args['hyparam']['GPGPU']['device_ids'][0]
            device= 'cuda'+':'+str(device_primary_num)
        else:
            device= 'cpu'
            print("device : cpu")
            
        self.args['hyparam']['GPGPU']['device'] = device
        
        return device
    
    def set_on(self):
        self.set_torch_method()
        self.device=self.randomseed_init()
       
        return self.args


class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args
        
    def config(self):
        # self.val_maker = Speech_datamake_for_doa(self.args['dataloader']['val']['maker'])
        self.val_maker = Speech_datamake_for_scl(self.args['dataloader']['val']['maker'])
        # self.test_maker = Gunshot_datamake(self.args['dataloader']['test']['maker'])
      
        return self.args   
          

class Trainer():

    def __init__(self, args):

        self.args=args

        self.hyperparameter=Hyparam_set(self.args)
        self.args=self.hyperparameter.set_on()
     
        self.dataloader=Dataloader_config(self.args)
        self.args=self.dataloader.config()        

    
    def run(self, ):
        
        for i in range(10):
            self.validation_for_scl(0)
        
        # self.validation_for_doa(0)
        # self.test(0)
        

    def validation_for_scl(self, epoch):

        with torch.no_grad():
            n_room = 8
            self.dataloader.val_maker.dataset.random_room_speech_select(n_room)
            for iter_num, (mixed, vad, speech_azi, num_spk) in enumerate(tqdm(self.dataloader.val_maker , desc='Test', total=len(self.dataloader.val_maker), )):
                self.dataloader.val_maker.dataset.random_room_speech_select(n_room)
                
    
    def validation_for_doa(self, epoch):
            
            with torch.no_grad():
                for iter_num, (mixed, vad, speech_azi, num_spk) in enumerate(tqdm(self.dataloader.val_maker , desc='Test', total=len(self.dataloader.val_maker), )):
                    continue


    def test(self, epoch):
      
        with torch.no_grad():
            for iter_num, (mixed, vad, speech_azi, num_spk) in enumerate(tqdm(self.dataloader.test_maker , desc='Test', total=len(self.dataloader.test_maker), )):
            #     break
                continue
                            


if __name__=='__main__':
    args=sys.argv[1:]
    
    args = ['dataloader /root/hyunsoo/clssl/SSL_src/dataloader/data_loader.yaml', 
            'hyparam /root/hyunsoo/clssl/SSL_src/hyparam/train.yaml']
    
    args=util.util.get_yaml_args(args)
    t=Trainer(args)
    t.run()