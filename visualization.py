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
        
        self.model=model_dir.get_model_for_scl(self.args['model']).to(self.device)

        self.args['hyparam']['model_for_finetune'] = "./results_03051800_scl/model_checkpoint/best_model.tar"

        if self.args['hyparam']['finetune']:
            trained=torch.load(self.args['hyparam']['model_for_finetune'], map_location=self.device)     
            self.model.load_state_dict(trained['model_state_dict'], )                

        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])   
        

    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']

        self.model_select()
        
        return self.args

   
class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args
        
    def config(self):

        self.args['dataloader']['train']['dataloader_dict']['batch_size'] = 1
        self.args['dataloader']['train']['dataloader_dict']['num_workers'] = 1
        self.args['dataloader']['val']['loader']['dataloader_dict']['batch_size'] = 1
        self.args['dataloader']['val']['loader']['dataloader_dict']['num_workers'] = 1
        self.args['dataloader']['val']['loader']['pkl_dir'] = './SSL_src/prepared/pkl/scl/'
        
        self.test_loader=Synth_dataload(self.args['dataloader']['val']['loader'])
        
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


    
    def run(self, ):
      
        self.test(0)


    def test(self, epoch):
        self.model.eval()

        for room_type in self.args['hyparam']['result_folder']['room_type']:
            room_type=str(room_type)    
            self.dataloader.test_loader.dataset.room_type=str(room_type)

            with torch.no_grad():

                for iter_num, (mixed, vad, speech_azi, white_snr, rt60) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader))):

                    mixed=mixed.to(self.hyperparameter.device)
                    vad=vad.to(self.hyperparameter.device)
                    speech_azi=speech_azi.to(self.hyperparameter.device)

                    out, embedding, speech_azi = self.model(mixed, vad, speech_azi)

                    speech_azi=speech_azi.cpu().numpy()
                    embedding=embedding.cpu().numpy()


                    selected_classes = np.arange(0, 31, 3)  # 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30


                    

            break




if __name__=='__main__':
    args=sys.argv[1:]


    args = ['model ./SSL_src/models/Causal_CRN_SPL_target/model_scl.yaml',
            'dataloader ./SSL_src/dataloader/data_loader.yaml', 
            'hyparam ./SSL_src/hyparam/test.yaml', 
            'learner ./SSL_src/hyparam/learner.yaml', 
            'logger ./SSL_src/hyparam/logger.yaml']
    args=util.util.get_yaml_args(args)    
    
    t=Tester(args)
    
    t.run()