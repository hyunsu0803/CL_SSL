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
        
        self.args['model']['CRN']['input_cnn_channel'] = 1
        self.model=model_dir.get_model_for_doa(self.args['model'], self.args['model_scl'], self.args['hyparam']).to(self.device)

        if self.args['hyparam']['finetune']:
            trained=torch.load(self.args['hyparam']['model_for_finetune'], map_location=self.device)     
            self.model.load_state_dict(trained['model_state_dict'], )                

        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])       
  

    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']

        self.model_select()
        
        return self.args


class Logger_config():
    def __init__(self, args) -> None:
        self.args=args
        self.result_folder=self.args['hyparam']['result_folder']
        

    def save_output(self, DB_type):
        try:
            now_dict=self.save_config_dict[DB_type]
        except:
            now_dict=self.save_config_dict[int(DB_type)]
            DB_type=int(DB_type)
        
        with open(self.result_folder['inference_folder']+'/'+DB_type+'/result.txt', 'w') as f:
            f.write('\nargmax_acc\n\n')

            k=(now_dict['argmax_acc']/now_dict['number_of_degrees'])
           
            for j in k:
                
            
               
                j=str(j)  
               
                f.write(j)
                f.write('\n')

            f.write('\nargmax_doa_error\n\n')
            k=(now_dict['argmax_doa_error']/now_dict['number_of_degrees'])
            for j in k:
                
                j=str(j)            
                f.write(j)
                f.write('\n')
            
            
          
            f.write('\n\n')

            f.write('\nsoftmax_acc\n\n')

            k=(now_dict['softmax_acc']/now_dict['number_of_degrees'])
       
            for j in k:
                
            
               
                j=str(j)  
               
                f.write(j)
                f.write('\n')

            f.write('\nsoftmax_doa_error\n\n')
            k=(now_dict['softmax_doa_error']/now_dict['number_of_degrees'])
            for j in k:
                
                j=str(j)            
                f.write(j)
                f.write('\n')
        
            f.write('\n\n')


            f.write('\nhalf_softmax_acc\n\n')

            k=(now_dict['half_softmax_acc']/now_dict['number_of_degrees'])
       
            for j in k:
                
            
               
                j=str(j)  
               
                f.write(j)
                f.write('\n')

            f.write('\nhalf_softmax_doa_error\n\n')
            k=(now_dict['half_softmax_doa_error']/now_dict['number_of_degrees'])
            for j in k:
                
                j=str(j)            
                f.write(j)
                f.write('\n')

    
    def error_update(self, DB_type, argmax_acc, softmax_acc, half_softmax_acc,argmax_doa_error, softmax_doa_error, half_softmax_doa_error,num):
        now_dict=self.save_config_dict[DB_type]
        
        now_dict['argmax_acc']+=argmax_acc
        now_dict['softmax_acc']+=softmax_acc
        now_dict['half_softmax_acc']+=half_softmax_acc
     
        now_dict['argmax_doa_error']+=argmax_doa_error
        now_dict['softmax_doa_error']+=softmax_doa_error
        now_dict['half_softmax_doa_error']+=half_softmax_doa_error

        now_dict['number_of_degrees']+=num
        self.save_config_dict[DB_type]=now_dict
  
    def config(self,):
        from copy import deepcopy

        self.save_config_dict=dict()

        metric_data={}
        metric_data['argmax_acc']=0
        metric_data['argmax_doa_error']=0
        

        metric_data['softmax_acc']=0
        metric_data['softmax_doa_error']=0

        metric_data['half_softmax_acc']=0
        metric_data['half_softmax_doa_error']=0



        metric_data['number_of_degrees']=0
        
        for room_type in self.result_folder['room_type']:
            os.makedirs(self.result_folder['inference_folder']+room_type, exist_ok=True)

            
            self.save_config_dict[room_type]=deepcopy(metric_data)

   

        return self.args
   
class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args

    
    def config(self):
        # self.test_loader = Synth_dataload(self.args['dataloader']['test']['loader'])
        self.test_loader = Real_dataload(self.args['dataloader']['test']['loader'])
       
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

    
    def run(self, ):
      
        self.test_RD(0)

    
    def plot_out_target(self, out, target, pkl_idx):

        ## save as png
        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(out, aspect='auto', vmin=0.0, vmax=1.0, interpolation='nearest')
        plt.xlabel('Time frame')
        plt.ylabel('Source angle')
        plt.title('Estimated DOA spatial spectrum')
        plt.subplot(2,1,2)
        plt.imshow(target, aspect='auto', vmin=0.0, vmax=1.0, interpolation='nearest')
        plt.xlabel('Time frame')
        plt.ylabel('Source angle')
        plt.title('Target DOA spatial spectrum')
        os.makedirs(self.logger.result_folder['inference_folder'] + '/pngs/' + pkl_idx.split('/')[-2] + '/', exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.logger.result_folder['inference_folder'] + '/pngs/' + pkl_idx.split('/')[-2] + '/' + pkl_idx.split('/')[-1].replace('.pkl', '.png'), dpi=600)
        plt.close()



    def test_RD(self, epoch):
        self.model.eval()

        for room_type in self.args['hyparam']['result_folder']['room_type']:
            room_type=str(room_type)    
            self.dataloader.test_loader.dataset.room_type=str(room_type)

            with torch.no_grad():

                for iter_num, (mixed, vad, pseudo_target, white_snr, coherent_snr, rt60) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader))):

                    mixed=mixed.to(self.hyperparameter.device)
                    vad=vad.to(self.hyperparameter.device)
                    pseudo_target=pseudo_target.to(self.hyperparameter.device)
                    temp_azi = torch.zeros(mixed.shape[0])

                    out, target, vad_block = self.model(mixed, vad, temp_azi)    # (B, 3, 360, n), (B, num_spk, n)


                    out=out.sigmoid().detach().cpu()  
                    # target=target.cpu()               
                    pseudo_target=pseudo_target.cpu()         # (B, 1)
                    # vad_block=vad_block.cpu()           # (B, num_spk, n)
                    
                    pkl_idx = self.dataloader.test_loader.dataset.pkl_list[iter_num]
                    # self.plot_out_target(out[0,1], pseudo_target[0,1], pkl_idx)

                    total_argmax_acc, total_softmax_acc, total_half_softmax_acc, \
                        total_argmax_doa_error, total_softmax_doa_error,total_half_softmax_doa_error, \
                            number_of_degrees_to_estimate=metric.mae.calc_mae_RD(out, pseudo_target.long(),\
                                                                        calc_layer=self.args['learner']['loss']['option']['train_map_num'],\
                                                                            acc_threshold=self.args['hyparam']['acc_threshold'],\
                                                                                local_maximum_distance=self.args['hyparam']['local_maximum_distance'])


                    self.logger.error_update(room_type, total_argmax_acc, total_softmax_acc,total_half_softmax_acc, 
                                             total_argmax_doa_error, total_softmax_doa_error, total_half_softmax_doa_error,
                                             number_of_degrees_to_estimate)

                    
                    self.learner.memory_delete([mixed, vad, pseudo_target, out, target, vad_block, total_argmax_acc, total_softmax_acc, total_half_softmax_acc,
                                             total_argmax_doa_error, total_softmax_doa_error, total_half_softmax_doa_error, number_of_degrees_to_estimate])
                  
                self.logger.save_output(room_type)

            break




if __name__=='__main__':
    args=sys.argv[1:]


    args = ['model ./SSL_src/models/Causal_CRN_SPL_target/model_doa.yaml', 
            'model_scl ./SSL_src/models/Causal_CRN_SPL_target/model_scl.yaml',
            'dataloader ./SSL_src/dataloader/data_loader.yaml', 
            'hyparam ./SSL_src/hyparam/test.yaml', 
            'learner ./SSL_src/hyparam/learner.yaml', 
            'logger ./SSL_src/hyparam/logger.yaml']
    args=util.util.get_yaml_args(args)    
    
    t=Tester(args)
    
    t.run()