import sys, os
import util
import torch
import numpy as np
import random
import importlib
import math
import wandb
from tqdm import tqdm
from dataloader.wrap_dataload import Train_dataload_for_doa, Synth_dataload, Real_dataload
import pandas as pd
import gc
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
            print("torch cuda is available")
            
            torch.cuda.manual_seed(self.args['hyparam']['randomseed'])

            device_primary_num=self.args['hyparam']['GPGPU']['device_ids'][0]
            device= 'cuda'+':'+str(device_primary_num)
        else:
            device= 'cpu'
            print("device : cpu")   
        
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
        
        self.model=model_dir.get_model_for_doa(self.args['model'], self.args['model_scl'], self.args['hyparam']).to(self.device)

        if self.args['hyparam']['finetune']:
            trained=torch.load(self.args['hyparam']['model_for_finetune'], map_location=self.device)     
            self.model.load_state_dict(trained['model_state_dict'], )                

        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])       


    def init_optimizer(self):

        self.args['learner']['optimizer']['config']['lr'] = 1.0e-3
        
        a=importlib.import_module('torch.optim')
        assert hasattr(a, self.args['learner']['optimizer']['type']), "optimizer {} is not in {}".format(self.args['learner']['optimizer']['type'], 'torch')
        a=getattr(a, self.args['learner']['optimizer']['type'])
     
        self.optimizer=a(self.model.parameters(), **self.args['learner']['optimizer']['config'])
        self.gradient_clip=self.args['learner']['optimizer']['gradient_clip']
    
        
    def init_optimzer_scheduler(self, ):

        self.args['learner']['optimizer_scheduler']['config']['min_lr'] = 1.0e-3

        a=importlib.import_module('torch.optim.lr_scheduler')
        assert hasattr(a, self.args['learner']['optimizer_scheduler']['type']), "optimizer scheduler {} is not in {}".format(self.args['learner']['optimizer']['type'], 'torch')
        a=getattr(a, self.args['learner']['optimizer_scheduler']['type'])

        self.optimizer_scheduler=a(self.optimizer, **self.args['learner']['optimizer_scheduler']['config'])



    def init_loss_func(self):

        
        self.loss_func=torch.nn.modules.loss.BCELoss(reduction='none')

        self.loss_train_map_num=self.args['learner']['loss']['option']['train_map_num']     # [0, 1, 2]
        self.loss_weight=self.args['learner']['loss']['option']['each_layer_weight']

        if self.args['learner']['loss']['optimize_method']=='min':
            self.best_val_loss=math.inf
            self.best_train_loss=math.inf
        else:
            self.best_val_loss=-math.inf
            self.best_train_loss=-math.inf


    def train_update(self, output, target):
            
        output=torch.sigmoid(output)
        loss = self.loss_func(output, target)

        loss_mean=loss.mean()
        
        if torch.isnan(loss).any():
            print('NaN occurred in loss')
            self.optimizer.zero_grad()
            return loss_mean

        if torch.isnan(loss_mean):
            print('nan occured')
            self.optimizer.zero_grad()
            return loss_mean

        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss_mean


    def test_update(self, output, target):
        
        output=torch.sigmoid(output)
        output=output[...,:target.shape[-1]]
        
        loss=self.loss_func(output, target)

        loss_mean=loss.mean()
        
        if torch.isnan(loss_mean):
            print('nan occured')
            self.optimizer.zero_grad()
            return loss_mean

        return loss_mean


    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']
        self.model_select()     # set self.model
        self.init_optimizer()
        self.init_optimzer_scheduler()
        self.init_loss_func()
        return self.args


class Logger_config():
    def __init__(self, args) -> None:
        self.args=args

        self.log_csv=dict()

        self.log_csv['train_epoch_loss']=[]
        self.log_csv['train_best_loss']=[]
        self.log_csv['test_epoch_mae']=[]
        self.log_csv['test_best_mae']=[]
        self.log_csv['test_epoch_acc']=[]
        self.log_csv['test_best_acc']=[]

        self.log_csv_dir=self.args['logger']['loss_csv']
        self.model_save_dir=self.args['logger']['model_save_dir']
        self.log_png_dir=self.args['logger']['loss_png_dir']


        if self.args['logger']['optimize_method']=='min':
            self.best_test_loss = math.inf
            self.best_train_loss = math.inf
            self.best_test_acc = -math.inf
            self.best_train_acc = -math.inf
            self.best_test_mae = math.inf
            self.best_train_mae = math.inf
        else:
            self.best_test_loss = -math.inf
            self.best_train_loss = -math.inf
            self.best_test_acc = math.inf
            self.best_train_acc = math.inf
            self.best_test_mae = -math.inf
            self.best_train_mae = -math.inf


###############################################
# train log
###############################################

    def train_iter_loss_log(self, loss):
        try:
            wandb.log({'train_iter_loss':loss})
        except:
            None
        self.epoch_train_loss.append(loss.cpu().detach().item())

       
    def train_epoch_loss_log(self):
        loss_mean=np.array(self.epoch_train_loss).mean()

        self.log_csv['train_epoch_loss'].append(loss_mean)

        self.model_save_loss=False
        if self.best_train_loss > loss_mean:
            self.best_train_loss = loss_mean 
            self.model_save_loss = True

        try:
            wandb.log({'train_epoch_loss':loss_mean})
            wandb.log({'train_best_loss':self.best_train_loss})
        except:
            None

        self.log_csv['train_best_loss'].append(self.best_train_loss)


###############################################
# test log
###############################################

    def test_iter_metric_log(self, out, pseudo_target):
        
        total_argmax_acc, total_softmax_acc, total_half_softmax_acc, \
                        total_argmax_doa_error, total_softmax_doa_error,total_half_softmax_doa_error, \
                            number_of_degrees_to_estimate=metric.mae.calc_mae_RD(out, pseudo_target.long(),\
                                                                        calc_layer=self.args['learner']['loss']['option']['train_map_num'],\
                                                                            acc_threshold=self.args['hyparam']['acc_threshold'],\
                                                                                local_maximum_distance=self.args['hyparam']['local_maximum_distance'])
        now_dict=self.save_test_config_dict
        
        now_dict['softmax_acc']+=total_softmax_acc
        now_dict['softmax_doa_error']+=total_softmax_doa_error
        now_dict['number_of_degrees']+=number_of_degrees_to_estimate


    def test_epoch_metric_log(self,):

        now_dict=self.save_test_config_dict

        softmax_acc = now_dict['softmax_acc'] / now_dict['number_of_degrees']
        softmax_doa_error = now_dict['softmax_doa_error'] / now_dict['number_of_degrees']

        softmax_acc = softmax_acc[1]
        softmax_doa_error = softmax_doa_error[1]

        self.log_csv['test_epoch_acc'].append(softmax_acc)
        self.log_csv['test_epoch_mae'].append(softmax_doa_error)


        self.model_save_acc = False
        if self.best_test_acc < softmax_acc:
            self.best_test_acc = softmax_acc
            self.model_save_acc = True

        self.model_save_mae = False
        if self.best_test_mae > softmax_doa_error:
            self.best_test_mae = softmax_doa_error
            self.model_save_mae = True

        try:
            wandb.log({'test_epoch_acc':softmax_acc})
            wandb.log({'test_best_acc':self.best_test_acc})
            wandb.log({'test_epoch_mae':softmax_doa_error})
            wandb.log({'test_best_mae':self.best_test_mae})
        except:
            None

        self.log_csv['test_best_acc'].append(self.best_test_acc)
        self.log_csv['test_best_mae'].append(self.best_test_mae)
        


###############################################
# epoch init, finish
###############################################

    def epoch_init(self,):
        self.epoch_train_loss=[]

        self.save_test_config_dict={}
        self.save_test_config_dict['softmax_acc']=0
        self.save_test_config_dict['softmax_doa_error']=0
        self.save_test_config_dict['number_of_degrees']=0

    

    def epoch_finish(self, epoch, model, optimizer):
    
        os.makedirs(os.path.dirname(self.log_csv_dir), exist_ok=True)
        pd.DataFrame(self.log_csv).to_csv(self.log_csv_dir)

        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }


        if self.model_save_loss:
            os.makedirs(os.path.dirname(self.model_save_dir + "best_loss_model.tar"), exist_ok=True)
            torch.save(checkpoint, self.model_save_dir + "best_loss_model.tar")
            print("new best model - loss\n")

        if self.model_save_mae:
            os.makedirs(os.path.dirname(self.model_save_dir + "best_mae_model.tar"), exist_ok=True)
            torch.save(checkpoint, self.model_save_dir + "best_mae_model.tar")
            print("new best model - mae\n")

        if self.model_save_acc:
            os.makedirs(os.path.dirname(self.model_save_dir + "best_acc_model.tar"), exist_ok=True)
            torch.save(checkpoint, self.model_save_dir + "best_acc_model.tar")
            print("new best model - acc\n")


        torch.save(checkpoint,  self.model_save_dir + "last_model.tar".format(epoch))

        util.util.draw_metric_pic(self.log_png_dir, epoch, self.log_csv['train_epoch_loss'],  self.log_csv['test_epoch_mae'], self.log_csv['test_epoch_acc'])


    def wandb_config(self):
        if self.args['logger']['wandb']['wandb_ok']:
            wandb.init(**self.args['logger']['wandb']['init'])      
        return self.args  
  
    def config(self,):
        self.wandb_config()
        return self.args
        

class Dataloader_config():
    def __init__(self, args) -> None:
        self.args=args
        
    def config(self):

        self.args['dataloader']['train']['dataloader_dict']['batch_size'] = 64
        self.args['dataloader']['train']['dataloader_dict']['num_workers'] = 8
        self.args['dataloader']['val']['loader']['dataloader_dict']['batch_size'] = 1
        self.args['dataloader']['val']['loader']['dataloader_dict']['num_workers'] = 8
        self.args['dataloader']['val']['loader']['pkl_dir'] = './SSL_src/prepared/pkl/doa/'
        
        self.train_loader=Train_dataload_for_doa(self.args['dataloader']['train'], self.args['hyparam']['randomseed'])
        self.val_loader=Real_dataload(self.args['dataloader']['val']['loader'])
        
        return self.args   
        
        
        

class Trainer():

    def __init__(self, args):

        self.args=args

        self.hyperparameter=Hyparam_set(self.args)
        self.args=self.hyperparameter.set_on()
     

        self.learner=Learner_config(self.args)
        self.args=self.learner.config()       

        self.model=self.learner.model
        self.optimizer=self.learner.optimizer
        self.optimizer_scheduler=self.learner.optimizer_scheduler

        self.dataloader=Dataloader_config(self.args)
        self.args=self.dataloader.config()

        self.logger=Logger_config(self.args)
        self.args=self.logger.config()

    
    def run(self, ):
      
        for epoch in range(self.args['hyparam']['resume_epoch'], self.args['hyparam']['last_epoch']):

            self.logger.epoch_init()
            
            self.train(epoch)
            self.validation(epoch)           
            
            self.logger.epoch_finish(epoch, self.model, self.optimizer)

            gc.collect()
            torch.cuda.empty_cache()

        print('\n*** Training is finished ***\n')
        self.learner.memory_delete([self.dataloader])
    

    def train(self, epoch):

        self.model.train()

        torch.cuda.empty_cache()
        
        for iter_num, (mixed, vad, speech_azi, speech_ele, white_snr, coherent_snr, rt60) in enumerate(tqdm(self.dataloader.train_loader, desc='Train {}'.format(epoch), total=len(self.dataloader.train_loader), )):
                
            mixed=mixed[0].to(self.hyperparameter.device)
            vad=vad.to(self.hyperparameter.device)
            speech_azi=speech_azi.to(self.hyperparameter.device)
            
            
            out, target, vad_block = self.model(mixed, vad, speech_azi)
            
            loss = self.learner.train_update(out, target)
                

            self.logger.train_iter_loss_log(loss)

            self.learner.memory_delete([mixed, vad, speech_azi, speech_ele, white_snr, coherent_snr, rt60, out, loss, target])
            gc.collect()
        
        
        self.logger.train_epoch_loss_log()

 
    def validation(self, epoch):

        self.model.eval()

        torch.cuda.empty_cache()
        
        
        with torch.no_grad():
            
            for iter_num, (mixed, vad, pseudo_target, white_snr, coherent_snr, rt60) in enumerate(tqdm(self.dataloader.val_loader, desc='Test', total=len(self.dataloader.val_loader), )):
                
                
                mixed=mixed.to(self.hyperparameter.device)
                vad=vad.to(self.hyperparameter.device)
                pseudo_target=pseudo_target.to(self.hyperparameter.device)
                temp_azi=torch.zeros(mixed.shape[0])

                
                out, target, vad_block = self.model(mixed, vad, temp_azi)

                out=out.sigmoid().detach().cpu()  
                target=target.cpu()               
                pseudo_target=pseudo_target.cpu()         # (B, 1)

                
                loss=self.learner.test_update(out, pseudo_target)
                    
                    
                self.logger.test_iter_metric_log(out, pseudo_target)

                self.learner.memory_delete([mixed, vad, pseudo_target, out, loss, target, white_snr, coherent_snr, rt60])
                gc.collect()
             
            
            self.logger.test_epoch_metric_log()
            


if __name__=='__main__':
    args=sys.argv[1:]
    
    args = ['model ./SSL_src/models/Causal_CRN_SPL_target/model_doa.yaml', 
            'model_scl ./SSL_src/models/Causal_CRN_SPL_target/model_scl.yaml',
            'dataloader ./SSL_src/dataloader/data_loader.yaml', 
            'hyparam ./SSL_src/hyparam/train.yaml', 
            'learner ./SSL_src/hyparam/learner.yaml', 
            'logger ./SSL_src/hyparam/logger.yaml']
    
    args=util.util.get_yaml_args(args)
    t=Trainer(args)
    t.run()