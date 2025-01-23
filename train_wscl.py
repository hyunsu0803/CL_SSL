import sys, os
import util
import torch
import numpy as np
import random
import importlib
import math
import wandb
from tqdm import tqdm
from dataloader.wrap_dataload import Train_dataload_for_scl, Synth_dataload, Real_dataload
import pandas as pd



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
        self.scaler = torch.cuda.amp.GradScaler()   # mixed precision

    def memory_delete(self, *args):
        for a in args:
            del a

    def model_select(self):
        model_name=self.args['model']['name']
        model_import='models.'+model_name+'.main'       # ./models/Causal_CRN~~/main.py

        
        model_dir=importlib.import_module(model_import)
        
        self.model=model_dir.get_model_for_scl(self.args['model']).to(self.device)
        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])   
        
    def model_select_for_finetune(self):
        model_name=self.args['model']['name']
        model_import='models.'+model_name+'.main'

        model_dir=importlib.import_module(model_import)
        
        self.model=model_dir.get_model_for_scl(self.args['model']).to(self.device)

        trained=torch.load(self.args['hyparam']['model'], map_location=self.device)     # only for infer
        self.model.load_state_dict(trained['model_state_dict'], )                       # only for infer
        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])       


    def init_optimizer(self):

        self.args['learner']['optimizer']['config']['lr'] = 1.0e-4
        
        a=importlib.import_module('torch.optim')
        assert hasattr(a, self.args['learner']['optimizer']['type']), "optimizer {} is not in {}".format(self.args['learner']['optimizer']['type'], 'torch')
        a=getattr(a, self.args['learner']['optimizer']['type'])
     
        self.optimizer=a(self.model.parameters(), **self.args['learner']['optimizer']['config'])
        self.gradient_clip=self.args['learner']['optimizer']['gradient_clip']

        
    def init_optimzer_scheduler(self, ):

        self.args['learner']['optimizer_scheduler']['config']['min_lr'] = 9.0e-5

        a=importlib.import_module('torch.optim.lr_scheduler')
        assert hasattr(a, self.args['learner']['optimizer_scheduler']['type']), "optimizer scheduler {} is not in {}".format(self.args['learner']['optimizer']['type'], 'torch')
        a=getattr(a, self.args['learner']['optimizer_scheduler']['type'])

        self.optimizer_scheduler=a(self.optimizer, **self.args['learner']['optimizer_scheduler']['config'])



    def init_loss_func(self):

        from loss.scl_loss import Weighted_SupConLoss
        
        self.loss_func=Weighted_SupConLoss()

        self.loss_train_map_num=self.args['learner']['loss']['option']['train_map_num']     # [0, 1, 2]
        self.loss_weight=self.args['learner']['loss']['option']['each_layer_weight']

        if self.args['learner']['loss']['optimize_method']=='min':
            self.best_val_loss=math.inf
            self.best_train_loss=math.inf
        else:
            self.best_val_loss=-math.inf
            self.best_train_loss=-math.inf


    def train_update(self, output, labels):
         
        with torch.cuda.amp.autocast():
            loss_mean = self.loss_func(output, labels)


        if torch.isnan(loss_mean):
            print('nan occured')
            self.optimizer.zero_grad()
            return loss_mean

        
        self.scaler.scale(loss_mean).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        

        return loss_mean


    def test_update(self, output, labels):
        
        with torch.cuda.amp.autocast():
            loss_mean = self.loss_func(output, labels)
        

        if torch.isnan(loss_mean):
            print('nan occured')
            self.optimizer.zero_grad()
            return loss_mean

        return loss_mean


    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']
        self.model_select()     # set self.model
        # self.model_select_for_finetune()
        self.init_optimizer()
        self.init_optimzer_scheduler()
        self.init_loss_func()
        return self.args


class Logger_config():
    def __init__(self, args) -> None:
        self.args=args
        self.csv=dict()
        self.csv['train_epoch_loss']=[]
        self.csv['train_best_loss']=[]
        self.csv['test_epoch_loss']=[]
        self.csv['test_best_loss']=[]

        self.csv_dir=self.args['logger']['save_csv']
        self.model_save_dir=self.args['logger']['model_save_dir']
        self.png_dir=self.args['logger']['png_dir']

        if self.args['logger']['optimize_method']=='min':
            self.best_test_loss=math.inf
            self.best_train_loss=math.inf
        else:
            self.best_test_loss=-math.inf
            self.best_train_loss=-math.inf

    def train_iter_log(self, loss):
        try:
            wandb.log({'train_iter_loss':loss})
        except:
            None
        self.epoch_train_loss.append(loss.cpu().detach().item())

       
    def train_epoch_log(self):
        loss_mean=np.array(self.epoch_train_loss).mean()

        self.csv['train_epoch_loss'].append(loss_mean)

        if self.best_train_loss > loss_mean:
            self.best_train_loss = loss_mean 

        try:
            wandb.log({'train_epoch_loss':loss_mean})
            wandb.log({'train_best_loss':self.best_train_loss})
        except:
            None

        self.csv['train_best_loss'].append(self.best_train_loss)


    def test_iter_log(self, loss):
        try:
            wandb.log({'test_iter_loss':loss})
        except:
            None
        self.epoch_test_loss.append(loss.cpu().detach().item())


    def test_epoch_log(self, optimizer_scheduler):
        loss_mean=np.array(self.epoch_test_loss).mean()
        self.csv['test_epoch_loss'].append(loss_mean)

        self.model_save=False
        if self.best_test_loss > loss_mean:
            self.model_save=True
            self.best_test_loss = loss_mean 
        try:
            wandb.log({'test_epoch_loss':loss_mean})
            wandb.log({'test_best_loss':self.best_test_loss})
        except:
            None
        self.csv['test_best_loss'].append(self.best_test_loss)

        optimizer_scheduler.step(loss_mean)
        

    def epoch_init(self,):
        self.epoch_train_loss=[]
        self.epoch_test_loss=[]
    

    def epoch_finish(self, epoch, model, optimizer):
    
        os.makedirs(os.path.dirname(self.csv_dir), exist_ok=True)
        pd.DataFrame(self.csv).to_csv(self.csv_dir)

        checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }

        os.makedirs(os.path.dirname(self.model_save_dir + "best_model.tar"), exist_ok=True)
        if self.model_save:
            os.makedirs(os.path.dirname(self.model_save_dir + "best_model.tar"), exist_ok=True)
            torch.save(checkpoint, self.model_save_dir + "best_model.tar")
            print("new best model\n")
        torch.save(checkpoint,  self.model_save_dir + "{}_model.tar".format(epoch))

        
        util.util.draw_result_pic(self.png_dir, epoch, self.csv['train_epoch_loss'],  self.csv['test_epoch_loss'])


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
        self.args['dataloader']['train']['dataloader_dict']['num_workers'] = 4
        self.args['dataloader']['val']['loader']['dataloader_dict']['batch_size'] = 64
        self.args['dataloader']['val']['loader']['dataloader_dict']['num_workers'] = 4
        self.args['dataloader']['val']['loader']['pkl_dir'] = '/root/clssl/SSL_src/prepared/pkl/scl/'
        
        self.train_loader=Train_dataload_for_scl(self.args['dataloader']['train'], self.args['hyparam']['randomseed'])
        self.val_loader=Synth_dataload(self.args['dataloader']['val']['loader'])
        
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
            
           
    def permute_n_augment(self, mixed, vad, speech_azi):
        
        mixed = mixed.reshape(-1, mixed.shape[-2], mixed.shape[-1])
        vad = vad.reshape(-1, vad.shape[-2], vad.shape[-1])
        speech_azi = speech_azi.reshape(-1, speech_azi.shape[-1])
            
        perm = torch.randperm(mixed.size(0)) # (size : 256)
        
        mixed = mixed.index_select(0, perm)  
        vad = vad.index_select(0, perm)  
        speech_azi = speech_azi.index_select(0, perm)
        
        return mixed, vad, speech_azi
    

    def train(self, epoch):

        self.model.train()

        torch.cuda.empty_cache()
        
        # with torch.cuda.amp.autocast():
            
        self.n_room = 8
        self.dataloader.train_loader.dataset.random_room_speech_select(self.n_room)
        for iter_num, (mixed, vad, speech_azi, speech_ele,_) in enumerate(tqdm(self.dataloader.train_loader, desc='Train {}'.format(epoch), total=len(self.dataloader.train_loader), )):
            # mixed : [64, 8, 4, 64000]
            # vad : [64, 8, 1, 64000]
            # speech_azi : [64, 8, 1]
            mixed, vad, speech_azi = self.permute_n_augment(mixed, vad, speech_azi)
            # mixed : [512, 4, 64000]   
            # vad : [512, 1, 64000]
            # speech_azi : [512, 1]
            
            mixed=mixed.to(self.hyperparameter.device)
            vad=vad.to(self.hyperparameter.device)
            speech_azi=speech_azi.to(self.hyperparameter.device)
            
                
            out, embedding, vad_frame = self.model(mixed, vad)
            
            loss = self.learner.train_update(out, speech_azi)
                

            self.logger.train_iter_log(loss)
            self.learner.memory_delete([mixed, vad, speech_azi, out, loss, embedding, vad_frame])
            
            self.dataloader.train_loader.dataset.random_room_speech_select(self.n_room)
        
        
        self.logger.train_epoch_log()

 
    def validation(self, epoch):
        self.model.eval()
        
        torch.cuda.empty_cache()
        
        # with torch.cuda.amp.autocast():
        with torch.no_grad():
            
            # mixed : (16, 4, 64000)
            # speech_azi : (16, 1)
            # num_spk : (16)
            for iter_num, (mixed, vad, speech_azi) in enumerate(tqdm(self.dataloader.val_loader, desc='Test', total=len(self.dataloader.val_loader), )):
                
                mixed, vad, speech_azi = self.permute_n_augment(mixed, vad, speech_azi)
                
                mixed=mixed.to(self.hyperparameter.device)
                vad=vad.to(self.hyperparameter.device)
                speech_azi=speech_azi.to(self.hyperparameter.device)

                
                
                out, embedding, vad_frame = self.model(mixed, vad)
                
                loss=self.learner.test_update(out, speech_azi)
                    
                    
                self.logger.test_iter_log(loss)
                self.learner.memory_delete([mixed, vad, speech_azi, out, loss, embedding, vad_frame])
             
            self.logger.test_epoch_log(self.optimizer_scheduler)
            

if __name__=='__main__':
    args=sys.argv[1:]
    
    args = ['model /root/clssl/SSL_src/models/Causal_CRN_SPL_target/model_scl.yaml', 
            'dataloader /root/clssl/SSL_src/dataloader/data_loader.yaml', 
            'hyparam /root/clssl/SSL_src/hyparam/train.yaml', 
            'learner /root/clssl/SSL_src/hyparam/learner.yaml', 
            'logger /root/clssl/SSL_src/hyparam/logger.yaml']
    
    args=util.util.get_yaml_args(args)
    t=Trainer(args)
    t.run()