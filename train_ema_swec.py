import sys, os
import util
import torch
import numpy as np
import random
import importlib
import math
import wandb
from tqdm import tqdm
from dataloader.wrap_dataload import Train_dataload_for_scl
import pandas as pd
import gc




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
        
        self.model=model_dir.get_model_for_scl(self.args['model']).to(self.device)

        if self.args['hyparam']['finetune']:
            trained=torch.load(self.args['hyparam']['model_for_finetune'], map_location=self.device)     
            self.model.load_state_dict(trained['model_state_dict'], )                

        self.model=torch.nn.DataParallel(self.model, self.args['hyparam']['GPGPU']['device_ids'])   
        

    def init_optimizer(self):

        self.args['learner']['optimizer']['config']['lr'] = 1.0e-4
        
        a=importlib.import_module('torch.optim')
        assert hasattr(a, self.args['learner']['optimizer']['type']), "optimizer {} is not in {}".format(self.args['learner']['optimizer']['type'], 'torch')
        a=getattr(a, self.args['learner']['optimizer']['type'])
     
        self.optimizer=a(self.model.parameters(), **self.args['learner']['optimizer']['config'])
        self.gradient_clip=self.args['learner']['optimizer']['gradient_clip']

        
    def init_optimzer_scheduler(self, ):

        self.args['learner']['optimizer_scheduler']['config']['min_lr'] = 1.0e-4

        a=importlib.import_module('torch.optim.lr_scheduler')
        assert hasattr(a, self.args['learner']['optimizer_scheduler']['type']), "optimizer scheduler {} is not in {}".format(self.args['learner']['optimizer']['type'], 'torch')
        a=getattr(a, self.args['learner']['optimizer_scheduler']['type'])

        self.optimizer_scheduler=a(self.optimizer, **self.args['learner']['optimizer_scheduler']['config'])



    def init_loss_func(self):

        from loss.ema_scl_loss import EMA_Weighted_SupConLoss
        
        self.loss_func=EMA_Weighted_SupConLoss()

        self.loss_train_map_num=self.args['learner']['loss']['option']['train_map_num']     # [0, 1, 2]
        self.loss_weight=self.args['learner']['loss']['option']['each_layer_weight']

        if self.args['learner']['loss']['optimize_method']=='min':
            self.best_val_loss=math.inf
            self.best_train_loss=math.inf
        else:
            self.best_val_loss=-math.inf
            self.best_train_loss=-math.inf


    def train_update(self, output_s, output_t, labels, vad_block):
        # outputs : [(B, 128, n), (B, 128, n), (B, 128, n)]
        # labels : (B, 1)
        # vad_block : (B, 1, n)


        B, num_spk, n = vad_block.shape

        vad_block_flat = vad_block.reshape(-1)  # (B*n)
        labels = labels.repeat_interleave(n, dim=0)  # (B*n, 1)
        labels[vad_block_flat==0] = 360
        labels[labels == 360] = 1000



        losses = []

        for out_s, out_t, sigma in zip(output_s, output_t, self.sigma):

            out_s = out_s.permute(0, 2, 1)  # (B, n, 128)
            out_s = out_s.reshape(-1, out_s.shape[-1])  # (B*n, 128)

            out_t = out_t.permute(0, 2, 1)  # (B, n, 128)
            out_t = out_t.reshape(-1, out_t.shape[-1])  # (B*n, 128)
            
            
            loss_mean = self.loss_func(out_s, out_t, labels, sigma)


            if torch.isnan(loss_mean):
                print('nan occured')
                self.optimizer.zero_grad()
                return loss_mean

            losses.append(loss_mean)


        loss_mean = sum(losses) / len(losses)
            
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        

        return loss_mean


    def config(self):
        self.device=self.args['hyparam']['GPGPU']['device']
        self.model_select()     # set self.model
        self.init_optimizer()
        self.init_optimzer_scheduler()
        self.init_loss_func()
        self.sigma = [0, 3, 9]
        return self.args


class Logger_config():
    def __init__(self, args) -> None:
        self.args=args
        self.csv=dict()
        self.csv['train_epoch_loss']=[]
        self.csv['train_best_loss']=[]

        self.csv_dir=self.args['logger']['loss_csv']
        self.model_save_dir=self.args['logger']['model_save_dir']
        self.png_dir=self.args['logger']['loss_png_dir']

        if self.args['logger']['optimize_method']=='min':
            self.best_train_loss=math.inf
        else:
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

        self.model_save = False
        if self.best_train_loss > loss_mean:
            self.best_train_loss = loss_mean 
            self.model_save = True

        try:
            wandb.log({'train_epoch_loss':loss_mean})
            wandb.log({'train_best_loss':self.best_train_loss})
        except:
            None

        self.csv['train_best_loss'].append(self.best_train_loss)
        

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

        os.makedirs(os.path.dirname(self.model_save_dir + "last_model.tar"), exist_ok=True)
        torch.save(checkpoint,  self.model_save_dir + "last_model.tar")

        if epoch % 50 == 49:
            torch.save(checkpoint, self.model_save_dir + "epoch_{}.tar".format(epoch))
        
        util.util.draw_result_pic(self.png_dir, epoch, self.csv['train_epoch_loss'],  self.csv['train_epoch_loss'], 'Loss')


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
        
        self.train_loader=Train_dataload_for_scl(self.args['dataloader']['train'], self.args['hyparam']['randomseed'])
        
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

        self.stage0 = 0
        self.stage1 = 50
        self.stage2 = 100
        self.stage3 = 200
        self.stage4 = 400

    
    def run(self, ):
      
        for epoch in range(self.args['hyparam']['resume_epoch'], self.args['hyparam']['last_epoch']):

            self.logger.epoch_init()
            
            self.train(epoch)       
            
            self.logger.epoch_finish(epoch, self.model, self.optimizer)

            gc.collect()
            torch.cuda.empty_cache()
            
        print('\n*** Training is finished ***\n')
        self.learner.memory_delete([self.dataloader])
    

    def train(self, epoch):

        self.model.train()

        torch.cuda.empty_cache()


        # if epoch == self.stage0:

        #     self.n_room = 1
        #     self.dataloader.train_loader.dataset.random_room_speech_select(self.n_room)

        # if epoch >= self.stage1:

        #     if epoch < self.stage2:
        #         self.n_room = 1
        #     elif epoch < self.stage3:
        #         self.n_room = 0
        #     elif epoch < self.stage4:
        #         self.n_room = 8

        #     self.dataloader.train_loader.dataset.random_room_speech_select(self.n_room)

        
        self.n_room = 8
        self.dataloader.train_loader.dataset.random_room_speech_select(self.n_room)        

        for iter_num, (mixed, vad, speech_azi, speech_ele, white_snr, coherent_snr, rt60) in enumerate(tqdm(self.dataloader.train_loader, desc='Train {}'.format(epoch), total=len(self.dataloader.train_loader), )):
            gc.collect()

            mixed_s, mixed_t = mixed
            mixed_s = mixed_s.to(self.hyperparameter.device)
            mixed_t = mixed_t.to(self.hyperparameter.device)
            vad = vad.to(self.hyperparameter.device)
            speech_azi = speech_azi.to(self.hyperparameter.device)
            
            outputs, embedding, speech_azi, vad_block = self.model(mixed_s, mixed_t, vad, speech_azi) # (B, 128, n), (B, 256, n) (B, 1)
            
            output_s, output_t = outputs

            loss = self.learner.train_update(output_s, output_t, speech_azi, vad_block)
                

            self.logger.train_iter_log(loss)
            self.learner.memory_delete([mixed_s, mixed_t, vad, speech_azi, speech_ele, white_snr, coherent_snr, rt60, outputs, loss, embedding, vad_block])

            
            # if epoch >= self.stage2:
            #     self.dataloader.train_loader.dataset.random_room_speech_select(self.n_room)

            self.dataloader.train_loader.dataset.random_room_speech_select(self.n_room)
        
        
        self.logger.train_epoch_log()            
            
            

if __name__=='__main__':
    args=sys.argv[1:]
    
    args = ['model ./SSL_src/models/Causal_CRN_SPL_target/model_scl.yaml', 
            'dataloader ./SSL_src/dataloader/data_loader.yaml', 
            'hyparam ./SSL_src/hyparam/train.yaml', 
            'learner ./SSL_src/hyparam/learner.yaml', 
            'logger ./SSL_src/hyparam/logger.yaml']
    
    args=util.util.get_yaml_args(args)
    t=Trainer(args)
    t.run()