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


# class Logger_config():
#     def __init__(self, args) -> None:
#         self.args=args
#         self.result_folder=self.args['hyparam']['result_folder']
#         self.room_type=self.result_folder['room_type']
        

#     def save_output(self, epoch):

#         macro_precision = 0
#         macro_recall = 0
#         sum_TP = 0
#         sum_FP = 0
#         sum_FN = 0
#         for i in range(360):
#             TP = self.save_config_dict['confusion_matrix'][str(i)]['TP']
#             FP = self.save_config_dict['confusion_matrix'][str(i)]['FP']
#             FN = self.save_config_dict['confusion_matrix'][str(i)]['FN']

#             sum_TP += TP
#             sum_FP += FP
#             sum_FN += FN

#             if TP+FP != 0:
#                 precision = TP/(TP+FP)
#                 macro_precision += precision
#             if TP+FN != 0:
#                 recall = TP/(TP+FN)
#                 macro_recall += recall

#         macro_precision = macro_precision/360 * 100
#         macro_recall = macro_recall/360 * 100
        
#         total_error_sum = self.save_config_dict['total_error_sum']
#         number_of_degrees = self.save_config_dict['number_of_degrees']

#         MAE = total_error_sum/number_of_degrees

#         acc_10 = self.save_config_dict['acc_10']/number_of_degrees  * 100
#         acc_5 = self.save_config_dict['acc_5']/number_of_degrees    * 100
#         acc_1 = self.save_config_dict['acc_1']/number_of_degrees    * 100

#         print(f"MAE : {MAE:.2f}\n")

#         print(f"acc_10 : {acc_10:.2f}\n")
#         print(f"acc_5 : {acc_5:.2f}\n")
#         print(f"acc_1 : {acc_1:.2f}\n")


#         print(f"macro_precision : {macro_precision:.2f}\n")
#         print(f"macro_recall : {macro_recall:.2f}\n")

#         os.makedirs(self.result_folder['inference_folder']+ self.room_type[0], exist_ok=True)
#         with open(self.result_folder['inference_folder']+ self.room_type[0]+f'/result_{MAE:.2f}.txt', 'w') as f:

#             f.write('\nargmax_doa_error\n')
#             f.write(str(MAE)+'\n\n')

#             f.write('\nacc_10\n')
#             f.write(str(acc_10))
#             f.write('\nacc_5\n')
#             f.write(str(acc_5))
#             f.write('\nacc_1\n')
#             f.write(str(acc_1))

#             f.write('\n\nmacro_precision\n')
#             f.write(str(macro_precision))
#             f.write('\nmacro_recall\n')
#             f.write(str(macro_recall))

    
#     def error_update(self, output_azi, ans_azi, vad_bool, coherent_snr, rt60):

#         # output_azi : (block_num)
#         # vad_block : (block_num)
#         # ans_azi : (1)

#         error = abs(output_azi - ans_azi)
#         error = np.minimum(error, 360-error)

#         for i, vad in enumerate(vad_bool):
#             if vad:

#                 self.save_config_dict['total_error_sum'] += error[i]
#                 self.save_config_dict['number_of_degrees'] += 1

#                 e = error[i]

#                 if e <= 10:
#                     self.save_config_dict['confusion_matrix'][str(ans_azi)]['TP'] += 1
#                 else:
#                     self.save_config_dict['confusion_matrix'][str(ans_azi)]['FN'] += 1
#                     self.save_config_dict['confusion_matrix'][str(output_azi[i])]['FP'] += 1


#                 if e <= 1:
#                     self.save_config_dict['acc_1'] += 1
#                 if e <= 5:
#                     self.save_config_dict['acc_5'] += 1
#                 if e <= 10:
#                     self.save_config_dict['acc_10'] += 1

                
#                 self.scatter_list_snr.append((coherent_snr, e))
#                 self.scatter_list_rt60.append((rt60, e))


#     def scatter_error_plot(self):
        
#         scatter_list_snr = self.scatter_list_snr
#         scatter_list_snr.sort(key=lambda x: x[0])

#         snr_values, error_values_snr = zip(*scatter_list_snr)

#         plt.figure()
#         plt.scatter(snr_values, error_values_snr, color='b', alpha=0.1, s=1, label='Data Points')
#         plt.xlabel('SNR')
#         plt.ylabel('DOA error')
#         plt.savefig(self.result_folder['inference_folder']+ self.room_type[0]+f'/scatter_snr.png', dpi=600)
#         plt.close()


#         scatter_list_rt60 = self.scatter_list_rt60
#         scatter_list_rt60.sort(key=lambda x: x[0])

#         rt60_values, error_values_rt60 = zip(*scatter_list_rt60)

#         plt.figure()
#         plt.scatter(rt60_values, error_values_rt60, color='g', alpha=0.1, label='Data Points')
#         plt.xlabel('RT60')
#         plt.ylabel('DOA error')
#         plt.savefig(self.result_folder['inference_folder']+ self.room_type[0]+f'/scatter_rt60.png', dpi=600)
#         plt.close()

                

  
#     def config(self,):

#         self.scatter_list_snr = []
#         self.scatter_list_rt60 = []
        
#         self.save_config_dict=dict()

#         self.save_config_dict['acc_1']=0
#         self.save_config_dict['acc_5']=0
#         self.save_config_dict['acc_10']=0
#         self.save_config_dict['total_error_sum']=0
#         self.save_config_dict['number_of_degrees']=0

#         self.save_config_dict['confusion_matrix']={}
#         for i in range(360):
#             self.save_config_dict['confusion_matrix'][str(i)]={'TP':0, 'FP':0, 'FN':0}

#         return self.args


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
        # self.test_loader=Synth_dataload(self.args['dataloader']['test']['loader'])
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
      
        self.test(0)

    
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
        os.makedirs(self.logger.result_folder['inference_folder'] + '/pngs/', exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.logger.result_folder['inference_folder'] + '/pngs/' + pkl_idx.split('/')[-1].replace('.pkl', '.png'), dpi=600)
        plt.close()



    def test(self, epoch):
        self.model.eval()

        for room_type in self.args['hyparam']['result_folder']['room_type']:
            room_type=str(room_type)    
            self.dataloader.test_loader.dataset.room_type=str(room_type)

            with torch.no_grad():

                for iter_num, (mixed, vad, speech_azi, white_snr, coherent_snr, rt60) in enumerate(tqdm(self.dataloader.test_loader, desc='Test', total=len(self.dataloader.test_loader))):

                    mixed=mixed.to(self.hyperparameter.device)
                    vad=vad.to(self.hyperparameter.device)
                    speech_azi=speech_azi.to(self.hyperparameter.device)
    

                    out, target, vad_block = self.model(mixed, vad, speech_azi)    # (B, 3, 360, n), (B, num_spk, n)

                    out=out.sigmoid().detach().cpu()  
                    target=target.cpu()               
                    speech_azi=speech_azi.cpu()         # (B, 1)
                    vad_block=vad_block.cpu()           # (B, num_spk, n)
                    
                    pkl_idx = self.dataloader.test_loader.dataset.pkl_list[iter_num]
                    self.plot_out_target(out[0,2], target[0,2], pkl_idx)

                    num_spk = vad_block.sum(axis=1).max()       # (1, )
                    num_spk = num_spk.item()
                    total_argmax_acc, total_softmax_acc, total_half_softmax_acc, \
                        total_argmax_doa_error, total_softmax_doa_error,total_half_softmax_doa_error, \
                            number_of_degrees_to_estimate=metric.mae.calc_mae(out, target, vad_block, num_spk, speech_azi,\
                                                                        calc_layer=self.args['learner']['loss']['option']['train_map_num'],\
                                                                            acc_threshold=self.args['hyparam']['acc_threshold'],\
                                                                                local_maximum_distance=self.args['hyparam']['local_maximum_distance'])
    
    
                    # output_azi = out[0,2].numpy().argmax(axis=0)        # (n, )
                    # vad_bool = target[0,2].numpy().sum(axis=0) > 0     # (n, )
                    # ans_azi = speech_azi[0,0].numpy()                 # (1, )
                    
                    # self.logger.error_update(output_azi, ans_azi, vad_bool, coherent_snr, rt60)

                    self.logger.error_update(room_type, total_argmax_acc, total_softmax_acc,total_half_softmax_acc, total_argmax_doa_error, total_softmax_doa_error, total_half_softmax_doa_error,number_of_degrees_to_estimate)

                    
                    self.learner.memory_delete([mixed, vad, speech_azi, out, target])
                  
                self.logger.save_output(room_type)
                # self.logger.scatter_error_plot()

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