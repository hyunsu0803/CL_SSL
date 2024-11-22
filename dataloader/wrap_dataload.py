from .base_loader.data_maker import train_data_maker, speech_data_maker, gunshot_data_maker
from .base_loader.data_loader import synth_data_loader, real_data_loader

from torch.utils.data import DataLoader 
import numpy as np
import random
import torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed) 
    
    
    
# data loaders
def Train_dataload(args, init_seed):
    g = torch.Generator()    
    g.manual_seed(init_seed)
    return DataLoader(train_data_maker(args),
                                            pin_memory=True,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            **args['dataloader_dict']
                                            )
def Synth_dataload(args):
    return DataLoader(synth_data_loader(args),
                                            pin_memory=True,
                                            **args['dataloader_dict']
                                            )
def Real_dataload(args):
    return DataLoader(real_data_loader(args),
                                            pin_memory=True,
                                            **args['dataloader_dict']
                                            )



# data makers
def Speech_datamake(args):
    return DataLoader(speech_data_maker(args),
                                            pin_memory=True,
                                            **args['dataloader_dict']
                                            )
def Gunshot_datamake(args):
    return DataLoader(gunshot_data_maker(args),
                                            pin_memory=True,
                                            **args['dataloader_dict']
                                            )
