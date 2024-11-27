# from .CRN_SPL_target import main_model
from .CL_CRN import main_model

def get_model(args):

    return main_model(args)
