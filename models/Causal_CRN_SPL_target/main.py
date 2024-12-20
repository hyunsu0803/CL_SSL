from .CRN_SPL_target import main_model_for_doa
from .CL_CRN import main_model_for_scl

def get_model_for_scl(args):

    return main_model_for_scl(args)


def get_model_for_doa(args, args_scl=None, hyparam=None):

    return main_model_for_doa(args, args_scl, hyparam)
