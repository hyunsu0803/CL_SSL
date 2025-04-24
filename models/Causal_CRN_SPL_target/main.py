from .DOA_classifier import main_model_for_doa
from .EMA_CL import main_model_for_scl

def get_model_for_scl(args, doa=False):

    return main_model_for_scl(args, doa=doa)


def get_model_for_doa(args, args_scl=None, hyparam=None):

    return main_model_for_doa(args, args_scl, hyparam)
