import os
import math
from typing import List
import torch
from diffusers.utils import(
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
try:
    # Try the old import path
    from diffusers.utils import is_compiled_module
except ImportError:
    # If the old import path is not available, use the new import path
    from diffusers.utils.torch_utils import is_compiled_module
import PIL
import torch.nn.functional as F
import numpy as np
from PIL import Image
from safetensors import safe_open
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.controlnet import ControlNetModel
from diffusers.models.embeddings import ImageProjection
import torch.nn as nn

from functools import partial
def find_index(functions,name):
    target_function_index = None
    for index, func in enumerate(functions):
        if (hasattr(func, "__name__") and func.__name__ == name) or (hasattr(func, "func") and hasattr(func.func, "__name__") and func.func.__name__ == name):
            target_function_index = index
            break
    return target_function_index
def apply_midIpAdapter(pipe):
    pipe.denoising_step_functions.insert(0, partial(ipAdapterSetter, pipe))
    pipe.denoising_step_functions.insert(0, partial(ipAdapterRemover, pipe))

    def remover_midIpAdapter():
        pipe.denoising_step_functions.pop(0)
        pipe.denoising_step_functions.pop(0)
        
    pipe.revert_functions.insert(0,remover_midIpAdapter)

def ipAdapterSetter(self,i,t,**kwargs):
    if kwargs.get('ip_start') is not None and kwargs.get('ip_start')==i and kwargs.get('activation_function') is not None:
        activation_function = kwargs.get('activation_function')
        kwargs=activation_function(self,**kwargs)
    return kwargs
def ipAdapterRemover(self,i,t,**kwargs):
    if kwargs.get('ip_end') is not None and kwargs.get('ip_end')==i and kwargs.get('removal_function') is not None:
        removal_function = kwargs.get('removal_function')
        kwargs=removal_function(self,**kwargs)
    return kwargs