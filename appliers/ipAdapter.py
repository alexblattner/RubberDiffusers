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
def apply_ipAdapter(pipe):
    #add 2 functions at checker_index
    pipe.inner_encode_input_ipAdapter=pipe.encode_input
    encode_input_index = find_index(pipe.denoising_functions,"encode_input")
    pipe.denoising_functions[encode_input_index]= partial(encode_input, pipe)
    prepare_extra_kwargs_index = find_index(pipe.denoising_functions,"prepare_extra_kwargs")
    pipe.denoising_functions.insert((prepare_extra_kwargs_index+1),partial(idAdapter_added_cond_kwargs, pipe))
    def remover_ipAdapter():
        pipe.denoising_functions.pop((prepare_extra_kwargs_index+1))
        pipe.encode_input=pipe.inner_encode_input_ipAdapter
        pipe.denoising_functions[encode_input_index]= pipe.encode_input
        delattr(pipe, f"inner_encode_input_ipAdapter")
        
    pipe.revert_functions.insert(0,remover_ipAdapter)
    
def ipAdapter_default(self,**kwargs):
    if kwargs.get('scale') is None:
        kwargs['scale']=1.0
    return kwargs

def encode_input(
    self,
    **kwargs,
):
    kwargs = self.inner_encode_input_ipAdapter(**kwargs)
    if kwargs.get('ip_adapter_image') is not None:
        output_hidden_state = False if isinstance(self.unet.encoder_hid_proj, ImageProjection) else True
        image_embeds, negative_image_embeds = self.encode_image(kwargs.get('ip_adapter_image'), kwargs.get('device'), kwargs.get('num_images_per_prompt'),output_hidden_state)
        if kwargs.get('do_classifier_free_guidance'):
            kwargs['image_embeds'] = torch.cat([negative_image_embeds, image_embeds])
    return kwargs

def idAdapter_added_cond_kwargs(self,**kwargs):
    kwargs['added_cond_kwargs'] = {"image_embeds": kwargs['image_embeds']} if kwargs.get('ip_adapter_image') is not None else None
    return kwargs