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
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel,MultiAdapter, T2IAdapter

from functools import partial
def find_index(functions,name):
    target_function_index = None
    for index, func in enumerate(functions):
        if (hasattr(func, "__name__") and func.__name__ == name) or (hasattr(func, "func") and hasattr(func.func, "__name__") and func.func.__name__ == name):
            target_function_index = index
            break
    return target_function_index
def apply_t2iAdapter(pipe):
    #insert defaults function
    pipe.denoising_functions.insert(0, partial(t2iAdapter_default, pipe))
    #insert t2iAdapter_default_check_inputs before checker function
    checker_index = find_index(pipe.denoising_functions,"checker")
    pipe.denoising_functions.insert(checker_index, partial(t2iAdapter_default_check_inputs, pipe))
    #insert setup_t2i after checker function
    pipe.denoising_functions.insert(checker_index+2, partial(setup_t2i, pipe))

    #insert setup_adapter_state after guidance_scale_embedding
    guidance_scale_embedding_index=find_index(pipe.denoising_functions,"guidance_scale_embedding")
    pipe.denoising_functions.insert(guidance_scale_embedding_index+1, partial(setup_adapter_state, pipe))
    
    #insert unet_kwargs_t2iAdapter after unet_kwargs
    unet_kwargs_index=find_index(pipe.denoising_step_functions,"unet_kwargs")
    pipe.denoising_step_functions.insert(unet_kwargs_index+1, partial(unet_kwargs_t2iAdapter, pipe))
    #reverse
    def remover_t2iAdapter():
        #remove unet_kwargs_t2iAdapter
        pipe.denoising_step_functions.pop(unet_kwargs_index+1)
        #remove setup_adapter_state
        pipe.denoising_functions.pop(guidance_scale_embedding_index+1)
        #remove setup_t2i
        pipe.denoising_functions.pop(checker_index+2)
        #remove t2iAdapter_default_check_inputs
        pipe.denoising_functions.pop(checker_index)
        #remove defaults function
        pipe.denoising_functions.pop(0)
    pipe.revert_functions.insert(0,remover_t2iAdapter)

def t2iAdapter_default(self,**kwargs):
    if kwargs.get('adapter_conditioning_scale') is None:
        kwargs['adapter_conditioning_scale'] = [1.0]
    elif not isinstance(kwargs.get('adapter_conditioning_scale'),list):
        kwargs['adapter_conditioning_scale'] = [kwargs.get('adapter_conditioning_scale')]
    if kwargs.get('adapters') is None:
        raise ValueError(
            f"You need to pass adapters."
        )
    elif not isinstance(kwargs.get('adapters'),list):
        kwargs['adapters']=kwargs.get('adapters')
    else:
        kwargs['adapters']=MultiAdapter(kwargs.get('adapters'))
    return kwargs
def t2iAdapter_default_check_inputs(self,**kwargs):
    adapters=kwargs.get('adapters')
    t2i_image=kwargs.get('t2i_image')
    adapter_conditioning_scale=kwargs.get('adapter_conditioning_scale')
    def check(inp):
        if inp < 0 or inp > 1:
            raise ValueError(f"The value of adapter_conditioning_scale should in [0.0, 1.0] but is {inp}")

    for i in adapter_conditioning_scale:
        check(i)
    if not isinstance(t2i_image,list) and isinstance(adapters,list):
        raise ValueError(
            f"MultiAdapter requires passing the same number of images as adapters. Given {len(t2i_image)} images and {len(adapters)} adapters."
        )
    elif isinstance(t2i_image,list) and not isinstance(adapters,list):
        raise ValueError(
            f"Adapter requires passing a single image."
        )
    else:
        
        if isinstance(t2i_image,list) and len(t2i_image) != len(adapters):
            raise ValueError(
                f"MultiAdapter requires passing the same number of images as adapters. Given {len(t2i_image)} images and {len(adapters)} adapters."
            )
        if isinstance(adapters,list) and len(adapter_conditioning_scale) != len(adapters):
            raise ValueError(
                f"adapter_conditioning_scale isn't the same length as adapters. Given len of {len(adapter_conditioning_scale)} for adapter_conditioning_scale and len of {len(adapters)} for adapters."
            )
    return kwargs
    
def _preprocess_adapter_image(image, height, width):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
        image = [
            i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image
        ]  # expand [h, w] or [h, w, c] to [b, h, w, c]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        else:
            raise ValueError(
                f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}"
            )
    return image
def setup_t2i(self, **kwargs):
    t2i_image=kwargs.get('t2i_image')
    height=kwargs.get('height')
    width=kwargs.get('width')
    dtype=kwargs.get('dtype')
    device=kwargs.get('device')
    adapters=kwargs.get('adapters')
    if isinstance(adapters, MultiAdapter):
        adapter_input = []

        for one_image in t2i_image:
            one_image = _preprocess_adapter_image(one_image, height, width)
            one_image = one_image.to(device=device, dtype=dtype)
            adapter_input.append(one_image)
    else:
        adapter_input = _preprocess_adapter_image(t2i_image, height, width)
        adapter_input = adapter_input.to(device=device, dtype=dtype)
    kwargs['adapter_input']=adapter_input
    return kwargs

def setup_adapter_state(self, **kwargs):
    adapters=kwargs.get('adapters')
    adapter_input=kwargs.get('adapter_input')
    adapter_conditioning_scale=kwargs.get('adapter_conditioning_scale')
    num_images_per_prompt=kwargs.get('num_images_per_prompt')
    if isinstance(adapters, MultiAdapter):
        adapter_state = adapters(adapter_input, adapter_conditioning_scale)
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v
    else:
        adapter_state = adapters(adapter_input)
        if isinstance(adapter_conditioning_scale,list):
            adapter_conditioning_scale=adapter_conditioning_scale[0]
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v * adapter_conditioning_scale
    if num_images_per_prompt > 1:
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v.repeat(num_images_per_prompt, 1, 1, 1)
    if kwargs.get('do_classifier_free_guidance'):
        for k, v in enumerate(adapter_state):
            adapter_state[k] = torch.cat([v] * 2, dim=0)
    kwargs['adapter_state']=adapter_state
    return kwargs

def unet_kwargs_t2iAdapter(self,i,t,**kwargs):
    adapter_state=kwargs.get('adapter_state')
    kwargs['unet_kwargs']['down_intrablock_additional_residuals']=[state.clone() for state in adapter_state]
    return kwargs