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
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from functools import partial
def find_index(functions,name):
    target_function_index = None
    for index, func in enumerate(functions):
        if (hasattr(func, "__name__") and func.__name__ == name) or (hasattr(func, "func") and hasattr(func.func, "__name__") and func.func.__name__ == name):
            target_function_index = index
            break
    return target_function_index
def apply_multiDiffusion2(pipe):
    determine_batch_size_index= find_index(pipe.denoising_functions,"determine_batch_size")
    pipe.inner_determine_batch_size_multiDiffusion=pipe.determine_batch_size
    pipe.determine_batch_size=partial(determine_batch_size, pipe)
    pipe.denoising_functions[determine_batch_size_index]=pipe.determine_batch_size

    #replace encode_input with a new function that contains it
    encode_input_index= find_index(pipe.denoising_functions,"encode_input")
    inner_encode_input_multiDiffusion=pipe.encode_input
    pipe.inner_encode_input_multiDiffusion=inner_encode_input_multiDiffusion
    pipe.encode_input=partial(encode_input, pipe)
    pipe.denoising_functions[encode_input_index]=pipe.encode_input
    
    #add prepare_mask_latent_variables function before the denoiser
    denoiser_index= find_index(pipe.denoising_functions,"denoiser")
    pipe.denoising_functions.insert(denoiser_index, partial(mask_prepare_multiDiffusion,pipe))
    
    pipe.multiDiffusionFunctions=[]
    si=find_index(pipe.denoising_step_functions,"unet_kwargs")
    ei=find_index(pipe.denoising_step_functions,"compute_previous_noisy_sample")
    print("count: ",ei-si)
    for i in range(ei-si):
        print(i)
        pipe.multiDiffusionFunctions.append(pipe.denoising_step_functions.pop(si))
    pipe.inner_unet_kwargs_multiDiffusion=pipe.unet_kwargs
    pipe.unet_kwargs=partial(unet_kwargs,pipe)
    pipe.denoising_step_functions.insert(si,pipe.unet_kwargs)
    #reverse
    def remover_multiDiffusion():
        pipe.denoising_step_functions.pop(si)
        pipe.unet_kwargs=pipe.inner_unet_kwargs_multiDiffusion
        delattr(pipe, f"inner_unet_kwargs_multiDiffusion")
        for i in range(ei-si):
            pipe.denoising_step_functions.insert(si,pipe.multiDiffusionFunctions[len(pipe.multiDiffusionFunctions)-1])
        delattr(pipe, f"multiDiffusionFunctions")
        pipe.denoising_functions.pop(denoiser_index)

        #replace encode_input with the old function
        pipe.encode_input=pipe.inner_encode_input_multiDiffusion
        pipe.denoising_functions[encode_input_index]=pipe.encode_input
        delattr(pipe, f"inner_encode_input_multiDiffusion")

        pipe.determine_batch_size=pipe.inner_determine_batch_size_multiDiffusion
        delattr(pipe, f"inner_determine_batch_size_multiDiffusion")
        pipe.denoising_functions[determine_batch_size_index]=pipe.determine_batch_size

    pipe.revert_functions.insert(0,remover_multiDiffusion)
def determine_batch_size(self, **kwargs):
    prompt=kwargs.get('prompt')
    prompt_embeds=kwargs.get('prompt_embeds')
    kwargs['prompt']=kwargs.get('prompt')[0] if kwargs.get('prompt') else kwargs.get('prompt')
    kwargs['prompt_embeds']=kwargs.get('prompt_embeds')[0] if kwargs.get('prompt_embeds') else kwargs.get('prompt_embeds')
    kwargs=self.inner_determine_batch_size_multiDiffusion(**kwargs)
    kwargs['prompt']=prompt
    kwargs['prompt_embeds']=prompt_embeds
    return kwargs
def encode_input(self, **kwargs):
    plen=len(kwargs.get('prompt_embeds')) if kwargs.get('prompt_embeds') else len(kwargs.get('prompt'))
    prompt_embedsArr=[]
    negative_prompt_embedsArr=[]
    for i in range(plen):
        prompt_embeds=kwargs.get('prompt_embeds')
        kwargs['prompt_embeds']=kwargs.get('prompt_embeds')[i] if kwargs.get('prompt_embeds') else None
        prompt=kwargs.get('prompt')
        kwargs['prompt']=kwargs.get('prompt')[i] if kwargs.get('prompt') else None
        negative_prompt_embeds=kwargs.get('negative_prompt_embeds')
        kwargs['negative_prompt_embeds']=kwargs.get('negative_prompt_embeds')[i] if kwargs.get('negative_prompt_embeds') else None
        negative_prompt=kwargs.get('negative_prompt')
        kwargs['negative_prompt']=kwargs.get('negative_prompt')[i] if kwargs.get('negative_prompt') else None
        kwargs=self.inner_encode_input_multiDiffusion(**kwargs)
        prompt_embedsArr.append(kwargs['prompt_embeds'])
        kwargs['prompt_embeds']=prompt_embeds
        kwargs['prompt']=prompt
        negative_prompt_embedsArr.append(kwargs['negative_prompt_embeds'])
        kwargs['negative_prompt_embeds']=negative_prompt_embeds
        kwargs['negative_prompt']=negative_prompt
    kwargs['prompt_embeds']=prompt_embedsArr
    kwargs['negative_prompt_embeds']=negative_prompt_embedsArr
    return kwargs
def mask_prepare_multiDiffusion(self,**kwargs):
    def create_rectangular_mask(height, width, y_start, x_start, block_height, block_width, strength, device='cpu'):
            mask = torch.zeros(height, width, device=device)
            mask[y_start:y_start + block_height, x_start:x_start + block_width] = strength
            return mask
    
    mask_list = []
    dtype=kwargs.get('dtype')
    pos=kwargs['multi_diffusion_pos']
    height=kwargs['height']
    width=kwargs['width']
    plen= len(kwargs['multi_diffusion_pos']) if kwargs['multi_diffusion_pos'] is not None  else len(kwargs['multi_diffusion_pos'])
    for i in range(plen):
        one_filter = None
        if isinstance(pos[i], str):
            pos_base = pos[i].split("-")
            pos_start = pos_base[0].split(":")
            pos_end = pos_base[1].split(":")
            block_height = abs(int(pos_start[1]) - int(pos_end[1])) // 8
            block_width = abs(int(pos_start[0]) - int(pos_end[0])) // 8
            y_start = int(pos_start[1]) // 8
            x_start = int(pos_start[0]) // 8
            one_filter = create_rectangular_mask(kwargs['height'] // 8, kwargs['width'] // 8, y_start, x_start, block_height, block_width,kwargs['multi_diffusion_mask_strengths'][i], device=kwargs['device'])
            # one_filter=one_filter.unsqueeze(0).expand(batch_size, 4, -1, -1).to(torch.float16)
        else:
            img = pos[i].convert('L').resize((kwargs['width'] // 8, kwargs['height'] // 8))

            # Convert image data to a numpy array
            np_data = np.array(img)

            # Normalize the data to range between 0 and 1
            np_data = np_data / 255

            np_data = (np_data > 0.5).astype(np.float32)
            # Convert the numpy array to a PyTorch tensor
            mask = torch.from_numpy(np_data)

            # Convert the numpy array to a PyTorch tensor
            one_filter = mask.to('cuda')
            one_filter *=kwargs['multi_diffusion_mask_strengths'][i]
        if dtype:
            one_filter=one_filter.to(dtype)
        mask_list.append(one_filter)

    base_mask = torch.zeros(height//8, width//8, device=kwargs['device'])
    if dtype:
        base_mask=base_mask.to(dtype)
    # For each pixel
    for x in range(kwargs['height']//8):
        for y in range(kwargs['width']//8):
            # Get the indices of the masks that are applied to this pixel
            applied_mask_indices = [idx for idx, mask in enumerate(mask_list) if mask[x, y] > 0]
            if len(applied_mask_indices)==0:
                base_mask[x, y] = 1
            elif len(applied_mask_indices)>0:
                value=0
                #count the value in that pixel
                for i in range(len(applied_mask_indices)):
                    value += mask_list[applied_mask_indices[i]][x, y]
                base_mask[x, y] = 1-value
    mask_list.insert(0,base_mask)
    kwargs['multi_diffusion_mask_list']=mask_list
    return kwargs
def unet_kwargs(self, i, t, **kwargs):
    mask_list=kwargs['multi_diffusion_mask_list']
    count=0
    result=None
    for j in mask_list:
        activation_function=kwargs.get('multiDiffusion_activation_functions')
        if activation_function is not None:
            activation_function=activation_function[count]
            if activation_function is not None:
                kwargs=activation_function(self,i,t,**kwargs)
        pe=kwargs['prompt_embeds']
        ne=kwargs['negative_prompt_embeds']
        kwargs['prompt_embeds']=kwargs['prompt_embeds'][count]
        kwargs['negative_prompt_embeds']=kwargs['negative_prompt_embeds'][count]
        tc=0
        for k in self.multiDiffusionFunctions:
            kwargs=k(i,t,**kwargs)
            tc+=1

        if count>0:
            result += kwargs['noise_pred'] * mask_list[count]
        else:
            result = kwargs['noise_pred'] * mask_list[count]
        removal_function=kwargs.get('multiDiffusion_removal_functions')
        if removal_function is not None:
            removal_function=removal_function[count]
            if removal_function is not None:
                kwargs=removal_function(self,i,t,**kwargs)
        count+=1
        kwargs['prompt_embeds']=pe
        kwargs['negative_prompt_embeds']=ne
    kwargs['noise_pred']=result
    return kwargs