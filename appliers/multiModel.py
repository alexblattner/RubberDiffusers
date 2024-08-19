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
from diffusers.models.controlnet import ControlNetModel
from functools import partial
def find_index(functions,name):
    target_function_index = None
    for index, func in enumerate(functions):
        if (hasattr(func, "__name__") and func.__name__ == name) or (hasattr(func, "func") and hasattr(func.func, "__name__") and func.func.__name__ == name):
            target_function_index = index
            break
    return target_function_index
def apply_multiModel(pipe):
    pipe.added_model=[]
    denoiser_index=find_index(pipe.denoising_functions,"denoiser")
    pipe.denoising_functions.insert(0,partial(set_other_pipes,pipe))
    pipe.denoising_functions.insert(denoiser_index,partial(mask_prepare_multiModel,pipe))

    compute_previous_noisy_sample_index=find_index(pipe.denoising_step_functions,"compute_previous_noisy_sample")
    
    pipe.denoising_step_functions.insert(compute_previous_noisy_sample_index,partial(diffusion_step,pipe))

    #reverse
    def remover_multiModel():
        del pipe.added_model
        pipe.denoising_step_functions.pop(compute_previous_noisy_sample_index)
        pipe.denoising_functions.pop(denoiser_index)
        pipe.denoising_functions.pop(0)

    pipe.revert_functions.insert(0,remover_multiModel)
    
def mask_prepare_multiModel(self,**kwargs):
    def create_rectangular_mask(height, width, y_start, x_start, block_height, block_width, strength, device='cpu'):
            mask = torch.zeros(height, width, device=device)
            mask[y_start:y_start + block_height, x_start:x_start + block_width] = strength
            return mask
    
    mask_list = []
    dtype=kwargs.get('dtype')
    pos=kwargs['pos']
    height=kwargs['height']
    width=kwargs['width']

    plen= len(kwargs['pos']) if kwargs['pos'] is not None  else len(kwargs['pos'])
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
            one_filter = create_rectangular_mask(kwargs['height'] // 8, kwargs['width'] // 8, y_start, x_start, block_height, block_width,kwargs['mask_strengths'][i], device=kwargs['device'])
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
            one_filter *=kwargs['mask_strengths'][i]
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
    kwargs['mask_list']=mask_list
    return kwargs
def set_other_pipes(self,**kwargs):
    for i in range(len(self.added_model)):
        model_kwargs=kwargs.get('model_kwargs')[i]
        defaults=['prompt','num_inference_steps','guidance_scale','guidance_rescale','eta','return_dict','callback_steps','num_images_per_prompt','output_type','clip_skip','cross_attention_kwargs','dtype','device','nsfw','generator','height','width']
        for k in kwargs.keys():
            if k not in model_kwargs and k in defaults:
                model_kwargs[k]=kwargs[k]
        si=0
        ei=find_index(self.added_model[i].denoising_functions,"denoiser")
        for j in range(ei-si):
            model_kwargs=self.added_model[i].denoising_functions[j](**model_kwargs)
        kwargs['model_kwargs'][i]=model_kwargs
    return kwargs
# def expand_and_scale_latents2(self,i,t,**kwargs):
#     latents2=kwargs.get('latents2') if kwargs.get('latents2') is not None else kwargs.get('latents')
#     latents=kwargs['latents']
#     latent_model_input=kwargs['latent_model_input']
#     kwargs['latents']=latents2
#     kwargs=self.expand_latents(i,t,**kwargs)
#     kwargs=self.scale_model_input(i,t,**kwargs)
#     kwargs['latents2']=kwargs['latents']
#     kwargs['latent_model_input2']=kwargs['latent_model_input']
#     kwargs['latent_model_input']=latent_model_input
#     kwargs['latents']=latents
#     return kwargs

def diffusion_step(self, i, t, **kwargs):
    latents = kwargs.get('latents')
    mask_list=kwargs.get('mask_list')
    noise_pred = kwargs.get('noise_pred')
    noises=[]
    noise_pred=noise_pred*mask_list[0]
    count=1
    for i in range(len(self.added_model)):
        model_kwargs=kwargs.get('model_kwargs')[i]
        si=find_index(self.added_model[i].denoising_step_functions,"unet_kwargs")
        ei=find_index(self.added_model[i].denoising_step_functions,"compute_previous_noisy_sample")
        model_kwargs['latent_model_input']=kwargs['latent_model_input']
        for j in range(ei-si):
            model_kwargs=self.added_model[i].denoising_step_functions[j+si](i,t,**model_kwargs)
        noise_pred+=model_kwargs['noise_pred']*mask_list[count]
        count+=1
        kwargs['model_kwargs'][i]=model_kwargs
    kwargs['noise_pred']=noise_pred
    return kwargs
# def compute_previous_noisy_sample(self, i, t, **kwargs):
#     latents = kwargs.get('latents')
#     mask_list=kwargs.get('mask_list')
#     noise_pred = kwargs.get('noise_pred')
#     noises=[]
#     model_kwargs=kwargs['model_kwargs']
#     for i in model_kwargs:
#         noises.append(i['noise_pred'])
#     noise_pred=noise_pred*mask_list[0]
#     count=1
#     for i in noises:
#         noise_pred+=i*mask_list[count]
#         count+=1
#     latents = self.scheduler.step(noise_pred, t, latents, **kwargs.get('extra_step_kwargs'), return_dict=False)[0]
#     kwargs['latents'] = latents
#     return kwargs
# def postProcess2(self,**kwargs):
#     latents=kwargs.get('latents')
#     kwargs['latents']=kwargs.get('latents2')
#     image=kwargs['image']
#     kwargs=self.postProcess(**kwargs)
#     kwargs['image']=kwargs['image']+image
#     return kwargs