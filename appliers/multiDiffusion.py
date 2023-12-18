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
def apply_multiDiffusion(pipe):
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
    si=find_index(pipe.denoising_step_functions,"predict_noise_residual")
    ei=find_index(pipe.denoising_step_functions,"compute_previous_noisy_sample")
    for i in range(ei-si):
        pipe.multiDiffusionFunctions.append(pipe.denoising_step_functions.pop(si))
    pipe.denoising_step_functions.insert(si,partial(mulitDiffusion_looping,pipe))
    #reverse
    def remover_multiDiffusion():
        pipe.denoising_step_functions.pop(si)
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
    def create_rectangular_mask(height, width, y_start, x_start, block_height, block_width, device='cpu'):
            mask = torch.zeros(height, width, device=device)
            mask[y_start:y_start + block_height, x_start:x_start + block_width] = 1
            return mask
    
    mask_list = []
    plen= len(kwargs['prompt']) if kwargs['prompt'] is not None  else len(kwargs['prompt_embeds'])
    pos=kwargs['pos']
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
            one_filter = create_rectangular_mask(kwargs['height'] // 8, kwargs['width'] // 8, y_start, x_start, block_height, block_width, device=kwargs['device'])
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
            # one_filter = one_filter.unsqueeze(0)
            # one_filter = one_filter.unsqueeze(0).expand(batch_size, 4, -1, -1).to(torch.float16)

        mask_list.append(one_filter)

    # For each pixel
    for x in range(kwargs['height']//8):
        for y in range(kwargs['width']//8):
            # Get the indices of the masks that are applied to this pixel
            applied_mask_indices = [idx for idx, mask in enumerate(mask_list) if mask[x, y] > 0]

            if applied_mask_indices:
                mask_strengths = [kwargs['mask_z_index'][idx] for idx in applied_mask_indices]
                # Calculate the weights for the applied masks
                totalM=0
                multi=len(mask_strengths)>2
                pxvals=dict()
                for i in applied_mask_indices:
                    val=mask_list[i][x, y].item()
                    val=val*kwargs['mask_z_index'][i]
                    pxvals[i]=val
                    totalM+=val
                total_weights=0
                for i in applied_mask_indices:
                    w=(pxvals[i]/totalM)
                    mask_list[i][x, y] *= w
                    total_weights+=w
                if total_weights>1:
                    mask_list[applied_mask_indices[0]][x, y] -= total_weights-1
                elif total_weights<1:
                    mask_list[applied_mask_indices[0]][x, y] += 1-total_weights
            else:
                raise ValueError(
                        "unoccupied pixel in the mask. {x}, {y}"
                    )
    for i in range(len(mask_list)):
        mask_list[i] = mask_list[i].unsqueeze(0).expand(kwargs['batch_size'], 4, -1, -1).to(torch.float16)
        torchvision.transforms.functional.to_pil_image(mask_list[i][0]*256).save(str(i)+".png")
    kwargs['mask_list']=mask_list
    return kwargs
def mulitDiffusion_looping(self, i, t, **kwargs):
    mask_list=kwargs['mask_list']
    count=0
    result=None
    for j in mask_list:
        pe=kwargs['prompt_embeds']
        kwargs['prompt_embeds']=kwargs['prompt_embeds'][count]
        for k in self.multiDiffusionFunctions:
            kwargs=k(i,t,**kwargs)

        if count>0:
            result += kwargs['noise_pred'] * mask_list[count]
        else:
            result = kwargs['noise_pred'] * mask_list[count]
        count+=1
        kwargs['prompt_embeds']=pe

    kwargs['noise_pred']=result
    return kwargs