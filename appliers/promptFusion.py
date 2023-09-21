import torch
import torch.nn.functional as F
from diffusers.utils import(
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
import PIL
import numpy as np
from functools import partial
def apply_promptFusion(pipe):
    #replace checker with a new function that contains it
    checker_index= pipe.denoising_functions.index(pipe.checker)
    inner_checker_promptFusion=pipe.checker
    pipe.inner_checker_promptFusion=inner_checker_promptFusion
    pipe.checker=partial(checker, pipe)
    pipe.denoising_functions[checker_index]=pipe.checker
    #replace encode_input with a new function that contains it
    encode_input_index= pipe.denoising_functions.index(pipe.encode_input)
    inner_encode_input_promptFusion=pipe.encode_input
    pipe.inner_encode_input_promptFusion=inner_encode_input_promptFusion
    pipe.encode_input=partial(encode_input, pipe)
    pipe.denoising_functions[encode_input_index]=pipe.encode_input

    #replace determine_batch_size with a new version of it
    determine_batch_size_index= pipe.denoising_functions.index(pipe.determine_batch_size)
    pipe.promptFusion_stored_determine_batch_size=pipe.determine_batch_size
    pipe.determine_batch_size=partial(determine_batch_size, pipe)
    pipe.denoising_functions[determine_batch_size_index]=pipe.determine_batch_size

    #insert prompt_fusion_step_modifier at start of the denoising_step_functions
    pipe.prompt_fusion_step_modifier=partial(prompt_fusion_step_modifier,pipe)
    pipe.denoising_step_functions.insert(0,pipe.prompt_fusion_step_modifier)
    #reverse
    def remover_promptFusion():
        #remove prompt_fusion_step_modifier from start of the denoising_step_functions
        pipe.denoising_step_functions.pop(0)
        #remove prompt_fusion_step_modifier
        delattr(pipe, f"prompt_fusion_step_modifier")

        #undo replacement of determine_batch_size with a new version of it
        pipe.determine_batch_size=pipe.promptFusion_stored_determine_batch_size
        pipe.denoising_functions[determine_batch_size_index]=pipe.determine_batch_size
        delattr(pipe, f"promptFusion_stored_determine_batch_size")

        #undo replacement of encode_input with a new function that contains it
        pipe.encode_input=inner_encode_input_promptFusion
        pipe.denoising_functions[encode_input_index]=pipe.encode_input
        delattr(pipe, f"inner_encode_input_promptFusion")

        #replace checker with a new function that contains it
        pipe.checker=pipe.inner_checker_promptFusion
        pipe.denoising_functions[checker_index]=pipe.checker
        delattr(pipe, f"inner_checker_promptFusion")

    pipe.revert_functions.insert(0,remover_promptFusion)
def checker(self, **kwargs):
    prompt=kwargs.get('prompt')
    negative_prompt=kwargs.get('negative_prompt')
    prompt_embeds=kwargs.get('prompt_embeds')
    negative_prompt_embeds=kwargs.get('negative_prompt_embeds')
    num_inference_steps=kwargs.get('num_inference_steps')
    nonEmptyArr=None
    if prompt is not None and isinstance(prompt, list):
        nonEmptyArr=prompt
        if negative_prompt is not None and isinstance(negative_prompt, list) and len(prompt)!=len(negative_prompt):
            raise ValueError(f"Both `prompt` and `negative_prompt` do not have the same length")
        if negative_prompt_embeds is not None and isinstance(negative_prompt_embeds, list) and len(prompt)!=len(negative_prompt_embeds):
            raise ValueError(f"Both `prompt` and `negative_prompt_embeds` do not have the same length")
    if prompt_embeds is not None and isinstance(prompt_embeds, list):
        nonEmptyArr=prompt_embeds
        if negative_prompt is not None and isinstance(negative_prompt, list) and len(prompt_embeds)!=len(negative_prompt):
            raise ValueError(f"Both `prompt_embeds` and `negative_prompt` do not have the same length")
        if negative_prompt_embeds is not None and isinstance(negative_prompt_embeds, list) and len(prompt_embeds)!=len(negative_prompt_embeds):
            raise ValueError(f"Both `prompt_embeds` and `negative_prompt_embeds` do not have the same length")
    if nonEmptyArr is None:
        raise ValueError(f"Both `prompt` and `prompt_embeds` are None or not lists")
    
    for idx, _ in enumerate(nonEmptyArr):
        if prompt is None:
            kwargs['prompt']=None  
        else:
            kwargs['prompt']=prompt[idx][0]
            if prompt[idx][1]>num_inference_steps:
                raise ValueError(f"prompt has a step greater than {num_inference_steps}")
        if negative_prompt is None:
            kwargs['negative_prompt']=None  
        else:
            kwargs['negative_prompt']=prompt[idx][0]
            if negative_prompt[idx][1]>num_inference_steps:
                raise ValueError(f"negative_prompt has a step greater than {num_inference_steps}")
        if prompt_embeds is None:
            kwargs['prompt_embeds']=None  
        else:
            kwargs['prompt_embeds']=prompt_embeds[idx][0]
            if prompt_embeds[idx][1]>num_inference_steps:
                raise ValueError(f"prompt_embeds has a step greater than {num_inference_steps}")
        if negative_prompt_embeds is None:
            kwargs['negative_prompt_embeds']=None  
        else:
            kwargs['negative_prompt_embeds']=negative_prompt_embeds[idx][0]
            if negative_prompt_embeds[idx][1]>num_inference_steps:
                raise ValueError(f"negative_prompt_embeds has a step greater than {num_inference_steps}")
        kwargs=self.inner_checker_promptFusion(**kwargs)
    kwargs['prompt']=prompt
    kwargs['negative_prompt']=negative_prompt
    kwargs['prompt_embeds']=prompt_embeds
    kwargs['negative_prompt_embeds']=negative_prompt_embeds
    return kwargs
def determine_batch_size(self, **kwargs):
    batch_size=kwargs.get('batch_size')
    if batch_size is None:
        prompt=kwargs['prompt']
        prompt_embeds=kwargs['prompt_embeds']
        if prompt is not None and isinstance(prompt[0][0], str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt[0][0], list):
            batch_size = prompt.shape[0]
        elif prompt_embeds is not None and isinstance(prompt_embeds[0][1], int):
            batch_size = 1
        elif prompt is not None and isinstance(prompt[0][1], list):
            batch_size = prompt_embeds.shape[0]
    kwargs['batch_size']=batch_size
    return kwargs
def encode_input(self, **kwargs):
    prompt=kwargs.get('prompt')
    negative_prompt=kwargs.get('negative_prompt')
    prompt_embeds=kwargs.get('prompt_embeds')
    negative_prompt_embeds=kwargs.get('negative_prompt_embeds')
    fusion_prompt_embeds=[]
    if prompt is not None and isinstance(prompt, list):
        nonEmptyArr=prompt
    if prompt_embeds is not None and isinstance(prompt_embeds, list):
        nonEmptyArr=prompt_embeds
    for idx, _ in enumerate(nonEmptyArr):
        kwargs['prompt']=None if prompt is None else prompt[idx][0]
        kwargs['negative_prompt']=None if negative_prompt is None else negative_prompt[idx][0]
        kwargs['prompt_embeds']=None if prompt_embeds is None else prompt_embeds[idx][0]
        kwargs['negative_prompt_embeds']=None if negative_prompt_embeds is None else negative_prompt_embeds[idx][0]
        kwargs=self.inner_encode_input_promptFusion(**kwargs)
        pe=[kwargs.get('prompt_embeds'),prompt_embeds[idx][1]] if prompt is None else [kwargs.get('prompt_embeds'),prompt[idx][1]]
        fusion_prompt_embeds.append(pe)
    kwargs['prompt_embeds']=fusion_prompt_embeds
    kwargs['fusion_prompt_embeds']=fusion_prompt_embeds
    return kwargs

def determineCurrent(arr,i):
    startIndex=0
    final=None
    if arr is None:
        return arr
    for a in arr:
        if a[1]>=i:
            final=a[0]
            break
    return final
def prompt_fusion_step_modifier(self, i, t, **kwargs):
    kwargs['prompt_embeds']=determineCurrent(kwargs.get('fusion_prompt_embeds'),i)
    return kwargs