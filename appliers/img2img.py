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
    from diffusers.utils import randn_tensor
except ImportError:
    # If the old import path is not available, use the new import path
    from diffusers.utils.torch_utils import randn_tensor
import PIL
import numpy as np
from PIL import Image
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union
def find_index(functions,name):
    target_function_index = None
    for index, func in enumerate(functions):
        if (hasattr(func, "__name__") and func.__name__ == name) or (hasattr(func, "func") and hasattr(func.func, "__name__") and func.func.__name__ == name):
            target_function_index = index
            break
    return target_function_index
def apply_img2img(pipe):
    #insert defaults function
    pipe.denoising_functions.insert(0, partial(img2img_default, pipe))

    checker_index = find_index(pipe.denoising_functions,"checker")
    pipe.img2img_stored_checker=pipe.checker
    pipe.checker=partial(checker,pipe)
    pipe.denoising_functions[checker_index]=pipe.checker
    #insert img2img_check_inputs before checker function
    pipe.denoising_functions.insert(checker_index, partial(img2img_check_inputs, pipe))
    #insert img2img_prepare_latents
    pipe.img2img_prepare_latents = partial(img2img_prepare_latents, pipe)

    #replace prepare_latent_var
    latent_var_index = find_index(pipe.denoising_functions,"prepare_latent_var")
    pipe.img2img_stored_prepare_latent_var=pipe.prepare_latent_var
    pipe.prepare_latent_var = partial(prepare_latent_var, pipe)
    pipe.denoising_functions[latent_var_index]=pipe.prepare_latent_var
    #insert img2img_preprocess_img before prepare_latent_var function
    pipe.denoising_functions.insert(latent_var_index, partial(img2img_preprocess_img, pipe))

    #insert new function
    pipe.img2img_get_timesteps=partial(img2img_get_timesteps, pipe)

    #replace prepare_timesteps
    timesteps_index= find_index(pipe.denoising_functions,"prepare_timesteps")
    pipe.img2img_stored_prepare_timesteps=pipe.prepare_timesteps
    pipe.prepare_timesteps=partial(prepare_timesteps, pipe)
    pipe.denoising_functions[timesteps_index]=pipe.prepare_timesteps

    #reverse
    def remover_img2img():
        
        #undo replacement of prepare_timesteps
        pipe.prepare_timesteps=pipe.img2img_stored_prepare_timesteps
        pipe.denoising_functions[timesteps_index]=pipe.prepare_timesteps
        #remove img2img_stored_prepare_timesteps
        delattr(pipe, f"img2img_stored_prepare_timesteps")
        
        #remove img2img_get_timesteps
        delattr(pipe, f"img2img_get_timesteps")

        #pop img2img_preprocess_img
        pipe.denoising_functions.pop(latent_var_index)

        #undo replacement of prepare_latent_var
        pipe.prepare_latent_var = pipe.img2img_stored_prepare_latent_var
        pipe.denoising_functions[latent_var_index]=pipe.prepare_latent_var
        delattr(pipe, f"img2img_stored_prepare_latent_var")

        #remove img2img_check_inputs before checker function
        pipe.denoising_functions.pop(checker_index)
        delattr(pipe, f"img2img_prepare_latents")
        pipe.checker=pipe.img2img_stored_checker
        pipe.denoising_functions[checker_index]=pipe.checker
        
        delattr(pipe, f"img2img_stored_checker")
        #remove defaults function
        pipe.denoising_functions.pop(0)

    pipe.revert_functions.insert(0,remover_img2img)
def checker(self,**kwargs):
    if kwargs.get('strength') < 0 or kwargs.get('strength') > 1:
        raise ValueError(f"The value of strength should in [0.0, 1.0] but is {kwargs.get('strength')}")

    if kwargs.get('callback_steps') is not None and (not isinstance(kwargs.get('callback_steps'), int) or kwargs.get('callback_steps') <= 0):
        raise ValueError(
            f"`callback_steps` has to be a positive integer but is {kwargs.get('callback_steps')} of type"
            f" {type(kwargs.get('callback_steps'))}."
        )

    if kwargs.get('callback_on_step_end_tensor_inputs') is not None and not all(
        k in self._callback_tensor_inputs for k in kwargs.get('callback_on_step_end_tensor_inputs')
    ):
        raise ValueError(
            f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in kwargs.get('callback_on_step_end_tensor_inputs') if k not in self._callback_tensor_inputs]}"
        )
    if kwargs.get('prompt') is None and kwargs.get('prompt_embeds') is None:
        raise ValueError(
            "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
        )
    elif kwargs.get('prompt') is not None and (not isinstance(kwargs.get('prompt'), str) and not isinstance(kwargs.get('prompt'), list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(kwargs.get('prompt'))}")

    if kwargs.get('prompt_embeds') is not None and kwargs.get('negative_prompt_embeds') is not None:
        if kwargs.get('prompt_embeds').shape != kwargs.get('negative_prompt_embeds').shape:
            raise ValueError(
                "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                f" got: `prompt_embeds` {kwargs.get('prompt_embeds').shape} != `negative_prompt_embeds`"
                f" {kwargs.get('negative_prompt_embeds').shape}."
            )
    return kwargs
def img2img_default(self,**kwargs):
    if kwargs.get('strength') is None:
        kwargs['strength']=0.75
    if kwargs.get('skip_noise') is None:
        kwargs['skip_noise']=False
    return kwargs
def img2img_check_inputs(self,**kwargs):
    strength=kwargs.get('strength')
    image=kwargs.get('image')
    callback_steps=kwargs.get('callback_steps')
    def check_single_image_dimensions(image):
        innnn=5
    if strength < 0 or strength > 1:
        raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
    
    if isinstance(image, list):
        for img in image:
            check_single_image_dimensions(img)
    else:
        check_single_image_dimensions(image)

    return kwargs

def img2img_preprocess_img(self, **kwargs):
    image = self.image_processor.preprocess(kwargs.get('image'))
    kwargs['image']=image
    return kwargs
def img2img_get_timesteps(self, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

    return timesteps, num_inference_steps - t_start
def prepare_timesteps(self, **kwargs):
    self.scheduler.set_timesteps(kwargs.get('num_inference_steps'), device=kwargs.get('device'))
    num_inference_steps=kwargs.get('num_inference_steps')
    strength=kwargs.get('strength')
    device=kwargs.get('device')
    timesteps, num_inference_steps = self.img2img_get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(kwargs.get('batch_size') * kwargs.get('num_images_per_prompt'))
    kwargs['timesteps']=timesteps
    kwargs['num_inference_steps']=num_inference_steps
    kwargs['latent_timestep']=latent_timestep
    
    return kwargs
def prepare_latent_var(self, **kwargs):
    num_channels_latents = self.unet.config.in_channels
    latents = self.img2img_prepare_latents(
        kwargs.get('image'),kwargs.get('latent_timestep'),
        kwargs.get('batch_size'), kwargs.get('num_images_per_prompt'),
        kwargs.get('dtype'),
        kwargs.get('device'),
        kwargs.get('generator'),
        kwargs.get('skip_noise'),
    )
    return {'num_channels_latents': num_channels_latents, 'latents': latents, **kwargs}
def img2img_prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None,skip_noise=False):
    if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
        raise ValueError(
            f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
        )

    image = image.to(device=device, dtype=dtype)

    batch_size = batch_size * num_images_per_prompt

    if image.shape[1] == 4:
        init_latents = image

    else:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        elif isinstance(generator, list):
            init_latents = [
                retrieve_latents(self.vae.encode(image), generator=generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        init_latents = self.vae.config.scaling_factor * init_latents
    if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
        # expand init_latents for batch_size
        deprecation_message = (
            f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
            " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
            " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
            " your script to pass as many initial images as text prompts to suppress this warning."
        )
        deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
        additional_image_per_prompt = batch_size // init_latents.shape[0]
        init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
    elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
        raise ValueError(
            f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
        )
    else:
        init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # get latents
    init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
    latents = init_latents

    return latents

def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")