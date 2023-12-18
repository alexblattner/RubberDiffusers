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
from PIL import Image
import numpy as np
from typing import Union
from diffusers.image_processor import VaeImageProcessor
from functools import partial
def find_index(functions,name):
    target_function_index = None
    for index, func in enumerate(functions):
        if (hasattr(func, "__name__") and func.__name__ == name) or (hasattr(func, "func") and hasattr(func.func, "__name__") and func.func.__name__ == name):
            target_function_index = index
            break
    return target_function_index
def apply_inpainting(pipe):
    #add mask processor
    pipe.mask_processor = VaeImageProcessor(
        vae_scale_factor=pipe.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )
    #make default function responsible for having default values be the first function
    pipe.denoising_functions.insert(0, partial(inpainting_default, pipe))
    #insert inpainting_check_inputs before checker function
    checker_index = find_index(pipe.denoising_functions,"checker")
    pipe.denoising_functions.insert(checker_index, partial(inpainting_check_inputs, pipe))
    #add prepare_mask_latents
    pipe.prepare_mask_latents=partial(prepare_mask_latents,pipe)
    #add prepare_latents
    pipe.prepare_latents = partial(prepare_latents, pipe)
    #add _encode_vae_image
    pipe._encode_vae_image = partial(_encode_vae_image, pipe)
    #replace prepare_latent_var with new version
    prepare_latent_var_index = find_index(pipe.denoising_functions,"prepare_latent_var")
    pipe.inpainting_stored_prepare_latent_var=pipe.prepare_latent_var
    pipe.prepare_latent_var = partial(prepare_latent_var, pipe)
    pipe.denoising_functions[prepare_latent_var_index]=pipe.prepare_latent_var
    #insert 2 functions before prepare_latent_var
    pipe.denoising_functions.insert(prepare_latent_var_index, partial(preprocess_img, pipe))
    pipe.denoising_functions.insert(prepare_latent_var_index, partial(set_init_image,pipe))
    #add get_timesteps
    pipe.get_timesteps=partial(get_timesteps, pipe)
    #replace prepare_timesteps with newer version
    prepare_timesteps_index= find_index(pipe.denoising_functions,"prepare_timesteps")
    pipe.inpainting_stored_prepare_timesteps=pipe.prepare_timesteps
    pipe.prepare_timesteps=partial(prepare_timesteps, pipe)
    pipe.denoising_functions[prepare_timesteps_index]=pipe.prepare_timesteps
    #add prepare_mask_latent_variables function before the denoiser
    denoiser_index= find_index(pipe.denoising_functions,"denoiser")
    pipe.denoising_functions.insert(denoiser_index, partial(prepare_mask_latent_variables,pipe))

    #replace expand_latents with newer version
    expand_latents_index = find_index(pipe.denoising_step_functions,"expand_latents")
    pipe.inpainting_stored_expand_latents=pipe.expand_latents
    pipe.expand_latents=partial(expand_latents, pipe)
    pipe.denoising_step_functions[expand_latents_index]=pipe.expand_latents
    
    #replace scale_model_input with newer version
    scale_model_input_index = find_index(pipe.denoising_step_functions,"scale_model_input")
    pipe.inpainting_stored_scale_model_input=pipe.scale_model_input
    pipe.scale_model_input=partial(scale_model_input, pipe)
    pipe.denoising_step_functions[scale_model_input_index]=pipe.scale_model_input
    #add num_channel4_conditional as before last function
    pipe.denoising_step_functions.insert(len(pipe.denoising_step_functions) - 1, partial(num_channel4_conditional, pipe))
    #reverse
    def remover_inpainting():
        #remove num_channel4_conditional as before last function
        pipe.denoising_step_functions.pop(len(pipe.denoising_step_functions) - 2)

        #undo replacement of scale_model_input with newer version
        pipe.scale_model_input=pipe.inpainting_stored_scale_model_input
        pipe.denoising_step_functions[scale_model_input_index]=pipe.scale_model_input
        delattr(pipe, f"inpainting_stored_scale_model_input")

        #undo replacement of expand_latents with newer version
        pipe.expand_latents=pipe.inpainting_stored_expand_latents
        pipe.denoising_step_functions[expand_latents_index]=pipe.expand_latents
        delattr(pipe, f"inpainting_stored_expand_latents")

        #remove prepare_mask_latent_variables function before the denoiser
        pipe.denoising_functions.pop(denoiser_index)

        #undo replacement of prepare_timesteps with newer version
        pipe.prepare_timesteps=pipe.inpainting_stored_prepare_timesteps
        pipe.denoising_functions[prepare_timesteps_index]=pipe.prepare_timesteps
        delattr(pipe, f"inpainting_stored_prepare_timesteps")

        #remove 2 functions before prepare_latent_var
        pipe.denoising_functions.pop(prepare_latent_var_index)
        pipe.denoising_functions.pop(prepare_latent_var_index)
        #remove get_timesteps
        delattr(pipe, f"get_timesteps")

        #replace prepare_latent_var with new version
        pipe.prepare_latent_var = pipe.inpainting_stored_prepare_latent_var
        pipe.denoising_functions[prepare_latent_var_index]=pipe.prepare_latent_var
        delattr(pipe,f"inpainting_stored_prepare_latent_var")

        #remove prepare_mask_latents
        delattr(pipe,f"prepare_mask_latents")
        #remove prepare_latents
        delattr(pipe,f"prepare_latents")
        #remove _encode_vae_image
        delattr(pipe,f"_encode_vae_image")

        #remove inpainting_check_inputs before checker function
        pipe.denoising_functions.pop(checker_index)

        #remove default function responsible for having default values
        pipe.denoising_functions.pop(0)
        #remove mask processor
        delattr(pipe,f"mask_processor")

    pipe.revert_functions.insert(0,remover_inpainting)
def inpainting_default(self,**kwargs):
    if kwargs.get('strength') is None:
        kwargs['strength']=0.75
    return kwargs
def inpainting_check_inputs(self,**kwargs):
    strength=kwargs.get('strength')
    height=kwargs.get('height')
    width=kwargs.get('width')
    callback_steps=kwargs.get('callback_steps')
    def check_image_type(image: Union[torch.FloatTensor, Image.Image], image_name: str):
        if not isinstance(image, (torch.FloatTensor, Image.Image)):
            raise TypeError(f"{image_name} should be of type torch.FloatTensor or PIL.Image.Image")
    if strength < 0 or strength > 1:
        raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    if (callback_steps is None) or (
        callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
    ):
        raise ValueError(
            f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
            f" {type(callback_steps)}."
        )
    
    check_image_type(kwargs.get('image'), "image")
    check_image_type(kwargs.get('mask_image'), "mask_image")
    return kwargs

def prepare_timesteps(self, **kwargs):
    num_inference_steps=kwargs.get('num_inference_steps')
    device=kwargs.get('device')
    strength=kwargs.get('strength')
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = self.get_timesteps(
        num_inference_steps=num_inference_steps, strength=strength, device=device
    )
    # check that number of inference steps is not < 1 - as this doesn't make sense
    if num_inference_steps < 1:
        raise ValueError(
            f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
            f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
        )
    # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
    latent_timestep = timesteps[:1].repeat(kwargs.get('batch_size') * kwargs.get('num_images_per_prompt'))
    # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
    is_strength_max = strength == 1.0
    kwargs['timesteps']=timesteps
    kwargs['num_inference_steps']=num_inference_steps
    kwargs['latent_timestep']=latent_timestep
    kwargs['is_strength_max']=is_strength_max
    return kwargs
def set_init_image(self, **kwargs):
    init_image = self.image_processor.preprocess(kwargs.get('image'), height=kwargs.get('height'), width=kwargs.get('width'))
    kwargs['init_image'] = init_image.to(dtype=torch.float32)
    return kwargs
def preprocess_img(self, **kwargs):
    init_image=kwargs.get('init_image')
    mask_condition = self.mask_processor.preprocess(kwargs.get('mask_image'), height=kwargs.get('height'), width=kwargs.get('width'))
    masked_image_latents=kwargs.get('masked_image_latents')
    if masked_image_latents is None:
        masked_image = init_image * (mask_condition < 0.5)
    else:
        masked_image = masked_image_latents
    kwargs['init_image']=init_image
    kwargs['masked_image']=masked_image
    kwargs['mask_condition']=mask_condition

    return kwargs


def prepare_latent_var(self, **kwargs):
    num_channels_latents = self.vae.config.latent_channels
    num_channels_unet = self.unet.config.in_channels
    return_image_latents = num_channels_unet == 4

    latents_outputs = self.prepare_latents(
        kwargs.get('batch_size') * kwargs.get('num_images_per_prompt'),
        num_channels_latents,
        kwargs.get('height'),
        kwargs.get('width'),
        kwargs.get('dtype'),
        kwargs.get('device'),
        kwargs.get('generator'),
        kwargs.get('latents'),
        image=kwargs.get('init_image'),
        timestep=kwargs.get('latent_timestep'),
        is_strength_max=kwargs.get('is_strength_max'),
        return_noise=True,
        return_image_latents=return_image_latents,
    )

    if return_image_latents:
        latents, noise, image_latents = latents_outputs
        kwargs['image_latents']=image_latents
    else:
        latents, noise = latents_outputs
    kwargs['latents']=latents
    kwargs['noise']=noise
    kwargs['num_channels_unet']=num_channels_unet
    kwargs['num_channels_latents']=num_channels_latents
    return kwargs
def prepare_mask_latent_variables(self, **kwargs):
    
    mask, masked_image_latents = self.prepare_mask_latents(
        kwargs.get('mask_condition'),
        kwargs.get('masked_image'),
        kwargs.get('batch_size') * kwargs.get('num_images_per_prompt'),
        kwargs.get('height'),
        kwargs.get('width'),
        kwargs.get('dtype'),
        kwargs.get('device'),
        kwargs.get('generator'),
        kwargs.get('do_classifier_free_guidance'),
    )
    if kwargs.get('num_channels_unet') == 9:
        # default case for runwayml/stable-diffusion-inpainting
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if kwargs.get('num_channels_latents') + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {kwargs.get('num_channels_latents')} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {kwargs.get('num_channels_latents')+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )
    elif kwargs.get('num_channels_latents') != 4:
        raise ValueError(
            f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
        )
    kwargs['mask']=mask
    kwargs['masked_image_latents']=masked_image_latents
    return kwargs
def expand_latents(self, i, t, **kwargs):
    # expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([kwargs.get('latents')] * 2) if kwargs.get('do_classifier_free_guidance') else kwargs.get('latents')
    kwargs['latent_model_input']=latent_model_input
    return kwargs
def scale_model_input(self, i, t, **kwargs):
    latent_model_input=kwargs.get('latent_model_input')
    prompt_embeds=kwargs.get('prompt_embeds')
    cross_attention_kwargs=kwargs.get('cross_attention_kwargs')
    # concat latents, mask, masked_image_latents in the channel dimension
    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

    if kwargs.get('num_channels_unet') == 9:
        latent_model_input = torch.cat([latent_model_input, kwargs.get('mask'), kwargs.get('masked_image_latents')], dim=1)
    kwargs['latent_model_input']=latent_model_input
    return kwargs
def num_channel4_conditional(self, i, t, **kwargs):
    if kwargs.get('num_channels_unet') == 4:
        init_latents_proper = kwargs.get('image_latents')[:1]
        init_mask = kwargs.get('mask')[:1]

        if i < len(kwargs.get('timesteps')) - 1:
            noise_timestep = kwargs.get('timesteps')[i + 1]
            init_latents_proper = self.scheduler.add_noise(
                init_latents_proper, kwargs.get('noise'), torch.tensor([noise_timestep])
            )

        kwargs['latents'] = (1 - init_mask) * init_latents_proper + init_mask * kwargs.get('latents')
    return kwargs


def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        timestep=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
    if isinstance(generator, list):
        image_latents = [
            self.vae.encode(image[i : i + 1]).latent_dist.sample(generator=generator[i])
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)

    image_latents = self.vae.config.scaling_factor * image_latents

    return image_latents

def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

def get_timesteps(self, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

    return timesteps, num_inference_steps - t_start
