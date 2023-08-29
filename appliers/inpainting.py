import torch
from diffusers.utils import(
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
import PIL
from PIL import Image
import numpy as np
from typing import Union
from functools import partial
def apply_inpainting(pipe):

    pipe.denoising_functions.insert(0, partial(inpainting_default, pipe))
    new_function_index = pipe.denoising_functions.index(pipe.checker)
    pipe.denoising_functions.insert(new_function_index, partial(inpainting_check_inputs, pipe))

    pipe.prepare_latents = partial(prepare_latents, pipe)
    pipe._encode_vae_image = partial(_encode_vae_image, pipe)
    new_function_index = pipe.denoising_functions.index(pipe.prepare_latent_var)
    pipe.prepare_latent_var = partial(prepare_latent_var, pipe)
    pipe.denoising_functions[new_function_index]=pipe.prepare_latent_var
    pipe.denoising_functions.insert(new_function_index, partial(preprocess_img, pipe))
    pipe.get_timesteps=partial(get_timesteps, pipe)
    timesteps_index= pipe.denoising_functions.index(pipe.prepare_timesteps)
    pipe.prepare_timesteps=partial(prepare_timesteps, pipe)
    pipe.denoising_functions[timesteps_index]=pipe.prepare_timesteps

    new_function_index = pipe.denoising_step_functions.index(pipe.expand_latents)
    pipe.denoising_step_functions[new_function_index]=partial(expand_latents, pipe)
    pipe.denoising_step_functions.insert(len(pipe.denoising_step_functions) - 1, partial(num_channel4_conditional, pipe))
    return pipe

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

def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):
    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image
def preprocess_img(self, **kwargs):
    mask, masked_image, init_image = prepare_mask_and_masked_image(
        kwargs.get('image'), kwargs.get('mask_image'), kwargs.get('height'), kwargs.get('width'), return_image=True
    )
    kwargs['mask_condition'] = mask.clone()
    kwargs['mask'] = mask
    kwargs['masked_image'] = masked_image
    kwargs['init_image'] = init_image
    return kwargs
def get_timesteps(self, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

    return timesteps, num_inference_steps - t_start
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
def prepare_latent_var(self, **kwargs):
    num_channels_latents = self.vae.config.latent_channels
    num_channels_unet = self.unet.config.in_channels
    return_image_latents = num_channels_unet == 4

    latents_outputs = self.prepare_latents(
        kwargs.get('batch_size') * kwargs.get('num_images_per_prompt'),
        num_channels_latents,
        kwargs.get('height'),
        kwargs.get('width'),
        kwargs.get('prompt_embeds_dtype'),
        kwargs.get('device'),
        kwargs.get('generator'),
        kwargs.get('latents'),
        image=kwargs.get('init_image'),
        timestep=kwargs.get('latent_timestep'),
        is_strength_max=kwargs.get('is_strength_max'),
        return_noise=True,
        return_image_latents=kwargs.get('return_image_latents'),
    )

    if kwargs.get('return_image_latents'):
        latents, noise, image_latents = latents_outputs
        kwargs['image_latents']=image_latents
    else:
        latents, noise = latents_outputs
    kwargs['latents']=latents
    kwargs['noise']=noise
    return kwargs
def prepare_mask_latent_variables(self, **kwargs):
    mask, masked_image_latents = self.prepare_mask_latents(
        kwargs.get('mask'),
        kwargs.get('masked_image'),
        kwargs.get('batch_size') * kwargs.get('num_images_per_prompt'),
        kwargs.get('height'),
        kwargs.get('width'),
        kwargs.get('prompt_embeds_dtype'),
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