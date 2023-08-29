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
import numpy as np
from PIL import Image
from functools import partial
def apply_img2img(pipe):
    pipe.denoising_functions.insert(0, partial(img2img_default, pipe))
    new_function_index = pipe.denoising_functions.index(pipe.checker)
    pipe.denoising_functions.insert(new_function_index, partial(img2img_check_inputs, pipe))

    pipe.prepare_latents = partial(prepare_latents, pipe)
    new_function_index = pipe.denoising_functions.index(pipe.prepare_latent_var)
    pipe.prepare_latent_var = partial(prepare_latent_var, pipe)
    pipe.denoising_functions[new_function_index]=pipe.prepare_latent_var
    pipe.denoising_functions.insert(new_function_index, partial(preprocess_img, pipe))
    pipe.get_timesteps=partial(get_timesteps, pipe)
    timesteps_index= pipe.denoising_functions.index(pipe.prepare_timesteps)
    pipe.prepare_timesteps=partial(prepare_timesteps, pipe)
    pipe.denoising_functions[timesteps_index]=pipe.prepare_timesteps
    return pipe

def img2img_default(self,**kwargs):
    if kwargs.get('strength') is None:
        kwargs['strength']=0.75
    return kwargs
def img2img_check_inputs(self,**kwargs):
    strength=kwargs.get('strength')
    image=kwargs.get('image')
    callback_steps=kwargs.get('callback_steps')
    def check_single_image_dimensions(image):
        if isinstance(image, torch.FloatTensor):
            image = image.numpy()
        elif isinstance(image, Image.Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray):   
            raise TypeError("Unsupported image type")

        height, width = image.shape[:2]

        if height % 8 != 0:
            raise ValueError(f"The height is not divisible by 8 and is {height}")
        if width % 8 != 0:
            raise ValueError(f"The width is not divisible by 8 and is {width}")
    if strength < 0 or strength > 1:
        raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")
    
    if isinstance(image, list):
        for img in image:
            check_single_image_dimensions(img)
    else:
        check_single_image_dimensions(image)

    return kwargs

def preprocess_img(self, **kwargs):
    image = self.image_processor.preprocess(kwargs.get('image'))
    kwargs['image']=image
    return kwargs
def get_timesteps(self, num_inference_steps, strength, device):
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
    timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
    latent_timestep = timesteps[:1].repeat(kwargs.get('batch_size') * kwargs.get('num_images_per_prompt'))
    kwargs['timesteps']=timesteps
    kwargs['num_inference_steps']=num_inference_steps
    kwargs['latent_timestep']=latent_timestep
    
    return kwargs
def prepare_latent_var(self, **kwargs):
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        kwargs.get('image'),kwargs.get('latent_timestep'),
        kwargs.get('batch_size'), kwargs.get('num_images_per_prompt'),
        kwargs.get('dtype'),
        kwargs.get('device'),
        kwargs.get('generator'),
    )
    return {'num_channels_latents': num_channels_latents, 'latents': latents, **kwargs}
def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
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
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

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