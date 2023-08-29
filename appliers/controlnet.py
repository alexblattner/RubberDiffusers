import torch
from diffusers.utils import(
    PIL_INTERPOLATION,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    is_compiled_module,
    logging,
    randn_tensor,
    replace_example_docstring,
)
import PIL
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.controlnet import ControlNetModel
from functools import partial
def apply_controlnet(pipe):
    pipe.control_image_processor = VaeImageProcessor(
        vae_scale_factor=pipe.vae_scale_factor, do_convert_rgb=True, do_normalize=False
    )
    pipe.denoising_functions.insert(0, partial(controlnet_default, pipe))
    pipe.loadControlnet=partial(loadControlnet, pipe)
    pipe.prepare_controlnet_image=partial(StableDiffusionControlNetPipeline.prepare_image, pipe)
    pipe.controlnet=None
    pipe.add_controlnet=partial(add_controlnet,pipe)
    pipe.remove_controlnet=partial(remove_controlnet,pipe)
    pipe.controlnet_sub_check_image=partial(StableDiffusionControlNetPipeline.check_image,pipe)
    new_function_index = pipe.denoising_functions.index(pipe.checker)
    pipe.denoising_functions.insert(new_function_index, partial(controlnet_check_inputs, pipe))
    pipe.denoising_functions.insert(new_function_index, partial(controlnet_adjustments, pipe))

    new_function_index = pipe.denoising_functions.index(pipe.encode_input)
    pipe.denoising_functions.insert(new_function_index, partial(controlnet_conditional_guess_adjustments, pipe))

    new_function_index = pipe.denoising_functions.index(pipe.prepare_timesteps)
    pipe.denoising_functions.insert(new_function_index, partial(controlnet_prepare_image, pipe))
    
    new_function_index = pipe.denoising_functions.index(pipe.denoiser)
    pipe.denoising_functions.insert(new_function_index, partial(controlnet_keep_set, pipe))
    
    new_function_index=pipe.denoising_step_functions.index(pipe.predict_noise_residual)
    pipe.predict_noise_residual=partial(predict_noise_residual,pipe)
    pipe.denoising_step_functions[new_function_index]=pipe.predict_noise_residual
    pipe.denoising_step_functions.insert(new_function_index,partial(controlnet_denoising,pipe))
    
    new_function_index = pipe.denoising_functions.index(pipe.postProcess)
    pipe.denoising_functions.insert(new_function_index, partial(controlnet_offloading, pipe))
    return pipe
class BetterMultiControlnet(MultiControlNetModel):
    def add_controlnet(self, controlnet: ControlNetModel):
        """
        Add a new controlnet model to the list of controlnet models.
        """
        self.nets.append(controlnet)

    def remove_controlnet(self, controlnet: ControlNetModel):
        """
        Remove a specific controlnet model from the list of controlnet models.
        """
        if controlnet in self.nets:
            self.nets.remove(controlnet)
        else:
            print("ControlNetModel not found in the list.")
def controlnet_default(self,**kwargs):
    if kwargs.get('guess_mode') is None:
        kwargs['guess_mode'] = False
    if kwargs.get('control_guidance_start') is None:
        kwargs['control_guidance_start'] = 0.0
    if kwargs.get('control_guidance_end') is None:
        kwargs['control_guidance_end'] = 1.0
    if kwargs.get('controlnet_conditioning_scale') is None:
        kwargs['controlnet_conditioning_scale'] = [1.0]
    if not isinstance(kwargs.get('controlnet_image'),list):
        kwargs['controlnet_image']=[kwargs.get('controlnet_image')]
    return kwargs
def loadControlnet(self, controlnet: ControlNetModel):
    self.controlnet=BetterMultiControlnet(controlnet)
def add_controlnet(self, controlnet: ControlNetModel):
    if self.controlnet is None:
        self.controlnet = BetterMultiControlnet([controlnet])
    elif isinstance(self.controlnet, MultiControlNetModel):
        self.controlnet.add_controlnet(controlnet)
    else:
        raise ValueError("Something that isn't Multicontrolnet occupies controlnet")
def remove_controlnet(self, controlnet: ControlNetModel):
    if self.controlnet is None:
        raise ValueError("Can't remove when there's nothing to remove...")
    elif isinstance(self.controlnet, MultiControlNetModel):
        self.controlet.remove_controlnet(controlnet)
    else:
        raise ValueError("Something that isn't Multicontrolnet occupies controlnet")
def controlnet_adjustments(self, **kwargs):
    controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
    control_guidance_start=kwargs.get('control_guidance_start')
    control_guidance_end=kwargs.get('control_guidance_end')
    # align format for control guidance
    if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        control_guidance_start = len(control_guidance_end) * [control_guidance_start]
    elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        control_guidance_end = len(control_guidance_start) * [control_guidance_end]
    elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
            control_guidance_end
        ]
    kwargs['controlnet']=controlnet
    kwargs['control_guidance_start']=control_guidance_start
    kwargs['control_guidance_end']=control_guidance_end
    return kwargs
def controlnet_check_inputs(self, **kwargs):
    prompt=kwargs.get('prompt')
    prompt_embeds=kwargs.get('prompt_embeds')
    controlnet_conditioning_scale=kwargs.get('controlnet_conditioning_scale')
    control_guidance_start=kwargs.get('control_guidance_start')
    control_guidance_end=kwargs.get('control_guidance_end')
    controlnet_image=kwargs.get('controlnet_image')
    # Check `controlnet_image`
    is_compiled = hasattr(F, "scaled_dot_product_attention") and isinstance(
        self.controlnet, torch._dynamo.eval_frame.OptimizedModule
    )
    if controlnet_image is None:
        raise TypeError("'controlnet_image' can't be None")
    if (
        isinstance(self.controlnet, ControlNetModel)
        or is_compiled
        and isinstance(self.controlnet._orig_mod, ControlNetModel)
    ):
        self.controlnet_sub_check_image(controlnet_image, prompt, prompt_embeds)
    elif (
        isinstance(self.controlnet, MultiControlNetModel)
        or is_compiled
        and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
    ):
        if not isinstance(controlnet_image, list):
            raise TypeError("For multiple controlnets: `controlnet_image` must be type `list`")

        # When `controlnet_image` is a nested list:
        # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
        elif any(isinstance(i, list) for i in controlnet_image):
            raise ValueError("A single batch of multiple conditionings are supported at the moment.")
        elif len(controlnet_image) != len(self.controlnet.nets):
            raise ValueError(
                f"For multiple controlnets: `controlnet_image` must have the same length as the number of controlnets, but got {len(controlnet_image)} controlnet_images and {len(self.controlnet.nets)} ControlNets."
            )

        for image in controlnet_image:
            self.controlnet_sub_check_image(image, prompt, prompt_embeds)
    else:
        assert False

    # Check `controlnet_conditioning_scale`
    if (
        isinstance(self.controlnet, ControlNetModel)
        or is_compiled
        and isinstance(self.controlnet._orig_mod, ControlNetModel)
    ):
        if not isinstance(controlnet_conditioning_scale, float):
            raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
    elif (
        isinstance(self.controlnet, MultiControlNetModel)
        or is_compiled
        and isinstance(self.controlnet._orig_mod, MultiControlNetModel)
    ):
        if isinstance(controlnet_conditioning_scale, list):
            if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
        elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
            self.controlnet.nets
        ):
            raise ValueError(
                "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                " the same length as the number of controlnets"
            )
    else:
        assert False

    if not isinstance(control_guidance_start, (tuple, list)):
        control_guidance_start = [control_guidance_start]

    if not isinstance(control_guidance_end, (tuple, list)):
        control_guidance_end = [control_guidance_end]

    if len(control_guidance_start) != len(control_guidance_end):
        raise ValueError(
            f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
        )

    if isinstance(self.controlnet, MultiControlNetModel):
        if len(control_guidance_start) != len(self.controlnet.nets):
            raise ValueError(
                f"`control_guidance_start`: {control_guidance_start} has {len(control_guidance_start)} elements but there are {len(self.controlnet.nets)} controlnets available. Make sure to provide {len(self.controlnet.nets)}."
            )

    for start, end in zip(control_guidance_start, control_guidance_end):
        if start >= end:
            raise ValueError(
                f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
            )
        if start < 0.0:
            raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
        if end > 1.0:
            raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")
    return kwargs
def controlnet_conditional_guess_adjustments(self,**kwargs):
    controlnet=kwargs.get('controlnet')
    guess_mode=kwargs.get('guess_mode')
    controlnet_conditioning_scale=kwargs.get('controlnet_conditioning_scale')
    if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

    global_pool_conditions = (
        controlnet.config.global_pool_conditions
        if isinstance(controlnet, ControlNetModel)
        else controlnet.nets[0].config.global_pool_conditions
    )
    guess_mode = guess_mode or global_pool_conditions
    kwargs['controlnet_conditioning_scale']=controlnet_conditioning_scale
    kwargs['guess_mode']=guess_mode
    return kwargs
def controlnet_prepare_image(self,**kwargs):
    controlnet=kwargs.get('controlnet')
    width=kwargs.get('width')
    height=kwargs.get('height')
    batch_size=kwargs.get('batch_size')
    num_images_per_prompt=kwargs.get('num_images_per_prompt')
    device=kwargs.get('device')
    dtype=kwargs.get('dtype')
    do_classifier_free_guidance=kwargs.get('do_classifier_free_guidance')
    guess_mode=kwargs.get('guess_mode')
    controlnet_image=kwargs.get('controlnet_image')

    if isinstance(controlnet, ControlNetModel):
        controlnet_image = self.prepare_controlnet_controlnet_image(
            image=controlnet_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        height, width = controlnet_image.shape[-2:]
    elif isinstance(controlnet, MultiControlNetModel):
        controlnet_images = []

        for controlnet_image_ in controlnet_image:
            controlnet_image_ = self.prepare_controlnet_image(
                image=controlnet_image_,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

            controlnet_images.append(controlnet_image_)

        controlnet_image = controlnet_images
        height, width = controlnet_image[0].shape[-2:]
    else:
        assert False
    kwargs['controlnet_image']=controlnet_image
    kwargs['height']=height
    kwargs['width']=width
    return kwargs
def prepare_controlnet_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        controlnet_image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = controlnet_image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        controlnet_image = controlnet_image.repeat_interleave(repeat_by, dim=0)

        controlnet_image = controlnet_image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            controlnet_image = torch.cat([controlnet_image] * 2)

        return controlnet_image
def controlnet_keep_set(self,**kwargs):
    controlnet_keep = []
    timesteps=kwargs.get('timesteps')
    control_guidance_start=kwargs.get('control_guidance_start')
    control_guidance_end=kwargs.get('control_guidance_end')
    controlnet=kwargs.get('controlnet')
    for i in range(len(timesteps)):
        keeps = [
            1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
            for s, e in zip(control_guidance_start, control_guidance_end)
        ]
        controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
    kwargs['controlnet_keep']=controlnet_keep
    return kwargs
def controlnet_denoising(self, i, t, **kwargs):
    guess_mode=kwargs.get('guess_mode')
    do_classifier_free_guidance=kwargs.get('do_classifier_free_guidance')
    latents=kwargs.get('latents')
    prompt_embeds=kwargs.get('prompt_embeds')
    latent_model_input=kwargs.get('latent_model_input')
    controlnet_conditioning_scale=kwargs.get('controlnet_conditioning_scale')
    controlnet_image=kwargs.get('controlnet_image')
    cond_scale=kwargs.get('cond_scale')
    controlnet_keep=kwargs.get('controlnet_keep')
    if guess_mode and do_classifier_free_guidance:
        # Infer ControlNet only for the conditional batch.
        control_model_input = latents
        control_model_input = self.scheduler.scale_model_input(control_model_input, t)
        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
    else:
        control_model_input = latent_model_input
        controlnet_prompt_embeds = prompt_embeds

    if isinstance(controlnet_keep[i], list):
        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
    else:
        controlnet_cond_scale = controlnet_conditioning_scale
        if isinstance(controlnet_cond_scale, list):
            controlnet_cond_scale = controlnet_cond_scale[0]
        cond_scale = controlnet_cond_scale * controlnet_keep[i]
    down_block_res_samples, mid_block_res_sample = self.controlnet(
        control_model_input,
        t,
        encoder_hidden_states=controlnet_prompt_embeds,
        controlnet_cond=controlnet_image,
        conditioning_scale=cond_scale,
        guess_mode=guess_mode,
        return_dict=False,
    )
    if guess_mode and do_classifier_free_guidance:
        # Infered ControlNet only for the conditional batch.
        # To apply the output of ControlNet to both the unconditional and conditional batches,
        # add 0 to the unconditional batch to keep it unchanged.
        down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
        mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
    kwargs['down_block_res_samples']=down_block_res_samples
    kwargs['mid_block_res_sample']=mid_block_res_sample
    return kwargs
def predict_noise_residual(self,i,t,**kwargs):
    prompt_embeds = kwargs.get('prompt_embeds')
    cross_attention_kwargs = kwargs.get('cross_attention_kwargs')
    latent_model_input = kwargs.get('latent_model_input')
    down_block_res_samples = kwargs.get('down_block_res_samples')
    mid_block_res_sample = kwargs.get('mid_block_res_sample')
    noise_pred = self.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        cross_attention_kwargs=cross_attention_kwargs,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
        return_dict=False,
    )[0]
    kwargs['noise_pred']=noise_pred
    return kwargs
def controlnet_offloading(self,**kwargs):
    if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()
    return kwargs