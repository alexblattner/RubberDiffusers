# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from packaging import version
import numpy as np
import torch.nn.functional as F
import PIL
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer,CLIPVisionModelWithProjection
import torchvision
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.utils import deprecate, is_accelerate_available, is_accelerate_version, logging, replace_example_docstring,PIL_INTERPOLATION
from diffusers.configuration_utils import FrozenDict
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler,KarrasDiffusionSchedulers,DDPMScheduler
from functools import partial

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
class StableDiffusionRubberPipeline(StableDiffusionPipeline):
    revert_functions = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        self.before_init()
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         safety_checker, feature_extractor, image_encoder,requires_safety_checker)

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.denoising_functions = [self.dimension, self.checker, self.determine_batch_size, self.call_params, self.encode_input,
                                    self.prepare_timesteps, self.prepare_latent_var, self.prepare_extra_kwargs,self.guidance_scale_embedding, self.denoiser, self.postProcess]
        self.denoising_step_functions = [self.expand_latents,self.scale_model_input, self.predict_noise_residual,
                                          self.perform_guidance, self.compute_previous_noisy_sample, self.call_callback]
        # The original functions were replaced with a bit different functions
        # self.denoising_step_functions = [
        #     self.latent_setter, self.predictor, self.step_performer, self.callbacker]
        self.after_init()

    def before_init(self):
        return

    def after_init(self):
        return

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        eta: float = 0.0,
        return_dict: bool = True,
        callback_steps: int = 1,
        num_images_per_prompt: Optional[int] = 1,
        output_type: Optional[str] = "pil",
        clip_skip: Optional[int] = None,
        cross_attention_kwargs: Optional[dict]=dict(),
        
        **kwargs
    ):
        kwargs['prompt'] = prompt
        kwargs['num_inference_steps'] = num_inference_steps
        kwargs['guidance_scale'] = guidance_scale
        kwargs['guidance_rescale'] = guidance_rescale
        kwargs['eta'] = eta
        kwargs['return_dict'] = return_dict
        kwargs['callback_steps'] = callback_steps
        kwargs['num_images_per_prompt'] = num_images_per_prompt
        kwargs['output_type'] = output_type
        kwargs['clip_skip']=clip_skip
        kwargs['cross_attention_kwargs']=cross_attention_kwargs

        for func in self.denoising_functions:
            kwargs = func(**kwargs)
        return kwargs

    def dimension(self, **kwargs):
        height = kwargs.get('height') or self.unet.config.sample_size * self.vae_scale_factor
        width = kwargs.get('width') or self.unet.config.sample_size * self.vae_scale_factor
        return {'height': height, 'width': width, **kwargs}

    def checker(self, **kwargs):
        self.check_inputs(
            kwargs.get('prompt'), kwargs.get('height'), kwargs.get('width'), kwargs.get('callback_steps'), kwargs.get(
                'negative_prompt'), kwargs.get('prompt_embeds'), kwargs.get('negative_prompt_embeds')
        )
        return kwargs
    def determine_batch_size(self, **kwargs):
        self._guidance_scale = kwargs['guidance_scale']
        self._guidance_rescale = kwargs['guidance_rescale']
        self._clip_skip = kwargs['clip_skip']
        self._cross_attention_kwargs = kwargs['cross_attention_kwargs']
        if kwargs.get('prompt') is not None and isinstance(kwargs.get('prompt'), str):
            batch_size = 1
        elif kwargs.get('prompt') is not None and isinstance(kwargs.get('prompt'), list):
            batch_size = len(kwargs.get('prompt'))
        else:
            batch_size = kwargs.get('prompt_embeds').shape[0]
        kwargs['batch_size']= batch_size
        return kwargs
    def call_params(self, **kwargs):
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = kwargs.get('guidance_scale') > 1.0
        
        return {'device': device, 'do_classifier_free_guidance': do_classifier_free_guidance, **kwargs}

    def encode_input(self, **kwargs):
        text_encoder_lora_scale = (
            kwargs.get('cross_attention_kwargs').get("scale", None) if kwargs.get(
                'cross_attention_kwargs') is not None else None
        )
        prompt = kwargs.get('prompt')
        device = kwargs.get('device')
        num_images_per_prompt = kwargs.get('num_images_per_prompt')
        do_classifier_free_guidance = kwargs.get('do_classifier_free_guidance')
        negative_prompt = kwargs.get('negative_prompt')
        prompt_embeds = kwargs.get('prompt_embeds')
        negative_prompt_embeds = kwargs.get('negative_prompt_embeds')
        clip_skip=kwargs.get('clip_skip')
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        kwargs['prompt_embeds'] = prompt_embeds
        kwargs['negative_prompt_embeds']=negative_prompt_embeds
        kwargs['dtype'] = prompt_embeds.dtype
        return {'text_encoder_lora_scale': text_encoder_lora_scale, 'prompt_embeds': prompt_embeds, **kwargs}

    def prepare_timesteps(self, **kwargs):
        kwargs['timesteps'], kwargs['num_inference_steps'] = retrieve_timesteps(self.scheduler, kwargs.get('num_inference_steps'), kwargs.get('device'), kwargs.get('timesteps'))
        return kwargs

    def prepare_latent_var(self, **kwargs):
        num_channels_latents = self.unet.config.in_channels
        
        latents = self.prepare_latents(
            kwargs.get('batch_size') * kwargs.get('num_images_per_prompt'),
            num_channels_latents,
            kwargs.get('height'),
            kwargs.get('width'),
            kwargs.get('dtype'),
            kwargs.get('device'),
            kwargs.get('generator'),
            kwargs.get('latents'),
        )
        return {'num_channels_latents': num_channels_latents, 'latents': latents, **kwargs}

    def prepare_extra_kwargs(self, **kwargs):
        generator = kwargs.get('generator')
        eta = kwargs.get('eta')
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        return {'extra_step_kwargs': extra_step_kwargs, **kwargs}

    def guidance_scale_embedding(self,**kwargs):
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(kwargs.get('batch_size') * kwargs.get('num_images_per_prompt'))
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=kwargs.get('device'), dtype=kwargs.get('dtype'))
        kwargs['timestep_cond']=timestep_cond
        self._num_timesteps = len(kwargs.get('timesteps'))
        return kwargs
    
    def expand_latents(self, i, t, **kwargs):
        latents = kwargs.get('latents')
        do_classifier_free_guidance = kwargs.get('do_classifier_free_guidance')
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        lss=kwargs.get("last_step_strength")
        num_inference_steps=kwargs.get('num_inference_steps')
        stop_step=kwargs.get('stop_step')
        if lss is not None and (i==num_inference_steps or (stop_step is not None and i==stop_step-1)):
            latent_model_input*=lss
        kwargs['latent_model_input'] = latent_model_input
        return kwargs
    def scale_model_input(self,i,t,**kwargs):
        kwargs['latent_model_input'] = self.scheduler.scale_model_input(kwargs.get('latent_model_input'), t)
        return kwargs
    def predict_noise_residual(self, i, t, **kwargs):
        prompt_embeds = kwargs.get('prompt_embeds')
        cross_attention_kwargs = kwargs.get('cross_attention_kwargs')
        latent_model_input = kwargs.get('latent_model_input')
        added_cond_kwargs=kwargs.get('added_cond_kwargs')
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, timestep_cond=kwargs.get('timestep_cond'),
                        cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs,return_dict=False)[0]
        kwargs['noise_pred'] = noise_pred
        return kwargs

    def perform_guidance(self, i, t, **kwargs):
        do_classifier_free_guidance = kwargs.get('do_classifier_free_guidance')
        guidance_scale = kwargs.get('guidance_scale')
        rescale_noise_cfg = kwargs.get('rescale_noise_cfg')
        noise_pred = kwargs.get('noise_pred')
        
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            kwargs['noise_pred_text']=noise_pred_text
        kwargs['noise_pred'] = noise_pred
        kwargs['noise_pred_uncond'] = noise_pred_uncond
        return kwargs

    def compute_previous_noisy_sample(self, i, t, **kwargs):
        latents = kwargs.get('latents')
        noise_pred = kwargs.get('noise_pred')

        latents = self.scheduler.step(noise_pred, t, latents, **kwargs.get('extra_step_kwargs'), return_dict=False)[0]
        kwargs['latents'] = latents
        return kwargs

    def call_callback(self, i, t, **kwargs):
        latents = kwargs.get('latents')
        timesteps = kwargs.get('timesteps')
        num_warmup_steps = len(timesteps) - kwargs.get('num_inference_steps') * self.scheduler.order
        callback = kwargs.get('callback')
        callback_steps = kwargs.get('callback_steps')
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        kwargs['latents'] = latents
        return kwargs

    
    def denoiser(self, **kwargs):
        timesteps=kwargs.get('timesteps')

        with self.progress_bar(total=kwargs.get('num_inference_steps')) as progress_bar:
            for i, t in enumerate(timesteps):
                if kwargs.get('stop_step') is not None and i == kwargs.get('stop_step'):
                    break
                for func in self.denoising_step_functions:
                    kwargs =func(i,t, **kwargs)

                if i == len(kwargs.get('timesteps')) - 1 or ((i + 1) > len(kwargs.get('timesteps')) - kwargs.get('num_inference_steps') * self.scheduler.order and 
                    (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        return kwargs

    def postProcess(self, **kwargs):
        output_type = kwargs.get("output_type")
        latents = kwargs.get("latents")
        return_dict = True
        device = self._execution_device
        prompt_embeds = kwargs.get("prompt_embeds")
        dtype=kwargs.get('dtype')
        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False)[0]
            if not kwargs.get('nsfw'):
                image, has_nsfw_concept = self.run_safety_checker(image, device, dtype)
            else:
                has_nsfw_concept=None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    def revert(self):
        print("revert")
        for func in self.revert_functions:
            func()
            del func
        self.revert_functions=[]