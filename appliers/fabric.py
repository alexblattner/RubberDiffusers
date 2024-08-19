import torch
import torch.nn as nn
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
import types
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
import numpy as np
from diffusers.models.attention_processor import LoRAAttnProcessor, AttnProcessor2_0, AttnProcessor
from diffusers.models.attention import BasicTransformerBlock, Attention
from functools import partial
import time
def find_index(functions,name):
    target_function_index = None
    for index, func in enumerate(functions):
        if (hasattr(func, "__name__") and func.__name__ == name) or (hasattr(func, "func") and hasattr(func.func, "__name__") and func.func.__name__ == name):
            target_function_index = index
            break
    return target_function_index
def apply_fabric(pipe):
    pipe.denoising_functions.insert(0, partial(fabric_default, pipe))
    pipe.unet_forward_with_cached_hidden_states=types.MethodType(unet_forward_with_cached_hidden_states,pipe)
    pipe.preprocess_feedback_images=types.MethodType(preprocess_feedback_images,pipe)
    pipe.get_unet_hidden_states=types.MethodType(get_unet_hidden_states,pipe)

    #add null_prompt_embbeder after encode_input and fabric_pos_neg_latents before
    encode_input_index = find_index(pipe.denoising_functions,"encode_input")
    pipe.denoising_functions.insert(encode_input_index+1, partial(null_prompt_embbeder, pipe))
    pipe.denoising_functions.insert(encode_input_index,partial(fabric_pos_neg_latents, pipe))

    #add ref_idx_setter before the denoiser
    denoiser_index=find_index(pipe.denoising_functions,"denoiser")
    pipe.denoising_functions.insert(denoiser_index,partial(ref_idx_setter, pipe))

    #replace predict_noise_residual function with a new predict_noise_residual function
    predict_noise_residual_index=find_index(pipe.denoising_step_functions,"predict_noise_residual")
    pipe.fabric_stored_predict_noise_residual=pipe.predict_noise_residual
    pipe.predict_noise_residual=partial(predict_noise_residual,pipe)
    pipe.denoising_step_functions[predict_noise_residual_index]=pipe.predict_noise_residual
    
    #reverse
    def remover_fabric():
        #undo replacement of predict_noise_residual function with a new predict_noise_residual function
        pipe.predict_noise_residual=pipe.fabric_stored_predict_noise_residual
        delattr(pipe, f"fabric_stored_predict_noise_residual")
        pipe.denoising_step_functions[predict_noise_residual_index]=pipe.predict_noise_residual
        #remove ref_idx_setter before the denoiser
        pipe.denoising_functions.pop(denoiser_index)
        #remove null_prompt_embbeder after encode_input and fabric_pos_neg_latents before
        pipe.denoising_functions.pop(encode_input_index)
        pipe.denoising_functions.pop(encode_input_index+1)

        delattr(pipe, f"get_unet_hidden_states")
        delattr(pipe, f"preprocess_feedback_images")
        delattr(pipe, f"unet_forward_with_cached_hidden_states")
        pipe.denoising_functions.pop(0)
    pipe.revert_functions.insert(0,remover_fabric)
def preprocess_feedback_images(self, images, dim, device, dtype, generator) -> torch.tensor:
    
    images_t = [self.image_processor.preprocess(img,height=dim[0], width=dim[1]).to(dtype=dtype)[0] for img in images]
    images_t = torch.stack(images_t).to(device)
    latents = self.vae.config.scaling_factor * self.vae.encode(images_t).latent_dist.sample(generator)

    return torch.cat([latents], dim=0)
def fabric_pos_neg_latents(self,**kwargs):
    liked=kwargs['liked_images']
    disliked=kwargs['disliked_images']
    dtype=kwargs.get('dtype')
    height=kwargs.get('height')
    width=kwargs.get('width')
    device=kwargs.get('device')
    generator=kwargs.get('generator')
    kwargs['positive_latents'] = (
        self.preprocess_feedback_images(liked, (height, width), device, dtype, generator)
        if liked and len(liked) > 0
        else torch.tensor(
            [],
            device=device,
            dtype=dtype,
        )
    )
    kwargs['negative_latents'] = (
        self.preprocess_feedback_images(disliked, (height, width), device, dtype, generator)
        if disliked and len(disliked) > 0
        else torch.tensor(
            [],
            device=device,
            dtype=dtype,
        )
    )
    kwargs['adjustor_tensor'] = torch.cat([kwargs.get('positive_latents'), kwargs.get('negative_latents')], dim=0)
    return kwargs
def fabric_default(self,**kwargs):
    if kwargs.get('liked_images') is None:
        kwargs['liked_images']=[]
    if kwargs.get('disliked_images') is None:
        kwargs['disliked_images']=[]
    if kwargs.get('feedback_start_ratio') is None:
        kwargs['feedback_start_ratio']=0.33
    if kwargs.get('feedback_end_ratio') is None:
        kwargs['feedback_end_ratio']=0.66
    if kwargs.get('min_weight') is None:
        kwargs['min_weight']=0.1
    if kwargs.get('max_weight') is None:
        kwargs['max_weight']=1.0
    if kwargs.get('neg_scale') is None:
        kwargs['neg_scale']=0.5
    if kwargs.get('pos_bottleneck_scale') is None:
        kwargs['pos_bottleneck_scale']=1.0
    if kwargs.get('neg_bottleneck_scale') is None:
        kwargs['neg_bottleneck_scale']=1.0
    return kwargs
def shape_setter(self, **kwargs):
    batch_size=kwargs.get('batch_size')
    num_images=kwargs.get('num_images')
    height=kwargs.get('height')
    width=kwargs.get('width')
    kwargs['shape'] = (
        batch_size * num_images,
        self.unet.config.in_channels,
        height // self.vae_scale_factor,
        width // self.vae_scale_factor,
    )
    return kwargs
def null_prompt_embbeder(self, **kwargs):
    device = kwargs.get('device')
    clip_skip=kwargs.get('clip_skip')
    dtype=kwargs.get('dtype')
    null_tokens = self.tokenizer(
        [""],
        return_tensors="pt",
        max_length=self.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
    )

    if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
        attention_mask = null_tokens.attention_mask.to(device)
    else:
        attention_mask = None

    null_prompt_emb = self.text_encoder(
        input_ids=null_tokens.input_ids.to(device),
        attention_mask=attention_mask,
    ).last_hidden_state

    kwargs['null_prompt_emb'] = null_prompt_emb.to(device=device, dtype=dtype)
    return kwargs
def ref_idx_setter(self, **kwargs):
    timesteps=kwargs.get('timesteps')
    feedback_start_ratio=kwargs.get('feedback_start_ratio')
    feedback_end_ratio=kwargs.get('feedback_end_ratio')
    kwargs['ref_start_idx'] = round(len(timesteps) * feedback_start_ratio)
    kwargs['ref_end_idx'] = round(len(timesteps) * feedback_end_ratio)
    return kwargs

def unet_forward_with_cached_hidden_states(
        self,
        z_all,
        t,
        prompt_embd,
        cached_pos_hiddens: Optional[List[torch.Tensor]] = None,
        cached_neg_hiddens: Optional[List[torch.Tensor]] = None,
        unet_kwargs: Optional[Dict[str, Any]] = None,
        pos_weights=(0.8, 0.8),
        neg_weights=(0.5, 0.5),
):

    def get_weights(hiddens, cached_hiddens, weight, module):
        if cached_hiddens is None:
            return module.old_forward(hiddens)
        cached_hs = cached_hiddens.pop(0).to(hiddens.device)
        combined_hs = torch.cat([hiddens, cached_hs], dim=1)
        batch_size, d_model = hiddens.shape[:2]
        weights = torch.ones(batch_size, d_model, device=hiddens.device, dtype=hiddens.dtype)
        weights = weights.repeat(1, 1 + cached_hs.shape[1] // d_model)
        weights[:, d_model:] = weight
        if isinstance(module, AttnProcessor2_0):
            def new_prepare_attention_mask(self, attention_mask, target_length, batch_size, out_dim=3):
                if weights is not None:
                    if weights.shape[0] != 1:
                        weights = weights.repeat_interleave(self.heads, dim=0)
                    attention_mask = weights.log().view(1, self.heads, 1, -1)
                return attention_mask
            module.old_prepare_attention_mask=module.prepare_attention_mask
            module.prepare_attention_mask=new_prepare_attention_mask.__get__(module)
        elif isinstance(module, AttnProcessor):
            module.old_get_attention_scores=module.get_attention_scores
            def new_get_attention_scores(self, query, key, attention_mask=None):
                attention_probs = module.old_get_attention_scores(query, key, attention_mask=None)
                if weights is not None:
                    if weights.shape[0] != 1:
                        weights = weights.repeat_interleave(self.heads, dim=0)
                    attention_probs = attention_probs * weights[:, None]
                    attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)
                return attention_probs
            module.get_attention_scores=new_get_attention_scores.__get__(module)

        return module.processor(module, hiddens, encoder_hidden_states=combined_hs, attention_mask=weights)

    if cached_pos_hiddens is None and cached_neg_hiddens is None:
        unet_kwargs['encoder_hidden_states']=prompt_embd
        return self.unet(z_all, t, **unet_kwargs)
    local_pos_weights = torch.linspace(*pos_weights, steps=len(self.unet.down_blocks) + 1)[:-1].tolist()
    local_neg_weights = torch.linspace(*neg_weights, steps=len(self.unet.down_blocks) + 1)[:-1].tolist()
    # Pre-calculate blocks and weights
    all_blocks = self.unet.down_blocks + [self.unet.mid_block] + self.unet.up_blocks
    all_pos_weights = local_pos_weights + [pos_weights[1]] + local_pos_weights[::-1]
    all_neg_weights = local_neg_weights + [neg_weights[1]] + local_neg_weights[::-1]

    # Extract all relevant modules only once
    all_modules = [mod for block in all_blocks for mod in block.modules() if isinstance(mod, BasicTransformerBlock)]
    zipper=zip(all_modules, all_pos_weights, all_neg_weights)
    for module, pos_weight, neg_weight in zipper:
        def new_forward(self, hidden_states, **kwargs):
            cond_hiddens, uncond_hiddens = hidden_states.chunk(2, dim=0)
            out_pos = get_weights(cond_hiddens, cached_pos_hiddens, pos_weight, self)
            out_neg = get_weights(uncond_hiddens, cached_neg_hiddens, neg_weight, self)
            return torch.cat([out_pos, out_neg], dim=0)
        
        module.attn1.old_forward = module.attn1.forward
        module.attn1.forward = new_forward.__get__(module.attn1)
    unet_kwargs['encoder_hidden_states']=prompt_embd
    out = self.unet(z_all, t, **unet_kwargs)

    for module in all_modules:
        if hasattr(module.attn1, "old_forward"):
            module.attn1.forward = module.attn1.old_forward
            del module.attn1.old_forward
            if isinstance(module, AttnProcessor2_0):
                module.prepare_attention_mask=module.old_prepare_attention_mask
                del module.old_prepare_attention_mask
            elif isinstance(module, AttnProcessor):
                module.get_attention_scores=module.old_get_attention_scores
                del module.old_get_attention_scores

    return out

def predict_noise_residual(self,i, t, **kwargs):
    latent_model_input=kwargs.get('latent_model_input')
    prompt_embeds=kwargs.get('prompt_embeds')
    adjustor_tensor=kwargs.get('adjustor_tensor')
    ref_start_idx=kwargs.get('ref_start_idx')
    ref_end_idx=kwargs.get('ref_end_idx')
    positive_latents=kwargs.get('positive_latents')
    negative_latents=kwargs.get('negative_latents')
    num_images=kwargs.get('num_images_per_prompt')
    cross_attention_kwargs = kwargs.get('cross_attention_kwargs')
    unet_kwargs=kwargs.get('unet_kwargs')
    if i >= ref_start_idx and i <= ref_end_idx:
        weight = kwargs.get('max_weight')
    else:
        weight = kwargs.get('min_weight')
    pos_ws = (weight, weight * kwargs.get('pos_bottleneck_scale'))
    neg_ws = (weight * kwargs.get('neg_scale'), weight * kwargs.get('neg_scale') * kwargs.get('neg_bottleneck_scale'))

    if adjustor_tensor.size(0) > 0 and weight > 0:
        noise = torch.randn_like(adjustor_tensor)
        newt=t if t.dim() >= 1 else t.unsqueeze(0)
        adjustor_tensor_noised = self.scheduler.add_noise(adjustor_tensor, noise, newt)

        ref_prompt_embd = torch.cat([kwargs.get('null_prompt_emb')] * (positive_latents.size(0) + negative_latents.size(0)), dim=0)

        cached_hidden_states = self.get_unet_hidden_states(
            adjustor_tensor_noised, t, ref_prompt_embd,unet_kwargs
        )

        n_pos, n_neg = positive_latents.shape[0], negative_latents.shape[0]
        cached_pos_hs, cached_neg_hs = [], []
        for hs in cached_hidden_states:
            cached_pos, cached_neg = hs.split([n_pos, n_neg], dim=0)
            cached_pos = cached_pos.view(
                1, -1, *cached_pos.shape[2:]
            ).expand(num_images, -1, -1)
            cached_neg = cached_neg.view(
                1, -1, *cached_neg.shape[2:]
            ).expand(num_images, -1, -1)
            cached_pos_hs.append(cached_pos)
            cached_neg_hs.append(cached_neg)

        if n_pos == 0:
            cached_pos_hs = None
        if n_neg == 0:
            cached_neg_hs = None
    else:
        cached_pos_hs, cached_neg_hs = None, None
    kwargs['noise_pred'] = self.unet_forward_with_cached_hidden_states(
        latent_model_input,
        t,
        prompt_embd=prompt_embeds,
        cached_pos_hiddens=cached_pos_hs,
        cached_neg_hiddens=cached_neg_hs,
        pos_weights=pos_ws,
        neg_weights=neg_ws,
        unet_kwargs=unet_kwargs
    )[0]
    return kwargs
def get_unet_hidden_states(self, z_all, t, prompt_embd,unet_kwargs):
        cached_hidden_states = []
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):

                def new_forward(self, hidden_states, *args, **kwargs):
                    cached_hidden_states.append(hidden_states.clone().detach().cpu())
                    return self.old_forward(hidden_states, *args, **kwargs)

                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)
        unet_kwargs['encoder_hidden_states']=prompt_embd
        # run forward pass to cache hidden states, output can be discarded
        _ = self.unet(z_all, t, **unet_kwargs)

        # restore original forward pass
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1.old_forward
                del module.attn1.old_forward

        return cached_hidden_states