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
def apply_fabric(pipe):
    pipe.denoising_functions.insert(0, partial(fabric_default, pipe))
    pipe.unet_forward_with_cached_hidden_states=types.MethodType(unet_forward_with_cached_hidden_states,pipe)
    pipe.preprocess_feedback_images=types.MethodType(preprocess_feedback_images,pipe)
    pipe.get_unet_hidden_states=types.MethodType(get_unet_hidden_states,pipe)

    #add null_prompt_embbeder after encode_input and fabric_pos_neg_latents before
    encode_input_index = pipe.denoising_functions.index(pipe.encode_input)
    pipe.denoising_functions.insert(encode_input_index+1, partial(null_prompt_embbeder, pipe))
    pipe.denoising_functions.insert(encode_input_index,partial(fabric_pos_neg_latents, pipe))

    #add ref_idx_setter before the denoiser
    denoiser_index=pipe.denoising_functions.index(pipe.denoiser)
    pipe.denoising_functions.insert(denoiser_index,partial(ref_idx_setter, pipe))

    #replace predict_noise_residual function with a new predict_noise_residual function
    predict_noise_residual_index=pipe.denoising_step_functions.index(pipe.predict_noise_residual)
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
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        pos_weights=(0.8, 0.8),
        neg_weights=(0.5, 0.5),
):

    attn_with_weights = AttnProcessor()

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
                        weights = weights.repeat_interleave(attn.heads, dim=0)
                    attention_probs = attention_probs * weights[:, None]
                    attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)
                return attention_probs
            module.get_attention_scores=new_get_attention_scores.__get__(module)

        return module.processor(module, hiddens, encoder_hidden_states=combined_hs, attention_mask=weights)

    if cached_pos_hiddens is None and cached_neg_hiddens is None:
        return self.unet(z_all, t, encoder_hidden_states=prompt_embd, 
                        cross_attention_kwargs=cross_attention_kwargs, return_dict=False)
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
    out = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

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



class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        weights=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if weights is not None:
            if weights.shape[0] != 1:
                weights = weights.repeat_interleave(attn.heads, dim=0)
            w_bias = weights.log()
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            if attention_mask.dtype == torch.bool:
                attention_mask.masked_fill(not attention_mask, -float('inf'))
            attention_mask += w_bias
        else:
            attention_mask = w_bias
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # if weights is not None:
        #     if weights.shape[0] != 1:
        #         weights = weights.repeat_interleave(attn.heads, dim=0)
        #     w_bias = weights.log()
        #     if attention_mask:
        #         if attention_mask.dtype == torch.bool:
        #             attention_mask.masked_fill(not attention_mask, -float('inf'))
        #         attention_mask += w_bias
        #     else:
        #         attention_mask = w_bias
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
def predict_noise_residual(self,i, t, **kwargs):
    latent_model_input=kwargs.get('latent_model_input')
    prompt_embeds=kwargs.get('prompt_embeds')
    adjustor_tensor=kwargs.get('adjustor_tensor')
    ref_start_idx=kwargs.get('ref_start_idx')
    ref_end_idx=kwargs.get('ref_end_idx')
    positive_latents=kwargs.get('positive_latents')
    negative_latents=kwargs.get('negative_latents')
    num_images=kwargs.get('num_images')
    cross_attention_kwargs = kwargs.get('cross_attention_kwargs')
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
            adjustor_tensor_noised, t, ref_prompt_embd
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
        cross_attention_kwargs=cross_attention_kwargs
    )[0]
    return kwargs
def get_unet_hidden_states(self, z_all, t, prompt_embd):
        cached_hidden_states = []
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):

                def new_forward(self, hidden_states, *args, **kwargs):
                    cached_hidden_states.append(hidden_states.clone().detach().cpu())
                    return self.old_forward(hidden_states, *args, **kwargs)

                module.attn1.old_forward = module.attn1.forward
                module.attn1.forward = new_forward.__get__(module.attn1)

        # run forward pass to cache hidden states, output can be discarded
        _ = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

        # restore original forward pass
        for module in self.unet.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1.old_forward
                del module.attn1.old_forward

        return cached_hidden_states
class FabricCrossAttnProcessor:
    def __init__(self):
        self.attntion_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        weights=None,
        lora_scale=1.0,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if isinstance(attn.processor, LoRAAttnProcessor):
            query = attn.to_q(hidden_states) + lora_scale * attn.processor.to_q_lora(hidden_states)
        else:
            query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if isinstance(attn.processor, LoRAAttnProcessor):
            key = attn.to_k(encoder_hidden_states) + lora_scale * attn.processor.to_k_lora(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states) + lora_scale * attn.processor.to_v_lora(encoder_hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if weights is not None:
            if weights.shape[0] != 1:
                weights = weights.repeat_interleave(attn.heads, dim=0)
            attention_probs = attention_probs * weights[:, None]
            attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        if isinstance(attn.processor, LoRAAttnProcessor):
            hidden_states = attn.to_out[0](hidden_states) + lora_scale * attn.processor.to_out_lora(hidden_states)
        else:
            hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
# noise_cond, noise_uncond = unet_out.chunk(2)
# guidance = noise_cond - noise_uncond
# noise_pred = noise_uncond + guidance_scale * guidance
# z = self.scheduler.step(noise_pred, t, z).prev_sample

# if i == len(timesteps) - 1 or (
#     (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
# ):
#     pbar.update()





# bad version
# def unet_forward_with_cached_hidden_states(
#     self,
#     z_all,
#     t,
#     prompt_embd,
#     cached_pos_hiddens: Optional[List[torch.Tensor]] = None,
#     cached_neg_hiddens: Optional[List[torch.Tensor]] = None,
#     cross_attention_kwargs: Optional[Dict[str, Any]] = None,
#     pos_weights=(0.8, 0.8),
#     neg_weights=(0.5, 0.5),
# ):
#     if cached_pos_hiddens is None and cached_neg_hiddens is None:
#         return self.unet(z_all, t, encoder_hidden_states=prompt_embd, 
#                     cross_attention_kwargs=cross_attention_kwargs, return_dict=False)

#     local_pos_weights = torch.linspace(*pos_weights, steps=len(self.unet.down_blocks) + 1)[:-1].tolist()
#     local_neg_weights = torch.linspace(*neg_weights, steps=len(self.unet.down_blocks) + 1)[:-1].tolist()
#     for block, pos_weight, neg_weight in zip(
#         self.unet.down_blocks + [self.unet.mid_block] + self.unet.up_blocks,
#         local_pos_weights + [pos_weights[1]] + local_pos_weights[::-1],
#         local_neg_weights + [neg_weights[1]] + local_neg_weights[::-1],
#     ):
#         for module in block.modules():
#             if isinstance(module, BasicTransformerBlock):
#                 print(module)

#                 def new_forward(
#                     self,
#                     hidden_states,
#                     pos_weight=pos_weight,
#                     neg_weight=neg_weight,
#                     **kwargs,
#                 ):
#                     cond_hiddens, uncond_hiddens = hidden_states.chunk(2, dim=0)
#                     batch_size, d_model = cond_hiddens.shape[:2]
#                     device, dtype = hidden_states.device, hidden_states.dtype

#                     weights = torch.ones(batch_size, d_model, device=device, dtype=dtype)
#                     out_pos = self.old_forward(hidden_states)
#                     out_neg = self.old_forward(hidden_states)

#                     if cached_pos_hiddens is not None:
#                         cached_pos_hs = cached_pos_hiddens.pop(0).to(hidden_states.device)
#                         cond_pos_hs = torch.cat([cond_hiddens, cached_pos_hs], dim=1)
#                         pos_weights = weights.clone().repeat(1, 1 + cached_pos_hs.shape[1] // d_model)
#                         pos_weights[:, d_model:] = pos_weight
#                         attn_with_weights = FabricCrossAttnProcessor()
#                         out_pos = attn_with_weights(
#                             self,
#                             cond_hiddens,
#                             encoder_hidden_states=cond_pos_hs,
#                             weights=pos_weights,
#                         )
#                     else:
#                         out_pos = self.old_forward(cond_hiddens)

#                     if cached_neg_hiddens is not None:
#                         cached_neg_hs = cached_neg_hiddens.pop(0).to(hidden_states.device)
#                         uncond_neg_hs = torch.cat([uncond_hiddens, cached_neg_hs], dim=1)
#                         neg_weights = weights.clone().repeat(1, 1 + cached_neg_hs.shape[1] // d_model)
#                         neg_weights[:, d_model:] = neg_weight
#                         attn_with_weights = FabricCrossAttnProcessor()
#                         out_neg = attn_with_weights(
#                             self,
#                             uncond_hiddens,
#                             encoder_hidden_states=uncond_neg_hs,
#                             weights=neg_weights,
#                         )
#                     else:
#                         out_neg = self.old_forward(uncond_hiddens)

#                     out = torch.cat([out_pos, out_neg], dim=0)
#                     return out

#                 module.attn1.old_forward = module.attn1.forward
#                 module.attn1.forward = new_forward.__get__(module.attn1)

#     out = self.unet(z_all, t, encoder_hidden_states=prompt_embd)

#     # restore original forward pass
#     for module in self.unet.modules():
#         if isinstance(module, BasicTransformerBlock):
#             module.attn1.forward = module.attn1.old_forward
#             del module.attn1.old_forward

#     return out