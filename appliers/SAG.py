import torch
import torch.nn.functional as F
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
from functools import partial
def apply_SAG(pipe):
    #insert function responsible for defaults
    pipe.denoising_functions.insert(0,partial(SAG_default, pipe ))
    #replace denoiser function with another that contains it
    denoiser_index= pipe.denoising_functions.index(pipe.denoiser)
    inner_denoiser_SAG=pipe.denoiser
    pipe.inner_denoiser_SAG=inner_denoiser_SAG
    pipe.denoiser=partial(denoiser, pipe)
    pipe.denoising_functions[denoiser_index]=pipe.denoiser
    #insert do_self_attention_guidance before call_params
    call_params_index= pipe.denoising_functions.index(pipe.call_params)
    pipe.denoising_functions.insert(call_params_index,partial(do_self_attention_guidance, pipe ))
    #add pred_x0 function
    pipe.pred_x0=partial(pred_x0,pipe)
    #add pred_epsilon function
    pipe.pred_epsilon=partial(pred_epsilon,pipe)
    #add sag_masking function
    pipe.sag_masking=partial(sag_masking,pipe)
    #add SAG_self_attention before compute_previous_noisy_sample in step functions
    compute_previous_noisy_sample_index = pipe.denoising_step_functions.index(pipe.compute_previous_noisy_sample)
    pipe.denoising_step_functions.insert(compute_previous_noisy_sample_index, partial(SAG_self_attention, pipe))
    #reverse
    def remover_SAG():
        #remove SAG_self_attention before compute_previous_noisy_sample from step functions
        pipe.denoising_step_functions.pop(compute_previous_noisy_sample_index)
        #remove pred_x0 function
        delattr(pipe, f"pred_x0")
        #remove pred_epsilon function
        delattr(pipe, f"pred_epsilon")
        #remove sag_masking function
        delattr(pipe, f"sag_masking")

        #remove do_self_attention_guidance before call_params
        pipe.denoising_functions.pop(call_params_index)

        #undo replacement of denoiser function with another that contains it
        pipe.denoiser=pipe.inner_denoiser_SAG
        pipe.denoising_functions[denoiser_index]=pipe.denoiser
        delattr(pipe, f"inner_denoiser_SAG")

        #remove function responsible for defaults
        pipe.denoising_functions.pop(0)

    pipe.revert_functions.insert(0,remover_SAG)
class CrossAttnStoreProcessor:
    def __init__(self):
        self.attention_probs = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        self.attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(self.attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
def SAG_default(self,**kwargs):
    if kwargs.get('sag_scale') is None:
        kwargs['sag_scale']=0.75
    return kwargs
def do_self_attention_guidance (self, **kwargs):
    kwargs['do_self_attention_guidance']=kwargs.get('sag_scale') > 0.0
    return kwargs
def SAG_self_attention(self, i, t, **kwargs):
    do_self_attention_guidance=kwargs.get('do_self_attention_guidance')
    do_classifier_free_guidance=kwargs.get('do_classifier_free_guidance')
    latents=kwargs.get('latents')
    noise_pred=kwargs.get('noise_pred')
    noise_pred_uncond=kwargs.get('noise_pred_uncond')
    store_processor=kwargs.get('store_processor')
    map_size=kwargs.get('map_size')
    prompt_embeds=kwargs.get('prompt_embeds')
    sag_scale=kwargs.get('sag_scale')
    if do_self_attention_guidance:
        # classifier-free guidance produces two chunks of attention map
        # and we only use unconditional one according to equation (25)
        # in https://arxiv.org/pdf/2210.00939.pdf
        if do_classifier_free_guidance:
            # DDIM-like prediction of x0
            pred_x0 = self.pred_x0(latents, noise_pred_uncond, t)
            # get the stored attention maps
            uncond_attn, cond_attn = store_processor.attention_probs.chunk(2)
            # self-attention-based degrading of latents
            degraded_latents = self.sag_masking(
                pred_x0, uncond_attn, map_size, t, self.pred_epsilon(latents, noise_pred_uncond, t)
            )
            uncond_emb, _ = prompt_embeds.chunk(2)
            # forward and give guidance
            degraded_pred = self.unet(degraded_latents, t, encoder_hidden_states=uncond_emb).sample
            noise_pred += sag_scale * (noise_pred_uncond - degraded_pred)
        else:
            # DDIM-like prediction of x0
            pred_x0 = self.pred_x0(latents, noise_pred, t)
            # get the stored attention maps
            cond_attn = store_processor.attention_probs
            # self-attention-based degrading of latents
            degraded_latents = self.sag_masking(
                pred_x0, cond_attn, map_size, t, self.pred_epsilon(latents, noise_pred, t)
            )
            # forward and give guidance
            degraded_pred = self.unet(degraded_latents, t, encoder_hidden_states=prompt_embeds).sample
            noise_pred += sag_scale * (noise_pred - degraded_pred)
    return kwargs



def sag_masking(self, original_latents, attn_map, map_size, t, eps):
    # Same masking process as in SAG paper: https://arxiv.org/pdf/2210.00939.pdf
    bh, hw1, hw2 = attn_map.shape
    b, latent_channel, latent_h, latent_w = original_latents.shape
    h = self.unet.config.attention_head_dim
    if isinstance(h, list):
        h = h[-1]
    # Produce attention mask
    attn_map = attn_map.reshape(b, h, hw1, hw2)
    attn_mask = attn_map.mean(1, keepdim=False).sum(1, keepdim=False) > 1.0
    attn_mask = (
        attn_mask.reshape(b, map_size[0][0], map_size[0][1])
        .unsqueeze(1)
        .repeat(1, latent_channel, 1, 1)
        .type(attn_map.dtype)
    )
    attn_mask = F.interpolate(attn_mask, (latent_h, latent_w))

    # Blur according to the self-attention mask
    degraded_latents = gaussian_blur_2d(original_latents, kernel_size=9, sigma=1.0)
    degraded_latents = degraded_latents * attn_mask + original_latents * (1 - attn_mask)
    newt=t.view(1)
    # Noise it again to match the noise level
    degraded_latents = self.scheduler.add_noise(degraded_latents, noise=eps, timesteps=newt)

    return degraded_latents

# Modified from diffusers.schedulers.scheduling_ddim.DDIMScheduler.step
# Note: there are some schedulers that clip or do not return x_0 (PNDMScheduler, DDIMScheduler, etc.)
def pred_x0(self, sample, model_output, timestep):

    # Move the tensor from GPU to CPU (if it's not already on CPU)
    numerical_value = int(timestep.tolist())

    # Convert the tensor to a NumPy array and get the numerical value
    alpha_prod_t = self.scheduler.alphas_cumprod[numerical_value]
    beta_prod_t = 1 - alpha_prod_t
    if self.scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif self.scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        # predict V
        model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
            " or `v_prediction`"
        )

    return pred_original_sample
def pred_epsilon(self, sample, model_output, timestep):
    numerical_value = int(timestep.tolist())
    alpha_prod_t = self.scheduler.alphas_cumprod[numerical_value]

    beta_prod_t = 1 - alpha_prod_t
    if self.scheduler.config.prediction_type == "epsilon":
        pred_eps = model_output
    elif self.scheduler.config.prediction_type == "sample":
        pred_eps = (sample - (alpha_prod_t**0.5) * model_output) / (beta_prod_t**0.5)
    elif self.scheduler.config.prediction_type == "v_prediction":
        pred_eps = (beta_prod_t**0.5) * sample + (alpha_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
            " or `v_prediction`"
        )

    return pred_eps
def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img
def denoiser(self,**kwargs):
    store_processor = CrossAttnStoreProcessor()
    self.unet.mid_block.attentions[0].transformer_blocks[0].attn1.processor = store_processor
    map_size = [None]
    def get_map_size(module, input, output):
        nonlocal map_size
        map_size[0] = output[0].shape[-2:]
    kwargs['store_processor']=store_processor
    kwargs['map_size']=map_size
    hook_handle = self.unet.mid_block.attentions[0].register_forward_hook(get_map_size)
    kwargs=self.inner_denoiser_SAG(**kwargs)
    hook_handle.remove()
    return kwargs