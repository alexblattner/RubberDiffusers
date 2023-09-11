# RubberDiffusers
This project aims to solve the rigidity problem that diffusers has. Instead of creating a pipeline for each variation and combination, you can just implement it for RubberDiffusers and the user will pick the variations he wants to enable or not. This is based on the base txt2img pipeline of diffusers.

# How to use
1. install diffusers:
   pip install git+https://github.com/huggingface/diffusers
2. run examples.py
3. choose whatever appliers you want, but warning, some appliers should be applied later if you're stacking them like promptFusion. Also, if you use inpainting, you can't use img2img

# Usage warnings
Don't use inpainting and img2img together. If you want to apply promptFusion, apply it last. If you want to apply SAG, apply it last (before promptFusion or after, it doesn't matter).

# vanilla RubberDiffusers
This works exactly the same as the regular txt2img pipeline from the diffusers library. I just removed the stuff related to this paper: https://arxiv.org/pdf/2305.08891.pdf. If you want to use it, use apply_Correction.

# apply_controlnet
In order to use controlnet, you need to do the following:
```
#import controlnets
openpose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32,local_files_only=True).to('cuda')
depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float32,local_files_only=True).to('cuda')
#apply controlnet
apply_controlnet(your_pipe)
#add controlnets on your pipe
your_pipe.add_controlnet(openpose)
your_pipe.add_controlnet(depth)
#all settings for controlnet pipelines
image=your_pipe("some guy on the beach",controlnet_image=[openpose_image,depth_image],controlnet_conditioning_scale=[0.5,0.5],control_guidance_start=0.0,control_guidance_end=1.0,guess_mode=False).images[0]
#if you don't want to use multicontrolnet, you can do this:
#image=your_pipe("some guy on the beach",controlnet_image=openpose_image,controlnet_conditioning_scale=0.5).images[0]
```
Default values:
guess_mode=False
control_guidance_start=0.0
control_guidance_end=1.0
controlnet_conditioning_scale=1.0 for each controlnet
Requirements:
controlnet_image=single image or list of images

# apply_Correction
It essentially applies https://arxiv.org/pdf/2305.08891.pdf. No other changes are made on the pipeline. So it ends up being the same as the one in diffusers.

# apply_img2img
