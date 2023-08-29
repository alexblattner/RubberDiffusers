import torch
from diffusers import ControlNetModel
from typing import List, Optional
from PIL import Image
import time
from io import BytesIO
import numpy as np
import os
import random
# Initialize the Celery app
from utils import kohya_lora_loader
from utils.rubberDiffusers import StableDiffusionRubberPipeline
from appliers.SAG import apply_SAG
from appliers.inpainting import apply_inpainting
from appliers.correctedScheduling import apply_Correction
from appliers.promptFusion import apply_promptFusion
from appliers.img2img import apply_img2img
from appliers.controlnet import apply_controlnet
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32,local_files_only=True).to('cuda')
pipe=StableDiffusionRubberPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32,safety_checker=None, requires_safety_checker=False,
).to("cuda")

#in this example, I am using diffusion correction, SAG, inpainting and controlnet. You can uncomment to use more things and the applyance order matters

# Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
pipe=apply_Correction(pipe)
pipe=apply_SAG(pipe)
pipe=apply_inpainting(pipe)
# pipe=apply_img2img(pipe)
pipe=apply_controlnet(pipe)
# pipe=apply_promptFusion(pipe)
generator = torch.manual_seed(2733424006)
# def printer(i, t, latents):
#     print("bbrrrrr")
prompt="woman, 4k, cyberpunk"
#you should uncomment prompt fusion for that:
#prompt=[["woman, 4k, cyberpunk",5],["woman, 4k, beautiful day at the park",20]]

buffer=open('img0.png', 'rb')
buffer.seek(0)
image_bytes = buffer.read()
imageC = Image.open(BytesIO(image_bytes))

#this should be applied with controlnet only!
pipe.add_controlnet(controlnet)

buffer=open('tmask.png', 'rb')
buffer.seek(0)
image_bytes = buffer.read()
mask_image = Image.open(BytesIO(image_bytes))
buffer=open('img111.png', 'rb')
buffer.seek(0)
image_bytes = buffer.read()
image = Image.open(BytesIO(image_bytes))
start_time = time.time()
fimage=pipe(
    prompt,
    prompt_embeds=None,
    num_inference_steps=20,
    generator=generator,
    # callback=printer,
    controlnet_conditioning_scale=[1.0],
    controlnet_image=[imageC],
    mask_image=mask_image,
    image=image,
    strength=0.75,
    sag_scale=0.75,
).images[0]
end_time = time.time()
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))
# # print(image)
fimage.save('img11.png', format='PNG')