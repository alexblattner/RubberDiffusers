# RubberDiffusers
This project aims to solve the rigidity problem that diffusers has. Instead of creating a pipeline for each variation and combination, you can just implement it for RubberDiffusers and the user will pick the variations he wants to enable or not. This is based on the base txt2img pipeline of diffusers.

There's a special parameter in this pipeline called "stop_step". It's the exact step you want the denoising to stop at.

# How to use
1. install diffusers:
   pip install git+https://github.com/huggingface/diffusers
2. run examples.py
3. choose whatever appliers you want, but warning, some appliers should be applied later if you're stacking them like promptFusion. Also, if you use inpainting, you can't use img2img

or copy this (change whatever you want, it works just like diffusers)
```
from rubberDiffusers import StableDiffusionRubberPipeline
pipe=StableDiffusionRubberPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32,local_files_only=True,safety_checker=None, requires_safety_checker=False,
)
```

# Usage warnings
Don't use inpainting and img2img together. If you want to apply promptFusion, apply it last. If you want to apply SAG, apply it last (before promptFusion or after, it doesn't matter).

# vanilla RubberDiffusers
This works exactly the same as the regular txt2img pipeline from the diffusers library. I just removed the stuff related to this paper: https://arxiv.org/pdf/2305.08891.pdf. If you want to use it, use apply_Correction.
```
pipe = StableDiffusionRubberPipeline.from_single_file(
   "your local model or use from_pretrained idk"
)
images=pipe("your prompt like usual").images
```
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
#load relevant images
buffer=open('openpose.png', 'rb') #this does not exist and is purely an example
buffer.seek(0)
image_bytes = buffer.read()
openpose_image = Image.open(BytesIO(image_bytes))
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

loading the controlnets

# apply_Correction
It essentially applies https://arxiv.org/pdf/2305.08891.pdf. No other changes are made on the pipeline. So it ends up being the same as the one in diffusers. To use it:
```
apply_Correction(your_pipe)
image=your_pipe("a dog",guidance_rescale=0.5).images[0]
```
Default values:

guidance_rescale=0.0

# apply_dynamicThreasholding
this applies this: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding
```
apply_dynamicThreasholding(pipe)
image=pipe("some prompt",mimic_scale=20,guidance_scale=5).images[0]
```
Default values:

mimic_scale=7.0

threshold_percentile=1.00

mimic_mode='Constant' #all possible values are: "Linear Down","Half Cosine Down","Cosine Down","Linear Up","Half Cosine Up","Cosine Up","Power Up","Power Down","Linear Repeating","Cosine Repeating","Sawtooth"

mimic_scale_min=0.0

cfg_mode='Constant' #all possible values are same as mimic_mode

cfg_scale_min=0.0

sched_val=4.0

experiment_mode=0 #can also be 1,2 or 3, nothing else

separate_feature_channels=True

scaling_startpoint='MEAN' #can also be 'ZERO'

variability_measure='AD' #can also be 'STD'

interpolate_phi=1.0

# apply_fabric
It applies this: https://github.com/sd-fabric/fabric

```
buffer=open('img111.png', 'rb')
buffer.seek(0)
image_bytes = buffer.read()
dimage = Image.open(BytesIO(image_bytes))
buffer=open('img11.png', 'rb')
buffer.seek(0)
image_bytes = buffer.read()
limage = Image.open(BytesIO(image_bytes))
buffer=open('img8.png', 'rb')
buffer.seek(0)
image_bytes = buffer.read()
limage2 = Image.open(BytesIO(image_bytes))
apply_fabric(pipe)
image=pipe("some prompt",liked_images=[limage,limage2],disliked_images=[dimage]).images[0]
```
Default values:

liked_images=[]

disliked_images=[]

feedback_start_ratio=0.33

feedback_end_ratio=0.66

min_weight=0.1

max_weight=1.0

neg_scale=0.5

pos_bottleneck_scale=1.0

neg_bottleneck_scale=1.0

# apply_img2img
Assuming you apply nothing else, it will work exactly like in diffusers. In order to use img2img, you need to do the following:
```
apply_img2img(your_pipe)
#load relevant image
buffer=open('mypic.png', 'rb') #this does not exist and is purely an example
buffer.seek(0)
image_bytes = buffer.read()
image = Image.open(BytesIO(image_bytes)) #can be an array of images too. it will create many images as a result
image=your_pipe("a handsome alien",image=image).images[0]
```
Default values:

strength=0.75

skip_noise=False #whether to skip the added noise from the strength procedure. Useful to simulate an efficient hires fix implementation

Requirements:

image= an image or list of images


# apply_inpainting
Assuming you apply nothing else, it will work exactly like in diffusers. In order to use inpainting, you need to do the following:
```
apply_inpainting(your_pipe)
#load relevant image
buffer=open('dogonbench.png', 'rb') #this does not exist and is purely an example
buffer.seek(0)
image_bytes = buffer.read()
image = Image.open(BytesIO(image_bytes)) 
buffer=open('dogmask.png', 'rb') #this does not exist and is purely an example
buffer.seek(0)
image_bytes = buffer.read()
mask_image = Image.open(BytesIO(image_bytes)) #can be an array of images too. it will create many images as a result
image=your_pipe("a handsome alien",image=image,mask_image=mask_image).images[0]
```
Default values:

strength=0.75

Requirements:

image= an image or list of images

mask_image= an image or list of images

# apply_ipAdapter
This will enable ip adapter for use. However, for full revert, you'll need to save the unet before applying because the diffusers implementation changes the unet too.
Usage:
```
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin",cache_dir="model_cache")
apply_ipAdapter(pipe)
prompt="a dog or something"
img=Image.open('your_image.png').convert('RGB')
image=pipe(prompt,num_inference_steps=20,ip_adapter_image=img).images[0]


#if you want faceID instead
#load ip adapter
pipe.load_ip_adapter("ip-adapter-faceid-plusv2_sd15.bin",subfolder=None, weight_name="ip-adapter-faceid-plusv2_sd15.bin",image_encoder_folder=None)
scale = 1
pipe.set_ip_adapter_scale(scale)
#get the faceids
ref_images_embeds = []
ip_adapter_images = []
app = FaceAnalysis(name="buffalo_l",root='insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
faces = app.get(image)
ip_adapter_images.append(face_align.norm_crop(image, landmark=faces[0].kps, image_size=224))
image = torch.from_numpy(faces[0].normed_embedding)
ref_images_embeds.append(image.unsqueeze(0))
ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)
neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)
id_embeds = torch.cat([neg_ref_images_embeds, ref_images_embeds]).to(dtype=torch.float16, device="cuda")

clip_embeds = pipe.prepare_ip_adapter_image_embeds( [ip_adapter_images], None, torch.device("cuda"), len(kwargs['generator']), True)[0]
pipe.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_embeds.to(dtype=torch.float16)
pipe.unet.encoder_hid_proj.image_projection_layers[0].shortcut = True # True if Plus v2

image=pipe(prompt,num_inference_steps=20,ip_adapter_image_embeds=[id_embeds]).images[0]

```
Default values:

same as the regular txt2img pipeline from diffusers

Requirements:

use load_ip_adapter first and use ip_adapter_image

# apply_multiDiffusion
This will let you apply prompts regionally simultaneously. To use with promptFusion, you must apply promptFusion after it (Example: [[["a beautiful night in the park, high quality","mona lisa, high quality","moon"],4],[["a beautiful day in the park, high quality","mona lisa, high quality","sun"],28]]).
Usage:
```
apply_multiDiffusion(pipe)
prompt=["a beautiful night in the park, high quality","mona lisa, high quality","moon"] # 3 prompts whose order matter
mask_image=Image.open('mask_image.png').convert('RGB')
pos=["0:0-512:512",mask_image,"8:8-128:128"] # those are the areas the prompts will be applied
mask_z_index=[1,12,2] # those are the strengths of each prompt 12 is the highest in strength in this case (no limits)

image=pipe(prompt,width=512,height=512,num_inference_steps=20,ip_adapter_image=img).images[0]
```
Default values:

same as the regular txt2img pipeline from diffusers

Requirements:

prompt/prompt_embeds = a list of strings/prompt_embeds

pos= a list of strings in the format (x1:y1-x2:y2 those values are in pixels and must be within the dimensions of the image, this makes a square) or a mask from an image. Also, the list must be the same length as prompt/prompt_embeds

mask_z_index= a list of ints that must be the same length as prompt/prompt_embeds

# apply_promptFusion
This will give you the ability to change prompt mid generation. The result is something like in here: https://github.com/ljleb/prompt-fusion-extension. My syntax is different though.
Syntax:
   list of prompt instructions.
   a prompt instruction is a list with the prompt or prompt_embedding as the first element and the second element being its last step.
   example:
      [["a beautiful park, 4k",5],["volcano, cartoon",20]]
      this will denoise with "a beautiful park, 4k" for the first 5 steps, then "volcano, cartoon" until the 20th step.
      the list can be as long as you want as long as there are enough steps.
Usage:
```
apply_promptFusion(pipe)
prompt=[["a beautiful park, 4k",5],["volcano, cartoon",20]]
image=pipe(prompt,num_inference_steps=20).images[0]
```
Default values:

same as the regular txt2img pipeline from diffusers

Requirements:

prompt= a list in a specific format

# apply_SAG
Assuming you apply nothing else, it will work exactly like in diffusers. In order to use inpainting, you need to do the following:
```
apply_SAG(pipe)
image=pipe("some prompt").images[0]
```
Default values:

sag_scale=0.75

Requirements:

prompt= a string or embedding

# apply_t2iAdapter

Usage:
```
from diffusers import StableDiffusionAdapterPipeline, MultiAdapter, T2IAdapter

adapters = MultiAdapter(
    [
        T2IAdapter.from_pretrained("TencentARC/t2iadapter_keypose_sd14v1"),
        T2IAdapter.from_pretrained("TencentARC/t2iadapter_depth_sd14v1"),
    ]
)
adapters = adapters.to(torch.float16)
t2i_image=[image1,image2]
image=pipe("some prompt",adapter=adapters,adapter_conditioning_scale=[1,1],t2i_image=t2i_image).images[0]
```
Default values:
adapter_conditioning_scale=[1.0]

Requirements:
adapters= the T2i adapter you want
t2i_image= the image or images to use

# undo appliers
Assuming you'd like to use the same pipeline with different functionalities, you can do something like this:
```
apply_SAG(pipe)
apply_promptFusion(pipe)
prompt=[["a beautiful park, 4k",5],["volcano, cartoon",20]]
image=pipe(prompt,num_inference_steps=20).images[0]
pipe.revert() #this will undo all changes made to your pipeline as if nothing happened to it
apply_img2img(your_pipe)
#import controlnets
openpose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32,local_files_only=True).to('cuda')
#apply controlnet
apply_controlnet(your_pipe)
#add controlnet on your pipe
your_pipe.add_controlnet(openpose)
#load relevant images
buffer=open('openpose.png', 'rb') #this does not exist and is purely an example
buffer.seek(0)
image_bytes = buffer.read()
openpose_image = Image.open(BytesIO(image_bytes))
image=your_pipe("a handsome alien",image=image,controlnet_image=openpose_image,controlnet_conditioning_scale=0.5).images[0]
```
in the code above, we generated an image with SAG and promptfusion, then used controlnet and img2img on it to create another.
to reset the pipe just do this:
```
pipe.revert()
```
