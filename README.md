# RubberDiffusers
This project aims to solve the rigidity problem that diffusers has. Instead of creating a pipeline for each variation and combination, you can just implement it for RubberDiffusers and the user will pick the variations he wants to enable or not. This is based on the base txt2img pipeline of diffusers.

# How to use
1. install diffusers:
   pip install git+https://github.com/huggingface/diffusers
2. run examples.py
3. choose whatever appliers you want, but warning, some appliers should be applied later if you're stacking them like promptFusion. Also, if you use inpainting, you can't use img2img
   
