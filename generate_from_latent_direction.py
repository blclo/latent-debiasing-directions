"""
In a nutshell, this code:
- Does not save the latents
- Generates from a pretrained latent direction
"""

import torch
from diffusers import StableDiffusionXLPipeline
import pickle
import os
import datetime
from PIL import Image   
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import json


def from_latent_direction_learned(latent_weight, latents, latent_direction_file):
    numpy_file_path = latent_direction_file
    numpy_array = np.load(numpy_file_path)

    # Convert the NumPy array to a PyTorch tensor
    latent_direction = torch.from_numpy(numpy_array)
    print("Shape of latent direction:", latent_direction.size())
    print("Shape of latents", latents.size())
    if latents.dtype != latent_direction.dtype:
        latent_direction = latent_direction.to(latents.dtype)

    latent_direction = latent_direction.to(latents.device)
    return latents + (latent_weight)*latent_direction

if __name__ == "__main__":
    latent_direction =  "path_to_latent_direction.npy"
    weights = [0, 14] # TODO: Define the array of weights to try out

    for w in weights:
        # TODO: Modify the vars below
        type_prompt = "your_prompt_theme" # just used for path saving purposes
        type_direction = "your_direction_type" # just used for path saving purposes
        weight_latent = w
        l = 0 # the latent step chosen for the latent direction training
        
        # declare initial noise
        for i in tqdm(range(100)):            
            pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            generator = torch.manual_seed(i+1)

            latents = torch.randn(
                (1, 4, 128, 128),
                dtype=torch.float16,
                generator=generator
            )
            latents = randn_tensor((1, 4, 128, 128), dtype=torch.float16, generator=generator)
            latents = from_latent_direction_learned(weight_latent, latents, latent_direction)
            
            prompt = "your_prompt_here" # TODO: Add prompt here
            out = pipe(prompt, latents=latents)

            time_stamp = datetime.datetime.now().strftime("%Y%m%d")
            os.makedirs(f"stableDiffusion_XL/{time_stamp}/{type_prompt}/l{l}w{weight_latent}", exist_ok=True)
            path_image = f"stableDiffusion_XL/{time_stamp}/{type_prompt}/l{l}w{weight_latent}/{type_direction}_w{weight_latent}_{i}.png"

            out.images[0].save(path_image)
            