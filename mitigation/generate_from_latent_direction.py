"""
This code generates images from a learned latent direction.

Requirements;
- latent_direction.npy: The learned latent direction.
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
    """
    Applies a learned latent direction to the given Gaussian latents.

    Args:
        latent_weight (float): The weight to apply to the latent direction.
        latents (torch.Tensor): The input latents.
        latent_direction_file (str): The file path to the learned latent direction.

    Returns:
        torch.Tensor: The modified latents after applying the learned latent direction.
    """
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
    # TODO: Modify the vars below
    latent_direction =  "latent_direction.npy" # path to the learned latent direction
    w = 10 # weight to apply to the latent direction
    n = 100 # number of images to generate for each weight
    type_prompt = "your_prompt_theme" # class for label in dictionary and name that will be used to identify the prompt when saving the images
    type_direction = "your_direction_type" # added to the filename to understand what latent direction has been applied
    weight_latent = w
    l = 0 # the latent step chosen for the latent direction training
    
    # generate images
    for i in tqdm(range(n)):            
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        generator = torch.manual_seed(i+1)

        latents = torch.randn(
            (1, 4, 128, 128),
            dtype=torch.float16,
            generator=generator
        )
        latents = randn_tensor((1, 4, 128, 128), dtype=torch.float16, generator=generator) # initial Gaussian noise latents 
        latents = from_latent_direction_learned(weight_latent, latents, latent_direction)
        
        prompt = "your_prompt_here" # TODO: Add prompt here
        out = pipe(prompt, latents=latents)

        time_stamp = datetime.datetime.now().strftime("%Y%m%d")
        os.makedirs(f"stableDiffusion_XL/{time_stamp}/{type_prompt}/l{l}w{weight_latent}", exist_ok=True)
        path_image = f"stableDiffusion_XL/{time_stamp}/{type_prompt}/l{l}w{weight_latent}/{type_direction}_w{weight_latent}_{i}.png"

        out.images[0].save(path_image)
        