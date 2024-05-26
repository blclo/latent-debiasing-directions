"""
Script to save latents from the generated images.
It doesnt use a latent direction to generate, just the Gaussian noise.

In a nutshell, this code is to generate the original images and save their latents for the latter training.
This script should be ran twice, once per class.

Requirements:
- Modify the vars below to ensure correct saving paths.
- Define the prompt.
"""
import torch
from diffusers import StableDiffusionXLPipeline
import pickle
import os
import datetime
from PIL import Image   
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import json

# TODO: Modify organisational vars
type_prompt = "your_identifier" # class for label in dictionaty and name that will be used to identify the prompt when saving the images
type_direction = "direction_type" # added to the filename to understand what latent direction has been applied
weight_latent = "Nil" # added to the filename to understand if a weight to a direction has been applied

path_latents = []
counter = 0


def save_latents(latents, n, path_latents, counter):
    """
    Save the given latents to a pickle file and return the updated list of file paths.

    Args:
        latents (Tensor): The latents to be saved.
        n (int): The step number.
        path_latents (list): The list of file paths to be updated.
        counter (int): The counter value.

    Returns:
        list: The updated list of file paths.
    """
    latents = 1 / 0.18215 * latents
    
    # Create the file path where you want to save the pickle file storing the latents
    time_stamp = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(f"latents/{time_stamp}", exist_ok=True)
    
    file_path = f'latents/{time_stamp}/{type_prompt}_{type_direction}_w{weight_latent}_latents_image_{counter}_step_{n}.pkl'
    path_latents.append(file_path)
        
    # Save the tensor using pickle
    with open(file_path, 'wb') as file:
        pickle.dump(latents, file)
    return path_latents
     
def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
    """
    Callback function used in the latent debiasing directions denoising process.
    
    Args:
        pipe (object): The pipe object used in the denoising process.
        step_index (int): The index of the current step in the denoising process.
        timestep (int): The current timestep in the denoising process.
        callback_kwargs (dict): Additional keyword arguments passed to the callback function.
        
    Returns:
        dict: The updated callback_kwargs dictionary.
    """
    global path_latents
    global counter
    # adjust the batch_size of prompt_embeds according to guidance_scale
    
    if step_index == 0:
        path_latents = []
    
    print(callback_kwargs["latents"].dtype)
    latents = callback_kwargs["latents"]
    # saving latents every 5 denoising steps
    if step_index % 5 == 0:
        path_latents = save_latents(latents, step_index, path_latents, counter)

    print("[info] Latents saved.")

    print(callback_kwargs["latents"].shape) # torch.Size([1, 4, 128, 128])
    return callback_kwargs


if __name__ == "__main__":
    list = []
    print("Starting code, will save latents every 5 steps.")
    
    # generate images. 50 by default. Smaller datasets haven't been tested for good results in the latent direction training. You are free to experiment tho :) A bigger dataset will yield better results.
    for i in tqdm(range(50)):
        counter = i
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(i+1)
        prompt = "your prompt" # TODO: add your prompt here
        out= pipe(prompt, generator=generator, callback_on_step_end = callback_dynamic_cfg, callback_on_step_end_tensor_inputs=['latents'], added_cond_kwargs = {})

        time_stamp = datetime.datetime.now().strftime("%Y%m%d")
        os.makedirs(f"stableDiffusion_XL/{time_stamp}/{type_prompt}", exist_ok=True)
        path_image = f"stableDiffusion_XL/{time_stamp}/{type_prompt}/{type_direction}_w{weight_latent}_{i}.png"

        out.images[0].save(path_image)
        list.append({
            "path": f"{path_image}",
            "latents": path_latents,
            "class":f"{type_prompt}", # label of the class
        })
        if i%25==0:
            os.makedirs("saved_dictionary", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
            path_dict = f"saved_dictionary/{type_prompt}_{type_direction}_w{weight_latent}_my_dict_{timestamp}.json"

    # save dictionary with pairs of paths, latens and direction types.
    os.makedirs("saved_dictionary", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    path_dict = f"saved_dictionary/{type_prompt}_{type_direction}_w{weight_latent}_my_dict_{timestamp}.json"

    # Save the JSON dictionary 
    with open(path_dict, 'w') as file:
        json.dump(list, file, indent=2)

    print(f'Dictionary saved to {path_dict}')