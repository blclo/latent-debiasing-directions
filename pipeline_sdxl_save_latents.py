"""
Script to save latents from the generated images.
It doesnt use a latent direction to generate, just the Gaussian noise.

In a nutshell, this code is to generate the original images and save their latents for the latter training.
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


type_prompt = "your_identifier" # only the name of the directory that will be used to identify the prompt when saving the images
type_direction = "direction_type" # added to the filename to understand if a direction has been applied
weight_latent = "Nil" # added to the filename to understand if a weight to a direction has been applied
path_latents = []
counter = 0

def save_latents(latents, n, path_latents, counter):
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
        global path_latents
        global counter
        # adjust the batch_size of prompt_embeds according to guidance_scale
        
        if step_index == 0:
            path_latents = []
        
        print(callback_kwargs["latents"].dtype)
        latents = callback_kwargs["latents"]
        # saving every 5 latents
        if step_index%5 == 0:
            path_latents = save_latents(latents, step_index, path_latents, counter)

        #plot_image_from_noise(latents, 0, step_index)
        print("[info] Latents saved.")

        print(callback_kwargs["latents"].shape)
        # torch.Size([1, 4, 128, 128])
        return callback_kwargs


if __name__ == "__main__":
    list = []
    print("Startind code, will save latents every 5 steps.")
    
    # generate images
    for i in tqdm(range(100)):
        counter = i
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(i+1)
        prompt = "your prompt" # add your prompt here
        out= pipe(prompt, generator=generator, callback_on_step_end = callback_dynamic_cfg, callback_on_step_end_tensor_inputs=['latents'], added_cond_kwargs = {})

        time_stamp = datetime.datetime.now().strftime("%Y%m%d")
        os.makedirs(f"stableDiffusion_XL/{time_stamp}/{type_prompt}", exist_ok=True)
        path_image = f"stableDiffusion_XL/{time_stamp}/{type_prompt}/{type_direction}_w{weight_latent}_{i}.png"

        out.images[0].save(path_image)
        list.append({
            "path": f"{path_image}",
            "latents": path_latents,
            "class":"define_for_your_prompt"
        })
        if i%25==0:
            # save dictionary every 100 images
            os.makedirs("saved_dictionary", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
            path_dict = f"saved_dictionary/{type_prompt}_{type_direction}_w{weight_latent}_my_dict_{timestamp}.json"

    # save dictionary with pairs of paths, latens and direction types.
    os.makedirs("saved_dictionary", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    path_dict = f"saved_dictionary/{type_prompt}_{type_direction}_w{weight_latent}_my_dict_{timestamp}.json"

    # Save the dictionary using json
    with open(path_dict, 'w') as file:
        json.dump(list, file, indent=2)

    print(f'Dictionary saved to {path_dict}')