import torch
import argparse
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import datetime
import os

class generateImageProfession:
    def __init__(self, prompt = str, negative_prompt=None, base_path = None, type=str, concept=None):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.type = type
        self.concept = concept
        self.base_path = base_path

    def get_path(self):
        # define main saving dir
        folder_path = "stableDifussionGenerated/"
        # Get the current date as a string, e.g., "2023-09-28"
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        # Create a new folder with the date as its name inside the original folder
        date_folder_path = os.path.join(folder_path, current_date)

        counter = 0
        if self.type == "concept":
            self.base_path = os.path.join(date_folder_path, "concepts")
            self.base_path = os.path.join(self.base_path, f"{self.concept}")
        else:
            self.base_path = os.path.join(date_folder_path, "attributes")
            self.base_path = os.path.join(self.base_path, f"{self.type}")

        os.makedirs(self.base_path, exist_ok=True)  # exist_ok=True will prevent errors if the folder already exists
        self.first_image = False
        return self.base_path
    
    def get_image(self):
        path = self.get_path()
        self.base_path = path
        model_id = "stabilityai/stable-diffusion-2-1"

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(0)
        image = pipe(self.prompt, negative_prompt = self.negative_prompt).images[0]
        
        if self.type == "concept":
            prompt = self.concept
        else:
            prompt = self.type
        image_name = f"{prompt}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png"
        full_image_path = os.path.join(self.base_path, image_name)

        # Save the image
        image.save(full_image_path)
        return self.base_path, full_image_path

if __name__ == "__main__":
    folder_path = "stableDifussionGenerated/"
    parser = argparse.ArgumentParser(description='Generate an image using Stable Diffusion based on a given prompt.')
    parser.add_argument('prompt', type=str, help='Prompt for image generation')
    parser.add_argument('--negative', type=str, help='Negative prompt to exclude certain elements', default=None)

    args = parser.parse_args()
    imageGenerator = generateImageProfession(args.prompt, args.negative)
    for i in range(5):
        image_name = imageGenerator.get_image()
