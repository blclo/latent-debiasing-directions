"""
This script allows you to define two different classes to understand the classification of your images
across them. While it is set up for two classes, it can be modified to account for more.

NOTE: We use CLIP for classification. 

- Requirements:
1. Provide the path for your image folder
2. Define the classes' prompts
"""

from PIL import Image
import requests
import numpy as np
import json
import os

from transformers import CLIPProcessor, CLIPModel

# load folder to get images from it
path = "path_to_your_images_folder"
prompt_class1 = "define your prompt" # e.g. "A picture of a woman"
prompt_class2 = "define your prompt" # e.g. "A picture of a man"

files = os.listdir(path)

# class classification
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print(f"Found a total of {len(files)} images in the folder.")

class1_count = 0
class2_count = 0
total_images = len(files)

for filename in os.listdir(path):
    img_path = os.path.join(path, filename)
    image = Image.open(img_path)

    inputs = processor(text=[f"{prompt_class1}", f"{prompt_class2}"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    print(probs)
    probs = probs.detach().numpy()
    is_class2 = np.argmax(probs)

    # fill missing field in the dictionary
    if is_class2:
        class2_count += 1
    else:
        class1_count += 1
    
# coger todos los latents 0 donde class sea mujer y hacer el avarage. Dar eso como ruido inicial para ver
percentage_class1 = (class1_count/total_images)*100
percentage_class2 = (class2_count/total_images)*100
print(f"Out of {total_images}, {class1_count} ({percentage_class1}%) are from class 1 and {class2_count} ({percentage_class2}%) are from class 2.")

stats = {
    "percentage_class1": percentage_class1,
    "percentage_class2": percentage_class2,
    "total_class1_count": class1_count,
    "total_class2_count": class2_count
}

# Specify the file path
json_file_path = f'{path}/social_predictions.json'

# Writing to the JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(stats, json_file, indent=4)

print(f'Dictionary saved to {path}')