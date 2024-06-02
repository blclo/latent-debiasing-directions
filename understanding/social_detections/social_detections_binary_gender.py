"""
* Specific script *
This script classifies your images across the binary genders: woman or male.

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
    # classify image
    image = Image.open(img_path)

    inputs = processor(text=["A picture of a woman", "A picture of a man"], images=image, return_tensors="pt", padding=True)

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
print(f"Out of {total_images}, {class1_count} ({percentage_class1}%) are woman and {class2_count} ({percentage_class2}%) are man.")


stats = {
    "percentage_man": percentage_class2,
    "percentage_woman": percentage_class1,
    "total_man": class2_count,
    "total_woman": class1_count
}


# Specify the file path
json_file_path = f'{path}/binary_gender_predictions.json'

# Writing to the JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(stats, json_file, indent=4)

print(f'Dictionary saved to {path}')