"""
* Specific script *
This script classifies your images across Indian, Latino, Black, Caucasian, Southeast Asian race, 
East Asian and Middle Eastern races.

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

class1_count, class2_count, class3_count, class4_count, class5_count, class6_count, class7_count = 0, 0, 0, 0, 0, 0, 0
total_images = len(files)

for filename in os.listdir(path):
    img_path = os.path.join(path, filename)
    # classify image
    image = Image.open(img_path)

    inputs = processor(text=["A person of Indian race", "A person of Latino race", "A person of Black race", "A person of Caucasian race", "A person of Southeast Asian race", "A person of East Asian race", "A person of Middle Eastern race"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    print(probs)
    probs = probs.detach().numpy()
    idx_max = np.argmax(probs)

    # fill missing field in the dictionary
    if idx_max == 0:
        class1_count += 1
    elif idx_max == 1:
        class2_count += 1
    elif idx_max == 2:
        class3_count += 1
    elif idx_max == 3:
        class4_count += 1
    elif idx_max == 4:
        class5_count += 1
    elif idx_max == 5:
        class6_count += 1
    else:
        class7_count += 1
    
# coger todos los latents 0 donde class sea mujer y hacer el avarage. Dar eso como ruido inicial para ver
percentage_class1 = (class1_count/total_images)*100
percentage_class2 = (class2_count/total_images)*100
percentage_class3 = (class3_count/total_images)*100
percentage_class4 = (class4_count/total_images)*100
percentage_class5 = (class5_count/total_images)*100
percentage_class6 = (class6_count/total_images)*100
percentage_class7 = (class7_count/total_images)*100
print(f"Out of {total_images}, {class1_count} ({percentage_class1}%) are Indian, {class2_count} ({percentage_class2}%) are Latino, {class3_count} ({percentage_class3}%) are Black, {class4_count} ({percentage_class4}%) are Caucasian, {class5_count} ({percentage_class5}%) are Southeast Asian, {class6_count} ({percentage_class6}%) East Asian, {class7_count} ({percentage_class7}%) Middle Eastern.")

