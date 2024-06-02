"""
This file runs images through the kosmos 2 model, obtains its descriptions, bboxes and entities and later stores the dictionary
results in a pickle file

Requirements:
- Define the path to the folder containing the images.
- Define the file_path used for saving the pickle dictionary
"""
 
import requests
import os 
import pickle
import datetime
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

prompt = "<grounding>An image of"

# TODO: Get the list of filenames in the folder
folder_path = "your_folder_here"
filenames = os.listdir(f"{folder_path}")

# TODO: Specify the file path
timestamp = datetime.datetime.now().strftime('%Y-%m-%d')
file_path = f'name_of_pickle_file_here_{timestamp}.pkl'

image_dict = {
    "filename":[],
    "description":[],
    "entities": []
}

# Print the filenames
for filename in filenames:
    print(filename)
    filename_path = f"{folder_path}/{filename}"
    image_dict["filename"].append(filename_path)
    image = Image.open(filename_path)
    
    # The original Kosmos-2 demo saves the image first then reload it. For some images, this will give slightly different image input and change the generation outputs.
    image.save(f"processed_{filename}")
    image = Image.open(f"processed_{filename}")

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Specify `cleanup_and_extract=False` in order to see the raw model generation.
    processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

    # By default, the generated  text is cleanup and the entities are extracted.
    processed_text, entities = processor.post_process_generation(generated_text)

    image_dict["description"].append(processed_text)
    print(processed_text)
    image_dict["entities"].append(entities)
    print(entities)

# Open the file in binary write mode ('wb')
with open(file_path, 'wb') as file:
    # Use pickle.dump() to save the dictionary to the file
    pickle.dump(image_dict, file)