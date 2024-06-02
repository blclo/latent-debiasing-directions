"""
This script creates a bar plot from pickle file 
and produces the images with bboxes overlay.

Requirements:
1. Provide path_to_graph for saving the resulting figure.
2. Path to pickle file where the dictionary was stored 
"""

import pickle
from PIL import Image, ImageDraw, ImageFont
import os
import torch
import torchvision.transforms as T
import cv2
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# methods to plot the bboxes
def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def draw_entity_boxes_on_image(image, entities, show=False, save_path=None):
    """_summary_
    Args:
        image (_type_): image or image path
        collect_entity_location (_type_): _description_
    """
    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)[:, :, [2, 1, 0]]
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    if len(entities) == 0:
        return image

    new_image = image.copy()
    previous_bboxes = []
    # size of text
    text_size = 0.4
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 3
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 3

    for entity_name, (start, end), bboxes in entities:
        for (x1_norm, y1_norm, x2_norm, y2_norm) in bboxes:
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
            # draw bbox
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

            x1 = orig_x1 - l_o
            y1 = orig_y1 - l_o

            if y1 < text_height + text_offset_original + 2 * text_spaces:
                y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                x1 = orig_x1 + r_o

            # add text background
            (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
            text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

            for prev_bbox in previous_bboxes:
                while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
                    text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                    text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                    y1 += (text_height + text_offset_original + 2 * text_spaces)

                    if text_bg_y2 >= image_h:
                        text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                        text_bg_y2 = image_h
                        y1 = image_h
                        break

            alpha = 0.5
            for i in range(text_bg_y1, text_bg_y2):
                for j in range(text_bg_x1, text_bg_x2):
                    if i < image_h and j < image_w:
                        if j < text_bg_x1 + 1.35 * c_width:
                            # original color
                            bg_color = color
                        else:
                            # white
                            bg_color = [255, 255, 255]
                        new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)

            cv2.putText(
                new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
            )
            # previous_locations.append((x1, y1))
            previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

    pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
    if save_path:
        pil_image.save(save_path)
    if show:
        pil_image.show()

    return new_image

if __name__ == '__main__':
    # TODO: define
    path_to_graph = "path_to_saving_the_graph.png"
    path_to_pickle_file = "path_to_your_pickle_file.pkl"
    
    # open pickle file
    file = open(path_to_pickle_file, "rb")
    data = pickle.load(file)
    file.close()

    # describe
    for key, val in data.items():
        print (key)

    # store key data
    filenames = data["filename"]
    descriptions = data["description"]
    entities = data["entities"]

    # discover unique set of image descriptions
    set_descriptions = set()
    for des in descriptions:
        set_descriptions.add(des)

    print(set_descriptions)
    print(f"Out of a total of {len(descriptions)} described images, {len(descriptions) - len(set_descriptions)} shared a description.")

    # create a dictionary counting the number of entities found in all images
    objects = {}
    for ent in entities:
        for i in range(len(ent)):
            found_object = ent[i][0]
            if found_object not in objects:
                objects[found_object] = 1
            else:
                objects[found_object] += 1
            found = ent[i][1]
            found_bbox = ent[i][2]

    # plot and save bar plot:
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(objects.items()), columns=['Concepts found', 'Count'])
    # Create a bar plot
    ax = sns.barplot(y=df['Concepts found'], x=df['Count'], data=df, orient='h')

    # Set ticks every unit
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Adjust labels to avoid cutting
    plt.tight_layout()

    plt.savefig(path_to_graph)

    # save all images in folder with the bounding boxes detected
    for i in range(len(filenames)):
        # open image
        image = Image.open(filenames[i])
        width, height = image.size

        # Save or display the modified image
        os.makedirs("bboxes_images", exist_ok=True)

        # Draw the bounding bboxes
        draw_entity_boxes_on_image(image, entities[i], show=False, save_path=f"bboxes_images/processed_{i}.png")
    
