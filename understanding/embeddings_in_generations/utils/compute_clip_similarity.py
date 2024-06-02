'''
* NOTE: Extra script *
Script to compute similarities for CLIP's vision encoder across two images when providing their paths. 
'''
import torch
import clip
from PIL import Image
import torch.nn as nn
import sys
from pathlib import Path
import datetime
import os

class CLIPSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.cos = torch.nn.CosineSimilarity(dim=0)

    def get_similarity(self, image1_path, image2_path):
        image1_preprocess = self.preprocess(Image.open(image1_path)).unsqueeze(0).to("cuda")
        image1_features = self.model.encode_image(image1_preprocess)

        image2_preprocess = self.preprocess(Image.open(image2_path)).unsqueeze(0).to("cuda")
        image2_features = self.model.encode_image(image2_preprocess)

        similarity = self.cos(image1_features[0],image2_features[0]).item()
        return similarity
    
def get_paths():
    if (args_count := len(sys.argv)) > 3:
        print(f"Two argument expected, got {args_count - 1}")
        raise SystemExit(2)
    elif args_count < 2:
        print("You must specify the directory of both images")
        raise SystemExit(2)

    image1 = Path(sys.argv[1])
    image2 = Path(sys.argv[2])
    return image1, image2

if __name__ == "__main__":
    image1_path, image2_path = get_paths()
    clip_similarity = CLIPSimilarity()
    similarity = clip_similarity.get_similarity(image1_path, image2_path)
    print("Image similarity", similarity)