import torch
import clip
from PIL import Image
import torch.nn as nn
import sys
from pathlib import Path
from transformers import CLIPTextModel, CLIPTokenizer

class CLIPTextSimilarity(nn.Module):
    def __init__(self, sentences_list):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.sentences_list = sentences_list

    def get_similarity(self):
        man_prompt = self.sentences_list[0]
        woman_prompt = self.sentences_list[1]
        profession_prompt = self.sentences_list[2]

        text_man_prompt = clip.tokenize(man_prompt).to("cuda")
        text_woman_prompt = clip.tokenize(woman_prompt).to("cuda")
        text_profession_prompt = clip.tokenize(profession_prompt).to("cuda")

        with torch.no_grad():
            text_man_features = self.model.encode_text(text_man_prompt)
            text_woman_features = self.model.encode_text(text_woman_prompt)
            text_profession_features = self.model.encode_text(text_profession_prompt)

        man_profession_similarity = self.cos(text_man_features[0], text_profession_features[0])
        woman_profession_similarity = self.cos(text_woman_features[0], text_profession_features[0])

        return man_profession_similarity, woman_profession_similarity

class ConceptAttributeCLIPTextSimilarity(nn.Module):
    def __init__(self, sentences_list):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.cos = torch.nn.CosineSimilarity(dim=0)
        self.sentences_list = sentences_list

         # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="tokenizer",
            revision=None,
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            subfolder="text_encoder",
            revision=None,
        )

        self.text_encoder.to("cuda")

    def get_similarity(self):
        main_concept = self.sentences_list[0]
        attribute = self.sentences_list[1]

        concept_input = self.tokenizer(main_concept, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        attribute_input = self.tokenizer(attribute, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
            main_concept_embeddings = self.text_encoder(concept_input.input_ids.to("cuda"))[0]
            attribute_embeddings = self.text_encoder(attribute_input.input_ids.to("cuda"))[0]

        similarity_mit_paper = self.cos(main_concept_embeddings[0], attribute_embeddings[0])

        main_concept_token = clip.tokenize(main_concept).to("cuda")
        attribute_token = clip.tokenize(attribute).to("cuda")

        with torch.no_grad():
            main_concept_embeddings = self.model.encode_text(main_concept_token)
            attribute_embeddings = self.model.encode_text(attribute_token)

        similarity_concept_attribute = self.cos(main_concept_embeddings[0], attribute_embeddings[0])

        print("Measuring similarity in the same way as MIT paper")
        return similarity_mit_paper
    
if __name__ == "__main__":
    sentence = f"An image of a doctor"
    sentences_list = ["A picture of a man",
                        "A picture of a woman",
                        f"{sentence}"]
    text_encoder = CLIPTextSimilarity(sentences_list)
    man_sim, woman_sim = text_encoder.get_similarity()
    print(f"The text encoder similarity is {man_sim}, {woman_sim}")