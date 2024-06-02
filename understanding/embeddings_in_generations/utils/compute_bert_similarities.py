from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BERTSimilarity:
    def __init__(self, sentences_list):
        self.model = SentenceTransformer("bert-base-uncased")
        self.sentences_list = sentences_list

    def get_similarity(self):
        embeddings = self.model.encode(self.sentences_list)
        similarity_man_prompt = cosine_similarity(embeddings[0:1], embeddings[2:3])
        similarity_woman_prompt = cosine_similarity(embeddings[1:2], embeddings[2:3])
        return similarity_man_prompt, similarity_woman_prompt

if __name__ == "__main__":
    # creating list of sentences
    sentences_list = ["An image of a man",
    "An image of a woman",
    "An image of a doctor"]
    
    bert_similarity = BERTSimilarity(sentences_list)
    sim_md, sim_wd = bert_similarity.get_similarity()
    print("Man-doctor prompt similarity", sim_md)
    print("Woman-doctor prompt similarity", sim_wd)