import openai
# from automated_extractions_of_attributes import GPT4_query
from utils.generate_images_sd import generateImageProfession
from utils.compute_clip_similarity import CLIPSimilarity
from utils.compute_clip_text_encoder import CLIPTextSimilarity, ConceptAttributeCLIPTextSimilarity
from utils.plotting_concept_w_relationships import PlotConceptAttributeRelations
import datetime
import json
import pickle
import os
import sys
import argparse

# modify the code to accept path with generated images to save time generating them

def get_argvs():
    parser = argparse.ArgumentParser(
                    prog='EmbeddingsGlobalReps',
                    description='Computes the relationship between the text and vision embeddings of concepts and attributes.',
                    epilog='Text at the bottom of help')
    
    parser.add_argument("--attributes", "-a")
    parser.add_argument("--concept", "-c")
    parser.add_argument("--number", "-n", default=1) # no images will be generated but we can't divide with 0
    parser.add_argument("--text", "-t", action='store_true')

    args = parser.parse_args()
    return  args.attributes, args.concept, args.number, args.text

'''
def get_concept_attribute_visual_similarity_from_existing_images(main_concept_image_name, att_image_name):
    clip_similarity = CLIPSimilarity()
    concept_attribute_visual_similarity = clip_similarity.get_similarity(main_concept_image_name, att_image_name)
    return concept_attribute_visual_similarity
'''

class embed():
    def __init__(self, attributes_text_file, concept, num_generated_images, text):
        # read attributes
        with open(attributes_text_file, "r") as file:
            self.attributes = file.read().splitlines()
            print("Attributes have been read from file.")

        self.concept = concept
        self.number_of_generated_images = int(num_generated_images) # number of generated images per attribute in VERU
        self.only_text = text # flag to compute only text embeddings if True
        self.path_for_results = None
        
        # list to save main data points
        self.data_points = []
        # list to collect information for dumping to file
        self.info_dump_all_atts = []

        self.base_path = None # will be used to store the generated images

    # ------------------------------------- helper methods >
    def get_concept_attribute_textual_similarity(self, att):
        sentences_list = [f"{self.concept}",
                        f"{att}"]
        clip_text_encoder = ConceptAttributeCLIPTextSimilarity(sentences_list)
        concept_attribute_text_similarity = clip_text_encoder.get_similarity()
        return concept_attribute_text_similarity

    def get_concept_attribute_visual_similarity(self, att, negative_prompt):
        self.base_path, self.main_concept_image_name = generateImageProfession(self.main_concept_prompt, negative_prompt, self.base_path, "concept", self.concept).get_image()
        self.base_path, self.att_image_name = generateImageProfession(self.attributes_prompt, negative_prompt, self.base_path, str(att)).get_image()
            
        clip_similarity = CLIPSimilarity()
        concept_attribute_visual_similarity = clip_similarity.get_similarity(self.main_concept_image_name, self.att_image_name)
        return concept_attribute_visual_similarity

    def compute_text_embeddings(self, attribute):
        conc_att_text_similarity = self.get_concept_attribute_textual_similarity(attribute)
        return conc_att_text_similarity        

    # Methods to append data
    def append_generation(self, att, conc_att_visual_similarity, text_similarity_for_att):
        # Append information to the info_dump list
        entry = {
            "Profession": self.concept,
            "Attribute": att,
            **({"Image prompts paths": {
                "Main concept prompt": self.main_concept_prompt,
                "Attribute prompt": self.attributes_prompt
            }} if not self.only_text else {}),
            "CLIP text": {
                f"Similarity {self.concept}/{att}": conc_att_visual_similarity
            },
            **({"CLIP vision": {
                "Concept file": self.main_concept_image_name,
                "Attribute file": self.att_image_name,
                f"Similarity {concept}/{att}": conc_att_visual_similarity
            }} if not self.only_text else {}),
            "Appended data point": [conc_att_visual_similarity, text_similarity_for_att.tolist(), f"{att}"]
        }
        self.info_dump_all_atts.append(entry)    

    def append_att_across_generations(self, vision_global, text_global, att):
        vision_global /= self.number_of_generated_images
        data_across_all_generations = {
            "Profession": self.concept,
            "Data points": [(vision_global, text_global.tolist(), f"{att}")]
        }
        # add to self.info_dump_all_atts to later append to json dictionary
        self.info_dump_all_atts.append(data_across_all_generations)

        # Add data to the data_points list for plotting later
        self.data_points.append((vision_global, text_global.tolist(), f"{att}"))

    # ------------------------------------- main methods >
    # Main method to compute the embeddings
    def compute_embeddings(self):
        # do for each attribute
        for att in self.attributes:
            print(f"Processing attribute: {att}.")
            vision_global = 0
            # compute textual similarity
            text_similarity_for_att = self.compute_text_embeddings(att)
            print(f"Text similarity between {self.concept}/{att} is {text_similarity_for_att}")

            if not self.only_text:
                negative_prompt = ""
                self.main_concept_prompt = f"photo of a {self.concept}"
                self.attributes_prompt = f"photo of {att}"
                for i in range(0, self.number_of_generated_images):
                    # append data for each generation
                    conc_att_visual_similarity = self.get_concept_attribute_visual_similarity(att, negative_prompt)
                    vision_global += conc_att_visual_similarity
                    self.append_generation(att, conc_att_visual_similarity, text_similarity_for_att)
            else:
                vision_global = 0

            self.append_att_across_generations(vision_global, text_similarity_for_att, att)

    def save(self):
        print("----------- Saving results -----------")
        # Save the information to a json file
        filename = f"info_dump_attributes_of_{self.concept}.json"
        base_path = f"/work3/s212725/hfstabledifussion/stableDifussionGenerated/{datetime.datetime.now().strftime('%Y-%m-%d')}"
        self.path_for_results = os.path.join(base_path, "results")
        if not os.path.exists(self.path_for_results):
            os.makedirs(self.path_for_results)
        json_path = os.path.join(self.path_for_results, filename)
        with open(json_path, "w") as file:
            json.dump(self.info_dump_all_atts, file, indent=4)
            print("JSON Dump can be found in " + json_path)

        # Save list of tuples into pickle file 
        pickle_filename = f"data_points_of_{self.concept}.pkl"
        path_for_pickle_results =  os.path.join(base_path, "results")
        pickle_path = os.path.join(path_for_pickle_results, pickle_filename)
        with open(pickle_path, 'wb') as file:
            pickle.dump(self.data_points, file)
        print("Pickle dump can be found in " + pickle_path)
        
    def plot(self):
        plot = PlotConceptAttributeRelations(self.path_for_results, self.data_points, self.concept)
        if not self.only_text:
            plot.point_chart()
        else:
            #plot.radar_chart()
            plot.single_bars()

if __name__ == "__main__":
    # Get args
    if len(sys.argv) == 1:
        print("No arguments provided. Use --help for usage information.")
        sys.exit(1)

    attributes_text_file, concept, num_generated_images, text = get_argvs()
    
    if text:
        print("Computing only text embeddings for attributes in concept.")

    embeddings = embed(attributes_text_file, concept, num_generated_images, text)
    embeddings.compute_embeddings() # save into self.info_dump_all_att for json and self.data_points for pkl file
    embeddings.save() # save the both arrays above into json and pkl file
    embeddings.plot() # plot using self.data_points in radar_chart() | single_bars()
    