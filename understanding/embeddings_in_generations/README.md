# TERU & VERU pipelines

TERU stands for Textual Embedding Relationship Understanding and VERU for Visual Embedding Relationship Understanding!
One focuses on the text embeddings while the other one on the embeddings from the vision encoder.

The code found here allows you to obtain a graph with the similarities between attributes and a specific concept.
The methodology of the pipelines can be seen in the following Figure:

![teru_pipeline](https://github.com/blclo/GlobalDataReps-RAI/blob/main/understanding/embeddings/media/TERU.png)

![veru_pipeline](https://github.com/blclo/GlobalDataReps-RAI/blob/main/understanding/embeddings/media/VERU_diagram_bigger_looking.png)

The following graph shows an example of the output for the `firefighter` concept. The cosine similarity between the concept and each of the provided attributes is presented on it. The y axis present the text embeddings similarities and the x axis the vision embeddings.

![firefighter_point_chart](https://github.com/blclo/GlobalDataReps-RAI/blob/main/understanding/embeddings/media/firefighterAttributesRelationshipPlot.png)

Some more possible charts have been explored, for other possible representations. These are the radar chart, the stacked bar chart and the single bars chart.

![radar_chart](https://github.com/blclo/GlobalDataReps-RAI/blob/main/understanding/embeddings/media/radar_chart_test.jpg)
![stacked_chart](https://github.com/blclo/GlobalDataReps-RAI/blob/main/understanding/embeddings/media/stacked_bar_test.svg)
![single_bars](https://github.com/blclo/GlobalDataReps-RAI/blob/main/understanding/embeddings/media/single_bar_test.svg)


## Running the automatic pipeline 
![single_bars_cleaner](https://github.com/blclo/GlobalDataReps-RAI/blob/main/understanding/embeddings/media/text_single_bars_cleaner_attributes_relationship_plot.png)

### Script usage
- `--attributes`: path to the text file containing the attributes you want to analyse
- `--concept`: concept to analyse
- `--text`: flag to enable only the textual pipeline

This pipeline allows you obtain the relationship between the text embeddings of the main [concept] to each of its related [attributes]:

`python teru_veru_pipeline.py --attributes "path_to_text_file" --concept "concept" --text`

will result in: "Bar chart stored at: path_to_image.png"

To include the vision encoder, remove the `--text` tag.

# In a nutshell
The command bellow will iterate through the attributes in the text file and will generate 20 images per attribute, comparing the visual embeddings of the generations to the ones of the concept. This is also done with the textual embeddings but without the need of generating an image. 

The more number of images to generate, the more realistic the results will be since the embeddings will be averaged across all samples.
`python teru_veru_pipeline.py "attributes.txt" "firefighter" 20`

### Extra scripts
Inside the `utils` directory you can also find some scripts used in through the project's development that can be used individually.
They are convenient for exploration:

- `compute_bert_similarities.py`: to compute similarities for BERT's text encoder across woman to profession (in this case doctor) and man to profession. 
- `compute_clip_similarities.py`: to compute similarities for CLIP's vision encoder across two images when providing their paths. 
