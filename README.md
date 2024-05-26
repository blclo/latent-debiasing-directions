# Debiasing text-to-image models through Latent Directions


# Code
You can find two main folders: `understanding` and `mitigation`.
- Under the `understanding` directory are all the scripts related to the understanding pipeline. 
- The `mitigation` directory contains all the scripts that can help you find the latent directions and apply them after their generation.

# Understanding biases
We provide a tool for developers for bias understanding which targets two key points: 
1. We aim to help comprehending the connections between embeddings and generations, analysing the embedding relationship between attributes and concepts in text and vision encoders. This can reveal innate biases and make us conscious of the existing problems.
2. The tool detects the social characteristics and objects presented in the image. This helps us understanding the impact of the biases, looking at the impacted generations and the statistics.

These two points of reference can help us verify the theory: the higher the cosine similarity between concept and attribute, the more likely is to see these attributes present in the generated images of the concept. If we find a high cosine similarity between a specific concept and attribute, but this is not so clear when looking at the statistics of their detection in the generated images, then we would have a misalignment and perhaps something to investigate!

In our paper we can see an example of how despite using the prompt ”A **wealthy** African man and his house”, the highest embedding similarities belong to attributes such as poverty-stricken or underprivileged.

# Mitigating biases through latent directions
The mitigation strategy consists of two main parts. First, we have to obtain the latent direction. Secondly, we need to apply it!

1. Finding the latent direction 🕵️‍♀
    - We need to generate two sample datasets and save their corresponding latents. The code found in `generate_dataset_save_latents.py`, helps you build this dataset and obtain a json dictionary with the latents files and class label. You need to run this script twice, once per each class/dataset.

        - What sample datasets should I choose? The datasets should represent the transition you aim to achieve through your latent direction. For instance, if you aim to debias light-skin color images, to generate more diversity with dark-skin color, you should choose to generate a dataset containing light-skin individuals and another one containing dark-skin ones. These way we can ensure the latents belong to the two different groups and we can train the latent direction to differentiate between them :)

    - Obtain the latent direction. The code found in `obtain_latent_direction.py` iterates through the merged* json dictionary, separating the data from the two classes. A SVM classifier separates linearly between the classes and returns 10 diffent latent directions per class. Each of those latent directions, differ in the latent step used for their training, from step 0 (Gaussian Noise) to step 45 (almost the complete denoised image).

*merged: the two dictionaries generates by `generate_dataset_save_latents.py` should be merged manually before using it in `obtain_latent_direction.py`.

2. Apply the latent direction with a chosen weight 🚀
   
    The paper shows how the impact of the weight selection is smaller than the impact of the latent step used in the training. As a result, we recommend choosing the latent direction obtained through step 10, which corresponds to `idx_latent` of 2. Once we have its file path, we can use the code found in `generate_from_latent_direction.py`, modify the weight and the number of images to generate and obtain debiased results!

   
# Citation
If this work is insightful and relevant for your research, we would highly appreciate if you could cite it:

For any questions, please do not hesitate to reach out to Carolina Lopez at clopezolmos@microsoft.com

