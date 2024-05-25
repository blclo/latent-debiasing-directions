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

# Citation
If this work is insightful and relevant for your research, we would highly appreciate if you could cite it:

For any questions, please do not hesitate to reach out to Carolina Lopez at clopezolmos@microsoft.com

