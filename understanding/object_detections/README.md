# Object Detection in Generations w/ Kosmos 2
The code found here allows you to obtain a histogram with the attributes found for a specific concept without the need of providing a text file with the attributes defined. The following figure presents the methodology of the pipeline.

![kosmos2_pipeline](https://github.com/blclo/latent-debiasing-directions/blob/main/understanding/kosmos2_ODIG/examples/ODIG_Kosmos2_wGraph.png)

To run the Object Detection In Generations pipeline using Kosmos 2, first define inside the `kosmos2_analysis.py` script the folder containing your images. Secondly, modify the name of the pickle file path where a dictionary will store the findings: attribute detected, ROI dimensions and bounding box coordinates. 

Once you have succesfully ran the `kosmos2_analysis.py` script you can proceed to provide the name of your pickle file to the `pkl_to_graph.py` script. With it you will obtain the graph showing the attribute count and all the detected attributes marked in the form of bounding boxes in the images. 

![example](https://github.com/blclo/latent-debiasing-directions/blob/main/understanding/kosmos2_ODIG/examples/vscode_kosmos2_example.png)