import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text
import datetime
from matplotlib.lines import Line2D
import os

"""
DCS: Difference in Cosine Similarity

This class is not used in the codebases' main scripts but can be used to obtain a graph 
computing the difference between embeddings of two classes across both text's and vision's encoders.

Reach out to: clopezolmos@microsoft.com for an example of this or more information.
"""

class LVEmbeddingsPlot:
    def __init__(self, base_path, points):
        self.data_points = points
        self.base_path = base_path
        with open("women_biased_professions.txt", "r") as file:
            self.w_biased_professions = file.read().splitlines()

    def plot(self):
        # Convert the data into a pandas DataFrame
        df = pd.DataFrame(self.data_points, columns=["DCS vision embeddings CLIP", "DCS text embeddings BERT", "Label"])

        # Add a column indicating whether the label is from women_biased_professions.txt
        df['Women_Biased'] = df['Label'].isin(self.w_biased_professions)

        # Use seaborn to create the scatter plot
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 7))

        plot = sns.scatterplot(data=df, x="DCS vision embeddings CLIP", y="DCS text embeddings BERT", hue="Label", style="Women_Biased", markers=["o", "X"], s=100, legend=False, palette="muted")

        # Custom legend for the markers
        legend_elements = [Line2D([0], [0], marker='X', color='black', label='Women stereotype', markersize=10, linestyle='None')]
        plt.legend(handles=legend_elements, loc='upper right')

        # Add more pronounced central axes
        plt.axhline(0, color='black', linewidth=1.5, zorder=0)
        plt.axvline(0, color='black', linewidth=1.5, zorder=0)

        plt.plot([-0.08, 0.08], [-0.08, 0.08], color='lightgrey', linestyle='--')  # Diagonal line

        # Annotate each point with its label 
        texts = [plt.text(x, y, label, fontname='Arial', fontsize=8) for x, y, label in zip(df["DCS vision embeddings CLIP"], df["DCS text embeddings BERT"], df["Label"])]
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

        # Add axis labels 
        plt.text(-0.0799, 0.00001, "Woman vision", ha='left', va='bottom', fontsize=10, fontname='Arial', color='black')
        plt.text(0.0799, 0.00001, "Man vision", ha='right', va='bottom', fontsize=10, fontname='Arial', color='black')
        plt.text(0.002, -0.0799, "Woman text", ha='left', va='bottom', fontsize=10, fontname='Arial', color='black')
        plt.text(0.002, 0.0799, "Man text", ha='left', va='top', fontsize=10, fontname='Arial', color='black')
        
        # Finalize the plot
        plt.xlim(-0.08, 0.08)
        plt.ylim(-0.08, 0.08)

        name_plot = "EmbeddingsPlotClipBert.png"
        path = os.path.join(self.base_path, name_plot)
        plt.savefig(path, dpi=300, format='png')
        plt.show()

