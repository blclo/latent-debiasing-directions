import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text
import datetime
from matplotlib.lines import Line2D
import os
import plotly.express as px
import sys
import pickle
import plotly.graph_objects as go

class PlotConceptAttributeRelations:
    def __init__(self, base_path = None, points = None, concept = None):
        self.data_points = points
        self.base_path = base_path
        self.concept = concept
        self.df = pd.DataFrame(self.data_points, columns=["Cosine similarity in vision embeddings CLIP", "Cosine similarity in text embeddings CLIP", "Label"])

    def point_chart(self):
        # Use seaborn to create the scatter plot
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 7))

        labels_to_plot = ['Mansion', 'Luxury car', 'Designer clothing', 'African house', 'Wooden house', 'thatched roof', 'mud walls', 'Traditional African clothing', 'Dashiki', 'Black man', 'White man', 'European man', 'Asian man'] 
        df_subset = self.df[self.df['Label'].isin(labels_to_plot)]
        df = df_subset
        plot = sns.scatterplot(data=df, x="Cosine similarity in vision embeddings CLIP", y="Cosine similarity in text embeddings CLIP", hue="Label", s=100, legend=False, palette="muted")

        # Add more pronounced central axes
        plt.axhline(0, color='black', linewidth=1.5, zorder=0)
        plt.axvline(0, color='black', linewidth=1.5, zorder=0)

        plt.plot([-1, 1], [-1, 1], color='lightgrey', linestyle='--')  # Diagonal line
        # plot
        plt.plot([0, 0], [1, 1], color='lightgrey', linestyle='--')  # Diagonal line

        # Annotate each point with its label 
        texts = [plt.text(x, y, label, fontname='Arial', fontsize=8) for x, y, label in zip(df["Cosine similarity in vision embeddings CLIP"], df["Cosine similarity in text embeddings CLIP"], df["Label"])]
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
        
        # Finalize the plot
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        name_plot = f"point_chart_{self.concept}_attributes_relationship_plot.svg"
        path = os.path.join(self.base_path, name_plot)
        plt.savefig(path, dpi=300, format='svg')
        print("Point chart has been saved at ", path)
        plt.show()

    def radar_chart(self):
        print(self.df)
        self.df = self.df.sort_values(by="Cosine similarity in text embeddings CLIP", ascending=False)
        fig = px.line_polar(self.df, r='Cosine similarity in text embeddings CLIP', theta='Label', line_close=True)
        name_plot = f"spider_chart_{self.concept}_attributes_relationship_plot_ordered.svg"
        filename_path = os.path.join(self.base_path, name_plot)
        
        fig.write_image(filename_path)
        print("Radar chart stored at: " + filename_path)

    def stacked_bars(self):
        name_plot = f"stacked_bars_{self.concept}_attributes_relationship_plot.svg"
        filename_path = os.path.join(self.base_path, name_plot)

        # Sort the DataFrame in descending order based on the "Distance in text embeddings CLIP" column
        ordered = self.df.sort_values(by="Cosine similarity in text embeddings CLIP", ascending=False)
        print(ordered)
        fig = px.bar(ordered, x="Cosine similarity in vision embeddings CLIP", y="Cosine similarity in text embeddings CLIP", color="Label")
        fig.write_image(filename_path)
        print("Stacked bars chart stored at: " + filename_path)

    def single_bars(self, threshold = None):
        type_embeddings = "text"
        name_plot = f"{type_embeddings}_single_bars_{self.concept}_attributes_relationship_plot.png"
        filename_path = os.path.join(self.base_path, name_plot)

        # Sort the DataFrame in descending order based on the "Distance in text embeddings CLIP" column
        ordered = self.df.sort_values(by=f"Cosine similarity in {type_embeddings} embeddings CLIP", ascending=False)
           # Add the filtering step here

        # Take the top 10 values
        top_10_values = ordered.head(10)
        print("Printing only the 10 top values.")
        if threshold:
            ordered = ordered[ordered[f'Cosine similarity in {type_embeddings} embeddings CLIP'] > threshold]
    
        print(top_10_values)

        fig = go.Figure(data=[go.Bar(
            y=top_10_values['Label'],    # Assigning 'Label' column to y due to Horizontal Bar Chart
            x=top_10_values[f'Cosine similarity in {type_embeddings} embeddings CLIP'],
            orientation='h',    # This creates a Horizontal Bar Chart
            marker=dict(color='rgba(255, 165, 0, 0.6)'),
        )])

        # Add x-axis and y-axis labels
        fig.update_xaxes(title_text=f"Cosine similarity in CLIP's {type_embeddings} embeddings")  # Replace with your desired x-axis title
        fig.update_yaxes(title_text='Attributes', tickangle=45)  # Replace with your desired y-axis title

        fig.write_image(filename_path)
        print("Bar chart stored at: " + filename_path)

    def combined_bars(self):
        name_plot = f"combined_single_bars_{self.concept}_attributes_relationship_plot.png"
        filename_path = os.path.join(self.base_path, name_plot)

        # Sort the DataFrame in descending order based on the "Distance in text embeddings CLIP" column
        vision= self.df.sort_values(by=f"Cosine similarity in vision embeddings CLIP", ascending=False)
        text = self.df.sort_values(by=f"Cosine similarity in text embeddings CLIP", ascending=False)
        vision_head = vision.head(15)
        text_head = text.head(15)

        # Merge DataFrames on the common 'Label' column
        merged_df = vision_head.merge(text_head, on='Label', suffixes=('_vision', '_text'))

        print(merged_df.columns)

    
        # Create a bar plot
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=merged_df['Label'],
            y=merged_df['Cosine similarity in vision embeddings CLIP_vision'],
            name='Vision embeddings',
            marker=dict(color='rgba(255, 165, 0, 0.6)'),
        ))

        fig.add_trace(go.Bar(
            x=merged_df['Label'],
            y=merged_df['Cosine similarity in text embeddings CLIP_text'],
            name='Text embeddings',
            marker=dict(color='rgba(0, 128, 255, 0.6)'),
        ))

        # Customize layout if needed
        fig.update_layout(
            title='Top 10 Values Comparison Plot',
            xaxis_title='Label',
            yaxis_title='Cosine Similarity',
            barmode='group',  # Group bars side by side
        )

        # Show the plot
        fig.show()
        fig.write_image(filename_path)
        

if __name__ == "__main__":
    points = sys.argv[1]

    with open(points, 'rb') as file:
        pickle_data = pickle.load(file)

    plot = PlotConceptAttributeRelations(".", pickle_data, "A wealthy African man and his house")
    plot.single_bars()
    #plot.point_chart()