"""
- Generates box plot per simulation (Sprint-Level)
"""
import matplotlib.pyplot as plt
import os 
import json
import logging
import config

class GeneratePlot:
    def __init__(self, files, title, x_label, labels, sim_id, sprint):
        self.files = files
        self.title = title
        self.x_label = x_label
        self.labels = labels
        self.sim_id = sim_id
        self.sprint = sprint

    def plot_graph(self, plot_type, values):
        boxplot_color = 'lightblue'
        plot_title = f'Median {plot_type} Distribution for {self.title}'

        plt.figure(figsize=(max(10, len(self.labels) * 1.2), 6))
        box = plt.boxplot(values, labels=self.labels, patch_artist=True)

        for patch in box['boxes']:
            patch.set_facecolor(boxplot_color)

        plt.ylabel(plot_type, fontsize=12)
        plt.xlabel(self.x_label)
        plt.xticks(rotation=45, ha='right', fontsize=12)  
        plt.yticks(rotation=0)  
        
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f"proj{self.sprint.project_id}_sprint{self.sprint.id}_sim{self.sim_id}_{plot_type.lower().replace(" ", "_")}.pdf"))
        plt.close()

    def generate_simulation_plot(self):
        logging.info(f"(Generate Plot) Generating plot for simulation {self.sim_id}, project {self.sprint.project_id}, sprint {self.sprint.id}")
        all_disagreements = []
        all_average_distances = []

        for file in self.files:
            logging.info(f"(Generate Plot) Files: {self.files}")
            with open(file, "r") as f:
                data = json.load(f)
                all_disagreements.append(data["median_disagreements"])
                all_average_distances.append(data["median_average_distances"])
        self.plot_graph("Disagreement", all_disagreements)
        self.plot_graph("Average Distance", all_average_distances)
        logging.info(f"(Generate Plot) All Disagreements: {all_disagreements}")
        logging.info(f"(Generate Plot) All Avg Distances: {all_average_distances}")


