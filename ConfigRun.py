"""
- runs solver for each config run 
- generates file for each run (with mean and median values)
"""

from Solver import Solver
import time
import statistics
import logging
from Project import Project
import json 
import config
import os

class ConfigRun:
    def __init__(self, sprint, constraints, num_elicitations, error_rate):
        logging.info(f"\n\n(ConfigRun) Initializing ConfigRun for project {sprint.project_id} and sprint {sprint.id}")
        self.sprint = sprint
        self.constraints = constraints
        self.num_elicitations = num_elicitations
        self.error_rate = error_rate
        self.num_runs = config.AVG_RUN_SIZE
        self.all_solutions = {}

        self.results = {"mean_disagreements": [], "mean_average_distances": [], "median_disagreements": [], "median_average_distances": [], "mean_costs": [], "median_costs": []}

    def run(self):
        start = time.time()

        for i in range(self.num_runs):
            logging.info(f"(ConfigRun) Run {i+1}/{self.num_runs}")
            solver = Solver(self.sprint, self.constraints, self.num_elicitations, self.error_rate)
            formatted_solutions = solver.solve()
            self.all_solutions[i] = formatted_solutions

            disagreements = []
            average_distances = []
            costs = []
            for sol, cost in formatted_solutions.items():
                disagreement = solver.calculate_disagreement(sol)[0]
                average_distance = solver.calculate_avdist(sol)
                logging.info(f"[SOLUTION] - {sol}")
                logging.info(f"[COST, DISAGREEMENT] - {cost}, {disagreement}")
                disagreements.append(disagreement)
                average_distances.append(average_distance)
                costs.append(cost)

            median_disagreement = statistics.median(disagreements)
            mean_disagreement = statistics.mean(disagreements)
            median_average_distance = statistics.median(average_distances)
            mean_average_distance = statistics.mean(average_distances)
            median_cost = statistics.median(costs)
            mean_cost = statistics.mean(costs)

            logging.info(f"[TOTAL SOLUTIONS] - {len(formatted_solutions)}")
            logging.info(f"[MEDIAN DISAGREEMENT] - {median_disagreement}")
            logging.info(f"[AVG DISAGREEMENT] - {mean_disagreement}")

            self.results["median_disagreements"].append(median_disagreement)
            self.results["mean_disagreements"].append(mean_disagreement)
            self.results["median_average_distances"].append(median_average_distance)
            self.results["mean_average_distances"].append(mean_average_distance)
            self.results["median_costs"].append(median_cost)
            self.results["mean_costs"].append(mean_cost)
        end = time.time()
        elapsed_minutes = (end - start) / 60
        logging.info(f"(ConfigRun) {self.constraints, self.num_elicitations, self.error_rate} time taken: {elapsed_minutes:.2f} minutes")
        logging.info(f"\n\n(ConfigRun) {self.constraints} All Solutions: {self.all_solutions}\n\n")
        file = self.create_file()
        return file

    def create_file(self):
        res = {
            "project_id": self.sprint.project_id,
            "sprint_id": self.sprint.id,
            **self.results 
        }
        
        filepath = os.path.join(config.FINAL_RESULTS, f"proj{self.sprint.project_id}_sprint{self.sprint.id}_{self.constraints}_e{self.num_elicitations}_err{self.error_rate}.json")
        with open(filepath, "w") as f:
            json.dump(res, f, indent=4)

        logging.info(f"(ConfigRun) [FILE SAVED] - Results saved to {filepath}")
        return filepath
