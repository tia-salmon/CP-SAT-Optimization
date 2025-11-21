"""
- runs all configs for each simulation
- creates graph for each simulation
"""
import config
import math
import os 

from ConfigRun import ConfigRun
from GeneratePlot import GeneratePlot
import logging

class SimulationRun:
    def __init__(self, sprint, id):
        self.sprint = sprint
        self.id = id
        
        self.files = []
        self.labels = []
        self.title = None
        self.x_label = None

        # not being used for paper
        self.n = len(sprint.issues)
        self.total_elicitations = (self.n * (self.n - 1)) // 2

    # not being used for paper
    def get_num_elicitations(self, percentage):
        return math.ceil(self.total_elicitations * percentage/100)
    
    def run(self):
        if self.id == 1:
            constraint = "c1p1"
            logging.info(f"(SimulationRun) Running simulation 1 for project {self.sprint.project_id} sprint {self.sprint.id}")
            self.files = []
            for n_eli in config.VARYING_ELI:
                if n_eli != 50:
                    cr = ConfigRun(self.sprint, constraint, n_eli, 0).run()
                    self.files.append(cr)
                    self.labels.append(f"eli{n_eli}")
            self.title = f"Varying Elicitations ({', '.join(str(e) for e in config.VARYING_ELI)})\n Under {constraint} Constraints with 0% Error - Project {self.sprint.project_id}, Sprint {self.sprint.id}"
            self.x_label = "Number of Elicitations"
        elif self.id == 2:
            logging.info(f"(SimulationRun) Running simulation 2 for project {self.sprint.project_id} sprint {self.sprint.id}")
            for const in config.VARYING_CONST:
                cr = ConfigRun(self.sprint, const, 50, 0).run()
                self.files.append(cr)
                self.labels.append(f"{const}")
            self.title = f"Varying Constraints ({', '.join(f'{e}' for e in config.VARYING_CONST)})\n with 50 Elicitations and 0% Error - Project {self.sprint.project_id}, Sprint {self.sprint.id}"
            self.x_label = "Constraints"
        elif self.id == 3:
            logging.info(f"(SimulationRun) Running simulation 3 for project {self.sprint.project_id} sprint {self.sprint.id}")
            for const in config.VARYING_WEIGHT:
                if const != "c1p1t1": # remove constraint already generated in diff simulation
                    cr = ConfigRun(self.sprint, const, 50, 0).run()
                    self.files.append(cr)
                    self.labels.append(f"{const}")
            global_res_file = os.path.join(config.FINAL_RESULTS, f"proj{self.sprint.project_id}_sprint{self.sprint.id}_c1p1t1_e50_err0.json")
            self.files = [global_res_file] + self.files
            self.labels = ["c1p1t1"] + self.labels
            self.title = f"Varying Weights ({', '.join(f'{e}' for e in config.VARYING_WEIGHT)})\n Under cpt Constraints with 50 Elicitations and 0% Error Rate - Project {self.sprint.project_id}, Sprint {self.sprint.id}"
            self.x_label = "Constraints"
        elif self.id == 4:
            constraint = "c1p1"
            logging.info(f"(SimulationRun) Running simulation 4 for project {self.sprint.project_id} sprint {self.sprint.id}")
            for err in config.VARYING_ERROR_RATE:
                if err != 0: # remove constraint already generated in diff simulation
                    cr = ConfigRun(self.sprint, constraint, 50, err).run()
                    self.files.append(cr)
                    self.labels.append(f"{err}%")

            self.files = [os.path.join(config.FINAL_RESULTS, f"proj{self.sprint.project_id}_sprint{self.sprint.id}_{constraint}_e50_err0.json")] + self.files
            self.labels = ["0%"] + self.labels
            cr_b = ConfigRun(self.sprint, "c1", 0, 0).run() # baseline config
            self.files.insert(0, cr_b)
            self.labels.insert(0, "b")
            self.title = f"Varying Error Rates ({f"b, {', '.join(f'{e}%' for e in config.VARYING_ERROR_RATE)}"})\n Under {constraint} Constraints with 50 Elicitations - Project {self.sprint.project_id}, Sprint {self.sprint.id}"
            self.x_label = "Error Rates"
            cr = ConfigRun(self.sprint, "cl", 50, 0).run()
        elif self.id == 5:
            logging.info(f"(SimulationRun) Running simulation 5 for project {self.sprint.project_id} sprint {self.sprint.id}")
            for const in config.VARYING_LOCAL_CONST:
                if const != "cl":
                    cr = ConfigRun(self.sprint, const, 50, 0).run()
                else:
                    cr = ConfigRun(self.sprint, const, 0, 0).run()
                self.files.append(cr)
                self.labels.append(f"{const}")
            self.title = f"Varying Constraints ({', '.join(f'{e}' for e in config.VARYING_CONST)})\n with 50 Elicitations and 0% Error - Project {self.sprint.project_id}, Sprint {self.sprint.id}"
            self.x_label = "Constraints"

        GeneratePlot(self.files, self.title, self.x_label, self.labels, self.id, self.sprint).generate_simulation_plot()

