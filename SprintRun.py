"""
- runs all simulations for each sprint
"""

from SimulationRun import SimulationRun
import logging 
import time

class SprintRun:
    def __init__(self, sprint):
        self.sprint = sprint

    def simulate(self):
        start = time.time()
        logging.info(f"\n\n(SprintRun) Simulating sprint {self.sprint.id}")
        SimulationRun(self.sprint, 1).run()
        logging.info(f"(SprintRun) Sprint {self.sprint.id} simulation 1 complete")
        SimulationRun(self.sprint, 2).run()
        logging.info(f"(SprintRun) Sprint {self.sprint.id} simulation 2 complete")
        SimulationRun(self.sprint, 3).run()
        logging.info(f"(SprintRun) Sprint {self.sprint.id} simulation 3 complete")
        SimulationRun(self.sprint, 4).run()
        logging.info(f"(SprintRun) Sprint {self.sprint.id} simulation 4 complete")
        SimulationRun(self.sprint, 5).run()
        logging.info(f"(SprintRun) Sprint {self.sprint.id} simulation 5 complete")
        end = time.time()
        elapsed_minutes = (end - start) / 60
        logging.info(f"\n\n(SprintRun) {self.sprint.id} time taken: {elapsed_minutes:.2f} minutes")

