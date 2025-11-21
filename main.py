"""
- runs program for each project and sprint
"""
import concurrent.futures
import logging
import config
from SprintRun import SprintRun
from Project import Project
import os
import time
import shutil
from concurrent.futures import as_completed

def setup_logger(log_file):
    logger = logging.getLogger()  
    logger.handlers = []         
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def run_sprint(args):
    start = time.time()
    sprint_id, project_id = args
    sprint_log = f"logs/sprint_{project_id}_{sprint_id}.log"
    setup_logger(sprint_log)

    print(f"Process {os.getpid()} starting sprint {sprint_id} project {project_id}")
    logging.info(f"Starting Sprint {sprint_id} in Project {project_id}")

    project = Project(project_id)
    sprint = next(s for s in project.sprints if s.id == sprint_id)

    SprintRun(sprint).simulate()

    logging.info(f"Finished Sprint {sprint_id} in Project {project_id}")
    end = time.time()
    elapsed_minutes = (end - start) / 60
    print(f"Process {os.getpid()} finished sprint {sprint_id} project {project_id} in {elapsed_minutes} minutes.")

def main():
    for path_ in ["logs", "plots", "results"]:
        if os.path.exists(path_):
            shutil.rmtree(path_)
        os.makedirs(path_, exist_ok=True)

    sprint_tasks = []
    for project_id in config.PROJECTS:
        project = Project(project_id)
        for sprint in project.sprints:
            sprint_tasks.append((sprint.id, project_id))
    max_workers = min(12, len(sprint_tasks))  # use up to 10 workers or less if fewer tasks
    print(f"Running with {max_workers} parallel workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_sprint, args): args for args in sprint_tasks}
        
        for future in as_completed(futures):
            args = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Sprint {args} failed: {e}")
    
    print("All sprint simulations complete.")

if __name__ == "__main__":
    main()
