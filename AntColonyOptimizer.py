import networkx as nx
import numpy as np
import ast
import random
import pandas as pd
import config
from Sprint import Sprint
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

# ===============================================================
# === AntColonyPrioritizer class===
# ===============================================================
class AntColonyPrioritizer:
    def __init__(self, info_df, project_id, num_ants=10, iterations=30, alpha=1, beta=1, rho=0.1):
        self.info_df = info_df.copy()
        self.project_id = project_id
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        if 'Creation_Date' in self.info_df.columns:
            self.info_df['Creation_Date'] = pd.to_datetime(self.info_df['Creation_Date'], errors='coerce')

        self.G = self.build_dependency_graph(self.info_df)
        self.issues = list(self.G.nodes)

        # Initialize pheromone levels
        self.pheromone = {
            (i, j): 1.0 for i in self.issues for j in self.issues if i != j
        }

    def build_dependency_graph(self, info_df):
        G = nx.DiGraph()

        for issue_id in info_df['Issue_ID']:
            G.add_node(issue_id)

        for _, row in info_df.iterrows():
            issue = row['Issue_ID']
            deps = row['Dependencies']

            if isinstance(deps, str):
                try:
                    deps = ast.literal_eval(deps)
                except:
                    deps = []

            if deps and isinstance(deps, list):
                for dep in deps:
                    G.add_edge(dep, issue)

        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Dependency graph is not acyclic!")
        return G

    def get_valid_candidates(self, completed):
        """Get issues whose dependencies are all completed."""
        valid = []
        for node in self.G.nodes:
            if node not in completed:
                deps = list(self.G.predecessors(node))
                if all(dep in completed for dep in deps):
                    valid.append(node)
        return valid

    def heuristic(self, i, j):
        """
        Heuristic desirability for moving from issue i → j.
        Estimates violations if j comes after i.
        Lower violations = higher heuristic.
        """
        row_i = self.info_df.loc[self.info_df['Issue_ID'] == i]
        row_j = self.info_df.loc[self.info_df['Issue_ID'] == j]

        violations = 0

        # update to only use c1p1::
        if float(row_j['Priority_Class'].values[0]) > float(row_i['Priority_Class'].values[0]):
            violations += 1
        if row_j['Creation_Date'].values[0] < row_i['Creation_Date'].values[0]:
            violations += 1

        heuristic_value = 1.0 / (1.0 + violations)
        return heuristic_value

    def heuristic_first_issue(self, j):
        """
        Heuristic for selecting the first issue (no previous issue to compare).
        Better issues (lower priority, earlier creation) = higher heuristic.
        """
        row_j = self.info_df.loc[self.info_df['Issue_ID'] == j]
        
        priority_score = 1.0 / (float(row_j['Priority_Class'].values[0]) + 1)
        
        date_val = pd.Timestamp(row_j['Creation_Date'].values[0])
        date_timestamp = date_val.timestamp()
        date_score = 1.0 / (date_timestamp / 86400 + 1)
        
        return priority_score + date_score

    def construct_solution(self):
        """Build one complete solution path using ACO probabilities."""
        completed = set()
        path = []

        for _ in range(len(self.issues)):
            candidates = self.get_valid_candidates(completed)
            if not candidates:
                break

            probs = []
            for c in candidates:
                if path:
                    tau = self.pheromone.get((path[-1], c), 1.0) ** self.alpha
                    eta = self.heuristic(path[-1], c) ** self.beta
                else:
                    tau = 1.0
                    eta = self.heuristic_first_issue(c) ** self.beta
                
                probs.append(tau * eta)

            probs = np.array(probs, dtype=float)
            probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)

            next_issue = np.random.choice(candidates, p=probs)
            path.append(next_issue)
            completed.add(next_issue)

        return path

    def evaluate_path(self, path):
        """
        Count total violations in the path.
        Lower score = better.
        """
        violations = 0

        for idx in range(len(path) - 1):
            current_issue = path[idx]
            next_issue = path[idx + 1]

            row_current = self.info_df.loc[self.info_df['Issue_ID'] == current_issue]
            row_next = self.info_df.loc[self.info_df['Issue_ID'] == next_issue]

            if float(row_next['Priority_Class'].values[0]) > float(row_current['Priority_Class'].values[0]):
                violations += 1
                
            if row_next['Creation_Date'].values[0] < row_current['Creation_Date'].values[0]:
                violations += 1

        return violations

    def run(self):
        """Run ACO algorithm for specified iterations."""
        best_path = None
        best_score = float('inf')

        for iteration in range(self.iterations):
            all_paths = []
            all_scores = []

            for _ in range(self.num_ants):
                path = self.construct_solution()
                score = self.evaluate_path(path)
                all_paths.append(path)
                all_scores.append(score)

            for key in self.pheromone:
                self.pheromone[key] *= (1 - self.rho)

            min_idx = np.argmin(all_scores)
            best_iter_path = all_paths[min_idx]
            best_iter_score = all_scores[min_idx]

            if best_iter_score < best_score:
                best_score = best_iter_score
                best_path = best_iter_path

            for i in range(len(best_iter_path) - 1):
                edge = (best_iter_path[i], best_iter_path[i + 1])
                if edge in self.pheromone:
                    self.pheromone[edge] += 1.0 / (best_iter_score + 1.0)

        return best_path, best_score


# NB: These must be defined at module level for pickling

def run_single_aco(info_df, project_id, run_idx, sprint_id, num_ants=10, iterations=50):
    """
    Run a single ACO instance - designed to be called in parallel.
    Each process gets its own random seed based on run_idx.
    """
    random.seed(run_idx)
    np.random.seed(run_idx)
    
    aco = AntColonyPrioritizer(info_df, project_id=project_id, 
                               num_ants=num_ants, iterations=iterations)
    best_order, best_score = aco.run()
    
    return {
        "project": int(project_id),
        "sprint": int(sprint_id),
        "run_idx": run_idx,
        "order": [int(x) for x in best_order],
        "score": float(best_score)
    }


def process_sprint_parallel(proj, sprint, loc, avg_run_size, num_ants=10, iterations=50, max_workers=None):
    """
    Process a single sprint with parallelized runs.
    
    Args:
        proj: Project ID
        sprint: Sprint number
        loc: Location path
        avg_run_size: Number of runs to perform
        num_ants: Number of ants for ACO
        iterations: Number of iterations for ACO
        max_workers: Maximum number of parallel workers (None = use all cores)
    """
    s = Sprint(sprint, loc)
    
    run_func = partial(run_single_aco, s.info_df, int(proj), 
                      sprint_id=sprint, num_ants=num_ants, iterations=iterations)
    
    sprint_runs = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_run = {executor.submit(run_func, x): x for x in range(avg_run_size)}
        
        for future in as_completed(future_to_run):
            run_idx = future_to_run[future]
            try:
                result = future.result()
                sprint_runs.append(result)
                print(f"Project {proj}, Sprint {sprint}, Run {run_idx+1} completed")
                print(f"  Score: {result['score']}")
            except Exception as exc:
                print(f"Project {proj}, Sprint {sprint}, Run {run_idx+1} generated exception: {exc}")
    
    # Sort runs by run_idx to maintain order
    sprint_runs.sort(key=lambda x: x['run_idx'])
    
    return {
        "project_id": int(proj),
        "sprint_id": int(sprint),
        "runs": sprint_runs
    }

"""Main execution function - MUST be called from if __name__ == '__main__' block"""
def main():
    aco_sols = []
    
    # Determine number of workers (leaving some cores free for system)
    max_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {max_workers} parallel workers")
    
    for proj in config.PROJECTS:
        loc = os.path.join(os.getcwd(), "input", f"P{proj}")
        
        for sprint in range(1, 6):
            print(f"\n=== Processing Project {proj}, Sprint {sprint} ===")
            
            sprint_result = process_sprint_parallel(
                proj, sprint, loc,
                avg_run_size=config.AVG_RUN_SIZE,
                num_ants=config.POPULATION_SIZE,
                iterations=50,
                max_workers=max_workers
            )
            
            aco_sols.append(sprint_result)
    
    with open("aco_results.json", "w") as f:
        json.dump(aco_sols, f, indent=2)
    
    print(f"Results saved to aco_results.json")


if __name__ == "__main__":
    # NB: This guard is required on Windows to prevent infinite process spawning
    mp.freeze_support() 
    main()