import networkx as nx
import numpy as np
import ast
import random
import pandas as pd
import os
import json
import config
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from Sprint import Sprint
import multiprocessing as mp


# ===============================================================
# === MOPSOPrioritizer class ===
# ===============================================================
class MOPSOPrioritizer:
    def __init__(self, info_df, project_id, num_particles=20, iterations=30, w=0.5, c1=1.5, c2=1.5, archive_size=50):
        self.info_df = info_df.copy()
        self.project_id = project_id
        self.num_particles = num_particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.archive_size = archive_size

        if 'Creation_Date' in self.info_df.columns:
            self.info_df['Creation_Date'] = pd.to_datetime(self.info_df['Creation_Date'], errors='coerce')

        self.G = self.build_dependency_graph(self.info_df)
        self.issues = list(self.G.nodes)
        self.n_issues = len(self.issues)

        self.particles = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_objectives = []
        self.archive = []
        self.archive_objectives = []

        self._initialize_swarm()

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

    def _initialize_swarm(self):
        for _ in range(self.num_particles):
            position = self._generate_valid_solution()
            self.particles.append(position)
            velocity = np.random.uniform(-2, 2, self.n_issues)
            self.velocities.append(velocity)
            self.personal_best_positions.append(position.copy())
            objectives = self.evaluate_solution(position)
            self.personal_best_objectives.append(objectives)
            self._update_archive(position, objectives)

    def _generate_valid_solution(self):
        completed = set()
        solution = []
        while len(completed) < self.n_issues:
            candidates = self._get_valid_candidates(completed)
            if not candidates:
                break
            next_issue = random.choice(candidates)
            solution.append(next_issue)
            completed.add(next_issue)
        return solution

    def _get_valid_candidates(self, completed):
        valid = []
        for node in self.G.nodes:
            if node not in completed:
                deps = list(self.G.predecessors(node))
                if all(dep in completed for dep in deps):
                    valid.append(node)
        return valid

    def evaluate_solution(self, solution):
        priority_violations = 0
        creation_violations = 0
        for idx in range(len(solution) - 1):
            current_issue = solution[idx]
            next_issue = solution[idx + 1]
            row_current = self.info_df.loc[self.info_df['Issue_ID'] == current_issue]
            row_next = self.info_df.loc[self.info_df['Issue_ID'] == next_issue]

            if row_next['Creation_Date'].values[0] < row_current['Creation_Date'].values[0]:
                creation_violations += 1
            if float(row_next['Priority_Class'].values[0]) > float(row_current['Priority_Class'].values[0]):
                    priority_violations += 1
      
        return (priority_violations, creation_violations)

    def dominates(self, obj1, obj2):
        better_in_one = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:
                return False
            if o1 < o2:
                better_in_one = True
        return better_in_one

    def _update_archive(self, solution, objectives):
        dominated = False
        to_remove = []
        for i, (arch_sol, arch_obj) in enumerate(zip(self.archive, self.archive_objectives)):
            if self.dominates(arch_obj, objectives):
                dominated = True
                break
            elif self.dominates(objectives, arch_obj):
                to_remove.append(i)
        for i in reversed(to_remove):
            del self.archive[i]
            del self.archive_objectives[i]
        if not dominated:
            self.archive.append(solution.copy())
            self.archive_objectives.append(objectives)
            if len(self.archive) > self.archive_size:
                self._prune_archive()

    def _prune_archive(self):
        if len(self.archive) <= self.archive_size:
            return
        min_dist = float('inf')
        remove_idx = 0
        for i in range(len(self.archive_objectives)):
            min_neighbor_dist = float('inf')
            for j in range(len(self.archive_objectives)):
                if i != j:
                    dist = sum((a - b) ** 2 for a, b in zip(self.archive_objectives[i], self.archive_objectives[j]))
                    min_neighbor_dist = min(min_neighbor_dist, dist)
            if min_neighbor_dist < min_dist:
                min_dist = min_neighbor_dist
                remove_idx = i
        del self.archive[remove_idx]
        del self.archive_objectives[remove_idx]

    def _select_global_best(self):
        if not self.archive:
            return None
        if len(self.archive) == 1:
            return self.archive[0]
        return random.choice(self.archive)

    def _apply_velocity_and_repair(self, particle_idx):
        particle = self.particles[particle_idx]
        velocity = self.velocities[particle_idx]
        personal_best = self.personal_best_positions[particle_idx]
        global_best = self._select_global_best()
        if global_best is None:
            global_best = particle
        r1, r2 = random.random(), random.random()
        personal_influence = np.zeros(self.n_issues)
        global_influence = np.zeros(self.n_issues)
        for i, issue in enumerate(particle):
            try:
                pb_pos = personal_best.index(issue)
                personal_influence[i] = pb_pos - i
            except ValueError:
                pass
            try:
                gb_pos = global_best.index(issue)
                global_influence[i] = gb_pos - i
            except ValueError:
                pass
        velocity = (self.w * velocity + 
                   self.c1 * r1 * personal_influence + 
                   self.c2 * r2 * global_influence)
        self.velocities[particle_idx] = velocity
        new_particle = particle.copy()
        num_swaps = int(min(self.n_issues / 2, abs(velocity).sum() / 10))
        for _ in range(num_swaps):
            idx1 = random.randint(0, self.n_issues - 1)
            idx2 = random.randint(0, self.n_issues - 1)
            new_particle[idx1], new_particle[idx2] = new_particle[idx2], new_particle[idx1]
            if not self._is_valid_ordering(new_particle):
                new_particle[idx1], new_particle[idx2] = new_particle[idx2], new_particle[idx1]
        self.particles[particle_idx] = new_particle

    def _is_valid_ordering(self, solution):
        position = {issue: idx for idx, issue in enumerate(solution)}
        for issue in solution:
            deps = list(self.G.predecessors(issue))
            for dep in deps:
                if dep not in position or position[dep] >= position[issue]:
                    return False
        return True

    def run(self):
        for iteration in range(self.iterations):
            for i in range(self.num_particles):
                self._apply_velocity_and_repair(i)
                objectives = self.evaluate_solution(self.particles[i])
                if self.dominates(objectives, self.personal_best_objectives[i]):
                    self.personal_best_positions[i] = self.particles[i].copy()
                    self.personal_best_objectives[i] = objectives
                self._update_archive(self.particles[i], objectives)
        if self.archive:
            best_idx = min(range(len(self.archive)), key=lambda i: sum(self.archive_objectives[i]))
            best_solution = self.archive[best_idx]
            best_objectives = self.archive_objectives[best_idx]
            return {
                'best_solution': best_solution,
                'best_objectives': best_objectives,
                'total_violations': sum(best_objectives),
                'pareto_front': self.archive.copy(),
                'pareto_objectives': self.archive_objectives.copy()
            }
        return None

# ===============================================================
# === Run a single MOPSO run ===
# ===============================================================
def run_mopso_once(project_id, sprint_id, run_id, info_df):
    mopso = MOPSOPrioritizer(info_df, project_id=int(project_id), num_particles=config.POPULATION_SIZE, iterations=50)
    result = mopso.run()
    print(f"Project {project_id}, Sprint {sprint_id}, Run {run_id} complete.")
    return {
        "project": int(project_id),
        "sprint": int(sprint_id),
        "run": run_id,
        "solution": [int(i) for i in result["best_solution"]],
        "total_violations": int(result["total_violations"]),
        "objectives": [int(v) for v in result["best_objectives"]]
    }

# ===============================================================
# === Parallelized execution over projects and sprints ===
# ===============================================================

def main():
    mopso_sols = []

    for proj in config.PROJECTS:
        loc = os.path.join(os.getcwd(), "input", f"P{proj}")
        for sprint in range(1, 6):
            print(f"\nRunning Project {proj}, Sprint {sprint}")
            s = Sprint(sprint, loc)
            info_df = s.info_df

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_mopso_once, proj, sprint, x + 1, info_df)
                        for x in range(config.AVG_RUN_SIZE)]
                sprint_runs = [f.result() for f in as_completed(futures)]

            mopso_sols.append({
                "project_id": int(proj),
                "sprint_id": int(sprint),
                "runs": sprint_runs
            })

    with open("mopso_results.json", "w") as f:
        json.dump(mopso_sols, f, indent=2)

    print("Results saved to mopso_results.json.")

if __name__ == "__main__":
    # NB: This guard is required on Windows to prevent infinite process spawning
    mp.freeze_support() 
    main()