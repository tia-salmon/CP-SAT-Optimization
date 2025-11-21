"""
- runs solver until final set of solutions generated
"""

from ortools.sat.python import cp_model
import random
from itertools import combinations
import config 
import math
import logging
import time


class Solver:
    def __init__(self, sprint, constraints, num_elicitations, error_rate):
        self.sprint = sprint
        self.constraints = constraints
        self.num_elicitations = num_elicitations
        self.error_rate = error_rate / 100
        self.max_error_pairs = math.ceil(self.num_elicitations * self.error_rate)

        self.model = None
        self.solver = None

        self.positions = []
        self.soft_constraint_penalties = []
        self.dep_constraints = set()
        self.elicited_pairs = set()
        self.error_pairs = set()
        self.solutions = {}    # raw solutions with cost
        self.formatted_solutions = {}  # formatted solutions with cost

        logging.info(f"Initialized solver - Sprint: {self.sprint.id}, Constraints: {self.constraints}, Elicitations: {self.num_elicitations}, Error Rate: {self.error_rate * 100:.1f}%, Num issues: {len(self.sprint.issues)}. Max error pairs: {self.max_error_pairs}")
        logging.info(f"Issues: {self.sprint.issues}")

        self.initialize_model()

    def initialize_model(self):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = 120

        n = len(self.sprint.issues)
        self.positions = [self.model.NewIntVar(0, n - 1, f'pos_{i}') for i in range(n)]
        self.model.AddAllDifferent(self.positions)

        self.add_dependency_constraints()

        self.add_soft_constraints()

        self.model.Minimize(sum(v * w for v, w, _ in self.soft_constraint_penalties))
        logging.info("Initialized model")

    def add_dependency_constraints(self):
        logging.info(f"Adding dependency constraints: {self.sprint.dep_df['Dependencies'].apply(lambda x: len(eval(x))).sum()} total")
        for _, row in self.sprint.dep_df.iterrows():
            issue_a = row['Issue_ID']
            dependencies = eval(row['Dependencies'])
            for dep_issue in dependencies:
                idx_a = self.sprint.idx_dict[issue_a]
                idx_dep = self.sprint.idx_dict[dep_issue]
                logging.info(f"[DEPENDENCY] - {(dep_issue, issue_a)}")
                self.model.Add(self.positions[idx_dep] < self.positions[idx_a])
                self.dep_constraints.add((dep_issue, issue_a))

    def add_soft_constraints(self):
        if self.constraints == "c1":
            self.add_creation_date_constraint(1)
        elif self.constraints == "c1p1":
            self.add_creation_date_constraint(1)
            self.add_priority_constraint(1)
        elif self.constraints == "c1t1":
            self.add_creation_date_constraint(1)
            self.add_type_constraint(1)
        elif self.constraints == "c1p1t1":
            self.add_creation_date_constraint(1)
            self.add_priority_constraint(1)
            self.add_type_constraint(1)
        elif self.constraints == "c2p1t1":
            self.add_creation_date_constraint(2)
            self.add_priority_constraint(1)
            self.add_type_constraint(1)
        elif self.constraints == "c1p2t1":
            self.add_creation_date_constraint(1)
            self.add_priority_constraint(2)
            self.add_type_constraint(1)
        elif self.constraints == "c1p1t2":
            self.add_creation_date_constraint(1)
            self.add_priority_constraint(1)
            self.add_type_constraint(2)
        elif self.constraints == "cl":
            self.add_creation_date_constraint_local(1)
        elif self.constraints == "clpl":
            self.add_creation_date_constraint_local(1)
            self.add_priority_constraint_local(1)
        elif self.constraints == "cltl":
            self.add_creation_date_constraint_local(1)
            self.add_type_constraint_local(1)
        elif self.constraints == "clpltl":
            self.add_creation_date_constraint_local(1)
            self.add_priority_constraint_local(1)
            self.add_type_constraint_local(1)
                
    def add_priority_constraint(self, w):
        logging.info(f"Adding priority constraints")
        prios = list(self.sprint.prio_df.set_index('Issue_ID')['Priority_Class'].to_dict().items())

        for i in range(len(prios)):
            issue_a, priority_a = prios[i]
            for j in range(i + 1, len(prios)):
                issue_b, priority_b = prios[j]

                if priority_a != priority_b:
                    idx_a = self.sprint.idx_dict[issue_a] # could make this into a function
                    idx_b = self.sprint.idx_dict[issue_b]
                    # issue a should come before issue b
                    violated, meta = self.create_violation_bool("prio", idx_a, idx_b, (issue_a, issue_b))

                    self.soft_constraint_penalties.append((violated, w, meta))
                    logging.info(f"[PRIORITY] - {(issue_a, issue_b)}")

    def add_creation_date_constraint(self, w):
        logging.info(f"Adding creation date constraints")
        creation_dates = list(self.sprint.creation_date_df.set_index('Issue_ID')['Creation_Date'].to_dict().items())
        for i in range(len(creation_dates)):
            issue_a, date_a = creation_dates[i]
            for j in range(i + 1, len(creation_dates)):
                issue_b, date_b = creation_dates[j]

                if date_a != date_b:
                    idx_a = self.sprint.idx_dict[issue_a] # could make this into a function
                    idx_b = self.sprint.idx_dict[issue_b]

                    violated, meta = self.create_violation_bool("creation_date", idx_a, idx_b, (issue_a, issue_b))

                    self.soft_constraint_penalties.append((violated, w, meta))
                    logging.info(f"[CREATION DATE] - {(issue_a, issue_b)}")

    def add_type_constraint(self, w):
        logging.info(f"Adding type constraints")
        types = list(self.sprint.type_df.set_index('Issue_ID')['Type_Class'].to_dict().items())

        for i in range(len(types)):
            issue_a, type_a = types[i]
            for j in range(i + 1, len(types)):
                issue_b, type_b = types[j]

                if type_a != type_b:
                    idx_a = self.sprint.idx_dict[issue_a] # could make this into a function
                    idx_b = self.sprint.idx_dict[issue_b]
                    violated, meta = self.create_violation_bool("type", idx_a, idx_b, (issue_a, issue_b))

                    self.soft_constraint_penalties.append((violated, w, meta))
                    logging.info(f"[TYPE] - {(issue_a, issue_b)}")

    def add_priority_constraint_local(self, w):
        logging.info(f"Adding local priority constraints")
        for _, prio_group in self.sprint.prio_df.groupby("Assignee_ID"):
            prios = list(prio_group.set_index('Issue_ID')['Priority_Class'].to_dict().items())

            if len(prios) > 1:
                for i in range(len(prios)):
                    issue_a, priority_a = prios[i]
                    for j in range(i + 1, len(prios)):
                        issue_b, priority_b = prios[j]

                        if priority_a != priority_b:
                            idx_a = self.sprint.idx_dict[issue_a]
                            idx_b = self.sprint.idx_dict[issue_b]

                            # Earlier-created issues should ideally come first
                            violated, meta = self.create_violation_bool("l_prio", idx_a, idx_b, (issue_a, issue_b))

                            self.soft_constraint_penalties.append((violated, w, meta))
                            logging.info(f"[LOCAL PRIO] - {(issue_a, issue_b)}")

    def add_creation_date_constraint_local(self, w):
        logging.info(f"Adding local creation date constraints")
        for _, date_group in self.sprint.creation_date_df.groupby("Assignee_ID"):
            creation_dates = list(date_group.set_index('Issue_ID')['Creation_Date'].to_dict().items())

            if len(creation_dates) > 1: 
                for i in range(len(creation_dates)):
                    issue_a, date_a = creation_dates[i]
                    for j in range(i + 1, len(creation_dates)):
                        issue_b, date_b = creation_dates[j]

                        if date_a != date_b:
                            idx_a = self.sprint.idx_dict[issue_a]
                            idx_b = self.sprint.idx_dict[issue_b]

                            # Earlier-created issues should ideally come first
                            violated, meta = self.create_violation_bool("l_creation_date", idx_a, idx_b, (issue_a, issue_b))
                            self.soft_constraint_penalties.append((violated, w, meta))
                            logging.info(f"[LOCAL CREATION DATE] - {(issue_a, issue_b)}")

    def add_type_constraint_local(self, w):
        logging.info(f"Adding local type constraints")
        # group issues by assignee
        for _, type_group in self.sprint.type_df.groupby('Assignee_ID'):
            types = list(type_group.set_index('Issue_ID')['Type_Class'].to_dict().items())
            if len(types) > 1: # only assignees with multiple issues
                for i in range(len(types)):
                    issue_a, type_a = types[i]
                    for j in range(i + 1, len(types)):
                        issue_b, type_b = types[j]

                        if type_a != type_b:
                            idx_a = self.sprint.idx_dict[issue_a]
                            idx_b = self.sprint.idx_dict[issue_b]

                            # Issue A should come before B (lower type_class number)
                            violated, meta = self.create_violation_bool("l_type", idx_a, idx_b, (issue_a, issue_b))
                            self.soft_constraint_penalties.append((violated, w, meta))
                            logging.info(f"[LOCAL TYPE] - {(issue_a, issue_b)}")

    def create_violation_bool(self, prefix, i, j, issues):
        violated = self.model.NewBoolVar(f'{prefix}_violated_{i}_{j}')
        self.model.Add(self.positions[j] < self.positions[i]).OnlyEnforceIf(violated)
        self.model.Add(self.positions[j] > self.positions[i]).OnlyEnforceIf(violated.Not())
        return violated, (prefix, *issues)

    def block_solution(self, solution):
        logging.info(f"[BLOCK] - Blocking solution: {self.format_solution(solution)}")
        # Add constraint to forbid this exact solution next time
        diff_vars = []
        for i, val in enumerate(solution):
            b = self.model.NewBoolVar(f'diff_{i}')
            self.model.Add(self.positions[i] != val).OnlyEnforceIf(b)
            self.model.Add(self.positions[i] == val).OnlyEnforceIf(b.Not())
            diff_vars.append(b)
        self.model.AddBoolOr(diff_vars)

    def add_solution(self, solution, cost):
        logging.info(f"[ADD] - Adding solution to list: {self.format_solution(solution)}")
        self.solutions[solution] = cost
        formatted = self.format_solution(solution)
        self.formatted_solutions[formatted] = cost

    def format_solution(self, solution):
        ordered_issues = [None] * len(solution)
        for issue, pos in zip(self.sprint.issues, solution):
            ordered_issues[pos] = issue
        return tuple(ordered_issues)

    def calculate_disagreement(self, other_order):
        gold_standard_index = self.sprint.idx_dict

        total_disagreements = 0
        disagree_pairs = []

        NUMREQ = len(other_order)

        for i in range(NUMREQ):
            for j in range(i + 1, NUMREQ):
                issue1 = other_order[i]
                issue2 = other_order[j]

                gs_idx1 = gold_standard_index[issue1]
                gs_idx2 = gold_standard_index[issue2]

                if (i < j and gs_idx1 > gs_idx2) or (i > j and gs_idx1 < gs_idx2):
                    total_disagreements += 1
                    disagree_pairs.append((issue1, issue2))
        return total_disagreements, disagree_pairs

    def calculate_avdist(self, req_order):
        gold_standard_index = self.sprint.idx_dict

        total_distance = 0
        numreq = len(req_order)

        for i, issue in enumerate(req_order):
            idx_individual = i
            
            idx_gold_standard = gold_standard_index[issue]
            
            distance = abs(idx_individual - idx_gold_standard)
            
            total_distance += distance
        
        average_distance = total_distance / numreq
        return average_distance

    def call_solver(self):
        status = self.solver.Solve(self.model)
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]: 
            logging.error("No more feasible solutions")
            return -1

        sol = tuple(self.solver.Value(var) for var in self.positions)
        cost = self.solver.ObjectiveValue()

        logging.info(f"Checking solution {self.format_solution(sol)}, cost {cost}, disagreement {self.calculate_disagreement(self.format_solution(sol))[0]}")
        for v, _, (prefix, i, j) in self.soft_constraint_penalties:
            if self.solver.Value(v) == 1:
                logging.info(f"[{prefix.upper()} VIOLATION] - {j} came before {i}")

        if sol in self.solutions: 
            logging.error(f"Solution {self.format_solution(sol)} already in solutions.")
            return

        if len(self.solutions) >= config.POPULATION_SIZE:
            worst_sol, worst_cost = max(self.solutions.items(), key=lambda x: x[1])
            logging.info(f"Comparing with worst solution and cost: {worst_sol}, {worst_cost}")
            if cost > worst_cost: 
                logging.error(f"Solution worse than worst solution.")
                return -1 # If worst cost, don't add
            else: 
                logging.info("Replacing worst solution with current solution")
                del self.solutions[worst_sol] 

        self.block_solution(sol)
        self.add_solution(sol, cost)
        logging.info(f"Added solution. Current solution size: {len(self.formatted_solutions)}")

    def solve(self):
        # Generate initial population of solutions
        for i in range(config.POPULATION_SIZE if self.num_elicitations == 0 else 2):
            logging.info(f"[GENERATE] - Generating solution {i + 1} of {config.POPULATION_SIZE if self.num_elicitations == 0 else 2} for initial population")
            if self.call_solver() == -1: 
                logging.warning(f"Solver has no more feasible or better solutions after intitial attempt {i + 1}")
                logging.info(f"[COMPLETE] - Final solutions: {self.formatted_solutions}")
                logging.info(f"[COMPLETE] - Final elicited pairs: {self.elicited_pairs}")
                logging.info(f"[COMPLETE] - Final error pairs: {self.error_pairs}")
                logging.info(f"[COMPLETE] - Final num error pairs: {len(self.error_pairs)}")
                return self.formatted_solutions

        # Elicit additional constraints and refine solutions
        while self.num_elicitations > len(self.elicited_pairs): # check if first call didnt reach num elicitations if this helps
            logging.info("[ELICIT] - Eliciting pairs")
            # Re-block all previous solutions before solving again
            for prev_sol in self.solutions.keys():
                self.block_solution(prev_sol)

            # Add elicitation constraints based on pairs and error rate
            self.add_elicitations()
            logging.info(f"[ELICIT] - Done eliciting: {len(self.elicited_pairs)}/{self.num_elicitations} elicited so far")

            num_sol_per_elicitations = 0
            # keep solving until status not improving or pop size regenerated
            
            while num_sol_per_elicitations < config.POPULATION_SIZE:
                res = self.call_solver()
                if res == -1: 
                    # below logs wont print if do this
                    logging.warning(f"Solver has no more feasible or better solutions at {num_sol_per_elicitations} elicitation attempt")
                    logging.info(f"[COMPLETE] - Final solutions: {self.formatted_solutions}")
                    logging.info(f"[COMPLETE] - Final elicited pairs: {self.elicited_pairs}")
                    logging.info(f"[COMPLETE] - Final error pairs: {self.error_pairs}")
                    logging.info(f"[COMPLETE] - Final num error pairs: {len(self.error_pairs)}")
                    return self.formatted_solutions
            
                num_sol_per_elicitations +=1 
                logging.info(f"Checking solution attempt {num_sol_per_elicitations}")
            
        logging.info(f"[COMPLETE] - Final solutions: {self.formatted_solutions}")
        logging.info(f"[COMPLETE] - Final elicited pairs: {self.elicited_pairs}")
        logging.info(f"[COMPLETE] - Final error pairs: {self.error_pairs}")
        logging.info(f"[COMPLETE] - Final num error pairs: {len(self.error_pairs)}")
        return self.formatted_solutions

    # all in one elicitations - elciitng based on prev solution not eliciting then rerunning then eliciting
    def add_elicitations(self):
        # Pick new pairs from existing solutions to elicit on
        solutions = list(self.formatted_solutions.keys())

        for sol1, sol2 in combinations(solutions, 2):
            for issue_a, issue_b in zip(sol1, sol2):
                if len(self.elicited_pairs) >= self.num_elicitations: 
                    logging.warning("Max elicitations reached.")
                    return
                if issue_a != issue_b:
                    idx_a = self.sprint.idx_dict[issue_a]
                    idx_b = self.sprint.idx_dict[issue_b]

                    if idx_a < idx_b:
                        pair = (issue_a, issue_b, idx_a, idx_b)
                    elif idx_b < idx_a:
                        pair = (issue_b, issue_a, idx_b, idx_a)

                    if (pair[0], pair[1]) in self.dep_constraints:
                        logging.info(f"[SKIP ELICIT] - Pair {pair} already in dependency constraints")
                        continue

                    if (pair[0], pair[1]) in self.elicited_pairs:
                        logging.warning(f"[ALREADY ELICITED] - Pair {(pair[0], pair[1])} already elicited.")
                        continue

                    self.elicited_pairs.add((pair[0], pair[1]))                        

                    # If haven't reached error rate then add error pairs constraint
                    if len(self.error_pairs) < self.max_error_pairs:
                        error_pair = (pair[1], pair[0], pair[3], pair[2])

                        self.error_pairs.add((error_pair[0], error_pair[1]))
                        self.model.Add(self.positions[error_pair[2]] < self.positions[error_pair[3]])
                        logging.info(f"[ADD ELICIT] - Adding error pair {error_pair}")
                    else: # Otherwise, add actual pair
                        self.model.Add(self.positions[pair[2]] < self.positions[pair[3]])
                        logging.info(f"[ADD ELICIT] - Adding normal pair {pair}")
