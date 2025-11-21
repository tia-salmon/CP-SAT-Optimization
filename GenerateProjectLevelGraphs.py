
import json
import os
import config
import statistics
import numpy as np
import matplotlib.pyplot as plt
import re

from scipy.stats import shapiro, f_oneway, kruskal, mannwhitneyu
from collections import defaultdict
from Sprint import Sprint
from itertools import combinations

"""
- Could update to not repeat list of keys each time (eli, eli_non_rollover, etc)
- Can be organized better and made more efficient

"""
res_folder = os.path.join("rollover",  "results")
before_res_folder = os.path.join("non_rollover",  "results")

# Get all files needed::
files = [f for f in os.listdir(res_folder) if f.endswith(".json")] # 260 files before rollover (conference); 
before_rollover_files = [f for f in os.listdir(before_res_folder) if f.endswith(".json")]

# Create dict to store files::
grouped = defaultdict(lambda: defaultdict(list))

# Example match: proj12_sprint1_c1p1_e50_err0
pattern = re.compile(r'(proj\d+)_sprint\d+_([a-z0-9]+)?_?e(\d+)_err(\d+)', re.IGNORECASE)

# Update dict::
for filename in files:
    match = pattern.search(filename)

    base = match.group(1)  # proj12
    condition = match.group(2) or "b"
    e_val = int(match.group(3))
    err_val = int(match.group(4))

    if condition == "b":
        grouped[base]["err"].append(filename)

    if condition == "c1p1" and err_val == 0 and e_val in config.VARYING_ELI:
        grouped[base]["eli"].append(filename)

    if condition in (config.VARYING_CONST + config.VARYING_WEIGHT) and e_val == 50 and err_val == 0:
        grouped[base]["const"].append(filename)

    if condition == "c1p1" and e_val == 50 and err_val in config.VARYING_ERROR_RATE:
        grouped[base]["err"].append(filename)
    
    if condition in (config.VARYING_CONST + config.VARYING_LOCAL_CONST) and e_val == 50 and err_val == 0:
        grouped[base]["l_const"].append(filename)

# Update dict with eli files before rollover (from conference)::
for b_filename in before_rollover_files:
    b_match = pattern.search(b_filename)
    if not b_match: continue

    b_base = b_match.group(1)  # proj12
    b_condition = b_match.group(2) or "b"
    b_e_val = int(b_match.group(3))
    b_err_val = int(b_match.group(4))
    
    if b_condition == "c1p1" and b_err_val == 0 and b_e_val in config.VARYING_ELI:
        grouped[b_base]["eli_non_rollover"].append(b_filename)

def eli_sort_key(filename):
    match = re.search(r'_e(\d+)_err0', filename)
    if match:
        return config.VARYING_ELI.index(int(match.group(1)))

def const_sort_key(filename):
    match = re.search(r'_([a-z0-9]+)_e50_err0', filename)
    if match:
        constraint = match.group(1)
        
        if constraint == "c1": # order by c1 first
            return (1, constraint)

        has_2 = "2" in constraint
        parts = re.findall(r'[a-z]+[0-9]*', constraint)
        num_parts = len(parts)

        if not has_2:
            if num_parts == 2:
                return (2, constraint)
            elif num_parts == 3:
                return (3, constraint)
        else:
            return (4, constraint)
    return (99, filename)  # unknown format goes last

def err_sort_key(filename):
    if "_b_" in filename:
        return -1  # b always first
    match = re.search(r'_e50_err(\d+)', filename)
    if match:
        return config.VARYING_ERROR_RATE.index(int(match.group(1)))

def l_const_sort_key(filename):
    pairs = {
        "c1p1": "clpl",
        "c1t1": "cltl",
        "c1p1t1": "clpltl",
    }

    match = re.search(r'_([a-z0-9]+)_e50_err0', filename)
    if not match:
        return (99, filename)  # unknown format goes last

    constraint = match.group(1)

    # If this constraint is one half of a pair, normalize it
    if constraint in pairs:
        group = pairs[constraint]
        return (0, group, constraint)
    elif constraint in pairs.values():
        group = constraint
        return (0, group, constraint)

    if constraint == "c1":
        return (1, constraint)

    has_2 = "2" in constraint
    parts = re.findall(r'[a-z]+[0-9]*', constraint)
    num_parts = len(parts)

    if not has_2:
        if num_parts == 2:
            return (2, constraint)
        elif num_parts == 3:
            return (3, constraint)
    else:
        return (4, constraint)

    return (98, constraint)  # fallback

# Order results in logical order::
for base in grouped:
    grouped[base]["eli"].sort(key=eli_sort_key)
    grouped[base]["eli_non_rollover"].sort(key=eli_sort_key)
    grouped[base]["const"].sort(key=const_sort_key)
    grouped[base]["err"].sort(key=err_sort_key)
    grouped[base]["l_const"].sort(key=l_const_sort_key)

# Get values, labels from each group::
project_data = defaultdict(lambda: {
    "eli": {"median_disagreements": [], "median_average_distances": [], "labels": []},
    "const": {"median_disagreements": [], "median_average_distances": [], "labels": []},
    "err": {"median_disagreements": [], "median_average_distances": [], "labels": []},
    "l_const": {"median_disagreements": [], "median_average_distances": [], "labels": []},
    "eli_non_rollover": {"median_disagreements": [], "median_average_distances": [], "labels": []},
    "aco": {"median_disagreements": [], "median_average_distances": [], "labels": []},
    "mopso": {"median_disagreements": [], "median_average_distances": [], "labels": []}
})



for proj, group in grouped.items():
    for group_type in ["eli", "const", "err", "l_const", "eli_non_rollover"]:
        if group_type == "eli_non_rollover":
            base_folder = before_res_folder 
        else:
            base_folder = res_folder

        for filename in group[group_type]:
            filepath = os.path.join(base_folder, filename)
            with open(filepath) as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Error loading json file: {filepath}")

                disagreements = data.get("median_disagreements")
                avg_distance = data.get("median_average_distances")

                # figure out the label
                label = "?"
                if "_b_" in filename:
                    label = "b"
                else:
                    # Extract constraint, e and err from filename
                    match = re.search(r'_([a-z0-9]+)_e(\d+)_err(\d+)', filename)
                    if match:
                        constraint = match.group(1)
                        e_val = int(match.group(2))
                        err_val = int(match.group(3))

                        if group_type == "eli" or group_type == "eli_non_rollover":
                            label = f"eli_{e_val}"
                        elif group_type == "const" or group_type == "l_const":
                            label = f"{constraint}"
                        elif group_type == "err":
                            label = f"err_{err_val}"

                # Store values and label
                if disagreements is not None:
                    project_data[proj][group_type]["median_disagreements"].append(disagreements)
                else:
                    print(f"missing disagreements for {filename}")
                if avg_distance is not None:
                    project_data[proj][group_type]["median_average_distances"].append(avg_distance)
                else:
                    print(f"missing disagreements for {filename}")

                if label not in project_data[proj][group_type]["labels"]:
                    project_data[proj][group_type]["labels"].append(label)

"""Could be updated to import from Solver (or redefined in another class/file)"""
def calculate_disagreement(sprint_id, proj_id, other_order):
    loc = os.path.join(os.getcwd(), "input", f"P{proj_id}")
    sprint = Sprint(sprint_id, loc)
    gold_standard_index = sprint.idx_dict
    total_disagreements = 0
    NUMREQ = len(other_order)
    for i in range(NUMREQ):
        for j in range(i + 1, NUMREQ):
            issue1 = other_order[i]; issue2 = other_order[j]
            if (gold_standard_index[issue1] > gold_standard_index[issue2]) != (i > j):
                total_disagreements += 1
    return total_disagreements

def calculate_avdist(sprint_id, proj_id, req_order):
    loc = os.path.join(os.getcwd(), "input", f"P{proj_id}")
    sprint = Sprint(sprint_id, loc)
    gold = sprint.idx_dict
    return sum(abs(i - gold[item]) for i, item in enumerate(req_order)) / len(req_order)

with open("mopso_results.json") as f:
    mopso_data = json.load(f)

with open("aco_results.json") as f:
    aco_data = json.load(f)

for entry in mopso_data:
    proj = str(entry['project_id'])
    spr = entry['sprint_id']
    project_key = f"proj{proj}"
    
    for run in entry['runs']:
        sol = run['solution']
        project_data[project_key]["mopso"]['median_disagreements'].append(calculate_disagreement(spr, int(proj), sol))
        project_data[project_key]["mopso"]['median_average_distances'].append(calculate_avdist(spr, int(proj), sol))

    if "MOPSO" not in project_data[project_key]["mopso"]["labels"]:
        project_data[project_key]["mopso"]["labels"].append("MOPSO")
    
for entry in aco_data:
    proj = str(entry['project_id'])
    spr = entry['sprint_id']
    project_key = f"proj{proj}"

    for run in entry['runs']:
        sol = run['solution']
        project_data[project_key]["aco"]['median_disagreements'].append(calculate_disagreement(spr, int(proj), sol))
        project_data[project_key]["aco"]['median_average_distances'].append(calculate_avdist(spr, int(proj), sol))

    if "ACO" not in project_data[project_key]["aco"]["labels"]:
        project_data[project_key]["aco"]["labels"].append("ACO")

with open("project_data.json", "w") as f:
    json.dump(project_data, f, indent=2)
with open("grouped_data.json", "w") as f:
    json.dump(grouped, f, indent=2)

def holm_correction(pvals):
    """Apply Holm correction to a list of p-values."""
    pvals = np.array(pvals)
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    corrected = np.empty(n)
    for i in range(n):
        corrected[i] = min((n - i) * sorted_pvals[i], 1.0)
    unsorted = np.empty(n)
    unsorted[sorted_idx] = corrected
    return unsorted.tolist()

def plot_graphs():
    os.makedirs("plots/png", exist_ok=True)
    os.makedirs("plots/pdf", exist_ok=True)
    os.makedirs("analysis/md", exist_ok=True)
    os.makedirs("analysis/stats", exist_ok=True)
    os.makedirs("plots/carry_mopso_aco_comparison", exist_ok=True)
    os.makedirs("plots/carry_mopso_aco_comparison/png", exist_ok=True)
    os.makedirs("plots/carry_mopso_aco_comparison/pdf", exist_ok=True)
    os.makedirs("plots/comparison_subplots", exist_ok=True)
    os.makedirs("plots/comparison_subplots/png", exist_ok=True)
    os.makedirs("plots/comparison_subplots/pdf", exist_ok=True)

    group_types_labels = {
        "eli": "Number of Elicitations",
        "const": "Constraints",
        "l_const": "Constraints",
        "err": "Error Rates (%)"
    }

    group_types_titles = {
        "eli": "Elicitations",
        "const": "Constraints",
        "l_const": "Local and Global Constraints",
        "err": "Error Rates"
    }

    for project, groups in project_data.items():
        for group_type in ["eli", "const", "err", "l_const", "eli_non_rollover", "mopso", "aco"]:
            for metric in ["median_disagreements", "median_average_distances"]:
                group = groups[group_type]
                if group_type not in ["aco", "mopso"]:
                    values = [[round(v, 2) for v in sublist] for sublist in group[metric]]
                else:
                    values = [[round(v, 2)] for v in group[metric]]
                labels = group["labels"]
                mapping = dict(zip(grouped[project][group_type], values))

                md_file = os.path.join("analysis/md", f"proj_{project}_{group_type}_{metric}.md")
                stats_file = os.path.join("analysis/stats", f"proj_{project}_{group_type}_{metric}_stats.txt")

                with open(md_file, "w") as f_md, open(stats_file, "w") as f_stat:
                    f_md.write(f"Project {project} - {group_types_titles.get(group_type, group_type)} - {metric}\n\n")
                    f_stat.write(f"Project {project} - {group_types_titles.get(group_type, group_type)} - {metric}\n\n")

                    # ====================== eli/const/err/l_const ======================
                    if group_type in ["eli", "const", "err", "l_const"]:
                        n_cols, n_rows = 3, 2
                        subplot_height = 4
                        subplot_width = max(5, len(labels) * 0.6)
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * subplot_width, n_rows * subplot_height))
                        axes = axes.flatten()

                        for sprint_idx in range(5):
                            sprint_vals = []
                            f_md.write(f"Sprint {sprint_idx + 1}\n\n")
                            f_stat.write(f"Sprint {sprint_idx + 1}\n\n")
                            for group_idx, label in enumerate(labels):
                                file_name = grouped[project][group_type][sprint_idx + group_idx * 5]
                                val = mapping[file_name]
                                sprint_vals.append(val)
                                arr = np.array(val)
                                f_md.write(f"{label}: n={len(arr)}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, median={np.median(arr):.2f}\n")

                            all_groups = [np.array(v) for v in sprint_vals]
                            if sprint_idx == 0:

                                # Collect project-wide values for each label
                                project_groups = {}
                                for gidx, label in enumerate(labels):
                                    combined = []
                                    for s_idx in range(5):
                                        fname = grouped[project][group_type][s_idx + gidx * 5]
                                        combined.extend(mapping[fname])
                                    project_groups[label] = np.array(combined)

                                # Write summary to markdown
                                f_md.write("\nPROJECT-WIDE SUMMARY (all sprints combined)\n\n")
                                for label, arr in project_groups.items():
                                    f_md.write(f"{label}: n={len(arr)}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, median={np.median(arr):.2f}\n")

                                # Prepare for stats
                                all_vals = list(project_groups.values())
                                all_labels = list(project_groups.keys())

                                normality = [shapiro(v)[1] > 0.05 for v in all_vals]

                                f_stat.write("\nPROJECT-WIDE STATS (all sprints combined)\n\n")

                                # ANOVA if all normal
                                if all(normality):
                                    f_stat.write("All groups normal - ANOVA\n")
                                    fval, pval = f_oneway(*all_vals)
                                    f_stat.write(f"F={fval:.4f}, p={pval:.4f}\n")

                                else:
                                    f_stat.write("Not all groups normal - Kruskal-Wallis\n")
                                    hval, pval = kruskal(*all_vals)
                                    f_stat.write(f"H={hval:.4f}, p={pval:.4f}\n")

                                    # Pairwise MW tests (Holm corrected)
                                    f_stat.write("Pairwise Mann-Whitney U (Holm corrected)\n")

                                    pvals = []
                                    pairs = []
                                    for i, j in combinations(range(len(all_vals)), 2):
                                        u, p = mannwhitneyu(all_vals[i], all_vals[j], alternative='two-sided')
                                        pvals.append(p)
                                        pairs.append((all_labels[i], all_labels[j]))

                                    corrected = holm_correction(pvals)

                                    for idx, (l1, l2) in enumerate(pairs):
                                        f_stat.write(f"{l1} vs {l2}: corrected p={corrected[idx]:.4f}\n")

                                f_stat.write("\n")

                            ax = axes[sprint_idx]
                            ax.boxplot(sprint_vals, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))
                            ax.set_title(f"Sprint {sprint_idx + 1}", fontsize=12)
                            ax.set_xlabel(group_types_labels[group_type], fontsize=12)
                            ax.tick_params(axis='x', rotation=45, labelsize=12)
                            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)

                        axes[5].set_visible(False)
                        plt.tight_layout(rect=[0, 0, 1, 0.97])
                        fig.set_size_inches(16, 8)
                        fig.subplots_adjust(left=0.06, right=0.96, wspace=0.35, hspace=0.45)
                        second_row_axes = [axes[3], axes[4]]
                        fig_center = 0.5
                        centers = [ax.get_position().x0 + ax.get_position().width / 2 for ax in second_row_axes]
                        avg_center = sum(centers) / len(centers)
                        shift = fig_center - avg_center
                        for ax in second_row_axes:
                            pos = ax.get_position()
                            ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
                        fig.suptitle(f"Project {project.replace('proj', '')} - Varying {group_types_titles[group_type]}", fontsize=14, y=1.03)
                        fig.savefig(os.path.join("plots", "png", f"{project.replace('proj', 'proj_')}_{group_type}_{metric}_all_sprints.png"), bbox_inches='tight', pad_inches=0.5)
                        fig.savefig(os.path.join("plots", "pdf", f"{project.replace('proj', 'proj_')}_{group_type}_{metric}_all_sprints.pdf"), bbox_inches='tight', pad_inches=0.5)
                        plt.close(fig)

                    # ====================== eli_non_rollover ======================
                    elif group_type == "eli_non_rollover":
                        rollover_group = groups["eli"]
                        non_group = groups["eli_non_rollover"]
                        metric_title = metric.replace("_", " ").title()

                        rollover_values = [[round(v, 2) for v in sublist] for sublist in rollover_group[metric]]
                        non_values = [[round(v, 2) for v in sublist] for sublist in non_group[metric]]
                        rollover_labels = rollover_group["labels"]
                        non_labels = non_group["labels"]
                        rollover_mapping = dict(zip(grouped[project]["eli"], rollover_values))
                        non_mapping = dict(zip(grouped[project]["eli_non_rollover"], non_values))

                        # --- Plot Rollover (Carry Over) ---
                        n_cols, n_rows = 3, 2
                        subplot_height = 4
                        subplot_width = max(5, len(rollover_labels) * 0.6)
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * subplot_width, n_rows * subplot_height))
                        axes = axes.flatten()

                        for sprint_idx in range(5):
                            sprint_vals = []
                            f_md.write(f"Sprint {sprint_idx + 1} (Carry Over)\n\n")
                            f_stat.write(f"Sprint {sprint_idx + 1} (Carry Over)\n\n")
                            for group_idx, label in enumerate(rollover_labels):
                                file_name = grouped[project]["eli"][sprint_idx + group_idx * 5]
                                val = rollover_mapping[file_name]
                                sprint_vals.append(val)
                                arr = np.array(val)
                                f_md.write(f"{label}: n={len(arr)}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, median={np.median(arr):.2f}\n")

                            all_groups = [np.array(v) for v in sprint_vals]
                            if sprint_idx == 0:
                                project_groups = {}
                                for gidx, label in enumerate(rollover_labels):
                                    combined = []
                                    for s_idx in range(5):
                                        fname = grouped[project]["eli"][s_idx + gidx * 5]
                                        combined.extend(rollover_mapping[fname])
                                    project_groups[label] = np.array(combined)

                                f_md.write("\nPROJECT-WIDE SUMMARY (Carry Over, all sprints combined)\n\n")
                                for label, arr in project_groups.items():
                                    f_md.write(f"{label}: n={len(arr)}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, median={np.median(arr):.2f}\n")

                                all_vals = list(project_groups.values())
                                all_labels = list(project_groups.keys())
                                normality = [shapiro(v)[1] > 0.05 for v in all_vals]

                                f_stat.write("\nPROJECT-WIDE STATS (Carry Over)\n\n")
                                if all(normality):
                                    f_stat.write("All groups normal - ANOVA\n")
                                    fval, pval = f_oneway(*all_vals)
                                    f_stat.write(f"F={fval:.4f}, p={pval:.4f}\n")
                                else:
                                    f_stat.write("Not all groups normal - Kruskal-Wallis\n")
                                    hval, pval = kruskal(*all_vals)
                                    f_stat.write(f"H={hval:.4f}, p={pval:.4f}\n")
                                    f_stat.write("Pairwise Mann-Whitney U (Holm corrected)\n")
                                    pvals = []
                                    pairs = []
                                    for i, j in combinations(range(len(all_vals)), 2):
                                        u, p = mannwhitneyu(all_vals[i], all_vals[j], alternative='two-sided')
                                        pvals.append(p)
                                        pairs.append((all_labels[i], all_labels[j]))
                                    corrected = holm_correction(pvals)
                                    for idx, (l1, l2) in enumerate(pairs):
                                        f_stat.write(f"{l1} vs {l2}: corrected p={corrected[idx]:.4f}\n")
                                f_stat.write("\n")

                            ax = axes[sprint_idx]
                            ax.boxplot(sprint_vals, tick_labels=rollover_labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))
                            ax.set_title(f"Sprint {sprint_idx + 1}", fontsize=12)
                            ax.set_xlabel("Elicitations", fontsize=12)
                            ax.tick_params(axis='x', rotation=45, labelsize=12)
                            ax.set_ylabel(metric_title, fontsize=12)

                        axes[5].set_visible(False)
                        plt.tight_layout(rect=[0, 0, 1, 0.97])
                        fig.set_size_inches(16, 8)
                        fig.subplots_adjust(left=0.06, right=0.96, wspace=0.35, hspace=0.45)
                        second_row_axes = [axes[3], axes[4]]
                        fig_center = 0.5
                        centers = [ax.get_position().x0 + ax.get_position().width / 2 for ax in second_row_axes]
                        avg_center = sum(centers) / len(centers)
                        shift = fig_center - avg_center
                        for ax in second_row_axes:
                            pos = ax.get_position()
                            ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
                        fig.suptitle(f"Project {project.replace('proj','')} - Varying Elicitations (Carry-Over)", fontsize=14, y=1.03)
                        fig.savefig(os.path.join("plots", "png", f"{project.replace('proj','proj_')}_eli_rollover_{metric}_all_sprints.png"), bbox_inches='tight', pad_inches=0.5)
                        fig.savefig(os.path.join("plots", "pdf", f"{project.replace('proj','proj_')}_eli_rollover_{metric}_all_sprints.pdf"), bbox_inches='tight', pad_inches=0.5)
                        plt.close(fig)

                        n_cols, n_rows = 3, 2
                        subplot_height = 4
                        subplot_width = max(5, len(non_labels) * 0.6)
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * subplot_width, n_rows * subplot_height))
                        axes = axes.flatten()

                        for sprint_idx in range(5):
                            sprint_vals = []
                            f_md.write(f"Sprint {sprint_idx + 1} (No Carry Over)\n\n")
                            f_stat.write(f"Sprint {sprint_idx + 1} (No Carry Over)\n\n")
                            for group_idx, label in enumerate(non_labels):
                                file_name = grouped[project]["eli_non_rollover"][sprint_idx + group_idx * 5]
                                val = non_mapping[file_name]
                                sprint_vals.append(val)
                                arr = np.array(val)
                                f_md.write(f"{label}: n={len(arr)}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, median={np.median(arr):.2f}\n")

                            all_groups = [np.array(v) for v in sprint_vals]
                            if sprint_idx == 0:
                                project_groups = {}
                                for gidx, label in enumerate(non_labels):
                                    combined = []
                                    for s_idx in range(5):
                                        fname = grouped[project]["eli_non_rollover"][s_idx + gidx * 5]
                                        combined.extend(non_mapping[fname])
                                    project_groups[label] = np.array(combined)

                                f_md.write("\nPROJECT-WIDE SUMMARY (No Carry Over, all sprints combined)\n\n")
                                for label, arr in project_groups.items():
                                    f_md.write(f"{label}: n={len(arr)}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, median={np.median(arr):.2f}\n")

                                all_vals = list(project_groups.values())
                                all_labels = list(project_groups.keys())
                                normality = [shapiro(v)[1] > 0.05 for v in all_vals]

                                f_stat.write("\nPROJECT-WIDE STATS (No Carry Over)\n\n")
                                if all(normality):
                                    f_stat.write("All groups normal - ANOVA\n")
                                    fval, pval = f_oneway(*all_vals)
                                    f_stat.write(f"F={fval:.4f}, p={pval:.4f}\n")
                                else:
                                    f_stat.write("Not all groups normal - Kruskal-Wallis\n")
                                    hval, pval = kruskal(*all_vals)
                                    f_stat.write(f"H={hval:.4f}, p={pval:.4f}\n")
                                    f_stat.write("Pairwise Mann-Whitney U (Holm corrected)\n")
                                    pvals = []
                                    pairs = []
                                    for i, j in combinations(range(len(all_vals)), 2):
                                        u, p = mannwhitneyu(all_vals[i], all_vals[j], alternative='two-sided')
                                        pvals.append(p)
                                        pairs.append((all_labels[i], all_labels[j]))
                                    corrected = holm_correction(pvals)
                                    for idx, (l1, l2) in enumerate(pairs):
                                        f_stat.write(f"{l1} vs {l2}: corrected p={corrected[idx]:.4f}\n")
                                f_stat.write("\n")

                            ax = axes[sprint_idx]
                            ax.boxplot(sprint_vals, tick_labels=non_labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))
                            ax.set_title(f"Sprint {sprint_idx + 1}", fontsize=12)
                            ax.set_xlabel("Number of Elicitations", fontsize=12)
                            ax.tick_params(axis='x', rotation=45, labelsize=12)
                            ax.set_ylabel(metric_title, fontsize=12)

                        axes[5].set_visible(False)
                        plt.tight_layout(rect=[0, 0, 1, 0.97])
                        fig.set_size_inches(16, 8)
                        fig.subplots_adjust(left=0.06, right=0.96, wspace=0.35, hspace=0.45)
                        second_row_axes = [axes[3], axes[4]]
                        fig_center = 0.5
                        centers = [ax.get_position().x0 + ax.get_position().width / 2 for ax in second_row_axes]
                        avg_center = sum(centers) / len(centers)
                        shift = fig_center - avg_center
                        for ax in second_row_axes:
                            pos = ax.get_position()
                            ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
                        fig.suptitle(f"Project {project.replace('proj','')} - Varying Elicitations (Non-Carry-Over)", fontsize=14, y=1.03)
                        fig.savefig(os.path.join("plots", "png", f"{project.replace('proj','proj_')}_eli_non_rollover_{metric}_all_sprints.png"), bbox_inches='tight', pad_inches=0.5)
                        fig.savefig(os.path.join("plots", "pdf", f"{project.replace('proj','proj_')}_eli_non_rollover_{metric}_all_sprints.pdf"), bbox_inches='tight', pad_inches=0.5)
                        plt.close(fig)

                    # ====================== mopso ======================
                    elif group_type == "mopso":
                        eli_group = groups["eli"]
                        mopso_group = groups["mopso"]
                        aco_group = groups["aco"]
                        eli_values = [[round(v, 2) for v in sublist] for sublist in eli_group[metric]]
                        mopso_values = mopso_group[metric]
                        aco_values = aco_group[metric]
                        eli_labels = eli_group["labels"]
                        eli_mapping = dict(zip(grouped[project]["eli"], eli_values))
                        target_eli_values = [0, 25, 50, 100]
                        filtered_eli_labels = [label for label in eli_labels if int(label.split('_')[1]) in target_eli_values]

                        fig, ax = plt.subplots(1, 1, figsize=(max(10, (len(filtered_eli_labels)+2)*1.5),6))
                        all_vals = []
                        combined_labels = []
                        f_md.write("CP-SAT vs MOPSO vs ACO\n\n")
                        f_stat.write("CP-SAT vs MOPSO vs ACO\n\n")
                        for label in filtered_eli_labels:
                            original_idx = eli_labels.index(label)
                            label_vals = []
                            for sprint_idx in range(5):
                                file_name = grouped[project]["eli"][sprint_idx + original_idx*5]
                                label_vals.extend(eli_mapping[file_name])
                            all_vals.append(label_vals)
                            combined_labels.append(label)
                            arr = np.array(label_vals)
                            f_md.write(f"{label} CP-SAT: n={len(arr)}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, median={np.median(arr):.2f}\n")
                        # MOPSO
                        all_vals.append(mopso_values)
                        combined_labels.append("MOPSO")
                        arr = np.array(mopso_values)
                        f_md.write(f"MOPSO: n={len(arr)}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, median={np.median(arr):.2f}\n")
                        # ACO
                        all_vals.append(aco_values)
                        combined_labels.append("ACO")
                        arr = np.array(aco_values)
                        f_md.write(f"ACO: n={len(arr)}, min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}, median={np.median(arr):.2f}\n")

                        normality = [shapiro(g)[1] > 0.05 for g in all_vals]
                        if all(normality):
                            f_stat.write("All groups normal - ANOVA\n")
                            f_val, p_val = f_oneway(*all_vals)
                            f_stat.write(f"F={f_val:.4f}, p={p_val:.4f}\n")
                        else:
                            f_stat.write("Not all groups normal - Kruskal-Wallis\n")
                            h_val, p_val = kruskal(*all_vals)
                            f_stat.write(f"H={h_val:.4f}, p={p_val:.4f}\n")
                            f_stat.write("Pairwise Mann-Whitney U with Holm correction:\n")
                            pvals = []
                            pairs = []
                            for i, j in combinations(range(len(all_vals)), 2):
                                u_stat, p_u = mannwhitneyu(all_vals[i], all_vals[j])
                                pvals.append(p_u)
                                pairs.append((combined_labels[i], combined_labels[j]))
                            corrected = holm_correction(pvals)
                            for k, (l1, l2) in enumerate(pairs):
                                f_stat.write(f"{l1} vs {l2}: U={u_stat:.4f}, corrected p={corrected[k]:.4f}\n")

                        bp = ax.boxplot(all_vals, tick_labels=combined_labels, patch_artist=True)
                        for patch_idx, patch in enumerate(bp['boxes']):
                            patch.set_facecolor("lightblue")
                        ax.set_title(f"Project {project.replace('proj','')} - All Sprints", fontsize=12)
                        ax.set_xlabel("Optimization Method", fontsize=12)
                        ax.tick_params(axis='x', rotation=45, labelsize=12)
                        ax.set_ylabel(metric.replace("_"," ").title(), fontsize=12)
                        plt.tight_layout(rect=[0,0,1,0.95])
                        fig.suptitle(f"Project {project.replace('proj','')} - CP-SAT vs MOPSO vs ACO", fontsize=14, y=0.98)
                        fig.savefig(os.path.join("plots", "carry_mopso_aco_comparison", "png", f'{project.replace("proj", "proj_")}_{metric}_comparison.png'),
                                    bbox_inches='tight', pad_inches=0.5)
                        fig.savefig(os.path.join("plots", "carry_mopso_aco_comparison", "pdf", f'{project.replace("proj", "proj_")}_{metric}_comparison.pdf'),
                                    bbox_inches='tight', pad_inches=0.5)
                        plt.close(fig)

if __name__ == "__main__":
    plot_graphs()