"""Head-to-head comparisons from outputs/seed_matrix_summary/per_run.csv.

For each pair (A, B) and metric: mean of A and B, seeds where A is better
(paired by seed number), whether every A run beats every B run, and the exact
Mann-Whitney U p-value (two-sided). Lower is better for every metric here.

Usage, from the repo root:

    .venv/Scripts/python.exe scripts/compare_seed_runs.py outputs/seed_matrix_summary/per_run.csv [outputs/seed_matrix_summary/per_run_3_seeds.csv]

The optional second file adds a "3-seed vs 5-seed means" table.
"""
import sys

import pandas as pd
from scipy.stats import mannwhitneyu

METRICS = [("carbon_kgco2", "carbon kg", 1), ("average_discomfort", "avg %", 100),
           ("worst_discomfort", "worst %", 100), ("discomfort_gap", "gap pts", 100)]
PAIRS = [
    ("marlisa", "independent_sac_training_encoders", "MARLISA vs SAC (team rule)"),
    ("marlisa", "independent_sac_training_encoders_final_episode", "MARLISA vs SAC (final episode)"),
    ("marlisa", "fair_carbon_aware_marlisa", "MARLISA vs Fair (both terms)"),
    ("fair_carbon_aware_marlisa", "fair_no_fairness", "Fair (both terms) vs no fairness"),
    ("fair_worst_case", "fair_no_fairness", "Worst-only vs no fairness"),
    ("fair_disparity", "fair_no_fairness", "Gap-only vs no fairness"),
    ("fair_worst_case", "fair_carbon_aware_marlisa", "Worst-only vs both terms"),
    ("fair_worst_case", "fair_disparity", "Worst-only vs gap-only"),
    ("fair_carbon_aware_marlisa", "fair_disparity", "Both terms vs gap-only"),
    ("marlisa", "fair_worst_case", "MARLISA vs worst-only"),
]

runs = pd.read_csv(sys.argv[1])
seeds = sorted(runs.loc[runs["config"] == "marlisa", "seed"])
print(f"seeds: {seeds}\n")

for a, b, title in PAIRS:
    print(f"== {title}")
    for metric, name, scale in METRICS:
        x = runs[runs["config"] == a].set_index("seed")[metric] * scale
        y = runs[runs["config"] == b].set_index("seed")[metric] * scale
        common = sorted(set(x.index) & set(y.index))
        wins = sum(x[s] < y[s] for s in common)
        separated = x.max() < y.min()
        reverse = y.max() < x.min()
        p = mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue
        flag = "A beats B in every run" if separated else ("B beats A in every run" if reverse else "")
        print(f"   {name:10s} A {x.mean():6.1f} (sd {x.std():4.1f})  B {y.mean():6.1f} (sd {y.std():4.1f})  "
              f"diff {x.mean() - y.mean():6.1f}  A better in {wins}/{len(common)} seeds  p={p:.3f}  {flag}")
    print()

if len(sys.argv) > 2:
    old = pd.read_csv(sys.argv[2])
    print("== 3-seed vs 5-seed means")
    for config in runs["config"].unique():
        o, n = old[old["config"] == config], runs[runs["config"] == config]
        if o.empty:
            continue
        cells = []
        for metric, name, scale in METRICS:
            cells.append(f"{name} {o[metric].mean() * scale:6.1f} -> {n[metric].mean() * scale:6.1f}")
        print(f"   {config:48s} " + " | ".join(cells))
