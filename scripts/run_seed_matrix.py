"""Run the multi-seed experiment matrix in parallel, one process per run.

Every run uses the settings of ``scripts/run_06_plan.py`` (the 3 Nov seed-42
runs): 50 training episodes on June 1-20, held-out test on June 26-30, pipeline
fixes off. Only the seed, the output folder and the named change differ.

    .venv\\Scripts\\python.exe scripts\\run_seed_matrix.py --list
    .venv\\Scripts\\python.exe scripts\\run_seed_matrix.py --launch --workers 18
    .venv\\Scripts\\python.exe scripts\\run_seed_matrix.py --summarize

A run is skipped when its seed folder already holds ``test_kpis.csv``, so
committed results such as ``outputs/marlisa/seed_42`` are never overwritten.
``--smoke`` runs a few short episodes into ``--output-root`` to check wiring.
"""

from __future__ import annotations

import os

# One thread per process: many runs share the CPU.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
EPISODES = 50
DEFAULT_SEED_COUNT = 3

FAIR_DEFAULTS = {"carbon_weight": 1.0, "comfort_weight": 0.25, "fairness_weight": 0.25, "comfort_band": 1.0}

#: name -> (label, output folder, runner, extra keyword arguments)
CONFIGS: dict[str, tuple[str, str, str, dict]] = {
    "marlisa": ("MARLISA", "marlisa", "marlisa", {"reward": "electricity_discomfort"}),
    "carbon_aware_marlisa": ("Carbon-aware MARLISA", "carbon_aware_marlisa", "carbon_aware", {}),
    "fair_carbon_aware_marlisa": (
        "Fair carbon-aware MARLISA", "fair_carbon_aware_marlisa", "fair", dict(FAIR_DEFAULTS),
    ),
    "independent_sac_training_encoders": (
        "Independent SAC (training-window input scaling)",
        "independent_sac_training_encoders",
        "sac",
        {"use_training_encoders": True},
    ),
    "fair_no_fairness": (
        "Fair carbon-aware MARLISA, fairness weight 0",
        "fair_carbon_aware_marlisa_no_fairness",
        "fair",
        {**FAIR_DEFAULTS, "fairness_weight": 0.0},
    ),
    "fair_worst_case": (
        "Fair carbon-aware MARLISA, worst-building term only",
        "fair_carbon_aware_marlisa_worst_case",
        "fair",
        {**FAIR_DEFAULTS, "fairness_mode": "worst_case"},
    ),
    "fair_disparity": (
        "Fair carbon-aware MARLISA, gap term only",
        "fair_carbon_aware_marlisa_disparity",
        "fair",
        {**FAIR_DEFAULTS, "fairness_mode": "disparity"},
    ),
}

#: Existing runs shown alongside the matrix: name -> (label, test KPI file).
REFERENCES = {
    "rbc": ("RBC", OUTPUTS_ROOT / "rbc" / "seed_42" / "test_kpis.csv"),
    "independent_sac_16oct": (
        "Independent SAC (16 Oct run, test-window input scaling)",
        OUTPUTS_ROOT / "independent_sac" / "seed_42" / "test_kpis.csv",
    ),
}


def default_seeds() -> list[int]:
    from src.dataset_splits import DEFAULT_SEEDS

    return list(DEFAULT_SEEDS[:DEFAULT_SEED_COUNT])


def seed_dir(output_root: Path, name: str, seed: int) -> Path:
    return output_root / CONFIGS[name][1] / f"seed_{seed}"


def run_job(name: str, seed: int, output_root: Path, smoke: bool) -> None:
    """Train and evaluate one configuration for one seed in this process."""
    import torch

    torch.set_num_threads(1)
    label, folder, runner, extra = CONFIGS[name]
    common = {"schema": SCHEMA, "output_directory": output_root / folder, "random_seed": seed}
    print(f"[{datetime.now():%H:%M:%S}] {label} | seed {seed} | smoke={smoke}", flush=True)

    if runner == "sac":
        from src.run_independent_sac import evaluate_independent_sac, run_independent_sac

        episodes = 1 if smoke else EPISODES
        result = run_independent_sac(**common, episodes=episodes, **extra)
        # MARLISA is graded on its final episode, so also grade SAC's final
        # episode (the runner itself grades the validation-selected one).
        final_dir = Path(result["output_directory"])
        evaluate_independent_sac(
            final_dir / "checkpoints" / f"episode_{episodes:03d}.pt",
            schema=SCHEMA,
            split="test",
            output_directory=final_dir / "test_final_episode",
            save_results=True,
        )
        return

    from src.run_carbon_aware_marlisa import run_carbon_aware_marlisa
    from src.run_fair_carbon_aware_marlisa import run_fair_carbon_aware_marlisa
    from src.run_marlisa import run_marlisa

    function = {"marlisa": run_marlisa, "carbon_aware": run_carbon_aware_marlisa, "fair": run_fair_carbon_aware_marlisa}[runner]
    length = {"episodes": 3, "simulation_end_time_step": 47} if smoke else {"episodes": EPISODES}
    function(**common, **length, **extra)


def launch(seeds: list[int], names: list[str], output_root: Path, workers: int, smoke: bool) -> None:
    log_dir = output_root / "seed_matrix_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / "run_log.csv"
    lock = threading.Lock()

    jobs = []
    for name in names:
        for seed in seeds:
            if (seed_dir(output_root, name, seed) / "test_kpis.csv").is_file():
                print(f"skip {name} seed {seed}: results already exist", flush=True)
                continue
            jobs.append((name, seed))
    print(f"{len(jobs)} runs, up to {workers} at once -> {output_root}", flush=True)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    def run_one(job: tuple[str, int]) -> None:
        name, seed = job
        log_path = log_dir / f"{name}_seed{seed}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--job", name, "--seed", str(seed), "--output-root", str(output_root)]
        if smoke:
            command.append("--smoke")
        started = datetime.now()
        start = perf_counter()
        with log_path.open("w", encoding="utf-8") as handle:
            code = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        minutes = (perf_counter() - start) / 60.0
        status = "completed" if code == 0 else f"failed (exit {code})"
        with lock:
            new_file = not run_log.exists()
            with run_log.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if new_file:
                    writer.writerow(["config", "seed", "started", "minutes", "status", "log"])
                writer.writerow([name, seed, f"{started:%Y-%m-%d %H:%M}", f"{minutes:.1f}", status, log_path.name])
            print(f"[{datetime.now():%H:%M:%S}] {name} seed {seed}: {status} in {minutes:.1f} min", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_one, jobs))
    print("all runs finished", flush=True)


def read_kpis(path: Path) -> dict | None:
    """Return the headline test KPIs from a ``test_kpis.csv``/``kpis.csv`` file."""
    import pandas as pd

    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    district = frame[frame["level"] == "district"].set_index("cost_function")["value"]
    buildings = frame[(frame["level"] == "building") & (frame["cost_function"] == "discomfort_proportion")]
    discomfort = buildings.set_index("name")["value"].sort_index()
    row = {
        "carbon_kgco2": float(district["carbon_emissions_kgco2"]),
        "electricity_kwh": float(district["electricity_consumption_kwh"]),
        "daily_peak_kwh": float(district["daily_peak_kwh"]),
        "average_discomfort": float(discomfort.mean()),
        "worst_discomfort": float(discomfort.max()),
        "best_discomfort": float(discomfort.min()),
        "discomfort_gap": float(discomfort.max() - discomfort.min()),
        "worst_building": str(discomfort.idxmax()),
    }
    for building, value in discomfort.items():
        row[f"discomfort_{building.lower()}"] = float(value)
    return row


def summarize(seeds: list[int], names: list[str], output_root: Path) -> None:
    import pandas as pd

    rbc = read_kpis(REFERENCES["rbc"][1])
    if rbc is None:
        raise FileNotFoundError(REFERENCES["rbc"][1])
    rows = []

    def add(config: str, label: str, seed: int, kpis: dict | None, note: str = "") -> None:
        if kpis is None:
            return
        rows.append({"config": config, "label": label, "seed": seed, **kpis, "carbon_ratio_vs_rbc": kpis["carbon_kgco2"] / rbc["carbon_kgco2"], "note": note})

    for key, (label, path) in REFERENCES.items():
        add(key, label, 42, read_kpis(path), "existing run (reference)")
    for name in names:
        label = CONFIGS[name][0]
        for seed in seeds:
            directory = seed_dir(output_root, name, seed)
            note = ""
            if CONFIGS[name][2] == "sac":
                selection = directory / "model_selection.json"
                if selection.is_file():
                    note = f"validation-selected episode {json.loads(selection.read_text())['selected_episode']}"
                add(f"{name}_final_episode", f"{label}, final episode", seed, read_kpis(directory / "test_final_episode" / "kpis.csv"), "final episode, as MARLISA")
            add(name, label, seed, read_kpis(directory / "test_kpis.csv"), note)

    per_run = pd.DataFrame(rows)
    summary_dir = output_root / "seed_matrix_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(summary_dir / "per_run.csv", index=False)

    metrics = ["carbon_kgco2", "carbon_ratio_vs_rbc", "electricity_kwh", "daily_peak_kwh", "average_discomfort", "worst_discomfort", "discomfort_gap"]
    metrics += sorted(column for column in per_run.columns if column.startswith("discomfort_building"))
    grouped = per_run.groupby(["config", "label"], sort=False)
    summary = grouped[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "seeds", grouped["seed"].apply(lambda s: " ".join(str(v) for v in sorted(s))))
    summary.insert(0, "n_seeds", grouped.size())
    summary.reset_index().to_csv(summary_dir / "summary.csv", index=False)
    print(summary.reset_index()[["label", "n_seeds", "carbon_kgco2_mean", "average_discomfort_mean", "worst_discomfort_mean", "discomfort_gap_mean"]].to_string(index=False))
    print(f"\nwritten: {summary_dir / 'per_run.csv'} and {summary_dir / 'summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="print the runs that --launch would start")
    action.add_argument("--launch", action="store_true", help="run every configuration and seed")
    action.add_argument("--summarize", action="store_true", help="collect test KPIs into summary CSVs")
    action.add_argument("--job", choices=sorted(CONFIGS), help="run one configuration (used by --launch)")
    parser.add_argument("--seed", type=int, help="seed for --job")
    parser.add_argument("--seeds", type=int, nargs="+", help="seeds (default: first 3 of DEFAULT_SEEDS)")
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), help="subset of configurations")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--smoke", action="store_true", help="short runs to check the wiring")
    args = parser.parse_args()

    seeds = args.seeds or default_seeds()
    names = args.configs or list(CONFIGS)
    output_root = args.output_root.resolve()

    if args.job:
        if args.seed is None:
            parser.error("--job needs --seed")
        run_job(args.job, args.seed, output_root, args.smoke)
    elif args.list:
        for name in names:
            for seed in seeds:
                exists = (seed_dir(output_root, name, seed) / "test_kpis.csv").is_file()
                print(f"{name:36s} seed {seed:<5d} {'exists, skipped' if exists else 'to run'}")
    elif args.launch:
        launch(seeds, names, output_root, args.workers, args.smoke)
    else:
        summarize(seeds, names, output_root)


if __name__ == "__main__":
    main()
