"""Execute the 06_agent_evaluation training plan outside Jupyter.

Mirrors the notebook configuration: seed=42, 50 episodes, skip Independent SAC,
held-out protocol defaults from the runners. Writes a timing table and log under
outputs/.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"
LOG_PATH = OUTPUTS_ROOT / "06_training_log.txt"
SUMMARY_PATH = OUTPUTS_ROOT / "06_experiment_run_log.csv"

EPISODES = 50
RANDOM_SEED = 42
SKIP_MODELS = {"independent_sac"}

CONTROLLERS = {
    "rbc": "RBC",
    "independent_sac": "Independent SAC",
    "marlisa": "MARLISA",
    "carbon_aware_marlisa": "Carbon-aware MARLISA",
    "fair_carbon_aware_marlisa": "Fair carbon-aware MARLISA",
}


def log(message: str) -> None:
    line = message.rstrip()
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    import pandas as pd
    from src.run_rbc import run_basic_rbc
    from src.run_independent_sac import run_independent_sac
    from src.run_marlisa import run_marlisa
    from src.run_carbon_aware_marlisa import run_carbon_aware_marlisa
    from src.run_fair_carbon_aware_marlisa import run_fair_carbon_aware_marlisa

    if not SCHEMA.is_file():
        raise FileNotFoundError(SCHEMA)

    if LOG_PATH.exists():
        LOG_PATH.unlink()

    log(f"Project: {PROJECT_ROOT}")
    log(f"Episodes={EPISODES} seed={RANDOM_SEED} skip={sorted(SKIP_MODELS)}")
    log(f"Schema: {SCHEMA}")

    jobs = [
        (
            "rbc",
            lambda: run_basic_rbc(
                schema=SCHEMA,
                output_directory=OUTPUTS_ROOT / "rbc",
                random_seed=RANDOM_SEED,
            ),
        ),
        (
            "independent_sac",
            lambda: run_independent_sac(
                schema=SCHEMA,
                output_directory=OUTPUTS_ROOT / "independent_sac",
                episodes=EPISODES,
                random_seed=RANDOM_SEED,
            ),
        ),
        (
            "marlisa",
            lambda: run_marlisa(
                schema=SCHEMA,
                output_directory=OUTPUTS_ROOT / "marlisa",
                episodes=EPISODES,
                random_seed=RANDOM_SEED,
                reward="electricity_discomfort",
            ),
        ),
        (
            "carbon_aware_marlisa",
            lambda: run_carbon_aware_marlisa(
                schema=SCHEMA,
                output_directory=OUTPUTS_ROOT / "carbon_aware_marlisa",
                episodes=EPISODES,
                random_seed=RANDOM_SEED,
            ),
        ),
        (
            "fair_carbon_aware_marlisa",
            lambda: run_fair_carbon_aware_marlisa(
                schema=SCHEMA,
                output_directory=OUTPUTS_ROOT / "fair_carbon_aware_marlisa",
                episodes=EPISODES,
                random_seed=RANDOM_SEED,
                carbon_weight=1.0,
                comfort_weight=0.25,
                fairness_weight=0.25,
                comfort_band=1.0,
            ),
        ),
    ]

    records = []
    overall_start = perf_counter()
    for model, run_job in jobs:
        label = CONTROLLERS[model]
        if model in SKIP_MODELS:
            log(f"\n{'=' * 72}\nSkipping {label} (reuse existing outputs)\n{'=' * 72}")
            records.append(
                {
                    "model": model,
                    "episodes": 1 if model == "rbc" else EPISODES,
                    "seed": RANDOM_SEED,
                    "minutes": 0.0,
                    "output_directory": str(
                        OUTPUTS_ROOT / model / f"seed_{RANDOM_SEED}"
                    ),
                    "status": "skipped",
                }
            )
            continue

        log(f"\n{'=' * 72}\nRunning {label}\n{'=' * 72}")
        start = perf_counter()
        result = run_job()
        elapsed_minutes = (perf_counter() - start) / 60.0
        out_dir = str(result["output_directory"])
        records.append(
            {
                "model": model,
                "episodes": 1 if model == "rbc" else EPISODES,
                "seed": RANDOM_SEED,
                "minutes": elapsed_minutes,
                "output_directory": out_dir,
                "status": "completed",
            }
        )
        log(f"Completed {model} in {elapsed_minutes:.1f} minutes -> {out_dir}")
        del result
        gc.collect()

    total_minutes = (perf_counter() - overall_start) / 60.0
    frame = pd.DataFrame(records)
    frame.to_csv(SUMMARY_PATH, index=False)
    log(f"\nSelected controllers finished in {total_minutes:.1f} minutes.")
    log(f"Timing table: {SUMMARY_PATH}")
    log(frame.to_string(index=False))


if __name__ == "__main__":
    main()
