"""Backfill observations.csv by replaying saved deterministic actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.experiment_data import (
        ObservationHistoryCityLearnEnv,
        observations_dataframe,
    )
except ModuleNotFoundError:
    from experiment_data import (
        ObservationHistoryCityLearnEnv,
        observations_dataframe,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
CANONICAL_CONTROLLERS = (
    "rbc",
    "independent_sac",
    "marlisa",
    "carbon_aware_marlisa",
    "fair_carbon_aware_marlisa",
)


def replay_observations(result_directory: str | Path) -> pd.DataFrame:
    """Replay one saved action sequence and write its observation trajectory."""

    result_directory = Path(result_directory).resolve()
    actions_path = result_directory / "actions.csv"
    metadata_path = result_directory / "metadata.json"

    if not actions_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Expected actions.csv and metadata.json in {result_directory}."
        )

    actions = pd.read_csv(actions_path)
    metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema = Path(metadata["schema"]).resolve()
    if not schema.is_file():
        project_schema = PROJECT_ROOT / "notebooks" / "data" / "schema.json"
        if not project_schema.is_file():
            raise FileNotFoundError(
                f"Schema from metadata does not exist: {schema}"
            )
        schema = project_schema
    random_seed = int(metadata.get("random_seed", 42))

    env = ObservationHistoryCityLearnEnv(
        schema=schema,
        central_agent=False,
        random_seed=random_seed,
    )
    env.reset(seed=random_seed)

    for time_step in sorted(actions["time_step"].unique()):
        if int(time_step) != env.time_step:
            raise ValueError(
                f"Expected action time step {env.time_step}, got {time_step}."
            )

        step_actions: list[list[float]] = []
        for building, action_names in zip(env.buildings, env.action_names):
            building_actions = actions.loc[
                (actions["time_step"] == time_step)
                & (actions["building"] == building.name),
                ["action", "value"],
            ]
            if building_actions["action"].duplicated().any():
                raise ValueError(
                    f"Duplicate actions for {building.name} at time step {time_step}."
                )

            values = building_actions.set_index("action")["value"]
            missing = [name for name in action_names if name not in values]
            if missing:
                raise ValueError(
                    f"Missing actions for {building.name} at time step "
                    f"{time_step}: {missing}"
                )
            step_actions.append([float(values[name]) for name in action_names])

        _, _, terminated, truncated, _ = env.step(step_actions)
        if (terminated or truncated) and time_step != actions["time_step"].max():
            raise ValueError("The replay episode ended before the saved actions did.")

    observations = observations_dataframe(env)

    kpis_path = result_directory / "kpis.csv"
    if kpis_path.is_file():
        saved_kpis = pd.read_csv(kpis_path)
        replayed_kpis = env.evaluate()
        comparison = saved_kpis.merge(
            replayed_kpis,
            on=["cost_function", "name", "level"],
            suffixes=("_saved", "_replayed"),
            how="inner",
        ).dropna(subset=["value_saved", "value_replayed"])
        max_kpi_difference = float(
            (comparison["value_saved"] - comparison["value_replayed"])
            .abs()
            .max()
        )
        if max_kpi_difference > 1e-5:
            raise ValueError(
                "Replayed actions did not reproduce the saved KPIs; maximum "
                f"absolute difference was {max_kpi_difference:.6g}."
            )
        observations.attrs["max_kpi_difference"] = max_kpi_difference

    observations.to_csv(result_directory / "observations.csv", index=False)
    return observations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create observations.csv from saved controller actions."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=list(CANONICAL_CONTROLLERS),
        help="Result folder names to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for controller in args.controllers:
        directory = args.output_root / controller
        observations = replay_observations(directory)
        print(
            f"{controller}: saved {len(observations)} rows and "
            f"{len(observations.columns)} columns; maximum KPI difference "
            f"{observations.attrs.get('max_kpi_difference', float('nan')):.3g}"
        )


if __name__ == "__main__":
    main()

