"""Run CityLearn's BasicRBC controller and save reproducible results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import citylearn
import pandas as pd
from citylearn.agents.rbc import BasicRBC

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
DEFAULT_SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "rbc"


def _get_hour(
    observation_names: list[list[str]],
    observations: list[list[float]],
    agent_index: int,
) -> int | None:
    """Return the current hour for one controller, if it is observable."""
    names = observation_names[agent_index]

    if "hour" not in names:
        return None

    return int(round(observations[agent_index][names.index("hour")]))


def _action_records(
    env: ObservationHistoryCityLearnEnv,
    observations: list[list[float]],
    actions: list[list[float]],
    time_step: int,
) -> list[dict[str, Any]]:
    """Convert nested CityLearn actions to tidy, building-level records."""
    records: list[dict[str, Any]] = []

    for building, names, values, agent_index in zip(
        env.buildings,
        env.action_names,
        actions,
        range(len(actions)),
    ):
        hour = _get_hour(env.observation_names, observations, agent_index)

        for name, value in zip(names, values):
            records.append(
                {
                    "time_step": time_step,
                    "hour": hour,
                    "building": building.name,
                    "action": name,
                    "value": float(value),
                }
            )

    return records


def run_basic_rbc(
    schema: str | Path = DEFAULT_SCHEMA,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    random_seed: int = 42,
    save_results: bool = True,
    simulate_power_outage: bool | None = None,
) -> dict[str, Any]:
    """Simulate one complete episode with CityLearn's ``BasicRBC``.

    The environment is decentralized so that each building has its own action
    vector. This is the same control arrangement used by independent SAC and
    MARLISA, which makes later comparisons easier to interpret.

    ``simulate_power_outage=None`` keeps the schema setting (outages on).
    """
    schema = Path(schema).resolve()
    output_directory = Path(output_directory).resolve()

    if not schema.is_file():
        raise FileNotFoundError(f"CityLearn schema was not found: {schema}")

    env_kwargs: dict[str, Any] = {
        "schema": schema,
        "central_agent": False,
        "random_seed": random_seed,
    }
    if simulate_power_outage is not None:
        env_kwargs["simulate_power_outage"] = simulate_power_outage
    env = ObservationHistoryCityLearnEnv(**env_kwargs)
    agent = BasicRBC(env)
    observations, _ = env.reset(seed=random_seed)
    terminated = truncated = False
    action_records: list[dict[str, Any]] = []
    reward_records: list[dict[str, Any]] = []

    while not (terminated or truncated):
        time_step = env.time_step
        actions = agent.predict(observations, deterministic=True)
        action_records.extend(_action_records(env, observations, actions, time_step))

        observations, rewards, terminated, truncated, _ = env.step(actions)

        for building, reward in zip(env.buildings, rewards):
            reward_records.append(
                {
                    "time_step": time_step,
                    "building": building.name,
                    "reward": float(reward),
                }
            )

    actions = pd.DataFrame(action_records)
    rewards = pd.DataFrame(reward_records)
    observations = observations_dataframe(env)
    kpis = env.evaluate()
    kpi_table = kpis.pivot(index="cost_function", columns="name", values="value")

    if save_results:
        output_directory.mkdir(parents=True, exist_ok=True)
        actions.to_csv(output_directory / "actions.csv", index=False)
        rewards.to_csv(output_directory / "rewards.csv", index=False)
        observations.to_csv(output_directory / "observations.csv", index=False)
        kpis.to_csv(output_directory / "kpis.csv", index=False)
        kpi_table.to_csv(output_directory / "kpi_table.csv")

        metadata = {
            "controller": "citylearn.agents.rbc.BasicRBC",
            "citylearn_version": citylearn.__version__,
            "schema": str(schema),
            "central_agent": False,
            "random_seed": random_seed,
            "simulate_power_outage": bool(env.buildings[0].simulate_power_outage),
            "time_steps": env.time_steps,
            "control_steps": int(actions["time_step"].nunique()),
            "observation_rows": len(observations),
            "observation_columns": list(observations.columns),
            "buildings": [building.name for building in env.buildings],
            "action_names": env.action_names,
            "action_map": agent.action_map,
        }
        (output_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    return {
        "env": env,
        "agent": agent,
        "actions": actions,
        "rewards": rewards,
        "observations": observations,
        "kpis": kpis,
        "kpi_table": kpi_table,
        "output_directory": output_directory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CityLearn BasicRBC for one episode."
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="Path to a CityLearn schema JSON file.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for CSV results and metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--no-power-outages",
        action="store_true",
        help="Disable the schema's power outages (WP1 fix).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_basic_rbc(
        schema=args.schema,
        output_directory=args.output_directory,
        random_seed=args.seed,
        simulate_power_outage=False if args.no_power_outages else None,
    )

    control_steps = result["actions"]["time_step"].nunique()
    print(f"\nCompleted {control_steps} control steps.")
    print("\nBasicRBC evaluation (lower is better for these KPIs):")
    print(result["kpi_table"].round(3).to_string())
    print(f"\nSaved results to: {result['output_directory']}")


if __name__ == "__main__":
    main()
