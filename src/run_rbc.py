"""Run CityLearn's BasicRBC controller and save reproducible results.

Defaults evaluate the rule-based controller on the held-out test split
(June 26–30) so RBC KPIs align with Independent SAC and MARLISA reports.
Use ``split='all'`` for a full-month reference run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import citylearn
import pandas as pd
from citylearn.agents.rbc import BasicRBC

try:
    from src.dataset_splits import (
        LSTM_LOOKBACK,
        TimeSplit,
        get_split,
        warmup_start_time_step,
    )
    from src.experiment_data import (
        ObservationHistoryCityLearnEnv,
        observations_dataframe,
    )
    from src.run_independent_sac import compute_split_kpis
    from src.sac_reward import DEFAULT_COMFORT_BAND
except ModuleNotFoundError:
    from dataset_splits import (
        LSTM_LOOKBACK,
        TimeSplit,
        get_split,
        warmup_start_time_step,
    )
    from experiment_data import (
        ObservationHistoryCityLearnEnv,
        observations_dataframe,
    )
    from run_independent_sac import compute_split_kpis
    from sac_reward import DEFAULT_COMFORT_BAND


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
    *,
    episode_start: int,
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
                    "time_step_local": time_step,
                    "time_step_absolute": episode_start + time_step,
                    "hour": hour,
                    "building": building.name,
                    "action": name,
                    "value": float(value),
                }
            )

    return records


def _fairness_summary(kpis: pd.DataFrame) -> pd.DataFrame:
    values = kpis.loc[
        (kpis["cost_function"] == "discomfort_proportion")
        & (kpis["level"] == "building"),
        ["name", "value"],
    ].dropna()
    if values.empty:
        return pd.DataFrame(
            columns=[
                "average_discomfort",
                "worst_discomfort",
                "best_discomfort",
                "discomfort_gap",
                "worst_building",
            ]
        )
    worst = values.loc[values["value"].idxmax()]
    return pd.DataFrame(
        [
            {
                "average_discomfort": float(values["value"].mean()),
                "worst_discomfort": float(values["value"].max()),
                "best_discomfort": float(values["value"].min()),
                "discomfort_gap": float(
                    values["value"].max() - values["value"].min()
                ),
                "worst_building": worst["name"],
            }
        ]
    )


def run_basic_rbc(
    schema: str | Path = DEFAULT_SCHEMA,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    random_seed: int = 42,
    save_results: bool = True,
    simulate_power_outage: bool | None = None,
    split: TimeSplit | str = "test",
    include_lstm_warmup: bool = True,
    comfort_band: float = DEFAULT_COMFORT_BAND,
    observe_simulated_temperature: bool = False,
) -> dict[str, Any]:
    """Simulate one episode with CityLearn's ``BasicRBC``.

    The environment is decentralized so that each building has its own action
    vector. Default ``split='test'`` scores June 26–30 with LSTM warm-up, matching
    the held-out Independent SAC / MARLISA protocol. Results are written under
    ``output_directory/seed_<seed>/``.

    ``simulate_power_outage=None`` keeps the schema setting (outages on), except
    warm-started held-out splits force outages off to avoid CityLearn crashes.
    """
    schema = Path(schema).resolve()
    base_output_directory = Path(output_directory).resolve()
    seed_directory = base_output_directory / f"seed_{random_seed}"
    output_directory = seed_directory

    if not schema.is_file():
        raise FileNotFoundError(f"CityLearn schema was not found: {schema}")

    is_full_dataset = split == "all" or split is None
    split_obj = None if is_full_dataset else (
        get_split(split) if isinstance(split, str) else split
    )

    start_step = None
    end_step = None
    scored_start = None
    if split_obj is not None:
        scored_start = split_obj.start_time_step
        if include_lstm_warmup and split_obj.start_time_step > 0:
            start_step = warmup_start_time_step(split_obj, LSTM_LOOKBACK)
            if simulate_power_outage is None:
                simulate_power_outage = False
        else:
            start_step = split_obj.start_time_step
        end_step = split_obj.end_time_step

    env_kwargs: dict[str, Any] = {
        "schema": schema,
        "central_agent": False,
        "random_seed": random_seed,
        "observe_simulated_temperature": observe_simulated_temperature,
    }
    if start_step is not None:
        env_kwargs["simulation_start_time_step"] = start_step
    if end_step is not None:
        env_kwargs["simulation_end_time_step"] = end_step
    if simulate_power_outage is not None:
        env_kwargs["simulate_power_outage"] = simulate_power_outage

    env = ObservationHistoryCityLearnEnv(**env_kwargs)
    episode_start = int(start_step if start_step is not None else 0)
    warmup_steps = (
        max(int(scored_start) - episode_start, 0) if scored_start is not None else 0
    )

    agent = BasicRBC(env)
    observations, _ = env.reset(seed=random_seed)
    terminated = truncated = False
    action_records: list[dict[str, Any]] = []
    reward_records: list[dict[str, Any]] = []
    local_step = 0

    while not (terminated or truncated):
        actions = agent.predict(observations, deterministic=True)
        action_records.extend(
            _action_records(
                env,
                observations,
                actions,
                local_step,
                episode_start=episode_start,
            )
        )

        observations, rewards, terminated, truncated, _ = env.step(actions)

        for building, reward in zip(env.buildings, rewards):
            reward_records.append(
                {
                    "time_step": local_step,
                    "time_step_local": local_step,
                    "time_step_absolute": episode_start + local_step,
                    "building": building.name,
                    "reward": float(reward),
                }
            )
        local_step += 1

    actions = pd.DataFrame(action_records)
    rewards = pd.DataFrame(reward_records)
    observations = observations_dataframe(env)

    if split_obj is not None:
        kpis = compute_split_kpis(
            env,
            scored_start_absolute=split_obj.start_time_step,
            comfort_band=comfort_band,
        )
        split_name = split_obj.name
    else:
        kpis = env.evaluate()
        split_name = "all"

    kpi_table = kpis.pivot(index="cost_function", columns="name", values="value")
    fairness = _fairness_summary(kpis)

    metadata = {
        "controller": "citylearn.agents.rbc.BasicRBC",
        "citylearn_version": citylearn.__version__,
        "schema": str(schema),
        "central_agent": False,
        "random_seed": random_seed,
        "split": split_name,
        "include_lstm_warmup": bool(include_lstm_warmup and split_obj is not None),
        "lstm_lookback": LSTM_LOOKBACK,
        "warmup_steps": warmup_steps,
        "simulate_power_outage": bool(env.buildings[0].simulate_power_outage),
        "observe_simulated_temperature": bool(env.observe_simulated_temperature),
        "time_steps": env.time_steps,
        "control_steps": int(actions["time_step"].nunique()) if not actions.empty else 0,
        "scored_transitions": max(local_step - warmup_steps, 0),
        "observation_rows": len(observations),
        "observation_columns": list(observations.columns),
        "buildings": [building.name for building in env.buildings],
        "action_names": env.action_names,
        "action_map": agent.action_map,
        "base_output_directory": str(base_output_directory),
    }

    if save_results:
        output_directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{split_name}_" if split_name != "all" else ""
        actions.to_csv(output_directory / f"{prefix}actions.csv", index=False)
        rewards.to_csv(output_directory / f"{prefix}rewards.csv", index=False)
        observations.to_csv(
            output_directory / f"{prefix}observations.csv", index=False
        )
        kpis.to_csv(output_directory / f"{prefix}kpis.csv", index=False)
        kpi_table.to_csv(output_directory / f"{prefix}kpi_table.csv")
        fairness.to_csv(
            output_directory / f"{prefix}fairness_summary.csv", index=False
        )
        # Alias used by comparison loaders / Independent SAC layout.
        if split_name == "test":
            kpis.to_csv(output_directory / "test_kpis.csv", index=False)
        elif split_name == "all":
            kpis.to_csv(output_directory / "kpis.csv", index=False)
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
        "fairness": fairness,
        "metadata": metadata,
        "output_directory": output_directory,
        "base_output_directory": base_output_directory,
        "split": split_name,
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
        "--split",
        choices=("all", "train", "validation", "test"),
        default="test",
        help="Dataset split to evaluate (default 'test' held-out protocol).",
    )
    parser.add_argument(
        "--no-lstm-warmup",
        action="store_true",
        help="Disable LSTM warm-up before the scored split.",
    )
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
        split=args.split,
        include_lstm_warmup=not args.no_lstm_warmup,
    )

    control_steps = (
        result["actions"]["time_step"].nunique() if not result["actions"].empty else 0
    )
    print(f"\nCompleted {control_steps} control steps on split={result['split']}.")
    print("\nBasicRBC evaluation (lower is better for these KPIs):")
    print(result["kpi_table"].round(3).to_string())
    print(f"\nSaved results to: {result['output_directory']}")


if __name__ == "__main__":
    main()
