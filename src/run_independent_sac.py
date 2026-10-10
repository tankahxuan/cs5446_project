"""Train and evaluate independent CityLearn SAC agents.

Each building is controlled by its own SAC policy because the environment is
created with ``central_agent=False``.  The final episode is deterministic and
is used for KPI reporting.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Literal

import citylearn
import numpy as np
import pandas as pd
import torch
from citylearn.agents.sac import SAC
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import IndependentSACReward


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "independent_sac"
RewardChoice = Literal["independent_sac", "schema"]


def _set_random_seeds(seed: int) -> None:
    """Seed the random number generators used by this experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _training_reward_records(
    env: CityLearnEnv,
    episodes: int,
) -> list[dict[str, Any]]:
    """Convert CityLearn's per-episode reward summaries to tidy records."""
    records: list[dict[str, Any]] = []
    building_names = [building.name for building in env.buildings]

    for episode, summary in enumerate(env.episode_rewards, start=1):
        if episode == episodes:
            phase = "deterministic_evaluation"
        elif episode == 1:
            phase = "initial_exploration"
        else:
            phase = "training"

        values_by_statistic = {
            statistic: np.atleast_1d(values).astype(float)
            for statistic, values in summary.items()
        }

        for index, building_name in enumerate(building_names):
            record: dict[str, Any] = {
                "episode": episode,
                "phase": phase,
                "building": building_name,
            }
            for statistic, values in values_by_statistic.items():
                record[statistic] = float(values[index])
            records.append(record)

    return records


def _final_action_records(agent: SAC, env: CityLearnEnv) -> list[dict[str, Any]]:
    """Return actions taken in the final deterministic episode."""
    completed_histories = [history[:-1] for history in agent.actions]
    control_steps = min(env.time_steps - 1, *(len(h) for h in completed_histories))
    records: list[dict[str, Any]] = []

    for building, names, history in zip(
        env.buildings,
        env.action_names,
        completed_histories,
    ):
        final_history = history[-control_steps:]
        for time_step, values in enumerate(final_history):
            for name, value in zip(names, values):
                records.append(
                    {
                        "time_step": time_step,
                        "building": building.name,
                        "action": name,
                        "value": float(value),
                    }
                )

    return records


def _final_reward_records(env: CityLearnEnv) -> list[dict[str, Any]]:
    """Return rewards from the final deterministic episode."""
    records: list[dict[str, Any]] = []

    for time_step, rewards in enumerate(env.rewards[1:]):
        for building, reward in zip(env.buildings, np.atleast_1d(rewards)):
            records.append(
                {
                    "time_step": time_step,
                    "building": building.name,
                    "reward": float(reward),
                }
            )

    return records


def _save_checkpoint(agent: SAC, path: Path, metadata: dict[str, Any]) -> None:
    """Save learned networks and normalization statistics for later analysis."""
    checkpoint = {
        "metadata": metadata,
        "policy_net": [network.state_dict() for network in agent.policy_net],
        "soft_q_net1": [network.state_dict() for network in agent.soft_q_net1],
        "soft_q_net2": [network.state_dict() for network in agent.soft_q_net2],
        "target_soft_q_net1": [
            network.state_dict() for network in agent.target_soft_q_net1
        ],
        "target_soft_q_net2": [
            network.state_dict() for network in agent.target_soft_q_net2
        ],
        "normalization": {
            "observation_mean": agent.norm_mean,
            "observation_std": agent.norm_std,
            "reward_mean": agent.r_norm_mean,
            "reward_std": agent.r_norm_std,
        },
    }
    torch.save(checkpoint, path)


def run_independent_sac(
    schema: str | Path = DEFAULT_SCHEMA,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    episodes: int = 3,
    random_seed: int = 42,
    reward: RewardChoice = "independent_sac",
    hidden_dimension: tuple[int, int] = (256, 256),
    discount: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    replay_buffer_capacity: int = 100_000,
    action_scaling_coefficient: float = 0.5,
    reward_scaling: float = 5.0,
    update_per_time_step: int = 1,
    standardize_start_time_step: int | None = None,
    end_exploration_time_step: int | None = None,
    simulation_end_time_step: int | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    """Train independent SAC policies and evaluate the final policy.

    At least three episodes are required by this workflow: initial replay-buffer
    collection, policy training, and a final deterministic evaluation episode.
    """
    if episodes < 3:
        raise ValueError(
            "episodes must be at least 3: exploration, training, and evaluation"
        )

    schema = Path(schema).resolve()
    output_directory = Path(output_directory).resolve()
    if not schema.is_file():
        raise FileNotFoundError(f"CityLearn schema was not found: {schema}")

    _set_random_seeds(random_seed)
    env_kwargs: dict[str, Any] = {
        "schema": schema,
        "central_agent": False,
        "random_seed": random_seed,
    }
    if simulation_end_time_step is not None:
        env_kwargs["simulation_end_time_step"] = simulation_end_time_step

    env = CityLearnEnv(**env_kwargs)
    if reward == "independent_sac":
        # CityLearn 3.0.2 otherwise forwards the schema's ComfortReward
        # arguments to this class when it is passed as a constructor override.
        env.reward_function = IndependentSACReward(env.get_metadata())
    elif reward != "schema":
        raise ValueError(f"Unsupported reward choice: {reward}")

    for index, action_space in enumerate(env.action_space):
        action_space.seed(random_seed + index)

    agent = SAC(
        env,
        hidden_dimension=list(hidden_dimension),
        discount=discount,
        tau=tau,
        alpha=alpha,
        lr=learning_rate,
        batch_size=batch_size,
        replay_buffer_capacity=replay_buffer_capacity,
        standardize_start_time_step=standardize_start_time_step,
        end_exploration_time_step=end_exploration_time_step,
        # CityLearn 3.0.2 spells this constructor argument "coefficienct".
        action_scaling_coefficienct=action_scaling_coefficient,
        reward_scaling=reward_scaling,
        update_per_time_step=update_per_time_step,
    )
    agent.learn(
        episodes=episodes,
        deterministic_finish=True,
        logging_level=30,
    )

    training_rewards = pd.DataFrame(_training_reward_records(env, episodes))
    actions = pd.DataFrame(_final_action_records(agent, env))
    rewards = pd.DataFrame(_final_reward_records(env))
    kpis = env.evaluate()
    kpi_table = kpis.pivot(index="cost_function", columns="name", values="value")

    metadata = {
        "controller": "citylearn.agents.sac.SAC",
        "control_layout": "one independent policy per building",
        "citylearn_version": citylearn.__version__,
        "torch_version": torch.__version__,
        "schema": str(schema),
        "reward_choice": reward,
        "reward_function": type(env.reward_function).__name__,
        "central_agent": False,
        "random_seed": random_seed,
        "episodes": episodes,
        "final_episode_is_deterministic": True,
        "episode_time_steps": env.time_steps,
        "buildings": [building.name for building in env.buildings],
        "observation_names": env.observation_names,
        "action_names": env.action_names,
        "hyperparameters": {
            "hidden_dimension": list(hidden_dimension),
            "discount": discount,
            "tau": tau,
            "alpha": alpha,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "replay_buffer_capacity": replay_buffer_capacity,
            "action_scaling_coefficient": action_scaling_coefficient,
            "reward_scaling": reward_scaling,
            "update_per_time_step": update_per_time_step,
            "standardize_start_time_step": agent.standardize_start_time_step,
            "end_exploration_time_step": agent.end_exploration_time_step,
        },
        "evaluation_note": (
            "The deterministic final episode reuses the same simulation period "
            "as training and is a development result, not a held-out estimate."
        ),
    }

    if save_results:
        output_directory.mkdir(parents=True, exist_ok=True)
        training_rewards.to_csv(
            output_directory / "training_rewards.csv", index=False
        )
        actions.to_csv(output_directory / "actions.csv", index=False)
        rewards.to_csv(output_directory / "rewards.csv", index=False)
        kpis.to_csv(output_directory / "kpis.csv", index=False)
        kpi_table.to_csv(output_directory / "kpi_table.csv")
        (output_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        _save_checkpoint(agent, output_directory / "checkpoint.pt", metadata)

    return {
        "env": env,
        "agent": agent,
        "training_rewards": training_rewards,
        "actions": actions,
        "rewards": rewards,
        "kpis": kpis,
        "kpi_table": kpi_table,
        "metadata": metadata,
        "output_directory": output_directory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate independent CityLearn SAC policies."
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reward",
        choices=("independent_sac", "schema"),
        default="independent_sac",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument(
        "--simulation-end-time-step",
        type=int,
        default=None,
        help="Optional shorter end step, useful only for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_independent_sac(
        schema=args.schema,
        output_directory=args.output_directory,
        episodes=args.episodes,
        random_seed=args.seed,
        reward=args.reward,
        batch_size=args.batch_size,
        update_per_time_step=args.updates_per_step,
        simulation_end_time_step=args.simulation_end_time_step,
    )
    print("\nIndependent SAC evaluation (lower is better for these KPIs):")
    print(result["kpi_table"].round(3).to_string())
    print(f"\nSaved results to: {result['output_directory']}")


if __name__ == "__main__":
    main()
