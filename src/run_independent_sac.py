"""Train and evaluate independent CityLearn SAC agents with held-out splits.

Training uses only June 1–20 (steps 0–479). Validation (June 21–25) selects
checkpoints. Testing (June 26–30) is a final held-out report. The reward
penalizes electricity imports and occupied thermal discomfort only.

This module preserves helper functions imported by ``run_marlisa.py``.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import citylearn
import numpy as np
import pandas as pd
import torch
from citylearn.agents.sac import SAC
from citylearn.cost_function import CostFunction

try:
    from src.dataset_splits import (
        DEFAULT_SEEDS,
        LSTM_LOOKBACK,
        SPLITS,
        TEST_SPLIT,
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
        TimeSplit,
        get_split,
        warmup_start_time_step,
    )
    from src.experiment_data import (
        ObservationHistoryCityLearnEnv,
        observations_dataframe,
    )
    from src.sac_reward import (
        DEFAULT_COMFORT_BAND,
        DEFAULT_DISCOMFORT_SCALE,
        DEFAULT_ELECTRICITY_SCALE,
        DEFAULT_HIGHER_EXPONENT,
        DEFAULT_LAMBDA_D,
        DEFAULT_LAMBDA_E,
        DEFAULT_LOWER_EXPONENT,
        ElectricityDiscomfortSACReward,
    )
except ModuleNotFoundError:
    from dataset_splits import (
        DEFAULT_SEEDS,
        LSTM_LOOKBACK,
        SPLITS,
        TEST_SPLIT,
        TRAIN_SPLIT,
        VALIDATION_SPLIT,
        TimeSplit,
        get_split,
        warmup_start_time_step,
    )
    from experiment_data import (
        ObservationHistoryCityLearnEnv,
        observations_dataframe,
    )
    from sac_reward import (
        DEFAULT_COMFORT_BAND,
        DEFAULT_DISCOMFORT_SCALE,
        DEFAULT_ELECTRICITY_SCALE,
        DEFAULT_HIGHER_EXPONENT,
        DEFAULT_LAMBDA_D,
        DEFAULT_LAMBDA_E,
        DEFAULT_LOWER_EXPONENT,
        ElectricityDiscomfortSACReward,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "independent_sac"

# Verified CityLearn SAC semantics for a 480-step training episode:
# - predict() explores while agent.time_step <= end_exploration_time_step
#   (code uses ``time_step > end_exploration_time_step`` for the policy branch)
# - agent.time_step increments inside predict() and persists across episodes
# - one train episode yields 479 transitions, leaving agent.time_step == 479
# - observation/reward standardization and Adam updates begin in update() when
#   agent.time_step >= standardize_start_time_step and the replay buffer is
#   at least batch_size
# Defaults below explore the first training episode, fit normalization at the
# end of episode 1 / start of episode 2, and use the learned policy from the
# first step of episode 2 onward.
DEFAULT_END_EXPLORATION_TIME_STEP = 478
DEFAULT_STANDARDIZE_START_TIME_STEP = 479
DEFAULT_TRAINING_EPISODES = 50
DEFAULT_VALIDATION_EVERY = 5
DEFAULT_DISCOMFORT_THRESHOLD = 0.20

RewardChoice = Literal["electricity_discomfort", "independent_sac", "schema"]


def _set_random_seeds(seed: int) -> None:
    """Seed the random number generators used by this experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _training_reward_records(
    env: ObservationHistoryCityLearnEnv,
    episodes: int,
) -> list[dict[str, Any]]:
    """Convert CityLearn's per-episode reward summaries to tidy records.

    Preserved for MARLISA compatibility. When the legacy
    ``deterministic_finish=True`` workflow is used, the final episode is labeled
    ``deterministic_evaluation``.
    """
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


def _final_action_records(agent: SAC, env: ObservationHistoryCityLearnEnv) -> list[dict[str, Any]]:
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


def _final_reward_records(env: ObservationHistoryCityLearnEnv) -> list[dict[str, Any]]:
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


class CountingSAC(SAC):
    """CityLearn SAC with environment-step and gradient-update counters."""

    def __init__(self, env: ObservationHistoryCityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.environment_steps = 0
        self.gradient_updates = 0

    def update(
        self,
        observations: list[list[float]],
        actions: list[list[float]],
        reward: list[float],
        next_observations: list[list[float]],
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.environment_steps += 1
        for index in range(len(self.action_space)):
            ready = (
                self.time_step >= self.standardize_start_time_step
                and self.batch_size <= len(self.replay_buffer[index])
            )
            if ready:
                self.gradient_updates += int(self.update_per_time_step)
        super().update(
            observations,
            actions,
            reward,
            next_observations,
            terminated,
            truncated,
        )


def _reward_kwargs(
    lambda_electricity: float,
    lambda_discomfort: float,
    electricity_scale: float,
    discomfort_scale: float,
    comfort_band: float,
    lower_exponent: float,
    higher_exponent: float,
) -> dict[str, float]:
    return {
        "lambda_electricity": lambda_electricity,
        "lambda_discomfort": lambda_discomfort,
        "electricity_scale": electricity_scale,
        "discomfort_scale": discomfort_scale,
        "band": comfort_band,
        "lower_exponent": lower_exponent,
        "higher_exponent": higher_exponent,
    }


def make_sac_env(
    schema: str | Path,
    split: TimeSplit | str,
    *,
    random_seed: int,
    reward: RewardChoice = "electricity_discomfort",
    reward_kwargs: dict[str, float] | None = None,
    include_lstm_warmup: bool = False,
    simulate_power_outage: bool | None = None,
) -> ObservationHistoryCityLearnEnv:
    """Create a decentralized CityLearn env for one dataset split."""
    split = get_split(split) if isinstance(split, str) else split
    schema = Path(schema).resolve()
    reward_kwargs = dict(reward_kwargs or {})

    if include_lstm_warmup and split.start_time_step > 0:
        start = warmup_start_time_step(split, LSTM_LOOKBACK)
        # Earlier starts can trip CityLearn's stochastic outage day alignment.
        if simulate_power_outage is None:
            simulate_power_outage = False
    else:
        start = split.start_time_step

    env_kwargs: dict[str, Any] = {
        "schema": schema,
        "central_agent": False,
        "random_seed": random_seed,
        "simulation_start_time_step": start,
        "simulation_end_time_step": split.end_time_step,
    }
    if simulate_power_outage is not None:
        env_kwargs["simulate_power_outage"] = simulate_power_outage

    env = ObservationHistoryCityLearnEnv(**env_kwargs)
    env._sac_split_name = split.name
    env._sac_scored_start = split.start_time_step
    env._sac_warmup_steps = split.start_time_step - start

    if reward in ("electricity_discomfort", "independent_sac"):
        env.reward_function = ElectricityDiscomfortSACReward(
            env.get_metadata(),
            **reward_kwargs,
        )
    elif reward != "schema":
        raise ValueError(f"Unsupported reward choice: {reward}")

    for index, action_space in enumerate(env.action_space):
        action_space.seed(random_seed + index)
    return env


def make_sac_agent(
    env: ObservationHistoryCityLearnEnv,
    *,
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
    standardize_start_time_step: int = DEFAULT_STANDARDIZE_START_TIME_STEP,
    end_exploration_time_step: int = DEFAULT_END_EXPLORATION_TIME_STEP,
) -> CountingSAC:
    return CountingSAC(
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


def build_inference_checkpoint(
    agent: CountingSAC,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build an inference-only checkpoint.

    Optimizer states, replay buffers, and RNG states are intentionally omitted.
    Loading this checkpoint cannot reproduce an uninterrupted training resume.
    """
    return {
        "checkpoint_type": "inference_only",
        "resumable_training": False,
        "resume_limitation": (
            "This checkpoint stores actor weights and observation-normalization "
            "statistics for deterministic evaluation only. It does not include "
            "optimizer states, replay buffers, or RNG states, so exact training "
            "resumption is not supported."
        ),
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
            "observation_mean": deepcopy(agent.norm_mean),
            "observation_std": deepcopy(agent.norm_std),
            "reward_mean": deepcopy(agent.r_norm_mean),
            "reward_std": deepcopy(agent.r_norm_std),
            "normalized": list(agent.normalized),
        },
        "architecture": {
            "hidden_dimension": list(agent.hidden_dimension),
            "observation_dimension": list(agent.observation_dimension),
            "action_dimension": list(agent.action_dimension),
        },
        "observation_names": deepcopy(agent.observation_names),
        "action_names": deepcopy(agent.action_names),
        "action_scaling_coefficient": float(agent.action_scaling_coefficient),
        "reward_scaling": float(agent.reward_scaling),
        "standardize_start_time_step": int(agent.standardize_start_time_step),
        "end_exploration_time_step": int(agent.end_exploration_time_step),
        "environment_steps": int(agent.environment_steps),
        "gradient_updates": int(agent.gradient_updates),
        "agent_time_step": int(agent.time_step),
    }


def save_inference_checkpoint(
    agent: CountingSAC,
    path: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = build_inference_checkpoint(agent, metadata=metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return checkpoint


def load_inference_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "policy_net" not in checkpoint:
        raise ValueError(f"Invalid SAC checkpoint: {path}")
    return checkpoint


def apply_checkpoint_to_agent(agent: SAC, checkpoint: dict[str, Any]) -> SAC:
    """Restore actor weights and normalization statistics for inference."""
    for index, state_dict in enumerate(checkpoint["policy_net"]):
        agent.policy_net[index].load_state_dict(state_dict)

    if "soft_q_net1" in checkpoint:
        for index, state_dict in enumerate(checkpoint["soft_q_net1"]):
            agent.soft_q_net1[index].load_state_dict(state_dict)
        for index, state_dict in enumerate(checkpoint["soft_q_net2"]):
            agent.soft_q_net2[index].load_state_dict(state_dict)
        for index, state_dict in enumerate(checkpoint["target_soft_q_net1"]):
            agent.target_soft_q_net1[index].load_state_dict(state_dict)
        for index, state_dict in enumerate(checkpoint["target_soft_q_net2"]):
            agent.target_soft_q_net2[index].load_state_dict(state_dict)

    normalization = checkpoint["normalization"]
    agent.norm_mean = deepcopy(normalization["observation_mean"])
    agent.norm_std = deepcopy(normalization["observation_std"])
    agent.r_norm_mean = deepcopy(normalization["reward_mean"])
    agent.r_norm_std = deepcopy(normalization["reward_std"])
    if "normalized" in normalization:
        agent.normalized = list(normalization["normalized"])
    else:
        agent.normalized = [mean is not None for mean in agent.norm_mean]

    for network_group in (
        agent.policy_net,
        agent.soft_q_net1,
        agent.soft_q_net2,
        agent.target_soft_q_net1,
        agent.target_soft_q_net2,
    ):
        for network in network_group:
            network.eval()
    return agent


def _episode_index_for_absolute_step(
    env: ObservationHistoryCityLearnEnv,
    absolute_time_step: int,
) -> int:
    start = int(env.episode_tracker.episode_start_time_step)
    return int(absolute_time_step - start)


def compute_split_kpis(
    env: ObservationHistoryCityLearnEnv,
    *,
    scored_start_absolute: int | None = None,
    comfort_band: float = DEFAULT_COMFORT_BAND,
) -> pd.DataFrame:
    """Compute district/building KPIs on the scored interval only.

    CityLearn legacy ``evaluate()`` electricity/carbon/cost values are
    baseline-normalized ratios. This helper also returns raw totals so reports
    do not misread normalized values as absolute kWh or percentages.
    """
    scored_start_absolute = (
        int(getattr(env, "_sac_scored_start", env.episode_tracker.episode_start_time_step))
        if scored_start_absolute is None
        else int(scored_start_absolute)
    )
    local_start = max(
        0,
        _episode_index_for_absolute_step(env, scored_start_absolute),
    )
    # After truncation CityLearn may leave time_step equal to the array length.
    local_end = min(int(env.time_step), int(env.time_steps) - 1)
    if env.buildings:
        series_len = len(env.buildings[0].net_electricity_consumption)
        local_end = min(local_end, series_len - 1)

    records: list[dict[str, Any]] = []
    district_net: list[float] = []
    district_emission: list[float] = []
    district_cost: list[float] = []

    for building in env.buildings:
        net = np.asarray(building.net_electricity_consumption[local_start : local_end + 1], dtype=float)
        emission = np.asarray(
            building.net_electricity_consumption_emission[local_start : local_end + 1],
            dtype=float,
        )
        cost = np.asarray(
            building.net_electricity_consumption_cost[local_start : local_end + 1],
            dtype=float,
        )
        indoor = building.indoor_dry_bulb_temperature[local_start : local_end + 1]
        cooling = building.indoor_dry_bulb_temperature_cooling_set_point[
            local_start : local_end + 1
        ]
        heating = building.indoor_dry_bulb_temperature_heating_set_point[
            local_start : local_end + 1
        ]
        occupants = building.occupant_count[local_start : local_end + 1]
        band = building.comfort_band[local_start : local_end + 1]

        if district_net:
            district_net = [a + b for a, b in zip(district_net, net.tolist())]
            district_emission = [
                a + b for a, b in zip(district_emission, emission.tolist())
            ]
            district_cost = [a + b for a, b in zip(district_cost, cost.tolist())]
        else:
            district_net = net.tolist()
            district_emission = emission.tolist()
            district_cost = cost.tolist()

        discomfort = CostFunction.discomfort(
            indoor_dry_bulb_temperature=indoor,
            dry_bulb_temperature_cooling_set_point=cooling,
            dry_bulb_temperature_heating_set_point=heating,
            band=comfort_band if comfort_band is not None else band,
            occupant_count=occupants,
        )
        electricity_positive = float(np.clip(net, 0.0, None).sum())
        carbon_positive = float(np.clip(emission, 0.0, None).sum())
        cost_positive = float(np.clip(cost, 0.0, None).sum())
        daily_peak = float(CostFunction.peak(net.tolist())[-1]) if len(net) else float("nan")
        ramping = float(CostFunction.ramping(net.tolist())[-1]) if len(net) else float("nan")

        building_metrics = {
            "electricity_consumption_kwh": electricity_positive,
            "carbon_emissions_kgco2": carbon_positive,
            "electricity_cost": cost_positive,
            "discomfort_proportion": float(discomfort[0][-1]) if discomfort[0] else float("nan"),
            "discomfort_cold_proportion": float(discomfort[1][-1]) if discomfort[1] else float("nan"),
            "discomfort_hot_proportion": float(discomfort[2][-1]) if discomfort[2] else float("nan"),
            "daily_peak_kwh": daily_peak,
            "ramping_kwh": ramping,
        }
        for metric, value in building_metrics.items():
            records.append(
                {
                    "name": building.name,
                    "level": "building",
                    "cost_function": metric,
                    "value": value,
                    "scored_local_start": local_start,
                    "scored_local_end": local_end,
                    "normalization_note": "raw_scored_interval",
                }
            )

    if district_net:
        discomfort_district = float("nan")
        building_discomfort = [
            r["value"]
            for r in records
            if r["cost_function"] == "discomfort_proportion"
        ]
        if building_discomfort:
            discomfort_district = float(np.nanmean(building_discomfort))

        district_metrics = {
            "electricity_consumption_kwh": float(np.clip(district_net, 0.0, None).sum()),
            "carbon_emissions_kgco2": float(np.clip(district_emission, 0.0, None).sum()),
            "electricity_cost": float(np.clip(district_cost, 0.0, None).sum()),
            "discomfort_proportion": discomfort_district,
            "daily_peak_kwh": float(CostFunction.peak(district_net)[-1]),
            "ramping_kwh": float(CostFunction.ramping(district_net)[-1]),
        }
        for metric, value in district_metrics.items():
            records.append(
                {
                    "name": "District",
                    "level": "district",
                    "cost_function": metric,
                    "value": value,
                    "scored_local_start": local_start,
                    "scored_local_end": local_end,
                    "normalization_note": "raw_scored_interval",
                }
            )

    return pd.DataFrame.from_records(records)


def _physical_state_records(
    env: ObservationHistoryCityLearnEnv,
    *,
    warmup_steps: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for building in env.buildings:
        end = min(
            int(env.time_step),
            int(env.time_steps) - 1,
            len(building.net_electricity_consumption) - 1,
        )
        for local_t in range(end + 1):
            absolute_t = int(env.episode_tracker.episode_start_time_step) + local_t
            scored = local_t >= warmup_steps
            cooling_set = float(
                building.indoor_dry_bulb_temperature_cooling_set_point[local_t]
            )
            indoor = float(building.indoor_dry_bulb_temperature[local_t])
            band = float(building.comfort_band[local_t])
            occupied = float(building.occupant_count[local_t]) > 0.0
            violation = 0.0
            if occupied and abs(indoor - cooling_set) > band:
                # With identical heating/cooling setpoints in this dataset,
                # KPI discomfort is equivalent to leaving the cooling band.
                if indoor > cooling_set + band or indoor < cooling_set - band:
                    violation = 1.0
            records.append(
                {
                    "time_step_local": local_t,
                    "time_step_absolute": absolute_t,
                    "scored": scored,
                    "building": building.name,
                    "indoor_dry_bulb_temperature": indoor,
                    "cooling_set_point": cooling_set,
                    "comfort_band": band,
                    "occupant_count": float(building.occupant_count[local_t]),
                    "comfort_violation": violation,
                    "electrical_storage_soc": float(
                        building.electrical_storage.soc[local_t]
                    ),
                    "dhw_storage_soc": float(building.dhw_storage.soc[local_t]),
                    "net_electricity_consumption": float(
                        building.net_electricity_consumption[local_t]
                    ),
                    "cooling_electricity_consumption": float(
                        building.cooling_electricity_consumption[local_t]
                    ),
                }
            )
    return records


def evaluate_independent_sac(
    checkpoint: dict[str, Any] | str | Path,
    *,
    schema: str | Path = DEFAULT_SCHEMA,
    split: TimeSplit | str = "validation",
    output_directory: str | Path | None = None,
    include_lstm_warmup: bool = True,
    save_results: bool = True,
) -> dict[str, Any]:
    """Deterministically evaluate a saved SAC policy on one held-out split.

    Creates a fresh environment and SAC instance, restores actor weights and
    observation-normalization statistics, and does not push transitions into a
    replay buffer or run optimizer updates.
    """
    if not isinstance(checkpoint, dict):
        checkpoint = load_inference_checkpoint(checkpoint)

    metadata = dict(checkpoint.get("metadata", {}))
    split = get_split(split) if isinstance(split, str) else split
    random_seed = int(metadata.get("random_seed", 42))
    reward_config = dict(metadata.get("reward_config", {}))
    hyper = dict(metadata.get("hyperparameters", {}))

    _set_random_seeds(random_seed)
    env = make_sac_env(
        schema,
        split,
        random_seed=random_seed,
        reward="electricity_discomfort",
        reward_kwargs=reward_config,
        include_lstm_warmup=include_lstm_warmup,
    )
    agent = make_sac_agent(
        env,
        hidden_dimension=tuple(hyper.get("hidden_dimension", (256, 256))),
        discount=float(hyper.get("discount", 0.99)),
        tau=float(hyper.get("tau", 0.005)),
        alpha=float(hyper.get("alpha", 0.2)),
        learning_rate=float(hyper.get("learning_rate", 3e-4)),
        batch_size=int(hyper.get("batch_size", 256)),
        replay_buffer_capacity=int(hyper.get("replay_buffer_capacity", 100_000)),
        action_scaling_coefficient=float(
            checkpoint.get(
                "action_scaling_coefficient",
                hyper.get("action_scaling_coefficient", 0.5),
            )
        ),
        reward_scaling=float(hyper.get("reward_scaling", 5.0)),
        update_per_time_step=int(hyper.get("update_per_time_step", 1)),
        standardize_start_time_step=int(
            checkpoint.get(
                "standardize_start_time_step",
                hyper.get(
                    "standardize_start_time_step",
                    DEFAULT_STANDARDIZE_START_TIME_STEP,
                ),
            )
        ),
        end_exploration_time_step=int(
            checkpoint.get(
                "end_exploration_time_step",
                hyper.get(
                    "end_exploration_time_step",
                    DEFAULT_END_EXPLORATION_TIME_STEP,
                ),
            )
        ),
    )
    apply_checkpoint_to_agent(agent, checkpoint)

    updates_before = int(agent.gradient_updates)
    steps_before = int(agent.environment_steps)
    optimizer_steps_before = [
        int(optimizer.state.get("step", 0)) if optimizer.state else 0
        for optimizer in agent.policy_optimizer
    ]

    observations, _ = env.reset(seed=random_seed)
    terminated = False
    truncated = False
    action_records: list[dict[str, Any]] = []
    reward_records: list[dict[str, Any]] = []
    local_step = 0
    warmup_steps = int(getattr(env, "_sac_warmup_steps", 0))

    while not (terminated or truncated):
        actions = agent.predict(observations, deterministic=True)
        # predict() advances agent.time_step and would otherwise leave training
        # side effects if update were called; evaluation never calls update.
        next_observations, rewards, terminated, truncated, _ = env.step(actions)

        for building, names, values in zip(env.buildings, env.action_names, actions):
            for name, value in zip(names, values):
                action_records.append(
                    {
                        "time_step_local": local_step,
                        "time_step_absolute": int(
                            env.episode_tracker.episode_start_time_step
                        )
                        + local_step,
                        "scored": local_step >= warmup_steps,
                        "building": building.name,
                        "action": name,
                        "value": float(value),
                    }
                )

        components = getattr(env.reward_function, "last_components", None)
        for index, (building, reward) in enumerate(
            zip(env.buildings, np.atleast_1d(rewards))
        ):
            record: dict[str, Any] = {
                "time_step_local": local_step,
                "time_step_absolute": int(
                    env.episode_tracker.episode_start_time_step
                )
                + local_step,
                "scored": local_step >= warmup_steps,
                "building": building.name,
                "reward": float(reward),
            }
            if components and index < len(components):
                record.update(
                    {
                        key: float(value)
                        for key, value in components[index].items()
                        if key != "reward"
                    }
                )
            reward_records.append(record)

        observations = next_observations
        local_step += 1

    updates_after = int(agent.gradient_updates)
    steps_after = int(agent.environment_steps)
    optimizer_steps_after = [
        int(optimizer.state.get("step", 0)) if optimizer.state else 0
        for optimizer in agent.policy_optimizer
    ]
    if updates_after != updates_before or steps_after != steps_before:
        raise RuntimeError("Evaluation unexpectedly collected training experience.")
    if optimizer_steps_after != optimizer_steps_before:
        raise RuntimeError("Evaluation unexpectedly updated optimizer state.")

    kpis = compute_split_kpis(
        env,
        scored_start_absolute=split.start_time_step,
        comfort_band=float(reward_config.get("band", DEFAULT_COMFORT_BAND)),
    )
    citylearn_kpis = env.evaluate(comfort_band=float(reward_config.get("band", DEFAULT_COMFORT_BAND)))
    observations_df = observations_dataframe(env)
    physical_state = pd.DataFrame(_physical_state_records(env, warmup_steps=warmup_steps))
    actions_df = pd.DataFrame(action_records)
    rewards_df = pd.DataFrame(reward_records)

    eval_metadata = {
        "phase": split.name,
        "split": split.as_dict(),
        "random_seed": random_seed,
        "include_lstm_warmup": include_lstm_warmup,
        "warmup_steps": warmup_steps,
        "transitions": local_step,
        "scored_transitions": max(local_step - warmup_steps, 0),
        "deterministic": True,
        "optimizer_updates_during_eval": 0,
        "citylearn_version": citylearn.__version__,
        "torch_version": torch.__version__,
        "checkpoint_type": checkpoint.get("checkpoint_type", "unknown"),
        "kpi_note": (
            "Primary KPIs in kpis.csv are raw scored-interval totals/proportions. "
            "citylearn_kpis.csv retains CityLearn legacy evaluate() values, where "
            "electricity/carbon/cost are baseline-normalized ratios and "
            "discomfort_proportion is an occupied-time discomfort fraction."
        ),
    }

    result = {
        "env": env,
        "agent": agent,
        "split": split,
        "actions": actions_df,
        "rewards": rewards_df,
        "observations": observations_df,
        "physical_state": physical_state,
        "kpis": kpis,
        "citylearn_kpis": citylearn_kpis,
        "metadata": eval_metadata,
        "output_directory": None,
    }

    if save_results and output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        actions_df.to_csv(output_directory / "actions.csv", index=False)
        rewards_df.to_csv(output_directory / "rewards.csv", index=False)
        observations_df.to_csv(output_directory / "observations.csv", index=False)
        physical_state.to_csv(output_directory / "physical_state.csv", index=False)
        kpis.to_csv(output_directory / "kpis.csv", index=False)
        citylearn_kpis.to_csv(output_directory / "citylearn_kpis.csv", index=False)
        (output_directory / "metadata.json").write_text(
            json.dumps(eval_metadata, indent=2),
            encoding="utf-8",
        )
        result["output_directory"] = output_directory

    return result


def select_checkpoint(
    validation_history: pd.DataFrame,
    *,
    discomfort_threshold: float = DEFAULT_DISCOMFORT_THRESHOLD,
) -> dict[str, Any]:
    """Select a checkpoint from validation history only.

    Selection rule
    --------------
    1. Prefer checkpoints with district ``discomfort_proportion`` <= threshold.
    2. Among feasible checkpoints, choose the lowest district electricity
       consumption (kWh).
    3. If none are feasible, report that fact and fall back to the checkpoint
       with the lowest discomfort; ties break by lower electricity.
    """
    if validation_history.empty:
        raise ValueError("validation_history is empty")

    required = {
        "episode",
        "checkpoint_path",
        "district_electricity_consumption_kwh",
        "district_discomfort_proportion",
    }
    missing = required - set(validation_history.columns)
    if missing:
        raise ValueError(f"validation_history missing columns: {sorted(missing)}")

    history = validation_history.copy()
    history["feasible"] = (
        history["district_discomfort_proportion"] <= float(discomfort_threshold)
    )
    feasible = history[history["feasible"]]
    if len(feasible):
        selected = feasible.sort_values(
            by=[
                "district_electricity_consumption_kwh",
                "district_discomfort_proportion",
                "episode",
            ],
            ascending=[True, True, True],
        ).iloc[0]
        rule = "comfort_feasible_then_min_electricity"
        fallback = False
    else:
        selected = history.sort_values(
            by=[
                "district_discomfort_proportion",
                "district_electricity_consumption_kwh",
                "episode",
            ],
            ascending=[True, True, True],
        ).iloc[0]
        rule = "fallback_least_discomfort_then_min_electricity"
        fallback = True

    return {
        "selection_rule": rule,
        "discomfort_threshold": float(discomfort_threshold),
        "no_feasible_checkpoint": bool(fallback),
        "selected_episode": int(selected["episode"]),
        "selected_checkpoint_path": str(selected["checkpoint_path"]),
        "district_electricity_consumption_kwh": float(
            selected["district_electricity_consumption_kwh"]
        ),
        "district_discomfort_proportion": float(
            selected["district_discomfort_proportion"]
        ),
    }


def _district_kpi_value(kpis: pd.DataFrame, metric: str) -> float:
    rows = kpis[(kpis["name"] == "District") & (kpis["cost_function"] == metric)]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0]["value"])


def _train_one_episode(agent: CountingSAC, deterministic: bool = False) -> dict[str, Any]:
    env = agent.env
    observations, _ = env.reset()
    terminated = False
    truncated = False
    reward_rows: list[dict[str, Any]] = []
    local_step = 0

    while not (terminated or truncated):
        actions = agent.predict(observations, deterministic=deterministic)
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        if not deterministic:
            agent.update(
                observations,
                actions,
                rewards,
                next_observations,
                terminated=terminated,
                truncated=truncated,
            )

        components = getattr(env.reward_function, "last_components", None)
        for index, (building, reward) in enumerate(
            zip(env.buildings, np.atleast_1d(rewards))
        ):
            row: dict[str, Any] = {
                "building": building.name,
                "reward": float(reward),
                "time_step_local": local_step,
            }
            if components and index < len(components):
                row.update(
                    {
                        key: float(value)
                        for key, value in components[index].items()
                        if key != "reward"
                    }
                )
            reward_rows.append(row)

        observations = [observation for observation in next_observations]
        local_step += 1

    reward_values = [float(row["reward"]) for row in reward_rows]
    return {
        "transitions": local_step,
        "mean_reward": float(np.mean(reward_values)) if reward_values else 0.0,
        "sum_reward": float(np.sum(reward_values)) if reward_values else 0.0,
        "reward_rows": reward_rows,
        "environment_steps": int(agent.environment_steps),
        "gradient_updates": int(agent.gradient_updates),
        "agent_time_step": int(agent.time_step),
    }


def train_independent_sac(
    schema: str | Path = DEFAULT_SCHEMA,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    episodes: int = DEFAULT_TRAINING_EPISODES,
    random_seed: int = 42,
    reward: RewardChoice = "electricity_discomfort",
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
    standardize_start_time_step: int = DEFAULT_STANDARDIZE_START_TIME_STEP,
    end_exploration_time_step: int = DEFAULT_END_EXPLORATION_TIME_STEP,
    lambda_electricity: float = DEFAULT_LAMBDA_E,
    lambda_discomfort: float = DEFAULT_LAMBDA_D,
    electricity_scale: float = DEFAULT_ELECTRICITY_SCALE,
    discomfort_scale: float = DEFAULT_DISCOMFORT_SCALE,
    comfort_band: float = DEFAULT_COMFORT_BAND,
    lower_exponent: float = DEFAULT_LOWER_EXPONENT,
    higher_exponent: float = DEFAULT_HIGHER_EXPONENT,
    validation_every: int = DEFAULT_VALIDATION_EVERY,
    discomfort_threshold: float = DEFAULT_DISCOMFORT_THRESHOLD,
    include_lstm_warmup: bool = True,
    save_results: bool = True,
    run_test: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train Independent SAC on the train split with validation-based selection."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1")

    def _log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    schema = Path(schema).resolve()
    output_directory = Path(output_directory).resolve()
    seed_directory = output_directory / f"seed_{random_seed}"
    checkpoint_directory = seed_directory / "checkpoints"
    if save_results:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)

    reward_config = _reward_kwargs(
        lambda_electricity,
        lambda_discomfort,
        electricity_scale,
        discomfort_scale,
        comfort_band,
        lower_exponent,
        higher_exponent,
    )

    _log(
        f"[SAC] seed={random_seed} | train episodes={episodes} | "
        f"split={TRAIN_SPLIT.start_time_step}-{TRAIN_SPLIT.end_time_step} "
        f"({TRAIN_SPLIT.n_transitions} transitions/episode)"
    )

    _set_random_seeds(random_seed)
    train_env = make_sac_env(
        schema,
        TRAIN_SPLIT,
        random_seed=random_seed,
        reward=reward,
        reward_kwargs=reward_config,
        include_lstm_warmup=False,
    )
    agent = make_sac_agent(
        train_env,
        hidden_dimension=hidden_dimension,
        discount=discount,
        tau=tau,
        alpha=alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        replay_buffer_capacity=replay_buffer_capacity,
        action_scaling_coefficient=action_scaling_coefficient,
        reward_scaling=reward_scaling,
        update_per_time_step=update_per_time_step,
        standardize_start_time_step=standardize_start_time_step,
        end_exploration_time_step=end_exploration_time_step,
    )

    hyperparameters = {
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
        "standardize_start_time_step": int(agent.standardize_start_time_step),
        "end_exploration_time_step": int(agent.end_exploration_time_step),
    }
    _log(
        f"[SAC] end_exploration={agent.end_exploration_time_step} | "
        f"standardize_start={agent.standardize_start_time_step} | "
        f"reward={type(train_env.reward_function).__name__}"
    )

    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for episode in range(1, episodes + 1):
        _log(f"[SAC] episode {episode}/{episodes} training...")
        episode_result = _train_one_episode(agent, deterministic=False)
        training_rows.append(
            {
                "episode": episode,
                "phase": "training",
                "split": "train",
                "mean_reward": episode_result["mean_reward"],
                "sum_reward": episode_result["sum_reward"],
                "transitions": episode_result["transitions"],
                "environment_steps": episode_result["environment_steps"],
                "gradient_updates": episode_result["gradient_updates"],
                "agent_time_step": episode_result["agent_time_step"],
            }
        )
        _log(
            f"[SAC] episode {episode}/{episodes} done | "
            f"mean_reward={episode_result['mean_reward']:.4f} | "
            f"sum_reward={episode_result['sum_reward']:.2f} | "
            f"env_steps={episode_result['environment_steps']} | "
            f"grad_updates={episode_result['gradient_updates']} | "
            f"agent_t={episode_result['agent_time_step']}"
        )

        metadata = {
            "controller": "citylearn.agents.sac.SAC",
            "control_layout": "one independent policy per building",
            "citylearn_version": citylearn.__version__,
            "torch_version": torch.__version__,
            "schema": str(schema),
            "reward_choice": reward,
            "reward_function": type(train_env.reward_function).__name__,
            "reward_config": reward_config,
            "central_agent": False,
            "random_seed": random_seed,
            "training_episodes_completed": episode,
            "training_episodes_target": episodes,
            "episode_time_steps": train_env.time_steps,
            "train_transitions_per_episode": TRAIN_SPLIT.n_transitions,
            "buildings": [building.name for building in train_env.buildings],
            "observation_names": train_env.observation_names,
            "action_names": train_env.action_names,
            "hyperparameters": hyperparameters,
            "dataset_splits": {name: split.as_dict() for name, split in SPLITS.items()},
            "environment_steps": int(agent.environment_steps),
            "gradient_updates": int(agent.gradient_updates),
            "agent_time_step": int(agent.time_step),
            "checkpoint_type": "inference_only",
            "resumable_training": False,
        }

        checkpoint_path = checkpoint_directory / f"episode_{episode:03d}.pt"
        if save_results:
            save_inference_checkpoint(agent, checkpoint_path, metadata)

        validate_now = (
            episode % validation_every == 0
            or episode == episodes
        )
        if validate_now and save_results:
            _log(f"[SAC] episode {episode}/{episodes} validating on held-out split...")
            checkpoint = load_inference_checkpoint(checkpoint_path)
            validation = evaluate_independent_sac(
                checkpoint,
                schema=schema,
                split=VALIDATION_SPLIT,
                output_directory=seed_directory / "validation" / f"episode_{episode:03d}",
                include_lstm_warmup=include_lstm_warmup,
                save_results=True,
            )
            val_row = {
                "episode": episode,
                "checkpoint_path": str(checkpoint_path),
                "district_electricity_consumption_kwh": _district_kpi_value(
                    validation["kpis"], "electricity_consumption_kwh"
                ),
                "district_discomfort_proportion": _district_kpi_value(
                    validation["kpis"], "discomfort_proportion"
                ),
                "district_carbon_emissions_kgco2": _district_kpi_value(
                    validation["kpis"], "carbon_emissions_kgco2"
                ),
                "district_electricity_cost": _district_kpi_value(
                    validation["kpis"], "electricity_cost"
                ),
                "district_daily_peak_kwh": _district_kpi_value(
                    validation["kpis"], "daily_peak_kwh"
                ),
                "district_ramping_kwh": _district_kpi_value(
                    validation["kpis"], "ramping_kwh"
                ),
            }
            validation_rows.append(val_row)
            _log(
                f"[SAC] validation episode {episode} | "
                f"electricity_kWh={val_row['district_electricity_consumption_kwh']:.3f} | "
                f"discomfort={val_row['district_discomfort_proportion']:.4f}"
            )

    training_rewards = pd.DataFrame(training_rows)
    validation_history = pd.DataFrame(validation_rows)
    if validation_history.empty:
        raise RuntimeError(
            "No validation checkpoints were produced. Enable save_results and "
            "ensure validation_every <= episodes."
        )

    selection = select_checkpoint(
        validation_history,
        discomfort_threshold=discomfort_threshold,
    )
    _log(
        f"[SAC] model selection | episode={selection['selected_episode']} | "
        f"rule={selection['selection_rule']} | "
        f"no_feasible={selection['no_feasible_checkpoint']}"
    )
    selected_checkpoint = load_inference_checkpoint(
        selection["selected_checkpoint_path"]
    )

    test_result = None
    if run_test:
        _log("[SAC] evaluating selected checkpoint on held-out test split...")
        test_result = evaluate_independent_sac(
            selected_checkpoint,
            schema=schema,
            split=TEST_SPLIT,
            output_directory=seed_directory / "test",
            include_lstm_warmup=include_lstm_warmup,
            save_results=save_results,
        )
        _log(
            "[SAC] test complete | "
            f"electricity_kWh={_district_kpi_value(test_result['kpis'], 'electricity_consumption_kwh'):.3f} | "
            f"discomfort={_district_kpi_value(test_result['kpis'], 'discomfort_proportion'):.4f}"
        )
    _log(f"[SAC] finished seed={random_seed} | outputs={seed_directory}")

    run_metadata = {
        "controller": "citylearn.agents.sac.SAC",
        "citylearn_version": citylearn.__version__,
        "torch_version": torch.__version__,
        "schema": str(schema),
        "random_seed": random_seed,
        "reward_choice": reward,
        "reward_config": reward_config,
        "hyperparameters": hyperparameters,
        "dataset_splits": {name: split.as_dict() for name, split in SPLITS.items()},
        "training_episodes": episodes,
        "validation_every": validation_every,
        "model_selection": selection,
        "environment_steps": int(agent.environment_steps),
        "gradient_updates": int(agent.gradient_updates),
        "lstm_lookback": LSTM_LOOKBACK,
        "include_lstm_warmup": include_lstm_warmup,
        "evaluation_note": (
            "Validation and test metrics are held-out. Training-period rewards "
            "are learning curves only and must not be reported as held-out KPIs."
        ),
        "dataset_note": (
            "The local dataset covers June only and does not support "
            "cross-season generalization claims."
        ),
    }

    if save_results:
        seed_directory.mkdir(parents=True, exist_ok=True)
        training_rewards.to_csv(seed_directory / "training_rewards.csv", index=False)
        validation_history.to_csv(
            seed_directory / "validation_history.csv", index=False
        )
        (seed_directory / "model_selection.json").write_text(
            json.dumps(selection, indent=2),
            encoding="utf-8",
        )
        (seed_directory / "metadata.json").write_text(
            json.dumps(run_metadata, indent=2),
            encoding="utf-8",
        )
        final_path = seed_directory / "selected_checkpoint.pt"
        torch.save(selected_checkpoint, final_path)
        if test_result is not None:
            test_result["kpis"].to_csv(seed_directory / "test_kpis.csv", index=False)

    return {
        "agent": agent,
        "env": train_env,
        "training_rewards": training_rewards,
        "validation_history": validation_history,
        "model_selection": selection,
        "selected_checkpoint": selected_checkpoint,
        "test_result": test_result,
        "metadata": run_metadata,
        "output_directory": seed_directory,
    }


def aggregate_seed_results(
    output_directory: str | Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> pd.DataFrame:
    """Aggregate held-out test KPIs across seeds using mean and std."""
    output_directory = Path(output_directory)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        kpi_path = output_directory / f"seed_{seed}" / "test_kpis.csv"
        if not kpi_path.is_file():
            continue
        kpis = pd.read_csv(kpi_path)
        for _, row in kpis.iterrows():
            rows.append(
                {
                    "seed": seed,
                    "name": row["name"],
                    "level": row["level"],
                    "cost_function": row["cost_function"],
                    "value": float(row["value"]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "name",
                "level",
                "cost_function",
                "mean",
                "std",
                "n_seeds",
            ]
        )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["name", "level", "cost_function"], as_index=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n_seeds=("value", "count"))
    )
    summary_path = output_directory / "seed_aggregate_test_kpis.csv"
    summary.to_csv(summary_path, index=False)
    frame.to_csv(output_directory / "seed_individual_test_kpis.csv", index=False)
    return summary


def run_independent_sac(
    schema: str | Path = DEFAULT_SCHEMA,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    episodes: int = DEFAULT_TRAINING_EPISODES,
    random_seed: int = 42,
    reward: RewardChoice = "electricity_discomfort",
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
    **kwargs: Any,
) -> dict[str, Any]:
    """Public entry point used by the notebook and CLI.

    ``simulation_end_time_step`` is accepted only for short smoke tests that
    truncate the training split end. It never reads validation/test data.
    """
    if reward == "independent_sac":
        reward = "electricity_discomfort"

    train_split = TRAIN_SPLIT
    if simulation_end_time_step is not None:
        end = min(int(simulation_end_time_step), TRAIN_SPLIT.end_time_step)
        train_split = TimeSplit(
            "train",
            TRAIN_SPLIT.start_time_step,
            end,
            TRAIN_SPLIT.start_date,
            TRAIN_SPLIT.end_date,
        )

    standardize = (
        DEFAULT_STANDARDIZE_START_TIME_STEP
        if standardize_start_time_step is None
        else int(standardize_start_time_step)
    )
    end_exploration = (
        DEFAULT_END_EXPLORATION_TIME_STEP
        if end_exploration_time_step is None
        else int(end_exploration_time_step)
    )

    # Smoke-test truncation: temporarily monkeypatch TRAIN_SPLIT usage by
    # running a dedicated short path when the end step is shortened.
    if simulation_end_time_step is not None:
        return _run_smoke_training(
            schema=schema,
            output_directory=output_directory,
            episodes=episodes,
            random_seed=random_seed,
            reward=reward,
            hidden_dimension=hidden_dimension,
            discount=discount,
            tau=tau,
            alpha=alpha,
            learning_rate=learning_rate,
            batch_size=batch_size,
            replay_buffer_capacity=replay_buffer_capacity,
            action_scaling_coefficient=action_scaling_coefficient,
            reward_scaling=reward_scaling,
            update_per_time_step=update_per_time_step,
            standardize_start_time_step=standardize,
            end_exploration_time_step=end_exploration,
            train_split=train_split,
            save_results=save_results,
            **kwargs,
        )

    return train_independent_sac(
        schema=schema,
        output_directory=output_directory,
        episodes=episodes,
        random_seed=random_seed,
        reward=reward,
        hidden_dimension=hidden_dimension,
        discount=discount,
        tau=tau,
        alpha=alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        replay_buffer_capacity=replay_buffer_capacity,
        action_scaling_coefficient=action_scaling_coefficient,
        reward_scaling=reward_scaling,
        update_per_time_step=update_per_time_step,
        standardize_start_time_step=standardize,
        end_exploration_time_step=end_exploration,
        save_results=save_results,
        **kwargs,
    )


def _run_smoke_training(
    *,
    schema: str | Path,
    output_directory: str | Path,
    episodes: int,
    random_seed: int,
    reward: RewardChoice,
    hidden_dimension: tuple[int, int],
    discount: float,
    tau: float,
    alpha: float,
    learning_rate: float,
    batch_size: int,
    replay_buffer_capacity: int,
    action_scaling_coefficient: float,
    reward_scaling: float,
    update_per_time_step: int,
    standardize_start_time_step: int,
    end_exploration_time_step: int,
    train_split: TimeSplit,
    save_results: bool,
    lambda_electricity: float = DEFAULT_LAMBDA_E,
    lambda_discomfort: float = DEFAULT_LAMBDA_D,
    electricity_scale: float = DEFAULT_ELECTRICITY_SCALE,
    discomfort_scale: float = DEFAULT_DISCOMFORT_SCALE,
    comfort_band: float = DEFAULT_COMFORT_BAND,
    lower_exponent: float = DEFAULT_LOWER_EXPONENT,
    higher_exponent: float = DEFAULT_HIGHER_EXPONENT,
    include_lstm_warmup: bool = True,
    **_: Any,
) -> dict[str, Any]:
    """Short end-to-end path used only for pipeline smoke tests."""
    schema = Path(schema).resolve()
    output_directory = Path(output_directory).resolve()
    seed_directory = output_directory / f"seed_{random_seed}" / "smoke"
    seed_directory.mkdir(parents=True, exist_ok=True)

    reward_config = _reward_kwargs(
        lambda_electricity,
        lambda_discomfort,
        electricity_scale,
        discomfort_scale,
        comfort_band,
        lower_exponent,
        higher_exponent,
    )
    _set_random_seeds(random_seed)
    env = make_sac_env(
        schema,
        train_split,
        random_seed=random_seed,
        reward=reward,
        reward_kwargs=reward_config,
        include_lstm_warmup=False,
    )
    # Scale exploration thresholds down for truncated smoke episodes.
    episode_transitions = train_split.n_transitions
    end_exploration = min(end_exploration_time_step, max(episode_transitions - 1, 1))
    standardize = min(standardize_start_time_step, max(episode_transitions, 2))
    agent = make_sac_agent(
        env,
        hidden_dimension=hidden_dimension,
        discount=discount,
        tau=tau,
        alpha=alpha,
        learning_rate=learning_rate,
        batch_size=min(batch_size, max(episode_transitions, 8)),
        replay_buffer_capacity=replay_buffer_capacity,
        action_scaling_coefficient=action_scaling_coefficient,
        reward_scaling=reward_scaling,
        update_per_time_step=update_per_time_step,
        standardize_start_time_step=standardize,
        end_exploration_time_step=end_exploration,
    )

    training_rows: list[dict[str, Any]] = []
    for episode in range(1, episodes + 1):
        episode_result = _train_one_episode(agent, deterministic=False)
        training_rows.append(
            {
                "episode": episode,
                "mean_reward": episode_result["mean_reward"],
                "sum_reward": episode_result["sum_reward"],
                "environment_steps": episode_result["environment_steps"],
                "gradient_updates": episode_result["gradient_updates"],
            }
        )

    metadata = {
        "controller": "citylearn.agents.sac.SAC",
        "citylearn_version": citylearn.__version__,
        "torch_version": torch.__version__,
        "schema": str(schema),
        "random_seed": random_seed,
        "reward_choice": reward,
        "reward_config": reward_config,
        "reward_function": type(env.reward_function).__name__,
        "smoke_train_split": train_split.as_dict(),
        "hyperparameters": {
            "hidden_dimension": list(hidden_dimension),
            "discount": discount,
            "tau": tau,
            "alpha": alpha,
            "learning_rate": learning_rate,
            "batch_size": int(agent.batch_size),
            "replay_buffer_capacity": replay_buffer_capacity,
            "action_scaling_coefficient": action_scaling_coefficient,
            "reward_scaling": reward_scaling,
            "update_per_time_step": update_per_time_step,
            "standardize_start_time_step": int(agent.standardize_start_time_step),
            "end_exploration_time_step": int(agent.end_exploration_time_step),
        },
        "observation_names": env.observation_names,
        "action_names": env.action_names,
        "buildings": [building.name for building in env.buildings],
        "environment_steps": int(agent.environment_steps),
        "gradient_updates": int(agent.gradient_updates),
        "dataset_splits": {name: split.as_dict() for name, split in SPLITS.items()},
        "checkpoint_type": "inference_only",
        "resumable_training": False,
    }
    checkpoint_path = seed_directory / "checkpoint.pt"
    checkpoint = save_inference_checkpoint(agent, checkpoint_path, metadata)

    validation = evaluate_independent_sac(
        checkpoint,
        schema=schema,
        split=VALIDATION_SPLIT,
        output_directory=seed_directory / "validation",
        include_lstm_warmup=include_lstm_warmup,
        save_results=save_results,
    )
    test_result = evaluate_independent_sac(
        checkpoint,
        schema=schema,
        split=TEST_SPLIT,
        output_directory=seed_directory / "test",
        include_lstm_warmup=include_lstm_warmup,
        save_results=save_results,
    )

    training_rewards = pd.DataFrame(training_rows)
    if save_results:
        training_rewards.to_csv(seed_directory / "training_rewards.csv", index=False)
        (seed_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    return {
        "agent": agent,
        "env": env,
        "training_rewards": training_rewards,
        "validation_result": validation,
        "test_result": test_result,
        "selected_checkpoint": checkpoint,
        "metadata": metadata,
        "output_directory": seed_directory,
        "checkpoint_path": checkpoint_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate independent CityLearn SAC policies."
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_TRAINING_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of seeds. When set, trains each seed sequentially.",
    )
    parser.add_argument(
        "--reward",
        choices=("electricity_discomfort", "independent_sac", "schema"),
        default="electricity_discomfort",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--validation-every", type=int, default=DEFAULT_VALIDATION_EVERY)
    parser.add_argument(
        "--discomfort-threshold",
        type=float,
        default=DEFAULT_DISCOMFORT_THRESHOLD,
    )
    parser.add_argument(
        "--lambda-electricity",
        type=float,
        default=DEFAULT_LAMBDA_E,
    )
    parser.add_argument(
        "--lambda-discomfort",
        type=float,
        default=DEFAULT_LAMBDA_D,
    )
    parser.add_argument(
        "--simulation-end-time-step",
        type=int,
        default=None,
        help="Optional shorter train end step for smoke tests only.",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip final held-out test evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds) if args.seeds is not None else (args.seed,)

    results = []
    for seed in seeds:
        result = run_independent_sac(
            schema=args.schema,
            output_directory=args.output_directory,
            episodes=args.episodes,
            random_seed=seed,
            reward=args.reward,
            batch_size=args.batch_size,
            update_per_time_step=args.updates_per_step,
            validation_every=args.validation_every,
            discomfort_threshold=args.discomfort_threshold,
            lambda_electricity=args.lambda_electricity,
            lambda_discomfort=args.lambda_discomfort,
            simulation_end_time_step=args.simulation_end_time_step,
            run_test=not args.no_test,
        )
        results.append(result)
        print(f"\nSeed {seed} finished. Outputs: {result['output_directory']}")
        if result.get("test_result") is not None:
            print("Held-out test KPIs (raw scored interval):")
            print(result["test_result"]["kpis"].round(3).to_string(index=False))
        elif result.get("validation_result") is not None:
            print("Validation KPIs (raw scored interval):")
            print(
                result["validation_result"]["kpis"].round(3).to_string(index=False)
            )

    if len(seeds) > 1 and args.simulation_end_time_step is None:
        summary = aggregate_seed_results(args.output_directory, seeds=seeds)
        print("\nSeed aggregate test KPIs:")
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
