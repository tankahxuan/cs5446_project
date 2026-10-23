"""Train and evaluate CityLearn's MARLISA controller.

The experiment follows the same schema, seed, SAC hyperparameters, training
phases, deterministic evaluation, and output layout as the independent SAC
baseline. Standard MARLISA uses CityLearn's district-coupled ``MARL`` reward
and additionally learns building demand estimators and shares coordination
variables between building policies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import citylearn
import numpy as np
import pandas as pd
import torch
from citylearn.agents.marlisa import MARLISA
from citylearn.reward_function import IndependentSACReward, MARL

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
    from src.observation_scaling import (
        apply_observation_space_limits,
        observation_space_limits,
    )
    from src.rewards import (
        CarbonAwareReward,
        DisparityAwareReward,
        FairCarbonAwareReward,
        WorstCaseFairReward,
    )
    from src.run_independent_sac import (
        _final_action_records,
        _final_reward_records,
        _set_random_seeds,
        _training_reward_records,
        compute_split_kpis,
    )
except ModuleNotFoundError:
    # Support direct execution: python src/run_marlisa.py
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
    from observation_scaling import (
        apply_observation_space_limits,
        observation_space_limits,
    )
    from rewards import (
        CarbonAwareReward,
        DisparityAwareReward,
        FairCarbonAwareReward,
        WorstCaseFairReward,
    )
    from run_independent_sac import (
        _final_action_records,
        _final_reward_records,
        _set_random_seeds,
        _training_reward_records,
        compute_split_kpis,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "marlisa"
RewardChoice = Literal[
    "independent_sac",
    "carbon",
    "fair_carbon",
    "worst_case_fair",
    "disparity_fair",
    "marl",
    "schema",
]


class ComparableMARLISA(MARLISA):
    """MARLISA with a CityLearn 3.0.2 building-policy indexing correction.

    The upstream method indexes ``policy_net`` using the coordination iteration.
    That causes all buildings to use the same policy in each iteration and leaves
    policies unused when the building count differs from ``iterations``. This
    override preserves the upstream calculation and indexes the policy using the
    current building instead.
    """

    def get_post_exploration_prediction_with_information_sharing(
        self,
        observations: list[list[float]],
        deterministic: bool,
    ) -> tuple[list[list[float]], list[list[float]]]:
        agent_count = len(self.action_dimension)
        actions: list[list[float] | None] = [None for _ in range(agent_count)]
        action_order = list(range(agent_count))
        next_agent_indexes = [
            sorted(action_order)[action_order[(index + 1) % agent_count]]
            for index in range(agent_count)
        ]
        coordination_variables = [[0.0, 0.0] for _ in range(agent_count)]
        expected_demand = [0.0 for _ in range(agent_count)]
        total_demand = 0.0

        for iteration in range(self.iterations):
            capacity_dispatched = 0.0

            for building, next_building, observation, raw_observation in zip(
                action_order,
                next_agent_indexes,
                observations,
                observations,
            ):
                encoded = self.get_encoded_observations(building, observation)
                encoded = np.hstack(
                    np.concatenate((encoded, coordination_variables[building]))
                )
                encoded = self.get_normalized_observations(building, encoded)
                encoded = self.pca[building].transform(encoded.reshape(1, -1))[0]
                tensor = torch.FloatTensor(encoded).unsqueeze(0).to(self.device)

                # CityLearn 3.0.2 uses policy_net[iteration] here. Each building
                # has its own trained policy, so the building index is required.
                sampled, _, mean = self.policy_net[building].sample(tensor)
                action_tensor = mean if deterministic else sampled
                action = list(action_tensor.detach().cpu().numpy()[0])
                actions[building] = action
                expected_demand[building] = self.predict_demand(
                    building, raw_observation, action
                )

                if not (
                    iteration == self.iterations - 1
                    and building == action_order[-1]
                ):
                    total_demand += (
                        expected_demand[building]
                        - expected_demand[next_building]
                    )
                    coordination_variables[next_building][0] = (
                        total_demand / self.total_coefficient
                    )

                coordination_variables[building][1] = capacity_dispatched
                capacity_dispatched += self.energy_size_coefficient[building]

        return actions, coordination_variables  # type: ignore[return-value]


def _array_or_none(value: Any) -> list[float] | None:
    """Convert a fitted sklearn value to a JSON/checkpoint-friendly list."""
    return None if value is None else np.asarray(value).tolist()


def _save_checkpoint(
    agent: MARLISA,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    """Save neural networks, normalization, PCA, and demand estimators."""
    estimators = []
    for estimator in agent.state_estimator:
        estimators.append(
            {
                "coef": _array_or_none(getattr(estimator, "coef_", None)),
                "intercept": _array_or_none(
                    getattr(estimator, "intercept_", None)
                ),
            }
        )

    pca_state = []
    for model in agent.pca:
        pca_state.append(
            {
                "components": _array_or_none(getattr(model, "components_", None)),
                "mean": _array_or_none(getattr(model, "mean_", None)),
                "explained_variance": _array_or_none(
                    getattr(model, "explained_variance_", None)
                ),
            }
        )

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
        "demand_estimators": estimators,
        "pca": pca_state,
        "regression_flag": agent.regression_flag,
        "pca_flag": agent.pca_flag,
        "energy_size_coefficient": agent.energy_size_coefficient,
        # Training-window limits, so a held-out evaluation can reuse the encoders.
        "observation_space_limits": observation_space_limits(agent),
    }
    torch.save(checkpoint, path)


def load_marlisa_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a saved MARLISA checkpoint dictionary."""
    checkpoint = torch.load(
        Path(path).resolve(), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dict checkpoint, got {type(checkpoint)}")
    return checkpoint


def apply_checkpoint_to_marlisa(
    agent: MARLISA,
    checkpoint: dict[str, Any],
) -> MARLISA:
    """Restore actor policies, observation normalization, PCA, and demand estimators."""
    for network, state in zip(agent.policy_net, checkpoint["policy_net"]):
        network.load_state_dict(state)

    normalization = checkpoint.get("normalization", {})
    if "observation_mean" in normalization:
        agent.norm_mean = normalization["observation_mean"]
        agent.norm_std = normalization["observation_std"]
    if "reward_mean" in normalization:
        agent.r_norm_mean = normalization["reward_mean"]
        agent.r_norm_std = normalization["reward_std"]

    for model, state in zip(agent.pca, checkpoint.get("pca", [])):
        if state.get("components") is not None:
            model.components_ = np.array(state["components"])
            model.mean_ = np.array(state["mean"])
            model.explained_variance_ = np.array(state["explained_variance"])

    for est, state in zip(
        agent.state_estimator, checkpoint.get("demand_estimators", [])
    ):
        if state.get("coef") is not None:
            est.coef_ = np.array(state["coef"])
            est.intercept_ = np.array(state["intercept"])

    if "regression_flag" in checkpoint:
        agent.regression_flag = checkpoint["regression_flag"]
    if "pca_flag" in checkpoint:
        agent.pca_flag = checkpoint["pca_flag"]
    if "energy_size_coefficient" in checkpoint:
        agent.energy_size_coefficient = checkpoint["energy_size_coefficient"]

    return agent


def _fairness_summary(kpis: pd.DataFrame) -> pd.DataFrame:
    """Summarize the spread in building discomfort for the final episode."""
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


def _comparison_table(
    current_kpis: pd.DataFrame,
    current_label: str,
) -> pd.DataFrame:
    """Combine available district KPI results using a shared table format."""
    controller_paths = {
        "BasicRBC": PROJECT_ROOT / "outputs" / "rbc" / "kpis.csv",
        "Independent SAC": (
            PROJECT_ROOT / "outputs" / "independent_sac" / "kpis.csv"
        ),
        "MARLISA": PROJECT_ROOT / "outputs" / "marlisa" / "kpis.csv",
        "Carbon-aware MARLISA": (
            PROJECT_ROOT / "outputs" / "carbon_aware_marlisa" / "kpis.csv"
        ),
        "Fair carbon-aware MARLISA": (
            PROJECT_ROOT / "outputs" / "fair_carbon_aware_marlisa" / "kpis.csv"
        ),
        "Worst-case fair MARLISA": (
            PROJECT_ROOT / "outputs" / "worst_case_fair_marlisa" / "kpis.csv"
        ),
        "Disparity-aware MARLISA": (
            PROJECT_ROOT / "outputs" / "disparity_aware_marlisa" / "kpis.csv"
        ),
    }
    series: dict[str, pd.Series] = {}

    for controller, path in controller_paths.items():
        if path.is_file():
            table = pd.read_csv(path)
            series[controller] = (
                table.loc[table["level"] == "district"]
                .set_index("cost_function")["value"]
            )

    series[current_label] = (
        current_kpis.loc[current_kpis["level"] == "district"]
        .set_index("cost_function")["value"]
    )
    return pd.concat(series, axis=1).sort_index()


def make_marlisa_env(
    schema: str | Path = DEFAULT_SCHEMA,
    split: TimeSplit | str | None = None,
    *,
    random_seed: int = 42,
    reward: RewardChoice = "marl",
    carbon_weight: float = 1.0,
    comfort_weight: float = 0.25,
    fairness_weight: float = 0.25,
    worst_case_weight: float | None = None,
    disparity_weight: float | None = None,
    fairness_mode: Literal["combined", "worst_case", "disparity"] = "combined",
    comfort_band: float = 1.0,
    simulation_start_time_step: int | None = None,
    simulation_end_time_step: int | None = None,
    observe_simulated_temperature: bool = False,
    simulate_power_outage: bool | None = None,
) -> ObservationHistoryCityLearnEnv:
    """Create and configure a CityLearn environment for MARLISA."""
    schema = Path(schema).resolve()
    if not schema.is_file():
        raise FileNotFoundError(f"CityLearn schema was not found: {schema}")

    start_step = simulation_start_time_step
    end_step = simulation_end_time_step

    if split is not None and split != "all":
        split_obj = get_split(split) if isinstance(split, str) else split
        if start_step is None:
            start_step = split_obj.start_time_step
        if end_step is None:
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

    if reward == "independent_sac":
        env.reward_function = IndependentSACReward(env.get_metadata())
    elif reward == "carbon":
        env.reward_function = CarbonAwareReward(
            env.get_metadata(), district_shared=False
        )
    elif reward in ("fair_carbon", "worst_case_fair", "disparity_fair"):
        resolved_mode = (
            "worst_case"
            if reward == "worst_case_fair"
            else "disparity"
            if reward == "disparity_fair"
            else fairness_mode
        )
        env.reward_function = FairCarbonAwareReward(
            env.get_metadata(),
            carbon_weight=carbon_weight,
            comfort_weight=comfort_weight,
            fairness_weight=fairness_weight,
            worst_case_weight=worst_case_weight,
            disparity_weight=disparity_weight,
            fairness_mode=resolved_mode,
            comfort_band=comfort_band,
        )
    elif reward == "marl":
        env.reward_function = MARL(env.get_metadata())
    elif reward != "schema":
        raise ValueError(f"Unsupported reward choice: {reward}")

    for index, action_space in enumerate(env.action_space):
        action_space.seed(random_seed + index)

    return env


def evaluate_marlisa(
    agent_or_checkpoint: ComparableMARLISA | MARLISA | dict[str, Any] | str | Path,
    *,
    schema: str | Path = DEFAULT_SCHEMA,
    split: TimeSplit | str = "test",
    output_directory: str | Path | None = None,
    random_seed: int = 42,
    reward: RewardChoice = "marl",
    carbon_weight: float = 1.0,
    comfort_weight: float = 0.25,
    fairness_weight: float = 0.25,
    worst_case_weight: float | None = None,
    disparity_weight: float | None = None,
    fairness_mode: Literal["combined", "worst_case", "disparity"] = "combined",
    comfort_band: float = 1.0,
    save_results: bool = True,
    observe_simulated_temperature: bool = False,
    use_training_encoders: bool = False,
    simulate_power_outage: bool | None = None,
    pipeline_fixes: bool = False,
    experiment_label: str | None = None,
) -> dict[str, Any]:
    """Deterministically evaluate a MARLISA policy on a specific dataset split."""
    if pipeline_fixes:
        observe_simulated_temperature = True
        use_training_encoders = True
        if simulate_power_outage is None:
            simulate_power_outage = False

    split_obj = get_split(split) if isinstance(split, str) else split

    env = make_marlisa_env(
        schema=schema,
        split=split_obj,
        random_seed=random_seed,
        reward=reward,
        carbon_weight=carbon_weight,
        comfort_weight=comfort_weight,
        fairness_weight=fairness_weight,
        worst_case_weight=worst_case_weight,
        disparity_weight=disparity_weight,
        fairness_mode=fairness_mode,
        comfort_band=comfort_band,
        observe_simulated_temperature=observe_simulated_temperature,
        simulate_power_outage=simulate_power_outage,
    )

    if isinstance(agent_or_checkpoint, (str, Path, dict)):
        checkpoint = (
            agent_or_checkpoint
            if isinstance(agent_or_checkpoint, dict)
            else load_marlisa_checkpoint(agent_or_checkpoint)
        )
        agent = ComparableMARLISA(
            env,
            hidden_dimension=checkpoint.get("metadata", {}).get(
                "sac_hyperparameters", {}
            ).get("hidden_dimension", [256, 256]),
        )
        apply_checkpoint_to_marlisa(agent, checkpoint)
        if use_training_encoders:
            limits = checkpoint.get("observation_space_limits")
            if limits is not None:
                apply_observation_space_limits(agent, limits)
    else:
        agent = agent_or_checkpoint
        if use_training_encoders:
            limits = getattr(agent, "_training_observation_space_limits", None)
            if limits is not None:
                apply_observation_space_limits(agent, limits)

    _set_random_seeds(random_seed)
    observations, _ = env.reset(seed=random_seed)
    terminated = False
    truncated = False
    local_step = 0
    action_records: list[dict[str, Any]] = []
    reward_records: list[dict[str, Any]] = []

    while not (terminated or truncated):
        actions = agent.predict(observations, deterministic=True)
        next_observations, rewards, terminated, truncated, _ = env.step(actions)

        for building, names, values in zip(env.buildings, env.action_names, actions):
            for name, value in zip(names, values):
                action_records.append(
                    {
                        "time_step_local": local_step,
                        "time_step_absolute": split_obj.start_time_step + local_step,
                        "building": building.name,
                        "action": name,
                        "value": float(value),
                    }
                )

        for building, reward_val in zip(env.buildings, np.atleast_1d(rewards)):
            reward_records.append(
                {
                    "time_step_local": local_step,
                    "time_step_absolute": split_obj.start_time_step + local_step,
                    "building": building.name,
                    "reward": float(reward_val),
                }
            )

        observations = next_observations
        local_step += 1

    kpis = env.evaluate()
    kpi_table = kpis.pivot(index="cost_function", columns="name", values="value")
    fairness = _fairness_summary(kpis)

    if experiment_label is None:
        experiment_label = {
            "carbon": "Carbon-aware MARLISA",
            "fair_carbon": (
                "Worst-case fair MARLISA"
                if fairness_mode == "worst_case"
                else "Disparity-aware MARLISA"
                if fairness_mode == "disparity"
                else "Fair carbon-aware MARLISA"
            ),
            "worst_case_fair": "Worst-case fair MARLISA",
            "disparity_fair": "Disparity-aware MARLISA",
        }.get(reward, "MARLISA")

    comparison = _comparison_table(kpis, f"{experiment_label} ({split_obj.name})")
    actions_df = pd.DataFrame(action_records)
    rewards_df = pd.DataFrame(reward_records)
    observations_df = observations_dataframe(env)

    if save_results and output_directory is not None:
        out_dir = Path(output_directory).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        kpis.to_csv(out_dir / f"{split_obj.name}_kpis.csv", index=False)
        kpi_table.to_csv(out_dir / f"{split_obj.name}_kpi_table.csv")
        fairness.to_csv(out_dir / f"{split_obj.name}_fairness_summary.csv", index=False)
        actions_df.to_csv(out_dir / f"{split_obj.name}_actions.csv", index=False)
        rewards_df.to_csv(out_dir / f"{split_obj.name}_rewards.csv", index=False)
        observations_df.to_csv(
            out_dir / f"{split_obj.name}_observations.csv", index=False
        )

    return {
        "env": env,
        "split": split_obj.name,
        "actions": actions_df,
        "rewards": rewards_df,
        "observations": observations_df,
        "kpis": kpis,
        "kpi_table": kpi_table,
        "fairness": fairness,
        "comparison": comparison,
    }


def run_marlisa(
    schema: str | Path = DEFAULT_SCHEMA,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    episodes: int = 3,
    random_seed: int = 42,
    reward: RewardChoice = "marl",
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
    regression_buffer_capacity: int = 30_000,
    start_regression_time_step: int = 2,
    regression_frequency: int = 2_500,
    information_sharing: bool = True,
    pca_compression: float = 1.0,
    iterations: int = 2,
    carbon_weight: float = 1.0,
    comfort_weight: float = 0.25,
    fairness_weight: float = 0.25,
    worst_case_weight: float | None = None,
    disparity_weight: float | None = None,
    fairness_mode: Literal["combined", "worst_case", "disparity"] = "combined",
    comfort_band: float = 1.0,
    simulation_end_time_step: int | None = None,
    save_results: bool = True,
    observe_simulated_temperature: bool = False,
    simulate_power_outage: bool | None = None,
    use_training_encoders: bool = False,
    pipeline_fixes: bool = False,
    split: TimeSplit | str = "all",
    run_test: bool | None = None,
) -> dict[str, Any]:
    """Train MARLISA and evaluate its performance.

    When ``split='all'`` (default), trains and evaluates on the complete month.
    When ``split='train'``, trains on June 1–20 and deterministically evaluates
    on the held-out test split (June 26–30).

    ``pipeline_fixes`` enables observation of simulated temperatures,
    training encoder reuse on held-out intervals, and disables power outages.
    """
    if episodes < 3:
        raise ValueError(
            "episodes must be at least 3: exploration, training, and evaluation"
        )

    if pipeline_fixes:
        observe_simulated_temperature = True
        use_training_encoders = True
        if simulate_power_outage is None:
            simulate_power_outage = False

    schema = Path(schema).resolve()
    output_directory = Path(output_directory).resolve()
    if not schema.is_file():
        raise FileNotFoundError(f"CityLearn schema was not found: {schema}")

    is_full_dataset = split == "all" or split is None
    active_split = (
        None
        if is_full_dataset
        else (get_split(split) if isinstance(split, str) else split)
    )
    is_train_split = active_split is not None and active_split.name == "train"

    if run_test is None:
        run_test = is_train_split

    _set_random_seeds(random_seed)

    env = make_marlisa_env(
        schema=schema,
        split=active_split,
        random_seed=random_seed,
        reward=reward,
        carbon_weight=carbon_weight,
        comfort_weight=comfort_weight,
        fairness_weight=fairness_weight,
        worst_case_weight=worst_case_weight,
        disparity_weight=disparity_weight,
        fairness_mode=fairness_mode,
        comfort_band=comfort_band,
        simulation_end_time_step=simulation_end_time_step,
        observe_simulated_temperature=observe_simulated_temperature,
        simulate_power_outage=simulate_power_outage,
    )

    # Adapt exploration, standardization, and batch size for truncated smoke episodes:
    episode_transitions = env.time_steps - 1
    if simulation_end_time_step is not None:
        actual_standardize = (
            standardize_start_time_step
            if standardize_start_time_step is not None
            else max(episode_transitions - 1, 2)
        )
        actual_end_exploration = (
            end_exploration_time_step
            if end_exploration_time_step is not None
            else episode_transitions
        )
        actual_batch_size = (
            min(batch_size, max(episode_transitions - 4, 8))
            if episode_transitions < batch_size
            else batch_size
        )
    else:
        actual_standardize = standardize_start_time_step
        actual_end_exploration = end_exploration_time_step
        actual_batch_size = batch_size

    agent = ComparableMARLISA(
        env,
        hidden_dimension=list(hidden_dimension),
        discount=discount,
        tau=tau,
        alpha=alpha,
        lr=learning_rate,
        batch_size=actual_batch_size,
        replay_buffer_capacity=replay_buffer_capacity,
        standardize_start_time_step=actual_standardize,
        end_exploration_time_step=actual_end_exploration,
        action_scaling_coefficienct=action_scaling_coefficient,
        reward_scaling=reward_scaling,
        update_per_time_step=update_per_time_step,
        regression_buffer_capacity=regression_buffer_capacity,
        start_regression_time_step=start_regression_time_step,
        regression_frequency=regression_frequency,
        information_sharing=information_sharing,
        pca_compression=pca_compression,
        iterations=iterations,
    )
    agent.learn(
        episodes=episodes,
        deterministic_finish=True,
        logging_level=30,
    )

    # Store training limits for possible downstream held-out evaluation.
    training_limits = observation_space_limits(agent)
    agent._training_observation_space_limits = training_limits

    training_rewards = pd.DataFrame(_training_reward_records(env, episodes))
    actions = pd.DataFrame(_final_action_records(agent, env))
    rewards = pd.DataFrame(_final_reward_records(env))
    observations = observations_dataframe(env)
    kpis = env.evaluate()
    kpi_table = kpis.pivot(index="cost_function", columns="name", values="value")
    fairness = _fairness_summary(kpis)

    experiment_label = {
        "carbon": "Carbon-aware MARLISA",
        "fair_carbon": (
            "Worst-case fair MARLISA"
            if fairness_mode == "worst_case"
            else "Disparity-aware MARLISA"
            if fairness_mode == "disparity"
            else "Fair carbon-aware MARLISA"
        ),
        "worst_case_fair": "Worst-case fair MARLISA",
        "disparity_fair": "Disparity-aware MARLISA",
    }.get(reward, "MARLISA")

    comp_label = (
        f"{experiment_label} ({active_split.name})"
        if active_split is not None
        else experiment_label
    )
    comparison = _comparison_table(kpis, comp_label)

    metadata = {
        "controller": "src.run_marlisa.ComparableMARLISA",
        "base_controller": "citylearn.agents.marlisa.MARLISA",
        "citylearn_3_0_2_policy_index_correction": True,
        "control_layout": "one coordinated policy per building",
        "citylearn_version": citylearn.__version__,
        "torch_version": torch.__version__,
        "schema": str(schema),
        "split": active_split.name if active_split is not None else "all",
        "reward_choice": reward,
        "reward_function": type(env.reward_function).__name__,
        "reward_formula": {
            "marl": (
                "-sign(net_electricity_consumption_i) * 0.01 * "
                "net_electricity_consumption_i**2 * "
                "max(district_net_electricity_consumption, 0)"
            ),
            "carbon": (
                "-max(net_electricity_consumption, 0) * carbon_intensity"
            ),
            "fair_carbon": (
                "-w_carbon*emissions_i - w_comfort*comfort_loss_i "
                "- w_worst*worst_loss - w_disp*disparity_loss"
            ),
            "worst_case_fair": (
                "-w_carbon*emissions_i - w_comfort*comfort_loss_i - w_fairness*worst_loss"
            ),
            "disparity_fair": (
                "-w_carbon*emissions_i - w_comfort*comfort_loss_i - w_fairness*disparity_loss"
            ),
        }.get(reward),
        "district_shared_reward": False if reward == "carbon" else None,
        "reward_weights": (
            {
                "carbon": carbon_weight,
                "comfort": comfort_weight,
                "fairness": fairness_weight,
                "worst_case": worst_case_weight,
                "disparity": disparity_weight,
                "fairness_mode": fairness_mode,
                "comfort_band_celsius": comfort_band,
            }
            if reward in ("fair_carbon", "worst_case_fair", "disparity_fair")
            else None
        ),
        "central_agent": False,
        "random_seed": random_seed,
        "episodes": episodes,
        "final_episode_is_deterministic": True,
        "episode_time_steps": env.time_steps,
        "observation_rows": len(observations),
        "observation_columns": list(observations.columns),
        "buildings": [building.name for building in env.buildings],
        "observation_names": env.observation_names,
        "action_names": env.action_names,
        "sac_hyperparameters": {
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
        "marlisa_hyperparameters": {
            "regression_buffer_capacity": regression_buffer_capacity,
            "start_regression_time_step": start_regression_time_step,
            "regression_frequency": regression_frequency,
            "information_sharing": information_sharing,
            "pca_compression": pca_compression,
            "iterations": iterations,
        },
        "pipeline_options": {
            "observe_simulated_temperature": bool(env.observe_simulated_temperature),
            "simulate_power_outage": bool(env.buildings[0].simulate_power_outage),
            "use_training_encoders": bool(use_training_encoders),
            "pipeline_fixes": bool(pipeline_fixes),
        },
        "has_held_out_test_evaluation": bool(run_test),
    }

    if save_results:
        output_directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{active_split.name}_" if active_split is not None else ""
        training_rewards.to_csv(
            output_directory / f"{prefix}training_rewards.csv", index=False
        )
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
        comparison.to_csv(
            output_directory / f"{prefix}controller_comparison.csv"
        )
        (output_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        _save_checkpoint(agent, output_directory / "checkpoint.pt", metadata)

    test_result = None
    if run_test:
        test_result = evaluate_marlisa(
            agent,
            schema=schema,
            split=TEST_SPLIT,
            output_directory=output_directory,
            random_seed=random_seed,
            reward=reward,
            carbon_weight=carbon_weight,
            comfort_weight=comfort_weight,
            fairness_weight=fairness_weight,
            worst_case_weight=worst_case_weight,
            disparity_weight=disparity_weight,
            fairness_mode=fairness_mode,
            comfort_band=comfort_band,
            save_results=save_results,
            observe_simulated_temperature=observe_simulated_temperature,
            use_training_encoders=use_training_encoders,
            simulate_power_outage=simulate_power_outage,
            experiment_label=experiment_label,
        )

    return {
        "env": env,
        "agent": agent,
        "training_rewards": training_rewards,
        "actions": actions,
        "rewards": rewards,
        "observations": observations,
        "kpis": kpis,
        "kpi_table": kpi_table,
        "fairness": fairness,
        "comparison": comparison,
        "metadata": metadata,
        "output_directory": output_directory,
        "test_result": test_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate CityLearn MARLISA."
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reward",
        choices=(
            "independent_sac",
            "carbon",
            "fair_carbon",
            "worst_case_fair",
            "disparity_fair",
            "marl",
            "schema",
        ),
        default="marl",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument(
        "--simulation-end-time-step",
        type=int,
        default=None,
        help="Optional shorter end step, useful only for smoke tests.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "test", "validation"),
        default="all",
        help="Dataset split to train on (default 'all').",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip held-out test split evaluation when training on the train split.",
    )
    parser.add_argument(
        "--fairness-mode",
        choices=("combined", "worst_case", "disparity"),
        default="combined",
        help="Fairness formulation for fair_carbon reward.",
    )
    parser.add_argument("--worst-case-weight", type=float, default=None)
    parser.add_argument("--disparity-weight", type=float, default=None)
    parser.add_argument("--carbon-weight", type=float, default=1.0)
    parser.add_argument("--comfort-weight", type=float, default=0.25)
    parser.add_argument("--fairness-weight", type=float, default=0.25)
    parser.add_argument("--comfort-band", type=float, default=1.0)
    parser.add_argument(
        "--observe-simulated-temperature",
        action="store_true",
        help="Agents observe the last simulated indoor temperature (WP1 fix).",
    )
    parser.add_argument(
        "--training-encoders",
        action="store_true",
        help="Scale validation/test inputs with the training-window limits.",
    )
    parser.add_argument(
        "--no-power-outages",
        action="store_true",
        help="Disable the schema's power outages (WP1 fix).",
    )
    parser.add_argument(
        "--pipeline-fixes",
        action="store_true",
        help="Turn on all environment and observation fixes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_marlisa(
        schema=args.schema,
        output_directory=args.output_directory,
        episodes=args.episodes,
        random_seed=args.seed,
        reward=args.reward,
        batch_size=args.batch_size,
        update_per_time_step=args.updates_per_step,
        simulation_end_time_step=args.simulation_end_time_step,
        split=args.split,
        run_test=not args.no_test if args.no_test else None,
        fairness_mode=args.fairness_mode,
        worst_case_weight=args.worst_case_weight,
        disparity_weight=args.disparity_weight,
        carbon_weight=args.carbon_weight,
        comfort_weight=args.comfort_weight,
        fairness_weight=args.fairness_weight,
        comfort_band=args.comfort_band,
        observe_simulated_temperature=args.observe_simulated_temperature,
        use_training_encoders=args.training_encoders,
        simulate_power_outage=False if args.no_power_outages else None,
        pipeline_fixes=args.pipeline_fixes,
    )
    print("\nMARLISA evaluation (lower is better for these KPIs):")
    print(result["kpi_table"].round(3).to_string())
    if result.get("test_result") is not None:
        print("\nHeld-out test KPIs:")
        print(result["test_result"]["kpi_table"].round(3).to_string())
    print("\nAvailable controller comparison:")
    print(result["comparison"].round(3).to_string())
    print(f"\nSaved results to: {result['output_directory']}")


if __name__ == "__main__":
    main()
