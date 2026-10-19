"""Train and evaluate CityLearn's MARLISA controller.

The experiment follows the same schema, seed, SAC hyperparameters, training
phases, deterministic evaluation, and output layout as the independent SAC
baseline.  Standard MARLISA uses CityLearn's district-coupled ``MARL`` reward
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
    from src.experiment_data import (
        ObservationHistoryCityLearnEnv,
        observations_dataframe,
    )
    from src.observation_scaling import observation_space_limits
    from src.rewards import CarbonAwareReward, FairCarbonAwareReward
    from src.run_independent_sac import (
        _final_action_records,
        _final_reward_records,
        _set_random_seeds,
        _training_reward_records,
    )
except ModuleNotFoundError:
    # Support direct execution: python src/run_marlisa.py
    from experiment_data import (
        ObservationHistoryCityLearnEnv,
        observations_dataframe,
    )
    from observation_scaling import observation_space_limits
    from rewards import CarbonAwareReward, FairCarbonAwareReward
    from run_independent_sac import (
        _final_action_records,
        _final_reward_records,
        _set_random_seeds,
        _training_reward_records,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "marlisa"
RewardChoice = Literal[
    "independent_sac", "carbon", "fair_carbon", "marl", "schema"
]


class ComparableMARLISA(MARLISA):
    """MARLISA with a CityLearn 3.0.2 building-policy indexing correction.

    The upstream method indexes ``policy_net`` using the coordination iteration.
    That causes all buildings to use the same policy in each iteration and leaves
    policies unused when the building count differs from ``iterations``.  This
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

                # CityLearn 3.0.2 uses policy_net[iteration] here.  Each building
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
    comfort_band: float = 1.0,
    simulation_end_time_step: int | None = None,
    save_results: bool = True,
    observe_simulated_temperature: bool = False,
    simulate_power_outage: bool | None = None,
) -> dict[str, Any]:
    """Train MARLISA and evaluate its final deterministic episode.

    Three episodes provide the minimum workflow used by the SAC notebook:
    initial exploration, one learning episode, and deterministic evaluation.

    ``observe_simulated_temperature`` and ``simulate_power_outage=False`` are
    WP1 environment fixes; their defaults keep the existing behaviour.
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
        "observe_simulated_temperature": observe_simulated_temperature,
    }
    if simulation_end_time_step is not None:
        env_kwargs["simulation_end_time_step"] = simulation_end_time_step
    if simulate_power_outage is not None:
        env_kwargs["simulate_power_outage"] = simulate_power_outage

    env = ObservationHistoryCityLearnEnv(**env_kwargs)
    if reward == "independent_sac":
        env.reward_function = IndependentSACReward(env.get_metadata())
    elif reward == "carbon":
        env.reward_function = CarbonAwareReward(
            env.get_metadata(), district_shared=False
        )
    elif reward == "fair_carbon":
        env.reward_function = FairCarbonAwareReward(
            env.get_metadata(),
            carbon_weight=carbon_weight,
            comfort_weight=comfort_weight,
            fairness_weight=fairness_weight,
            comfort_band=comfort_band,
        )
    elif reward == "marl":
        env.reward_function = MARL(env.get_metadata())
    elif reward != "schema":
        raise ValueError(f"Unsupported reward choice: {reward}")

    for index, action_space in enumerate(env.action_space):
        action_space.seed(random_seed + index)

    agent = ComparableMARLISA(
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
        # CityLearn 3.0.2 spells this inherited argument "coefficienct".
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

    training_rewards = pd.DataFrame(_training_reward_records(env, episodes))
    actions = pd.DataFrame(_final_action_records(agent, env))
    rewards = pd.DataFrame(_final_reward_records(env))
    observations = observations_dataframe(env)
    kpis = env.evaluate()
    kpi_table = kpis.pivot(index="cost_function", columns="name", values="value")
    fairness = _fairness_summary(kpis)
    experiment_label = {
        "carbon": "Carbon-aware MARLISA",
        "fair_carbon": "Fair carbon-aware MARLISA",
    }.get(reward, "MARLISA")
    comparison = _comparison_table(kpis, experiment_label)

    metadata = {
        "controller": "src.run_marlisa.ComparableMARLISA",
        "base_controller": "citylearn.agents.marlisa.MARLISA",
        "citylearn_3_0_2_policy_index_correction": True,
        "control_layout": "one coordinated policy per building",
        "citylearn_version": citylearn.__version__,
        "torch_version": torch.__version__,
        "schema": str(schema),
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
                "- w_fairness*(worst_comfort_loss + comfort_loss_gap)"
            ),
        }.get(reward),
        "district_shared_reward": False if reward == "carbon" else None,
        "reward_weights": (
            {
                "carbon": carbon_weight,
                "comfort": comfort_weight,
                "fairness": fairness_weight,
                "comfort_band_celsius": comfort_band,
            }
            if reward == "fair_carbon"
            else None
        ),
        "fairness_definition": (
            "At each step, occupied comfort loss is 0 inside the comfort "
            "band and 1 + degrees outside otherwise. The shared fairness "
            "term is worst loss plus the worst-best loss gap."
            if reward == "fair_carbon"
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
        },
        "comparison_note": (
            "The SAC hyperparameters match the independent SAC runner. Standard "
            "MARLISA uses CityLearn's district-coupled MARL reward and adds "
            "demand estimation and information sharing."
        ),
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
        observations.to_csv(output_directory / "observations.csv", index=False)
        kpis.to_csv(output_directory / "kpis.csv", index=False)
        kpi_table.to_csv(output_directory / "kpi_table.csv")
        fairness.to_csv(output_directory / "fairness_summary.csv", index=False)
        comparison.to_csv(output_directory / "controller_comparison.csv")
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
        "observations": observations,
        "kpis": kpis,
        "kpi_table": kpi_table,
        "fairness": fairness,
        "comparison": comparison,
        "metadata": metadata,
        "output_directory": output_directory,
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
        "--observe-simulated-temperature",
        action="store_true",
        help="Agents observe the last simulated indoor temperature (WP1 fix).",
    )
    parser.add_argument(
        "--no-power-outages",
        action="store_true",
        help="Disable the schema's power outages (WP1 fix).",
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
        observe_simulated_temperature=args.observe_simulated_temperature,
        simulate_power_outage=False if args.no_power_outages else None,
    )
    print("\nMARLISA evaluation (lower is better for these KPIs):")
    print(result["kpi_table"].round(3).to_string())
    print("\nAvailable controller comparison:")
    print(result["comparison"].round(3).to_string())
    print(f"\nSaved results to: {result['output_directory']}")


if __name__ == "__main__":
    main()
