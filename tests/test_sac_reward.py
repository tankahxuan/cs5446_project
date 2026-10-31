"""Unit tests for the Independent SAC electricity + discomfort reward."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_splits import TRAIN_SPLIT
from src.rewards import CarbonAwareReward, FairCarbonAwareReward
from src.run_independent_sac import (
    DEFAULT_OUTPUT_DIRECTORY,
    default_output_directory,
    make_sac_agent,
    make_sac_env,
    parse_args,
    run_independent_sac,
    select_checkpoint,
)
from src.sac_reward import ElectricityDiscomfortSACReward

SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"


def _obs(**overrides):
    base = {
        "net_electricity_consumption": 0.0,
        "indoor_dry_bulb_temperature": 24.0,
        "indoor_dry_bulb_temperature_cooling_set_point": 24.0,
        "indoor_dry_bulb_temperature_heating_set_point": 24.0,
        "occupant_count": 1.0,
        "hvac_mode": 1,
        "comfort_band": 1.0,
        "heating_demand": 0.0,
        "cooling_demand": 1.0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def reward_fn() -> ElectricityDiscomfortSACReward:
    return ElectricityDiscomfortSACReward(
        env_metadata={"central_agent": False},
        lambda_electricity=1.0,
        lambda_discomfort=1.0,
        electricity_scale=10.0,
        discomfort_scale=4.0,
        band=1.0,
        lower_exponent=2.0,
        higher_exponent=3.0,
    )


def test_zero_electricity_in_band(reward_fn):
    rewards = reward_fn.calculate([_obs()])
    assert rewards == [0.0]
    assert reward_fn.last_components[0]["electricity_penalty"] == 0.0
    assert reward_fn.last_components[0]["discomfort_penalty"] == 0.0


def test_positive_electricity(reward_fn):
    rewards = reward_fn.calculate([_obs(net_electricity_consumption=5.0)])
    assert rewards[0] == pytest.approx(-0.5)
    assert reward_fn.last_components[0]["electricity_penalty"] == pytest.approx(0.5)
    assert reward_fn.last_components[0]["discomfort_penalty"] == pytest.approx(0.0)


def test_negative_net_electricity_no_credit(reward_fn):
    rewards = reward_fn.calculate([_obs(net_electricity_consumption=-3.0)])
    assert rewards == [0.0]
    assert reward_fn.last_components[0]["electricity_positive"] == 0.0


def test_temperature_inside_band(reward_fn):
    rewards = reward_fn.calculate(
        [_obs(indoor_dry_bulb_temperature=24.5)]  # within +/- 1 C
    )
    assert rewards == [0.0]
    assert reward_fn.last_components[0]["discomfort_raw"] == 0.0


def test_mild_and_severe_violations(reward_fn):
    mild = reward_fn.calculate(
        [_obs(indoor_dry_bulb_temperature=25.5)]  # 1.5 C above setpoint
    )[0]
    severe = reward_fn.calculate(
        [_obs(indoor_dry_bulb_temperature=27.0)]  # 3 C above setpoint
    )[0]
    assert mild < 0.0
    assert severe < mild
    assert math.isfinite(mild) and math.isfinite(severe)


def test_unoccupied_ignores_temperature_violation(reward_fn):
    rewards = reward_fn.calculate(
        [
            _obs(
                indoor_dry_bulb_temperature=30.0,
                occupant_count=0.0,
                net_electricity_consumption=0.0,
            )
        ]
    )
    assert rewards == [0.0]
    assert reward_fn.last_components[0]["discomfort_raw"] == 0.0


def test_occupied_penalizes_temperature_violation(reward_fn):
    rewards = reward_fn.calculate(
        [_obs(indoor_dry_bulb_temperature=30.0, occupant_count=2.0)]
    )
    assert rewards[0] < 0.0
    assert reward_fn.last_components[0]["discomfort_raw"] > 0.0


def test_multiple_buildings(reward_fn):
    rewards = reward_fn.calculate(
        [
            _obs(net_electricity_consumption=10.0, indoor_dry_bulb_temperature=24.0),
            _obs(net_electricity_consumption=0.0, indoor_dry_bulb_temperature=27.0),
        ]
    )
    assert len(rewards) == 2
    assert rewards[0] == pytest.approx(-1.0)
    assert rewards[1] < 0.0
    assert rewards[0] != rewards[1]


def test_reward_sign_and_finiteness(reward_fn):
    rewards = reward_fn.calculate(
        [
            _obs(net_electricity_consumption=4.0, indoor_dry_bulb_temperature=26.5),
            _obs(net_electricity_consumption=-1.0, indoor_dry_bulb_temperature=23.0),
        ]
    )
    for reward, component in zip(rewards, reward_fn.last_components):
        assert reward <= 0.0
        assert math.isfinite(reward)
        assert math.isclose(
            reward,
            -(component["electricity_penalty"] + component["discomfort_penalty"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


def test_components_sum_to_total(reward_fn):
    reward_fn.calculate(
        [_obs(net_electricity_consumption=2.5, indoor_dry_bulb_temperature=26.0)]
    )
    component = reward_fn.last_components[0]
    assert component["reward"] == pytest.approx(
        -(component["electricity_penalty"] + component["discomfort_penalty"])
    )


def test_make_sac_env_carbon_reward():
    env = make_sac_env(SCHEMA, TRAIN_SPLIT, random_seed=42, reward="carbon")
    assert isinstance(env.reward_function, CarbonAwareReward)
    assert env.reward_function.district_shared is False


def test_make_sac_env_fair_carbon_modes():
    # Test worst_case mode
    env_wc = make_sac_env(
        SCHEMA,
        TRAIN_SPLIT,
        random_seed=42,
        reward="fair_carbon",
        reward_kwargs={"fairness_mode": "worst_case", "fairness_weight": 0.35},
    )
    assert isinstance(env_wc.reward_function, FairCarbonAwareReward)
    assert env_wc.reward_function.worst_case_weight == pytest.approx(0.35)
    assert env_wc.reward_function.disparity_weight == pytest.approx(0.0)

    # Test disparity mode
    env_disp = make_sac_env(
        SCHEMA,
        TRAIN_SPLIT,
        random_seed=42,
        reward="fair_carbon",
        reward_kwargs={"fairness_mode": "disparity", "fairness_weight": 0.45},
    )
    assert isinstance(env_disp.reward_function, FairCarbonAwareReward)
    assert env_disp.reward_function.worst_case_weight == pytest.approx(0.0)
    assert env_disp.reward_function.disparity_weight == pytest.approx(0.45)

    # Test shortcut aliases
    env_shortcut_wc = make_sac_env(SCHEMA, TRAIN_SPLIT, random_seed=42, reward="worst_case_fair")
    assert env_shortcut_wc.reward_function.worst_case_weight == pytest.approx(0.25)
    assert env_shortcut_wc.reward_function.disparity_weight == pytest.approx(0.0)

    env_shortcut_disp = make_sac_env(SCHEMA, TRAIN_SPLIT, random_seed=42, reward="disparity_fair")
    assert env_shortcut_disp.reward_function.worst_case_weight == pytest.approx(0.0)
    assert env_shortcut_disp.reward_function.disparity_weight == pytest.approx(0.25)


def test_sac_fair_carbon_step_and_gradient_update():
    """Verify that an Independent SAC agent can step and update under FairCarbonAwareReward."""
    env = make_sac_env(
        SCHEMA,
        TRAIN_SPLIT,
        random_seed=42,
        reward="fair_carbon",
        reward_kwargs={"fairness_mode": "combined", "fairness_weight": 0.25},
    )
    agent = make_sac_agent(
        env,
        batch_size=8,
        hidden_dimension=(32, 32),
        standardize_start_time_step=8,
        end_exploration_time_step=10,
    )
    observations, _ = env.reset()

    for _ in range(15):
        actions = agent.predict(observations, deterministic=False)
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        assert len(rewards) == len(env.buildings)
        assert all(math.isfinite(r) for r in rewards)

        agent.update(
            observations,
            actions,
            rewards,
            next_observations,
            terminated=terminated,
            truncated=truncated,
        )
        observations = next_observations

    assert agent.gradient_updates > 0


def _validation_history():
    return pd.DataFrame(
        {
            "episode": [5, 10, 15, 20],
            "checkpoint_path": ["a.pt", "b.pt", "c.pt", "d.pt"],
            "district_electricity_consumption_kwh": [100.0, 200.0, 300.0, 400.0],
            "district_discomfort_proportion": [0.1, 0.1, 0.1, 0.1],
            "district_carbon_emissions_kgco2": [50.0, 10.0, 30.0, 40.0],
            "worst_building_discomfort": [0.5, 0.4, 0.1, 0.3],
            "building_discomfort_gap": [0.4, 0.3, 0.2, 0.05],
        }
    )


@pytest.mark.parametrize(
    "reward, episode, label",
    [
        ("electricity_discomfort", 5, "electricity"),
        ("independent_sac", 5, "electricity"),
        ("schema", 5, "electricity"),
        ("carbon", 10, "carbon"),
        ("fair_carbon", 10, "carbon"),
        ("worst_case_fair", 15, "worst_building_discomfort"),
        ("disparity_fair", 20, "discomfort_gap"),
    ],
)
def test_select_checkpoint_ranks_by_reward_objective(reward, episode, label):
    selection = select_checkpoint(_validation_history(), reward=reward)
    assert selection["selected_episode"] == episode
    assert selection["selection_rule"] == f"comfort_feasible_then_min_{label}"
    assert selection["no_feasible_checkpoint"] is False


def test_select_checkpoint_fallback_breaks_ties_by_objective():
    selection = select_checkpoint(
        _validation_history(), discomfort_threshold=0.05, reward="carbon"
    )
    assert selection["no_feasible_checkpoint"] is True
    assert selection["selected_episode"] == 10
    assert selection["selection_rule"] == "fallback_least_discomfort_then_min_carbon"


def test_select_checkpoint_defaults_to_the_electricity_rule():
    selection = select_checkpoint(_validation_history())
    assert selection["selected_episode"] == 5
    assert selection["selection_metric"] == "district_electricity_consumption_kwh"


def test_carbon_weight_scales_the_carbon_reward():
    observations = [{"net_electricity_consumption": 2.0, "carbon_intensity": 0.5}]
    unweighted = CarbonAwareReward({"central_agent": False})
    doubled = CarbonAwareReward({"central_agent": False}, carbon_weight=2.0)
    assert unweighted.calculate(observations) == pytest.approx([-1.0])
    assert doubled.calculate(observations) == pytest.approx([-2.0])
    with pytest.raises(ValueError, match="carbon_weight"):
        CarbonAwareReward({"central_agent": False}, carbon_weight=-1.0)


def test_make_sac_env_passes_carbon_weight():
    env = make_sac_env(
        SCHEMA,
        TRAIN_SPLIT,
        random_seed=42,
        reward="carbon",
        reward_kwargs={"carbon_weight": 2.0},
    )
    assert env.reward_function.carbon_weight == pytest.approx(2.0)


@pytest.mark.parametrize("key", ["band", "comfort_band"])
def test_either_band_alias_reaches_both_rewards(key):
    fair = make_sac_env(
        SCHEMA, TRAIN_SPLIT, random_seed=42, reward="fair_carbon", reward_kwargs={key: 2.0}
    )
    assert fair.reward_function.comfort_band == pytest.approx(2.0)
    baseline = make_sac_env(SCHEMA, TRAIN_SPLIT, random_seed=42, reward_kwargs={key: 2.0})
    assert baseline.reward_function.band == pytest.approx(2.0)


def test_conflicting_band_aliases_are_rejected():
    with pytest.raises(ValueError, match="conflicts with comfort_band"):
        make_sac_env(
            SCHEMA,
            TRAIN_SPLIT,
            random_seed=42,
            reward_kwargs={"band": 1.0, "comfort_band": 2.0},
        )


def test_baseline_reward_only_receives_its_own_settings():
    env = make_sac_env(
        SCHEMA,
        TRAIN_SPLIT,
        random_seed=42,
        reward_kwargs={"carbon_weight": 9.0, "fairness_mode": "worst_case", "lambda_electricity": 3.0},
    )
    assert isinstance(env.reward_function, ElectricityDiscomfortSACReward)
    assert not hasattr(env.reward_function, "carbon_weight")
    assert not hasattr(env.reward_function, "fairness_mode")


@pytest.mark.parametrize(
    "reward, kwargs",
    [
        ("worst_case_fair", {"fairness_mode": "disparity"}),
        ("disparity_fair", {"fairness_mode": "combined"}),
        ("worst_case_fair", {"disparity_weight": 0.1}),
        ("disparity_fair", {"worst_case_weight": 0.1}),
    ],
)
def test_alias_rewards_reject_contradictory_settings(reward, kwargs):
    with pytest.raises(ValueError):
        make_sac_env(SCHEMA, TRAIN_SPLIT, random_seed=42, reward=reward, reward_kwargs=kwargs)


def test_each_reward_has_its_own_default_output_directory():
    assert default_output_directory("electricity_discomfort") == DEFAULT_OUTPUT_DIRECTORY
    assert default_output_directory("independent_sac") == DEFAULT_OUTPUT_DIRECTORY
    rewards = ("carbon", "fair_carbon", "worst_case_fair", "disparity_fair")
    directories = {default_output_directory(reward) for reward in rewards}
    assert len(directories) == len(rewards)
    assert DEFAULT_OUTPUT_DIRECTORY not in directories


def test_cli_defaults_leave_mode_and_directory_to_the_reward(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_independent_sac.py", "--reward", "worst_case_fair"])
    args = parse_args()
    assert args.fairness_mode is None
    assert args.output_directory is None


@pytest.mark.parametrize(
    "reward, expected_mode",
    [
        ("worst_case_fair", "worst_case"),
        ("disparity_fair", "disparity"),
        ("fair_carbon", "combined"),
    ],
)
def test_run_metadata_records_the_mode_that_trained(tmp_path, reward, expected_mode):
    result = run_independent_sac(
        schema=SCHEMA,
        output_directory=tmp_path,
        episodes=1,
        random_seed=42,
        reward=reward,
        batch_size=8,
        simulation_end_time_step=23,
    )
    assert result["metadata"]["reward_config"]["fairness_mode"] == expected_mode
