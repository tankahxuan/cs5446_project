"""Unit tests for the Independent SAC electricity + discomfort reward."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_splits import TRAIN_SPLIT
from src.rewards import CarbonAwareReward, FairCarbonAwareReward
from src.run_independent_sac import (
    make_sac_agent,
    make_sac_env,
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
        fairness_mode="worst_case",
        fairness_weight=0.35,
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
        fairness_mode="disparity",
        fairness_weight=0.45,
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
        fairness_mode="combined",
        fairness_weight=0.25,
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
