"""Tests for the WP1 environment fixes (all off by default).

1. Agents observe the last simulated indoor temperature.
2. Validation/test agents reuse the training-window observation encoders.
3. Power outages can be switched off.
4. Periodic validation no longer changes the training trajectory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.run_independent_sac as ris
from src.dataset_splits import TRAIN_SPLIT, TimeSplit
from src.experiment_data import INDOOR_TEMPERATURE, ObservationHistoryCityLearnEnv
from src.observation_scaling import (
    apply_observation_space_limits,
    observation_space_limits,
)
from src.run_marlisa import ComparableMARLISA

SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"
SHORT_END = 47  # 48-step episodes keep these tests fast


class TemperatureProbeReward(RewardFunction):
    """Zero reward that records the indoor temperature the reward function sees."""

    def __init__(self, env_metadata, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.seen: list[list[float]] = []

    def calculate(self, observations):
        self.seen.append([float(o[INDOOR_TEMPERATURE]) for o in observations])
        return [0.0 for _ in observations]


def _short_env(env_class=ObservationHistoryCityLearnEnv, **kwargs):
    return env_class(
        schema=SCHEMA,
        central_agent=False,
        random_seed=42,
        simulation_end_time_step=SHORT_END,
        simulate_power_outage=False,
        **kwargs,
    )


def _cooling_off(env):
    return [[0.0 for _ in names] for names in env.action_names]


def _run_episode(env):
    """Run one cooling-off episode; return observations after reset and each step."""
    observations, _ = env.reset()
    trajectory = [observations]
    terminated = truncated = False
    while not (terminated or truncated):
        observations, _, terminated, truncated, _ = env.step(_cooling_off(env))
        trajectory.append(observations)
    return trajectory


def _indoor(env, trajectory):
    return np.array(
        [
            [values[names.index(INDOOR_TEMPERATURE)] for names, values in zip(env.observation_names, step)]
            for step in trajectory
        ]
    )


def _encoder_signature(encoders):
    return [
        [(type(e).__name__, {k: np.asarray(v).tolist() for k, v in vars(e).items()}) for e in building]
        for building in encoders
    ]


# 1. Observation fix --------------------------------------------------------


def test_flag_off_matches_stock_citylearn():
    stock = _short_env(CityLearnEnv)
    wrapped = _short_env(observe_simulated_temperature=False)
    assert _run_episode(wrapped) == _run_episode(stock)


def test_agent_observes_the_temperature_the_reward_scored():
    env = _short_env(observe_simulated_temperature=True)
    env.reward_function = TemperatureProbeReward(env.get_metadata())
    trajectory = _run_episode(env)
    observed = _indoor(env, trajectory)

    # After every step, including the terminal one, the agent sees the
    # temperature the reward just scored for the action it took.
    seen = np.array(env.reward_function.seen)
    assert len(seen) == len(trajectory) - 1
    assert np.allclose(observed[1:], seen)
    assert env.observation_history == trajectory

    # With cooling off the room heats up, so this differs from stock CityLearn.
    stock = _short_env(CityLearnEnv)
    assert np.abs(observed - _indoor(stock, _run_episode(stock))).max() > 1.0


def test_rewards_unchanged_by_observation_flag():
    rewards = []
    for flag in (False, True):
        env = _short_env(observe_simulated_temperature=flag)
        _run_episode(env)
        rewards.append(np.asarray(env.rewards[1:], dtype=float))
    assert np.array_equal(rewards[0], rewards[1])


def test_observation_flag_rejects_central_agent():
    with pytest.raises(ValueError, match="central_agent=False"):
        ObservationHistoryCityLearnEnv(
            schema=SCHEMA,
            central_agent=True,
            simulation_end_time_step=SHORT_END,
            observe_simulated_temperature=True,
        )


# 2. Training encoders ------------------------------------------------------


@pytest.mark.parametrize(
    "make_agent",
    [ris.make_sac_agent, ComparableMARLISA],
    ids=["sac", "marlisa"],
)
def test_training_limits_rebuild_training_encoders(make_agent):
    train_agent = make_agent(ris.make_sac_env(SCHEMA, "train", random_seed=42))
    test_env = ris.make_sac_env(SCHEMA, "test", random_seed=42, include_lstm_warmup=True)
    test_agent = make_agent(test_env)
    observations, _ = test_env.reset()

    def encoded(agent):
        return [agent.get_encoded_observations(i, o) for i, o in enumerate(observations)]

    # The bug: the same raw observation is encoded differently on the test env.
    assert not all(np.allclose(a, b) for a, b in zip(encoded(train_agent), encoded(test_agent)))

    apply_observation_space_limits(test_agent, observation_space_limits(train_agent))
    assert _encoder_signature(test_agent.encoders) == _encoder_signature(train_agent.encoders)
    assert all(np.allclose(a, b) for a, b in zip(encoded(train_agent), encoded(test_agent)))
    if hasattr(train_agent, "regression_encoders"):
        assert _encoder_signature(test_agent.regression_encoders) == _encoder_signature(
            train_agent.regression_encoders
        )


def test_mismatched_observation_names_are_rejected():
    agent = ris.make_sac_agent(ris.make_sac_env(SCHEMA, "train", random_seed=42))
    limits = observation_space_limits(agent)
    limits[0]["observation_names"] = list(reversed(limits[0]["observation_names"]))
    with pytest.raises(ValueError, match="do not match"):
        apply_observation_space_limits(agent, limits)


# 3. Power outages ----------------------------------------------------------


def test_power_outages_can_be_switched_off():
    assert ris.make_sac_env(SCHEMA, "train", random_seed=42).buildings[0].simulate_power_outage
    env = ris.make_sac_env(SCHEMA, "train", random_seed=42, simulate_power_outage=False)
    assert not any(building.simulate_power_outage for building in env.buildings)


# End-to-end: all fixes through the SAC smoke path ---------------------------


@pytest.fixture(scope="module")
def fixed_smoke(tmp_path_factory):
    return ris.run_independent_sac(
        schema=SCHEMA,
        output_directory=tmp_path_factory.mktemp("sac_fixed"),
        episodes=2,
        random_seed=42,
        batch_size=32,
        simulation_end_time_step=SHORT_END,
        observe_simulated_temperature=True,
        use_training_encoders=True,
        simulate_power_outage=False,
        isolate_validation_rng=True,
    )


def test_checkpoint_records_fixes_and_training_limits(fixed_smoke):
    checkpoint = fixed_smoke["selected_checkpoint"]
    assert checkpoint["metadata"]["pipeline_options"] == {
        "observe_simulated_temperature": True,
        "use_training_encoders": True,
        "simulate_power_outage": False,
        "isolate_validation_rng": True,
    }
    assert checkpoint["observation_space_limits"] == observation_space_limits(fixed_smoke["agent"])


def test_evaluation_follows_checkpoint_options(fixed_smoke):
    for phase in ("validation_result", "test_result"):
        result = fixed_smoke[phase]
        assert result["metadata"]["observe_simulated_temperature"] is True
        assert result["metadata"]["use_training_encoders"] is True
        assert result["metadata"]["simulate_power_outage"] is False
        assert result["env"].observe_simulated_temperature
        assert _encoder_signature(result["agent"].encoders) == _encoder_signature(
            fixed_smoke["agent"].encoders
        )


def test_training_encoders_need_stored_limits(fixed_smoke):
    checkpoint = dict(fixed_smoke["selected_checkpoint"])
    del checkpoint["observation_space_limits"]
    with pytest.raises(ValueError, match="observation_space_limits"):
        ris.evaluate_independent_sac(checkpoint, schema=SCHEMA, split="validation", save_results=False)


# 4. Validation RNG isolation -----------------------------------------------


def test_validation_does_not_change_training_when_isolated(monkeypatch, tmp_path):
    short_train = TimeSplit("train", 0, SHORT_END, TRAIN_SPLIT.start_date, TRAIN_SPLIT.end_date)
    monkeypatch.setattr(ris, "TRAIN_SPLIT", short_train)

    def episode_rewards(validation_every, isolate):
        result = ris.train_independent_sac(
            schema=SCHEMA,
            output_directory=tmp_path / f"every{validation_every}_isolate{isolate}",
            episodes=2,
            random_seed=42,
            batch_size=16,
            # Explore for most of episode 1, then sample from the policy, so
            # episode 2 depends on the global torch/python RNG state.
            end_exploration_time_step=30,
            standardize_start_time_step=31,
            validation_every=validation_every,
            isolate_validation_rng=isolate,
            run_test=False,
            verbose=False,
        )
        return result["training_rewards"]["sum_reward"].tolist()

    # validation_every=2 validates only after the last episode: the reference.
    reference = episode_rewards(2, isolate=False)
    assert episode_rewards(1, isolate=True) == reference

    leaky = episode_rewards(1, isolate=False)
    assert leaky[0] == reference[0]
    assert leaky[1] != reference[1]  # the 16 Oct behaviour this option fixes
