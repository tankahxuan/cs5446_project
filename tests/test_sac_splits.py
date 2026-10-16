"""Verify train/validation/test split lengths against CityLearn 3.0.2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_splits import (
    LSTM_LOOKBACK,
    SPLITS,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from src.run_independent_sac import make_sac_env

SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"


@pytest.mark.parametrize(
    "split,expected_time_steps,expected_transitions",
    [
        (TRAIN_SPLIT, 480, 479),
        (VALIDATION_SPLIT, 120, 119),
        (TEST_SPLIT, 120, 119),
    ],
)
def test_split_transition_counts(split, expected_time_steps, expected_transitions):
    env = make_sac_env(
        SCHEMA,
        split,
        random_seed=0,
        include_lstm_warmup=False,
    )
    assert env.time_steps == expected_time_steps
    assert split.n_time_steps == expected_time_steps
    assert split.n_transitions == expected_transitions

    observations, _ = env.reset()
    transitions = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        actions = [space.sample() for space in env.action_space]
        observations, _, terminated, truncated, _ = env.step(actions)
        transitions += 1

    assert transitions == expected_transitions
    assert env.buildings[0].dynamics.lookback == LSTM_LOOKBACK


def test_lstm_warmup_before_dynamics():
    env = make_sac_env(
        SCHEMA,
        VALIDATION_SPLIT,
        random_seed=0,
        include_lstm_warmup=False,
    )
    env.reset()
    building = env.buildings[0]
    dynamics_flags = []
    for _ in range(LSTM_LOOKBACK + 2):
        actions = [np.zeros(space.shape, dtype=float) for space in env.action_space]
        env.step(actions)
        dynamics_flags.append(bool(building.simulate_dynamics))

    assert dynamics_flags[:LSTM_LOOKBACK] == [False] * LSTM_LOOKBACK
    assert dynamics_flags[LSTM_LOOKBACK] is True


def test_warmup_extended_validation_env():
    env = make_sac_env(
        SCHEMA,
        VALIDATION_SPLIT,
        random_seed=0,
        include_lstm_warmup=True,
    )
    assert env._sac_warmup_steps == LSTM_LOOKBACK
    assert env.episode_tracker.episode_start_time_step == (
        VALIDATION_SPLIT.start_time_step - LSTM_LOOKBACK
    )
    assert env.time_steps == VALIDATION_SPLIT.n_time_steps + LSTM_LOOKBACK
