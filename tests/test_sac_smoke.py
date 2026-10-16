"""End-to-end Independent SAC smoke checks before full training."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.run_independent_sac import (
    evaluate_independent_sac,
    load_inference_checkpoint,
    run_independent_sac,
)

SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"
SMOKE_OUTPUT = PROJECT_ROOT / "outputs" / "independent_sac_smoke"


@pytest.fixture(scope="module")
def smoke_result():
    return run_independent_sac(
        schema=SCHEMA,
        output_directory=SMOKE_OUTPUT,
        episodes=2,
        random_seed=42,
        batch_size=32,
        update_per_time_step=1,
        simulation_end_time_step=47,  # truncated train split for speed
        save_results=True,
        include_lstm_warmup=True,
    )


def test_smoke_training_produces_checkpoint(smoke_result):
    checkpoint_path = Path(smoke_result["checkpoint_path"])
    assert checkpoint_path.is_file()
    checkpoint = load_inference_checkpoint(checkpoint_path)
    assert checkpoint["checkpoint_type"] == "inference_only"
    assert checkpoint["resumable_training"] is False
    assert len(checkpoint["policy_net"]) == 3
    assert smoke_result["metadata"]["environment_steps"] > 0


def test_checkpoint_reload_and_deterministic_reproducibility(smoke_result):
    checkpoint_path = smoke_result["checkpoint_path"]
    first = evaluate_independent_sac(
        checkpoint_path,
        schema=SCHEMA,
        split="validation",
        output_directory=SMOKE_OUTPUT / "seed_42" / "smoke" / "validation_repeat_a",
        include_lstm_warmup=True,
        save_results=True,
    )
    second = evaluate_independent_sac(
        checkpoint_path,
        schema=SCHEMA,
        split="validation",
        output_directory=SMOKE_OUTPUT / "seed_42" / "smoke" / "validation_repeat_b",
        include_lstm_warmup=True,
        save_results=True,
    )

    assert first["actions"]["value"].to_numpy().tolist() == second["actions"][
        "value"
    ].to_numpy().tolist()
    assert np.allclose(
        first["kpis"]["value"].to_numpy(dtype=float),
        second["kpis"]["value"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_no_optimizer_updates_during_evaluation(smoke_result):
    result = smoke_result["validation_result"]
    assert result["metadata"]["optimizer_updates_during_eval"] == 0
    agent = result["agent"]
    for optimizer in agent.policy_optimizer:
        assert all(state.get("step", 0) == 0 for state in optimizer.state.values())


def test_reward_components_sum_to_total(smoke_result):
    rewards = smoke_result["validation_result"]["rewards"]
    assert {
        "electricity_penalty",
        "discomfort_penalty",
        "reward",
    }.issubset(rewards.columns)
    reconstructed = -(
        rewards["electricity_penalty"] + rewards["discomfort_penalty"]
    )
    assert np.allclose(rewards["reward"], reconstructed, atol=1e-10)


def test_logged_actions_are_finite(smoke_result):
    actions = smoke_result["validation_result"]["actions"]
    assert actions["value"].notna().all()
    assert np.isfinite(actions["value"].to_numpy(dtype=float)).all()


def test_kpis_finite_or_explained(smoke_result):
    kpis = smoke_result["test_result"]["kpis"]
    for _, row in kpis.iterrows():
        value = float(row["value"])
        if math.isnan(value):
            # Empty occupant windows can yield NaN discomfort; electricity should not.
            assert "discomfort" in row["cost_function"]
        else:
            assert math.isfinite(value)
