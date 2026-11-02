"""Tests for MARLISA dataset splits, decoupled fairness rewards, and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_splits import (
    SPLITS,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from src.rewards import FairCarbonAwareReward
from src.run_marlisa import (
    ComparableMARLISA,
    evaluate_marlisa,
    make_marlisa_env,
    run_marlisa,
)

SCHEMA = PROJECT_ROOT / "notebooks" / "data" / "schema.json"


@pytest.mark.parametrize(
    "split_name,expected_steps",
    [
        ("train", 480),
        ("validation", 120),
        ("test", 120),
        ("all", 720),
    ],
)
def test_make_marlisa_env_splits(split_name: str, expected_steps: int) -> None:
    env = make_marlisa_env(
        schema=SCHEMA,
        split=split_name,
        random_seed=42,
    )
    assert env.time_steps == expected_steps


def test_make_marlisa_env_decoupled_fairness_rewards() -> None:
    # Test worst_case mode
    env_wc = make_marlisa_env(
        schema=SCHEMA,
        reward="fair_carbon",
        fairness_mode="worst_case",
        fairness_weight=0.3,
    )
    assert isinstance(env_wc.reward_function, FairCarbonAwareReward)
    assert env_wc.reward_function.worst_case_weight == pytest.approx(0.3)
    assert env_wc.reward_function.disparity_weight == pytest.approx(0.0)

    # Test disparity mode
    env_disp = make_marlisa_env(
        schema=SCHEMA,
        reward="fair_carbon",
        fairness_mode="disparity",
        fairness_weight=0.4,
    )
    assert isinstance(env_disp.reward_function, FairCarbonAwareReward)
    assert env_disp.reward_function.worst_case_weight == pytest.approx(0.0)
    assert env_disp.reward_function.disparity_weight == pytest.approx(0.4)

    # Test shortcut reward names
    env_shortcut_wc = make_marlisa_env(schema=SCHEMA, reward="worst_case_fair")
    assert env_shortcut_wc.reward_function.worst_case_weight == pytest.approx(0.25)
    assert env_shortcut_wc.reward_function.disparity_weight == pytest.approx(0.0)

    env_shortcut_disp = make_marlisa_env(schema=SCHEMA, reward="disparity_fair")
    assert env_shortcut_disp.reward_function.worst_case_weight == pytest.approx(0.0)
    assert env_shortcut_disp.reward_function.disparity_weight == pytest.approx(0.25)


def test_marlisa_smoke_train_and_test(tmp_path: Path) -> None:
    # Run a fast 3-episode smoke training with truncated end step
    result = run_marlisa(
        schema=SCHEMA,
        output_directory=tmp_path / "marlisa_smoke",
        episodes=3,
        random_seed=42,
        batch_size=32,
        reward="fair_carbon",
        fairness_mode="worst_case",
        split="train",
        simulation_end_time_step=47,  # truncated train split for speed
        run_test=False,
        save_results=True,
    )
    assert result["metadata"]["split"] == "train"
    assert result["metadata"]["reward_choice"] == "fair_carbon"
    assert result["metadata"]["reward_weights"]["fairness_mode"] == "worst_case"
    assert (tmp_path / "marlisa_smoke" / "train_kpis.csv").is_file()
    assert (tmp_path / "marlisa_smoke" / "checkpoint.pt").is_file()


def test_evaluation_env_warms_up_but_scores_only_the_split() -> None:
    """Held-out MARLISA evaluation must match the SAC runner's warm-up."""
    from src.dataset_splits import LSTM_LOOKBACK

    cold = make_marlisa_env(schema=SCHEMA, split=TEST_SPLIT)
    warm = make_marlisa_env(schema=SCHEMA, split=TEST_SPLIT, include_lstm_warmup=True)

    assert cold._marlisa_episode_start == TEST_SPLIT.start_time_step
    assert warm._marlisa_episode_start == TEST_SPLIT.start_time_step - LSTM_LOOKBACK
    # Both score the same interval; only the simulated run-up differs.
    assert cold._marlisa_scored_start == warm._marlisa_scored_start
    assert warm._marlisa_scored_start == TEST_SPLIT.start_time_step
    assert warm.time_steps == cold.time_steps + LSTM_LOOKBACK


def test_train_split_is_not_warmed_up() -> None:
    """The train split starts at step 0, so no earlier data exists."""
    env = make_marlisa_env(schema=SCHEMA, split=TRAIN_SPLIT, include_lstm_warmup=True)
    assert env._marlisa_episode_start == TRAIN_SPLIT.start_time_step == 0


def test_reward_metadata_records_the_objective_that_ran(tmp_path: Path) -> None:
    import json

    result = run_marlisa(
        schema=SCHEMA,
        output_directory=tmp_path / "disparity",
        episodes=3,
        random_seed=42,
        batch_size=32,
        reward="disparity_fair",
        split="train",
        simulation_end_time_step=47,
        run_test=False,
        save_results=True,
    )
    weights = json.loads(
        (tmp_path / "disparity" / "metadata.json").read_text(encoding="utf-8")
    )["reward_weights"]

    assert weights["fairness_mode"] == "disparity"
    assert weights["worst_case"] == pytest.approx(0.0)
    assert weights["disparity"] == pytest.approx(0.25)
    assert "disparity_loss" in result["metadata"]["reward_formula"]
    assert "worst_loss" not in result["metadata"]["reward_formula"]


def test_make_marlisa_env_passes_carbon_weight():
    env = make_marlisa_env(SCHEMA, TRAIN_SPLIT, reward="carbon", carbon_weight=2.0)
    assert env.reward_function.carbon_weight == pytest.approx(2.0)


def test_make_marlisa_env_alias_uses_its_own_mode_by_default():
    env = make_marlisa_env(SCHEMA, TRAIN_SPLIT, reward="worst_case_fair")
    assert env.reward_function.fairness_mode == "worst_case"


def test_make_marlisa_env_alias_rejects_a_different_mode():
    with pytest.raises(ValueError, match="always uses"):
        make_marlisa_env(
            SCHEMA, TRAIN_SPLIT, reward="worst_case_fair", fairness_mode="disparity"
        )
