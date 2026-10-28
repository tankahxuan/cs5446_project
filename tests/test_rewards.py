"""Tests for project-specific CityLearn reward functions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rewards import (
    CarbonAwareReward,
    DisparityAwareReward,
    FairCarbonAwareReward,
    WorstCaseFairReward,
)


OBSERVATIONS = [
    {"net_electricity_consumption": 2.0, "carbon_intensity": 0.5},
    {"net_electricity_consumption": -1.0, "carbon_intensity": 0.5},
]


def test_carbon_reward_penalizes_import_and_does_not_credit_export() -> None:
    reward = CarbonAwareReward({"central_agent": False})
    assert reward.calculate(OBSERVATIONS) == pytest.approx([-1.0, 0.0])


def test_carbon_reward_can_share_district_emissions() -> None:
    reward = CarbonAwareReward(
        {"central_agent": False}, district_shared=True
    )
    assert reward.calculate(OBSERVATIONS) == pytest.approx([-1.0, -1.0])


def test_carbon_reward_sums_for_a_central_agent() -> None:
    reward = CarbonAwareReward({"central_agent": True})
    assert reward.calculate(OBSERVATIONS) == pytest.approx([-1.0])


FAIR_OBSERVATIONS = [
    {
        "net_electricity_consumption": 2.0,
        "carbon_intensity": 0.5,
        "indoor_dry_bulb_temperature": 26.0,
        "indoor_dry_bulb_temperature_cooling_set_point": 24.0,
        "occupant_count": 2.0,
    },
    {
        "net_electricity_consumption": 1.0,
        "carbon_intensity": 0.5,
        "indoor_dry_bulb_temperature": 24.0,
        "indoor_dry_bulb_temperature_cooling_set_point": 24.0,
        "occupant_count": 1.0,
    },
]


def test_fair_carbon_reward_combines_local_and_shared_losses() -> None:
    reward = FairCarbonAwareReward({"central_agent": False})
    assert reward.calculate(FAIR_OBSERVATIONS) == pytest.approx([-2.5, -1.5])


def test_fair_carbon_reward_ignores_comfort_when_unoccupied() -> None:
    observations = [dict(FAIR_OBSERVATIONS[0], occupant_count=0.0)]
    reward = FairCarbonAwareReward({"central_agent": False})
    assert reward.calculate(observations) == pytest.approx([-1.0])


def test_fair_carbon_reward_applies_shared_term_once_centrally() -> None:
    reward = FairCarbonAwareReward({"central_agent": True})
    assert reward.calculate(FAIR_OBSERVATIONS) == pytest.approx([-3.0])


def test_fair_carbon_reward_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="fairness_weight"):
        FairCarbonAwareReward(
            {"central_agent": False}, fairness_weight=-0.1
        )
    with pytest.raises(ValueError, match="worst_case_weight"):
        FairCarbonAwareReward(
            {"central_agent": False}, worst_case_weight=-0.1
        )
    with pytest.raises(ValueError, match="disparity_weight"):
        FairCarbonAwareReward(
            {"central_agent": False}, disparity_weight=-0.1
        )


def test_fair_carbon_reward_rejects_invalid_fairness_mode() -> None:
    with pytest.raises(ValueError, match="Unknown fairness_mode"):
        FairCarbonAwareReward(
            {"central_agent": False}, fairness_mode="unsupported_mode"
        )


def test_worst_case_fair_reward_penalizes_worst_only() -> None:
    reward = FairCarbonAwareReward(
        {"central_agent": False}, fairness_mode="worst_case"
    )
    # worst_loss=2.0 -> penalty=0.25*2.0=0.5
    # b0: -(1.0 + 0.5 + 0.5) = -2.0; b1: -(0.5 + 0.0 + 0.5) = -1.0
    assert reward.calculate(FAIR_OBSERVATIONS) == pytest.approx([-2.0, -1.0])

    dedicated = WorstCaseFairReward({"central_agent": False})
    assert dedicated.calculate(FAIR_OBSERVATIONS) == pytest.approx([-2.0, -1.0])

    central = WorstCaseFairReward({"central_agent": True})
    assert central.calculate(FAIR_OBSERVATIONS) == pytest.approx([-2.5])


def test_disparity_aware_reward_penalizes_spread_only() -> None:
    reward = FairCarbonAwareReward(
        {"central_agent": False}, fairness_mode="disparity"
    )
    # spread=2.0-0.0=2.0 -> penalty=0.25*2.0=0.5
    assert reward.calculate(FAIR_OBSERVATIONS) == pytest.approx([-2.0, -1.0])

    dedicated = DisparityAwareReward({"central_agent": False})
    assert dedicated.calculate(FAIR_OBSERVATIONS) == pytest.approx([-2.0, -1.0])

    central = DisparityAwareReward({"central_agent": True})
    assert central.calculate(FAIR_OBSERVATIONS) == pytest.approx([-2.5])


def test_disparity_reward_zero_when_comfort_losses_are_equal() -> None:
    # Both buildings have same indoor temp (26.0) -> comfort_loss=2.0 each
    equal_discomfort_observations = [
        dict(FAIR_OBSERVATIONS[0]),
        dict(FAIR_OBSERVATIONS[0], net_electricity_consumption=1.0),
    ]
    # Disparity penalty should be 0 since 2.0 - 2.0 == 0.0
    disparity_reward = DisparityAwareReward({"central_agent": False})
    # b0: -(1.0 + 0.5 + 0.0) = -1.5, b1: -(0.5 + 0.5 + 0.0) = -1.0
    assert disparity_reward.calculate(
        equal_discomfort_observations
    ) == pytest.approx([-1.5, -1.0])

    # WorstCase reward still penalizes the worst (2.0) with penalty 0.5
    worst_case_reward = WorstCaseFairReward({"central_agent": False})
    # b0: -(1.0 + 0.5 + 0.5) = -2.0, b1: -(0.5 + 0.5 + 0.5) = -1.5
    assert worst_case_reward.calculate(
        equal_discomfort_observations
    ) == pytest.approx([-2.0, -1.5])


def test_explicit_fairness_weights_override_mode() -> None:
    reward = FairCarbonAwareReward(
        {"central_agent": False},
        worst_case_weight=0.1,
        disparity_weight=0.2,
    )
    assert reward.worst_case_weight == 0.1
    assert reward.disparity_weight == 0.2
    # penalty = 0.1 * 2.0 + 0.2 * 2.0 = 0.2 + 0.4 = 0.6
    # b0: -(1.0 + 0.5 + 0.6) = -2.1, b1: -(0.5 + 0.0 + 0.6) = -1.1
    assert reward.calculate(FAIR_OBSERVATIONS) == pytest.approx([-2.1, -1.1])



def test_single_explicit_weight_keeps_the_mode_partner() -> None:
    """An explicit weight rescales one term; it must not zero the other."""
    reward = FairCarbonAwareReward(
        {"central_agent": False},
        fairness_mode="combined",
        worst_case_weight=0.5,
    )
    assert reward.worst_case_weight == pytest.approx(0.5)
    assert reward.disparity_weight == pytest.approx(0.25)


def test_weight_contradicting_the_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="switches off worst_case_weight"):
        FairCarbonAwareReward(
            {"central_agent": False},
            fairness_mode="disparity",
            worst_case_weight=0.5,
        )
    with pytest.raises(ValueError, match="switches off disparity_weight"):
        WorstCaseFairReward({"central_agent": False}, disparity_weight=0.5)


def test_zero_fairness_weight_does_not_contradict_combined_mode() -> None:
    """Only the mode switches a term off; fairness_weight only scales it."""
    reward = FairCarbonAwareReward(
        {"central_agent": False},
        fairness_mode="combined",
        fairness_weight=0.0,
        disparity_weight=0.4,
    )
    assert reward.worst_case_weight == pytest.approx(0.0)
    assert reward.disparity_weight == pytest.approx(0.4)
    # Only the disparity term is scaled up: penalty = 0.0 * 2.0 + 0.4 * 2.0.
    # b0: -(1.0 + 0.5 + 0.8) = -2.3, b1: -(0.5 + 0.0 + 0.8) = -1.3
    assert reward.calculate(FAIR_OBSERVATIONS) == pytest.approx([-2.3, -1.3])


def test_zero_weight_matching_the_mode_is_allowed() -> None:
    reward = DisparityAwareReward(
        {"central_agent": False}, worst_case_weight=0.0
    )
    assert reward.worst_case_weight == pytest.approx(0.0)
    assert reward.disparity_weight == pytest.approx(0.25)
