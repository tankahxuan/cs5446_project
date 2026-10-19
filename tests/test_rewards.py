"""Tests for project-specific CityLearn reward functions."""

import pytest

from src.rewards import CarbonAwareReward, FairCarbonAwareReward


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
