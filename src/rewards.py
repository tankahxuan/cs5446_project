"""Custom reward functions for the CityLearn experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from citylearn.reward_function import RewardFunction


class CarbonAwareReward(RewardFunction):
    r"""Penalize operational carbon emissions from grid imports.

    For building :math:`i` at time :math:`t`, the local reward is:

    .. math::

        r_{i,t} = -c_t \max(E_{i,t}, 0)

    where ``c`` is grid carbon intensity and ``E`` is net electricity
    consumption. Grid export receives neither a penalty nor a credit, matching
    CityLearn's carbon-emissions KPI calculation.

    Parameters
    ----------
    env_metadata:
        CityLearn environment metadata.
    district_shared:
        If ``True`` in a decentralized environment, every agent receives the
        same negative district-emissions reward. The default ``False`` keeps
        one local emissions reward per building, preserving the reward layout
        used by the independent SAC baseline.
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        district_shared: bool = False,
    ) -> None:
        super().__init__(env_metadata)
        self.district_shared = bool(district_shared)

    def get_required_observation_names(self) -> Iterable[str]:
        return ("net_electricity_consumption", "carbon_intensity")

    def calculate(self, observations: list[Mapping[str, float]]) -> list[float]:
        emissions = [
            max(float(o["net_electricity_consumption"]), 0.0)
            * max(float(o["carbon_intensity"]), 0.0)
            for o in observations
        ]

        if self.central_agent:
            return [-sum(emissions)]

        if self.district_shared:
            district_reward = -sum(emissions)
            return [district_reward for _ in emissions]

        return [-value for value in emissions]


class FairCarbonAwareReward(RewardFunction):
    r"""Combine carbon, occupied comfort, and cross-building fairness.

    The occupied comfort loss for building :math:`i` is:

    .. math::

        L_i = \mathbb{1}_{occupied}\mathbb{1}_{d_i > 0}(1 + d_i)

    where :math:`d_i = \max(|T_i - T_i^{set}| - b, 0)` and ``b`` is the
    comfort band. The shared fairness term is:

    .. math::

        F = \max_i L_i + (\max_i L_i - \min_i L_i)

    Thus, the decentralized reward is:

    .. math::

        r_i = -w_c C_i - w_h L_i - w_f F

    The first part minimizes local emissions, the second protects each
    building's occupied comfort, and the shared third part penalizes both the
    worst current building and inequality between buildings.
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        carbon_weight: float = 1.0,
        comfort_weight: float = 0.25,
        fairness_weight: float = 0.25,
        comfort_band: float = 1.0,
    ) -> None:
        super().__init__(env_metadata)
        self.carbon_weight = float(carbon_weight)
        self.comfort_weight = float(comfort_weight)
        self.fairness_weight = float(fairness_weight)
        self.comfort_band = float(comfort_band)

        for name in (
            "carbon_weight",
            "comfort_weight",
            "fairness_weight",
            "comfort_band",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def get_required_observation_names(self) -> Iterable[str]:
        return (
            "net_electricity_consumption",
            "carbon_intensity",
            "indoor_dry_bulb_temperature",
            "indoor_dry_bulb_temperature_cooling_set_point",
            "occupant_count",
        )

    def _comfort_loss(self, observation: Mapping[str, float]) -> float:
        occupied = float(observation["occupant_count"]) > 0.0
        if not occupied:
            return 0.0

        indoor_temperature = float(
            observation["indoor_dry_bulb_temperature"]
        )
        set_point = float(
            observation[
                "indoor_dry_bulb_temperature_cooling_set_point"
            ]
        )
        degrees_outside_band = max(
            abs(indoor_temperature - set_point) - self.comfort_band,
            0.0,
        )
        return (
            0.0
            if degrees_outside_band == 0.0
            else 1.0 + degrees_outside_band
        )

    def calculate(self, observations: list[Mapping[str, float]]) -> list[float]:
        emissions = [
            max(float(o["net_electricity_consumption"]), 0.0)
            * max(float(o["carbon_intensity"]), 0.0)
            for o in observations
        ]
        comfort_losses = [self._comfort_loss(o) for o in observations]
        worst_loss = max(comfort_losses, default=0.0)
        best_loss = min(comfort_losses, default=0.0)
        fairness_loss = worst_loss + (worst_loss - best_loss)

        if self.central_agent:
            total_loss = (
                self.carbon_weight * sum(emissions)
                + self.comfort_weight * sum(comfort_losses)
                + self.fairness_weight * fairness_loss
            )
            return [-total_loss]

        return [
            -(
                self.carbon_weight * emission
                + self.comfort_weight * comfort_loss
                + self.fairness_weight * fairness_loss
            )
            for emission, comfort_loss in zip(emissions, comfort_losses)
        ]
