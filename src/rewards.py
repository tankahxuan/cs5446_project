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
    comfort band. The shared fairness term can penalize the worst-case
    discomfort (:math:`\max_i L_i`), inter-building disparity
    (:math:`\max_i L_i - \min_i L_i`), or both (combined):

    .. math::

        F = w_{\text{worst}} \max_i L_i + w_{\text{disp}} (\max_i L_i - \min_i L_i)

    Thus, the decentralized reward is:

    .. math::

        r_i = -w_c C_i - w_h L_i - F

    Parameters
    ----------
    env_metadata:
        CityLearn environment metadata.
    carbon_weight:
        Weight on local operational emissions (:math:`w_c`).
    comfort_weight:
        Weight on local occupied thermal discomfort (:math:`w_h`).
    fairness_weight:
        Base weight on fairness term(s). Used to set ``worst_case_weight``
        and/or ``disparity_weight`` according to ``fairness_mode`` if those
        weights are not explicitly given.
    comfort_band:
        Allowable deviation from setpoint without penalty (:math:`b`).
    worst_case_weight:
        Optional explicit weight on :math:`\max_i L_i`.
    disparity_weight:
        Optional explicit weight on :math:`\max_i L_i - \min_i L_i`.
    fairness_mode:
        One of ``"combined"`` (default, penalizes both), ``"worst_case"``
        (Proposal RQ1), or ``"disparity"`` (Proposal RQ2).
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        carbon_weight: float = 1.0,
        comfort_weight: float = 0.25,
        fairness_weight: float = 0.25,
        comfort_band: float = 1.0,
        worst_case_weight: float | None = None,
        disparity_weight: float | None = None,
        fairness_mode: str = "combined",
    ) -> None:
        super().__init__(env_metadata)
        self.carbon_weight = float(carbon_weight)
        self.comfort_weight = float(comfort_weight)
        self.fairness_weight = float(fairness_weight)
        self.comfort_band = float(comfort_band)
        self.fairness_mode = str(fairness_mode).lower()

        if self.fairness_mode not in ("combined", "worst_case", "disparity"):
            raise ValueError(
                f"Unknown fairness_mode {fairness_mode!r}. "
                "Expected 'combined', 'worst_case', or 'disparity'."
            )

        if worst_case_weight is not None or disparity_weight is not None:
            self.worst_case_weight = float(
                worst_case_weight if worst_case_weight is not None else 0.0
            )
            self.disparity_weight = float(
                disparity_weight if disparity_weight is not None else 0.0
            )
        else:
            if self.fairness_mode == "worst_case":
                self.worst_case_weight = self.fairness_weight
                self.disparity_weight = 0.0
            elif self.fairness_mode == "disparity":
                self.worst_case_weight = 0.0
                self.disparity_weight = self.fairness_weight
            else:  # combined
                self.worst_case_weight = self.fairness_weight
                self.disparity_weight = self.fairness_weight

        for name in (
            "carbon_weight",
            "comfort_weight",
            "fairness_weight",
            "worst_case_weight",
            "disparity_weight",
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
        disparity_loss = worst_loss - best_loss
        fairness_penalty = (
            self.worst_case_weight * worst_loss
            + self.disparity_weight * disparity_loss
        )

        if self.central_agent:
            total_loss = (
                self.carbon_weight * sum(emissions)
                + self.comfort_weight * sum(comfort_losses)
                + fairness_penalty
            )
            return [-total_loss]

        return [
            -(
                self.carbon_weight * emission
                + self.comfort_weight * comfort_loss
                + fairness_penalty
            )
            for emission, comfort_loss in zip(emissions, comfort_losses)
        ]


class WorstCaseFairReward(FairCarbonAwareReward):
    r"""Reward function that specifically penalizes worst-building discomfort.

    Corresponds to the Worst-Case Fair MARL policy in the project proposal.
    Sets ``fairness_mode='worst_case'`` so the shared fairness penalty is:

    .. math::

        F_{\text{worst}} = \max_i L_i
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        carbon_weight: float = 1.0,
        comfort_weight: float = 0.25,
        fairness_weight: float = 0.25,
        comfort_band: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            env_metadata=env_metadata,
            carbon_weight=carbon_weight,
            comfort_weight=comfort_weight,
            fairness_weight=fairness_weight,
            comfort_band=comfort_band,
            fairness_mode="worst_case",
            **kwargs,
        )


class DisparityAwareReward(FairCarbonAwareReward):
    r"""Reward function that specifically penalizes inter-building discomfort disparity.

    Corresponds to the Disparity-Aware MARL policy in the project proposal.
    Sets ``fairness_mode='disparity'`` so the shared fairness penalty is:

    .. math::

        F_{\text{disp}} = \max_i L_i - \min_i L_i
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        carbon_weight: float = 1.0,
        comfort_weight: float = 0.25,
        fairness_weight: float = 0.25,
        comfort_band: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            env_metadata=env_metadata,
            carbon_weight=carbon_weight,
            comfort_weight=comfort_weight,
            fairness_weight=fairness_weight,
            comfort_band=comfort_band,
            fairness_mode="disparity",
            **kwargs,
        )
