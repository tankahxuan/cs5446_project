"""Custom reward functions for the CityLearn experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from citylearn.reward_function import RewardFunction

DEFAULT_FAIRNESS_MODE = "combined"

#: Which fairness terms each mode switches on, as multipliers of
#: ``fairness_weight``: (worst-case term, disparity term).
FAIRNESS_MODE_MULTIPLIERS: dict[str, tuple[float, float]] = {
    "combined": (1.0, 1.0),
    "worst_case": (1.0, 0.0),
    "disparity": (0.0, 1.0),
}

DEFAULT_CARBON_WEIGHT = 1.0
DEFAULT_COMFORT_WEIGHT = 0.25
DEFAULT_FAIRNESS_WEIGHT = 0.25

#: Reward names that fix the fairness mode, and the mode each one fixes.
FORCED_FAIRNESS_MODE: dict[str, str] = {
    "worst_case_fair": "worst_case",
    "disparity_fair": "disparity",
}

#: Config keys that ``build_carbon_reward`` forwards to ``FairCarbonAwareReward``.
FAIR_REWARD_KEYS = (
    "carbon_weight",
    "comfort_weight",
    "fairness_weight",
    "worst_case_weight",
    "disparity_weight",
    "comfort_band",
)


class CarbonAwareReward(RewardFunction):
    r"""Penalize operational carbon emissions from grid imports.

    For building :math:`i` at time :math:`t`, the local reward is:

    .. math::

        r_{i,t} = -w_c c_t \max(E_{i,t}, 0)

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
    carbon_weight:
        Weight on operational emissions (:math:`w_c`).
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        district_shared: bool = False,
        carbon_weight: float = DEFAULT_CARBON_WEIGHT,
    ) -> None:
        super().__init__(env_metadata)
        self.district_shared = bool(district_shared)
        self.carbon_weight = float(carbon_weight)
        if self.carbon_weight < 0.0:
            raise ValueError("carbon_weight must be non-negative")

    def get_required_observation_names(self) -> Iterable[str]:
        return ("net_electricity_consumption", "carbon_intensity")

    def calculate(self, observations: list[Mapping[str, float]]) -> list[float]:
        emissions = [
            self.carbon_weight
            * max(float(o["net_electricity_consumption"]), 0.0)
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
        carbon_weight: float = DEFAULT_CARBON_WEIGHT,
        comfort_weight: float = DEFAULT_COMFORT_WEIGHT,
        fairness_weight: float = DEFAULT_FAIRNESS_WEIGHT,
        comfort_band: float = 1.0,
        worst_case_weight: float | None = None,
        disparity_weight: float | None = None,
        fairness_mode: str | None = None,
    ) -> None:
        super().__init__(env_metadata)
        self.carbon_weight = float(carbon_weight)
        self.comfort_weight = float(comfort_weight)
        self.fairness_weight = float(fairness_weight)
        self.comfort_band = float(comfort_band)

        mode_was_given = fairness_mode is not None
        self.fairness_mode = str(
            DEFAULT_FAIRNESS_MODE if fairness_mode is None else fairness_mode
        ).lower()

        if self.fairness_mode not in FAIRNESS_MODE_MULTIPLIERS:
            raise ValueError(
                f"Unknown fairness_mode {fairness_mode!r}. "
                "Expected 'combined', 'worst_case', or 'disparity'."
            )

        # The mode fixes which fairness terms are active; an explicit weight
        # only rescales a term the mode already switches on. A weight that
        # contradicts the mode is a mistake, not an override, so it is
        # rejected rather than silently zeroed.
        multipliers = dict(
            zip(
                ("worst_case_weight", "disparity_weight"),
                FAIRNESS_MODE_MULTIPLIERS[self.fairness_mode],
            )
        )
        given = {
            "worst_case_weight": worst_case_weight,
            "disparity_weight": disparity_weight,
        }

        for name, value in given.items():
            multiplier = multipliers[name]
            # Only the multiplier says whether the mode switched a term off.
            # A zero ``fairness_weight`` is just the default scale, so an
            # explicit weight is still free to raise a term off the floor.
            if (
                mode_was_given
                and value is not None
                and multiplier == 0.0
                and float(value) != 0.0
            ):
                raise ValueError(
                    f"fairness_mode={self.fairness_mode!r} switches off "
                    f"{name}, but {name}={value!r} was given. Pass a "
                    "different fairness_mode, or drop the weight."
                )
            setattr(
                self,
                name,
                float(
                    multiplier * self.fairness_weight
                    if value is None
                    else value
                ),
            )

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
        carbon_weight: float = DEFAULT_CARBON_WEIGHT,
        comfort_weight: float = DEFAULT_COMFORT_WEIGHT,
        fairness_weight: float = DEFAULT_FAIRNESS_WEIGHT,
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
        carbon_weight: float = DEFAULT_CARBON_WEIGHT,
        comfort_weight: float = DEFAULT_COMFORT_WEIGHT,
        fairness_weight: float = DEFAULT_FAIRNESS_WEIGHT,
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


def resolve_fairness_mode(reward: str, fairness_mode: str | None = None) -> str:
    """Return the fairness mode a reward name actually trains with.

    ``worst_case_fair`` and ``disparity_fair`` fix their own mode, so asking for
    a different one is an error rather than something to ignore.
    """
    forced = FORCED_FAIRNESS_MODE.get(reward)
    if forced is not None and fairness_mode not in (None, forced):
        raise ValueError(
            f"reward={reward!r} always uses fairness_mode={forced!r}, "
            f"but fairness_mode={fairness_mode!r} was given."
        )
    return forced or fairness_mode or DEFAULT_FAIRNESS_MODE


def build_carbon_reward(
    reward: str,
    env_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> RewardFunction:
    """Build a carbon or fairness reward from its name and config.

    Shared by the SAC and MARLISA runners so a reward name means the same thing
    for both algorithms.
    """
    if reward == "carbon":
        return CarbonAwareReward(
            env_metadata,
            district_shared=False,
            carbon_weight=config.get("carbon_weight", DEFAULT_CARBON_WEIGHT),
        )
    return FairCarbonAwareReward(
        env_metadata,
        fairness_mode=resolve_fairness_mode(reward, config.get("fairness_mode")),
        **{key: config[key] for key in FAIR_REWARD_KEYS if key in config},
    )
