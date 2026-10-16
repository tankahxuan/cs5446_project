"""Electricity + thermal-discomfort reward for the Independent SAC baseline.

This reward is intentionally free of carbon, pricing, fairness, and district
coupling terms. Those objectives are reserved for later MARLISA experiments.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from citylearn.reward_function import RewardFunction


# Fixed scales so electricity and discomfort terms have comparable, documented
# magnitudes. Electricity is hourly building net import (kWh). Discomfort uses
# ComfortReward-style delta**exponent magnitudes; a 2 C cooling violation with
# lower_exponent=2 contributes 4 before scaling.
DEFAULT_ELECTRICITY_SCALE = 10.0
DEFAULT_DISCOMFORT_SCALE = 4.0
DEFAULT_LAMBDA_E = 1.0
DEFAULT_LAMBDA_D = 1.0
DEFAULT_COMFORT_BAND = 1.0
DEFAULT_LOWER_EXPONENT = 2.0
DEFAULT_HIGHER_EXPONENT = 3.0


class ElectricityDiscomfortSACReward(RewardFunction):
    r"""Per-building Independent SAC reward.

    For building :math:`i` at time :math:`t`:

    .. math::

        r_i(t) = -\lambda_E \hat{E}_i^+(t) - \lambda_D D_i(t)

    where

    .. math::

        \hat{E}_i^+(t) = \max(E_i(t), 0) / E_{\mathrm{scale}}

    and :math:`D_i(t)` is an occupancy-gated comfort violation aligned with
    CityLearn's discomfort KPI band logic and the schema ``ComfortReward``
    exponents (``band=1.0``, ``lower_exponent=2.0``, ``higher_exponent=3.0``).

    Occupied comfort violations outside the band are penalized with the same
    asymmetric exponents as ``ComfortReward``. Unoccupied steps and in-band
    temperatures contribute zero discomfort, matching the KPI's occupied /
    outside-band definition more closely than ComfortReward's soft in-band
    penalties.

    Parameters
    ----------
    env_metadata:
        CityLearn environment metadata.
    lambda_electricity / lambda_e:
        Weight on normalized positive electricity import.
    lambda_discomfort / lambda_d:
        Weight on normalized discomfort.
    electricity_scale:
        Positive scale for electricity normalization (kWh).
    discomfort_scale:
        Positive scale for discomfort normalization.
    band:
        Comfort band (+/- C) around the relevant setpoints.
    lower_exponent:
        Exponent for cooling-mode overheating / heating-mode undercooling
        violations (ComfortReward ``lower_exponent``).
    higher_exponent:
        Exponent for cooling-mode undercooling / heating-mode overheating
        violations (ComfortReward ``higher_exponent``).
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        lambda_electricity: float | None = None,
        lambda_discomfort: float | None = None,
        lambda_e: float | None = None,
        lambda_d: float | None = None,
        electricity_scale: float = DEFAULT_ELECTRICITY_SCALE,
        discomfort_scale: float = DEFAULT_DISCOMFORT_SCALE,
        band: float = DEFAULT_COMFORT_BAND,
        lower_exponent: float = DEFAULT_LOWER_EXPONENT,
        higher_exponent: float = DEFAULT_HIGHER_EXPONENT,
        **kwargs: Any,
    ) -> None:
        # CityLearn may forward ComfortReward schema kwargs; accept and ignore
        # unknown extras after reading the documented comfort parameters.
        band = float(kwargs.pop("band", band))
        lower_exponent = float(kwargs.pop("lower_exponent", lower_exponent))
        higher_exponent = float(kwargs.pop("higher_exponent", higher_exponent))
        super().__init__(env_metadata, **kwargs)

        if lambda_electricity is None:
            lambda_electricity = (
                DEFAULT_LAMBDA_E if lambda_e is None else float(lambda_e)
            )
        if lambda_discomfort is None:
            lambda_discomfort = (
                DEFAULT_LAMBDA_D if lambda_d is None else float(lambda_d)
            )

        self.lambda_electricity = float(lambda_electricity)
        self.lambda_discomfort = float(lambda_discomfort)
        self.electricity_scale = float(electricity_scale)
        self.discomfort_scale = float(discomfort_scale)
        self.band = float(band)
        self.lower_exponent = float(lower_exponent)
        self.higher_exponent = float(higher_exponent)
        self.last_components: list[dict[str, float]] = []

        for name in (
            "lambda_electricity",
            "lambda_discomfort",
            "electricity_scale",
            "discomfort_scale",
            "band",
            "lower_exponent",
            "higher_exponent",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.electricity_scale <= 0.0:
            raise ValueError("electricity_scale must be positive")
        if self.discomfort_scale <= 0.0:
            raise ValueError("discomfort_scale must be positive")

    @property
    def lambda_e(self) -> float:
        return self.lambda_electricity

    @property
    def lambda_d(self) -> float:
        return self.lambda_discomfort

    def get_required_observation_names(self) -> Iterable[str]:
        return (
            "net_electricity_consumption",
            "indoor_dry_bulb_temperature",
            "indoor_dry_bulb_temperature_cooling_set_point",
            "indoor_dry_bulb_temperature_heating_set_point",
            "occupant_count",
            "hvac_mode",
            "comfort_band",
            "heating_demand",
            "cooling_demand",
        )

    def config_dict(self) -> dict[str, float]:
        return {
            "lambda_electricity": self.lambda_electricity,
            "lambda_discomfort": self.lambda_discomfort,
            "electricity_scale": self.electricity_scale,
            "discomfort_scale": self.discomfort_scale,
            "band": self.band,
            "lower_exponent": self.lower_exponent,
            "higher_exponent": self.higher_exponent,
        }

    def _comfort_band(self, observation: Mapping[str, float]) -> float:
        if self.band is not None:
            return float(self.band)
        return float(observation.get("comfort_band", DEFAULT_COMFORT_BAND))

    def _discomfort(self, observation: Mapping[str, float]) -> float:
        """Return non-negative discomfort magnitude before scaling."""
        if float(observation.get("occupant_count", 0.0)) <= 0.0:
            return 0.0

        temperature = float(observation["indoor_dry_bulb_temperature"])
        cooling_set_point = float(
            observation["indoor_dry_bulb_temperature_cooling_set_point"]
        )
        heating_set_point = float(
            observation["indoor_dry_bulb_temperature_heating_set_point"]
        )
        band = self._comfort_band(observation)
        heating_demand = float(observation.get("heating_demand", 0.0))
        cooling_demand = float(observation.get("cooling_demand", 0.0))
        heating = heating_demand > cooling_demand
        hvac_mode = int(observation.get("hvac_mode", 1))

        # KPI-aligned outside-band detection using cooling/heating setpoints.
        too_hot = temperature > cooling_set_point + band
        too_cold = temperature < heating_set_point - band
        if not too_hot and not too_cold:
            return 0.0

        if hvac_mode in (1, 2):
            set_point = (
                cooling_set_point if hvac_mode == 1 else heating_set_point
            )
            delta = abs(temperature - set_point)
            if temperature < set_point - band:
                exponent = (
                    self.lower_exponent if hvac_mode == 2 else self.higher_exponent
                )
            else:
                # temperature > set_point + band
                exponent = (
                    self.higher_exponent if heating else self.lower_exponent
                )
            return float(delta**exponent)

        # Auto / off: ComfortReward dual-setpoint branch.
        if too_cold:
            delta = abs(temperature - heating_set_point)
            exponent = (
                self.higher_exponent if not heating else self.lower_exponent
            )
            return float(delta**exponent)

        delta = abs(temperature - cooling_set_point)
        exponent = self.higher_exponent if heating else self.lower_exponent
        return float(delta**exponent)

    def calculate(self, observations: list[Mapping[str, float]]) -> list[float]:
        rewards: list[float] = []
        components: list[dict[str, float]] = []

        for observation in observations:
            electricity = float(observation["net_electricity_consumption"])
            electricity_positive = max(electricity, 0.0)
            electricity_hat = electricity_positive / self.electricity_scale
            discomfort_raw = self._discomfort(observation)
            discomfort_hat = discomfort_raw / self.discomfort_scale

            electricity_penalty = self.lambda_electricity * electricity_hat
            discomfort_penalty = self.lambda_discomfort * discomfort_hat
            total = -(electricity_penalty + discomfort_penalty)

            rewards.append(float(total))
            components.append(
                {
                    "net_electricity_consumption": electricity,
                    "electricity_positive": electricity_positive,
                    "electricity_hat": electricity_hat,
                    "discomfort_raw": discomfort_raw,
                    "discomfort_hat": discomfort_hat,
                    "electricity_penalty": electricity_penalty,
                    "discomfort_penalty": discomfort_penalty,
                    "reward": float(total),
                }
            )

        if self.central_agent:
            combined = {
                "net_electricity_consumption": sum(
                    c["net_electricity_consumption"] for c in components
                ),
                "electricity_positive": sum(
                    c["electricity_positive"] for c in components
                ),
                "electricity_hat": sum(c["electricity_hat"] for c in components),
                "discomfort_raw": sum(c["discomfort_raw"] for c in components),
                "discomfort_hat": sum(c["discomfort_hat"] for c in components),
                "electricity_penalty": sum(
                    c["electricity_penalty"] for c in components
                ),
                "discomfort_penalty": sum(
                    c["discomfort_penalty"] for c in components
                ),
                "reward": float(sum(rewards)),
            }
            self.last_components = [combined]
            return [float(sum(rewards))]

        self.last_components = components
        return rewards

    def reset(self) -> None:
        self.last_components = []
