"""Shared data capture utilities for CityLearn experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv


INDOOR_TEMPERATURE = "indoor_dry_bulb_temperature"


class ObservationHistoryCityLearnEnv(CityLearnEnv):
    """CityLearn environment that retains the most recent episode observations.

    ``reset`` clears the history, so after an agent completes its deterministic
    final episode this list contains that episode only.  The first item is the
    initial observation and every subsequent item is returned by ``step``.

    ``observe_simulated_temperature=True`` replaces each building's
    ``indoor_dry_bulb_temperature`` observation with the temperature most
    recently simulated by its LSTM model. Stock CityLearn 3.0.2 observes the
    dataset (ideal-load) temperature instead, so agents never see the effect of
    their own cooling, although the reward is computed from the simulated value.
    Only the values returned by ``reset``/``step`` (and the history) change; the
    reward, KPIs and the ``observations`` property are untouched.
    """

    observation_history: list[list[list[float]]]

    def __init__(
        self,
        *args: Any,
        observe_simulated_temperature: bool = False,
        **kwargs: Any,
    ):
        # CityLearnEnv.__init__ calls reset(), so the flag must exist first.
        self.observe_simulated_temperature = bool(observe_simulated_temperature)
        super().__init__(*args, **kwargs)

    def reset(self, *args: Any, **kwargs: Any):
        observations, info = super().reset(*args, **kwargs)
        observations = self._agent_observations(observations)
        self.observation_history = [deepcopy(observations)]
        return observations, info

    def step(self, actions: Any):
        observations, rewards, terminated, truncated, info = super().step(actions)
        observations = self._agent_observations(observations)
        self.observation_history.append(deepcopy(observations))
        return observations, rewards, terminated, truncated, info

    def _agent_observations(self, observations: Any) -> Any:
        if not self.observe_simulated_temperature:
            return observations
        if self.central_agent:
            raise ValueError("observe_simulated_temperature requires central_agent=False.")

        patched = []
        for building, names, values in zip(self.buildings, self.observation_names, observations):
            values = list(values)
            if INDOOR_TEMPERATURE in names:
                time_step = int(building.time_step)
                # Buildings stop advancing on the terminal transition, so their
                # current step has already been simulated there.
                last_simulated = time_step if self.terminated else max(time_step - 1, 0)
                series = building.energy_simulation.indoor_dry_bulb_temperature
                values[names.index(INDOOR_TEMPERATURE)] = float(series[last_simulated])
            patched.append(values)
        return patched


def observations_dataframe(env: ObservationHistoryCityLearnEnv) -> pd.DataFrame:
    """Return the latest episode observations in dashboard-friendly wide form."""

    if env.central_agent:
        raise ValueError(
            "Observation export currently expects central_agent=False so that "
            "each observation vector maps to exactly one building."
        )

    history = getattr(env, "observation_history", None)
    if not history:
        raise ValueError("No observations were recorded. Run an episode first.")

    building_names = [building.name for building in env.buildings]
    records: list[dict[str, Any]] = []

    for time_step, building_observations in enumerate(history):
        if len(building_observations) != len(building_names):
            raise ValueError(
                "Recorded observation count does not match the building count."
            )

        for building_name, names, values in zip(
            building_names,
            env.observation_names,
            building_observations,
        ):
            if len(names) != len(values):
                raise ValueError(
                    f"Observation names and values differ for {building_name} "
                    f"at time step {time_step}."
                )

            record: dict[str, Any] = {
                "time_step": time_step,
                "building": building_name,
            }
            for name, value in zip(names, values):
                if isinstance(value, np.generic):
                    value = value.item()
                record[name] = value
            records.append(record)

    return pd.DataFrame.from_records(records)

