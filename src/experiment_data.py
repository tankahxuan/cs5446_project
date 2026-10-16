"""Shared data capture utilities for CityLearn experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv


class ObservationHistoryCityLearnEnv(CityLearnEnv):
    """CityLearn environment that retains the most recent episode observations.

    ``reset`` clears the history, so after an agent completes its deterministic
    final episode this list contains that episode only.  The first item is the
    initial observation and every subsequent item is returned by ``step``.
    """

    observation_history: list[list[list[float]]]

    def reset(self, *args: Any, **kwargs: Any):
        observations, info = super().reset(*args, **kwargs)
        self.observation_history = [deepcopy(observations)]
        return observations, info

    def step(self, actions: Any):
        observations, rewards, terminated, truncated, info = super().step(actions)
        self.observation_history.append(deepcopy(observations))
        return observations, rewards, terminated, truncated, info


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

