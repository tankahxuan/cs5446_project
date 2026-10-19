"""Diagnostic env wrapper: agents observe the last *simulated* indoor temperature.

CityLearn 3.0.2 shows agents the dataset (ideal-load) indoor temperature, not the
temperature its LSTM simulated for their own actions. This wrapper replaces that
observation with the most recent simulated value. It is the 18 Oct prototype;
the project version belongs in ``src/experiment_data.py`` (implementation plan WP1.1).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.experiment_data import ObservationHistoryCityLearnEnv  # noqa: E402


class LastSimulatedTemperatureEnv(ObservationHistoryCityLearnEnv):
    """Expose the most recent simulated indoor temperature to the agent."""

    def _patch(self, observations):
        patched = []
        for b, names, values in zip(self.buildings, self.observation_names, observations):
            values = list(values)
            ix = names.index("indoor_dry_bulb_temperature")
            series = b.energy_simulation.indoor_dry_bulb_temperature
            t = int(b.time_step)
            # Building time stops advancing on the terminal transition, so the
            # current index is already simulated there.
            last = t if self.terminated else max(t - 1, 0)
            values[ix] = float(series[last])
            patched.append(values)
        return patched

    def reset(self, *args, **kwargs):
        observations, info = super().reset(*args, **kwargs)
        observations = self._patch(observations)
        self.observation_history[-1] = observations
        return observations, info

    def step(self, actions):
        observations, rewards, terminated, truncated, info = super().step(actions)
        observations = self._patch(observations)
        self.observation_history[-1] = observations
        return observations, rewards, terminated, truncated, info
