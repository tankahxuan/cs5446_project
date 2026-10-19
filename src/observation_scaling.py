"""Reuse training-window observation scaling when an agent is evaluated elsewhere.

CityLearn 3.0.2 derives each building's observation-space limits from the
simulation window the env covers (``Building._get_observation_space_limits_data``
in ``citylearn/building.py``), and its RL agents build their input encoders from
those limits when they are created (``RLC.set_encoders``). An agent created on a
validation, test or other-dataset env therefore rescales its inputs differently
from the agent that was trained. These helpers record the training limits so an
evaluation agent can rebuild its encoders from them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


def observation_space_limits(agent: Any) -> list[dict[str, list]]:
    """Return each building's observation names and low/high limits."""
    return [
        {
            "observation_names": list(names),
            "low": np.asarray(space.low, dtype=float).tolist(),
            "high": np.asarray(space.high, dtype=float).tolist(),
        }
        for names, space in zip(agent.observation_names, agent.observation_space)
    ]


def apply_observation_space_limits(agent: Any, limits: list[dict[str, list]]) -> Any:
    """Rebuild ``agent``'s encoders from stored observation-space limits.

    The agent keeps the stored limits as its ``observation_space`` so that any
    later ``set_encoders`` call gives the same result. MARLISA's regression
    encoders are rebuilt too.
    """
    if len(limits) != len(agent.observation_space):
        raise ValueError(
            f"Stored limits cover {len(limits)} buildings; the agent has "
            f"{len(agent.observation_space)}."
        )

    observation_space = []
    for index, (names, space, stored) in enumerate(
        zip(agent.observation_names, agent.observation_space, limits)
    ):
        if list(names) != list(stored["observation_names"]):
            raise ValueError(
                f"Observation names for building index {index} do not match the "
                "stored limits."
            )
        observation_space.append(
            spaces.Box(
                low=np.asarray(stored["low"], dtype=space.dtype),
                high=np.asarray(stored["high"], dtype=space.dtype),
                dtype=space.dtype,
            )
        )

    agent.observation_space = observation_space
    agent.encoders = agent.set_encoders()
    if hasattr(agent, "set_regression_encoders"):
        agent.regression_encoders = agent.set_regression_encoders()
    return agent
