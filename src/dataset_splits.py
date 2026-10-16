"""Fixed June train/validation/test splits for the local CityLearn dataset.

The dataset covers 720 hourly steps (June 1–30). These definitions are shared
so later experiments can reuse the same held-out intervals without depending on
MARLISA or Independent SAC internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SplitName = Literal["train", "validation", "test"]

# LSTMDynamicsBuilding lookback used by the local schema buildings.
LSTM_LOOKBACK = 12


@dataclass(frozen=True)
class TimeSplit:
    """Inclusive absolute time-step interval for one evaluation phase."""

    name: SplitName
    start_time_step: int
    end_time_step: int
    start_date: str
    end_date: str

    @property
    def n_time_steps(self) -> int:
        """Number of inclusive environment time steps in the interval."""
        return self.end_time_step - self.start_time_step + 1

    @property
    def n_transitions(self) -> int:
        """Number of ``env.step`` transitions CityLearn executes in the interval.

        CityLearn truncates when ``time_step >= time_steps - 1``, so an episode
        with ``N`` inclusive time steps yields ``N - 1`` transitions.
        """
        return max(self.n_time_steps - 1, 0)

    def as_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "start_time_step": self.start_time_step,
            "end_time_step": self.end_time_step,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_time_steps": self.n_time_steps,
            "n_transitions": self.n_transitions,
        }


TRAIN_SPLIT = TimeSplit("train", 0, 479, "June 1", "June 20")
VALIDATION_SPLIT = TimeSplit("validation", 480, 599, "June 21", "June 25")
TEST_SPLIT = TimeSplit("test", 600, 719, "June 26", "June 30")

SPLITS: dict[SplitName, TimeSplit] = {
    "train": TRAIN_SPLIT,
    "validation": VALIDATION_SPLIT,
    "test": TEST_SPLIT,
}

DEFAULT_SEEDS = (42, 123, 2024, 2025, 2026)


def get_split(name: SplitName | str) -> TimeSplit:
    try:
        return SPLITS[name]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(
            f"Unknown split {name!r}. Expected one of {sorted(SPLITS)}."
        ) from exc


def warmup_start_time_step(
    split: TimeSplit,
    lookback: int = LSTM_LOOKBACK,
) -> int:
    """Return the earliest absolute start step that includes LSTM warm-up.

    When ``split.start_time_step`` is large enough, evaluation environments can
    begin ``lookback`` steps earlier so dynamics are warm before scored steps.
    Training always starts at 0, so no earlier warm-up data exists.
    """
    return max(0, split.start_time_step - int(lookback))
