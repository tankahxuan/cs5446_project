"""Run MARLISA with the project's time-varying carbon reward."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from src.run_marlisa import DEFAULT_SCHEMA, run_marlisa
except ModuleNotFoundError:
    from run_marlisa import DEFAULT_SCHEMA, run_marlisa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "carbon_aware_marlisa"


def run_carbon_aware_marlisa(
    schema: str | Path = DEFAULT_SCHEMA,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train MARLISA with local carbon-emissions rewards per building."""
    if "reward" in kwargs:
        raise ValueError(
            "run_carbon_aware_marlisa fixes reward='carbon'; remove reward from kwargs"
        )

    return run_marlisa(
        schema=schema,
        output_directory=output_directory,
        reward="carbon",
        **kwargs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate carbon-aware CityLearn MARLISA."
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument(
        "--simulation-end-time-step",
        type=int,
        default=None,
        help="Optional shorter end step, useful only for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_carbon_aware_marlisa(
        schema=args.schema,
        output_directory=args.output_directory,
        episodes=args.episodes,
        random_seed=args.seed,
        batch_size=args.batch_size,
        update_per_time_step=args.updates_per_step,
        simulation_end_time_step=args.simulation_end_time_step,
    )
    print("\nCarbon-aware MARLISA evaluation (lower is better):")
    print(result["kpi_table"].round(3).to_string())
    print("\nAvailable controller comparison:")
    print(result["comparison"].round(3).to_string())
    print(f"\nSaved results to: {result['output_directory']}")


if __name__ == "__main__":
    main()
