"""Run MARLISA with carbon, comfort, and building-fairness rewards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from src.run_marlisa import DEFAULT_SCHEMA, run_marlisa
except ModuleNotFoundError:
    from run_marlisa import DEFAULT_SCHEMA, run_marlisa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "fair_carbon_aware_marlisa"
)


def run_fair_carbon_aware_marlisa(
    schema: str | Path = DEFAULT_SCHEMA,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    carbon_weight: float = 1.0,
    comfort_weight: float = 0.25,
    fairness_weight: float = 0.25,
    comfort_band: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train MARLISA with the project's fair carbon-aware reward."""
    if "reward" in kwargs:
        raise ValueError(
            "This runner fixes reward='fair_carbon'; remove reward from kwargs"
        )

    return run_marlisa(
        schema=schema,
        output_directory=output_directory,
        reward="fair_carbon",
        carbon_weight=carbon_weight,
        comfort_weight=comfort_weight,
        fairness_weight=fairness_weight,
        comfort_band=comfort_band,
        **kwargs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fair carbon-aware CityLearn MARLISA."
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--carbon-weight", type=float, default=1.0)
    parser.add_argument("--comfort-weight", type=float, default=0.25)
    parser.add_argument("--fairness-weight", type=float, default=0.25)
    parser.add_argument("--comfort-band", type=float, default=1.0)
    parser.add_argument(
        "--simulation-end-time-step",
        type=int,
        default=None,
        help="Optional shorter end step, useful only for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_fair_carbon_aware_marlisa(
        schema=args.schema,
        output_directory=args.output_directory,
        episodes=args.episodes,
        random_seed=args.seed,
        batch_size=args.batch_size,
        update_per_time_step=args.updates_per_step,
        carbon_weight=args.carbon_weight,
        comfort_weight=args.comfort_weight,
        fairness_weight=args.fairness_weight,
        comfort_band=args.comfort_band,
        simulation_end_time_step=args.simulation_end_time_step,
    )
    print("\nFair carbon-aware MARLISA evaluation (lower is better):")
    print(result["kpi_table"].round(3).to_string())
    print("\nAvailable controller comparison:")
    print(result["comparison"].round(3).to_string())
    print(f"\nSaved results to: {result['output_directory']}")


if __name__ == "__main__":
    main()
