"""Run MARLISA with carbon, comfort, and building-fairness rewards."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

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
    worst_case_weight: float | None = None,
    disparity_weight: float | None = None,
    fairness_mode: Literal["combined", "worst_case", "disparity"] = "combined",
    comfort_band: float = 1.0,
    split: str = "all",
    run_test: bool | None = None,
    pipeline_fixes: bool = False,
    observe_simulated_temperature: bool = False,
    simulate_power_outage: bool | None = None,
    use_training_encoders: bool = False,
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
        worst_case_weight=worst_case_weight,
        disparity_weight=disparity_weight,
        fairness_mode=fairness_mode,
        comfort_band=comfort_band,
        split=split,
        run_test=run_test,
        pipeline_fixes=pipeline_fixes,
        observe_simulated_temperature=observe_simulated_temperature,
        simulate_power_outage=simulate_power_outage,
        use_training_encoders=use_training_encoders,
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
    parser.add_argument(
        "--split",
        choices=("all", "train", "test", "validation"),
        default="all",
        help="Dataset split to train on (default 'all').",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip held-out test split evaluation when training on the train split.",
    )
    parser.add_argument(
        "--fairness-mode",
        choices=("combined", "worst_case", "disparity"),
        default="combined",
        help="Fairness objective mode: 'worst_case' (RQ1), 'disparity' (RQ2), or 'combined'.",
    )
    parser.add_argument(
        "--worst-case-weight",
        type=float,
        default=None,
        help="Explicit weight on worst-building discomfort (overrides fairness-mode).",
    )
    parser.add_argument(
        "--disparity-weight",
        type=float,
        default=None,
        help="Explicit weight on building discomfort disparity (overrides fairness-mode).",
    )
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
    parser.add_argument(
        "--observe-simulated-temperature",
        action="store_true",
        help="Agents observe the last simulated indoor temperature (WP1 fix).",
    )
    parser.add_argument(
        "--training-encoders",
        action="store_true",
        help="Scale validation/test inputs with the training-window limits.",
    )
    parser.add_argument(
        "--no-power-outages",
        action="store_true",
        help="Disable the schema's power outages (WP1 fix).",
    )
    parser.add_argument(
        "--pipeline-fixes",
        action="store_true",
        help="Turn on all environment and observation fixes.",
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
        split=args.split,
        run_test=not args.no_test if args.no_test else None,
        fairness_mode=args.fairness_mode,
        worst_case_weight=args.worst_case_weight,
        disparity_weight=args.disparity_weight,
        carbon_weight=args.carbon_weight,
        comfort_weight=args.comfort_weight,
        fairness_weight=args.fairness_weight,
        comfort_band=args.comfort_band,
        simulation_end_time_step=args.simulation_end_time_step,
        pipeline_fixes=args.pipeline_fixes,
        observe_simulated_temperature=args.observe_simulated_temperature,
        use_training_encoders=args.training_encoders,
        simulate_power_outage=False if args.no_power_outages else None,
    )
    print("\nFair carbon-aware MARLISA evaluation (lower is better):")
    print(result["kpi_table"].round(3).to_string())
    if result.get("test_result") is not None:
        print("\nHeld-out test KPIs:")
        print(result["test_result"]["kpi_table"].round(3).to_string())
    print("\nAvailable controller comparison:")
    print(result["comparison"].round(3).to_string())
    print(f"\nSaved results to: {result['output_directory']}")


if __name__ == "__main__":
    main()
