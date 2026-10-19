"""Single-seed diagnostic: does fixing SAC's inputs change held-out comfort?

Variants (all use the project's held-out SAC pipeline unchanged otherwise):
  baseline       current code
  obsfix         agent observes the last *simulated* indoor temperature instead of
                 the dataset (ideal-load) temperature CityLearn 3.0.2 exposes
  obsfix_encfix  obsfix + validation/test agents reuse the training-window
                 observation encoders instead of rebuilding them per split

Usage (from the repo root):
  python scripts/diagnostics/ab_sac_inputs.py <variant> <episodes> <seed> <out_dir>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from sim_temperature_env import LastSimulatedTemperatureEnv

import src.run_independent_sac as ris  # noqa: E402  (sim_temperature_env puts the repo on sys.path)

VARIANTS = ("baseline", "obsfix", "obsfix_encfix")

if len(sys.argv) != 5 or sys.argv[1] not in VARIANTS:
    sys.exit(__doc__)
variant, episodes, seed, out_dir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
torch.set_num_threads(1)

if variant in ("obsfix", "obsfix_encfix"):
    ris.ObservationHistoryCityLearnEnv = LastSimulatedTemperatureEnv

if variant == "obsfix_encfix":
    _original_make_agent = ris.make_sac_agent
    _train_encoders = {}

    def make_agent_with_train_encoders(env, **kwargs):
        agent = _original_make_agent(env, **kwargs)
        if getattr(env, "_sac_split_name", "train") != "train":
            if "encoders" not in _train_encoders:
                train_env = ris.make_sac_env(ris.DEFAULT_SCHEMA, "train", random_seed=seed)
                _train_encoders["encoders"] = _original_make_agent(train_env).encoders
            agent.encoders = _train_encoders["encoders"]
        return agent

    ris.make_sac_agent = make_agent_with_train_encoders

start = time.perf_counter()
result = ris.train_independent_sac(
    schema=ris.DEFAULT_SCHEMA,
    output_directory=out_dir / variant,
    episodes=episodes,
    random_seed=seed,
    validation_every=5,
    verbose=False,
)
minutes = (time.perf_counter() - start) / 60
history = result["validation_history"][
    ["episode", "district_electricity_consumption_kwh", "district_discomfort_proportion", "district_carbon_emissions_kgco2"]
]
test = result["test_result"]["kpis"]
summary = {
    "variant": variant,
    "seed": seed,
    "episodes": episodes,
    "minutes": round(minutes, 1),
    "validation": history.round(4).to_dict(orient="records"),
    "selected": result["model_selection"],
    "test_building_discomfort": test[(test.level == "building") & (test.cost_function == "discomfort_proportion")]
    .set_index("name")["value"].round(3).to_dict(),
    "test_district": test[test.level == "district"].set_index("cost_function")["value"].round(3).to_dict(),
}
(out_dir / f"{variant}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
