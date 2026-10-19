"""Evaluate the same saved checkpoints on the test split with per-split vs training encoders.

Reads the selected checkpoints written by ab_sac_inputs.py.

Usage (from the repo root):
  python scripts/diagnostics/paired_encoder_eval.py <out_dir> [seed]
"""

import sys
from pathlib import Path

import pandas as pd
import torch

from sim_temperature_env import LastSimulatedTemperatureEnv

import src.run_independent_sac as ris  # noqa: E402  (sim_temperature_env puts the repo on sys.path)

if len(sys.argv) not in (2, 3):
    sys.exit(__doc__)
runs = Path(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) == 3 else 42
OriginalEnv = ris.ObservationHistoryCityLearnEnv
original_make_agent = ris.make_sac_agent
torch.set_num_threads(1)

rows = []
for variant, obs_patch in [("baseline", False), ("obsfix", True), ("obsfix_encfix", True)]:
    selected = runs / variant / f"seed_{seed}" / "selected_checkpoint.pt"
    for encoders in ("per_split", "train"):
        ris.ObservationHistoryCityLearnEnv = LastSimulatedTemperatureEnv if obs_patch else OriginalEnv
        if encoders == "train":
            train_agent = original_make_agent(ris.make_sac_env(ris.DEFAULT_SCHEMA, "train", random_seed=seed))

            def make_agent(env, _encoders=train_agent.encoders, **kwargs):
                agent = original_make_agent(env, **kwargs)
                agent.encoders = _encoders
                return agent

            ris.make_sac_agent = make_agent
        else:
            ris.make_sac_agent = original_make_agent
        result = ris.evaluate_independent_sac(selected, split="test", save_results=False)
        k = result["kpis"]
        b = k[(k.level == "building") & (k.cost_function == "discomfort_proportion")].set_index("name")["value"]
        d = k[k.level == "district"].set_index("cost_function")["value"]
        rows.append({
            "run": variant, "checkpoint": "selected", "test_encoders": encoders,
            "kgCO2": round(d["carbon_emissions_kgco2"], 1), "disc_avg": round(d["discomfort_proportion"], 3),
            "B1": round(b["Building_1"], 3), "B2": round(b["Building_2"], 3), "B3": round(b["Building_3"], 3),
        })
print(pd.DataFrame(rows).to_string(index=False))
