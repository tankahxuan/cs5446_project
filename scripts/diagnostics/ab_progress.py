"""Print validation/test KPIs collected so far by ab_sac_inputs.py runs.

Usage (from the repo root):
  python scripts/diagnostics/ab_progress.py <out_dir> [seed]
"""

import sys
from pathlib import Path

import pandas as pd

if len(sys.argv) not in (2, 3):
    sys.exit(__doc__)
out = Path(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) == 3 else 42
rows = []
for variant in ["baseline", "obsfix", "obsfix_encfix"]:
    seed_dir = out / variant / f"seed_{seed}"
    phases = sorted(seed_dir.glob("validation/episode_*/kpis.csv"))
    if (seed_dir / "test" / "kpis.csv").is_file():
        phases.append(seed_dir / "test" / "kpis.csv")
    for path in phases:
        kpis = pd.read_csv(path)
        district = kpis[kpis.level == "district"].set_index("cost_function")["value"]
        building = kpis[(kpis.level == "building") & (kpis.cost_function == "discomfort_proportion")].set_index("name")["value"]
        label = path.parent.name if path.parent.name != "test" else "TEST(selected)"
        rows.append({
            "variant": variant,
            "phase": label,
            "kWh": round(district["electricity_consumption_kwh"], 1),
            "kgCO2": round(district["carbon_emissions_kgco2"], 1),
            "disc_avg": round(district["discomfort_proportion"], 3),
            "B1": round(building["Building_1"], 3),
            "B2": round(building["Building_2"], 3),
            "B3": round(building["Building_3"], 3),
            "worst": round(building.max(), 3),
        })
print(pd.DataFrame(rows).to_string(index=False))
