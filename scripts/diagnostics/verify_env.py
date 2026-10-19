"""Simulator probes behind findings F1, F2, F4 and F6 of the implementation plan.

Sections: (1) what indoor temperature the agent, the reward and the simulator see;
(2) reference policies on full June; (3) power-outage timing; (4) observation
encoder limits per split.

Usage (from the repo root): python scripts/diagnostics/verify_env.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from citylearn.agents.rbc import BasicRBC  # noqa: E402
from citylearn.citylearn import CityLearnEnv  # noqa: E402
from citylearn.cost_function import CostFunction  # noqa: E402
from citylearn.reward_function import RewardFunction  # noqa: E402

SCHEMA = REPO / "notebooks" / "data" / "schema.json"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


class ProbeReward(RewardFunction):
    """Records the indoor temperature the reward function sees."""

    def __init__(self, env_metadata, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.seen: list[list[float]] = []

    def calculate(self, observations):
        self.seen.append([float(o["indoor_dry_bulb_temperature"]) for o in observations])
        return [0.0 for _ in observations]


def make_env(start=0, end=719, outages=False, seed=42):
    return CityLearnEnv(
        schema=str(SCHEMA),
        central_agent=False,
        random_seed=seed,
        simulation_start_time_step=start,
        simulation_end_time_step=end,
        simulate_power_outage=outages,
    )


def oracle_actions(env):
    """Ideal-load tracking: deliver the dataset's cooling demand; no storage use."""
    t = env.time_step
    actions = []
    for b in env.buildings:
        demand = float(b.energy_simulation.cooling_demand_without_control[t])
        cop = float(b.cooling_device.get_cop(b.weather.outdoor_dry_bulb_temperature[t], heating=False))
        a = demand / max(b.cooling_device.nominal_power * cop, 1e-9)
        actions.append([0.0, 0.0, float(np.clip(a, 0.0, 1.0))])
    return actions


def run_policy(env, policy, probe=False):
    if probe:
        env.reward_function = ProbeReward(env.get_metadata())
    obs, _ = env.reset(seed=42)
    names = env.observation_names[0]
    ix = names.index("indoor_dry_bulb_temperature")
    observed = []
    rbc = BasicRBC(env) if policy == "rbc" else None
    done = False
    while not done:
        observed.append([o[ix] for o in obs])
        if policy == "rbc":
            actions = rbc.predict(obs, deterministic=True)
        elif policy == "oracle":
            actions = oracle_actions(env)
        elif policy == "cool_off":
            actions = [[0.0, 0.0, 0.0] for _ in env.buildings]
        elif policy == "cool_full":
            actions = [[0.0, 0.0, 1.0] for _ in env.buildings]
        else:
            raise ValueError(policy)
        obs, _, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
    return np.array(observed)


def discomfort_rate(b, band, start=0, end=None, temps=None):
    T = np.asarray(b.indoor_dry_bulb_temperature if temps is None else temps, dtype=float)
    n = len(b.net_electricity_consumption) if end is None else end + 1
    T = T[start:n]
    sp = np.asarray(b.indoor_dry_bulb_temperature_cooling_set_point[start:n], dtype=float)
    occ = np.asarray(b.occupant_count[start:n], dtype=float) > 0
    hot = (T - sp > band) & occ
    cold = (T - sp < -band) & occ
    return (hot | cold).sum() / max(occ.sum(), 1), hot.sum() / max(occ.sum(), 1), cold.sum() / max(occ.sum(), 1)


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100, flush=True)


# ---------------------------------------------------------------------------
section("1. What indoor temperature does the AGENT observe vs the REWARD vs the simulator? (cooling OFF, 4 days)")
env = make_env(0, 95)
observed = run_policy(env, "cool_off", probe=True)
seen = np.array(env.reward_function.seen)
b0 = env.buildings[0]
sim = np.asarray(b0.indoor_dry_bulb_temperature, dtype=float)
csv = np.asarray(b0.energy_simulation.indoor_dry_bulb_temperature_without_control, dtype=float)
n = len(observed)
print(f"steps={n}  sim series len={len(sim)}")
rows = []
for t in [0, 11, 12, 13, 20, 40, 60, 80, 94]:
    rows.append({
        "t": t,
        "agent_obs_T[t]": observed[t, 0],
        "dataset_T[t]": csv[t],
        "simulated_T[t]": sim[t],
        "simulated_T[t-1]": sim[t - 1] if t > 0 else np.nan,
        "reward_sees_T[t]": seen[t, 0] if t < len(seen) else np.nan,
    })
print(pd.DataFrame(rows).round(2).to_string(index=False))
dyn = slice(13, n)
print("mean |agent_obs - dataset|   (t>=13):", np.abs(observed[dyn, 0] - csv[dyn]).mean().round(4))
print("mean |agent_obs - simulated| (t>=13):", np.abs(observed[dyn, 0] - sim[dyn]).mean().round(3))
print("mean |reward_T - simulated|  (t>=13):", np.abs(seen[13:n - 1, 0] - sim[13:n - 1]).mean().round(4))

# ---------------------------------------------------------------------------
section("2. Reference policies on full June (720 h), outages OFF: carbon ratio vs comfort (1 C and 2 C bands)")
records = []
for policy in ["oracle", "rbc", "cool_off", "cool_full"]:
    t0 = time.perf_counter()
    env = make_env(0, 719, outages=False)
    run_policy(env, policy)
    k = env.evaluate()
    kd = k[k.level == "district"].set_index("cost_function")["value"]
    k1 = env.evaluate(comfort_band=1.0)
    for b in env.buildings:
        d1 = discomfort_rate(b, 1.0)
        d2 = discomfort_rate(b, 2.0)
        records.append({
            "policy": policy, "building": b.name,
            "carbon_ratio_district": kd["carbon_emissions_total"],
            "carbon_kg_building": float(np.clip(np.asarray(b.net_electricity_consumption_emission, float), 0, None).sum()),
            "kpi_disc_2C": float(k[(k.name == b.name) & (k.cost_function == "discomfort_proportion")]["value"].iloc[0]),
            "kpi_disc_1C": float(k1[(k1.name == b.name) & (k1.cost_function == "discomfort_proportion")]["value"].iloc[0]),
            "own_disc_1C(hot/cold)": f"{d1[0]:.3f} ({d1[1]:.2f}/{d1[2]:.2f})",
        })
    print(f"  {policy} done in {time.perf_counter() - t0:.1f}s", flush=True)
ref = pd.DataFrame(records)
print(ref.round(3).to_string(index=False))

env = make_env(0, 719, outages=False)
run_policy(env, "oracle")
print("\nNo-control reference itself (dataset temperatures = what env.evaluate() divides energy by):")
for b in env.buildings:
    csvT = np.asarray(b.energy_simulation.indoor_dry_bulb_temperature_without_control, float)[: len(b.net_electricity_consumption)]
    print(f"  {b.name}: discomfort 1C={discomfort_rate(b, 1.0, temps=csvT)[0]:.3f}  2C={discomfort_rate(b, 2.0, temps=csvT)[0]:.3f}")

# ---------------------------------------------------------------------------
section("3. Power outages: when do they happen? (schema model seed fixed at 73055)")
for (start, end, label) in [(0, 719, "full June"), (0, 479, "train split"), (588, 719, "test+warmup")]:
    for seed in (42, 123):
        try:
            env = make_env(start, end, outages=True, seed=seed)
            env.reset(seed=seed)
            sig = np.asarray(env.buildings[0]._Building__power_outage_signal)
            hrs = np.flatnonzero(sig) + start
            sigs2 = []
            for _ in range(2):
                env.reset()
                sigs2.append(int(np.asarray(env.buildings[0]._Building__power_outage_signal).sum()))
            print(f"  {label:12s} seed={seed}: outage hours={int(sig.sum())} abs steps={hrs.tolist()[:3]}..{hrs.tolist()[-1:] if len(hrs) else ''} | next resets: {sigs2}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:12s} seed={seed}: ERROR {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
section("4. SAC input scaling: observation-space limits per split window (Building_1)")
from src.run_independent_sac import make_sac_agent, make_sac_env  # noqa: E402

agents = {}
for split, warm in [("train", False), ("validation", True), ("test", True)]:
    e = make_sac_env(SCHEMA, split, random_seed=42, include_lstm_warmup=warm)
    agents[split] = make_sac_agent(e)
names = agents["train"].observation_names[0]
rows = []
for i, name in enumerate(names):
    encs = {s: a.encoders[0][i] for s, a in agents.items()}
    if type(encs["train"]).__name__ != "Normalize":
        continue
    lo = {s: float(e.x_min) for s, e in encs.items()}
    hi = {s: float(e.x_max) for s, e in encs.items()}
    if lo["train"] == lo["test"] and hi["train"] == hi["test"] and lo["train"] == lo["validation"] and hi["train"] == hi["validation"]:
        continue
    mid = (lo["train"] + hi["train"]) / 2
    enc_train = (mid - lo["train"]) / (hi["train"] - lo["train"])
    enc_test = (mid - lo["test"]) / (hi["test"] - lo["test"])
    rows.append({"feature": name, "train[lo,hi]": f"[{lo['train']:.3g}, {hi['train']:.3g}]",
                 "val[lo,hi]": f"[{lo['validation']:.3g}, {hi['validation']:.3g}]",
                 "test[lo,hi]": f"[{lo['test']:.3g}, {hi['test']:.3g}]",
                 "same raw value encoded train->test": f"{enc_train:.2f} -> {enc_test:.2f}"})
print(pd.DataFrame(rows).to_string(index=False))
print(f"{len(rows)} of {len(names)} features are scaled differently between train and held-out envs.")
