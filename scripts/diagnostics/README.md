# Diagnostics (18 Oct 2026)

Probes of CityLearn 3.0.2 and of the held-out SAC pipeline. They found that agents never see the indoor temperature the simulator computes for their actions, and that held-out evaluation scales observations differently from training. Results are single-seed development checks, not reportable findings.

Run from the repository root with the project's virtual environment:

```shell
python scripts/diagnostics/verify_env.py
python scripts/diagnostics/ab_sac_inputs.py baseline 20 42 runs/ab          # one variant, about 13 min single-threaded
python scripts/diagnostics/ab_sac_inputs.py obsfix 20 42 runs/ab
python scripts/diagnostics/ab_sac_inputs.py obsfix_encfix 20 42 runs/ab
python scripts/diagnostics/ab_progress.py runs/ab                           # KPIs collected so far
python scripts/diagnostics/paired_encoder_eval.py runs/ab                   # re-test the selected checkpoints
```

The three `ab_sac_inputs.py` runs are independent and can run in parallel. `runs/` is git-ignored.

| Script | What it checks |
| --- | --- |
| `verify_env.py` | (1) the indoor temperature the agent observes vs the one the reward and the simulator use; (2) carbon and comfort of reference policies (ideal load, BasicRBC, cooling off, cooling full) over June; (3) when power outages occur; (4) which observation encoders differ between the train, validation and test envs |
| `ab_sac_inputs.py` | Trains held-out SAC with the current code (`baseline`), with the last simulated temperature observed (`obsfix`), and additionally with training encoders reused at evaluation (`obsfix_encfix`) |
| `ab_progress.py` | Tabulates validation and test KPIs written by `ab_sac_inputs.py` |
| `paired_encoder_eval.py` | Evaluates each run's selected checkpoint on the test split with per-split and with training encoders |
| `sim_temperature_env.py` | The observation-fix prototype shared by the scripts above |

The scripts patch `src.run_independent_sac` at runtime and do not change `src/`. The project versions of both fixes are now options in `src/` (19 Oct): `observe_simulated_temperature` in `src/experiment_data.py` and `use_training_encoders` with `src/observation_scaling.py`, or `--pipeline-fixes` on `src/run_independent_sac.py`.
