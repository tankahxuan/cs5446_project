# Fair Carbon-Aware Multi-Agent Reinforcement Learning for Building Energy Management

Course project using [CityLearn](https://www.citylearn.net/) to study how building-level control agents can reduce electricity-related carbon emissions while limiting thermal discomfort across buildings.

**Status:** Experiments complete. Every controller has been trained and tested on held-out June data with five random seeds (see [Multi-seed experiments](#multi-seed-experiments)); the report is being finalised.

To access Proposal/Report here: [Working Report](https://www.overleaf.com/1514918655gmnfrgdhgmtj#c644c2)

## Research questions

1. Can coordinated multi-agent reinforcement learning reduce district carbon emissions while limiting disproportionate thermal discomfort in the worst-affected buildings?
2. How do different fairness objectives affect the trade-off between carbon emissions and comfort?

We measure building-level fairness through the worst building's discomfort rate and the difference in discomfort rates between buildings. This measure does not describe fairness across demographic groups.

## Approach and final settings

The study compares five controllers, each with one agent per building:

- **BasicRBC:** fixed hourly rules, no learning (the reference).
- **Independent SAC:** one Soft Actor-Critic agent per building with no information sharing; the reward is electricity use plus discomfort.
- **MARLISA:** one SAC agent per building that also shares its predicted demand with the others; same reward as Independent SAC.
- **Carbon-aware MARLISA:** MARLISA with a carbon-only reward.
- **Fair carbon-aware MARLISA:** MARLISA with carbon, comfort and fairness penalties (worst-off building and gap between buildings).

Three variants of the fair reward (no fairness term, worst-building term only, gap term only) isolate the effect of the fairness term.

| Setting | Value |
| --- | --- |
| Simulator | CityLearn 3.0.2 |
| Dataset | `citylearn_challenge_2023_phase_2_local_evaluation` (local copy in `notebooks/data/`): 3 buildings, hourly, 1–30 June |
| Splits | Train 1–20 June (steps 0–479), validation 21–25 June (480–599), test 26–30 June (600–719). Validation and test runs start 12 hours early to warm up the temperature model; 119 test hours are scored |
| Training | 50 episodes per run; MARLISA's last episode is a deterministic test pass |
| Seeds | 42, 123, 2024, 2025, 2026 (`DEFAULT_SEEDS` in `src/dataset_splits.py`) |
| Comfort measure | Share of occupied hours more than 1 °C from the setpoint |
| Rewards | Standard: −E/10 − D/4 (`src/sac_reward.py`); carbon: −E·c; fair: −E·c − 0.25·L − (0.25·worst L + 0.25·gap in L) (`src/rewards.py`) |
| Pipeline fixes | Off, except that the multi-seed Independent SAC runs reuse training-window input scaling, as MARLISA does |

E is electricity imported from the grid (kWh), c the grid's carbon intensity, and D and L are discomfort terms (defined in the report). Outcomes are district carbon emissions, daily peak demand, average discomfort, worst-building discomfort and the discomfort gap between buildings.

## Getting started

Use **Python 3.11** and CityLearn **3.0.2**. Create and activate a local virtual environment for your platform:

### Windows (PowerShell)

```powershell
# 1. Create a Python 3.11 virtual environment
py -3.11 -m venv .venv
# or if python is 3.11: python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

> **Tip (Windows users)**: If PowerShell restricts running scripts, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, or execute commands directly with `.\.venv\Scripts\python.exe`.

> **Windows: `import torch` fails with `WinError 1114`**: This happens when the venv is built from Miniconda's Python, which loads an older Visual C++ runtime. Build the venv from a uv-managed Python instead:
> ```powershell
> uv venv --python 3.11 --python-preference only-managed .venv
> uv pip install -r requirements-lock.txt --python .venv\Scripts\python.exe
> ```

### macOS / Linux / WSL

On macOS, if `python3.11` is not found, install it with Homebrew (`brew install python@3.11`) or use uv (see the tip below).

```bash
# 1. Create a Python 3.11 virtual environment
python3.11 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Tip (if Python 3.11 is not installed)**: You can use [uv](https://docs.astral.sh/uv/) to automatically download Python 3.11 and create the environment without root privileges:
> ```bash
> uv venv --python 3.11 .venv
> source .venv/bin/activate
> uv pip install -r requirements.txt
> ```

### Exact versions for experiment runs

`requirements.txt` pins only CityLearn. `requirements-lock.txt` pins every package to the versions that ran the tests and diagnostics, for Python 3.11.5 or newer on Windows, macOS and Linux. Use it for any run whose results go into the report:

```shell
pip install -r requirements-lock.txt          # or: uv pip install -r requirements-lock.txt
python -m pytest tests/test_rewards.py tests/test_sac_reward.py tests/test_sac_splits.py   # fast checks, about 20 s
# or run the full test suite (including pipeline fixes and smoke tests):
python -m pytest
```

In VS Code, install the Python and Jupyter extensions, open a notebook, then select **Kernel → Python Environments → .venv**. The virtual environment is excluded from Git by `.gitignore`; each collaborator creates their own.

> **Note on running scripts**: All command-line examples in this repository assume that `.venv` is activated in your current terminal session (indicated by a `(.venv)` prompt prefix). When opening a new terminal session, remember to activate it first (`.\.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`).

Start with [`notebooks/00_Getting_Started.ipynb`](notebooks/00_Getting_Started.ipynb). It explores CityLearn's `citylearn_challenge_2023_phase_2_local_evaluation` dataset, which all experiments use (local copy in `notebooks/data/`).

## Repository contents

| Path | Purpose |
| --- | --- |
| `notebooks/00_Getting_Started.ipynb` | Initial CityLearn exploration |
| `notebooks/01_Rule_Based_Control.ipynb` | Run and inspect the `BasicRBC` reference controller |
| `notebooks/02_Independent_SAC.ipynb` | Train and evaluate one SAC policy per building |
| `notebooks/03_MARLISA.ipynb` | Train coordinated MARLISA policies and compare all controllers |
| `notebooks/04_Carbon_Aware_MARLISA.ipynb` | Train carbon-aware MARLISA and compare all four controllers |
| `notebooks/05_Fair_Carbon_Aware_MARLISA.ipynb` | Train fair carbon-aware MARLISA and discuss all five controllers |
| `notebooks/06_agent_evaluation.ipynb` | Compare all controllers with interactive Plotly visualisations |
| `notebooks/07_pipeline_checks.ipynb` | Multi-controller checks, decoupled fairness verification, and held-out test split evaluation |
| `src/run_rbc.py` | Reproducible command-line runner for the RBC experiment |
| `src/run_independent_sac.py` | Reusable independent SAC training and evaluation runner |
| `src/run_marlisa.py` | Reusable MARLISA training, evaluation, and comparison runner |
| `src/run_carbon_aware_marlisa.py` | Carbon-aware MARLISA experiment runner |
| `src/run_fair_carbon_aware_marlisa.py` | Fair carbon-aware MARLISA experiment runner |
| `src/export_observations.py` | Reconstruct wide observation CSVs from controller actions |
| `src/rewards.py` | Project-specific carbon reward functions |
| `src/sac_reward.py` | Electricity + discomfort reward used by Independent SAC and standard MARLISA |
| `src/dataset_splits.py` | Fixed June train/validation/test splits and the default seed list |
| `src/observation_scaling.py` | Reuses training-window observation scaling when an agent is evaluated on another split |
| `src/experiment_data.py` | Shared data-capture utilities for the runners |
| `tests/` | Unit tests (rewards, splits, pipeline fixes) and smoke tests (SAC, MARLISA) |
| `scripts/diagnostics/` | 18 Oct simulator and SAC-input diagnostics (see its README) |
| `scripts/run_06_plan.py`, `scripts/execute_06_viz.py` | Run notebook 06's seed-42 training plan, or refresh its charts, outside Jupyter |
| `scripts/run_seed_matrix.py` | Launch the multi-seed experiment matrix in parallel and build the summary tables |
| `scripts/compare_seed_runs.py` | Head-to-head comparisons between controllers: seeds won and exact p-values |
| `scripts/make_report_figures.py` | Draw the report's two figures |
| `outputs/` | Experiment results; `outputs/seed_matrix_summary/` holds the multi-seed tables |
| `requirements.txt` | Python packages for the project |
| `requirements-lock.txt` | Exact package versions for reproducible runs |
| `.gitignore` | Excludes local environments and generated files |

### Notebook status

Notebooks 00–06 stay as their authors left them, as a record of the work so far. New work goes into new notebooks numbered from 07, and new options in `src/` default to the old behaviour so these notebooks keep running as before. Their numbers are development results (in-sample or single-seed), not findings for the report.

| Notebook | Pipeline | Notes |
| --- | --- | --- |
| `00_Getting_Started` | Data exploration | Current |
| `01_Rule_Based_Control` | BasicRBC on held-out test split by default | Current |
| `02_Independent_SAC` | SAC on train/validation/test splits, 1 °C band | Current; one seed |
| `03_MARLISA` – `05_Fair_Carbon_Aware_MARLISA` | MARLISA variants on the shared held-out protocol, 50 episodes | Standard MARLISA uses electricity+discomfort; carbon / fair variants change only the reward |
| `06_agent_evaluation` | Plotly comparison on held-out outputs (`seed_*/test_kpis.csv`) | Set `RUN_ALL_MODELS=True` to regenerate |
| `07_pipeline_checks` | Multi-controller checks, decoupled fairness verification, and held-out test split evaluation | Current |
| Multi-seed results (scripts, no notebook) | `scripts/run_seed_matrix.py` → `outputs/seed_matrix_summary/` | Source of the report's numbers (five seeds) |

## Rule-based controller

Open [`notebooks/01_Rule_Based_Control.ipynb`](notebooks/01_Rule_Based_Control.ipynb) for the guided experiment, or run from the command line:

```shell
python src/run_rbc.py
```

The experiment uses decentralized control so each building receives its own action vector, matching Independent SAC and MARLISA. By default it evaluates the held-out test split (June 26–30) with LSTM warm-up and writes under `outputs/rbc/seed_<seed>/`. Use `--split all` for a full-month reference.

`BasicRBC` follows a fixed hourly schedule. It charges storage overnight and discharges it during the day; it also changes the available cooling power according to the hour. It does not learn from rewards.

## Independent SAC

Open [`notebooks/02_Independent_SAC.ipynb`](notebooks/02_Independent_SAC.ipynb) for the guided experiment, or run from the command line:

```shell
# Basic run:
python src/run_independent_sac.py --episodes 3

# With pipeline fixes (simulated temperature, training encoders, no outages):
python src/run_independent_sac.py --episodes 3 --pipeline-fixes

# With a carbon-aware or fairness-aware reward:
python src/run_independent_sac.py --episodes 3 --reward worst_case_fair --pipeline-fixes
```

The default workflow uses one SAC policy per building and `ElectricityDiscomfortSACReward`; `--reward independent_sac` selects the same reward. Training runs on the June 1–20 split (`TRAIN_SPLIT`), validation on June 21–25 (`VALIDATION_SPLIT`) selects the best checkpoint, and deterministic testing evaluates on the held-out June 26–30 split (`TEST_SPLIT`). Three episodes are useful for checking the pipeline but are not enough for a final research result. Increase the episode count and repeat the experiment across fixed random seeds for reported comparisons.

`--reward` sets what the agents train on, how validation ranks checkpoints, and where results go by default:

| `--reward` | Training objective | Checkpoints ranked by | Default output directory |
| --- | --- | --- | --- |
| `electricity_discomfort` (default), `independent_sac` | Electricity imports and thermal discomfort | Electricity consumption | `outputs/independent_sac/` |
| `carbon` | Carbon emissions | Carbon emissions | `outputs/carbon_sac/` |
| `fair_carbon` | Carbon, discomfort, and the fairness terms set by `--fairness-mode` (default `combined`) | Carbon emissions | `outputs/fair_carbon_sac/` |
| `worst_case_fair` | Carbon, discomfort, and worst-building discomfort | Worst-building discomfort | `outputs/worst_case_fair_sac/` |
| `disparity_fair` | Carbon, discomfort, and the discomfort gap between buildings | Discomfort gap between buildings | `outputs/disparity_fair_sac/` |
| `schema` | The reward defined in the schema | Electricity consumption | `outputs/schema_sac/` |

Validation first keeps checkpoints at or below `--discomfort-threshold`, then picks the lowest value of the ranking metric. If none qualify, it picks the lowest discomfort and uses the ranking metric to break ties.

The default reward's weights are `--lambda-electricity` and `--lambda-discomfort`. `--carbon-weight` applies to the carbon and fairness rewards, while `--comfort-weight` and `--fairness-weight` apply to the fairness rewards only. `--worst-case-weight` and `--disparity-weight` rescale a fairness term and raise an error if the fairness mode turns that term off. `worst_case_fair` and `disparity_fair` fix their own fairness mode and reject a different `--fairness-mode`.

Each run saves validation and held-out test observations, episode rewards, final actions and rewards, KPI tables, metadata, and PyTorch checkpoints. Runs with the same reward and seed share a directory, so give each run in a weight sweep its own `--output-directory`.

## MARLISA

Open [`notebooks/03_MARLISA.ipynb`](notebooks/03_MARLISA.ipynb) for the guided experiment, or run from the command line:

```shell
# Default research protocol (train/val/test, 50 episodes, electricity+discomfort):
python src/run_marlisa.py

# Short smoke / debug on the full month:
python src/run_marlisa.py --episodes 3 --split all

# Optional pipeline fixes (off by default; existing SAC runs did not use them):
python src/run_marlisa.py --pipeline-fixes
```

The default MARLISA run uses the electricity+discomfort reward (no carbon/fairness) on the shared held-out protocol: train June 1–20, validation June 21–25, test June 26–30. It keeps the same local schema, seed, network sizes, and SAC hyperparameters as Independent SAC. CityLearn's `learn()` final episode is deterministic; validation/test then score that final policy (mid-training checkpoint selection is not exposed by CityLearn MARLISA). Coordination still comes from demand estimators, PCA, and information sharing. The runner corrects a CityLearn 3.0.2 post-exploration policy-indexing issue. Results are saved under `outputs/marlisa/seed_<seed>/`, including `test_kpis.csv` and `selected_checkpoint.pt`. Use `--reward marl` only if you need CityLearn's district-coupled `MARL` reward.

## Carbon-aware MARLISA

Open [`notebooks/04_Carbon_Aware_MARLISA.ipynb`](notebooks/04_Carbon_Aware_MARLISA.ipynb) to train the carbon-aware controller and compare RBC, independent SAC, MARLISA, and carbon-aware MARLISA in one place. The notebook displays experiment settings, district KPIs, improvement relative to RBC, building-level carbon, comfort fairness, and the carbon-comfort trade-off.

Or run the carbon-aware experiment from the command line:

```shell
# Default research protocol (held-out train/val/test, 50 episodes):
python src/run_carbon_aware_marlisa.py

# Short smoke / debug on the full month:
python src/run_carbon_aware_marlisa.py --episodes 3 --split all
```

The custom reward is `-max(net_electricity_consumption, 0) * carbon_intensity` for each building and time step. It penalizes grid imports more heavily during carbon-intensive periods and gives no credit for export, consistent with CityLearn's emissions KPI. Comfort and fairness stay off for this baseline. The only intended change from standard MARLISA is the reward. Results are stored under `outputs/carbon_aware_marlisa/seed_<seed>/`.

## Fair carbon-aware MARLISA

Open [`notebooks/05_Fair_Carbon_Aware_MARLISA.ipynb`](notebooks/05_Fair_Carbon_Aware_MARLISA.ipynb) for the complete five-controller comparison and written discussion, or run from the command line:

```shell
# Default research protocol (held-out, 50 episodes, combined fairness):
python src/run_fair_carbon_aware_marlisa.py

# Worst-case fairness only (penalizes max_i L_i)
python src/run_fair_carbon_aware_marlisa.py --fairness-mode worst_case

# Gap (disparity) fairness only (penalizes max_i L_i - min_i L_i)
python src/run_fair_carbon_aware_marlisa.py --fairness-mode disparity
```

The reward combines local carbon emissions, each building's occupied temperature loss outside a 1°C band, and shared fairness penalties. To support the project's research questions, the fairness objective can be configured via `--fairness-mode`:
- `combined` (default): penalizes both worst building loss and inter-building disparity.
- `worst_case`: penalizes only the worst-off building's discomfort loss ($\max_i L_i$).
- `disparity`: penalizes only the spread in discomfort across buildings ($\max_i L_i - \min_i L_i$).

Comparing these modes, plus a run with `--fairness-weight 0` (no fairness term), isolates the fairness term for research question 2. Custom weights can also be specified directly via `--worst-case-weight` and `--disparity-weight`. Results are saved under `outputs/fair_carbon_aware_marlisa/seed_<seed>/`.

## Dashboard observation data

Every controller runner now saves the final deterministic episode to `observations.csv`, with one wide row per time step and building. New experiment runs create this file automatically. To backfill it from the deterministic actions of existing runs without retraining the policies, run:

```shell
python src/export_observations.py
```

The replay utility processes the five canonical controller folders, reconstructs the observation trajectories, and checks that the replayed KPIs match each run's saved KPIs. Pass `--controllers` followed by specific output folder names to process only selected controllers.

## Results dashboard (not built)

A standalone Plotly Dash application (`dashboard/app.py`) was planned but has not been built. [`notebooks/06_agent_evaluation.ipynb`](notebooks/06_agent_evaluation.ipynb) provides interactive Plotly comparisons of the seed-42 runs; the multi-seed results are summarised under [Multi-seed experiments](#multi-seed-experiments).

## Pipeline checks

Open [`notebooks/07_pipeline_checks.ipynb`](notebooks/07_pipeline_checks.ipynb) for interactive verification of the held-out splits (June 1–20 train, June 21–25 validation, June 26–30 test), decoupled fairness objectives (worst-case and disparity), and candidate controller environments.

Or run the automated checks directly from the command line:

```shell
# Fast unit checks (reward formulations, splits, and pipeline fixes):
python -m pytest tests/test_rewards.py tests/test_marlisa_splits.py -k "not smoke"

# End-to-end smoke execution on held-out test splits:
python -m pytest tests/test_marlisa_splits.py -k smoke
```

Add experiment code, configurations, and documented results as the project develops. Formal KPI/metadata/checkpoint outputs under `outputs/<method>/seed_*/` are tracked for team comparison; smoke runs under `outputs/*_smoke/` stay ignored.

## Multi-seed experiments

These runs produce the report's numbers. `scripts/run_seed_matrix.py` trains and evaluates every configuration for each seed in parallel, one single-threaded process per run, with the same settings as the seed-42 runs in `scripts/run_06_plan.py`: 50 episodes, held-out test on 26–30 June, pipeline fixes off. A run is skipped if its results already exist, so committed results are never overwritten.

| Configuration | Output folder |
| --- | --- |
| MARLISA, carbon-aware MARLISA, fair carbon-aware MARLISA | `outputs/marlisa/`, `outputs/carbon_aware_marlisa/`, `outputs/fair_carbon_aware_marlisa/` |
| Independent SAC with training-window input scaling (also tested on its final episode, in `test_final_episode/`) | `outputs/independent_sac_training_encoders/` |
| Fair reward with no fairness term, worst-building term only, gap term only | `outputs/fair_carbon_aware_marlisa_no_fairness/`, `outputs/fair_carbon_aware_marlisa_worst_case/`, `outputs/fair_carbon_aware_marlisa_disparity/` |

Each run writes to `<folder>/seed_<seed>/`; logs and timings go to `outputs/seed_matrix_logs/`.

### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\run_seed_matrix.py --list                  # show which runs are still to do
python scripts\run_seed_matrix.py --launch --workers 8    # run them in parallel
python scripts\run_seed_matrix.py --summarize --seeds 42 123 2024 2025 2026
python scripts\compare_seed_runs.py outputs\seed_matrix_summary\per_run.csv
python scripts\make_report_figures.py --output-dir outputs\report_figures
```

### macOS (Terminal)

```bash
source .venv/bin/activate
python scripts/run_seed_matrix.py --list                  # show which runs are still to do
python scripts/run_seed_matrix.py --launch --workers 8    # run them in parallel
python scripts/run_seed_matrix.py --summarize --seeds 42 123 2024 2025 2026
python scripts/compare_seed_runs.py outputs/seed_matrix_summary/per_run.csv
python scripts/make_report_figures.py --output-dir outputs/report_figures
```

- **Choosing runs:** `--seeds` picks the seeds (default: the first three of `DEFAULT_SEEDS`) and `--configs` picks configurations. `--smoke --output-root <folder>` makes short runs to check the setup.
- **Workers:** set `--workers` to no more than the number of CPU cores. Each run needs about 0.5 GB of memory. On a 24-core Windows PC, 18 runs in parallel took about 26 minutes.
- **Outputs:**
  - `--summarize` writes `outputs/seed_matrix_summary/summary.csv` (mean, standard deviation, lowest and highest per configuration) and `per_run.csv` (every run).
  - `compare_seed_runs.py` prints, for key pairs of controllers, how many seeds each wins and exact p-values. Its saved output is `head_to_head.txt`.

**Headline results** (test period, mean over five seeds; discomfort is the share of occupied hours more than 1 °C from the setpoint):

| Controller | Carbon (kg CO₂) | Avg discomfort | Worst building | Gap (pp) |
| --- | --- | --- | --- | --- |
| BasicRBC (one run) | 365.6 | 98.2% | 99.2% | 2.7 |
| Independent SAC | 179.0 | 21.3% | 32.7% | 23.1 |
| MARLISA | 179.4 | 15.9% | 22.4% | 13.0 |
| Carbon-aware MARLISA | 83.8 | 96.5% | 99.2% | 5.5 |
| Fair carbon-aware MARLISA | 179.6 | 34.0% | 46.4% | 20.1 |
| Fair: no fairness term | 165.7 | 41.1% | 54.7% | 26.1 |
| Fair: worst-building term only | 174.9 | 28.4% | 35.9% | 14.0 |
| Fair: gap term only | 174.5 | 33.7% | 47.7% | 23.3 |

- **MARLISA vs Independent SAC:** MARLISA is the most comfortable controller and narrows the discomfort gap between buildings in every seed. Rewards are per-building, so this does not prove that coordination itself is the cause.
- **Fairness terms:** the worst-building term improves comfort over no fairness term in every seed, for about 6% more carbon. The gap term does not reliably help.
- **Carbon only:** carbon-aware MARLISA almost switches cooling off.

## Collaboration

- Work on a branch for each change and open a pull request for a teammate to review.
- Record dataset names, CityLearn version, time splits, reward settings, and random seeds with each experiment.
- Before committing, check `git status` and avoid adding local environments, credentials, or generated data.

## Team

- Chen Zihan
- Li Jiaqi
- Ma Yongjie
- Tan Kah Xuan
- Wee Guan Hao Meldrick

Roles and individual contributions are listed in the report's appendix.

## References

- [CityLearn documentation](https://www.citylearn.net/)
- [CityLearn source repository](https://github.com/citylearn-project/CityLearn)
