# Fair Carbon-Aware Multi-Agent Reinforcement Learning for Building Energy Management

Course project using [CityLearn](https://www.citylearn.net/) to study how building-level control agents can reduce electricity-related carbon emissions while limiting thermal discomfort across buildings.

**Status:** Proposal and environment exploration. The research design below is planned; experimental results are not yet available.

To access Proposal/Report here: [Working Report](https://www.overleaf.com/1514918655gmnfrgdhgmtj#c644c2)

## Research question

Can coordinated reinforcement learning reduce district carbon emissions without leaving one building with substantially worse thermal comfort than the others?

We measure building-level fairness through the worst building's discomfort rate and the difference in discomfort rates between buildings. This measure does not describe fairness across demographic groups.

## Planned approach

1. Run CityLearn's no-control and rule-based controllers as reference policies.
2. Train building-level reinforcement learning agents using a shared objective for district carbon emissions and average comfort.
3. Add a penalty for the worst building's discomfort while keeping the agents and training setup comparable.
4. Evaluate all policies on held-out time periods and multiple random seeds.

Planned outcomes include district carbon emissions, peak electricity demand, average discomfort, worst-building discomfort, and the gap between buildings. The final dataset, building selection, time split, reward weights, and seeds will be recorded here before reporting results.

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

### Linux / macOS / WSL

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

Start with [`notebooks/00_Getting_Started.ipynb`](notebooks/00_Getting_Started.ipynb). It uses CityLearn's `citylearn_challenge_2023_phase_2_local_evaluation` dataset for initial exploration. This pilot dataset is not yet the final experimental choice.

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
| `dashboard/app.py` (planned) | Local Plotly Dash results dashboard |
| `tests/` | Unit tests (rewards, splits, pipeline fixes) and smoke tests (SAC, MARLISA) |
| `scripts/diagnostics/` | 18 Oct simulator and SAC-input diagnostics (see its README) |
| `requirements.txt` | Python packages for the project |
| `requirements-lock.txt` | Exact package versions for reproducible runs |
| `.gitignore` | Excludes local environments and generated files |

### Notebook status

Notebooks 00–06 stay as their authors left them, as a record of the work so far. New work goes into new notebooks numbered from 07, and new options in `src/` default to the old behaviour so these notebooks keep running as before. Their numbers are development results (in-sample or single-seed), not findings for the report.

| Notebook | Pipeline | Notes |
| --- | --- | --- |
| `00_Getting_Started` | Data exploration | Current |
| `01_Rule_Based_Control` | BasicRBC, all of June, 2 °C comfort band | Current |
| `02_Independent_SAC` | SAC on train/validation/test splits, 1 °C band | Current; one seed |
| `03_MARLISA` – `05_Fair_Carbon_Aware_MARLISA` | MARLISA variants, trained and scored on all of June, 3 episodes, 2 °C band | Their Independent SAC comparison rows come from the pre-16 Oct electricity-only reward. 04 and 05 expect the old `outputs/independent_sac/` layout. |
| `06_agent_evaluation` | Plotly comparison of 10-episode runs on all of June | Leaves out SAC unless old-layout SAC outputs exist; comfort charts plot dataset temperatures |
| `07_pipeline_checks` | Multi-controller checks, decoupled fairness verification, and held-out test split evaluation | Current |
| `08_results` (planned) | Held-out, multi-seed results | Source of the report's numbers |

## Rule-based controller

Open [`notebooks/01_Rule_Based_Control.ipynb`](notebooks/01_Rule_Based_Control.ipynb) for the guided experiment, or run from the command line:

```shell
python src/run_rbc.py
```

The experiment uses decentralized control so each building receives its own action vector, matching the planned independent SAC and MARLISA setup. It saves observations, actions, rewards, KPIs, a pivoted KPI table, and run metadata under `outputs/rbc/`. The `outputs/` directory is intentionally ignored because these files can be regenerated.

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
# Basic run:
python src/run_marlisa.py --episodes 3

# Recommended research run (held-out splits and pipeline fixes):
python src/run_marlisa.py --episodes 3 --split train --pipeline-fixes
```

The default MARLISA run uses CityLearn's district-coupled `MARL` reward. It keeps the same local schema, seed, neural-network sizes, SAC hyperparameters, exploration schedule, and deterministic final evaluation as the independent SAC baseline. This treats each controller and its recommended reward as one complete approach; reward totals therefore must not be compared directly across SAC and MARLISA. MARLISA also adds learned demand estimators, PCA preprocessing, and information-sharing coordination variables. The runner corrects a CityLearn 3.0.2 post-exploration policy-indexing issue so each building uses its own trained policy; this correction is recorded in the run metadata. Results are saved under `outputs/marlisa/`, including a controller comparison table when RBC and independent SAC results are available.

The runner supports held-out splits (`--split train` with automatic held-out test evaluation on `--split test`), as well as `--pipeline-fixes` matching Independent SAC.

## Carbon-aware MARLISA

Open [`notebooks/04_Carbon_Aware_MARLISA.ipynb`](notebooks/04_Carbon_Aware_MARLISA.ipynb) to train the carbon-aware controller and compare RBC, independent SAC, MARLISA, and carbon-aware MARLISA in one place. The notebook displays experiment settings, district KPIs, improvement relative to RBC, building-level carbon, comfort fairness, and the carbon-comfort trade-off.

Or run the carbon-aware experiment from the command line:

```shell
# Basic run:
python src/run_carbon_aware_marlisa.py --episodes 3

# Recommended research run (held-out splits and pipeline fixes):
python src/run_carbon_aware_marlisa.py --episodes 3 --split train --pipeline-fixes
```

The custom reward is `-max(net_electricity_consumption, 0) * carbon_intensity` for each building and time step. It penalizes grid imports more heavily during carbon-intensive periods and gives no credit for export, consistent with CityLearn's emissions KPI. The only intended change from standard MARLISA is the reward; its controller and training settings are held fixed. Like standard MARLISA, the runner supports held-out splits (`--split train` with automatic held-out test evaluation on `--split test`) and pipeline fixes (`--pipeline-fixes`). Results are stored under `outputs/carbon_aware_marlisa/`.

## Fair carbon-aware MARLISA

Open [`notebooks/05_Fair_Carbon_Aware_MARLISA.ipynb`](notebooks/05_Fair_Carbon_Aware_MARLISA.ipynb) for the complete five-controller comparison and written discussion, or run from the command line:

```shell
# Basic run (3 episodes, default combined fairness):
python src/run_fair_carbon_aware_marlisa.py --episodes 3

# Recommended research runs (held-out splits and pipeline fixes):
# Proposal RQ1: Worst-case fairness (penalizes max_i L_i)
python src/run_fair_carbon_aware_marlisa.py --episodes 3 --split train --fairness-mode worst_case --pipeline-fixes

# Proposal RQ2: Disparity-aware fairness (penalizes max_i L_i - min_i L_i)
python src/run_fair_carbon_aware_marlisa.py --episodes 3 --split train --fairness-mode disparity --pipeline-fixes
```

The reward combines local carbon emissions, each building's occupied temperature loss outside a 1°C band, and shared fairness penalties. To support the project's research questions, the fairness objective can be configured via `--fairness-mode`:
- `combined` (default): penalizes both worst building loss and inter-building disparity.
- `worst_case` (Proposal RQ1): specifically penalizes the worst-off building's discomfort loss ($\max_i L_i$).
- `disparity` (Proposal RQ2): specifically penalizes the spread in discomfort across buildings ($\max_i L_i - \min_i L_i$).

Custom weights can also be specified directly via `--worst-case-weight` and `--disparity-weight`. Held-out splits (`--split train`) and pipeline fixes (`--pipeline-fixes`) are supported. Results and the complete run configuration are saved under `outputs/fair_carbon_aware_marlisa/`.

## Dashboard observation data

Every controller runner now saves the final deterministic episode to `observations.csv`, with one wide row per time step and building. New experiment runs create this file automatically. To backfill it from the deterministic actions of existing runs without retraining the policies, run:

```shell
python src/export_observations.py
```

The replay utility processes the five canonical controller folders, reconstructs the observation trajectories, and checks that the replayed KPIs match each run's saved KPIs. Pass `--controllers` followed by specific output folder names to process only selected controllers.

## Results dashboard (planned)

A standalone Plotly Dash application (`dashboard/app.py`) is planned to present
the saved results as a five-stage data story: overall outcome, carbon-comfort trade-off,
building fairness, time-series evidence, and one-day building replay. In the meantime,
[`notebooks/06_agent_evaluation.ipynb`](notebooks/06_agent_evaluation.ipynb) provides
interactive Plotly comparisons across all controllers.

## Pipeline checks

Open [`notebooks/07_pipeline_checks.ipynb`](notebooks/07_pipeline_checks.ipynb) for interactive verification of the held-out splits (June 1–20 train, June 21–25 validation, June 26–30 test), decoupled fairness objectives (Proposal RQ1 & RQ2), and candidate controller environments.

Or run the automated checks directly from the command line:

```shell
# Fast unit checks (reward formulations, splits, and pipeline fixes):
python -m pytest tests/test_rewards.py tests/test_marlisa_splits.py -k "not smoke"

# End-to-end smoke execution on held-out test splits:
python -m pytest tests/test_marlisa_splits.py -k smoke
```

Add experiment code, configurations, and documented results as the project develops. Keep large generated outputs in the ignored `outputs/` or `runs/` folders; commit only the figures and tables needed to reproduce the report.

## Collaboration

- Work on a branch for each change and open a pull request for a teammate to review.
- Record dataset names, CityLearn version, time splits, reward settings, and random seeds with each experiment.
- Before committing, check `git status` and avoid adding local environments, credentials, or generated data.

## Team

| Member | Role |
| --- | --- |
| [Name] | Environment and data |
| [Name] | Reference policies |
| [Name] | Learning agents |
| [Name] | Reward design and fairness analysis |
| [Name] | Evaluation and reporting |
| [Name, if applicable] | Reproducibility and presentation |

## References

- [CityLearn documentation](https://www.citylearn.net/)
- [CityLearn source repository](https://github.com/citylearn-project/CityLearn)
