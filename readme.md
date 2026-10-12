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
| `src/run_rbc.py` | Reproducible command-line runner for the RBC experiment |
| `src/run_independent_sac.py` | Reusable independent SAC training and evaluation runner |
| `src/run_marlisa.py` | Reusable MARLISA training, evaluation, and comparison runner |
| `src/run_carbon_aware_marlisa.py` | Carbon-aware MARLISA experiment runner |
| `src/run_fair_carbon_aware_marlisa.py` | Fair carbon-aware MARLISA experiment runner |
| `src/rewards.py` | Project-specific carbon reward functions |
| `requirements.txt` | Python packages for the project |
| `.gitignore` | Excludes local environments and generated files |

## Rule-based controller

Open [`notebooks/01_Rule_Based_Control.ipynb`](notebooks/01_Rule_Based_Control.ipynb) for the guided experiment, or run from the command line:

```shell
python src/run_rbc.py
```

The experiment uses decentralized control so each building receives its own action vector, matching the planned independent SAC and MARLISA setup. It saves actions, rewards, KPIs, a pivoted KPI table, and run metadata under `outputs/rbc/`. The `outputs/` directory is intentionally ignored because these files can be regenerated.

`BasicRBC` follows a fixed hourly schedule. It charges storage overnight and discharges it during the day; it also changes the available cooling power according to the hour. It does not learn from rewards.

## Independent SAC

Open [`notebooks/02_Independent_SAC.ipynb`](notebooks/02_Independent_SAC.ipynb) for the guided experiment, or run from the command line:

```shell
python src/run_independent_sac.py --episodes 3
```

The default workflow uses one SAC policy per building and CityLearn's `IndependentSACReward`. Episode 1 collects experience, the middle episode(s) train the policies, and the last episode evaluates deterministic actions. Three episodes are useful for checking the pipeline but are not enough for a final research result. Increase the episode count and repeat the experiment across fixed random seeds for reported comparisons.

Results are saved under `outputs/independent_sac/`, including episode rewards, final actions and rewards, KPI tables, metadata, and a PyTorch checkpoint. The final deterministic episode uses the same 720-step period as training, so these results are for development. A held-out evaluation period should be defined before drawing research conclusions.

## MARLISA

Open [`notebooks/03_MARLISA.ipynb`](notebooks/03_MARLISA.ipynb) for the guided experiment, or run from the command line:

```shell
python src/run_marlisa.py --episodes 3
```

The default MARLISA run uses the same local schema, seed, independent SAC reward, neural-network sizes, SAC hyperparameters, exploration schedule, and deterministic final evaluation as the independent SAC baseline. MARLISA adds learned demand estimators, PCA preprocessing, and information-sharing coordination variables. The runner also corrects a CityLearn 3.0.2 post-exploration policy-indexing issue so each building uses its own trained policy; this correction is recorded in the run metadata. Results are saved under `outputs/marlisa/`, including a controller comparison table when RBC and independent SAC results are available.

## Carbon-aware MARLISA

Open [`notebooks/04_Carbon_Aware_MARLISA.ipynb`](notebooks/04_Carbon_Aware_MARLISA.ipynb) to train the carbon-aware controller and compare RBC, independent SAC, MARLISA, and carbon-aware MARLISA in one place. The notebook displays experiment settings, district KPIs, improvement relative to RBC, building-level carbon, comfort fairness, and the carbon-comfort trade-off.

Or run the carbon-aware experiment from the command line:

```shell
python src/run_carbon_aware_marlisa.py --episodes 3
```

The custom reward is `-max(net_electricity_consumption, 0) * carbon_intensity` for each building and time step. It penalizes grid imports more heavily during carbon-intensive periods and gives no credit for export, consistent with CityLearn's emissions KPI. The only intended change from standard MARLISA is the reward; its controller and training settings are held fixed. Results are stored under `outputs/carbon_aware_marlisa/`.

## Fair carbon-aware MARLISA

Open [`notebooks/05_Fair_Carbon_Aware_MARLISA.ipynb`](notebooks/05_Fair_Carbon_Aware_MARLISA.ipynb) for the complete five-controller comparison and written discussion, or run from the command line:

```shell
python src/run_fair_carbon_aware_marlisa.py --episodes 3
```

The reward combines local carbon emissions, each building's occupied temperature loss outside a 1°C band, and a shared penalty for the worst current building plus the worst-best comfort gap. Default weights are 1.0 for carbon, 0.25 for local comfort, and 0.25 for fairness. Results and the complete run configuration are saved under `outputs/fair_carbon_aware_marlisa/`.

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
