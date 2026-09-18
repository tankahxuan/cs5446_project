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

Use **Python 3.11** and CityLearn **3.0.2**. From the project folder in a PowerShell terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, install the Python and Jupyter extensions, open a notebook, then select **Kernel → Python Environments → .venv**. The virtual environment is excluded from Git by `.gitignore`; each collaborator creates their own.

Start with [`notebooks/00_Getting_Started.ipynb`](notebooks/00_Getting_Started.ipynb). It uses CityLearn's `citylearn_challenge_2023_phase_2_local_evaluation` dataset for initial exploration. This pilot dataset is not yet the final experimental choice.

## Repository contents

| Path | Purpose |
| --- | --- |
| `notebooks/00_Getting_Started.ipynb` | Initial CityLearn exploration |
| `requirements.txt` | Python packages for the project |
| `.gitignore` | Excludes local environments and generated files |

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
