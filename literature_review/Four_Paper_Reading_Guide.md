# Four core papers for the CityLearn project

Reference set requested by the user on 12 October 2026. Notes are based on the supplied PDFs, including rendered figures and tables. Use these papers as the core reading set for this project. This is a reference record, not a claim about memory outside this workspace or conversation.

## 1. Original CityLearn framework

Source: `C:/Users/tanka/Dropbox/08-1 NUS Master in Computing AI specialization/CS5546 AI Planning and Decision Making/03 Project/CityLearn - Standardizing Research in Multi-Agent Reinforcement Learning for Demand Response and Urban Energy Management.pdf`

Authors shown: Jose R. Vazquez-Canteli, Sourav Dey, Gregor Henze, Zoltan Nagy. 11 PDF pages. Do not confuse this paper with the separate 2019 two-page CityLearn v1.0 conference paper.

- Main contribution: a common Gym-compatible simulation and evaluation environment for district demand response, supporting central and decentralized agents, customizable rewards, and thermal/electrical storage.
- Important assumption: building loads and temperatures are pre-simulated; storage control does not change indoor temperature. Backup logic ensures thermal demand is met. Section 2.1 and the action-constraint discussion explain this.
- States: weather/time, load and solar generation, storage charge, and configurable building information. Actions primarily change active storage; the challenge version did not include batteries or net-electricity-consumption observations.
- Energy storage shifts load; conversion and storage losses must be modeled.
- Evaluation: ramping, one minus load factor, average daily peak, peak demand, total net electricity, with quadratic demand also provided but excluded from the 2020 challenge.
- Challenge: four climates, nine buildings per district, design/evaluation/final-challenge datasets. Scores normalized to RBC. Table 1 winner score 0.879, second-place ProtoBuds 0.967. These are composite scores, not carbon reductions.
- Project relevance: shared schema and controlled comparisons; record the metric denominator. The current project uses a newer simulator and direct cooling control, so the original comfort guarantee must not be assumed.
- Read first: Sections 2.1-2.3, Figure 7 and Table 1 on PDF p.7, Appendix A.5-A.8.

## 2. CityLearn v2

Source: `C:/Users/tanka/Dropbox/08-1 NUS Master in Computing AI specialization/CS5546 AI Planning and Decision Making/03 Project/CityLearn v2 - Energy-flexible, resilient, occupant-centric, and carbon-aware management of grid-interactive communities.pdf`

Lead author Kingsley Nweye, with 14 coauthors. Supplied file is arXiv:2405.03848v1, 2 May 2024, a preprint with 28 PDF pages. Distinguish the uploaded preprint from later published versions.

- Main contribution: expanded benchmark supporting thermal dynamics, occupant thermostat overrides, power outages/resilience, carbon signals, EVs, and EULP-based community creation.
- LSTM thermal model predicts indoor temperature from temperature history, weather, thermal load, solar irradiance, occupancy, and time. Its training model is distinct from the RL policy network.
- Occupant model: preference clustering, logistic override probability, and override-magnitude modeling. Not all datasets enable every module.
- Figure 3 on PDF p.8 distinguishes central single-agent, independent multi-agent, and coordinated multi-agent configurations. RBC/MPC/RL can use the same environment.
- Results are examples across distinct tasks, not one universal benchmark ranking. Cost/emissions example uses a 17-day period, with weak emissions variation; RLC cost reduction up to 6.7% versus RBC and little additional emissions reduction.
- Table 2 on PDF p.11: overcooling-penalty coefficient m=12 lowers electricity by 12.0% but raises the listed aggregate discomfort measure by 98.0%, relative to m=1. This discomfort column uses degree-hours, not the project's discomfort-proportion KPI.
- Limitations: surrogate and simplified device models, occupant models from Quebec winter data, location-dependent performance, observation feasibility, limited real-world validation.
- Project relevance: carbon/comfort trade-offs; inspect the actual schema, action meanings, comfort band, occupancy definition, and evaluation reference. Cross-building fairness is a project objective and is not established as solved by this paper.
- Read first: Sections 2.1.5, 2.1.6, 2.3, Figure 3, Section 4.1.3/Table 2, Section 5, appendices on actions/rewards/KPIs.

## 3. Soft Actor-Critic

Source: `C:/Users/tanka/Dropbox/08-1 NUS Master in Computing AI specialization/CS5546 AI Planning and Decision Making/03 Project/Soft Actor-Critic - Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.pdf`

Haarnoja, Zhou, Abbeel, Levine. Supplied arXiv:1801.01290v2, 8 August 2018. 14 PDF pages.

- Problem: sample inefficiency and unstable learning for continuous control.
- Method: off-policy stochastic actor-critic maximizing reward plus policy entropy. Alpha controls the relative entropy weight; replay reuses experiences; two critics reduce overestimation; slowly updated targets stabilize learning.
- Original formulation includes actor, two Q networks, separate soft V network and target V. The installed CityLearn SAC uses actor, two Q critics and target Q critics without separate V, so do not claim identical architectures.
- Algorithm 1 on PDF p.5 separates environment interactions from minibatch learning updates. The soft-policy-iteration convergence theory has assumptions; it is not a blanket convergence guarantee for neural implementations.
- Experiments: continuous-control locomotion benchmarks; DDPG, PPO, SQL, TD3 comparisons; five seeds and evaluation every 1,000 environment steps. Figure 1 bands represent min-max across seeds, not confidence intervals.
- Ablations support stochastic training/entropy and deterministic mean-action evaluation. Reward scale is influential.
- Appendix D: learning rate 0.0003, gamma 0.99, 2x256 hidden layers, batch 256, tau 0.005; original replay capacity 1 million. Shared settings do not establish optimality for CityLearn.
- Project relevance: SAC baseline and MARLISA's learning backbone; track transitions and held-out deterministic KPIs, report multiple seeds and accurately defined variation, document reward normalization.
- Read first: Section 3.2/Eq.1, Section 4.2/Algorithm 1, Figures 1-3, Section 5.2, Appendix D.

## 4. Centralised SAC for CityLearn

Source: `C:/Users/tanka/Dropbox/08-1 NUS Master in Computing AI specialization/CS5546 AI Planning and Decision Making/03 Project/A Centralised Soft Actor Critic Deep Reinforcement Learning Approach to District Demand Side Management through CityLearn.pdf`

Kathirgamanathan, Twardowski, Mangina, Finn. RLEM 2020, pp.11-14; DOI 10.1145/3427773.3427869. Four PDF pages.

- One centralized SAC controller observes and controls all nine buildings, mainly cooling and DHW storage. This is distinct from the project's independent or coordinated building policies.
- Objective: equal-weighted five-metric district load-shaping score normalized against a manually tuned RBC; no explicit carbon or comfort objective.
- Reward: district/load-dependent virtual price and strong hand-shaped nighttime charging/daytime discharging terms, scaled/clipped to [-1,1]. The printed Eq.1 has a positive consumption term and ambiguous clock notation; verify implementation before reproducing it.
- State selection uses calendar/weather, load, PV, cooling/DHW storage charge. Table 4: deployment changes learning rate and minibatch size, allowing adaptation; do not describe results as unequivocal frozen-policy zero-shot evaluation.
- Results: below RBC within roughly 10 episodes in the presented training case; design climate 1 score 0.893; average final challenge score 0.967, second place, a 3.3% improvement in the composite score versus RBC.
- Table 5 challenge climate 1: ramping 0.779, peak demand 1.131, net electricity 1.015, composite 0.984. Improved composite does not guarantee improvement in each constituent metric; net electricity exceeds RBC in all four challenge climates.
- Limitations: manual shaping and generalization, unknown scaling beyond nine buildings, original environment's precomputed cooling and comfort limitations, no carbon/fairness evaluation.
- Project relevance: useful centralized baseline motivation, reward design and KPI separation; not evidence that MARLISA outperforms central SAC, nor direct validation of carbon/comfort fairness.
- Read first: Section 3.1, Tables 3-4, Section 3.3, Figure 1, Table 5, limitations in Section 5.

## Synthesis for the project

The original CityLearn paper motivates reproducible benchmarking; v2 supports carbon and occupant-sensitive simulation; SAC supplies continuous-control learning; centralized SAC demonstrates a district application and the limitations of aggregate scores. Together they motivate controlled experiments examining carbon, average and worst-building discomfort, and between-building gaps. They do not establish novelty of fairness-aware building control or general superiority of one architecture. A separate MARLISA paper is needed for that algorithm's specific contribution.
