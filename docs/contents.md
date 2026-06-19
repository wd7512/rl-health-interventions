# docs/

## design/

| File | Description |
|------|-------------|
| `initial_design.tex` | Academic LaTeX design document: problem statement, MDP formalisation (14 state variables, 6 actions), 4 user archetypes, multi-timescale reward. Long-term vision — current MVP is a deliberate simplification. |

## mvp/

| File | Description |
|------|-------------|
| `mvp_specification.tex` | MDP formulation for the current binary-state, binary-action simulator with transition probabilities, reward function, agent formulations, and 10-seed results table. |

## initial_experiments/

| File | Description |
|------|-------------|
| `initial_experiments.tex` | Scaffold for state space, action space, and reward function extensions bridging MVP to the full design vision. Sections to be filled by Issues #124, #126, #132. |
| `chart_generator.py` | Stub — generate charts for initial experiments results. |
| `benchmark_all_agents.py` | Stub — benchmark all agents on initial experiments configs. |
| `images/` | Charts and figures for initial experiments. |

## learned/

| File | Description |
|------|-------------|
| `learned_specification.tex` | Placeholder — learned transition models from real wearable data. See `config/learned.yaml`. |

## llm/

| File | Description |
|------|-------------|
| `llm_specification.tex` | Placeholder — LLM-based user simulation. See `config/llm_based.yaml`. |

## plans/

| File | Description |
|------|-------------|
| `README.md` | Overview of the archive; links to current docs |
| `ROADMAP.md` | Milestone roadmap (M-01 through M-10), success metrics, critical path |
| `ROADMAP_TECHNICAL_DEPS.md` | Technical dependency graph (Mermaid), risk register (R-01 to R-13), TRL table |
| `subphase_1a_data_layer.md` | Phase 2: config-driven data layer, FeaturePipeline, Dataset ABC |
| `subphase_1c_user_simulation.md` | Phase 2: UserProfile (4 archetypes), ResponseModel, dataset exploration |
| `stale/` | Superseded implementation blueprints (code_design, subphase plans, issue-101-mvp) |
| `2026-06-14_research-assistant-plan.md` | Research PR plan (statistical analysis, decision trees) |

## sources/

| File | Description |
|------|-------------|
| `data_sources.md` | Feasibility study of All of Us and UK Biobank; MDP variable gap analysis; recommends synthetic data for Phase 1 |
| `additional_data_sources.md` | Survey of 12 JITAI trial and wearable datasets; gap analysis matrix; priority-ordered recommendations |
| `data_availability_schema.md` | Schema reference and credential setup for all 14 datasets; access status, column types, and EDA instructions |
| `all_of_us_fitbit_dataset.md` | Deep dive into All of Us Fitbit BigQuery schema (7 tables); SQL examples; MDP variable mapping |
| `bidsleep_sleep_staging.md` | Apple Watch HR/accel + Dreem 2 EEG gold-standard sleep labels; multi-night variability |
| `4tu_coaching_datasets.md` | 5 open-access coaching intervention datasets with step response data; only open-access intervention response data available |
| `dreamt_dataset.md` | 100 participants, Empatica E4 wristband + polysomnography; largest open wearable+sleep dataset |
| `fitbit_fitness_tracker.md` | Kaggle CC0 dataset (33 users); immediate download; synthetic data parameterization |
| `harth_dataset.md` | Multi-demographic activity recognition (adults, children, older adults, wrist variant); MIT license |
| `mhealth_dataset.md` | 10 subjects, chest/wrist/ankle sensors with ECG; multi-body sensor placement validation |
| `pmdata_multi_source_fitness.md` | Garmin + WHOOP + HRV4Training + Apple Health (1 GB); largest open multi-source fitness dataset |
| `scientisst_move_biosignals.md` | Multimodal biosignals during naturalistic everyday activities; ecological validity for synthetic data |
| `stepcount_jitai_dataset.md` | NeurIPS 2024 Workshop; RL simulation environment for PA JITAIs built from real MRT data |
| `wesad_dataset.md` | 15 subjects, chest + wrist sensors; stress/affect labels for burden and fatigue modelling |
