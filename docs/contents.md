# docs/

## initial_design.tex

Initial academic LaTeX design document covering problem statement, MDP formalisation,
architecture, timeline, and future work. This is a draft awaiting supervisor review;
implementation docs in `code/` derive from it.

## code/

| File | Description |
|------|-------------|
| `codebase_plan.md` | Top-level architecture, 8-week Phase 1 scope, tech stack, confirmed design decisions |
| `code_design.md` | Detailed software design: module structure, ABC interfaces, validation layers, logging spec |
| `phase_1_execution_plan.md` | Dependency graph (Tracks A/B/C), gates summary, risk register |
| `subphase_1a_data_layer.md` | Data pipeline, Dataset ABC, feature engineering, synthetic data generation |
| `subphase_1b_mdp_environment.md` | MDP environment, TransitionModel, state/action/reward shaping |
| `subphase_1c_user_simulation.md` | User profiling, ResponseModel, archetype simulation |
| `subphase_1d_agent_library.md` | Agent ABC, Thompson Sampling, extension interface |
| `subphase_1e_experiment_runner.md` | Experiment orchestration, CLI, result tracking, checkpoint/resume |

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
