---
title: "PEARL Deep Analysis — Simulator Design Reference"
status: "analysis complete"
date: "2026-07-23"
paper: "Lee, A. A., Hegde, N., Deliu, N., et al. (2025). A Personalized Exercise Assistant using Reinforcement Learning (PEARL): Results from a four-arm Randomized-controlled Trial. arXiv:2508.10060."
purpose: "Extract all figures, tables, and experimental details from the PEARL paper to inform PEARL-matched simulator config design."
related: "pearl-rct-2025.md · pearl-constitution.md · pearl_reference.json"
---

# PEARL Deep Analysis — Simulator Design Reference

## 1. Decision Point Timing

**Confirmed: 1 decision point per day, delivered at either morning (6 AM) or afternoon (3 PM)**

- 12 distinct action classes = 6 nudge themes × 2 delivery times
- One text-only in-app push notification per day (delivered at either 6 AM or 3 PM, not both)
- Nudges are dismissible and presented once per day
- All delivered nudges remain accessible on the study home page throughout the day

**Simulator impact:** Must model 1 decision point per day with a choice of delivery time, not 5 steps. The current config uses `steps_per_day: 5` which is incorrect for PEARL. Set `steps_per_day: 1` while preserving the 12 action classes as theme/time combinations.

## 2. Nudge Bank

**155 nudges in the bank, ~180 total messages (including LLM-generated supplements)**

| COM-B Theme | Initial Messages | LLM-Supplemented | Total |
|-------------|------------------|------------------|-------|
| Ability | ~20 | ~10 | ~30 |
| Perceived Benefit | ~20 | ~10 | ~30 |
| Physical Opportunity | ~20 | ~10 | ~30 |
| Planning | ~20 | ~10 | ~30 |
| Prioritization | ~20 | ~10 | ~30 |
| Social Opportunity | ~20 | ~10 | ~30 |

**Nudge content:** Text-only in-app push notifications. Upon selection of an action class, a specific message was randomly sampled from the corresponding theme's repository.

**Simulator impact:** The action space is 12 discrete actions (6 themes × 2 times), not the current 4 actions (idle, movement_suggestion, goal_reminder, journal). Each action should map to a COM-B theme.

## 3. Intervention Types (COM-B Framework)

The six nudge themes map to COM-B components:

| COM-B Component | Sub-Construct | Theme | Thumbs-Up Rate | RL Allocation |
|-----------------|---------------|-------|----------------|---------------|
| **Capability** | Psychological | Ability | 90% | 27.0% |
| **Motivation** | Reflective | Perceived Benefit | 88% | 15.4% |
| **Motivation** | Reflective | Planning | 85% | 14.1% |
| **Motivation** | Reflective | Prioritization | 84% | 12.8% |
| **Opportunity** | Social | Social Opportunity | 83% | 15.1% |
| **Opportunity** | Physical | Physical Opportunity | 79% | 15.6% |

**Key finding:** The RL agent allocated 27% of nudges to Ability vs. the Fixed arm's 11.8%, showing a distinct preference pattern. Ability-themed nudges also received the highest favorability rates in user feedback (90% thumbs-up).

**Simulator impact:** Theme-specific reward weights are needed. Physical Opportunity nudges should have lower acceptance probability than Ability nudges.

## 4. RL Algorithm Details

### Algorithm Type
**Contextual Multi-Armed Bandit (C-MAB) with ε-greedy exploration**

- **Exploration strategy:** ε-greedy with ε = 0.7 or 0.8
  - With probability ε, follow the policy's recommendation (i.e., the action that maximizes expected reward)
  - With probability 1-ε, choose uniformly at random from the set of 12 possible actions
  - A lower ε leads to higher exploration; ε=0.7 is used when the model has high variability across ensemble models
- **Reward prediction model:** XGBoost decision trees
- **Policy evaluation:** Importance sampling (unbiased estimator of policy value)
- **Optimization:** Empirical risk minimization oracle

### State Variables (Table 7 from paper)

| Feature Class | State Class | Cadence |
|---------------|-------------|---------|
| **User Demographics** | Age, Gender, Device Type | Static: Start of study |
| **Education, Area type, weather** | Start of study | Static |
| **User COM-B Survey** | 20 survey questions (1-5 Likert) | Static: Start of study |
| **User Pre-study steps** | Mean and standard deviation | Static: Start of study |
| **Pre-study walk pattern** | Low/High walker, Regular/Irregular across days of week | Static: Start of study |
| **Recent steps mean/std** | Last 7 days | Dynamic |
| **Slope of regression line** | Fitting recent steps | Dynamic: Last 7 days |
| **Daily nudge feedback stats** | Thumbs up/down counts | Dynamic: Last 7 days |
| **Day of the week** | Each day | Dynamic |
| **Recent Study Walk Pattern** | Low/High walker, Regular/Irregular | Dynamic: Last 7 days |
| **Avg steps Morning/Evening** | Morning (<12pm) vs Evening (>12pm) | Dynamic: Last 7 days |
| **Percentage of missing data** | In steps | Dynamic: Last 7 days |
| **Past Week Nudge History** | Actions taken | Dynamic: Each day |

### Action Space
- **12 discrete actions** = 6 nudge themes × 2 delivery times (morning, afternoon)
- Action = (COM-B theme, delivery time)

### Reward Function
**Relative change in steps walked between nudge delivery and the following 24 hours, relative to individual pre-study baseline walking pattern.**

Formally:
- Let `M` = morning nudge indicator, `W` = weekday indicator
- Baseline `B(M,W)` = average daily step count within 24-hour period starting from morning for user, calculated over 30-day pre-study window for each (M,W) combination
- Reward = `(steps_in_24h_after_nudge - B(M,W)) / B(M,W)`

### Deployment Pipeline
1. **12 AM:** Cron starts
2. **12-2 AM:** Data collection (90 mins)
3. **2-3 AM:** Reward computation (30 mins)
4. **3-4 AM:** Dynamic feature update (30 mins)
5. **4-5 AM:** Model inference (30 mins)
6. **5 AM:** Model decision finalized
7. **6 AM / 3 PM:** Nudge delivered

**Simulator impact:** The RL algorithm is a contextual bandit with ε-greedy exploration, not Thompson Sampling. The current config uses `type: thompson_sampling` which needs to be changed. Also, the ε-greedy convention is: ε = exploit (follow policy), 1-ε = explore (random action).

## 5. Four-Arm Design

| Arm | N (randomized) | N (mITT) | Description | Agent Type |
|-----|----------------|----------|-------------|------------|
| **Control** | 3,304 | 1,813 | No nudges | FixedAgent(action="idle") |
| **Random** | 3,337 | 1,902 | Nudges selected uniformly at random from 12 actions | RandomAgent |
| **Fixed** | 3,419 | 2,003 | Nudges selected by COM-B survey logic | FixedAgent (rule-based) |
| **RL** | 3,403 | 1,993 | Nudges selected by C-MAB algorithm | ContextualBandit |

### Fixed Arm Logic
- **Content selection:** Barrier score = 5 - Likert score for each COM-B theme. Theme sampled from multinomial distribution weighted by barrier scores (higher barrier = more likely to receive nudge for that theme).
- **Timing selection:** Based on stated preference (morning, afternoon, no preference). 70% preferred time, 30% other; 50/50 for no preference.
- **Static policy:** Derived solely from initial survey responses, no adaptation.

### Random Arm Logic
- Uniform random selection from 12 actions (6 themes × 2 times)

### RL Arm Logic
- ε-greedy C-MAB with XGBoost reward models
- Learns population-level preferences (e.g., Ability theme, evening timing)
- Adapts to individual trajectories over time

**Simulator impact:** The current config maps Fixed to `movement_suggestion` but PEARL's Fixed arm uses survey-based logic, not a single action. Need a rule-based agent that selects based on COM-B barrier scores.

## 6. All Tables Extracted

### Table 1: Literature Review
Comparative overview of mHealth studies (HeartSteps, DIAMANTE, mSTAR, etc.). PEARL is the largest (N=7,711) and longest (60 days) RL-for-PA RCT.

### Table 2: Baseline Characteristics (N=7,711)

| Characteristic | Control (N=1,813) | Random (N=1,902) | Fixed (N=2,003) | RL (N=1,993) |
|----------------|-------------------|------------------|-----------------|--------------|
| Age (years) | 42.0 (9.1) | 42.1 (9.0) | 42.0 (8.8) | 42.2 (9.0) |
| Female | 1,578 (87.0%) | 1,641 (86.3%) | 1,705 (85.1%) | 1,733 (87.0%) |
| Weight (kg) | 91.1 (29.5) | 90.8 (23.6) | 90.9 (24.9) | 90.4 (23.8) |
| Baseline avg daily steps | 7,248 (3,064) | 7,160 (3,004) | 7,170 (3,089) | 7,171 (3,073) |
| Urban | 261 (14.4%) | 295 (15.5%) | 311 (15.5%) | 301 (15.1%) |
| Suburban | 1,153 (63.6%) | 1,216 (63.9%) | 1,267 (63.3%) | 1,240 (62.2%) |
| Rural | 376 (20.7%) | 375 (19.7%) | 408 (20.4%) | 435 (21.8%) |

**Note:** Baseline steps in Table 2 (7,160-7,248) differ from the abstract (5,618.2) and Table 3 (5,580). The source of this discrepancy is unresolved — it may reflect different populations (ITT vs mITT) or different baseline window definitions. The arm counts and other values in Table 2 are retained as reported.

### Table 3: Step Count Across Study Phases

| Phase | Control | Random | Fixed | RL |
|-------|---------|--------|-------|-----|
| Baseline | 5,580.0 (1,499.0) | 5,574.4 (1,499.6) | 5,580.6 (1,517.0) | 5,580.4 (1,495.0) |
| Month 1 | 6,050.0 (1,922.0) | 6,061.2 (1,897.5) | 6,054.8 (1,875.9) | 6,282.2 (1,956.0) |
| Month 2 | 5,948.6 (2,076.9) | 5,947.6 (2,068.8) | 5,969.4 (2,016.1) | 6,096.3 (2,117.4) |

### Table 4: Difference-in-Differences Regression Results

| Comparison | Month 1 Δ (SE, p) | Month 2 Δ (SE, p) |
|------------|-------------------|-------------------|
| RL vs Control | +295.9 (79.6, p=0.0002) | +210.1 (83.8, p=0.0122) |
| RL vs Random | +218.3 (78.5, p=0.005) | +147.0 (82.8, p=0.076) |
| RL vs Fixed | +238.4 (77.6, p=0.002) | +135.0 (81.5, p=0.098) |
| Random vs Control | +77.6 (79.7, p=0.330) | +63.1 (84.0, p=0.452) |
| Fixed vs Control | +57.5 (78.8, p=0.465) | +75.1 (82.7, p=0.364) |
| Fixed vs Random | -20.1 (77.7, p=0.796) | +12.0 (81.7, p=0.883) |

### Table 6: GEE Model Results

| Covariate | Estimate | 95% CI | p-value |
|-----------|----------|--------|---------|
| (Intercept) | 6,125.2 | (6,028.1, 6,222.4) | <0.001 |
| Treatment Arm: Random | -14.1 | (-148.1, 119.8) | 0.836 |
| Treatment Arm: Fixed | -18.7 | (-151.6, 114.3) | 0.783 |
| Treatment Arm: RL | 208.0 | (73.6, 342.4) | 0.002 |
| Day in study | -4.5 | (-6.6, -2.5) | <0.001 |
| Random × study day | 0.5 | (-2.3, 3.4) | 0.707 |
| Fixed × study day | 1.3 | (-1.5, 4.1) | 0.373 |
| RL × study day | -0.6 | (-3.4, 2.2) | 0.696 |

### Table 7: RL Feature Set
See Section 4 (RL Algorithm Details) above for full feature list.

## 7. All Figures Analyzed

### Figure 1: CONSORT Flow Diagram
- 15,000 assessed → 13,463 randomized → 7,711 in mITT analysis
- Withdrawal rates: Control 13.5%, Random 10.5%, Fixed 10.8%, RL 10.4%
- Control group had higher attrition (likely due to no nudges)

### Figure 2: Step Count Trajectory (Monthly)
- All arms start at ~5,580 steps/day
- RL peaks at ~6,290 at Month 1, decays to ~6,090 at Month 2
- Control gains ~370 steps from baseline (placebo/observation effect)

### Figure 3: Daily Step Trajectory
- RL group separates from other groups after ~5-7 days (learning curve)
- Weekly cyclical pattern visible (weekday/weekend effects)
- All groups converge toward end of study (Days 55-60)
- Day-to-day fluctuations ±200-300 steps

### Figure 4: COM-B Survey Results
- RL group: 37% "Very useful" vs Fixed 24.5%, Random 26.7%
- ~95% of all respondents found nudges at least "Somewhat useful"

### Figure 5: Change in COM-B Survey
- Shows changes in COM-B scores from baseline to study exit

### Figure 6: Exit Survey — Would Recommend Nudges
- RL: 77.6%, Random: 74.0%, Fixed: 70.6%, Control: 48.9%

### Figure 7: Exit Survey — Nudge Customization
- RL: 37%, Random: 26.7%, Fixed: 24.5%, Control: 10.8%

### Figure 8: Nudge Feedback by COM-B Theme
- Ability: 90% thumbs-up, Physical Opportunity: 79% thumbs-up
- Clear gradient: Capability > Motivation > Opportunity

### Figure 9: Nudge Content Distribution
- RL allocated 27% to Ability (vs 17% random baseline), showing a distinct preference pattern
- Fixed: 24.3% Planning, 21.5% Physical Opportunity

### Figure 10: Nudge Timing Distribution
- RL delivered 7% more evening nudges than Random group (per paper text)
- The paper reports delivery windows at 6 AM and 3 PM; timing allocation varies by arm

### Figure 11: Nudge Distribution Over Time
- RL initially prioritizes Perceived Benefit and Planning
- Shifts to Ability theme in later stages (temporal adaptation)

### Figure 12: Most Influential Features in RL Model
- Top features: pre-study baseline steps, weight, age, previous day step count

### Figure 13: RL Deployment Pipeline
- Pipeline runs daily at midnight, model decision by 5 AM
- Nudge delivered at 6 AM or 3 PM
- Per-user inference (not batched)

## 8. Comparison Table: PEARL Design vs Current Simulator

| Design Dimension | PEARL (Paper) | Current Simulator | Gap |
|------------------|---------------|-------------------|-----|
| **Decision points/day** | 1 (delivered at either 6 AM or 3 PM) | 5 steps/day | Major mismatch |
| **Action space** | 12 (6 themes × 2 times) | 4 (idle, movement_suggestion, goal_reminder, journal) | Need 12 actions |
| **Nudge bank size** | 155 (~180 messages) | 4 actions | Need COM-B themes |
| **RL algorithm** | C-MAB with ε-greedy (ε=0.7-0.8) | Thompson Sampling | Different algorithm |
| **Episode length** | 60 days | 60 days | Match |
| **Baseline period** | 30-day pre-study window | 7 days (35 steps) | Different definition |
| **Arms** | 4 (Control, Random, Fixed, RL) | 4 (idle, random, fixed, thompson) | Similar structure |
| **Fixed arm logic** | COM-B barrier-score weighted | Always movement_suggestion | Major mismatch |
| **Random arm** | Uniform over 12 actions | Uniform over 4 actions | Need 12 actions |
| **Reward** | Relative step change vs individual baseline | alpha * step_bin + (1-alpha) * sleep - penalty | Different formula |
| **State variables** | 20+ features (static + dynamic) | 4 state variables (step_bin, sleep, day_of_week, burden) | Major gap |
| **Personas** | N/A (real participants) | 5 personas (base, goal_driven, etc.) | Different approach |
| **Attrition** | 1,522 withdrew (11.3%); 7,711 of 13,463 met inclusion criteria (57.3%) | Not modeled | Need attrition model |
| **Weekly seasonality** | Confirmed (weekday/weekend) | day_of_week state variable | Partially matched |
| **Time-of-day effects** | RL delivered 7% more evening nudges than Random | Not modeled | Need timing |
| **Exploration** | Explicit ε-greedy | Thompson Sampling (implicit) | Different mechanism |
| **Feature engineering** | Static (COM-B, demographics) + Dynamic (7-day windows) | Minimal | Major gap |
| **Baseline steps** | 5,618 (mITT), 7,160-7,248 (ITT) | Configurable | Can match 5,618 |

## 9. Open Questions for Config Design

1. **Action space mapping:** Should we map PEARL's 12 actions to our existing actions, or redesign the action space entirely?
2. **State representation:** How much of PEARL's 20+ feature set should we implement? COM-B survey requires new data collection.
3. **RL algorithm:** Should we implement ε-greedy C-MAB or keep Thompson Sampling? The paper explicitly chose ε-greedy over other options.
4. **Fixed arm implementation:** The fixed arm requires COM-B survey data and barrier-score weighting. Should we implement this or simplify?
5. **Attrition modeling:** Should we model dropout as a function of nudge receipt (as suggested by the CONSORT diagram)?
6. **Reward function:** Should we use PEARL's relative step change formula or our current weighted formula?
7. **Baseline period:** Should we use 30-day pre-study baseline (PEARL) or 7-day baseline (current)?
8. **Nudge message generation:** Should we generate ~180 text messages for the 6 COM-B themes, or use symbolic representations?

## 10. Recommended Next Steps

1. **Create PEARL-matched config** (Issue #252) with:
   - 1 decision point per day with choice of delivery time (6 AM or 3 PM)
   - 12 actions (6 COM-B themes × 2 times)
   - ε-greedy C-MAB agent (not Thompson Sampling)
   - PEARL's reward formula (relative step change)
   - 60-day episodes
   - Baseline period: 30-day pre-study

2. **Generate COM-B nudge messages** for the 6 themes (~30 messages per theme)

3. **Implement COM-B barrier-score fixed agent** for the Fixed arm

4. **Add attrition model** based on CONSORT diagram data

5. **Test with small-scale experiments** (Issue #253) before full recalibration

---

*Analysis completed 2026-07-23. Source: Lee et al. (2025) arXiv:2508.10060, 27-page PDF.*
