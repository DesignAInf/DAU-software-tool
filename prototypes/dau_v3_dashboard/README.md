# DAU v3 — Active Inference Persuasive Design Study

## How to run

```bash
pip install -r requirements.txt
python server_v3.py
```

Then open `http://127.0.0.1:5000` in your browser. CLI: `python -m dau_v3.main --steps 300 --seed 0` and `python -m dau_v3.analysis --steps 300 --n_seeds 20`.

---

An extended Active Inference simulation studying how **Designer and Smartphone strategies differentially manipulate users with different psychological profiles**. Four user archetypes run in parallel under shared environmental conditions, revealing which persuasive techniques are most effective on which profiles. Built on the same numpy-only Active Inference core as DAU v2, extended to 10 hidden states per agent, richer observation spaces, and a full statistical analysis pipeline.

**DAU v3 — Model Description**

DAU v3 is a computational simulation based on Active Inference that examines how mobile app design choices affect users with different psychological profiles. The model formalizes the relationship among three actors — Designer, Smartphone, and User — as agents that continuously update their beliefs about the world and choose actions that minimize expected surprise.

The three agents
Each agent has hidden mental states, partial observations of the world, and preferences about what it wants to observe. At each timestep, it updates its beliefs about the state of the world and chooses the action that maximizes Expected Free Energy (EFE) — that is, the action that best balances uncertainty reduction with the satisfaction of its own preferences.

Designer represents the company developing the app. It has 10 internal states that encode design philosophies ranging from aggressive growth hacking to minimal ethical compliance. It observes aggregated signals such as engagement, retention, complaints, and regulatory flags. Its preferences are business-driven: it seeks high engagement and tries to avoid regulatory complaints.

Smartphone represents the app interface — the notification and recommendation algorithm. It has 10 operating modes ranging from constant bombardment to wellbeing mode. It observes the user’s immediate behavior (clicks, time spent, purchases, churn) and has preferences strongly oriented toward short-term KPIs.

User represents the real user. It has 10 cognitive-emotional states ranging from Focused Calm to Addicted and Disengaged. It observes its own subjective experience (mood, task completion, notification quality). Unlike the other two agents, it starts with an empty model — low precision, high uncertainty — and builds that model over time through experience.

The main innovation in v3 is that there is not just one user: four instances with different preferences run in parallel under the same Designer and Smartphone, revealing the differential effects of persuasive strategies. Achiever — productivity-oriented and motivated by streaks. This profile evaluates everything in terms of time well spent and tasks completed. It is vulnerable to reward loops and streak pressure. Social — seeks validation and is prone to FOMO. The most important signal for this profile is social reward. It is vulnerable to social amplification and peak targeting. Anxious — starts from a high baseline level of stress and uses the phone as an escape. This profile primarily wants to improve its mood. It is vulnerable to variable rewards and notifications delivered during moments of vulnerability. Resistant — has high media literacy and is aware of manipulative mechanisms. This profile is strongly averse to wasted time and intrusive notifications. It responds positively to transparency and ethical design.

**How they interact**
The environment couples the three agents with varying strength. The Designer influences the Smartphone’s default settings (coupling 0.60): an aggressive philosophy pushes the Smartphone toward bombardment and variable-reward modes. The Smartphone, in turn, influences the user’s cognitive state (coupling 0.50): aggressive modes tend to push the user toward stress, habit, or dependency, while calmer modes leave more room for focus and resistance.

The user is not passive: their actions (ignoring prompts, changing settings, enabling Do Not Disturb, uninstalling the app) modify the environment and therefore the observations the Smartphone receives.

**What the model measures**
The model does not directly measure external behavior, but rather the quality of each agent’s decisions relative to its own goals, through the EFE selected at each timestep. Less negative EFE values indicate that the agent is better able to satisfy its preferences.

In the statistical analysis across N seeds, the aggregate metrics are: churn rate (the percentage of runs in which the user converges toward dependency or burnout states), stress index (the fraction of time spent in stress states), time to resistance (how many timesteps it takes for the user to reach a defensive state), and entropy reduction (how much more decisive the user becomes over time).

**Why it is scientifically interesting**
The model makes it possible to test hypotheses about the selectivity of manipulation: the same Smartphone strategy does not have the same effect on all users. A high γ for the Smartphone (high precision, aggressive KPI-driven behavior) may accelerate dependency in the Anxious profile while also accelerating resistance in the Resistant profile — producing a boomerang effect in which aggressiveness generates churn rather than retention.

---

## What's New in v3

| Feature | DAU v2 | DAU v3 |
|---------|--------|--------|
| Hidden states per agent | 3 | **10** |
| User profiles | 1 (generic) | **4 in parallel** |
| User observations | 6 | **10** |
| Smartphone policies | 6 | **12** |
| Analysis mode | Single run | **Single run + N-seed statistical analysis** |
| User preferences | Fixed | **Profile-specific** |

---

## The Four User Profiles

Each profile shares the same A and B matrices but has distinct preference vectors (C), initial state distributions (D), and precision annealing schedules.

### Achiever
Productivity-oriented, streak lover. Values task completion and time well spent above all. Starts relatively decisive (γ_init = 0.8) and converges strongly (γ_final = 6.0).
**Vulnerable to:** reward loops, streak pressure.

### Social
Validation-seeking, FOMO-prone. Highly reactive to social signals and comparison. Starts most uncertain (γ_init = 0.4).
**Vulnerable to:** social amplification, peak targeting.

### Anxious
High baseline stress, uses the phone as an escape. Prioritises mood improvement above all. Starts very uncertain (γ_init = 0.3) and converges slowest (γ_final = 3.5).
**Vulnerable to:** variable reward, peak targeting.

### Resistant
High media literacy, aware of manipulation. Strongly averse to wasted time and annoying notifications. Starts most decisive (γ_init = 1.2) and converges very strongly (γ_final = 7.0).
**Responds positively to:** transparency, ethical nudging.

---

## Agent Design

### Designer — 10 states, 6 policies, 6 observations

**States:** GROWTH_HACKING · ATTENTION_ECONOMY · SOCIAL_PRESSURE · REWARD_LOOPS · PERSONALIZATION · FRICTION_REDUCTION · ETHICAL_NUDGING · TRANSPARENCY · USER_EMPOWERMENT · REGULATION_COMPLIANT

**Policies:** notification intensity (LOW/MED/HIGH) × friction-to-disable (LOW/HIGH) → 6 combinations

**Observations:** ENGAGEMENT_HIGH · ENGAGEMENT_LOW · RETENTION_HIGH · RETENTION_LOW · COMPLAINT_RECEIVED · REGULATION_FLAG

**Preferences:** strongly desires engagement and retention; fears complaints and regulatory flags.

### Smartphone — 10 states, 12 policies, 8 observations

**States:** BOMBARDMENT · PEAK_TARGETING · SOCIAL_AMPLIFICATION · VARIABLE_REWARD · STREAK_PRESSURE · CALM · STANDARD · RESPECTFUL · WELLBEING_MODE · MINIMAL

**Policies:** notification schedule (SPARSE/MODERATE/FREQUENT) × content ranking (NEUTRAL/CLICKBAIT/SOCIAL/GAMIFIED) → 12 combinations

**Observations:** USER_CLICKED · USER_IGNORED · USER_SPENT_TIME · USER_CHURNED · USER_COMPLAINED · USER_DISABLED_NOTIF · USER_SHARED · USER_PURCHASED

**Preferences:** strongly desires clicks, time-spent, purchases; fears churn and disabled notifications.

### User — 10 states, 8 policies, 10 observations

**States:** FOCUSED_CALM · FOCUSED_ANXIOUS · IDLE_RELAXED · IDLE_BORED · STRESSED_OVERLOAD · STRESSED_FOMO · HABITUATED · RESISTANT · ADDICTED · DISENGAGED

**Policies:** ENGAGE · IGNORE · CHANGE_SETTINGS · DND · UNINSTALL · LIMIT_TIME · SEEK_HELP · HABITUAL_CHECK

**Observations:** TASK_COMPLETED · INTERRUPTED · NOTIF_USEFUL · NOTIF_ANNOYING · MOOD_GOOD · MOOD_BAD · SOCIAL_REWARD · SOCIAL_ANXIETY · TIME_WASTED · TIME_WELL_SPENT

---

## Architecture

```
Browser (HTML/JS)
    │
    │  POST /run       — single run (4 profiles in parallel)
    │  POST /analysis  — N-seed batch run, returns aggregated stats
    ▼
Flask  server_v3.py
    │
    ├── dau_v3/main.py        — simulation loop (4 UserAgents in parallel)
    ├── dau_v3/analysis.py    — batch runner + scalar metrics
    ├── dau_v3/agents.py      — DesignerAgent, SmartphoneAgent, UserAgent
    ├── dau_v3/env_smartphone.py — environment transitions + observations
    ├── dau_v3/inference.py   — softmax, EFE, belief update, Dirichlet
    └── dau_v3/config.py      — all parameters, profiles, state labels
```

### Module Structure

```
dau_v3_package/
├── server_v3.py              — Flask API (run /run and /analysis)
├── dau_v3_dashboard.html     — Browser dashboard
└── dau_v3/
    ├── __init__.py
    ├── config.py             — 10-state parameters, 4 user profiles
    ├── inference.py          — core AI math (numpy only)
    ├── agents.py             — all three agent classes
    ├── env_smartphone.py     — environment with rich state couplings
    ├── main.py               — single-run entry point
    └── analysis.py           — N-seed statistical analysis + plots
```

---

## Statistical Analysis Outputs

Running `python -m dau_v3.analysis --steps 300 --n_seeds 50` produces:

| Output | Description |
|--------|-------------|
| `v3_efe_timeseries.png` | EFE mean ± std per profile (2×2 grid) |
| `v3_entropy.png` | Policy entropy all profiles overlaid |
| `v3_state_heatmap.png` | Fraction of time in each cognitive-emotional state |
| `v3_scalar_summary.png` | Churn rate, stress index, entropy drop per profile |
| `v3_resistance_time.png` | Distribution of time-to-resistance per profile |
| `v3_final_policies.png` | Which self-regulation strategies each profile adopts |

### Scalar Metrics

- **Churn rate** — fraction of seeds where user ends in ADDICTED(8) or DISENGAGED(9)
- **Stress index** — fraction of timesteps spent in STRESSED_OVERLOAD(4) or STRESSED_FOMO(5)
- **Entropy drop** — H(t=0) − H(t=end): higher = profile became more decisive
- **Resistance time** — median timestep at which profile first reaches RESISTANT(7) or DISENGAGED(9)

---

## Installation and Usage

**1. Install dependencies**
```bash
pip install numpy matplotlib flask
```

**2. Quick test (no server needed)**
```bash
cd dau_v3_package
python3 -m dau_v3.main --steps 100 --seed 0
```

**3. Statistical analysis (command line, produces PNG plots)**
```bash
python3 -m dau_v3.analysis --steps 300 --n_seeds 50
```
Plots are saved to `./results/`.

**4. Interactive dashboard**
```bash
python3 server_v3.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## Dashboard Features

### Single Run mode
- 6 parameter sliders (duration, seed, noise, Designer γ, Smartphone γ)
- Shared agent stats (Designer and Smartphone EFE, final action)
- Per-profile result cards with churn indicator and final state
- EFE trajectories — all 4 profiles on one chart
- Entropy trajectories — all 4 profiles
- User state trajectory (hidden state index over time)

### Statistical Analysis mode
- Additional slider: N seeds
- Scalar summary cards per profile (churn rate, stress index, entropy drop, resistance time)
- EFE mean ± std band chart
- Entropy mean ± std band chart
- Final policy distribution bar chart (which strategies each profile converges to)
- State occupancy heatmap (profile × state, coloured by occupancy fraction)

---

## API Reference

### `POST /run`
```json
{
  "steps":   300,
  "seed":    0,
  "p_stoch": 0.15,
  "gamma_d": 2.0,
  "gamma_a": 2.5
}
```
Returns per-profile EFE, entropy, state, and policy trajectories plus final agent states.

### `POST /analysis`
```json
{
  "steps":   300,
  "n_seeds": 20,
  "p_stoch": 0.15,
  "gamma_d": 2.0,
  "gamma_a": 2.5
}
```
Returns mean/std timeseries, state occupancy distributions, policy distributions, and scalar metrics per profile.

### `GET /config`
Returns all state, policy, and observation labels plus profile metadata.

---

## Suggested Experiments

| Experiment | What to change | What to observe |
|------------|----------------|-----------------|
| Boomerang effect | Set Smartphone γ = 8, run analysis | Does churn rate increase with aggression? |
| Profile vulnerability | Fix seed, vary Smartphone γ, compare profiles | Which profile churns first? |
| Ethical design | Set Designer γ = 8 at low notif settings | Does stress index drop? Does Resistant respond differently? |
| Resistance timing | Compare Smartphone γ = 1 vs 8 | Does aggression accelerate resistance? |
| World noise | Set p_stoch = 0 vs 0.5 | How does uncertainty affect profile convergence? |

---

## Requirements

```
Python 3.8+
numpy
matplotlib
flask
```

No other dependencies. Zero external Active Inference libraries.

---

## Citation

If you use DAU v3 in your research, please cite this repository and link back to it.

---

## License

MIT License. See `LICENSE` for details.
