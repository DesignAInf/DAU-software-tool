# DAU v3 — Active Inference Persuasive Design Study

An extended Active Inference simulation studying how **Designer and Smartphone strategies differentially manipulate users with different psychological profiles**. Four user archetypes run in parallel under shared environmental conditions, revealing which persuasive techniques are most effective on which profiles.

Built on the same numpy-only Active Inference core as DAU v2, extended to 10 hidden states per agent, richer observation spaces, and a full statistical analysis pipeline.

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
