# DAU v2 — Active Inference Dashboard

An interactive research dashboard for simulating and visualizing the behavior of three Active Inference agents — **Designer**, **Smartphone**, and **User** — in a smartphone interaction scenario. Built with a Python/Flask backend running the real simulation engine and a browser-based frontend for real-time visualization. The question the model asks is: if three agents with these goals interact according to the principles of Active Inference, what patterns of behavior emerge?

---

## What Is This?

DAU v2 models the smartphone triad as a multi-agent Active Inference system. Each agent continuously updates its beliefs about the world and selects actions to minimize surprise — a principle drawn from computational neuroscience and cognitive science. The dashboard makes this simulation accessible to researchers, students, and stakeholders without requiring any programming knowledge.

The three agents operate at different timescales and pursue different objectives:

| Agent | Timescale | Objective |
|-------|-----------|-----------|
| **Designer** | Slow | Sets global smartphone defaults (notification intensity, friction-to-disable). Optimizes for engagement metrics. |
| **Smartphone** | Medium | Controls the interface layer (notification schedule, ranking bias). Optimizes for immediate KPIs and anti-churn. |
| **User** | Fast | Responds with self-regulation strategies (engage, ignore, change settings, DND, uninstall). Optimizes for task completion, low interruption, and positive mood. |

A key feature of the model is the **empty user**: the User starts with no clear preferences (near-uniform policy distribution) and gradually becomes more decisive over time through precision annealing and Dirichlet learning — simulating how a person forms habits and preferences through repeated exposure to an interface.

---

## Features

### Interactive Parameter Control
Six sliders allow real-time configuration of the simulation before each run:

- **Duration** — number of timesteps (50–500)
- **Random Seed** — controls which random events occur; same seed = reproducible run
- **World Noise** — how unpredictable the environment is (0 = fully deterministic, 0.5 = highly stochastic)
- **Designer Decisiveness (γ)** — how firmly the Designer commits to its preferred strategy
- **Smartphone Decisiveness (γ)** — how aggressively the Smartphone optimizes for engagement
- **User Final Commitment (γ)** — the endpoint of the User's precision annealing schedule

### Live Results Visualization
After each run, the dashboard updates with:

- **Decision Quality over Time** — Expected Free Energy (EFE) trajectories for all three agents, showing how well each agent's chosen actions aligned with its goals at every timestep
- **User Uncertainty over Time** — policy entropy curve showing the User's transition from an undecided to a committed state
- **User Commitment Growth** — the γ annealing schedule, visualizing the User's increasing decisiveness
- **Strategy Probability Bars** — final-timestep policy posteriors q(π) for each agent, showing which actions each agent considers most likely
- **Belief State Bars** — final-timestep posterior beliefs q(s) for each agent, showing how each agent perceives its current hidden state
- **Summary Statistics** — mean, standard deviation, and final values for EFE and entropy metrics per agent

### Connected to Real Python Code
Unlike standalone JavaScript simulations, this dashboard calls the actual `dau_v2` Python simulation engine via a local Flask API. Every run button press executes the real Active Inference math — belief updates, EFE computation, policy posteriors, Dirichlet learning — in Python and streams the results to the browser as JSON.

---

## Architecture

```
Browser (HTML/JS)
    │
    │  POST /run  {steps, seed, γD, γA, γU, p_stoch}
    ▼
Flask server (server.py)
    │
    │  calls
    ▼
dau_v2/main.py  →  agents.py  →  inference.py
                →  env_smartphone.py
                →  config.py
    │
    │  returns JSON {efe arrays, summary stats, final agent states}
    ▼
Browser renders charts, bars, and statistics
```

### Module Structure

```
dau_project/
├── server.py                        — Flask API server
├── dau_v2_dashboard_connected.html  — Frontend dashboard
└── dau_v2/
    ├── __init__.py
    ├── config.py        — All scenario parameters (states, actions, C, D, β, γ)
    ├── inference.py     — Core math: softmax, belief update, EFE, policy posterior
    ├── agents.py        — DesignerAgent, SmartphoneAgent, UserAgent classes
    ├── env_smartphone.py — Environment: world transitions + observation generation
    ├── main.py          — Simulation loop + summary printer
    ├── plotting.py      — Static matplotlib plots (optional)
    └── self_check.py    — Sanity tests
```

---

## Active Inference — Key Concepts

For readers unfamiliar with Active Inference, here is a brief glossary of the terms used in the dashboard:

**Expected Free Energy (EFE)** — The core quantity each agent minimizes. It combines two terms: *epistemic value* (how informative will my action be?) and *pragmatic value* (how likely is my action to produce outcomes I prefer?). See the full explanation below.

**Policy precision (γ)** — Controls how decisively an agent commits to its preferred policy. Low γ → near-random exploration. High γ → always picks the best-scoring action.

**Policy entropy H(q(π))** — Measures how spread out an agent's preferences are across its available strategies. High entropy = undecided. Low entropy = committed.

**Belief state q(s)** — The agent's probability distribution over its hidden states. Reflects what the agent currently believes about the world.

**Dirichlet learning** — The mechanism by which the User's policy prior becomes more peaked over time. Concentration parameters α are updated online proportionally to how often each policy is selected.

---

## How EFE Is Computed

At every timestep, each agent must choose among its available policies. To evaluate each policy, it computes the **Expected Free Energy** — an estimate of how "good" that policy will be at the next moment.

The formula is:

```
G(π) = -H[q(o|π)]  +  q(o|π) · C
```

It has two components.

### Component 1 — Epistemic value: `-H[q(o|π)]`

The agent predicts what observations it would expect if it chose that policy. It then computes the entropy of that predicted distribution. If entropy is low (the agent knows well what to expect), this term is close to zero — the policy is informative and reduces uncertainty.

### Component 2 — Pragmatic value: `q(o|π) · C`

`C` is the agent's preference vector — how much it desires each possible observation. This term measures: how well do the observations predicted by this policy match what the agent wants? If predicted observations align with high-preference outcomes, this term is high.

The result `G(π)` is a number. **Higher (less negative) = better policy.**

### Important: EFE values are always negative

In this simulation EFE values are typically between **-0.1** and **-1.5**. This is because:

- The epistemic term `-H[q(o|π)]` is always ≤ 0 (entropy is always ≥ 0)
- The pragmatic term `q(o|π) · C` can be positive, but the preference vectors are calibrated such that the sum remains negative

This means that in the dashboard charts **higher (less negative) is better**:

```
-0.28  is a better decision than  -1.20
```

When you see a line **rising toward zero**, the agent is choosing better policies. When it **drops toward -1.5**, the agent is choosing suboptimal policies or the world is surprising it.

### Concrete example — Smartphone agent

The Smartphone has 6 policies. For each one (e.g. `FREQUENT+CLICKBAIT`) it:

1. Predicts the next state: `q(s') = B_π · q(s)`
2. Predicts observations: `q(o) = A · q(s')`
3. Computes `-H[q(o)]` → epistemic term
4. Computes `q(o) · C` using `C = [+3.0, -1.0, +0.5, -3.0]` for `[USER_ENGAGED, USER_IGNORED, USER_SATISFIED, USER_CHURNED]`

If `FREQUENT+CLICKBAIT` predicts `USER_ENGAGED` with high probability, the pragmatic term is large and positive → `G` is high → the Smartphone favors this policy.

After computing G for all 6 policies it selects one via:

```
q(π) = softmax(γ · G)
```

### What `EFE_selected` and `EFE_max` mean in the dashboard

The dashboard records two values per agent per timestep:

- **`EFE_selected(t)`** — the G value of the policy that was actually chosen (sampled from q(π))
- **`EFE_max(t)`** — the G value of the best available policy at that step

The gap between the two indicates how often the agent picks the optimal policy. If they are equal, the agent is fully deterministic (very high γ). If they diverge, the agent is exploring.

### Reading the EFE chart

| What you see | What it means |
|---|---|
| Flat line | Agent locked onto one policy — always the same EFE |
| Oscillating line | Agent switching policies in response to observations |
| `EFE_selected` ≈ `EFE_max` | High γ — agent is decisive |
| Large gap between the two | Low γ — agent is exploring |
| User line consistently lower | See next section |

---

## Why the User Has the Lowest EFE

You will notice that the User's EFE line is consistently lower (more negative) than the Designer's and Smartphone's. This is not a bug — it is a structural result of the model, with three distinct causes.

### Reason 1 — The User's preferences are harder to satisfy

The User's preference vector `C` is ambitious:

```
+3.0   TASK_DONE        ← strongly desired
-2.0   INTERRUPTED      ← aversive
+1.0   NOTIF_USEFUL     ← mildly desired
-1.5   NOTIF_ANNOYING   ← aversive
+1.5   MOOD_OK          ← desired
-2.5   MOOD_BAD         ← strongly aversive
```

But the world the User inhabits is largely hostile. The Smartphone operates in STANDARD or AGGRESSIVE mode most of the time, generating frequent INTERRUPTED and NOTIF_ANNOYING observations. The pragmatic term `q(o) · C` therefore tends to be very negative, because the predicted observations rarely match what the User wants.

### Reason 2 — The User's model starts empty

The User begins with γ = 0.5 — near-random policy selection. Its policy distribution is nearly uniform, so it frequently selects suboptimal actions (e.g. ENGAGE when DND would be better). This means `EFE_selected` is often far from `EFE_max` — the User cannot yet exploit its best available policies. As γ anneals upward this gap closes, but the overall EFE level remains lower than the other agents.

### Reason 3 — The User observes more

The User has **6 possible observations**; Designer and Smartphone have only **4**. More observations means the predicted distribution `q(o|π)` is more spread out, which means higher entropy H, which means the epistemic term `-H[q(o|π)]` is more negative. This alone structurally lowers the User's EFE relative to the other agents, regardless of how well it chooses its policies.

### Summary

```
Agent       Obs space   Preference difficulty   γ regime        EFE level
─────────────────────────────────────────────────────────────────────────
Designer    4 obs       easy (engagement)       fixed 2.0       high (~-0.3)
Smartphone  4 obs       easy (engagement)       fixed 2.5       high (~-0.2)
User        6 obs       hard (wellbeing)        annealed 0.5→5  low  (~-0.7)
```

This asymmetry is the central finding of the model: **the User is structurally the most disadvantaged agent** because its goal (wellbeing, task completion, low interruption) is the hardest to achieve in an environment that Designer and Smartphone have built to optimize for engagement.

---

## Requirements

```
Python 3.8+
numpy
matplotlib
flask
```

No other dependencies. The simulation engine (`dau_v2/`) uses only `numpy` — no pymdp or other Active Inference libraries.

---

## Installation and Usage

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

**2. Install dependencies**
```bash
pip install numpy matplotlib flask
```

**3. Start the server**
```bash
cd dau_project
python3 server.py
```

**4. Open the dashboard**

Navigate to `http://127.0.0.1:5000` in your browser.

**5. Run the simulation**

Adjust the sliders and press **Run Simulation**. Results appear within a few seconds.

---

## API Endpoints

The Flask server exposes three endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the dashboard HTML |
| `/run` | POST | Runs the simulation with provided parameters, returns JSON results |
| `/config` | GET | Returns agent state/action/observation labels |

### `/run` — Request body

```json
{
  "steps":   200,
  "seed":    0,
  "gamma_d": 2.0,
  "gamma_a": 2.5,
  "gamma_u": 5.0,
  "p_stoch": 0.2
}
```

### `/run` — Response (abbreviated)

```json
{
  "steps": 200,
  "efe_dsg_sel": [...],
  "efe_art_sel": [...],
  "efe_usr_sel": [...],
  "usr_entropy":  [...],
  "summary": {
    "designer":   { "efe_sel_mean": -0.501, "last_action": 4 },
    "smartphone": { "efe_sel_mean": -0.289, "last_action": 2 },
    "user":       { "entropy_t0": 1.595, "entropy_final": 1.392 }
  },
  "final_states": {
    "designer":   { "qs": [...], "qpi": [...] },
    "smartphone": { "qs": [...], "qpi": [...] },
    "user":       { "qs": [...], "qpi": [...] }
  }
}
```

---

## Running the Sanity Tests

```bash
python3 -m dau_v2.self_check
```

Five tests run automatically:

1. User q(π) is approximately uniform at t=0 (entropy ≥ 95% of maximum)
2. User entropy decreases over time (first-quarter mean > last-quarter mean)
3–5. EFE arrays for all three agents have correct length and contain only finite values

---

## Suggested Experiments

| Experiment | What to change | What to observe |
|------------|----------------|-----------------|
| Rigid Designer | Set Designer γ = 8 | Designer locks onto one strategy; EFE line goes flat |
| Passive User | Set User final γ = 1 | User entropy stays high; User never commits |
| Chaotic world | Set World Noise = 0.5 | All EFE traces become more variable |
| Reproducibility | Keep seed fixed, change one parameter | Compare outcomes cleanly |
| Learning speed | Compare γ_user = 2 vs γ_user = 10 | Entropy curve drops faster with higher γ |

---

## Citation

If you use this simulation in your research, please cite accordingly and link back to this repository.

---

## License

MIT License. See `LICENSE` for details.
