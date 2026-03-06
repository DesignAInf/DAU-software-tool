# DAU v2 – Smartphone Triad Active Inference Simulation

A new, from-scratch Active Inference implementation for the Designer–Artifact–User triad
applied to the **smartphone** context.  Three **non-identical** agents compete and cooperate
across different timescales, with misaligned preferences that create realistic tension.

> **Does not touch the original DAU implementation** – all new code lives in `dau_v2/`.

---

## Quick Start

```bash
# From the repository root
python -m dau_v2.main --steps 200 --seed 0
```

Sanity tests:

```bash
python -m dau_v2.self_check
```

---

## Output Files

| File | Description |
|------|-------------|
| `results/dau_v2_efe_timeseries.png` | EFE trajectories for all three agents |
| `results/dau_v2_user_decisiveness.png` | User policy entropy + max q(π) over time |

---

## Agent Design

### Why three non-identical agents?

Each agent operates at a **different timescale**, has **different state/action spaces**,
and holds **conflicting preferences** – reflecting real smartphone ecosystems where
designers optimise for business KPIs, the phone executes those policies, and the user
tries to protect their own attention and well-being.

| Agent | Hidden states | Actions | Preferences C | γ (policy prec.) |
|---|---|---|---|---|
| **Designer** | engage / balanced / minimal | 4 | Prefers `high_engagement` | 2.0 |
| **Smartphone** | aggressive / moderate / calm | 4 | Prefers `user_engaged` (KPI) | 3.0 |
| **User** | distracted / focused / DND | **5** | Prefers `notif_low`, dislikes `notif_high` | 0.5 → 4.0 ↑ |

---

## Active Inference Maths

Each agent implements the standard generative model:

```
A_i : P(o | s)          Likelihood matrix   [n_obs × n_states]
B_i : P(s'| s, a)       Transition tensor   [n_states × n_states × n_actions]
C_i : log P*(o)         Log-preferences     [n_obs]
D_i : P(s)              Prior over states   [n_states]
```

### Belief update (perception)

```
log q(s) ∝ log D(s) + β · log A(o|s)
```

- β (likelihood precision) scales how much the agent trusts its senses vs. its prior.

### Expected Free Energy (H = 1 look-ahead)

```
G(a) = risk(a) + ambiguity(a)

risk(a)      = -E_{q(o'|a)}[ C(o') ]           # pragmatic value
ambiguity(a) = Σ_s' q(s'|a) · H[P(o|s')]       # epistemic value
```

- Lower G → action is preferred.

### Action selection (softmax policy)

```
q(π) = softmax( -γ · G )
```

- γ (policy precision) controls exploration vs. exploitation.
- γ → 0: uniform (random); γ → ∞: deterministic argmin G.

---

## "Empty User Model" → Becomes Definite

The user starts with:
- **Uniform prior D**: equal belief across all states (no prior commitment).
- **Low γ = 0.5**: policy distribution q(π) is nearly flat → user behaves almost randomly.

Over T steps, γ is linearly annealed from **0.5 → 4.0**:

```
γ_user(t) = 0.5 + (4.0 - 0.5) × t / (T - 1)
```

This means the user progressively **becomes more decisive** – their preferred actions
emerge as they accumulate evidence and precision increases.  The `user_decisiveness` plot
shows this as a decreasing entropy curve.

---

## Coupling Between Agents

```
Designer  ──(design influence)──►  Smartphone B matrix (perturbed each step)
                                        │
                          (action maps to user observation)
                                        │
                                        ▼
                                      User
                                        │
                          (action maps to smartphone observation)
                                        ▼
                                   Smartphone
                                        │
                          (state maps to designer observation)
                                        ▼
                                    Designer
```

### Coupling rules

| Source → Target | Mechanism |
|---|---|
| Designer → Smartphone | Designer's action biases smartphone's B matrix by `designer_influence=0.3` toward a promoted smartphone action |
| Smartphone → User | Phone action deterministically sets user observation: `send_notif → notif_high`, `suppress → notif_low`, `add_friction → friction_barrier` |
| User → Smartphone | User action maps to phone observation: `engage → user_engaged`, `dnd/limit → user_resistant` |
| Smartphone state → Designer | argmax(q_s_phone) maps to designer observation: `aggressive→high_engagement`, `calm→low_engagement` |

---

## EFE Logging

Two scalars are logged per agent per timestep:

| Scalar | Meaning |
|---|---|
| `EFE_selected(t)` | EFE of the *chosen* action (what the agent actually paid) |
| `EFE_min(t)` | Minimum EFE across all candidate actions (best possible) |

Both appear in `dau_v2_efe_timeseries.png` as solid (selected) and dashed (min) lines.

---

## Module Structure

```
dau_v2/
├── __init__.py          Package header
├── config.py            All parameters (AgentConfig, SimConfig)
├── inference.py         Core maths: update_belief, compute_efe, select_action
├── agents.py            ActiveInferenceAgent class + 3 build_*_agent() factories
├── env_smartphone.py    SmartphoneEnvironment – step logic + coupling
├── plotting.py          Matplotlib plots (non-interactive backend)
├── main.py              CLI entry point
└── self_check.py        Sanity tests

results/
├── dau_v2_efe_timeseries.png
└── dau_v2_user_decisiveness.png
```

---

## Dependencies

```
numpy
matplotlib
```

No other external packages required.

---

## CLI Reference

```
python -m dau_v2.main [--steps N] [--seed S] [--no-decisiveness-plot]

Options:
  --steps INT              Simulation length (default: 200)
  --seed  INT              Random seed (default: 0)
  --no-decisiveness-plot   Skip saving the user decisiveness plot
```

---

## Design Choices & Limitations

- **H = 1 look-ahead** only (single-step EFE).  Extending to H = 2 requires rolling
  out the B matrix twice and marginalising over intermediate states – straightforward
  but adds code complexity.
- **No online learning** of A or B (Dirichlet parameter learning omitted for clarity).
  The user's prior D is uniform but fixed; precision annealing is the sole "learning" mechanism.
- **Designer influence** is a simple linear blend of B columns, not a full likelihood
  update over the phone's prior. This is intentionally transparent and easy to tune.
- **No stochastic observations** (observations are deterministically set by couplings).
  Adding a noise model (e.g., confusion matrix over coupling outputs) is a natural extension.
