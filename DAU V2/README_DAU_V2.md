# DAU v2 — Active Inference Smartphone Triad

> A clean-room, numpy-only implementation of three non-identical Active Inference agents (Designer, Smartphone, User) modeling the smartphone interaction triad. Features per-agent precision weighting, an "empty user model" that becomes definite over time via γ-annealing and Dirichlet learning, and full EFE tracking.

---

## Table of Contents

1. [Scenario & Design Choices](#1-scenario--design-choices)
2. [Module Structure](#2-module-structure)
3. [Equations](#3-equations)
4. [Running the Simulation](#4-running-the-simulation)
5. [Output Files](#5-output-files)
6. [Self-check / Tests](#6-self-check--tests)
7. [Configuration Reference](#7-configuration-reference)

---

## 1. Scenario & Design Choices

### 1.1 The Smartphone Triad

Three agents interact across different timescales around a smartphone interface:

| Agent | Timescale | Role |
|-------|-----------|------|
| **Designer** | Slow | Sets global smartphone defaults (notification intensity, friction-to-disable). Optimizes for engagement metrics (business objective). |
| **Smartphone** | Medium | Controls the interface layer (notification schedule, recommendation ranking bias). Optimizes for immediate KPIs (engagement, anti-churn). |
| **User** | Fast | Self-regulates in response to the interface (engage, ignore, change settings, DND, uninstall). Optimizes for well-being, task completion, low interruption. |

### 1.2 Hidden States and Action Spaces

**Designer**

| | |
|--|--|
| Hidden state | `design_stance` ∈ {ENGAGEMENT, WELLBEING, BALANCED} |
| Actions (6 policies) | `(notif_intensity, friction)` ∈ {LOW, MED, HIGH} × {LOW, HIGH} |
| Observations | HIGH_ENGAGEMENT, LOW_ENGAGEMENT, HIGH_WELLBEING, LOW_WELLBEING |
| Preferences C | +2.0 HIGH_ENG, −2.0 LOW_ENG, +0.5 HIGH_WB, −0.5 LOW_WB |

**Smartphone**

| | |
|--|--|
| Hidden state | `interface_mode` ∈ {CALM, STANDARD, AGGRESSIVE} |
| Actions (6 policies) | `(notif_schedule, ranking_bias)` ∈ {SPARSE, MODERATE, FREQUENT} × {NEUTRAL, CLICKBAIT} |
| Observations | USER_ENGAGED, USER_IGNORED, USER_SATISFIED, USER_CHURNED |
| Preferences C | +3.0 USER_ENGAGED, −1.0 USER_IGNORED, +0.5 USER_SATISFIED, −3.0 USER_CHURNED |

**User**

| | |
|--|--|
| Hidden state | `need_state` ∈ {FOCUSED, IDLE, STRESSED} |
| Actions (5 policies) | ENGAGE, IGNORE, CHANGE_SETTINGS, DND, UNINSTALL |
| Observations | TASK_DONE, INTERRUPTED, NOTIF_USEFUL, NOTIF_ANNOYING, MOOD_OK, MOOD_BAD |
| Preferences C | +3.0 TASK_DONE, −2.0 INTERRUPTED, +1.0 NOTIF_USEFUL, −1.5 NOTIF_ANNOYING, +1.5 MOOD_OK, −2.5 MOOD_BAD |

### 1.3 Stochastic Generative Models

All A and B matrices include stochasticity:

- **Observation noise:** all A matrices blended 85%/15% toward uniform — `A_noisy = 0.85·A + 0.15/n_obs`
- **Transition noise:** all B columns use `p_stoch = 0.20` — target state gets 0.80 mass, remainder spread uniformly

### 1.4 The "Empty User Model"

The User starts with an uncommitted, near-uniform policy distribution and becomes progressively more decisive through two mechanisms:

1. **γ-annealing:** policy precision linearly increases from `γ=0.5` (near-uniform) to `γ=5.0` (peaked) over the full episode
2. **Dirichlet learning:** a concentration parameter vector `α` (uniform at t=0) is updated online with `Δα = lr · q(π)`, shifting the policy prior toward frequently-selected policies

A dedicated plot (`dau_v2_user_decisiveness.png`) tracks policy entropy, `max(q_π)`, and the γ schedule over time.

### 1.5 Cross-Agent Couplings

The environment mediates all agent interactions:

```
Designer action (notif_intensity)
    │  weight: DSG_TO_ART_STRENGTH = 0.60
    ▼
Smartphone mode  ←──── Smartphone action (notif_schedule, ranking_bias)
    │  weight: ART_TO_USR_STRENGTH = 0.50
    ▼
User state  ←──────── User action (engage / DND / uninstall / …)
    │
    ▼
Observations fed back to Designer and Smartphone (aggregate KPI / well-being signal)
```

---

## 2. Module Structure

```
dau_v2/
├── __init__.py          — package marker
├── config.py            — ALL scenario parameters (states, actions, C, D, β, γ)
├── inference.py         — stateless core math: softmax, belief update, EFE, policy posterior
├── agents.py            — DesignerAgent, SmartphoneAgent, UserAgent classes
├── env_smartphone.py    — environment: world state transitions + observation generation
├── main.py              — simulation loop + CLI entry point + summary printer
├── plotting.py          — matplotlib visualisation (EFE timeseries + user decisiveness)
└── self_check.py        — 5 sanity tests (no external test framework required)

results/
├── dau_v2_efe_timeseries.png
└── dau_v2_user_decisiveness.png
```

### Key design principle

`inference.py` is completely stateless — every function takes plain numpy arrays and returns numpy arrays. Agents import and call these functions; they do not inherit from any base class. This makes the math easy to test and reason about independently.

---

## 3. Equations

### 3.1 β-weighted belief update — `infer_states()`

$$\log q(s) \;\propto\; \log D[s] \;+\; \beta \cdot \log A^{\text{noisy}}[o, s]$$

$$q(s) = \text{softmax}\!\left(\log D[s] + \beta \cdot \log A^{\text{noisy}}[o, s]\right)$$

where $A^{\text{noisy}}[o,s] = (1-\epsilon)\,A[o,s] + \epsilon/n_{\text{obs}}$ with $\epsilon = 0.15$.

**Effect of β:** `β=1` → standard Bayes; `β→0` → posterior collapses to prior; `β>1` → hypersensitive to evidence.

### 3.2 State propagation

$$q(s' \mid \pi) = B_\pi \cdot q(s)$$

where $B_\pi$ is the stochastic transition matrix for policy $\pi$, with $p_{\text{stoch}} = 0.20$.

### 3.3 Mean-field observation prediction

$$q(o \mid \pi) = A^{\text{noisy}} \cdot q(s' \mid \pi)$$

### 3.4 Expected Free Energy — `compute_efe()`

$$G(\pi) = \underbrace{-H\!\left[q(o \mid \pi)\right]}_{\text{epistemic value}} + \underbrace{q(o \mid \pi) \cdot C}_{\text{pragmatic value}}$$

where $H[p] = -\sum_o p(o)\log p(o)$ is the Shannon entropy of predicted observations and $C$ is the log-prior-preference vector.

- **Epistemic term:** negative entropy — policies that yield informative (low-entropy) predictions are preferred
- **Pragmatic term:** expected preference — policies predicting preferred outcomes score higher

Both `EFE_selected(t)` (EFE of the chosen policy) and `EFE_max(t)` (EFE of the best available policy) are logged at every timestep.

### 3.5 γ-weighted policy posterior — `policy_posterior()`

$$q(\pi) = \text{softmax}\!\left(\gamma \cdot G(\pi) + \log E(\pi)\right)$$

where $E(\pi) = \text{normalise}(\alpha)$ is the Dirichlet-derived policy prior (only used for the User; Designer and Smartphone use a flat prior).

**Effect of γ:** `γ→0` → uniform policy selection; `γ→∞` → argmax selection.

### 3.6 Dirichlet policy prior update — `update_dirichlet()`

$$\alpha \;\leftarrow\; \alpha + \text{lr} \cdot q(\pi)$$

Soft online accumulation: the concentration parameters grow proportionally to how often each policy is selected, making the prior progressively more peaked.

### 3.7 γ-annealing schedule (User only)

$$\gamma_{\text{user}}(t) = \gamma_{\text{init}} + \frac{t}{T-1}\left(\gamma_{\text{final}} - \gamma_{\text{init}}\right)$$

with $\gamma_{\text{init}} = 0.5$ and $\gamma_{\text{final}} = 5.0$.

### 3.8 Summary table

| Equation | Function | Precision |
|----------|----------|-----------|
| $A^{\text{noisy}} = (1-\epsilon)A + \epsilon/n_o$ | `_add_noise()` | — |
| $q(s) = \text{softmax}(\log D + \beta \log A^{\text{noisy}}[o])$ | `infer_states()` | **β** |
| $q(s'\|\pi) = B_\pi \cdot q(s)$ | `compute_efe()` | — |
| $q(o\|\pi) = A^{\text{noisy}} \cdot q(s'\|\pi)$ | `compute_efe()` | — |
| $G(\pi) = -H[q(o\|\pi)] + q(o\|\pi) \cdot C$ | `compute_efe()` | — |
| $q(\pi) = \text{softmax}(\gamma \cdot G + \log E)$ | `policy_posterior()` | **γ** |
| $\alpha \leftarrow \alpha + \text{lr} \cdot q(\pi)$ | `update_dirichlet()` | — |
| $\gamma(t) = \gamma_0 + (t/T)(\gamma_T - \gamma_0)$ | `UserAgent.step()` | **γ annealing** |

---

## 4. Running the Simulation

### Requirements

```
Python 3.8+
numpy
matplotlib
```

No other dependencies. No pymdp. Everything runs locally.

### Commands

```bash
# Standard run — 200 steps, seed 0
python -m dau_v2.main --steps 200 --seed 0

# Longer run for smoother plots
python -m dau_v2.main --steps 400 --seed 42

# Run sanity tests
python -m dau_v2.self_check
```

---

## 5. Output Files

```
results/dau_v2_efe_timeseries.png
```

Two-panel figure:
- **Top:** `EFE_selected(t)` — EFE of the policy actually chosen at each step, for all three agents
- **Bottom:** `EFE_max(t)` — EFE of the best available policy at each step

```
results/dau_v2_user_decisiveness.png
```

Three-panel figure:
- **Panel 1:** Policy entropy `H(q_π)` — decreases as User becomes more committed
- **Panel 2:** `max(q_π)` — increases as User converges on a preferred policy
- **Panel 3:** `γ_user(t)` — linear annealing schedule from 0.5 to 5.0

### Example console output

```
[run] steps=200  seed=0
[plot] saved → results/dau_v2_efe_timeseries.png
[plot] saved → results/dau_v2_user_decisiveness.png

============================================================
DAU v2 — Simulation Summary
============================================================

  Designer:
    EFE_selected — mean=-0.501  median=-0.299  std=0.350  [-1.200, -0.299]
    EFE_max      — mean=-0.299  median=-0.299  std=0.000  [-0.299, -0.299]

  Smartphone:
    EFE_selected — mean=-0.289  median=-0.194  std=0.214  [-1.124, -0.194]
    EFE_max      — mean=-0.194  median=-0.194  std=0.000  [-0.194, -0.194]

  User:
    EFE_selected — mean=-0.685  median=-0.675  std=0.094  [-1.617, -0.675]
    EFE_max      — mean=-0.675  median=-0.675  std=0.000  [-0.675, -0.675]

  User policy entropy:
    t=0     H = 1.5949  (high → uniform)
    t=199   H = 1.3919  (lower → more committed)
    Final max(q_π) = 0.2498
    Final γ_user   = 5.0000
============================================================
```

---

## 6. Self-check / Tests

```bash
python -m dau_v2.self_check
```

Five tests run automatically:

| # | Test | What it checks |
|---|------|----------------|
| 1 | Initial user q_π ≈ uniform | `H(q_π)` at t=0 ≥ 95% of maximum possible entropy |
| 2 | User entropy decreases | Mean entropy in last quarter < mean entropy in first quarter |
| 3–5 | EFE arrays are valid | Each agent's EFE array has exactly `steps` entries and contains only finite values |

---

## 7. Configuration Reference

All parameters are in `dau_v2/config.py`. No other file contains magic numbers.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DSG_BETA` | `1.0` | Designer likelihood precision |
| `DSG_GAMMA` | `2.0` | Designer policy precision |
| `ART_BETA` | `1.0` | Smartphone likelihood precision |
| `ART_GAMMA` | `2.5` | Smartphone policy precision (more decisive than Designer) |
| `USR_BETA` | `1.0` | User likelihood precision (fixed) |
| `USR_GAMMA_INIT` | `0.5` | User initial policy precision (low → near-uniform) |
| `USR_GAMMA_FINAL` | `5.0` | User final policy precision after annealing |
| `USR_DIRICHLET_ALPHA_INIT` | `1.0` | Initial Dirichlet concentration (uniform) |
| `USR_DIRICHLET_LR` | `0.05` | Dirichlet learning rate |
| `DSG_TO_ART_STRENGTH` | `0.60` | How strongly Designer's action biases smartphone mode |
| `ART_TO_USR_STRENGTH` | `0.50` | How strongly smartphone mode stresses the user |
| `ENV_P_STOCH` | `0.20` | Transition noise for all B matrices |
