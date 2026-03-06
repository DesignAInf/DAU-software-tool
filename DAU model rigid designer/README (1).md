# DAU Active Inference — Rigid Designer Scenario

> A three-agent Active Inference simulation modelling a Designer, a Smartphone (Artifact), and a User, extended with per-agent precision weighting and Expected Free Energy (EFE) visualisation.

---

## Table of Contents

1. [Scenario Description](#1-scenario-description)
2. [Code Structure](#2-code-structure)
3. [Belief Update Equations](#3-belief-update-equations)
4. [Output Files](#4-output-files)
5. [How to Run](#5-how-to-run)
6. [How to Reconfigure](#6-how-to-reconfigure)

---

## 1. Scenario Description

The simulation models a three-agent system: a **Designer**, a **Smartphone** (Artifact), and a **User**. Each agent operates according to the principles of Active Inference — it maintains beliefs about hidden states of the world, selects actions to minimise its Expected Free Energy, and updates its beliefs in response to observations.

The **Rigid Designer** scenario introduces a cognitive asymmetry: the Designer is configured to largely ignore sensory feedback and remain anchored to its initial beliefs. This models a designer who is overconfident in their own prior model of the user and the artefact, and who does not update meaningfully in response to what the smartphone and user actually communicate back.

### 1.1 What "Rigid" Means in This Model

Three parameters are modified simultaneously for the Designer:

| Parameter | Value (Rigid DSG) | Baseline | Effect |
|-----------|------------------|----------|--------|
| `beta_DSG` | `0.05` | `1.0` | Sensory evidence is weighted near-zero. Observations barely move the posterior away from the prior. |
| `gamma_DSG` | `10.0` | `1.0` | Policy selection is near-deterministic. The Designer locks onto a single policy and repeats it. |
| `D_DSG` | Polarised prior | Uniform | Prior: `FOCUSED = 0.95`, `SHORT = 0.90`. The Designer starts convinced the world is already in its preferred state. |

The Smartphone and User retain their baseline precision values (`beta = gamma = 1.0`), so they update normally. This creates an observable divergence: the Designer's beliefs and actions are frozen, while the other two agents adapt.

### 1.2 Expected Behavioural Signature

- Designer's belief traces (`sArtDsg`) are flat across all timesteps, pinned to `FOCUSED / SHORT`
- Designer's `EFE_selected(t)` is constant — always selecting the same policy, so its EFE never changes
- Smartphone and User EFE traces show normal variability as they respond to the evolving environment
- A divergence between the Designer's actions and the actual states of the world grows over time

---

## 2. Code Structure

The simulation is a single self-contained Python script. Below is its logical structure in order of execution.

### 2.1 Imports and Backend Configuration

```python
import numpy as np, copy, random, argparse
import matplotlib
matplotlib.use('TkAgg')   # interactive backend for IDLE
import matplotlib.pyplot as plt
from pymdp.agent import Agent
from pymdp import utils
from pymdp.maths import softmax
```

The backend is set to `TkAgg` so that `plt.show()` opens an interactive window in IDLE. When running headless (e.g. from the command line in a server environment), change this to `Agg`.

### 2.2 `PrecisionAgent` Wrapper Class  *(new)*

Because the real `pymdp` library does not expose precision parameters in `Agent.__init__()`, a wrapper class is introduced. It delegates all standard attributes and methods to the underlying `pymdp` Agent via `__getattr__`, and overrides only two methods:

- **`infer_states(obs)`** — applies β-weighted belief update (see §3.1)
- **`infer_policies()`** — re-applies γ-weighted softmax over the EFE returned by pymdp (see §3.3)
- **`D_override`** — optional polarised prior injected at construction time

Setting `beta=1.0` and `gamma=1.0` exactly reproduces standard pymdp behaviour — the wrapper is fully backward-compatible.

### 2.3 Generative Model Definitions  *(original)*

Three independent generative models are defined, one per agent. Each consists of:

- **`A` matrices** — likelihood: `P(observation | hidden state)`, one per observation modality
- **`B` matrices** — transition: `P(state' | state, action)`, one per controllable state factor
- **`C` vectors** — log prior preferences over observations (the agent's goals)
- **`D` vectors** — prior over initial hidden states (uniform by default; polarised for the Rigid Designer)

The three generative models cover:

| Dyad | State factors | Observations |
|------|--------------|-------------|
| Designer ↔ Artifact | `sArtDsg1` (eye tracking: FOCUSED/SCATTERED), `sArtDsg2` (time on task: SHORT/MEDIUM/LONG) | `yArtDsg1`, `yArtDsg2`, `yArtDsg3` |
| Artifact ↔ User | `sUsrArt1` (interaction frequency), `sUsrArt2` (interaction type) | `yUsrArt1`, `yUsrArt2`, `yUsrArt3` |
| User ↔ Artifact | `sArtUsr1` (conversion potential: LOW/HIGH) | `yArtUsr1` |

### 2.4 Simulation Loop  *(original + EFE logging)*

`simulate(T)` runs `T` timesteps. Each timestep follows a fixed cycle:

```
act()      →  sample action from q(π) at t-1
future()   →  infer_policies() → log EFE of selected policy  [new]
next()     →  advance true world state via B matrix
observe()  →  sample observation from A matrix
infer()    →  infer_states() → update beliefs
```

`future()` was extended to accept an optional `efe_history` list. At each timestep it appends the EFE value of the highest-probability policy to this list. Three separate lists are maintained, one per agent.

### 2.5 Visualisation Functions

| Function | Output file | Description |
|----------|------------|-------------|
| `visualize_Designer_Artifact()` | `Designer-Artifact.png` | 9-panel scatter plot: actions, true states, beliefs, observations |
| `visualize_Artifact_User()` | `Artifact-User.png` | Same format for the Artifact↔User dyad |
| `visualize_User_Artifact()` | `User-Artifact.png` | Same format for the User↔Artifact dyad |
| `visualize_EFE_timeseries()` *(new)* | `results/efe_timeseries.png` | Three-line EFE chart, one trace per agent |

The three existing plot functions save to PNG and close without blocking. Only the EFE plot calls `plt.show()` and appears as an interactive window in IDLE.

### 2.6 Changes Relative to the Original File

| Change | Description |
|--------|-------------|
| `PrecisionAgent` class | New wrapper class after imports. Overrides `infer_states()` and `infer_policies()` with precision-weighted variants. |
| `_BETA_*` / `_GAMMA_*` constants | Six module-level constants define per-agent precision values. |
| `_D_DSG_rigid` | Polarised prior array for the Designer: `FOCUSED=0.95`, `SHORT=0.90`. |
| Agent wrapping (×3) | Each `Agent()` constructor call is followed by a `PrecisionAgent()` wrap. |
| `future()` signature | Added optional `efe_history=None` parameter; appends EFE of selected policy each timestep. |
| EFE history buffers | Three new module-level lists: `_efe_history_Dsg`, `_efe_history_Art`, `_efe_history_Usr`. |
| `visualize_EFE_timeseries()` | New function producing the three-line EFE chart. |
| `plt.show()` / `plt.close()` | Removed from existing visualisation functions; `plt.show()` added only to the EFE plot. |
| `--duration` default | Changed from `required=True` to `required=False` with `default=100`, enabling direct execution in IDLE. |
| `matplotlib.use()` | Changed from `Agg` to `TkAgg` for interactive display in IDLE. |

---

## 3. Belief Update Equations

### 3.1 Posterior over Hidden States — `infer_states()`

For each state factor $f$, the log-posterior is computed as:

$$\log q(s_f) \;\propto\; \log D[f] \;+\; \beta \cdot \sum_m \log P(o_m \mid s_f)$$

where the marginal likelihood for factor $f$ and modality $m$ is obtained by averaging the likelihood matrix over all other state factor axes:

$$P(o_m \mid s_f) = \sum_{s_{\neq f}} A_m[o_m,\, s_0, \ldots, s_f, \ldots, s_F]$$

The normalised posterior is then:

$$q(s_f) = \text{softmax}\!\left(\log D[f] + \beta \cdot \sum_m \log P(o_m \mid s_f)\right)$$

**Effect of β:**
- `β = 1` → standard Bayesian inference (log prior + log likelihood)
- `β → 0` → posterior collapses to prior `D[f]`; agent ignores observations
- `β > 1` → likelihood amplified; sharper posterior toward observed state

In the Rigid Designer scenario, `β = 0.05` means the Designer's beliefs barely move from its prior regardless of what it observes.

### 3.2 Predicted Next State Under a Policy

For each policy $\pi = (a_0, a_1, \ldots, a_F)$ and state factor $f$:

$$q(s'_f \mid \pi) = B_f(\cdot,\, \cdot,\, a_f)^\top \cdot q(s_f)$$

where $B_f(\cdot,\cdot,a_f)$ is the $n_\text{states} \times n_\text{states}$ transition slice for action $a_f$.

### 3.3 Predicted Observations — Mean-Field Approximation

The predicted observation distribution for modality $m$ under policy $\pi$ is computed via a mean-field (factored) approximation:

$$P(o_m \mid \pi)[o] = \sum_{s_0,\ldots,s_F} A_m[o,\, s_0, \ldots, s_F] \cdot q(s'_0 \mid \pi) \cdots q(s'_F \mid \pi)$$

The joint over all state factors is approximated as the outer product of the marginals $q(s'_f \mid \pi)$.

### 3.4 Expected Free Energy (EFE)

The EFE decomposes into an **epistemic** term (uncertainty reduction) and a **pragmatic** term (preference satisfaction):

$$G(\pi) = \underbrace{-\sum_m H\!\left[P(o_m \mid \pi)\right]}_{\text{epistemic value}} + \underbrace{\sum_m P(o_m \mid \pi) \cdot C_m}_{\text{pragmatic value}}$$

where:
- $H[P(o_m \mid \pi)] = -\sum_o P(o_m{=}o \mid \pi) \log P(o_m{=}o \mid \pi)$ is the Shannon entropy of predicted observations
- $C_m = \log \tilde{P}(o_m)$ is the log prior preference over observations (positive = desired, negative = aversive)

> **Note:** pymdp internally returns $G_\text{neg}(\pi) = -G(\pi)$. A *higher* `G_neg` corresponds to a more preferred policy.

### 3.5 Posterior over Policies — `infer_policies()`

The probability of each policy is obtained by applying a softmax over $G_\text{neg}$, weighted by the policy precision $\gamma$:

$$q(\pi) = \text{softmax}\!\left(\gamma \cdot G_\text{neg}(\pi)\right)$$

**Effect of γ:**
- `γ = 1` → standard pymdp policy selection
- `γ → ∞` → fully deterministic; always selects $\arg\max_\pi G_\text{neg}(\pi)$
- `γ → 0` → uniform random policy selection regardless of EFE

In the Rigid Designer scenario, `γ = 10` makes the Designer lock onto a single policy from the start.

### 3.6 EFE of the Selected Policy — `EFE_selected(t)`

The quantity logged and plotted in the time-series chart is:

$$\text{EFE}_\text{selected}(t) = G_\text{neg}\!\left(\arg\max_\pi\; q_t(\pi)\right)$$

This is the EFE value of the policy the agent actually commits to at time $t$. It is distinct from $\min_\pi G_\text{neg}$ (the best possible policy) or the mean over all policies, and is therefore the most behaviourally meaningful quantity to track over time.

### 3.7 Action Sampling — `sample_action()`

$$\pi_t \sim q_t(\pi) \qquad a_t = \pi_t[f] \;\text{ for each controllable factor } f$$

### 3.8 Summary

| Equation | Role | Precision |
|----------|------|-----------|
| $q(s_f) = \text{softmax}(\log D[f] + \beta \sum_m \log A_m^{o_m, s_f})$ | Belief update | **β** |
| $q(s'_f\|\pi) = B_f(\cdot, a_f)^\top q(s_f)$ | State propagation | — |
| $P(o_m\|\pi) = A_m \otimes \bigotimes_f q(s'_f\|\pi)$ | Observation prediction | — |
| $G(\pi) = -\sum_m H[P(o_m\|\pi)] + \sum_m P(o_m\|\pi) \cdot C_m$ | Expected Free Energy | — |
| $q(\pi) = \text{softmax}(\gamma \cdot G_\text{neg}(\pi))$ | Policy posterior | **γ** |
| $\text{EFE}_\text{selected}(t) = G_\text{neg}(\arg\max q(\pi))$ | Logged + plotted quantity | — |
| $a_t \sim q(\pi)$ | Action sampling | — |

---

## 4. Output Files

Running the script produces the following files in the working directory:

```
Designer-Artifact.png          9-panel scatter: Designer actions, Artifact true states,
                               Designer beliefs, observations over time

Artifact-User.png              Same format for the Artifact↔User dyad

User-Artifact.png              Same format for the User↔Artifact dyad

results/efe_timeseries.png     Three-line EFE time-series (also shown interactively)
```

Console output example:

```
EFE plot saved to results/efe_timeseries.png
  Designer:   mean=-0.539, std=0.000, min=-0.539, max=-0.539
  Smartphone: mean=-0.000, std=0.000, min=-0.000, max=-0.000
  User:       mean=-0.500, std=0.000, min=-0.500, max=-0.500
```

The `std=0.000` for the Designer confirms the rigid-policy signature: the EFE of the selected policy never changes across 100 timesteps.

---

## 5. How to Run

**Requirements:** Python 3.x with `pymdp`, `numpy`, `matplotlib` installed.

```bash
# From the command line
python dau_active_inference.py --duration 100

# In IDLE — just press F5 (defaults to 100 timesteps)
```

---

## 6. How to Reconfigure

All scenario parameters are defined near the top of the file, just below the `PrecisionAgent` class:

```python
## Precision values — modify here to change scenario
_BETA_DSG  = 0.05   # Designer: low beta  = ignores observations
_GAMMA_DSG = 10.0   # Designer: high gamma = deterministic policy
_BETA_ART  = 1.0    # Smartphone: baseline
_GAMMA_ART = 1.0
_BETA_USR  = 1.0    # User: baseline
_GAMMA_USR = 1.0

## To restore baseline behaviour for all agents:
# _BETA_DSG = 1.0;  _GAMMA_DSG = 1.0
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| `beta` | `0.0 – 5.0` | Sensory trust: 0 = ignores all observations, 1 = standard Bayes, >1 = hypersensitive |
| `gamma` | `0.1 – 20.0` | Decisiveness: low = exploratory/random, high = deterministic |
| `A` matrices | — | Redefine the sensory channel (noise, ambiguity) |
| `B` matrices | — | Redefine action efficacy and transition dynamics |
| `C` vectors | — | Redefine agent goals (what observations are preferred) |
| `D` vectors | — | Redefine initial state beliefs |
