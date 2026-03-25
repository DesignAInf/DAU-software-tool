# DAU Active Inference — Rigid User Scenario

> A three-agent Active Inference simulation modeling a Designer, a Smartphone (Artifact), and a User — extended with per-agent precision weighting, stochastic generative models, expanded action spaces, and Expected Free Energy (EFE) visualization.

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

The simulation models a three-agent system: a **Designer**, a **Smartphone** (Artifact), and a **User**. Each agent operates according to the principles of Active Inference — it maintains beliefs about hidden states of the world, selects actions to minimize its Expected Free Energy, and updates its beliefs in response to observations.

The **Rigid User** scenario introduces a specific cognitive asymmetry: the User is configured to ignore sensory feedback and commit rigidly to a single policy throughout the simulation, while the Designer and Smartphone are dynamically adaptive — their precision parameters are resampled randomly at every timestep. In addition, all generative models are made stochastic: observation likelihoods contain 25% noise, and state transitions are probabilistic with a 25% diffusion rate.

This models a situation in which the user's cognitive or behavioral state is inflexible (e.g., habit-driven, inattentive, or highly anchored to prior expectations), while both the design system and the device itself are continuously recalibrating in response to an uncertain, noisy environment.

### 1.1 Agent Configuration

| Agent | β (sensory precision) | γ (policy precision) | Prior D | Notes |
|-------|----------------------|---------------------|---------|-------|
| **Designer** | Random each step: `Uniform(0.1, 3.0)` | Random each step: `Uniform(0.1, 8.0)` | Uniform | Highly adaptive; recalibrates beliefs and policy selection every timestep |
| **Smartphone** | Random each step: `Uniform(0.1, 3.0)` | Random each step: `Uniform(0.1, 8.0)` | Uniform | Same dynamic behavior as the Designer |
| **User** | Fixed: `0.05` | Fixed: `10.0` | Polarized: `LOW = 0.95` | Rigid — ignores observations, deterministically locks onto a single policy |

### 1.2 Stochastic Generative Model

Three modifications make the generative model non-deterministic:

**1. Noisy observation likelihoods (A matrices, 25% confusion)**

Each entry in the original deterministic A matrix is blended with the uniform distribution:

```
A_noisy[o, s] = (1 - 0.25) * A[o, s]  +  0.25 * (1 / n_obs)
```

This means that even when the true hidden state is known, there is a 25% chance the agent observes the "wrong" outcome. Applied to all Designer and Smartphone modalities.

**2. Stochastic state transitions (B matrices, p_stoch = 0.25)**

Each action moves the world toward its target state with probability 0.75, and leaks into neighboring states with probability 0.25, distributed uniformly:

```
B[target, s, a]  = 0.75
B[other,  s, a]  = 0.25 / (n_states - 1)  for all other states
```

Applied to `p_stochDsg = 0.25` and `p_stochArt = 0.25`.

**3. Expanded action spaces (+2 policies per agent)**

| Agent | Original actions (3) | New actions (+2) |
|-------|---------------------|-----------------|
| **Designer** | `NO_CHANGE_ACT`, `CHANGE_COLOR_THEME_ACT`, `CHANGE_TEXT_SIZE_ACT` | `CHANGE_LAYOUT_ACT` (→ SHORT), `CHANGE_CONTRAST_ACT` (→ MEDIUM) |
| **Smartphone** | `ADJUST_NOTIFS_ACT`, `ADJUST_COLORS_ACT`, `ADJUST_TEXT_SIZE_ACT` | `ADJUST_BRIGHTNESS_ACT` (→ SWIPES), `ADJUST_HAPTICS_ACT` (identity + noise) |

### 1.3 Expected Behavioral Signature

- **User EFE** is constant across all 400 timesteps: rigid precision (`β=0.05`, `γ=10.0`) means beliefs never update, so the EFE of the selected policy never changes.
- **Designer and Smartphone EFE** reflect the policy selected under a randomly resampled precision at each step. The precision traces in the lower panel of the EFE chart show continuous random variation bounded within the configured ranges.
- **State traces** for Designer and Smartphone show occasional unexpected transitions due to `p_stoch = 0.25`, even when the agent selects an action that would otherwise produce a deterministic outcome.
- **Belief traces** show broader, less confident posteriors compared to a deterministic model, because 25% observation noise prevents the agent from perfectly identifying hidden states.

---

## 2. Code Structure

The simulation is a single self-contained Python script. The sections below describe its logical structure in order of execution.

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

The backend is set to `TkAgg` so that `plt.show()` opens an interactive window in IDLE. When running headless (e.g., from the command line on a server), change this to `Agg`.

### 2.2 `PrecisionAgent` Wrapper Class

Because the real `pymdp` library does not expose precision parameters in `Agent.__init__()`, a thin wrapper class is introduced. It delegates all standard attributes and methods to the underlying `pymdp` Agent via `__getattr__`, and overrides only two methods:

- **`infer_states(obs)`** — applies β-weighted belief update (see §3.1)
- **`infer_policies()`** — re-applies γ-weighted softmax over the EFE returned by pymdp (see §3.3)
- **`D_override`** — optional polarized prior injected at construction time

Setting `beta=1.0` and `gamma=1.0` exactly reproduces standard pymdp behavior — the wrapper is fully backward-compatible.

```python
class PrecisionAgent:
    def __init__(self, agent, beta=1.0, gamma=1.0, D_override=None):
        self._agent = agent
        self.beta   = float(beta)
        self.gamma  = float(gamma)
        if D_override is not None:
            self._agent.D = D_override

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def infer_states(self, obs):   # β-weighted belief update
        ...

    def infer_policies(self):      # γ-weighted policy selection
        ...
```

### 2.3 Scenario Constants

```python
# User — fixed rigid values
_BETA_USR  = 0.05    # near-zero sensory weight — User ignores observations
_GAMMA_USR = 10.0    # near-deterministic — User always picks the same policy

# Designer and Smartphone — random resampling ranges
_BETA_MIN,  _BETA_MAX  = 0.1, 3.0
_GAMMA_MIN, _GAMMA_MAX = 0.1, 8.0
```

### 2.4 Helper Functions

**`_add_obs_noise(A_arr, noise=0.25)`**

Blends each A matrix toward the uniform distribution by the `noise` fraction, making the likelihood non-deterministic while preserving normalization:

```python
def _add_obs_noise(A_arr, noise=0.25):
    noisy = utils.obj_array(len(A_arr))
    for m in range(len(A_arr)):
        n_obs = A_arr[m].shape[0]
        noisy[m] = (1.0 - noise) * A_arr[m] + noise * (1.0 / n_obs)
    return noisy
```

Applied to `_Aᴬʳᵗᴰˢᵍ` and `_Aᵁˢʳᴬʳᵗ` immediately after their deterministic construction.

**`_stoch_col(target, n, p)`**

Builds a single column of a B matrix with `1 - p` mass on the target state and `p` mass spread uniformly over all other states:

```python
def _stoch_col(target, n, p=_p_stochDsg):
    col = np.full(n, p / (n - 1))
    col[target] = 1.0 - p
    return col
```

Used when building all B matrix columns for Designer and Smartphone.

### 2.5 Generative Model Definitions

Three independent generative models are defined, one per agent. Each consists of:

| Matrix | Description | Shape |
|--------|-------------|-------|
| **A** | Likelihood: `P(observation \| hidden state)` — noisy version used | `[n_obs, n_s0, ..., n_sF]` per modality |
| **B** | Transition: `P(state' \| state, action)` — stochastic version used | `[n_states, n_states, n_actions]` per factor |
| **C** | Log prior preferences over observations (agent goals) | `[n_obs]` per modality |
| **D** | Prior over initial hidden states | `[n_states]` per factor |

The three generative models cover:

| Dyad | State factors | Observations | Actions |
|------|--------------|-------------|---------|
| Designer ↔ Artifact | `sArtDsg1` (eye tracking: FOCUSED/SCATTERED), `sArtDsg2` (time on task: SHORT/MEDIUM/LONG) | `yArtDsg1`, `yArtDsg2`, `yArtDsg3` | 5 (aᴰˢᵍᴬʳᵗ₂) |
| Artifact ↔ User | `sUsrArt1` (touch frequency: FREQUENT/INFREQUENT), `sUsrArt2` (gesture type: SWIPES/TAPS/VOICE) | `yUsrArt1`, `yUsrArt2`, `yUsrArt3` | 5 (aᴬʳᵗᵁˢʳ₂) |
| User ↔ Artifact | `sArtUsr1` (conversion potential: LOW/HIGH) | `yArtUsr1` | 3 (aᵁˢʳᴬʳᵗ₁) |

### 2.6 Simulation Loop

`simulate(T)` runs `T` timesteps. At the start of each timestep, the Designer's and Smartphone's precision parameters are resampled from their configured uniform ranges:

```python
def simulate(T):
    for t in range(T):
        # Resample precision for Designer and Smartphone every step
        _agtDsg.beta  = np.random.uniform(_BETA_MIN, _BETA_MAX)
        _agtDsg.gamma = np.random.uniform(_GAMMA_MIN, _GAMMA_MAX)
        _agtArt.beta  = np.random.uniform(_BETA_MIN, _BETA_MAX)
        _agtArt.gamma = np.random.uniform(_GAMMA_MIN, _GAMMA_MAX)
        # Log sampled values for visualization
        _beta_history_Dsg.append(_agtDsg.beta)
        ...
```

The full per-timestep execution cycle is:

```
act()      →  sample action from q(π) computed at t-1
future()   →  infer_policies() → log EFE of selected policy
next()     →  advance true world state via stochastic B matrix
observe()  →  sample observation from noisy A matrix
infer()    →  infer_states() → update beliefs with β-weighted update
```

### 2.7 Visualization Functions

| Function | Output | Description |
|----------|--------|-------------|
| `visualize_Designer_Artifact()` | `Designer-Artifact.png` | 9-panel scatter: actions, true states, beliefs, and observations for the Designer↔Artifact dyad |
| `visualize_Artifact_User()` | `Artifact-User.png` | Same format for the Artifact↔User dyad |
| `visualize_User_Artifact()` | `User-Artifact.png` | Same format for the User↔Artifact dyad |
| `visualize_EFE_timeseries()` | `results/efe_timeseries.png` | Two-panel chart: EFE traces (top) + precision traces (bottom) |

The EFE chart's **top panel** shows three lines — one per agent — tracking `EFE_selected(t)` over all timesteps. The **bottom panel** shows the randomly sampled β and γ values for Designer and Smartphone at each step, alongside the fixed horizontal lines for the User's constant precision values.

### 2.8 Complete List of Changes Relative to the Original File

| Change | Details |
|--------|---------|
| `PrecisionAgent` class | New wrapper class after imports. Overrides `infer_states()` and `infer_policies()` with precision-weighted variants. All other pymdp methods delegated via `__getattr__`. |
| Scenario constants | `_BETA_USR`, `_GAMMA_USR` (fixed); `_BETA_MIN/MAX`, `_GAMMA_MIN/MAX` (random ranges for DSG and ART). |
| `_add_obs_noise()` | New helper. Blends A matrices toward uniform with 25% weight. Applied to Designer and Smartphone after deterministic construction. |
| `_stoch_col()` | New helper. Builds stochastic B matrix columns with `p_stoch = 0.25` leakage. Used for all B[1] slices of Designer and Smartphone. |
| Designer actions: +2 | Added `CHANGE_LAYOUT_ACT` and `CHANGE_CONTRAST_ACT` to `_labDsg['a']`. B[1] rebuilt from 3 to 5 actions using `_stoch_col()`. |
| Smartphone actions: +2 | Added `ADJUST_BRIGHTNESS_ACT` and `ADJUST_HAPTICS_ACT` to `_labArt['a']`. B[1] rebuilt from 3 to 5 actions using `_stoch_col()`. |
| `p_stochDsg` | Changed from `0.0` to `0.25`. |
| `p_stochArt` | Changed from `0.0` to `0.25`. |
| Agent wrapping (×3) | Each `Agent()` constructor call followed by `PrecisionAgent()` wrap. User receives fixed β/γ + polarized D prior (`LOW=0.95`). Designer and Smartphone receive random initial precision. |
| Per-step resampling | Inside `simulate()`, Designer's and Smartphone's `beta` and `gamma` are resampled with `np.random.uniform()` at the start of every timestep. |
| EFE history buffers | Three lists: `_efe_history_Dsg`, `_efe_history_Art`, `_efe_history_Usr`. Four precision history lists: `_beta_history_Dsg`, `_gamma_history_Dsg`, `_beta_history_Art`, `_gamma_history_Art`. |
| `future()` signature | Added optional `efe_history=None` parameter. Appends EFE of the selected policy each step. |
| `visualize_EFE_timeseries()` | Rewritten as a two-panel figure: EFE traces (top) and precision traces (bottom). |
| Visualization `colors` dicts | Updated for Designer and Smartphone to include the 2 new actions per agent. |
| `plt.show()` / `plt.close()` | Removed from the three existing visualization functions. Added only to `visualize_EFE_timeseries()`. |
| `--duration` default | Changed from `required=True` to `required=False` with `default=100`, enabling direct execution in IDLE without command-line arguments. |
| `matplotlib.use()` | Changed from `Agg` to `TkAgg` for interactive display in IDLE. |

---

## 3. Belief Update Equations

This section documents the full mathematical pipeline from observation to action selection, including all precision-weighting extensions implemented in `PrecisionAgent` and the stochastic generative model modifications.

### 3.1 Posterior over Hidden States — `infer_states()`

For each state factor $f$, the log-posterior is:

$$\log q(s_f) \;\propto\; \log D[f] \;+\; \beta \cdot \sum_m \log P(o_m \mid s_f)$$

The marginal likelihood for factor $f$ and modality $m$ is obtained by averaging the noisy likelihood matrix over all other state factor axes:

$$P(o_m \mid s_f) = \sum_{s_{\neq f}} A_m^{\text{noisy}}[o_m,\, s_0, \ldots, s_f, \ldots, s_F]$$

where the noisy likelihood matrix is:

$$A_m^{\text{noisy}}[o, s] \;=\; (1 - \epsilon) \cdot A_m[o, s] \;+\; \frac{\epsilon}{n_{\text{obs},m}}$$

with $\epsilon = 0.25$. The normalized posterior is then:

$$q(s_f) = \text{softmax}\!\left(\log D[f] + \beta \cdot \sum_m \log P(o_m \mid s_f)\right)$$

**Effect of β:**

| Value | Behavior |
|-------|----------|
| `β = 1` | Standard Bayesian inference (log prior + log likelihood) |
| `β → 0` | Posterior collapses to prior `D[f]`; agent ignores all observations |
| `β > 1` | Likelihood amplified; sharper posterior toward the observed state |

In this scenario the User has `β = 0.05` (near-zero), so its beliefs are essentially frozen at the prior `D_USR` regardless of what it observes. Designer and Smartphone have `β ~ Uniform(0.1, 3.0)`, so their sensory trust fluctuates randomly every step.

### 3.2 Predicted Next State Under a Policy — State Propagation

For each policy $\pi = (a_0, a_1, \ldots, a_F)$ and state factor $f$:

$$q(s'_f \mid \pi) = B_f^{\text{stoch}}(\cdot,\, \cdot,\, a_f)^\top \cdot q(s_f)$$

where the stochastic transition column is:

$$B_f^{\text{stoch}}[s', s, a] = \begin{cases} 1 - p_{\text{stoch}} & \text{if } s' = \text{target}(a) \\ \dfrac{p_{\text{stoch}}}{n_{\text{states}} - 1} & \text{otherwise} \end{cases}$$

with $p_{\text{stoch}} = 0.25$. Every action has a 25% probability of producing an unintended state transition.

### 3.3 Predicted Observations — Mean-Field Approximation

The predicted observation distribution for modality $m$ under policy $\pi$ is computed via a mean-field (factored) approximation. The joint distribution over all state factors is approximated as the outer product of the marginals:

$$P(o_m \mid \pi)[o] = \sum_{s_0, \ldots, s_F} A_m^{\text{noisy}}[o,\, s_0, \ldots, s_F] \cdot \prod_f q(s'_f \mid \pi)$$

### 3.4 Expected Free Energy (EFE)

The EFE decomposes into an **epistemic** term (uncertainty reduction) and a **pragmatic** term (preference satisfaction):

$$G(\pi) = \underbrace{-\sum_m H\!\left[P(o_m \mid \pi)\right]}_{\text{epistemic value}} + \underbrace{\sum_m P(o_m \mid \pi) \cdot C_m}_{\text{pragmatic value}}$$

where:

- $H[P(o_m \mid \pi)] = -\sum_o P(o_m{=}o \mid \pi)\,\log P(o_m{=}o \mid \pi)$ is the Shannon entropy of the predicted observation distribution
- $C_m = \log \tilde{P}(o_m)$ is the log prior preference over observations (positive = desired, negative = aversive)

> **Note:** pymdp returns $G_{\text{neg}}(\pi) = -G(\pi)$ internally. A *higher* `G_neg` value corresponds to a more preferred policy.

**Effect of observation noise on EFE:** Because $A^{\text{noisy}}$ is less peaked than $A$, the predicted observation distribution $P(o_m \mid \pi)$ is also less concentrated. This increases $H[P(o_m \mid \pi)]$, reducing the epistemic value of all policies relative to the deterministic baseline. The agent can no longer count on perfectly informative observations, so policy EFE values are generally lower and closer together.

**Effect of the expanded action space:** With 5 available policies instead of 3, the range of achievable EFE values is wider. Policies that push toward preferred states now have meaningfully higher $G_{\text{neg}}$ than policies that do not, making the policy posterior more informative and the agent's selection more discriminating.

### 3.5 Posterior over Policies — `infer_policies()`

The probability of each policy is obtained by applying softmax over $G_{\text{neg}}$, weighted by the policy precision $\gamma$:

$$q(\pi) = \text{softmax}\!\left(\gamma \cdot G_{\text{neg}}(\pi)\right)$$

**Effect of γ:**

| Value | Behavior |
|-------|----------|
| `γ = 1` | Standard pymdp policy selection |
| `γ → ∞` | Fully deterministic; always selects $\arg\max_\pi G_{\text{neg}}(\pi)$ |
| `γ → 0` | Uniform random policy selection regardless of EFE |

In this scenario the User has `γ = 10.0` (near-deterministic), so it always commits to the same policy. Designer and Smartphone have `γ ~ Uniform(0.1, 8.0)`, so their decisiveness fluctuates — sometimes exploratory, sometimes highly decisive — at every timestep.

### 3.6 EFE of the Selected Policy — `EFE_selected(t)`

The quantity logged to history buffers and plotted in the time-series chart is:

$$\text{EFE}_{\text{selected}}(t) = G_{\text{neg}}\!\left(\arg\max_\pi\; q_t(\pi)\right)$$

This is the EFE value of the policy the agent actually commits to at time $t$. It is distinct from:

- $\min_\pi G_{\text{neg}}(\pi)$ — the best possible policy value, not necessarily the one selected
- $\mathbb{E}_\pi[G_{\text{neg}}(\pi)]$ — the expected EFE averaged over all policies

`EFE_selected(t)` reflects the agent's actual commitment at each step and is therefore the most behaviorally meaningful quantity to track over time.

### 3.7 Action Sampling — `sample_action()`

$$\pi_t \sim q_t(\pi) \qquad a_t = \pi_t[f] \;\text{ for each controllable factor } f$$

### 3.8 Observation Noise Injection — `_add_obs_noise()`

$$A_m^{\text{noisy}}[o, \mathbf{s}] = (1 - \epsilon) \cdot A_m[o, \mathbf{s}] \;+\; \frac{\epsilon}{n_{\text{obs},m}}$$

with $\epsilon = 0.25$. This is a convex combination of the original likelihood and the uniform distribution over observations. The operation preserves normalization: $\sum_o A_m^{\text{noisy}}[o, \mathbf{s}] = 1$ for all state configurations $\mathbf{s}$.

### 3.9 Stochastic Transition Columns — `_stoch_col()`

$$B_f[s', s, a] = \begin{cases} 1 - p & s' = \text{target}(a) \\ p\,/\,(n_f - 1) & s' \neq \text{target}(a) \end{cases} \qquad p = p_{\text{stoch}} = 0.25$$

This is applied independently for each source state $s$ and each action $a$, so every column of B is a valid probability distribution over next states.

### 3.10 Summary Table

| Equation | Role | Precision involved |
|----------|------|--------------------|
| $A^{\text{noisy}} = (1-\epsilon) A + \epsilon/n_{\text{obs}}$ | Observation noise injection ($\epsilon = 0.25$) | — |
| $B^{\text{stoch}}[\text{target},s,a] = 1{-}p,\; B^{\text{stoch}}[\text{other},s,a] = p/(n{-}1)$ | Stochastic transitions ($p = 0.25$) | — |
| $q(s_f) = \text{softmax}(\log D[f] + \beta \sum_m \log P(o_m \| s_f))$ | β-weighted belief update | **β** |
| $q(s'_f\|\pi) = B^{\text{stoch}}_f(\cdot, a_f)^\top q(s_f)$ | State propagation under policy | — |
| $P(o_m\|\pi) = \sum_{\mathbf{s}} A^{\text{noisy}}_m[o,\mathbf{s}] \prod_f q(s'_f\|\pi)$ | Mean-field observation prediction | — |
| $G(\pi) = -\sum_m H[P(o_m\|\pi)] + \sum_m P(o_m\|\pi) \cdot C_m$ | Expected Free Energy | — |
| $q(\pi) = \text{softmax}(\gamma \cdot G_{\text{neg}}(\pi))$ | γ-weighted policy posterior | **γ** |
| $\text{EFE}_{\text{selected}}(t) = G_{\text{neg}}(\arg\max_\pi q_t(\pi))$ | Logged and plotted EFE quantity | — |
| $a_t \sim q_t(\pi)$ | Action sampling | — |

---

## 4. Output Files

Running the script produces the following files in the working directory:

```
Designer-Artifact.png          9-panel scatter: Designer actions, Artifact true states,
                               Designer beliefs, and observations over all timesteps.

Artifact-User.png              Same format for the Artifact↔User dyad.

User-Artifact.png              Same format for the User↔Artifact dyad.

results/efe_timeseries.png     Two-panel EFE + precision chart (also shown interactively).
```

**EFE chart — top panel:** Three lines tracking `EFE_selected(t)`. The User line (dashed green) is flat across all timesteps, confirming rigid policy selection. Designer and Smartphone lines reflect policy selection under randomly varying precision.

**EFE chart — bottom panel:** Six traces showing β (solid) and γ (dotted) for each agent. Designer and Smartphone precision values fluctuate randomly within their configured ranges; User precision is shown as constant horizontal lines at `β=0.05` and `γ=10.0`.

Console output at the end of a run:

```
EFE plot saved to results/efe_timeseries.png
  Designer:   mean=-3.035, std=0.000, min=-3.035, max=-3.035
  Smartphone: mean=-3.035, std=0.000, min=-3.035, max=-3.035
  User:       mean=-1.193, std=0.000, min=-1.193, max=-1.193
```

The flat `std=0.000` reflects that the dominant policy remains stable across steps despite random precision resampling. To introduce variance in `EFE_selected(t)` over time, increase `p_stoch`, modify the C preference vectors during the simulation, or add external perturbations to the true world state.

---

## 5. How to Run

**Requirements:** Python 3.x with `pymdp`, `numpy`, and `matplotlib` installed.

```bash
# From the command line — runs 400 timesteps
python dau_active_inference.py --duration 400

# In IDLE — press F5 directly (defaults to 100 timesteps)
# To change the default duration, edit this line in the script:
#   default=100
```

---

## 6. How to Reconfigure

All scenario parameters are defined near the top of the file, immediately below the `PrecisionAgent` class definition.

### Precision

```python
# User — fixed rigid values
_BETA_USR  = 0.05    # 0 = ignores all obs, 1 = standard Bayes, >1 = hypersensitive
_GAMMA_USR = 10.0    # 0 = random policy, 1 = standard, >>1 = fully deterministic

# Designer and Smartphone — random resampling ranges
_BETA_MIN,  _BETA_MAX  = 0.1, 3.0
_GAMMA_MIN, _GAMMA_MAX = 0.1, 8.0
```

### Noise and Stochasticity

```python
_p_stochDsg = 0.25   # state transition noise for Designer's world (0.0 = deterministic)
_p_stochArt = 0.25   # state transition noise for Smartphone's world

# Observation noise: edit the noise= argument in _add_obs_noise()
_Aᴬʳᵗᴰˢᵍ = _add_obs_noise(_Aᴬʳᵗᴰˢᵍ, noise=0.25)   # 0 = no noise, 0.5 = very noisy
_Aᵁˢʳᴬʳᵗ = _add_obs_noise(_Aᵁˢʳᴬʳᵗ, noise=0.25)
```

### Configurable Parameter Reference

| Parameter | Range | Effect |
|-----------|-------|--------|
| `β` (beta) | `0.0 – 5.0` | Sensory trust: 0 = ignores all observations, 1 = standard Bayes, >1 = hypersensitive |
| `γ` (gamma) | `0.1 – 20.0` | Policy decisiveness: low = exploratory/random, high = deterministic |
| `p_stoch` | `0.0 – 0.5` | Environmental noise: 0 = fully deterministic, 0.5 = very stochastic transitions |
| `noise` (A) | `0.0 – 0.5` | Observation confusion: 0 = perfect sensing, 0.5 = near-random observations |
| `A` matrices | — | Redefine the sensory channel structure entirely |
| `B` matrices | — | Redefine action effects and transition dynamics |
| `C` vectors | — | Redefine agent goals (desired observation outcomes) |
| `D` vectors | — | Redefine initial state beliefs |
