# DAU v2 – Smartphone Triad Active Inference Simulation

A computational model of the **Designer–Artifact–User (DAU)** triad applied to smartphone interaction.
Three autonomous agents, each running Active Inference, co-exist in a shared environment:

| Agent | Role | Preference |
|---|---|---|
| **Designer** | Sets app defaults & choice architecture | High engagement |
| **Smartphone** | Mediates interface (notifications, friction) | User engaged (KPI) |
| **User** | Self-regulates their own attention | Focus & well-being |

The agents have **conflicting preferences**, which creates emergent tension — the Designer and Smartphone push toward engagement while the User resists.

---

## Files

```
DAU advanced version/
├── config.py          # All parameters (AgentConfig, SimConfig)
├── agents.py          # Agent definitions (A, B, C, D matrices + step loop)
├── inference.py       # Core Active Inference maths (belief update, EFE, action selection)
├── env_smartphone.py  # Environment: coupling rules between the three agents
├── main.py            # Entry point (CLI), simulation loop, summary, plots
├── plotting.py        # EFE time-series and user decisiveness plots
└── self_check.py      # Sanity checks on matrix shapes and normalisation
```

---

## Quick Start

```bash
pip install numpy matplotlib
python -m "DAU advanced version".main --steps 200 --seed 0
```

**CLI options:**

| Flag | Default | Description |
|---|---|---|
| `--steps` | 200 | Number of simulation timesteps |
| `--seed` | 0 | RNG seed for reproducibility |
| `--no-decisiveness-plot` | off | Skip the user policy-entropy plot |

**Outputs** (written to `results/`):
- `dau_v2_efe_timeseries.png` – EFE over time for all three agents
- `dau_v2_user_decisiveness.png` – User policy entropy (commitment over time)

---

## Architecture

### Generative Model

Every agent holds four matrices:

| Symbol | Shape | Meaning |
|---|---|---|
| **A** | `[n_obs, n_states]` | Likelihood: P(observation \| hidden state) |
| **B** | `[n_states, n_states, n_actions]` | Transition: P(next state \| current state, action) |
| **C** | `[n_obs]` | Log-preferences over observations |
| **D** | `[n_states]` | Prior belief over hidden states |

All columns of **A** and **B** are proper probability distributions (sum to 1).

### Interaction Loop (one timestep)

```
[Designer] ─── observes smartphone state ──► action: boost/maintain/friction/wellbeing
     │
     ▼  (blends smartphone's B-matrix via designer_influence = 0.3)
[Smartphone] ── observes user's last action ──► action: send_notif/suppress/recommend/friction
     │
     ▼  (deterministic observation routing)
[User] ──── observes smartphone's action ──► action: engage/ignore/settings/DND/limit
     │
     └── feeds back as next smartphone observation
```

---

## Belief Update Equations

### 1. Perception — updating q(s)

Given observation *o* at time *t*, each agent updates its posterior belief over hidden states using **variational Bayes** (mean-field approximation):

```
log q(s) ∝ log D(s) + β · log A(o | s)
```

In exponential form:

```
q(s) = σ( log D(s) + β · log A(o | s) )
```

where σ(·) denotes normalisation (softmax / L1-norm after exponentiation).

- **D(s)** — prior over states (acts as regulariser each step)
- **A(o | s)** — likelihood of the observed *o* given state *s*
- **β** — likelihood precision (β > 1: trust senses more; β < 1: rely on prior)

Implementation: `inference.py → update_belief(A, D, q_s, obs, beta)`

---

### 2. Planning — Expected Free Energy G(a)

For each candidate action *a*, the agent predicts one step ahead and computes:

```
G(a) = risk(a) + ambiguity(a)
```

**Predicted next state** (one-step roll-out):

```
q(s' | a) = B[:, :, a] · q(s)          (matrix-vector product, then normalised)
```

**Predicted observation**:

```
q(o' | a) = A · q(s' | a)              (marginalising over predicted states)
```

**Risk** (pragmatic value — how far predicted outcomes are from preferred ones):

```
risk(a) = −Σ_o  q(o' | a) · C(o)      (negative expected log-preference)
```

**Ambiguity** (epistemic value — expected uncertainty in the likelihood):

```
ambiguity(a) = Σ_{s'}  q(s' | a) · H[ A(:, s') ]

where  H[ A(:, s') ] = −Σ_o  A(o | s') · log A(o | s')
```

Full EFE:

```
G(a) = −Σ_o q(o'|a)·C(o)  +  Σ_{s'} q(s'|a)·H[A(:,s')]
```

Lower G(a) → action *a* is preferred (better balance of reward and information gain).

Implementation: `inference.py → compute_efe(A, B, C, q_s)`

---

### 3. Action Selection — q(π)

The policy distribution is a **softmax over negative EFE**:

```
q(π = a) = softmax( −γ · G(a) )  =  exp(−γ · G(a)) / Σ_{a'} exp(−γ · G(a'))
```

- **γ** (gamma) — policy precision (inverse temperature)
  - γ → 0 : uniform distribution (random behaviour)
  - γ → ∞ : deterministic argmin G (fully committed)

An action is then **sampled** from q(π) (stochastic policy).

Implementation: `inference.py → select_action(G, gamma, rng)`

---

### 4. User Policy Precision Annealing

The User agent starts with very low precision (near-random behaviour) and becomes increasingly decisive over time:

```
γ_user(t) = γ_init + (γ_final − γ_init) · t / (T − 1)
```

| Parameter | Value |
|---|---|
| γ_init | 0.5 (nearly uniform q(π)) |
| γ_final | 4.0 (fairly decisive) |
| T | `steps` (default 200) |

This models a user who starts without clear habits but gradually learns to act on their preferences.

Implementation: `env_smartphone.py → SmartphoneEnvironment.step()`

---

### 5. Designer Influence on Smartphone's B-matrix

The Designer does not directly control the Smartphone's actions — instead, it **perturbs the Smartphone's transition model** each step:

```
B_perturbed[:, :, a*] = (1 − α) · B_original[:, :, a*]  +  α · B_push
```

where:
- **a\*** is the smartphone action promoted by the Designer's choice (e.g., `boost_notif` promotes `send_notif`)
- **α** = `designer_influence` = 0.3 (blending weight)
- **B_push** is a matrix that strongly drives toward the `aggressive` smartphone state

This coupling is **transient** — B is restored to its original value after each step, so the influence must be re-exerted continuously.

---

## Agent Configurations

### Designer (γ = 2.0)

| State | Observation | Action |
|---|---|---|
| engage_mode | high_engagement | boost_notif |
| balanced_mode | med_engagement | maintain |
| minimal_mode | low_engagement | add_friction |
| — | — | wellbeing_nudge |

**Preferences:** C = [+2.0, 0.0, −1.0] — strongly prefers high engagement.

### Smartphone (γ = 3.0)

| State | Observation | Action |
|---|---|---|
| aggressive | user_engaged | send_notif |
| moderate | user_neutral | suppress_notif |
| calm | user_resistant | show_recommendation |
| — | — | add_friction |

**Preferences:** C = [+3.0, 0.0, −2.0] — maximises engagement, punishes churn.

### User (γ: 0.5 → 4.0)

| State | Observation | Action |
|---|---|---|
| distracted | notif_high | engage |
| focused | notif_low | ignore |
| dnd | friction_barrier | change_settings |
| — | — | enable_dnd |
| — | — | limit_usage |

**Preferences:** C = [−2.0, +1.0, +0.5] — dislikes intrusive notifications, prefers quiet focus.

**Prior D:** uniform [1/3, 1/3, 1/3] — the User starts with no committed model of their own behaviour.

---

## Observation Routing (Environment Coupling)

```
Smartphone action → User observation
  send_notif          → notif_high        (0)
  suppress_notif      → notif_low         (1)
  show_recommendation → notif_high        (0)  [treated as intrusive]
  add_friction        → friction_barrier  (2)

User action → Smartphone observation
  engage              → user_engaged      (0)
  ignore              → user_neutral      (1)
  change_settings     → user_neutral      (1)
  enable_dnd          → user_resistant    (2)
  limit_usage         → user_resistant    (2)

Smartphone state → Designer observation
  aggressive (argmax q_s = 0) → high_engagement (0)
  moderate   (argmax q_s = 1) → med_engagement  (1)
  calm       (argmax q_s = 2) → low_engagement  (2)
```

---

## Key References

- Friston et al. (2017). *Active inference and learning.* Neuroscience & Biobehavioral Reviews.
- Friston et al. (2015). *Active inference and epistemic value.* Cognitive Neuroscience.
- Parr & Friston (2019). *Generalised free energy and active inference.* Biological Cybernetics.
- Da Costa et al. (2020). *Active inference on discrete state-spaces.* Journal of Mathematical Psychology.
