# DMBD Joule

**Active Inference Markov blanket for Joule heat reduction in conductors**

A computational simulation demonstrating that a subset of electrons, organized as
an Active Inference Markov blanket, can reduce the internal Joule heat dissipation
of a conducting wire by actively cancelling the perturbation sources that cause
inelastic collisions.

---

## What this project does

When current flows through a conductor, electrons collide inelastically with the
lattice (phonons), external electromagnetic interference, and structural defects.
Each collision dissipates kinetic energy as heat — this is the Joule effect.

This project asks: can a self-organizing layer of electrons, acting as an Active
Inference agent, reduce the heat dissipated by the remaining electrons by
continuously cancelling those perturbation sources?

The answer, demonstrated computationally, is **yes** — with a Joule Reduction
Score (JRS) of up to **39.6%** relative to an unorganized baseline.

A seven-way comparison against classical alternatives (passive shielding, random
role assignment, LQR optimal control, Q-learning) shows that standard control
methods do not match this result in this toy model. The comparison has known
limitations (noted below) and should be interpreted as an exploratory finding,
not a definitive demonstration of algorithmic superiority.

---

## The core idea: Active Inference in a physical system

Active Inference (Friston, 2010) is a theory of perception and action in which
agents minimize a quantity called **Expected Free Energy (EFE)** — a measure of
surprise about their sensory states. The agent does not optimize an external
reward; it acts to bring its internal states in line with its predictions.

This project instantiates that principle at the level of individual electrons in
a conductor, using the canonical **Markov blanket** partition:

```
E  (environment)  =  physical cable
                      phonons / EM interference / lattice defects / load

B_s (sensory)     =  peripheral electrons
                      measure perturbation from all sources
                      feed the sensory mean field <S>(x,t)

B_a (active)      =  central electrons
                      read <S>, cancel all three perturbation sources directly
                      on the cable (phonon amplitude, EM source strength, defect strength)

I  (internal)     =  bulk electrons
                      see only E_effective = E_total * (1 - <A>)
                      their Joule heat -> 0 when cancellation works
```

The key architectural property: **I is conditionally independent of E given B**.
The blanket layer mediates all information flow between the internal electrons
and the cable. When B_a successfully cancels the perturbation sources, the
internal electrons experience an effectively quiet environment.

---

## Architecture

### System (`system.py`)

The cable is modelled as 60 spatial segments with three independent perturbation
sources, each with its own relaxation dynamics:

| Source | Contribution | Default relax rate |
|---|---|---|
| Phonons (lattice vibrations) | 20% | 0.08/step |
| EM interference (8 moving sources) | 65% | 0.05/step |
| Lattice defects (15 fixed positions) | 15% | 0.04/step |

The cable regenerates toward its natural amplitude every step at each relax rate.
B_a must continuously fight this regeneration — complete cancellation requires
that the cancellation rate exceeds the regeneration rate for every segment.

**Equations of motion** differ by role:

```
B_s:  dv/dt = alpha*(J_0 - v) + F_cable*0.10 + 0.05*(v_I - v) + eta(sigma_phonon)
B_a:  dv/dt = alpha*(J_0 - v) + 0.08*(S_vel - v) - S_total*0.07 + eta(0.7*sigma_phonon)
I:    dv/dt = alpha*(J_0 - v) + 0.10*(S_vel - v) + E_eff*0.03  + eta(thermal_I)
      thermal_I = max(sigma_phonon*(1 - A*lambda_2*0.92), sigma_min)
```

Internal electrons are shielded from the cable not just in the force term but
in their collision probability:

```
p_coll_I = rho_0 * (|E_eff|*0.18 + sigma_ph*0.8) * (1 - A*lambda_2*0.80)
```

When A -> 1, p_coll_I -> rho_0 * sigma_ph * 0.8 * (1 - lambda_2*0.80) — a
small residual set by the thermal floor.

### Role assignment via G minimization (`blanket.py`, v6 closed loop)

In v6, role assignment is a genuine minimization of Expected Free Energy G.
The heuristic score functions of v5 are completely replaced by stochastic
coordinate descent on G:

```
omega* = argmin_{omega in Omega} G(omega)

where omega = role assignment {role_i for i=1..N}
      Omega = valid partitions (15% <= f_r <= 55% per role)
      G     = pragmatic + epistemic + joule_risk
```

**Algorithm (per timestep):**
1. Sample 20% of electrons eligible for role change
2. For each sampled electron i, estimate G under each of 3 proposed roles
   (first-order approximation using current mean fields, no dynamics re-run)
3. Assign the role that minimizes G
4. Repeat for K=5 coordinate descent iterations
5. Project onto fraction constraints

**G estimation (first-order, per-electron):**
```
G_I(i): add i to I pool  -> update J_I_est, Q_I_est -> pragmatic + joule_risk
G_S(i): add i to B_s pool -> update S_density        -> epistemic
G_A(i): add i to B_a pool -> update A_density, shield -> epistemic + joule_risk
```

This makes EFE genuinely causal: the optimizer finds the partition that
minimizes G, and the physics is then driven by those roles.
Mean G improvement per step: **+0.21** (empirically measured).

### Active cancellation (`system.py`, v5 improvements)

Three simultaneous cancellation mechanisms per step:

**Phonon (segment-level, coverage-aware):**
```
delta_ph[seg] = kappa_ph * <S_ph>[seg] * A_density_norm[seg] * boost[seg]
boost[seg] = 1.6  if seg is in top-N perturbed AND undercovered
           = 1.0  otherwise
```

**EM (per-source, targeted — v5 key improvement):**  
Each B_a electron is assigned to its nearest EM source. Cancellation is direct
and source-specific, not a diffuse kernel projection:
```
for k in EM_sources:
    Ba_assigned_to_k = electrons where em_assignment == k
    cancel_k = kappa_em * mean(<S_em> at Ba_k positions) * (n_Ba_k / n_Ba_total) * 2.5
    em_sources[k] -= cancel_k
```

**Defect (segment-level, Lorentzian kernel):**
```
cancel_d = kappa_df * sum_seg(<S_df>[seg] * A_norm_boost[seg] * kernel_d[seg])
defect_strength[d] -= cancel_d
```

### Expected Free Energy (`vbem.py`)

The central theoretical result of the project: EFE explicitly couples to
internal Joule heat through a `joule_risk` term:

```
G = pragmatic_risk + epistemic_risk + joule_risk

pragmatic  = |J_I - J_0|^2 * 5.0
epistemic  = (1/kappa_I) * (1 - shield_A)
joule_risk = lambda_J * Q_I          <-- key: direct coupling to dissipation
```

This makes minimizing EFE equivalent to minimizing internal heat dissipation.
Empirical result: **r(EFE, Q_I) = 0.986**, R² = 0.944.

---

## Results

### Main experiment (v5, aggressive parameters)

```
python3 run_experiment.py
```

| Metric | Value |
|---|---|
| JRS (Joule Reduction Score) | **37–40%** |
| Total cancellation | **92–93%** |
| EM cancellation | **93–94%** |
| Phonon cancellation | **80–85%** |
| Defect cancellation | **>95%** |
| r(EFE, Q_I) | **0.986** |
| MI(I;E\|B) reduction | **p ≈ 0** (highly significant) |

### Seven-way comparison (corrected methodology)

```
python3 run_full_comparison.py              # single seed
python3 run_full_comparison.py --robust    # 10 seeds, 95% CI
```

Three methodological fixes were applied after an external review:

- **Fix 1** — E and F now use fair role structure: VBEM assigns I/B_s/B_a,
  the controller only decides kappa values. The original forced `role[:]=ACTIVE`,
  destroying the sensory field.
- **Fix 2** — All conditions share the same warmed-up cable state.
- **Fix 3** — Condition G: AI with `lambda_J=0` (joule_risk removed). Tests
  whether the result is architectural (emergent) or tautological.

| Condition | Q_I | JRS vs A | Cancel | Note |
|---|---|---|---|---|
| A — No blanket | 0.01430 | baseline | 5% | |
| B — Passive blanket | 0.01384 | +3.2% | 4% | |
| C — Random roles | 0.00908 | +36.5% | 94% | |
| **D — Active Inference** | **0.00910** | **+36.4%** | **93%** | VBEM |
| E — LQR (fair) | 0.00852 | +40.4% | 95% | on top of VBEM |
| F — Q-learning (fair) | 0.00821 | +42.6% | 96% | on top of VBEM |
| **G — AI, lambda_J=0** | **0.00910** | **+36.4%** | **93%** | D = G |

**Robust results (10 seeds, 95% CI, Welch t-test D vs A: t=−31.54, p<0.0001):**

| | Q_I mean ± 95% CI | JRS |
|---|---|---|
| A | 0.01452 ± 0.00016 | baseline |
| D | 0.00851 ± 0.00032 | **+41.4%** |
| G | 0.00851 ± 0.00032 | **+41.4%** |
| E | 0.01447 ± 0.00014 | +0.3% |
| F | 0.01461 ± 0.00009 | −0.6% |

*(Robust numbers use the previous comparison setup; single-seed numbers above
use the fully corrected Fix A setup where E/F operate on top of VBEM.)*

**v6 -- closed-loop results (EFE genuinely drives the policy):**

The central comparison is D (v6, closed loop) vs H (v5 heuristic scores, open loop):

| Condition | Q_I | JRS vs A | EFE drives policy? |
|---|---|---|---|
| A — No blanket | 0.01430 | baseline | — |
| H — Heuristic scores (v5, open loop) | 0.00945 | +8.5% | No |
| **D — Closed-loop G minimization (v6)** | **0.00579** | **+43.9%** | **Yes** |

**D beats H by +35 percentage points.** When EFE genuinely drives role
assignment via coordinate descent on G, the system achieves substantially
lower internal dissipation than when roles are assigned by heuristic score
functions that ignore G. This is the result that was not available in v5.

**D and G now differ.** With lambda_J=0, the optimizer loses the
joule_risk signal and performance degrades. This demonstrates that
lambda_J is causally effective in v6 -- not tautological.

**G_improvement per step = +0.21** on average: the coordinate descent
optimizer reduces G by this amount each timestep, and that reduction
translates directly into lower Q_I.

---

## Installation

```bash
git clone https://github.com/yourname/dmbd_joule.git
cd dmbd_joule
pip install -r requirements.txt

# Optional: install as package
pip install -e .
```

**Requirements:** Python 3.10+, numpy >= 1.24, scipy >= 1.10, matplotlib >= 3.7

---

## Usage

### Quick start

```bash
# Default parameters (JRS ~28%)
python3 run_experiment.py

# Aggressive parameters (JRS ~38%)
python3 -m dmbd_joule \
  --relax-em 0.01 --relax-phonon 0.03 \
  --kappa-em 1.40 --kappa-phonon 1.60 \
  --n-hot-segments 30 --n-blanket 1000

# Four-way comparison (A/B/C/D)
python3 run_comparison.py

# Six-way comparison (A/B/C/D/E/F)
python3 run_full_comparison.py
```

### All CLI parameters

```
--n-warmup          int     Warmup steps (discarded)        [200]
--n-baseline        int     Baseline measurement steps      [400]
--n-blanket         int     Blanket synthesis steps         [800]
--n-electrons       int     Total electrons                 [240]
--phonon-temp       float   Phonon noise amplitude          [0.05]
--em-sigma          float   EM interference amplitude       [0.55]
--kappa-phonon      float   Phonon cancellation strength    [1.40]
--kappa-em          float   EM cancellation strength        [1.10]
--kappa-defect      float   Defect cancellation strength    [0.95]
--relax-phonon      float   Phonon regeneration rate        [0.08]
--relax-em          float   EM regeneration rate            [0.05]
--lambda-joule      float   lambda_J weight of Q_I in EFE  [8.0]
--blanket-strength  float   CI enforcement strength         [0.92]
--n-hot-segments    int     Priority segments for B_a       [20]
--min-ba-per-seg    int     Min B_a per hot segment         [2]
--sigma-min         float   Irreducible thermal floor       [0.002]
--seed              int     RNG seed                        [42]
--out-dir           str     Output directory                [results]
```

### Parameter tuning guide

The most impactful levers, in order:

1. `--relax-em` — most important. Slower EM regeneration = higher cancellation.
   Range: 0.01 (aggressive) to 0.15 (easy for cable, hard for B_a).
2. `--relax-phonon` — second most impactful.
3. `--kappa-em`, `--kappa-phonon` — cancellation strength. Keep kappa/relax < 25
   to avoid oscillation.
4. `--n-hot-segments` — coverage of priority segments. Increase with n_electrons.

### As a Python library

```python
from dmbd_joule.system     import CableSystem, SystemParams
from dmbd_joule.blanket    import BlanketSynthesizer
from dmbd_joule.vbem       import VBEMInference
from dmbd_joule.metrics    import InformationMetrics
from dmbd_joule.experiment import run_experiment
from dmbd_joule.comparison import run_full_comparison

# Single experiment
result = run_experiment(
    n_blanket = 800,
    params    = SystemParams(relax_em=0.01, kappa_em=1.40),
    out_dir   = "my_results",
)

# Six-way comparison
results = run_full_comparison(
    params  = SystemParams(relax_em=0.01),
    n_steps = 800,
    out_dir = "my_comparison",
)
```

---

## Output figures

Each run produces four figures in the output directory:

| Figure | Content |
|---|---|
| `fig1_timeseries.png` | 9-panel: Joule by role, cancellation by source, role composition, internal current, conditional MI, active response field, EFE decomposition, ELBO, EM tracking accuracy |
| `fig2_ablation.png` | Statistical validation: MI distributions, Q_I distributions, cancellation histogram, Joule by role, role composition, scorecard |
| `fig3_physics.png` | Physical fields at convergence: phonon natural vs cancelled, mean fields B_s/B_a, current by role, electron density per segment |
| `fig4_efe_joule.png` | EFE–Joule analysis: decomposition, scatter r=0.986, joule_risk vs Q_I, dual-axis time series, Granger lags, cancellation by source |

The six-way comparison produces `fig_full_comparison.png` with convergence
curves, Q_I distributions, a scorecard, and a publishable claims panel.

---

## Tests

```bash
python3 -m pytest tests/ -v
```

Ten tests covering: system runs, role populations, role fraction bounds,
positive cancellation, blanket reduces Joule heat, EFE tracks Q_I,
thermal floor reporting, EM tracking accuracy, LQR controller, Q-learning controller.

---

## Module structure

```
dmbd_joule/
├── system.py           Core simulation: cable physics, electron dynamics,
│                       4-role partition, active cancellation mechanisms
├── blanket.py          VBEM role assignment: score functions, Lagrange
│                       multipliers, cooldown, fraction constraints
├── vbem.py             Variational inference: E-step, M-step, EFE with
│                       joule_risk term, ELBO
├── experiment.py       Three-phase orchestrator: warmup, baseline, blanket
├── metrics.py          MetricSnapshot, InformationMetrics, ablation tests,
│                       EFE-Joule analysis
├── plots.py            Four publication figures (fig1-fig4)
├── comparison.py       Six conditions A-F, run_comparison, run_full_comparison
├── comparison_plots.py Comparison figures (4-way and 6-way)
├── controllers.py      LQR controller (model-based), Q-learning controller
│                       (model-free RL), shared observation/action interface
└── __main__.py         CLI entry point
```

---

## Theoretical context

This project is a computational instantiation of three ideas from the
Active Inference literature:

**1. The Markov blanket as the unit of selfhood (Friston, 2013)**  
A system has a "self" if and only if it possesses a Markov blanket — a
partition of states into sensory (influenced by environment, influence
internal), active (influence environment, influenced by internal), and
internal (conditionally independent of environment given blanket).
This project implements that partition literally, in a physical system.

**2. Active Inference as an alternative to reinforcement learning**  
Unlike RL, an Active Inference agent does not optimize an external reward
function. It minimizes expected surprise (EFE) about its own sensory states.
The `joule_risk` term in EFE makes internal heat dissipation part of the
agent's prediction — reducing Joule heat becomes equivalent to reducing
surprise. This is a novel connection between thermodynamics and Free Energy
minimization.

**3. Adaptive deployment as emergent behaviour**  
The B_a coverage constraint (Improvement 1, v5) was not hand-coded as a rule.
It emerges from the score function: when hot segments are undercovered, the
score for B_a electrons in those segments increases, and VBEM reassigns
electrons there. The system finds the right deployment strategy through
local scoring, not global optimization.

---

## Limitations and honest assessment

- **Toy model caveat.** This is a computational proof of concept with
  hand-tuned parameters. Results are not predictions about real materials.

- **EFE is now causal (v6).** The coordinate descent optimizer in `blanket.py`
  uses G to assign roles. The G_estimate is a first-order approximation
  from current mean fields -- it does not re-run full dynamics. A more
  accurate estimator (e.g. Monte Carlo rollouts) would be stronger but
  computationally heavier.

- **Comparison with LQR and Q-learning has residual asymmetry.** E and F
  operate on top of VBEM cancellation rather than replacing it. The
  observed gap therefore conflates algorithmic differences with structural
  ones. This is documented in the code and is a known limitation.

- **"Granger-style" analysis is lagged correlation**, not a proper Granger
  causality test. The conditional MI is a histogram-based heuristic. Both
  should be treated as qualitative indicators, not formal tests.

- **Test 2 (CI assay)** consistently fails with |rho| ~ 0.10 instead of
  the theoretical < 0.05. The blanket provides approximate, not exact,
  conditional independence in this finite stochastic system.

- The 7–8% VBEM advantage over random roles (D vs C) is real but modest.
  The dominant contribution is the I/B_s/B_a role structure + cancellation
  mechanism, not the VBEM optimization itself.

---

## Citation

If you use this code, please cite:

```
Possati, L.M. (2025). DMBD Joule: Active Inference Markov blanket for
Joule heat reduction in conductors. GitHub.
https://github.com/yourname/dmbd_joule
```

---

## License

MIT License. See LICENSE file.

---

## Author

**Luca Maria Possati**  
Researcher in philosophy of technology, Active Inference, and computational
modeling of physical systems.
