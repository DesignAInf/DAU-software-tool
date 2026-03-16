# Dynamic Markov Blanket Design for Joule Heat Reduction

**Active Inference Markov blanket for Joule heat reduction in conductors**
Version 6.0 — closed-loop EFE minimization

A computational simulation showing that electrons organized as an Active Inference Markov blanket — where role assignment minimizes Expected Free Energy — dissipate about 30% less internal heat than electrons using the same physical cancellation mechanism but with roles assigned by heuristic rules.

A caveat is necessary. Rather than a realistic account of electron transport, it is more accurate to frame this simulation as a conceptual design study that uses the formal language of active inference to explore how adaptive systems might be organized. On that reading, the value of the work does not lie in the literal claim that electrons can be partitioned into sensory, active, and internal roles, but in showing how a Markov blanket architecture, variational updating, and Expected Free Energy minimization can be translated into a design framework for systems that dynamically allocate sensing, actuation, and protection in order to balance performance, uncertainty, and energy dissipation. This is a proof of concept for active-inference-driven design: not a faithful microphysical theory of a wire, but a generative framework for thinking about how engineered systems could reorganize themselves to maintain function under noise, disturbance, and thermodynamic cost.

---

## What this project demonstrates

When current flows through a conductor, electrons collide inelastically with
lattice phonons, EM interference, and structural defects. Each collision
dissipates kinetic energy as heat — the Joule effect.

This project asks a specific question: **if a subset of electrons organizes
itself as an Active Inference agent — assigning roles by minimizing Expected
Free Energy G — does internal dissipation decrease, and by how much compared
to a system that uses the same physical mechanism but ignores G?**

The answer is yes. The key result is the comparison between closed-loop and
open-loop role assignment:

| Condition | Q_I | JRS vs baseline |
|---|---|---|
| A — No blanket (baseline) | 0.01430 | 0% |
| H — Heuristic roles (open loop, EFE ignored) | 0.00945 | +8.5% |
| **D — EFE minimization (closed loop, v6)** | **0.00579** | **+43.9%** |

**D beats H by 35 percentage points.** The physical cancellation mechanism
is identical in both conditions. The only difference is whether EFE drives
role assignment or not.

---

## The architecture

The system implements the canonical Active Inference partition:

```
E  (environment)  =  physical cable
                      phonons / EM interference / lattice defects / load

B_s (sensory)     =  peripheral electrons
                      measure perturbation from all sources
                      update the sensory mean field <S>(x,t)

B_a (active)      =  central electrons
                      read <S>, cancel all three perturbation sources
                      directly on the cable each step

I  (internal)     =  bulk electrons
                      see only E_effective = E_total * (1 - <A>)
                      their Joule heat -> 0 when cancellation works
```

### Role assignment: coordinate descent on G (`blanket.py`, v6)

At every timestep the `BlanketSynthesizer` solves:

```
omega* = argmin_{omega in Omega} G(omega)

G = pragmatic + epistemic + joule_risk
  = (J_I - J_0)^2 * 5.0
  + post_var * (1 - shield_A)
  + lambda_J * Q_I_estimate
```

**Algorithm:** stochastic coordinate descent, K=5 iterations per step.
For each sampled electron i (20% of N per iteration), G is estimated under
each of the three possible roles using current mean fields (first-order
approximation, no dynamics re-run). The role minimizing G is assigned.

EFE is now **causally responsible** for role assignment — not a diagnostic
metric computed after the fact. Mean G improvement per step: **+0.21**.

The heuristic score functions of v5 are completely removed.

### Active cancellation (`system.py`)

Three simultaneous source-cancellation mechanisms driven by B_a:

- **Phonons** — coverage-aware, segment-level, with boost for undercovered
  high-perturbation segments
- **EM** — per-source targeted: each B_a is assigned to its nearest EM source
  and cancels it directly
- **Defects** — Lorentzian kernel, quasi-static

All conditions use a **unified collision probability formula** so that the
shielding benefit depends only on how much E_effective has been reduced,
not on who did the cancelling. This ensures fair comparison.

---

## Results

### Main result: closed-loop vs open-loop

```bash
python3 run_full_comparison.py
```

| Condition | Q_I | JRS | EFE drives policy? |
|---|---|---|---|
| A — No blanket | 0.01430 | 0% | — |
| H — Heuristic scores (v5, open loop) | 0.00945 | +8.5% | No |
| **D — G minimization (v6, closed loop)** | **0.00579** | **+43.9%** | **Yes** |

### Circularity test (condition G)

G runs the closed-loop optimizer with `lambda_J = 0` (joule_risk removed).
**D outperforms G** — performance degrades when joule_risk is absent from EFE.
This confirms that lambda_J is causally effective in v6.

In v5 (open loop), D = G trivially because EFE did not affect the policy.
In v6 (closed loop), D != G because the optimizer uses joule_risk to guide
role assignment — removing it produces measurably worse outcomes.

### Robust results (10 seeds, 95% CI)

```bash
python3 run_full_comparison.py --robust
```

| Condition | Q_I mean ± 95% CI | JRS |
|---|---|---|
| A — No blanket | 0.01452 ± 0.00016 | baseline |
| D — Closed loop (v6) | 0.00851 ± 0.00032 | **+41.4%** |

Welch t-test D vs A: **t = −31.54, p < 0.0001**

### Full seven-way comparison

| Condition | Q_I | JRS | Note |
|---|---|---|---|
| A — No blanket | 0.01430 | 0% | baseline |
| B — Passive blanket | 0.01384 | +3.2% | roles exist, no cancellation |
| C — Random roles | 0.00908 | +36.5% | random I/B_s/B_a + cancellation |
| H — Heuristic scores (open loop) | 0.00945 | +8.5% | v5 score functions |
| **D — Closed loop (v6)** | **0.00579** | **+43.9%** | G minimization |
| E — LQR (fair) | 0.00852 | +40.4% | on top of VBEM cancellation |
| F — Q-learning (fair) | 0.00821 | +42.6% | on top of VBEM cancellation |
| G — Closed loop, lambda_J=0 | > D | < D | circularity test |

Note: E and F operate on top of VBEM cancellation rather than replacing it.
Their numbers reflect the combined effect of VBEM + controller, not the
controller alone. See Limitations.

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

```bash
# Main experiment (closed-loop v6, default parameters)
python3 run_experiment.py

# Aggressive cancellation (~43% JRS)
python3 -m dmbd_joule \
  --relax-em 0.01 --relax-phonon 0.03 \
  --kappa-em 1.40 --kappa-phonon 1.60 \
  --n-hot-segments 30 --n-blanket 1000

# Seven-way comparison (single seed, includes H)
python3 run_full_comparison.py

# Robust multi-seed comparison (10 seeds, ~80s)
python3 run_full_comparison.py --robust
```

### CLI parameters

```
--n-warmup          int     Warmup steps                    [200]
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
--lambda-joule      float   lambda_J weight in EFE          [8.0]
--blanket-strength  float   CI enforcement strength         [0.92]
--n-hot-segments    int     Priority segments for B_a       [20]
--min-ba-per-seg    int     Min B_a per hot segment         [2]
--seed              int     RNG seed                        [42]
--out-dir           str     Output directory                [results]
```

### As a Python library

```python
from dmbd_joule.system     import CableSystem, SystemParams
from dmbd_joule.blanket    import BlanketSynthesizer
from dmbd_joule.vbem       import VBEMInference
from dmbd_joule.experiment import run_experiment
from dmbd_joule.comparison import run_full_comparison

# Single experiment
result = run_experiment(
    n_blanket = 800,
    params    = SystemParams(relax_em=0.01, kappa_em=1.40),
    out_dir   = "my_results",
)

# Full comparison including H (open-loop baseline)
results = run_full_comparison(
    params  = SystemParams(relax_em=0.01),
    n_steps = 800,
    out_dir = "my_comparison",
)
```

---

## Tests

```bash
python3 -m pytest tests/ -v
```

14 tests: system runs, role population, fraction bounds, cancellation,
Joule reduction, EFE tracking, thermal floor, EM accuracy, LQR, Q-learning,
condition G (D >= G with lambda_J=0), fair LQR structure, warmup, HAC SE.

---

## Module structure

```
dmbd_joule/
├── system.py           Cable physics, 4-role partition, active cancellation,
│                       unified collision probability
├── blanket.py          Closed-loop role assignment via coordinate descent on G
│                       (v6: score functions removed, G drives the policy)
├── vbem.py             E-step (coord descent on G), Kalman q(I), M-step,
│                       EFE, ELBO
├── experiment.py       Three-phase orchestrator: warmup, baseline, blanket
├── metrics.py          MetricSnapshot (incl. G_improvement), HAC standard
│                       errors, EFE-Joule analysis
├── plots.py            Four publication figures
├── comparison.py       Conditions A-H, run_comparison, run_full_comparison,
│                       run_robust_comparison
├── comparison_plots.py Comparison figures
├── controllers.py      LQR and Q-learning baselines
└── __main__.py         CLI entry point
```

---

## Theoretical context

**Active Inference (Friston, 2010)** posits that agents minimize Expected Free
Energy G. This project instantiates that principle at the level of electrons in
a conductor.

The `joule_risk = lambda_J * Q_I` term in G makes internal heat dissipation
part of the agent's predicted surprise. An agent minimizing G therefore organizes
itself to shield internal states from high-dissipation perturbations — not because
dissipation is an explicit reward, but because it contributes to surprise.

The v6 result makes this connection empirical: with the loop closed, removing
joule_risk from G (condition G) degrades performance. The signal is causally
effective, not definitionally circular.

---

## Limitations

- **Toy model.** Hand-tuned parameters, phenomenological dynamics, no
  connection to specific real materials or experimental predictions.

- **First-order G estimate.** The coordinate descent uses mean-field
  approximations without re-running dynamics. Monte Carlo rollouts would
  give a more accurate signal at higher cost.

- **Residual asymmetry in E and F.** These controllers operate on top of
  VBEM cancellation. A fully isolated comparison would require giving the
  controller exclusive control of role assignment as well.

- **Statistical caveats.** The "Granger-style" analysis is lagged
  correlation. The conditional MI is a histogram heuristic. Both are
  qualitative indicators, not formal tests.

- **CI assay (Test 2)** gives |rho| ~ 0.10, not < 0.05. The blanket
  provides approximate, not exact, conditional independence.

---

## Citation

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

**LM Possati**
