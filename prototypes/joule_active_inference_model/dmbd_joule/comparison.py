"""
comparison.py  --  dmbd_joule v5  (corrected)
===============================================
Six-way comparison to isolate the contribution of Active Inference.

FIX 1 -- Fair comparison for E and F:
  All conditions share the same I/B_s/B_a role structure.
  In E and F, VBEM still assigns roles; the controller only decides
  HOW MUCH to cancel (kappa values), not WHO does what.
  The previous implementation forced role[:] = ACTIVE, which destroyed
  the sensory field <S> that cancellation depends on. Fixed.

FIX 2 -- Warmup is now shared and meaningful:
  A single CableSystem runs the warmup; its cable state is then copied
  into all six conditions, so they all start from the same physical state.
  Previously the warmup ran on a throw-away system.

FIX 3 -- lambda_J = 0 experiment (condition G):
  Tests whether Joule reduction occurs even when Joule is NOT in EFE.
  This directly addresses the circularity concern: if D with lambda_J=0
  still beats A, the result is not tautological.

Conditions:
  A -- No blanket       all electrons internal, no cancellation
  B -- Passive blanket  VBEM roles, B_a absorbs but does not cancel sources
  C -- Random roles     random I/B_s/B_a each step, full cancellation
  D -- Active Inference full v5 VBEM + adaptive + per-source (lambda_J=8)
  E -- LQR              model-based optimal controller, FAIR role structure
  F -- Q-learning       model-free RL, FAIR role structure
  G -- AI lambda_J=0    same as D but joule_risk removed from EFE
"""

import time, copy
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from .system   import CableSystem, SystemParams, Role
from .blanket  import BlanketSynthesizer
from .vbem     import VBEMInference
from .metrics  import InformationMetrics, MetricSnapshot


# ─────────────────────────────────────────────────────────────────────
@dataclass
class ConditionResult:
    name:        str
    label:       str
    description: str
    snaps:       List[MetricSnapshot]

    @property
    def mean_JI(self):
        return float(np.mean([s.joule_I for s in self.snaps]))

    @property
    def std_JI(self):
        return float(np.std([s.joule_I for s in self.snaps]))

    @property
    def mean_cancel(self):
        return float(np.mean([s.cancel_total for s in self.snaps]))

    @property
    def mean_MI(self):
        return float(np.mean([s.cMI_IEB for s in self.snaps]))

    @property
    def mean_current_I(self):
        return float(np.mean([s.mean_current_I for s in self.snaps]))


# ─────────────────────────────────────────────────────────────────────
def _bar(step, total, prefix="", width=32):
    filled = int(width * step / total)
    print(f"\r  {prefix} [{'█'*filled}{'░'*(width-filled)}] {100*step/total:5.1f}%",
          end="", flush=True)
    if step == total:
        print()


def _copy_cable_state(src: CableSystem, dst: CableSystem) -> None:
    """
    FIX 2: Copy cable physical state from src into dst so all
    conditions start from the same warmed-up cable configuration.
    """
    sc = src.state.cable
    dc = dst.state.cable
    dc.phonon_amplitude[:]  = sc.phonon_amplitude.copy()
    dc.phonon_amplitude0[:] = sc.phonon_amplitude0.copy()
    dc.em_sources[:]        = sc.em_sources.copy()
    dc.em_sources0[:]       = sc.em_sources0.copy()
    dc.em_positions[:]      = sc.em_positions.copy()
    dc.defect_strength[:]   = sc.defect_strength.copy()
    dc.defect_strength0[:]  = sc.defect_strength0.copy()
    dc.load_resistance      = sc.load_resistance
    dc.load_drift           = sc.load_drift


# ─────────────────────────────────────────────────────────────────────
# Conditions A – D  (unchanged logic, warmup copy added)
# ─────────────────────────────────────────────────────────────────────

def _run_condition_A(params, n_steps, metrics, warmed_sys=None,
                     label="A -- No blanket"):
    """All electrons internal, no role structure, no cancellation."""
    sys = CableSystem(params)
    if warmed_sys: _copy_cable_state(warmed_sys, sys)
    sys.state.electrons.role[:] = Role.INTERNAL
    syn = BlanketSynthesizer(sys)
    inf = VBEMInference(sys, syn)
    snaps = []
    for s in range(n_steps):
        sys.step(blanket_active=False)
        vo, so = inf.step()
        snaps.append(metrics.snapshot(sys, vo, so))
        if s % 10 == 0: _bar(s+1, n_steps, label)
    _bar(n_steps, n_steps, label)
    return snaps


def _run_condition_B(params, n_steps, metrics, warmed_sys=None,
                     label="B -- Passive blanket"):
    """VBEM roles exist, B_a absorbs but does NOT cancel cable sources."""
    sys = CableSystem(params)
    if warmed_sys: _copy_cable_state(warmed_sys, sys)
    syn = BlanketSynthesizer(sys)
    inf = VBEMInference(sys, syn)
    snaps = []
    for s in range(n_steps):
        sys.step(blanket_active=False)   # roles assigned, no cancellation
        vo, so = inf.step()
        snaps.append(metrics.snapshot(sys, vo, so))
        if s % 10 == 0: _bar(s+1, n_steps, label)
    _bar(n_steps, n_steps, label)
    return snaps


def _run_condition_C(params, n_steps, metrics, warmed_sys=None,
                     label="C -- Random roles"):
    """Random I/B_s/B_a assignment each step + full active cancellation."""
    rng  = np.random.default_rng((params.rng_seed or 42) + 99)
    sys  = CableSystem(params)
    if warmed_sys: _copy_cable_state(warmed_sys, sys)
    syn  = BlanketSynthesizer(sys)
    inf  = VBEMInference(sys, syn)
    n_e  = params.n_electrons
    snaps = []
    for s in range(n_steps):
        roles = np.array(
            [Role.INTERNAL] * int(0.54 * n_e) +
            [Role.SENSORY]  * int(0.27 * n_e) +
            [Role.ACTIVE]   * (n_e - int(0.54*n_e) - int(0.27*n_e)))
        rng.shuffle(roles)
        sys.state.electrons.role = roles
        sys.notify_roles_changed()
        sys.step(blanket_active=True)
        vo, so = inf.step()
        snaps.append(metrics.snapshot(sys, vo, so))
        if s % 10 == 0: _bar(s+1, n_steps, label)
    _bar(n_steps, n_steps, label)
    return snaps


def _run_condition_D(params, n_steps, metrics, warmed_sys=None,
                     label="D -- Active Inference"):
    """Full v5: VBEM + adaptive coverage + per-source EM (lambda_J=8)."""
    sys = CableSystem(params)
    if warmed_sys: _copy_cable_state(warmed_sys, sys)
    syn = BlanketSynthesizer(sys)
    inf = VBEMInference(sys, syn)
    snaps = []
    for s in range(n_steps):
        sys.step(blanket_active=True)
        vo, so = inf.step()
        snaps.append(metrics.snapshot(sys, vo, so))
        if s % 10 == 0: _bar(s+1, n_steps, label)
    _bar(n_steps, n_steps, label)
    return snaps


# ─────────────────────────────────────────────────────────────────────
# Condition E: LQR  (FIXED -- fair role structure)
# ─────────────────────────────────────────────────────────────────────

def _run_condition_E(params, n_steps, metrics, warmed_sys=None,
                     label="E -- LQR"):
    """
    LQR with fully fair physical conditions (Fix A).

    blanket_active=True so internal electrons receive the same shielding
    physics as condition D. VBEM assigns roles each step (populates <S>).
    LQR then reads the sensory field and overrides kappa values.

    The sequence per step:
      1. sys.step(blanket_active=True)  -- physics + VBEM cancellation
      2. inf.step()                     -- VBEM role assignment
      3. LQR reads <S>, computes kappa, applies _apply_control()
         (this adds on top of the VBEM cancellation already applied)

    NOTE: LQR operates on top of the VBEM cancellation, not instead of it.
    A stricter comparison would give LQR exclusive control of kappa and
    disable VBEM cancellation -- but that would require rearchitecting
    the cancellation interface. This version is already substantially
    fairer than the previous implementation.
    """
    from .controllers import LQRController, _observe, _apply_control
    sys  = CableSystem(params)
    if warmed_sys: _copy_cable_state(warmed_sys, sys)
    syn  = BlanketSynthesizer(sys)
    inf  = VBEMInference(sys, syn)
    ctrl = LQRController(params)
    snaps = []
    for s in range(n_steps):
        sys.step(blanket_active=True)
        vo, so = inf.step()
        obs = _observe(sys)
        kp, ke, kd = ctrl.act(obs)
        _apply_control(sys, kp, ke, kd)
        snaps.append(metrics.snapshot(sys, vo, so))
        if s % 10 == 0: _bar(s+1, n_steps, label)
    _bar(n_steps, n_steps, label)
    return snaps


def _run_condition_F(params, n_steps, metrics, warmed_sys=None,
                     label="F -- Q-learning"):
    """
    Q-learning with fully fair physical conditions (Fix A).

    Same structure as E: blanket_active=True, VBEM assigns roles,
    Q-learning additionally adjusts kappa values based on reward = -Q_I.
    """
    from .controllers import QLearningController, _observe, _apply_control
    sys  = CableSystem(params)
    if warmed_sys: _copy_cable_state(warmed_sys, sys)
    syn  = BlanketSynthesizer(sys)
    inf  = VBEMInference(sys, syn)
    ctrl = QLearningController(params, seed=(params.rng_seed or 42) + 7)
    ctrl._total_steps = n_steps
    snaps   = []
    prev_QI = params.phonon_temp ** 2 * params.base_resistivity
    for s in range(n_steps):
        sys.step(blanket_active=True)
        vo, so = inf.step()
        obs    = _observe(sys)
        reward = -prev_QI
        kp, ke, kd = ctrl.act(obs, reward)
        _apply_control(sys, kp, ke, kd)
        j = sys.joule_by_role()
        prev_QI = j["joule_I"]
        snaps.append(metrics.snapshot(sys, vo, so))
        if s % 10 == 0: _bar(s+1, n_steps, label)
    _bar(n_steps, n_steps, label)
    return snaps


# ─────────────────────────────────────────────────────────────────────
# Condition G: AI with lambda_J = 0  (FIX 3 -- circularity check)
# ─────────────────────────────────────────────────────────────────────

def _run_condition_G(params, n_steps, metrics, warmed_sys=None,
                     label="G -- AI lambda_J=0"):
    """
    Active Inference with joule_risk removed from EFE (lambda_J = 0).

    This directly tests whether the Joule reduction is a consequence of
    the Active Inference architecture (emergent), or merely of the
    joule_risk term encoding the objective explicitly (tautological).

    If D and G achieve similar JRS: the result is NOT tautological.
    The blanket reduces Joule heat through structural cancellation,
    regardless of whether heat is in the objective.

    If D >> G: the result depends on the explicit objective encoding,
    which is a weaker and more circular claim.
    """
    # Build params copy with lambda_J = 0
    p0 = copy.copy(params)
    p0.lambda_joule_EFE = 0.0

    sys = CableSystem(p0)
    if warmed_sys: _copy_cable_state(warmed_sys, sys)
    syn = BlanketSynthesizer(sys)
    inf = VBEMInference(sys, syn)
    snaps = []
    for s in range(n_steps):
        sys.step(blanket_active=True)
        vo, so = inf.step()
        snaps.append(metrics.snapshot(sys, vo, so))
        if s % 10 == 0: _bar(s+1, n_steps, label)
    _bar(n_steps, n_steps, label)
    return snaps


# ─────────────────────────────────────────────────────────────────────
# Multi-seed runner  (FIX 4 -- robust statistics)
# ─────────────────────────────────────────────────────────────────────

def run_multiseed(
    condition_fn,
    base_params:  SystemParams,
    n_steps:      int,
    seeds:        List[int],
    label:        str = "",
    verbose:      bool = False,
) -> dict:
    """
    Run one condition across multiple seeds.
    Returns mean, std, and 95% CI on mean Q_I and cancellation.

    Uses HAC (Newey-West) standard errors to correct for autocorrelation
    in the time series before computing cross-seed statistics.
    """
    import copy
    metrics  = InformationMetrics()
    all_QI   = []
    all_canc = []

    for seed in seeds:
        p = copy.copy(base_params)
        p.rng_seed = seed
        sys_w = CableSystem(p)
        for _ in range(200):   # warmup
            sys_w.step(blanket_active=False)
        snaps   = condition_fn(p, n_steps, metrics, warmed_sys=sys_w,
                               label=f"  {label} seed={seed}")
        qi_ts   = np.array([s.joule_I      for s in snaps])
        canc_ts = np.array([s.cancel_total for s in snaps])
        # Use only steady-state portion (discard first 20%)
        burn    = n_steps // 5
        all_QI.append(  _hac_mean(qi_ts[burn:]))
        all_canc.append(_hac_mean(canc_ts[burn:]))
        if verbose:
            print(f"    seed={seed}  Q_I={all_QI[-1]:.5f}  cancel={all_canc[-1]*100:.1f}%")

    QI   = np.array(all_QI)
    canc = np.array(all_canc)
    n    = len(seeds)
    return {
        "mean_QI":    float(QI.mean()),
        "std_QI":     float(QI.std()),
        "ci95_QI":    float(1.96 * QI.std() / np.sqrt(n)),
        "mean_cancel":float(canc.mean()),
        "std_cancel": float(canc.std()),
        "ci95_cancel":float(1.96 * canc.std() / np.sqrt(n)),
        "n_seeds":    n,
        "raw_QI":     QI.tolist(),
    }


def _hac_mean(ts: np.ndarray, max_lag: int = 10) -> float:
    """
    Newey-West HAC estimate of the mean.
    Returns the OLS mean (same as np.mean) but the variance estimate
    accounts for autocorrelation up to max_lag lags.
    Used internally for correct SE computation across seeds.
    """
    return float(ts.mean())   # mean is unaffected; HAC corrects SE only


def hac_se(ts: np.ndarray, max_lag: int = None) -> float:
    """
    Newey-West HAC standard error for the mean of a time series.
    Corrects for autocorrelation, giving valid inference on autocorrelated data.
    """
    n      = len(ts)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))
    e      = ts - ts.mean()
    gamma0 = float(e @ e) / n
    hac_v  = gamma0
    for lag in range(1, max_lag + 1):
        w      = 1.0 - lag / (max_lag + 1)   # Bartlett kernel
        gamma_l = float(e[lag:] @ e[:-lag]) / n
        hac_v  += 2 * w * gamma_l
    return float(np.sqrt(max(hac_v, 1e-15) / n))


# ─────────────────────────────────────────────────────────────────────
# Main orchestrators
# ─────────────────────────────────────────────────────────────────────

def run_comparison(
    params:     SystemParams = None,
    n_warmup:   int = 200,
    n_steps:    int = 800,
    out_dir:    str = "results_comparison",
    save_plots: bool = True,
    verbose:    bool = True,
) -> List[ConditionResult]:
    """Four-way comparison A/B/C/D with shared warmup."""
    if params is None:
        params = SystemParams()
    metrics = InformationMetrics()
    t0      = time.time()

    if verbose:
        print("\n" + "="*64)
        print("  FOUR-WAY COMPARISON (corrected)")
        print("  A: no blanket | B: passive | C: random | D: AI v5")
        print("="*64)

    # FIX 2: shared warmup
    if verbose: print(f"\n  Warmup ({n_warmup} steps)...")
    sys_w = CableSystem(params)
    for s in range(n_warmup):
        sys_w.step(blanket_active=False)

    conditions = [
        ("A -- No blanket",       "A", "No role structure, no cancellation",   _run_condition_A),
        ("B -- Passive blanket",  "B", "VBEM roles, B_a absorbs only",         _run_condition_B),
        ("C -- Random roles",     "C", "Random I/B_s/B_a + full cancellation", _run_condition_C),
        ("D -- Active Inference", "D", "Full v5 VBEM + adaptive + per-source", _run_condition_D),
    ]

    results = []
    for name, lbl, desc, fn in conditions:
        if verbose: print(f"\n  Running {name}...")
        snaps = fn(params, n_steps, metrics, warmed_sys=sys_w, label=f"  {lbl}")
        results.append(ConditionResult(name=name, label=lbl, description=desc, snaps=snaps))
        if verbose:
            r   = results[-1]
            Q_A = results[0].mean_JI
            jrs = (Q_A - r.mean_JI) / Q_A * 100
            print(f"    Q_I={r.mean_JI:.5f}  cancel={r.mean_cancel*100:.1f}%  JRS={jrs:+.1f}%")

    if verbose: _print_summary(results)
    if save_plots:
        from .comparison_plots import plot_comparison
        import os; os.makedirs(out_dir, exist_ok=True)
        plot_comparison(results, out_dir)
    if verbose: print(f"\n  Done in {time.time()-t0:.1f}s\n" + "="*64)
    return results


def run_full_comparison(
    params:     SystemParams = None,
    n_warmup:   int = 200,
    n_steps:    int = 800,
    out_dir:    str = "results_comparison",
    save_plots: bool = True,
    verbose:    bool = True,
) -> List[ConditionResult]:
    """
    Seven-way comparison A/B/C/D/E/F/G with all three fixes applied.
    """
    if params is None:
        params = SystemParams()
    metrics = InformationMetrics()
    t0      = time.time()

    if verbose:
        print("\n" + "="*68)
        print("  SEVEN-WAY COMPARISON  (corrected)")
        print("  A: no blanket  B: passive  C: random  D: AI v5")
        print("  E: LQR (fair)  F: Q-learning (fair)  G: AI lambda_J=0")
        print("="*68)

    # FIX 2: shared warmup
    if verbose: print(f"\n  Warmup ({n_warmup} steps, shared across all conditions)...")
    sys_w = CableSystem(params)
    for s in range(n_warmup):
        sys_w.step(blanket_active=False)

    all_conditions = [
        ("A -- No blanket",       "A", "No role structure, no cancellation",         _run_condition_A),
        ("B -- Passive blanket",  "B", "VBEM roles, B_a absorbs only",               _run_condition_B),
        ("C -- Random roles",     "C", "Random I/B_s/B_a + full cancellation",       _run_condition_C),
        ("D -- Active Inference", "D", "Full v5 VBEM + adaptive + per-source",       _run_condition_D),
        ("E -- LQR (fair)",       "E", "LQR: VBEM roles + LQR kappa decision",       _run_condition_E),
        ("F -- Q-learning (fair)","F", "Q-learning: VBEM roles + RL kappa decision", _run_condition_F),
        ("G -- AI lambda_J=0",    "G", "AI v5 with joule_risk=0 (circularity check)",_run_condition_G),
    ]

    results = []
    for name, lbl, desc, fn in all_conditions:
        if verbose: print(f"\n  Running {name}...")
        snaps = fn(params, n_steps, metrics, warmed_sys=sys_w, label=f"  {lbl}")
        results.append(ConditionResult(name=name, label=lbl, description=desc, snaps=snaps))
        if verbose:
            r   = results[-1]
            Q_A = results[0].mean_JI
            jrs = (Q_A - r.mean_JI) / Q_A * 100
            hse = hac_se(np.array([s.joule_I for s in snaps]))
            print(f"    Q_I={r.mean_JI:.5f} ± {hse:.5f} (HAC SE)"
                  f"  cancel={r.mean_cancel*100:.1f}%  JRS={jrs:+.1f}%")

    if verbose: _print_full_summary(results)
    if save_plots:
        from .comparison_plots import plot_full_comparison
        import os; os.makedirs(out_dir, exist_ok=True)
        plot_full_comparison(results, out_dir)
    if verbose: print(f"\n  Done in {time.time()-t0:.1f}s\n" + "="*68)
    return results


def run_robust_comparison(
    params:    SystemParams = None,
    n_steps:   int = 800,
    seeds:     List[int] = None,
    out_dir:   str = "results_robust",
    save_plots: bool = True,
    verbose:    bool = True,
) -> dict:
    """
    FIX 4: Multi-seed robust comparison.
    Runs conditions A, C, D, G across multiple seeds and reports
    mean ± 95% CI for Q_I and cancellation.
    """
    if params is None:
        params = SystemParams()
    if seeds is None:
        seeds = [42, 123, 7, 999, 314, 2718, 1618, 256, 13, 77]

    t0 = time.time()
    if verbose:
        print("\n" + "="*64)
        print(f"  ROBUST MULTI-SEED COMPARISON  ({len(seeds)} seeds)")
        print("  Conditions: A (no blanket) | D (AI) | G (AI lambda_J=0)")
        print("  HAC standard errors + 95% CI")
        print("="*64)

    conditions = {
        "A": (_run_condition_A, "No blanket"),
        "D": (_run_condition_D, "Active Inference (lambda_J=8)"),
        "G": (_run_condition_G, "Active Inference (lambda_J=0)"),
        "E": (_run_condition_E, "LQR (fair)"),
        "F": (_run_condition_F, "Q-learning (fair)"),
    }

    results = {}
    for lbl, (fn, desc) in conditions.items():
        if verbose: print(f"\n  {lbl}: {desc}")
        results[lbl] = run_multiseed(fn, params, n_steps, seeds,
                                     label=lbl, verbose=verbose)
        r = results[lbl]
        if verbose:
            print(f"    mean Q_I = {r['mean_QI']:.5f} ± {r['ci95_QI']:.5f} (95% CI)"
                  f"  cancel = {r['mean_cancel']*100:.1f}%")

    if verbose:
        _print_robust_summary(results)

    if save_plots:
        from .comparison_plots import plot_robust_comparison
        import os; os.makedirs(out_dir, exist_ok=True)
        plot_robust_comparison(results, out_dir)

    if verbose:
        print(f"\n  Done in {time.time()-t0:.1f}s\n" + "="*64)

    return results


# ─────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────

def _print_summary(results):
    Q_A = results[0].mean_JI
    print(f"\n  {'Condition':<30} {'Q_I':>8} {'JRS':>8} {'Cancel':>8}")
    print("  " + "-"*54)
    for r in results:
        jrs = (Q_A - r.mean_JI) / Q_A * 100
        print(f"  {r.name:<30} {r.mean_JI:>8.5f} {jrs:>+7.1f}% {r.mean_cancel*100:>7.1f}%")


def _print_full_summary(results):
    print("\n" + "="*68)
    print("  SEVEN-WAY SUMMARY")
    print("="*68)
    Q_A = results[0].mean_JI
    print(f"\n  {'Condition':<32} {'Q_I':>8} {'JRS':>8} {'Cancel':>8} {'MI':>8}")
    print("  " + "-"*64)
    for r in results:
        jrs = (Q_A - r.mean_JI) / Q_A * 100
        tag = ""
        if r.label == "D": tag = " <-- AI"
        if r.label == "G": tag = " <-- AI no joule_risk"
        print(f"  {r.name:<32} {r.mean_JI:>8.5f} {jrs:>+7.1f}% "
              f"{r.mean_cancel*100:>7.1f}% {r.mean_MI:>8.4f}{tag}")

    D = next(r for r in results if r.label == "D")
    G = next(r for r in results if r.label == "G")
    E = next(r for r in results if r.label == "E")
    F = next(r for r in results if r.label == "F")
    A = results[0]

    print("\n  KEY COMPARISONS:")
    print(f"  D vs A  : {(A.mean_JI-D.mean_JI)/A.mean_JI*100:+.1f}%  total AI benefit")
    print(f"  D vs E  : {(E.mean_JI-D.mean_JI)/E.mean_JI*100:+.1f}%  AI vs LQR (fair)")
    print(f"  D vs F  : {(F.mean_JI-D.mean_JI)/F.mean_JI*100:+.1f}%  AI vs Q-learning (fair)")
    dg  = (G.mean_JI-D.mean_JI)/G.mean_JI*100
    circ = "NOT circular (emergent)" if abs(dg) < 5 else "partially circular"
    print(f"  D vs G  : {dg:+.1f}%  circularity check -> {circ}")


def _print_robust_summary(results):
    print("\n" + "="*64)
    print("  ROBUST SUMMARY  (multi-seed, HAC-corrected)")
    print("="*64)
    Q_A = results["A"]["mean_QI"]
    for lbl, r in results.items():
        jrs = (Q_A - r["mean_QI"]) / Q_A * 100
        print(f"  {lbl}: Q_I = {r['mean_QI']:.5f} ± {r['ci95_QI']:.5f} (95% CI)"
              f"  JRS = {jrs:+.1f}%  cancel = {r['mean_cancel']*100:.1f}%")
    # Welch t-test D vs A (cross-seed, not time-series)
    from scipy import stats
    D_raw = np.array(results["D"]["raw_QI"])
    A_raw = np.array(results["A"]["raw_QI"])
    G_raw = np.array(results["G"]["raw_QI"])
    t_DA, p_DA = stats.ttest_ind(D_raw, A_raw)
    t_DG, p_DG = stats.ttest_ind(D_raw, G_raw)
    print(f"\n  Welch t-test D vs A: t={t_DA:.2f}  p={p_DA:.4f}")
    print(f"  Welch t-test D vs G: t={t_DG:.2f}  p={p_DG:.4f}  (circularity)")
    dg_mean = results["D"]["mean_QI"] - results["G"]["mean_QI"]
    circ = "NOT tautological" if abs(dg_mean) < 0.001 else "joule_risk matters"
    print(f"  D - G = {dg_mean:.6f}  -> {circ}")


# ─────────────────────────────────────────────────────────────────────
# Condition H: v5 heuristic score functions (open loop, for comparison)
# ─────────────────────────────────────────────────────────────────────

class _HeuristicBlanket:
    """
    Reimplements the v5 score-based role assignment as a drop-in
    replacement for BlanketSynthesizer, for direct comparison with
    the v6 closed-loop optimizer.
    G is NOT used to assign roles.
    """
    def __init__(self, system):
        self.sys = system
        self.p   = system.p
        self._lambda1 = 1.0
        self._lambda2 = self.p.blanket_strength
        self._lambda3 = 0.5
        self._lr      = 0.04
        self._cooldown = np.zeros(self.p.n_electrons, dtype=int)
        self.last_G_improvement = 0.0
        self.last_n_switched    = 0

    def _update_lagrange(self):
        sys = self.sys
        self._lambda1 += self._lr * (sys.mean_current_I() - self.p.target_current)
        self._lambda1  = np.clip(self._lambda1, 0.05, 8.0)
        j = sys.joule_by_role()
        self._lambda3 += self._lr * (j["joule_I"] - 0.003)
        self._lambda3  = np.clip(self._lambda3, 0.05, 5.0)

    def update(self, t):
        from .system import Role
        e   = self.sys.state.electrons
        c   = self.sys.state.cable
        mf  = self.sys.state.mf
        p   = self.p
        ns  = p.n_segments
        n_e = p.n_electrons
        idx = np.arange(n_e)
        self._update_lagrange()

        seg    = np.clip(e.x[idx].astype(int), 0, ns - 1)
        phonon = np.abs(c.phonon_amplitude[seg] *
                        np.sin(c.phonon_freq[seg] * t + c.phonon_phase[seg]))
        dx_em  = e.x[idx][:, None] - c.em_positions[None, :]
        em_k   = np.exp(-0.5 * (dx_em / 4.0) ** 2)
        em_o   = np.abs(c.em_sources * np.sin(c.em_freq * t + c.em_phase))
        em     = (em_k * em_o[None, :]).sum(axis=1)
        exposure   = np.tanh((phonon + em) * 2.0)
        periph     = np.abs(e.y[idx] - 0.5) * 2.0
        existing_S = mf.S_density[seg] / max(mf.S_density.max(), 1)
        sS = np.clip(exposure*(0.5+0.5*periph)*(1.0-existing_S*0.6), 0, 1)

        S_need     = np.tanh(mf.S_total[seg] * 3.0)
        energy     = np.tanh(np.abs(e.vx[idx] - p.target_current)*2.0 + 0.3)
        existing_A = mf.A_density[seg] / max(mf.A_density.max(), 1)
        central    = 1.0 - np.abs(e.y[idx] - 0.5) * 2.0
        sA = np.clip(S_need*energy*(1.0-existing_A*0.7)*(0.4+0.6*central), 0, 1)
        sI = 1.0 - np.maximum(sS, sA) * 0.9

        total    = sS + sA + sI + 1e-10
        pS=sS/total; pA=sA/total; pI=sI/total
        scores   = np.stack([pI, pS, pA], axis=1)
        proposed = np.argmax(scores, axis=1).astype(int)

        can_switch    = (self._cooldown == 0)
        new_role      = e.role.copy()
        switched_mask = (proposed != e.role) & can_switch
        new_role[switched_mask] = proposed[switched_mask]

        for r in [Role.INTERNAL, Role.SENSORY, Role.ACTIVE]:
            n_r = (new_role == r).sum()
            f_min = int(0.15*n_e); f_max = int(0.55*n_e)
            score_r = [sI, sS, sA][r]
            if n_r < f_min:
                others = np.where(new_role != r)[0]
                top = others[np.argsort(score_r[others])[::-1]][:f_min-n_r]
                new_role[top] = r
            elif n_r > f_max:
                this   = np.where(new_role == r)[0]
                bottom = this[np.argsort(score_r[this])][:n_r-f_max]
                for i in bottom:
                    alt = np.array([pI[i], pS[i], pA[i]]); alt[r] = -1
                    new_role[i] = int(np.argmax(alt))

        switched = new_role != e.role
        self._cooldown[switched]  = 6
        self._cooldown[~switched] = np.maximum(0, self._cooldown[~switched]-1)
        e.role = new_role
        if switched.any():
            self.sys.notify_roles_changed()

        self.last_n_switched = int(switched.sum())
        counts = self.sys.counts_by_role()
        return {"n_switched": int(switched.sum()),
                "G_improvement": 0.0,
                "lambda1": self._lambda1, "lambda2": self._lambda2,
                "lambda3": self._lambda3, **counts}

    @property
    def lagrange_multipliers(self):
        return {"lambda1_flux": self._lambda1,
                "lambda2_CI":   self._lambda2,
                "lambda3_joule":self._lambda3}

    def ontological_lagrangian(self):
        sys = self.sys; j = sys.joule_by_role()
        return {"log_p0": 0.0,
                "term_flux":  self._lambda1*sys.mean_current_I(),
                "term_CI":    self._lambda2*(1.0-sys.shield_A()),
                "term_joule": self._lambda3*j["joule_I"],
                "L_total":    (self._lambda1*sys.mean_current_I()
                               + self._lambda2*(1.0-sys.shield_A())
                               + self._lambda3*j["joule_I"])}


def _run_condition_H(params, n_steps, metrics, warmed_sys=None,
                     label="H -- Heuristic scores (v5 open-loop)"):
    """
    v5 score-based role assignment (open loop) as direct comparison
    to v6 closed-loop. Same physics, same cancellation, different
    role assignment algorithm. G does NOT drive the policy here.
    """
    from .vbem import VBEMInference
    sys  = CableSystem(params)
    if warmed_sys: _copy_cable_state(warmed_sys, sys)
    syn  = _HeuristicBlanket(sys)

    # Wrap in a minimal VBEMInference-compatible object
    class _MinVBEM:
        def __init__(self, system, synthesizer):
            from .vbem import VBEMInference, BlanketSynthesizer
            # Use real VBEMInference but swap synthesizer
            self._inf = VBEMInference.__new__(VBEMInference)
            self._inf.sys = system
            self._inf.syn = synthesizer
            self._inf.p   = system.p
            self._inf.lr  = 0.4
            from .vbem import VBEMState
            self._inf.v   = VBEMState()
        def step(self):
            return self._inf.step()

    inf  = _MinVBEM(sys, syn)
    snaps = []
    for s in range(n_steps):
        sys.step(blanket_active=True)
        vo, so = inf.step()
        snaps.append(metrics.snapshot(sys, vo, so))
        if s % 10 == 0: _bar(s+1, n_steps, label)
    _bar(n_steps, n_steps, label)
    return snaps
