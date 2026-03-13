"""metrics.py  —  dmbd_joule v4"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import List, Tuple
from .system import CableSystem, Role


@dataclass
class MetricSnapshot:
    t: float
    # Joule by role
    joule_I:     float
    joule_Bs:    float
    joule_Ba:    float
    joule_total: float
    joule_I_net:  float   # Q_I above thermal floor
    Q_I_floor:   float   # irreducible thermal floor
    em_track_acc: float  # per-source EM tracking accuracy
    # Current
    mean_current_I: float
    current_var_I:  float
    # Blanket structure
    n_I: int; n_Bs: int; n_Ba: int
    n_switched: int
    # Cancellation — full breakdown
    cancel_phonon: float
    cancel_em:     float
    cancel_defect: float
    cancel_total:  float
    shield_A:      float
    E_effective:   float
    # Information
    MI_IE:       float
    cMI_IEB:     float
    partial_corr: float
    # Inference — EFE with Joule coupling
    free_energy: float
    EFE:         float
    pragmatic:   float
    epistemic:   float
    joule_risk:  float   # λ_J · Q_I in EFE
    ELBO:        float
    lambda1: float; lambda2: float; lambda3: float
    G_improvement: float = 0.0   # v6: G reduction achieved by coord descent


class InformationMetrics:

    def __init__(self, n_bins: int = 12):
        self.n_bins = n_bins

    def _MI(self, x, y):
        if len(x) < 6: return 0.0
        H, _, _ = np.histogram2d(x, y, bins=self.n_bins)
        H /= H.sum() + 1e-10
        Hx = H.sum(axis=1); Hy = H.sum(axis=0)
        outer = np.outer(Hx, Hy)
        mask = (H > 0) & (outer > 0)
        return max(0.0, float((H[mask] * np.log2(H[mask] / outer[mask])).sum()))

    def compute(self, sys: CableSystem) -> Tuple[float, float, float]:
        e  = sys.state.electrons; mf = sys.state.mf
        ns = sys.p.n_segments;    t  = sys.state.t
        is_I = (e.role == Role.INTERNAL)
        if is_I.sum() < 6: return 0.0, 0.0, 0.0

        seg_I  = np.clip(e.x[is_I].astype(int), 0, ns - 1)
        I_vals = e.vx[is_I]
        E_vals = mf.S_total[seg_I]      # sensory signal = measured perturbation
        B_vals = mf.A_response[seg_I]   # active response = cancellation

        mi_IE = self._MI(I_vals, E_vals)
        mi_IB = self._MI(I_vals, B_vals)
        mi_EB = self._MI(E_vals, B_vals)
        cmi   = max(0.0, mi_IE - 0.65 * (mi_IB + mi_EB) / 2.0)

        def safe_r(a, b):
            if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        rIE = safe_r(I_vals, E_vals)
        rIB = safe_r(I_vals, B_vals)
        rEB = safe_r(E_vals, B_vals)
        denom = np.sqrt(max((1-rIB**2)*(1-rEB**2), 1e-18))
        return mi_IE, cmi, float((rIE - rIB*rEB)/denom)

    def snapshot(self, sys: CableSystem,
                 vbem_out: dict, synth_out: dict) -> MetricSnapshot:
        mi_IE, cmi, pc = self.compute(sys)
        e   = sys.state.electrons
        is_I = (e.role == Role.INTERNAL)
        j   = sys.joule_by_role()
        cs  = sys.cancellation_by_source()
        return MetricSnapshot(
            t              = sys.state.t,
            joule_I        = j["joule_I"],
            joule_Bs       = j["joule_Bs"],
            joule_Ba       = j["joule_Ba"],
            joule_total    = j["joule_total"],
            joule_I_net    = j.get("joule_I_net", 0.0),
            Q_I_floor      = j.get("Q_I_floor",   0.0),
            em_track_acc   = sys.em_tracking_accuracy(),
            mean_current_I = sys.mean_current_I(),
            current_var_I  = float(e.vx[is_I].var()) if is_I.any() else 0.0,
            n_I            = synth_out.get("n_I", 0),
            n_Bs           = synth_out.get("n_Bs", 0),
            n_Ba           = synth_out.get("n_Ba", 0),
            n_switched     = synth_out.get("n_switched", 0),
            cancel_phonon  = cs["cancel_phonon"],
            cancel_em      = cs["cancel_em"],
            cancel_defect  = cs["cancel_defect"],
            cancel_total   = cs["cancel_total"],
            shield_A       = sys.shield_A(),
            E_effective    = sys.E_effective_mean(),
            MI_IE          = mi_IE,
            cMI_IEB        = cmi,
            partial_corr   = pc,
            free_energy    = vbem_out.get("free_energy", 0.0),
            EFE            = vbem_out.get("EFE", 0.0),
            pragmatic      = vbem_out.get("pragmatic", 0.0),
            epistemic      = vbem_out.get("epistemic", 0.0),
            joule_risk     = vbem_out.get("joule_risk", 0.0),
            ELBO           = vbem_out.get("ELBO", 0.0),
            lambda1        = synth_out.get("lambda1", 0.0),
            lambda2        = synth_out.get("lambda2", 0.0),
            lambda3        = synth_out.get("lambda3", 0.0),
            G_improvement  = vbem_out.get("G_improvement", 0.0),
        )


# ─────────────────────────────────────────────────────────────────────
# EFE — Joule Analysis
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EFEJouleAnalysis:
    """
    Statistical analysis of the EFE — Joule relationship.

    Tests whether EFE is a causal predictor of Joule_I reduction,
    not just a correlated quantity.
    """
    # Correlation analysis
    corr_EFE_JouleI:       float   # Pearson r(EFE, Q_I)
    corr_pragmatic_JouleI: float   # r(pragmatic, Q_I)
    corr_epistemic_JouleI: float   # r(epistemic, Q_I)
    corr_joulerick_JouleI: float   # r(joule_risk, Q_I) — should be ~1
    corr_EFE_cancel:       float   # r(EFE, cancellation) — does high EFE drive more cancel?

    # Granger-style lag analysis: does EFE(t) predict Q_I(t+k)?
    granger_lag1:  float   # corr(EFE(t), Q_I(t+1))
    granger_lag5:  float   # corr(EFE(t), Q_I(t+5))
    granger_lag10: float   # corr(EFE(t), Q_I(t+10))

    # Regression: Q_I = β₀ + β₁·EFE + β₂·cancel + ε
    beta_EFE:    float
    beta_cancel: float
    r_squared:   float

    # EFE decomposition at convergence
    mean_EFE:       float
    mean_pragmatic: float
    mean_epistemic: float
    mean_joule_risk: float
    joule_frac_EFE: float   # joule_risk / EFE — fraction of EFE driven by Joule

    def summary(self) -> str:
        return (
            f"\n{'='*64}\n"
            f"  EFE — JOULE REDUCTION ANALYSIS\n"
            f"{'='*64}\n"
            f"  Correlation analysis:\n"
            f"    r(EFE,       Q_I) = {self.corr_EFE_JouleI:+.4f}\n"
            f"    r(pragmatic, Q_I) = {self.corr_pragmatic_JouleI:+.4f}\n"
            f"    r(epistemic, Q_I) = {self.corr_epistemic_JouleI:+.4f}\n"
            f"    r(joule_risk,Q_I) = {self.corr_joulerick_JouleI:+.4f}  ← λ_J·Q_I term\n"
            f"    r(EFE, cancel)    = {self.corr_EFE_cancel:+.4f}\n"
            f"\n  Granger-style lag predictability:\n"
            f"    r(EFE(t), Q_I(t+1))  = {self.granger_lag1:+.4f}\n"
            f"    r(EFE(t), Q_I(t+5))  = {self.granger_lag5:+.4f}\n"
            f"    r(EFE(t), Q_I(t+10)) = {self.granger_lag10:+.4f}\n"
            f"\n  Linear regression  Q_I = β₀ + β·EFE + β·cancel:\n"
            f"    β_EFE    = {self.beta_EFE:.6f}\n"
            f"    β_cancel = {self.beta_cancel:.6f}\n"
            f"    R²       = {self.r_squared:.4f}\n"
            f"\n  EFE decomposition at convergence:\n"
            f"    G total      = {self.mean_EFE:.5f}\n"
            f"    pragmatic    = {self.mean_pragmatic:.5f}  ({100*self.mean_pragmatic/max(self.mean_EFE,1e-10):.1f}%)\n"
            f"    epistemic    = {self.mean_epistemic:.5f}  ({100*self.mean_epistemic/max(self.mean_EFE,1e-10):.1f}%)\n"
            f"    joule_risk   = {self.mean_joule_risk:.5f}  ({100*self.joule_frac_EFE:.1f}%)\n"
            f"{'='*64}"
        )


def analyze_EFE_joule(blanket_snaps: List[MetricSnapshot],
                      burnin: int = 50) -> EFEJouleAnalysis:
    """Compute full EFE — Joule statistical analysis from snapshot history."""
    snaps = blanket_snaps[burnin:]
    n = len(snaps)

    def ts(f): return np.array([getattr(s, f) for s in snaps])

    EFE  = ts("EFE"); QI   = ts("joule_I")
    prag = ts("pragmatic"); epis = ts("epistemic")
    jrsk = ts("joule_risk"); canc = ts("cancel_total")

    def safe_corr(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12: return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    # Granger lags
    def lag_corr(x, y, lag):
        if lag >= len(x): return 0.0
        return safe_corr(x[:-lag], y[lag:])

    # Linear regression Q_I ~ EFE + cancel
    X = np.column_stack([np.ones(n), EFE, canc])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, QI, rcond=None)
        QI_pred = X @ beta
        ss_res  = np.sum((QI - QI_pred)**2)
        ss_tot  = np.sum((QI - QI.mean())**2)
        r2 = 1 - ss_res/(ss_tot + 1e-12)
    except Exception:
        beta = [0, 0, 0]; r2 = 0.0

    mean_EFE = float(EFE.mean())
    return EFEJouleAnalysis(
        corr_EFE_JouleI       = safe_corr(EFE,  QI),
        corr_pragmatic_JouleI = safe_corr(prag, QI),
        corr_epistemic_JouleI = safe_corr(epis, QI),
        corr_joulerick_JouleI = safe_corr(jrsk, QI),
        corr_EFE_cancel       = safe_corr(EFE,  canc),
        granger_lag1          = lag_corr(EFE, QI, 1),
        granger_lag5          = lag_corr(EFE, QI, 5),
        granger_lag10         = lag_corr(EFE, QI, 10),
        beta_EFE    = float(beta[1]),
        beta_cancel = float(beta[2]),
        r_squared   = float(r2),
        mean_EFE        = mean_EFE,
        mean_pragmatic  = float(prag.mean()),
        mean_epistemic  = float(epis.mean()),
        mean_joule_risk = float(jrsk.mean()),
        joule_frac_EFE  = float(jrsk.mean()) / max(mean_EFE, 1e-10),
    )


@dataclass
class AblationResult:
    mi_blanket_mean:  float; mi_blanket_std:  float
    mi_baseline_mean: float; mi_baseline_std: float
    t_stat: float; p_value: float; significant: bool
    pc_blanket: float; pc_baseline: float; ci_passed: bool
    joule_I_baseline: float; joule_I_blanket: float; jrs_percent: float
    cancel_phonon: float; cancel_em: float; cancel_defect: float; cancel_total: float
    E_eff_mean: float
    elbo_blanket: float; elbo_baseline: float; elbo_wins: bool
    mean_n_I: float; mean_n_Bs: float; mean_n_Ba: float
    lambda1: float; lambda2: float; lambda3: float

    def summary(self) -> str:
        s1 = "✓ SIGNIFICANT"  if self.significant  else "✗ not sig."
        s2 = "✓ PASSED"       if self.ci_passed    else "✗ failed"
        s5 = "✓ BLANKET WINS" if self.elbo_wins    else "✗ baseline"
        return (
            f"\n{'='*64}\n"
            f"  ABLATION VALIDATION  —  v4\n"
            f"{'='*64}\n"
            f"  Test 1 – MI(I;E|B) reduced\n"
            f"    MI|B* = {self.mi_blanket_mean:.5f}±{self.mi_blanket_std:.5f}"
            f"  base = {self.mi_baseline_mean:.5f}±{self.mi_baseline_std:.5f}\n"
            f"    t={self.t_stat:.2f}, p={self.p_value:.4f}  → {s1}\n"
            f"\n  Test 2 – CI  |ρ_{{IE·B}}| < 0.05\n"
            f"    ρ|B* = {self.pc_blanket:.4f}   ρ|base = {self.pc_baseline:.4f}  → {s2}\n"
            f"\n  Test 3 – Joule Reduction Score (internals)\n"
            f"    Q_I baseline = {self.joule_I_baseline:.5f}\n"
            f"    Q_I blanket  = {self.joule_I_blanket:.5f}\n"
            f"    JRS = {self.jrs_percent:.1f}%\n"
            f"\n  Test 4 – Complete active cancellation\n"
            f"    phonon cancel = {self.cancel_phonon*100:.1f}%\n"
            f"    EM     cancel = {self.cancel_em*100:.1f}%\n"
            f"    defect cancel = {self.cancel_defect*100:.1f}%\n"
            f"    TOTAL  cancel = {self.cancel_total*100:.1f}%\n"
            f"    E_eff residual = {self.E_eff_mean:.5f}\n"
            f"\n  Test 5 – ELBO\n"
            f"    ELBO(B*) = {self.elbo_blanket:.4f}"
            f"   ELBO(base) = {self.elbo_baseline:.4f}  → {s5}\n"
            f"\n  Role composition at convergence:\n"
            f"    ⟨n_I⟩={self.mean_n_I:.0f}  ⟨n_Bs⟩={self.mean_n_Bs:.0f}"
            f"  ⟨n_Ba⟩={self.mean_n_Ba:.0f}\n"
            f"    λ₁={self.lambda1:.3f}  λ₂={self.lambda2:.3f}  λ₃={self.lambda3:.3f}\n"
            f"{'='*64}"
        )


def run_ablation(blanket_snaps, baseline_snaps, lagrange, alpha=0.05):
    def ts(snaps, f): return np.array([getattr(s, f) for s in snaps])

    mi_b = ts(blanket_snaps,"cMI_IEB"); mi_a = ts(baseline_snaps,"cMI_IEB")
    QI_b = ts(blanket_snaps,"joule_I"); QI_a = ts(baseline_snaps,"joule_I")
    el_b = ts(blanket_snaps,"ELBO");    el_a = ts(baseline_snaps,"ELBO")
    pc_b = ts(blanket_snaps,"partial_corr"); pc_a = ts(baseline_snaps,"partial_corr")
    cs_b = {k: float(ts(blanket_snaps,k).mean())
            for k in ["cancel_phonon","cancel_em","cancel_defect","cancel_total"]}

    t_stat, p_val = stats.ttest_ind(mi_b, mi_a, equal_var=False)
    QI_base = float(QI_a.mean()); QI_bl = float(QI_b.mean())
    jrs = max(0.0, (QI_base - QI_bl)/(QI_base+1e-10)*100)

    return AblationResult(
        mi_blanket_mean  = float(mi_b.mean()),
        mi_blanket_std   = float(mi_b.std()),
        mi_baseline_mean = float(mi_a.mean()),
        mi_baseline_std  = float(mi_a.std()),
        t_stat   = float(t_stat), p_value = float(p_val),
        significant = (p_val < alpha) and (mi_b.mean() < mi_a.mean()),
        pc_blanket  = float(np.abs(pc_b).mean()),
        pc_baseline = float(np.abs(pc_a).mean()),
        ci_passed   = float(np.abs(pc_b).mean()) < 0.05,
        joule_I_baseline = QI_base, joule_I_blanket = QI_bl, jrs_percent = jrs,
        **cs_b,
        E_eff_mean    = float(ts(blanket_snaps,"E_effective").mean()),
        elbo_blanket  = float(el_b.mean()),
        elbo_baseline = float(el_a.mean()),
        elbo_wins     = float(el_b.mean()) > float(el_a.mean()) + 0.05,
        mean_n_I  = float(ts(blanket_snaps,"n_I").mean()),
        mean_n_Bs = float(ts(blanket_snaps,"n_Bs").mean()),
        mean_n_Ba = float(ts(blanket_snaps,"n_Ba").mean()),
        lambda1 = lagrange.get("lambda1_flux", 0.0),
        lambda2 = lagrange.get("lambda2_CI",   0.0),
        lambda3 = lagrange.get("lambda3_joule",0.0),
    )
