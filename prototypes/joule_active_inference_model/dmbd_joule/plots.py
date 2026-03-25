"""plots.py  --  dmbd_joule v4"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from .metrics import MetricSnapshot, AblationResult

# Color palette
BG    = "#05080f"; SURF  = "#0b1422"; BORD  = "#1a2e50"; WHITE = "#e0eeff"
CI    = "#4488ff"; CBS   = "#00ff9d"; CBA   = "#ffcc00"; CE    = "#ff8844"
CHOT  = "#ff4444"; CCOLD = "#00d4ff"; CEFE  = "#cc88ff"; CDIM  = "#3a5070"


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG,   "axes.facecolor":  SURF,
        "axes.edgecolor":   BORD, "axes.labelcolor": WHITE,
        "axes.titlecolor":  WHITE,"xtick.color":     CDIM,
        "ytick.color":      CDIM, "grid.color":      BORD,
        "grid.linewidth":   0.5,  "text.color":      WHITE,
        "font.family":      "monospace", "font.size": 9,
        "axes.titlesize":   10,   "legend.facecolor": SURF,
        "legend.edgecolor": BORD, "legend.fontsize":  8,
        "lines.linewidth":  1.4,  "figure.dpi":       120,
    })


def _ts(snaps, field):
    return np.array([getattr(s, field) for s in snaps])


def _decorate(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, pad=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORD)


# -----------------------------------------------------------------
def plot_timeseries(base, blk, save=None):
    _style()
    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    fig.suptitle("DMBD Joule v5  --  I / B_s / B_a / E=cable",
                 color=WHITE, fontsize=11, y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35,
                           left=0.06, right=0.97, top=0.92, bottom=0.06)
    tl = _ts(blk, "t"); tb = _ts(base, "t")

    # 1. Joule heat by role
    ax = fig.add_subplot(gs[0, 0])
    _decorate(ax, "Joule heat by role", "t", "Q")
    ax.plot(tb, _ts(base, "joule_I"),  color=CHOT, label="Q_I baseline", alpha=0.7)
    ax.plot(tl, _ts(blk,  "joule_I"),  color=CI,   label="Q_I blanket",  lw=2)
    ax.plot(tl, _ts(blk,  "joule_Bs"), color=CBS,  label="Q_Bs", ls="--")
    ax.plot(tl, _ts(blk,  "joule_Ba"), color=CBA,  label="Q_Ba", ls=":")
    if hasattr(blk[0], "Q_I_floor"):
        ax.axhline(_ts(blk,"Q_I_floor").mean(), color=WHITE, lw=1, ls=":",
                   alpha=0.6, label="thermal floor")
    ax.legend()

    # 2. Active cancellation by source
    ax = fig.add_subplot(gs[0, 1])
    _decorate(ax, "Active cancellation by source", "t", "fraction")
    ax.plot(tl, _ts(blk, "cancel_phonon"), color=CCOLD, label="phonon")
    ax.plot(tl, _ts(blk, "cancel_em"),     color=CBA,   label="EM field")
    ax.plot(tl, _ts(blk, "cancel_defect"), color=CBS,   label="defects")
    ax.plot(tl, _ts(blk, "cancel_total"),  color=WHITE, lw=2, ls="--", label="total")
    ax.axhline(1.0, color=CHOT, lw=0.8, ls=":", alpha=0.5, label="100%")
    ax.set_ylim(0, 1.15); ax.legend()

    # 3. Role composition over time
    ax = fig.add_subplot(gs[0, 2])
    _decorate(ax, "Role composition over time", "t", "electron count")
    ax.stackplot(tl,
                 _ts(blk, "n_I"), _ts(blk, "n_Bs"), _ts(blk, "n_Ba"),
                 labels=["I (internal)", "B_s (sensory)", "B_a (active)"],
                 colors=[CI, CBS, CBA], alpha=0.75)
    ax.legend(loc="upper right")

    # 4. Internal current vs target J_0
    ax = fig.add_subplot(gs[1, 0])
    _decorate(ax, "Internal current vs J_0", "t", "J")
    ax.plot(tb, _ts(base, "mean_current_I"), color=CHOT, label="baseline", alpha=0.8)
    ax.plot(tl, _ts(blk,  "mean_current_I"), color=CI,   label="blanket",  lw=2)
    ax.axhline(0.75, color=WHITE, lw=0.8, ls="--", alpha=0.5, label="J_0")
    ax.legend()

    # 5. Conditional MI
    ax = fig.add_subplot(gs[1, 1])
    _decorate(ax, "Conditional MI: I(I;E|B)", "t", "bits")
    ax.plot(tb, _ts(base, "cMI_IEB"), color=CHOT, label="baseline", alpha=0.8)
    ax.plot(tl, _ts(blk,  "cMI_IEB"), color=CBS,  label="blanket",  lw=2)
    ax.axhline(0.05, color=WHITE, lw=0.8, ls="--", alpha=0.5, label="epsilon=0.05")
    ax.legend()

    # 6. Active response field <A>
    ax = fig.add_subplot(gs[1, 2])
    _decorate(ax, "Active response field <A>", "t")
    ax.plot(tl, _ts(blk, "shield_A"), color=CBA, lw=2, label="<A>(x,t) mean")
    ax.fill_between(tl, 0, _ts(blk, "shield_A"), color=CBA, alpha=0.2)
    ax.set_ylabel("<A>", color=CBA); ax.legend()

    # 7. EFE decomposition
    ax = fig.add_subplot(gs[2, 0])
    _decorate(ax, "Expected Free Energy G", "t", "G")
    ax.plot(tl, _ts(blk, "EFE"),        color=CEFE, label="G total")
    ax.plot(tl, _ts(blk, "pragmatic"),  color=CI,   label="pragmatic",  ls="--")
    ax.plot(tl, _ts(blk, "epistemic"),  color=CBS,  label="epistemic",  ls=":")
    ax.plot(tl, _ts(blk, "joule_risk"), color=CHOT, label="joule_risk", ls="-.")
    ax.legend()

    # 8. ELBO
    ax = fig.add_subplot(gs[2, 1])
    _decorate(ax, "ELBO", "t", "L")
    ax.plot(tb, _ts(base, "ELBO"), color=CHOT, label="baseline", alpha=0.8)
    ax.plot(tl, _ts(blk,  "ELBO"), color=CBS,  label="blanket",  lw=2)
    ax.legend()

    # 9. EM tracking accuracy + role switches
    ax = fig.add_subplot(gs[2, 2])
    _decorate(ax, "EM tracking accuracy", "t")
    if hasattr(blk[0], "em_track_acc"):
        ax.plot(tl, _ts(blk,"em_track_acc")*100, color=CBS, lw=2, label="EM track acc %")
        ax.set_ylabel("accuracy %", color=CBS)
    ax2b = ax.twinx()
    ax2b.bar(tl, _ts(blk, "n_switched"), color=CCOLD, alpha=0.4, width=1, label="switches")
    ax2b.set_ylabel("n_switched", color=CCOLD)
    ax.legend(loc="upper left"); ax2b.legend(loc="upper right")

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  -> {save}")
    return fig


# -----------------------------------------------------------------
def plot_ablation(result, base, blk, save=None):
    _style()
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.suptitle("Ablation Validation v5  --  adaptive deployment + per-source EM",
                 color=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36,
                           left=0.07, right=0.97, top=0.91, bottom=0.08)

    def hist_panel(ax, d1, d2, c1, c2, l1, l2, title, xlabel):
        ax.set_facecolor(SURF)
        ax.set_title(title, pad=5); ax.set_xlabel(xlabel)
        ax.hist(d1, bins=20, color=c1, alpha=0.65, label=l1, density=True)
        ax.hist(d2, bins=20, color=c2, alpha=0.65, label=l2, density=True)
        ax.axvline(d1.mean(), color=c1, lw=2, ls="--")
        ax.axvline(d2.mean(), color=c2, lw=2, ls="--")
        ax.legend(); ax.grid(True, alpha=0.3)
        for sp in ax.spines.values(): sp.set_edgecolor(BORD)

    hist_panel(fig.add_subplot(gs[0, 0]),
               _ts(blk, "cMI_IEB"), _ts(base, "cMI_IEB"),
               CBS, CHOT,
               f"blanket  mean={result.mi_blanket_mean:.4f}",
               f"baseline mean={result.mi_baseline_mean:.4f}",
               "Test 1 -- MI(I;E|B)", "bits")

    hist_panel(fig.add_subplot(gs[0, 1]),
               _ts(blk, "joule_I"), _ts(base, "joule_I"),
               CI, CHOT,
               f"blanket  mean={result.joule_I_blanket:.4f}",
               f"baseline mean={result.joule_I_baseline:.4f}",
               "Test 3 -- Internal Joule heat Q_I", "Q_I")
    fig.axes[-1].text(0.97, 0.97, f"JRS = {result.jrs_percent:.1f}%",
                      transform=fig.axes[-1].transAxes,
                      ha="right", va="top", color=CI, fontsize=10, fontweight="bold")

    # Cancellation histogram
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor(SURF)
    ax3.set_title("Test 4 -- Total active cancellation", pad=5)
    ce = _ts(blk, "cancel_total") * 100
    ax3.hist(ce, bins=20, color=CBA, alpha=0.7, density=True, label="total cancel. %")
    ax3.axvline(ce.mean(), color=WHITE, lw=2, ls="--", label=f"mean={ce.mean():.1f}%")
    ax3.set_xlabel("cancellation %"); ax3.legend(); ax3.grid(True, alpha=0.3)
    for sp in ax3.spines.values(): sp.set_edgecolor(BORD)

    # Joule by role
    ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor(SURF)
    ax4.set_title("Joule heat distribution by role", pad=5)
    ax4.hist(_ts(blk, "joule_I"),  bins=20, color=CI,  alpha=0.7, label="I",   density=True)
    ax4.hist(_ts(blk, "joule_Bs"), bins=20, color=CBS, alpha=0.7, label="B_s", density=True)
    ax4.hist(_ts(blk, "joule_Ba"), bins=20, color=CBA, alpha=0.7, label="B_a", density=True)
    ax4.set_xlabel("Q"); ax4.legend(); ax4.grid(True, alpha=0.3)
    for sp in ax4.spines.values(): sp.set_edgecolor(BORD)

    # Mean role composition bar chart
    ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor(SURF)
    ax5.set_title("Mean role composition", pad=5)
    roles  = ["I", "B_s", "B_a"]
    vals   = [result.mean_n_I, result.mean_n_Bs, result.mean_n_Ba]
    colors = [CI, CBS, CBA]
    bars   = ax5.bar(roles, vals, color=colors, alpha=0.8)
    for bar, v in zip(bars, vals):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{v:.0f}", ha="center", color=WHITE, fontsize=9)
    ax5.set_ylabel("electron count"); ax5.grid(True, alpha=0.3, axis="y")
    for sp in ax5.spines.values(): sp.set_edgecolor(BORD)

    # Scorecard
    ax6 = fig.add_subplot(gs[1, 2]); ax6.set_facecolor(SURF); ax6.axis("off")
    ax6.set_title("Scorecard", pad=5)
    items = [
        ("Test 1 -- MI reduced",       result.significant,           f"p={result.p_value:.3f}"),
        ("Test 2 -- CI assay",         result.ci_passed,             f"|rho|={result.pc_blanket:.3f}"),
        ("Test 3 -- JRS",              result.jrs_percent > 5,       f"{result.jrs_percent:.1f}%"),
        ("Test 4 -- total cancel",     result.cancel_total > 0.10,   f"{result.cancel_total*100:.1f}%"),
        ("Test 4 -- EM cancel",        result.cancel_em > 0.30,      f"{result.cancel_em*100:.1f}%"),
        ("Test 5 -- ELBO",             result.elbo_wins,             f"delta={result.elbo_blanket-result.elbo_baseline:.3f}"),
        ("lambda_1/2/3",               True,                         f"{result.lambda1:.2f}/{result.lambda2:.2f}/{result.lambda3:.2f}"),
    ]
    for i, (label, ok, val) in enumerate(items):
        y = 0.93 - i * 0.13
        ax6.text(0.04, y, "+" if ok else "x", color=CI if ok else CHOT,
                 fontsize=12, va="center", transform=ax6.transAxes)
        ax6.text(0.15, y, label, color=WHITE, fontsize=8.5,
                 va="center", transform=ax6.transAxes)
        ax6.text(0.97, y, val, color=CCOLD, fontsize=8.5, ha="right",
                 va="center", transform=ax6.transAxes)

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  -> {save}")
    return fig


# -----------------------------------------------------------------
def plot_physics(blk, sys_final, save=None):
    _style()
    fig = plt.figure(figsize=(15, 5), facecolor=BG)
    fig.suptitle("Physical fields at convergence  --  v5", color=WHITE)
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38,
                           left=0.05, right=0.97, top=0.88, bottom=0.12)

    s   = sys_final.state
    seg = np.arange(sys_final.p.n_segments)

    # Phonons: natural vs cancelled
    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor(SURF)
    ax1.fill_between(seg, 0, s.cable.phonon_amplitude0, color=CHOT, alpha=0.4,
                     label="natural phonon")
    ax1.fill_between(seg, 0, s.cable.phonon_amplitude,  color=CBA,  alpha=0.6,
                     label="phonon after cancellation")
    ax1.plot(seg, s.cancellation_total * 10, color=WHITE, lw=1, ls="--",
             label="cancel. field x10")
    ax1.set_title("Phonons: natural vs cancelled"); ax1.set_xlabel("x")
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.2)
    for sp in ax1.spines.values(): sp.set_edgecolor(BORD)

    # Sensory and active mean fields
    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor(SURF)
    ax2.plot(seg, s.mf.S_total,    color=CBS, lw=1.5, label="<S> total")
    ax2.plot(seg, s.mf.A_response, color=CBA, lw=1.5, label="<A> active response")
    ax2.fill_between(seg, 0, s.mf.E_effective, color=CHOT, alpha=0.4,
                     label="E_eff (residual)")
    ax2.set_title("Mean fields B_s / B_a"); ax2.set_xlabel("x")
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.2)
    for sp in ax2.spines.values(): sp.set_edgecolor(BORD)

    # Current by role
    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor(SURF)
    ax3.plot(seg, s.current_I_field,  color=CI,  lw=1.5, label="J_I")
    ax3.plot(seg, s.current_Bs_field, color=CBS, lw=1.5, label="J_Bs", ls="--")
    ax3.plot(seg, s.current_Ba_field, color=CBA, lw=1.5, label="J_Ba", ls=":")
    ax3.axhline(0.75, color=WHITE, lw=0.8, ls=":", alpha=0.5, label="J_0")
    ax3.set_title("Current by role"); ax3.set_xlabel("x")
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.2)
    for sp in ax3.spines.values(): sp.set_edgecolor(BORD)

    # Electron density by role
    ax4 = fig.add_subplot(gs[3]); ax4.set_facecolor(SURF)
    from .system import Role
    e     = s.electrons
    seg_e = np.clip(e.x.astype(int), 0, sys_final.p.n_segments - 1)
    n_I   = np.bincount(seg_e[e.role == Role.INTERNAL], minlength=sys_final.p.n_segments)
    n_Bs  = np.bincount(seg_e[e.role == Role.SENSORY],  minlength=sys_final.p.n_segments)
    n_Ba  = np.bincount(seg_e[e.role == Role.ACTIVE],   minlength=sys_final.p.n_segments)
    ax4.bar(seg, n_I,              color=CI,  alpha=0.7, label="I",   width=1)
    ax4.bar(seg, n_Bs, bottom=n_I,            color=CBS, alpha=0.7, label="B_s", width=1)
    ax4.bar(seg, n_Ba, bottom=n_I + n_Bs,     color=CBA, alpha=0.7, label="B_a", width=1)
    ax4.set_title("Electron density per segment"); ax4.set_xlabel("x")
    ax4.legend(fontsize=7); ax4.grid(True, alpha=0.2, axis="y")
    for sp in ax4.spines.values(): sp.set_edgecolor(BORD)

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  -> {save}")
    return fig


# -----------------------------------------------------------------
def plot_efe_joule(blk, efe_analysis, save=None):
    """
    Figure 4 -- EFE / Joule relationship analysis.
    Shows how the three EFE components relate to Q_I over time and the
    Granger-style lag predictability.
    """
    _style()
    fig = plt.figure(figsize=(15, 8), facecolor=BG)
    fig.suptitle("EFE -- Joule Reduction Analysis  (v5)",
                 color=WHITE, fontsize=11)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.36,
                           left=0.07, right=0.97, top=0.91, bottom=0.09)

    tl   = _ts(blk, "t")
    QI   = _ts(blk, "joule_I")
    EFE  = _ts(blk, "EFE")
    prag = _ts(blk, "pragmatic")
    epis = _ts(blk, "epistemic")
    jrsk = _ts(blk, "joule_risk")
    canc = _ts(blk, "cancel_total")

    # 1. EFE decomposition stacked area
    ax = fig.add_subplot(gs[0, 0])
    _decorate(ax, "EFE decomposition over time", "t", "G")
    ax.stackplot(tl, prag, epis, jrsk,
                 labels=["pragmatic", "epistemic", "joule_risk = lambda_J * Q_I"],
                 colors=[CI, CBS, CHOT], alpha=0.75)
    ax.legend(fontsize=7)

    # 2. EFE vs Q_I scatter with regression line
    ax = fig.add_subplot(gs[0, 1])
    _decorate(ax, "EFE vs Q_I", "G", "Q_I")
    ax.scatter(EFE[50:], QI[50:], c=tl[50:], cmap="plasma", s=8, alpha=0.6)
    m, b = np.polyfit(EFE[50:], QI[50:], 1)
    xf   = np.linspace(EFE[50:].min(), EFE[50:].max(), 100)
    ax.plot(xf, m * xf + b, color=WHITE, lw=1.5, ls="--",
            label=f"r = {efe_analysis.corr_EFE_JouleI:+.3f}")
    ax.legend()

    # 3. joule_risk term vs Q_I (should be r=1 by construction)
    ax = fig.add_subplot(gs[0, 2])
    _decorate(ax, "joule_risk (lambda_J * Q_I) vs Q_I", "lambda_J * Q_I", "Q_I")
    ax.scatter(jrsk[50:], QI[50:], c=CBS, s=8, alpha=0.6)
    m2, b2 = np.polyfit(jrsk[50:], QI[50:], 1)
    xf2    = np.linspace(jrsk[50:].min(), jrsk[50:].max(), 100)
    ax.plot(xf2, m2 * xf2 + b2, color=WHITE, lw=1.5, ls="--",
            label=f"r = {efe_analysis.corr_joulerick_JouleI:+.3f}")
    ax.legend()

    # 4. EFE and Q_I dual-axis time series
    ax = fig.add_subplot(gs[1, 0])
    _decorate(ax, "EFE and Q_I over time", "t")
    ax.plot(tl, EFE, color=CEFE, lw=1.5, label="EFE (G)")
    ax.set_ylabel("G", color=CEFE); ax.tick_params(axis="y", colors=CEFE)
    ax2 = ax.twinx()
    ax2.plot(tl, QI, color=CHOT, lw=1.5, label="Q_I")
    ax2.set_ylabel("Q_I", color=CHOT); ax2.tick_params(axis="y", colors=CHOT)
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    # 5. Granger lag bar chart
    ax = fig.add_subplot(gs[1, 1])
    _decorate(ax, "Granger lag: r(EFE(t), Q_I(t+k))", "lag k", "r")
    lags  = [1, 5, 10]
    rcors = [efe_analysis.granger_lag1,
             efe_analysis.granger_lag5,
             efe_analysis.granger_lag10]
    bar_colors = [CI if r > 0 else CHOT for r in rcors]
    ax.bar([str(k) for k in lags], rcors, color=bar_colors, alpha=0.8)
    ax.axhline(0, color=WHITE, lw=0.8)
    ax.set_ylim(-1, 1)
    for i, r in enumerate(rcors):
        ax.text(i, r + 0.04 * np.sign(r + 0.01), f"{r:+.3f}",
                ha="center", color=WHITE, fontsize=9)

    # 6. Cancellation source breakdown over time
    ax = fig.add_subplot(gs[1, 2])
    _decorate(ax, "Cancellation by source over time", "t", "fraction cancelled")
    ax.plot(tl, _ts(blk, "cancel_phonon"), color=CCOLD, lw=1.5, label="phonon")
    ax.plot(tl, _ts(blk, "cancel_em"),     color=CBA,   lw=1.5, label="EM field")
    ax.plot(tl, _ts(blk, "cancel_defect"), color=CBS,   lw=1.5, label="defects")
    ax.plot(tl, canc, color=WHITE, lw=2, ls="--", label="total")
    ax.axhline(1.0, color=CHOT, lw=0.8, ls=":", alpha=0.5, label="100%")
    ax.set_ylim(0, 1.15); ax.legend(fontsize=7)

    if save:
        os.makedirs(os.path.dirname(save) or ".", exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  -> {save}")
    return fig


# -----------------------------------------------------------------
def plot_all(base, blk, result, sys_final, efe_analysis, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plot_timeseries(base, blk,                 f"{out_dir}/fig1_timeseries.png")
    plot_ablation(result, base, blk,           f"{out_dir}/fig2_ablation.png")
    plot_physics(blk, sys_final,               f"{out_dir}/fig3_physics.png")
    plot_efe_joule(blk, efe_analysis,          f"{out_dir}/fig4_efe_joule.png")
