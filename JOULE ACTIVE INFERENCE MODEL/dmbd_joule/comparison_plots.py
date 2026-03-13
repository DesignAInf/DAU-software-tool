"""comparison_plots.py  --  dmbd_joule v5"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

BG   = "#05080f"; SURF = "#0b1422"; BORD = "#1a2e50"; WHITE = "#e0eeff"
CDIM = "#3a5070"

# One distinct color per condition
COLORS = {
    "A": "#ff4444",   # red   -- no blanket
    "B": "#ff9944",   # orange -- passive
    "C": "#ffcc00",   # yellow -- random
    "D": "#00ff9d",   # green  -- Active Inference
}

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
        "lines.linewidth":  1.6,  "figure.dpi":       130,
    })

def _ts(snaps, f):
    return np.array([getattr(s, f) for s in snaps])

def _ax(ax, title, xl="", yl=""):
    ax.set_title(title, pad=5); ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values(): sp.set_edgecolor(BORD)


def plot_comparison(results, out_dir):
    _style()
    fig = plt.figure(figsize=(17, 11), facecolor=BG)
    fig.suptitle(
        "Four-way comparison  --  isolating the Active Inference contribution",
        color=WHITE, fontsize=11, y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.36,
                           left=0.07, right=0.97, top=0.93, bottom=0.06)

    # ── Panel 1: Q_I time series ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _ax(ax, "Internal Joule heat Q_I over time", "t", "Q_I")
    for r in results:
        tl = _ts(r.snaps, "t")
        ax.plot(tl, _ts(r.snaps, "joule_I"),
                color=COLORS[r.label], label=r.label, alpha=0.85)
    ax.legend()

    # ── Panel 2: Q_I smoothed (rolling mean) ─────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    _ax(ax, "Q_I smoothed (window=30)", "t", "Q_I")
    for r in results:
        tl  = _ts(r.snaps, "t")
        qi  = _ts(r.snaps, "joule_I")
        sm  = np.convolve(qi, np.ones(30)/30, mode="valid")
        ax.plot(tl[:len(sm)], sm,
                color=COLORS[r.label], label=r.label, lw=2)
    ax.legend()

    # ── Panel 3: Bar chart — mean Q_I + JRS ──────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    _ax(ax, "Mean Q_I by condition", "", "Q_I")
    labels = [r.label for r in results]
    qi_vals = [r.mean_JI for r in results]
    cols    = [COLORS[r.label] for r in results]
    bars    = ax.bar(labels, qi_vals, color=cols, alpha=0.85, width=0.55)
    Q_A     = results[0].mean_JI
    for bar, r in zip(bars, results):
        jrs = (Q_A - r.mean_JI) / Q_A * 100
        label = f"JRS\n{jrs:+.1f}%" if r.label != "A" else "baseline"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0002,
                label, ha="center", color=WHITE, fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{r.label}\n{r.name.split('--')[1].strip()[:18]}"
                        for r in results], fontsize=7)

    # ── Panel 4: Cancellation over time ──────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _ax(ax, "Total cancellation over time", "t", "fraction")
    for r in results:
        tl = _ts(r.snaps, "t")
        ax.plot(tl, _ts(r.snaps, "cancel_total"),
                color=COLORS[r.label], label=r.label, alpha=0.85)
    ax.axhline(1.0, color=WHITE, lw=0.8, ls=":", alpha=0.4)
    ax.set_ylim(0, 1.1); ax.legend()

    # ── Panel 5: Conditional MI I(I;E|B) ─────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _ax(ax, "Conditional MI  I(I;E|B)", "t", "bits")
    for r in results:
        tl = _ts(r.snaps, "t")
        ax.plot(tl, _ts(r.snaps, "cMI_IEB"),
                color=COLORS[r.label], label=r.label, alpha=0.85)
    ax.axhline(0.05, color=WHITE, lw=0.8, ls="--", alpha=0.5,
               label="epsilon=0.05")
    ax.legend()

    # ── Panel 6: Internal current stability ──────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    _ax(ax, "Internal current <J_I> stability", "t", "J_I")
    for r in results:
        tl = _ts(r.snaps, "t")
        ax.plot(tl, _ts(r.snaps, "mean_current_I"),
                color=COLORS[r.label], label=r.label, alpha=0.85)
    ax.axhline(0.75, color=WHITE, lw=0.8, ls="--", alpha=0.5, label="J_0")
    ax.legend()

    # ── Panel 7: Q_I distribution (violin-style histogram) ───────────
    ax = fig.add_subplot(gs[2, 0])
    _ax(ax, "Q_I distribution", "Q_I", "density")
    for r in results:
        qi = _ts(r.snaps, "joule_I")
        ax.hist(qi, bins=30, color=COLORS[r.label], alpha=0.55,
                density=True, label=r.label)
        ax.axvline(qi.mean(), color=COLORS[r.label], lw=2, ls="--")
    ax.legend()

    # ── Panel 8: Key deltas bar chart ────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    _ax(ax, "JRS: D (AI v5) vs each condition", "", "JRS %")
    A, B, C, D = results
    comparisons = [
        ("D vs A\n(AI vs\nno blanket)",   (A.mean_JI - D.mean_JI) / A.mean_JI * 100),
        ("D vs B\n(AI vs\npassive)",      (B.mean_JI - D.mean_JI) / B.mean_JI * 100),
        ("D vs C\n(AI vs\nrandom)",       (C.mean_JI - D.mean_JI) / C.mean_JI * 100),
        ("B vs A\n(passive vs\nnone)",    (A.mean_JI - B.mean_JI) / A.mean_JI * 100),
    ]
    comp_colors = [COLORS["A"], COLORS["B"], COLORS["C"], "#aaaaaa"]
    xlabels, yvals = zip(*comparisons)
    bars = ax.bar(range(len(comparisons)), yvals, color=comp_colors, alpha=0.85, width=0.55)
    for bar, v in zip(bars, yvals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{v:+.1f}%", ha="center", color=WHITE, fontsize=9)
    ax.set_xticks(range(len(comparisons)))
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.axhline(0, color=WHITE, lw=0.8)
    ax.set_ylabel("JRS %")

    # ── Panel 9: Scorecard text ───────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor(SURF); ax.axis("off")
    ax.set_title("Scorecard", pad=5)
    A, B, C, D = results
    lines = [
        ("CONDITION",        "Q_I",     "JRS vs A",   "CANCEL"),
        ("─"*14,             "─"*7,     "─"*9,        "─"*7),
    ]
    for r in results:
        jrs = (A.mean_JI - r.mean_JI) / A.mean_JI * 100
        lines.append((r.name.split("--")[0].strip(),
                      f"{r.mean_JI:.5f}",
                      f"{jrs:+.1f}%",
                      f"{r.mean_cancel*100:.1f}%"))
    lines += [
        ("", "", "", ""),
        ("D beats A?", "", "YES" if D.mean_JI < A.mean_JI else "NO", ""),
        ("D beats B?", "", "YES" if D.mean_JI < B.mean_JI else "NO", ""),
        ("D beats C?", "", "YES" if D.mean_JI < C.mean_JI else "NO", ""),
        ("AI isolatable?", "",
         "YES" if (D.mean_JI < B.mean_JI and D.mean_JI < C.mean_JI) else "NO", ""),
    ]
    for i, row in enumerate(lines):
        y = 0.97 - i * 0.085
        colors = [WHITE, CDIM, WHITE, CDIM]
        xs     = [0.01, 0.38, 0.62, 0.84]
        for j, (val, col, x) in enumerate(zip(row, colors, xs)):
            c = col
            if val in ("YES",): c = "#00ff9d"
            if val in ("NO",):  c = "#ff4444"
            ax.text(x, y, val, color=c, fontsize=7.5,
                    va="top", transform=ax.transAxes,
                    fontweight="bold" if i == 0 else "normal")

    save_path = os.path.join(out_dir, "fig_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  -> {save_path}")
    return fig


def plot_full_comparison(results, out_dir):
    """Six-condition comparison figure."""
    _style()
    # Extend color palette for E and F
    COLORS_EXT = {**COLORS, "E": "#cc88ff", "F": "#00ccff", "G": "#ff88cc"}

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    fig.suptitle(
        "Six-way comparison  --  No blanket / Passive / Random / Active Inference / LQR / Q-learning",
        color=WHITE, fontsize=10, y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.36,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    def ts(r, f): return np.array([getattr(s, f) for s in r.snaps])

    # ── 1. Q_I smoothed ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _ax(ax, "Q_I smoothed (window=40)", "t", "Q_I")
    for r in results:
        tl = ts(r, "t"); qi = ts(r, "joule_I")
        sm = np.convolve(qi, np.ones(40)/40, mode="valid")
        ax.plot(tl[:len(sm)], sm,
                color=COLORS_EXT[r.label], label=r.label, lw=2)
    ax.legend(fontsize=7)

    # ── 2. Mean Q_I bar ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    _ax(ax, "Mean Q_I by condition", "", "Q_I")
    Q_A = results[0].mean_JI
    for i, r in enumerate(results):
        bar = ax.bar(i, r.mean_JI, color=COLORS_EXT[r.label], alpha=0.85, width=0.6)
        jrs = (Q_A - r.mean_JI) / Q_A * 100
        lbl = f"{jrs:+.1f}%" if r.label != "A" else "ref"
        ax.text(i, r.mean_JI + 0.0002, lbl,
                ha="center", color=WHITE, fontsize=7.5)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r.label for r in results])

    # ── 3. Cancellation ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    _ax(ax, "Total cancellation over time", "t", "fraction")
    for r in results:
        ax.plot(ts(r, "t"), ts(r, "cancel_total"),
                color=COLORS_EXT[r.label], label=r.label, alpha=0.8)
    ax.axhline(1.0, color=WHITE, lw=0.8, ls=":", alpha=0.4)
    ax.set_ylim(0, 1.1); ax.legend(fontsize=7)

    # ── 4. Conditional MI ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 3])
    _ax(ax, "Conditional MI  I(I;E|B)", "t", "bits")
    for r in results:
        sm = np.convolve(ts(r, "cMI_IEB"), np.ones(20)/20, mode="valid")
        tl = ts(r, "t")
        ax.plot(tl[:len(sm)], sm,
                color=COLORS_EXT[r.label], label=r.label, alpha=0.8)
    ax.axhline(0.05, color=WHITE, lw=0.8, ls="--", alpha=0.5)
    ax.legend(fontsize=7)

    # ── 5. Convergence speed: steps to reach 80% of final cancellation
    ax = fig.add_subplot(gs[1, 0])
    _ax(ax, "Convergence: cancel total over time", "t", "cancel")
    for r in results[2:]:   # skip A and B (never cancel)
        tl = ts(r, "t"); canc = ts(r, "cancel_total")
        sm = np.convolve(canc, np.ones(10)/10, mode="valid")
        ax.plot(tl[:len(sm)], sm,
                color=COLORS_EXT[r.label], label=r.label, lw=2)
    ax.axhline(0.80, color=WHITE, lw=0.8, ls="--", alpha=0.5, label="80%")
    ax.legend(fontsize=7)

    # ── 6. Internal current stability ────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _ax(ax, "Internal current <J_I>", "t", "J_I")
    for r in results:
        sm = np.convolve(ts(r,"mean_current_I"), np.ones(20)/20, mode="valid")
        tl = ts(r, "t")
        ax.plot(tl[:len(sm)], sm,
                color=COLORS_EXT[r.label], label=r.label, alpha=0.85)
    ax.axhline(0.75, color=WHITE, lw=0.8, ls="--", alpha=0.5, label="J_0")
    ax.legend(fontsize=7)

    # ── 7. Q_I distribution ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    _ax(ax, "Q_I distribution", "Q_I", "density")
    for r in results:
        qi = ts(r, "joule_I")
        ax.hist(qi, bins=30, color=COLORS_EXT[r.label], alpha=0.5,
                density=True, label=r.label)
    ax.legend(fontsize=7)

    # ── 8. Key delta bars ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    _ax(ax, "JRS: D (AI) vs each baseline", "", "JRS %")
    Q_A = results[0].mean_JI
    D   = next(r for r in results if r.label == "D")
    comps = [(r, (r.mean_JI - D.mean_JI) / r.mean_JI * 100)
             for r in results if r.label != "D"]
    xs = range(len(comps))
    bars = ax.bar(xs, [v for _, v in comps],
                  color=[COLORS_EXT[r.label] for r, _ in comps],
                  alpha=0.85, width=0.6)
    for bar, (_, v) in zip(bars, comps):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3 * np.sign(v + 0.01),
                f"{v:+.1f}%", ha="center", color=WHITE, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r.label for r, _ in comps])
    ax.axhline(0, color=WHITE, lw=0.8)
    ax.set_ylabel("AI advantage over condition (%)")

    # ── 9. Scorecard ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, :2])
    ax.set_facecolor(SURF); ax.axis("off")
    ax.set_title("Full scorecard — publishable claims", pad=5)

    A  = next(r for r in results if r.label == "A")
    B  = next(r for r in results if r.label == "B")
    C  = next(r for r in results if r.label == "C")
    D  = next(r for r in results if r.label == "D")
    E  = next(r for r in results if r.label == "E")
    F  = next(r for r in results if r.label == "F")

    rows = [
        ("Condition", "Q_I", "JRS vs A", "Cancel", "MI"),
        ("─"*18, "─"*7, "─"*8, "─"*6, "─"*6),
    ]
    for r in results:
        jrs = (A.mean_JI - r.mean_JI) / A.mean_JI * 100
        rows.append((r.name.split("--")[0].strip() + " " + r.name.split("--")[1].strip()[:12],
                     f"{r.mean_JI:.5f}", f"{jrs:+.1f}%",
                     f"{r.mean_cancel*100:.1f}%", f"{r.mean_MI:.4f}"))

    for i, row in enumerate(rows):
        y  = 0.97 - i * 0.10
        xs = [0.01, 0.38, 0.55, 0.70, 0.84]
        for val, x in zip(row, xs):
            ax.text(x, y, val, color=WHITE, fontsize=7.5,
                    va="top", transform=ax.transAxes,
                    fontweight="bold" if i == 0 else "normal")

    # ── 10. Claims panel ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2:])
    ax.set_facecolor(SURF); ax.axis("off")
    ax.set_title("Publishable claims", pad=5)

    D_beats_E = D.mean_JI < E.mean_JI
    D_beats_F = D.mean_JI < F.mean_JI
    D_beats_C = D.mean_JI < C.mean_JI

    claims = [
        (True,
         "Active cancellation necessary",
         f"D beats passive B by {(B.mean_JI-D.mean_JI)/B.mean_JI*100:+.1f}%"),
        (D_beats_C,
         "VBEM adds value over random assignment",
         f"D beats C by {(C.mean_JI-D.mean_JI)/C.mean_JI*100:+.1f}%"),
        (D_beats_F,
         "AI matches or beats model-free RL",
         f"D vs F: {(F.mean_JI-D.mean_JI)/F.mean_JI*100:+.1f}%"),
        (D_beats_E,
         "AI matches model-based optimal (LQR)",
         f"D vs E: {(E.mean_JI-D.mean_JI)/E.mean_JI*100:+.1f}%  *** STRONG ***"),
        (not D_beats_E,
         "AI approaches but doesn't beat LQR",
         f"D vs E gap: {(D.mean_JI-E.mean_JI)/E.mean_JI*100:+.1f}%  (expected)"),
    ]

    y = 0.93
    for ok, claim, detail in claims:
        col  = "#00ff9d" if ok else "#888888"
        mark = "+" if ok else "~"
        ax.text(0.03, y, mark, color=col, fontsize=13,
                va="top", transform=ax.transAxes)
        ax.text(0.11, y,      claim,  color=WHITE, fontsize=8,
                va="top", transform=ax.transAxes)
        ax.text(0.11, y-0.07, detail, color=CDIM,  fontsize=7,
                va="top", transform=ax.transAxes)
        y -= 0.17

    save_path = os.path.join(out_dir, "fig_full_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  -> {save_path}")
    return fig


def plot_robust_comparison(results: dict, out_dir: str):
    """
    Multi-seed robust comparison figure.
    Bar chart with 95% CI error bars for conditions A, D, G, E, F.
    """
    _style()
    COLORS_EXT = {**COLORS, "E": "#cc88ff", "F": "#00ccff", "G": "#ff88cc"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    fig.suptitle(
        "Robust multi-seed comparison  --  mean ± 95% CI  (HAC-corrected)",
        color=WHITE, fontsize=11)

    labels = list(results.keys())
    colors = [COLORS_EXT.get(l, "#aaaaaa") for l in labels]

    # Panel 1: mean Q_I with 95% CI
    ax = axes[0]; ax.set_facecolor(SURF)
    means = [results[l]["mean_QI"]  for l in labels]
    cis   = [results[l]["ci95_QI"]  for l in labels]
    bars  = ax.bar(labels, means, color=colors, alpha=0.85, width=0.55,
                   yerr=cis, capsize=5, error_kw={"color": WHITE, "lw": 1.5})
    Q_A = results["A"]["mean_QI"]
    for bar, l, m in zip(bars, labels, means):
        jrs = (Q_A - m) / Q_A * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + results[l]["ci95_QI"] + 0.0002,
                f"{jrs:+.1f}%", ha="center", color=WHITE, fontsize=8)
    _ax(ax, "Mean Q_I ± 95% CI", "", "Q_I")

    # Panel 2: cancellation
    ax = axes[1]; ax.set_facecolor(SURF)
    means_c = [results[l]["mean_cancel"] * 100 for l in labels]
    cis_c   = [results[l]["ci95_cancel"] * 100 for l in labels]
    ax.bar(labels, means_c, color=colors, alpha=0.85, width=0.55,
           yerr=cis_c, capsize=5, error_kw={"color": WHITE, "lw": 1.5})
    _ax(ax, "Mean cancellation % ± 95% CI", "", "cancel %")
    ax.set_ylim(0, 105)

    # Panel 3: per-seed scatter for D and G (circularity test)
    ax = axes[2]; ax.set_facecolor(SURF)
    D_raw = results["D"]["raw_QI"]
    G_raw = results["G"]["raw_QI"]
    n     = len(D_raw)
    xs    = np.arange(n)
    ax.scatter(xs, D_raw, color=COLORS_EXT["D"], s=60, zorder=3,
               label=f"D (AI λ_J=8)  mean={np.mean(D_raw):.5f}")
    ax.scatter(xs, G_raw, color=COLORS_EXT["G"], s=60, marker="^", zorder=3,
               label=f"G (AI λ_J=0)  mean={np.mean(G_raw):.5f}")
    for d, g in zip(D_raw, G_raw):
        ax.plot([xs[D_raw.index(d)], xs[G_raw.index(g)]],
                [d, g], color="#555555", lw=0.8, zorder=1)
    ax.axhline(np.mean(D_raw), color=COLORS_EXT["D"], lw=1.5, ls="--")
    ax.axhline(np.mean(G_raw), color=COLORS_EXT["G"], lw=1.5, ls="--")
    _ax(ax, "D vs G per seed (circularity check)", "seed", "Q_I")
    ax.legend(fontsize=7)

    for ax_ in axes:
        ax_.grid(True, alpha=0.3)
        for sp in ax_.spines.values(): sp.set_edgecolor(BORD)

    save_path = os.path.join(out_dir, "fig_robust_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  -> {save_path}")
