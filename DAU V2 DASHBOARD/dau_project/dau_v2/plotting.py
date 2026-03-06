"""
plotting.py — DAU v2 visualisation functions (matplotlib only).

Produces:
  results/dau_v2_efe_timeseries.png   — 3-curve EFE plot
  results/dau_v2_user_decisiveness.png — user policy entropy + max-prob
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _ensure_results():
    os.makedirs(_RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────

def plot_efe_timeseries(
    efe_dsg:  np.ndarray,
    efe_art:  np.ndarray,
    efe_usr:  np.ndarray,
    efe_dsg_min: np.ndarray,
    efe_art_min: np.ndarray,
    efe_usr_min: np.ndarray,
    steps: int,
    filename: str = "dau_v2_efe_timeseries.png",
):
    """
    Two-row figure:
      Top    — EFE_selected(t) for all three agents.
      Bottom — EFE_min(t) (best available policy) for all three agents.

    Shaded band between EFE_min and EFE_selected shows how far the chosen
    policy is from optimal at each step.
    """
    _ensure_results()
    t = np.arange(steps)

    colors = {
        "Designer":   "#E05C5C",
        "Smartphone": "#4C9BE8",
        "User":       "#52B788",
    }

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1]})
    fig.patch.set_facecolor("#F7F7F7")

    labels = ["Designer", "Smartphone", "User"]
    sel    = [efe_dsg,     efe_art,      efe_usr]
    mins   = [efe_dsg_min, efe_art_min,  efe_usr_min]

    for ax_idx, (ax, title) in enumerate(zip(
            axes, ["EFE_selected(t)  —  EFE of chosen policy",
                   "EFE_max(t)  —  EFE of best available policy"])):
        ax.set_facecolor("#FFFFFF")
        data = sel if ax_idx == 0 else mins
        for lbl, d in zip(labels, data):
            ax.plot(t, d, color=colors[lbl], linewidth=1.8, label=lbl)
        ax.set_ylabel(title, fontsize=10)
        ax.legend(fontsize=9, framealpha=0.7, loc="upper right")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title(
        "DAU v2 — Expected Free Energy over time  (Rigid User scenario)",
        fontsize=13, fontweight="bold",
    )
    axes[1].set_xlabel("Timestep  $t$", fontsize=12)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    out = os.path.join(_RESULTS_DIR, filename)
    plt.savefig(out, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot] saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_user_decisiveness(
    entropy_hist: np.ndarray,
    maxprob_hist: np.ndarray,
    gamma_hist:   np.ndarray,
    steps: int,
    filename: str = "dau_v2_user_decisiveness.png",
):
    """
    Three-panel figure showing the User's evolving commitment over time:
      Panel 1 — policy entropy H(q_π)  (high = uncertain, low = committed)
      Panel 2 — max(q_π)               (high = decisive)
      Panel 3 — γ_user(t)              (policy precision schedule)
    """
    _ensure_results()
    t = np.arange(steps)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.patch.set_facecolor("#F7F7F7")

    data_and_meta = [
        (entropy_hist, "Policy entropy  H(q_π)",    "#E05C5C", "lower = more committed"),
        (maxprob_hist, "Max policy probability",      "#4C9BE8", "higher = more decisive"),
        (gamma_hist,   "Policy precision  γ_user(t)", "#52B788", "linear annealing schedule"),
    ]

    for ax, (d, ylabel, color, note) in zip(axes, data_and_meta):
        ax.set_facecolor("#FFFFFF")
        ax.plot(t, d, color=color, linewidth=1.8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.text(0.98, 0.92, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#555555")
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title(
        "DAU v2 — User Decisiveness over time  (empty model → definite)",
        fontsize=13, fontweight="bold",
    )
    axes[2].set_xlabel("Timestep  $t$", fontsize=12)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    out = os.path.join(_RESULTS_DIR, filename)
    plt.savefig(out, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plot] saved → {out}")
