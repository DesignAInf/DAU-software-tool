"""
plotting.py – All matplotlib visualisation for DAU v2.

Functions
---------
plot_efe_timeseries(designer, smartphone, user, path)
    Three EFE curves (selected + min) on one figure.
    Saved to: results/dau_v2_efe_timeseries.png

plot_user_decisiveness(user, path)
    Policy entropy and max(q_pi) for the user over time.
    Saved to: results/dau_v2_user_decisiveness.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (works on servers / CI)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .agents import ActiveInferenceAgent


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

COLOURS = {
    "Designer":   "#E07B39",   # orange
    "Smartphone": "#3A7DC9",   # blue
    "User":       "#5BBD72",   # green
}
ALPHA_MIN = 0.45   # Transparency for EFE_min (dashed) curves


# ---------------------------------------------------------------------------
# EFE timeseries plot
# ---------------------------------------------------------------------------

def plot_efe_timeseries(
    designer:   ActiveInferenceAgent,
    smartphone: ActiveInferenceAgent,
    user:       ActiveInferenceAgent,
    save_path:  str,
) -> None:
    """
    Plot EFE_selected (solid) and EFE_min (dashed) for all three agents.

    Layout: single axis, two line styles per agent → 6 lines total.
    """
    agents = [designer, smartphone, user]
    T = len(designer.log_efe_selected)
    steps = np.arange(T)

    fig, ax = plt.subplots(figsize=(11, 5))

    for agt in agents:
        name  = agt.cfg.name
        col   = COLOURS[name]
        efe_s = agt.efe_selected
        efe_m = agt.efe_min

        ax.plot(steps, efe_s, color=col, linewidth=2.0,
                label=f"{name}  EFE selected", zorder=3)
        ax.plot(steps, efe_m, color=col, linewidth=1.2, linestyle="--",
                alpha=ALPHA_MIN, label=f"{name}  EFE min", zorder=2)

    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel("Expected Free Energy  G(a)  [nats]", fontsize=13)
    ax.set_title("DAU v2 – EFE Trajectories: Designer | Smartphone | User", fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# User decisiveness plot
# ---------------------------------------------------------------------------

def plot_user_decisiveness(
    user:      ActiveInferenceAgent,
    save_path: str,
) -> None:
    """
    Two-panel plot for the user's policy distribution evolution:
      Top:    Shannon entropy of q(π) over time (high = uncommitted)
      Bottom: max(q(π)) over time (high = decisive)
    """
    T     = len(user.log_q_pi)
    steps = np.arange(T)

    entropy_series = user.policy_entropy_series
    max_q_series   = np.array([qp.max() for qp in user.log_q_pi])
    col            = COLOURS["User"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Entropy panel ---
    ax1.plot(steps, entropy_series, color=col, linewidth=2.0)
    ax1.axhline(np.log(user.cfg.n_actions), color="grey", linestyle=":",
                linewidth=1.0, label=f"Max entropy (uniform, {user.cfg.n_actions} actions)")
    ax1.set_ylabel("Policy entropy  H[q(π)]  [nats]", fontsize=12)
    ax1.set_title("DAU v2 – User Policy Decisiveness", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # --- Max probability panel ---
    ax2.plot(steps, max_q_series, color=col, linewidth=2.0)
    ax2.axhline(1.0 / user.cfg.n_actions, color="grey", linestyle=":",
                linewidth=1.0, label=f"Uniform max (1/{user.cfg.n_actions})")
    ax2.set_xlabel("Timestep", fontsize=12)
    ax2.set_ylabel("max q(π)", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot saved] {path}")
