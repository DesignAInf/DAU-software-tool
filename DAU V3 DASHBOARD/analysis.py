"""
analysis.py — DAU v3 statistical analysis.

Runs the simulation across N seeds and produces:
  1. EFE timeseries per profile (mean ± std band)
  2. Policy entropy per profile (mean ± std band)
  3. User state distribution heatmap (profile × state)
  4. Manipulation effectiveness heatmap (Smartphone strategy × profile)
  5. Final policy distribution per profile
  6. Resistance convergence: timestep at which each profile first reaches
     RESISTANT or DISENGAGED state

Usage
-----
    python -m dau_v3.analysis --steps 300 --n_seeds 50
    python -m dau_v3.analysis --steps 300 --n_seeds 50 --no_plots
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from . import config as cfg
from .main import run

# ── Visual style ──────────────────────────────────────────────────────────────
PROFILE_COLORS = {
    "Achiever":  "#E05C5C",
    "Social":    "#4C9BE8",
    "Anxious":   "#F4A261",
    "Resistant": "#52B788",
}
DARK_BG   = "#0C0C14"
SURFACE   = "#13131E"
SURFACE2  = "#1C1C2C"
BORDER    = "#2A2A40"
TEXT      = "#E8E8F0"
TEXT_DIM  = "#5A5A78"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE2)
    ax.spines[:].set_color(BORDER)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    if title:  ax.set_title(title, color=TEXT, fontsize=11, pad=8, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=BORDER, linewidth=0.5, alpha=0.5)


def run_batch(steps: int, n_seeds: int):
    """
    Run the simulation n_seeds times and aggregate results.

    Returns
    -------
    agg : dict
        {profile_name: {metric_key: ndarray shape (n_seeds, steps)}}
    scalar_agg : dict
        {profile_name: {scalar_key: ndarray shape (n_seeds,)}}
    """
    profiles = list(cfg.USER_PROFILES.keys())
    metric_keys = ["efe_usr_sel", "usr_entropy", "usr_maxprob",
                   "usr_gamma", "user_state", "act_usr",
                   "efe_dsg_sel", "efe_art_sel"]

    agg = {p: {k: np.zeros((n_seeds, steps)) for k in metric_keys}
           for p in profiles}

    print(f"Running {n_seeds} seeds × {steps} steps × 4 profiles …")
    for i, seed in enumerate(range(n_seeds)):
        hist_all, _, _, _ = run(steps=steps, seed=seed)
        for p in profiles:
            for k in metric_keys:
                if k in hist_all[p]:
                    agg[p][k][i] = hist_all[p][k]
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_seeds} done")

    return agg


def compute_scalars(agg, steps):
    """Derive scalar metrics from the batch results."""
    profiles = list(cfg.USER_PROFILES.keys())
    scalars = {}
    for p in profiles:
        states = agg[p]["user_state"]          # (n_seeds, steps)
        # First timestep where user reaches RESISTANT (7) or DISENGAGED (9)
        resist_t = []
        for seed_i in range(states.shape[0]):
            traj = states[seed_i]
            hits = np.where((traj == 7) | (traj == 9))[0]
            resist_t.append(hits[0] if len(hits) > 0 else steps)
        # Churn rate: fraction of seeds where user ends in DISENGAGED(9) or ADDICTED(8)
        final_states = states[:, -1]
        churn_rate   = float(np.mean((final_states == 9) | (final_states == 8)))
        # Stress index: fraction of timesteps in STRESSED states (4,5)
        stress_idx   = float(np.mean((states == 4) | (states == 5)))
        scalars[p] = {
            "resist_t":    np.array(resist_t),
            "churn_rate":  churn_rate,
            "stress_idx":  stress_idx,
            "efe_mean":    float(agg[p]["efe_usr_sel"].mean()),
            "efe_std":     float(agg[p]["efe_usr_sel"].std()),
            "entropy_drop": float(agg[p]["usr_entropy"][:, 0].mean()
                                 - agg[p]["usr_entropy"][:, -1].mean()),
        }
    return scalars


# ── Plot 1: EFE timeseries per profile ───────────────────────────────────────

def plot_efe_timeseries(agg, steps, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor=DARK_BG)
    fig.suptitle("EFE Selected — per User Profile (mean ± 1 std)",
                 color=TEXT, fontsize=13, fontweight="bold")
    t = np.arange(steps)
    for ax, p in zip(axes.flat, cfg.USER_PROFILES.keys()):
        col = PROFILE_COLORS[p]
        vals = agg[p]["efe_usr_sel"]
        mu   = vals.mean(axis=0)
        sd   = vals.std(axis=0)
        _style_ax(ax, title=p, xlabel="timestep", ylabel="EFE selected")
        ax.fill_between(t, mu - sd, mu + sd, alpha=0.2, color=col)
        ax.plot(t, mu, color=col, linewidth=1.8, label=p)
        ax.axhline(0, color=BORDER, linewidth=0.7, linestyle="--")
        desc = cfg.USER_PROFILES[p]["desc"]
        ax.set_title(f"{p}\n{desc}", color=TEXT, fontsize=9, pad=6)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "v3_efe_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 2: Entropy timeseries per profile ───────────────────────────────────

def plot_entropy(agg, steps, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    _style_ax(ax, title="User Policy Entropy over Time — all profiles",
              xlabel="timestep", ylabel="H(q(π))")
    t = np.arange(steps)
    for p, col in PROFILE_COLORS.items():
        vals = agg[p]["usr_entropy"]
        mu   = vals.mean(axis=0)
        sd   = vals.std(axis=0)
        ax.fill_between(t, mu - sd, mu + sd, alpha=0.15, color=col)
        ax.plot(t, mu, color=col, linewidth=1.8, label=p)
    ax.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "v3_entropy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 3: User state distribution heatmap ──────────────────────────────────

def plot_state_heatmap(agg, out_dir):
    profiles = list(cfg.USER_PROFILES.keys())
    n_states = cfg.USR_N_STATES
    matrix   = np.zeros((len(profiles), n_states))
    for i, p in enumerate(profiles):
        states = agg[p]["user_state"]  # (n_seeds, steps)
        for s in range(n_states):
            matrix[i, s] = float(np.mean(states == s))

    fig, ax = plt.subplots(figsize=(13, 4), facecolor=DARK_BG)
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=matrix.max())
    ax.set_facecolor(SURFACE2)
    ax.set_xticks(range(n_states))
    ax.set_xticklabels([s.replace("_", "\n") for s in cfg.USR_STATES],
                       fontsize=7, color=TEXT_DIM)
    ax.set_yticks(range(len(profiles)))
    ax.set_yticklabels(profiles, fontsize=9, color=TEXT)
    ax.set_title("User State Occupancy Heatmap (fraction of time in each state)",
                 color=TEXT, fontsize=11, fontweight="bold", pad=10)
    for i in range(len(profiles)):
        for j in range(n_states):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if matrix[i,j] > 0.15 else TEXT_DIM)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=TEXT_DIM, labelsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "v3_state_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 4: Scalar summary bar chart ─────────────────────────────────────────

def plot_scalar_summary(scalars, out_dir):
    profiles = list(scalars.keys())
    metrics  = ["churn_rate", "stress_idx", "entropy_drop"]
    labels   = ["Churn Rate\n(ADDICTED or DISENGAGED at end)",
                "Stress Index\n(fraction of time in STRESSED states)",
                "Entropy Drop\n(t=0 → t=end, higher = more decisive)"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=DARK_BG)
    fig.suptitle("Scalar Outcomes per Profile (N seeds)",
                 color=TEXT, fontsize=13, fontweight="bold")
    for ax, metric, label in zip(axes, metrics, labels):
        vals  = [scalars[p][metric] for p in profiles]
        colors= [PROFILE_COLORS[p] for p in profiles]
        bars  = ax.bar(profiles, vals, color=colors, edgecolor=BORDER, linewidth=0.8)
        _style_ax(ax, title=label)
        ax.set_xticklabels(profiles, rotation=15, ha="right", fontsize=8, color=TEXT_DIM)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "v3_scalar_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 5: Time to resistance ────────────────────────────────────────────────

def plot_resistance_time(scalars, steps, out_dir):
    profiles = list(scalars.keys())
    fig, ax  = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
    _style_ax(ax, title="Time to Resistance / Disengagement per Profile",
              xlabel="timestep", ylabel="density")
    for p in profiles:
        col = PROFILE_COLORS[p]
        rt  = scalars[p]["resist_t"]
        # Histogram as step plot
        counts, edges = np.histogram(rt, bins=20, range=(0, steps), density=True)
        ax.step(edges[:-1], counts, where="mid", color=col, linewidth=1.8, label=p)
        ax.axvline(np.median(rt), color=col, linewidth=0.9, linestyle="--", alpha=0.7)
    ax.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "v3_resistance_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {path}")


# ── Plot 6: Final policy distribution ────────────────────────────────────────

def plot_final_policies(agg, steps, out_dir):
    profiles = list(cfg.USER_PROFILES.keys())
    n_pol    = cfg.USR_N_POLICIES
    x        = np.arange(n_pol)
    width    = 0.18

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
    _style_ax(ax, title="Final Timestep Policy Distribution per Profile",
              xlabel="policy", ylabel="mean q(π)")
    for i, p in enumerate(profiles):
        # approximate final q(π) from act_usr distribution in last 20% of run
        last_t  = int(steps * 0.8)
        acts    = agg[p]["act_usr"][:, last_t:]   # (n_seeds, last_steps)
        dist    = np.array([np.mean(acts == pi) for pi in range(n_pol)])
        col     = PROFILE_COLORS[p]
        bars    = ax.bar(x + i * width, dist, width, color=col,
                         label=p, edgecolor=BORDER, linewidth=0.6)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([p.replace("_", "\n") for p in cfg.USR_POLICIES],
                       fontsize=8, color=TEXT_DIM)
    ax.legend(facecolor=SURFACE2, labelcolor=TEXT, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "v3_final_policies.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DAU v3 — Statistical analysis across N seeds"
    )
    parser.add_argument("--steps",    type=int, default=cfg.DEFAULT_STEPS)
    parser.add_argument("--n_seeds",  type=int, default=cfg.N_SEEDS)
    parser.add_argument("--out_dir",  type=str, default="results")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    agg = run_batch(steps=args.steps, n_seeds=args.n_seeds)
    scalars = compute_scalars(agg, args.steps)

    # Print scalar table
    print("\n" + "=" * 60)
    print("DAU v3 — Statistical Summary")
    print("=" * 60)
    print(f"  {'Profile':<12} {'EFE mean':>10} {'Churn':>8} {'Stress':>8} {'ΔEntropy':>10} {'Resist_t (med)':>15}")
    print("  " + "-" * 60)
    for p, sc in scalars.items():
        print(f"  {p:<12} {sc['efe_mean']:>+10.4f} {sc['churn_rate']:>8.3f} "
              f"{sc['stress_idx']:>8.3f} {sc['entropy_drop']:>10.4f} "
              f"{int(np.median(sc['resist_t'])):>15}")
    print("=" * 60)

    if not args.no_plots:
        print("\nGenerating plots …")
        plot_efe_timeseries(agg, args.steps, args.out_dir)
        plot_entropy(agg, args.steps, args.out_dir)
        plot_state_heatmap(agg, args.out_dir)
        plot_scalar_summary(scalars, args.out_dir)
        plot_resistance_time(scalars, args.steps, args.out_dir)
        plot_final_policies(agg, args.steps, args.out_dir)
        print(f"\nAll plots saved to ./{args.out_dir}/")


if __name__ == "__main__":
    main()
