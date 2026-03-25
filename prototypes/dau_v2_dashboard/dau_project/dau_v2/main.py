"""
main.py — DAU v2 simulation entry point.

Usage
-----
    python -m dau_v2.main --steps 200 --seed 0

Outputs
-------
    results/dau_v2_efe_timeseries.png
    results/dau_v2_user_decisiveness.png
"""

import argparse
import numpy as np

from .agents          import DesignerAgent, SmartphoneAgent, UserAgent
from .env_smartphone  import SmartphoneEnvironment
from .plotting        import plot_efe_timeseries, plot_user_decisiveness
from . import config  as cfg


def run(steps: int = cfg.DEFAULT_STEPS, seed: int = cfg.DEFAULT_SEED):
    """Execute the full simulation and return history dictionaries."""

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ── Instantiate agents and environment ────────────────────────────────────
    designer   = DesignerAgent()
    smartphone = SmartphoneAgent()
    user       = UserAgent()
    env        = SmartphoneEnvironment(seed=seed)

    obs_dsg, obs_art, obs_usr = env.reset()

    # ── History buffers ────────────────────────────────────────────────────────
    hist = {
        # EFE_selected and EFE_max per agent
        "efe_dsg_sel":  np.zeros(steps),
        "efe_art_sel":  np.zeros(steps),
        "efe_usr_sel":  np.zeros(steps),
        "efe_dsg_max":  np.zeros(steps),
        "efe_art_max":  np.zeros(steps),
        "efe_usr_max":  np.zeros(steps),
        # User decisiveness
        "usr_entropy":  np.zeros(steps),
        "usr_maxprob":  np.zeros(steps),
        "usr_gamma":    np.zeros(steps),
        # Actions chosen (for transparency)
        "act_dsg":      np.zeros(steps, dtype=int),
        "act_art":      np.zeros(steps, dtype=int),
        "act_usr":      np.zeros(steps, dtype=int),
    }

    # ── Main loop ─────────────────────────────────────────────────────────────
    for t in range(steps):

        # Each agent takes one step given its observation
        act_dsg, q_pi_dsg, G_dsg, efe_d_sel, efe_d_max = \
            designer.step(obs_dsg, t=t, T=steps)

        act_art, q_pi_art, G_art, efe_a_sel, efe_a_max = \
            smartphone.step(obs_art, t=t, T=steps)

        act_usr, q_pi_usr, G_usr, efe_u_sel, efe_u_max = \
            user.step(obs_usr, t=t, T=steps)

        # Environment transitions and produces next observations
        obs_dsg, obs_art, obs_usr = env.step(act_dsg, act_art, act_usr)

        # Record
        hist["efe_dsg_sel"][t]  = efe_d_sel
        hist["efe_art_sel"][t]  = efe_a_sel
        hist["efe_usr_sel"][t]  = efe_u_sel
        hist["efe_dsg_max"][t]  = efe_d_max
        hist["efe_art_max"][t]  = efe_a_max
        hist["efe_usr_max"][t]  = efe_u_max
        hist["usr_entropy"][t]  = user.policy_entropy()
        hist["usr_maxprob"][t]  = user.policy_max_prob()
        hist["usr_gamma"][t]    = user.gamma
        hist["act_dsg"][t]      = act_dsg
        hist["act_art"][t]      = act_art
        hist["act_usr"][t]      = act_usr

    return hist, designer, smartphone, user


def print_summary(hist: dict, designer, smartphone, user, steps: int):
    """Print end-of-run statistics to stdout."""
    print("\n" + "=" * 60)
    print("DAU v2 — Simulation Summary")
    print("=" * 60)

    for label, sel_key, max_key in [
        ("Designer",   "efe_dsg_sel", "efe_dsg_max"),
        ("Smartphone", "efe_art_sel", "efe_art_max"),
        ("User",       "efe_usr_sel", "efe_usr_max"),
    ]:
        s = hist[sel_key]
        m = hist[max_key]
        print(f"\n  {label}:")
        print(f"    EFE_selected  — mean={s.mean():.3f}  "
              f"median={np.median(s):.3f}  "
              f"std={s.std():.3f}  "
              f"[{s.min():.3f}, {s.max():.3f}]")
        print(f"    EFE_max       — mean={m.mean():.3f}  "
              f"median={np.median(m):.3f}  "
              f"std={m.std():.3f}  "
              f"[{m.min():.3f}, {m.max():.3f}]")

    final_entropy = hist["usr_entropy"][-1]
    init_entropy  = hist["usr_entropy"][0]
    final_maxp    = hist["usr_maxprob"][-1]
    final_gamma   = hist["usr_gamma"][-1]
    print(f"\n  User policy entropy:")
    print(f"    t=0     H = {init_entropy:.4f}  (high → uniform)")
    print(f"    t={steps-1:<4} H = {final_entropy:.4f}  "
          f"(lower → more committed)")
    print(f"    Final max(q_π) = {final_maxp:.4f}")
    print(f"    Final γ_user   = {final_gamma:.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DAU v2 — Active Inference smartphone triad simulation"
    )
    parser.add_argument("--steps", type=int, default=cfg.DEFAULT_STEPS,
                        help="Number of simulation timesteps (default: 200)")
    parser.add_argument("--seed",  type=int, default=cfg.DEFAULT_SEED,
                        help="Random seed (default: 0)")
    args = parser.parse_args()

    print(f"[run] steps={args.steps}  seed={args.seed}")

    hist, designer, smartphone, user = run(steps=args.steps, seed=args.seed)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_efe_timeseries(
        hist["efe_dsg_sel"], hist["efe_art_sel"], hist["efe_usr_sel"],
        hist["efe_dsg_max"], hist["efe_art_max"], hist["efe_usr_max"],
        steps=args.steps,
    )

    plot_user_decisiveness(
        hist["usr_entropy"],
        hist["usr_maxprob"],
        hist["usr_gamma"],
        steps=args.steps,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(hist, designer, smartphone, user, args.steps)

    return hist


if __name__ == "__main__":
    main()
