"""
main.py – Entry point for DAU v2 smartphone triad simulation.

Usage
-----
    python -m dau_v2.main --steps 200 --seed 0
    python -m dau_v2.main --steps 100 --seed 42 --no-decisiveness-plot

Arguments
---------
--steps  INT   Number of simulation timesteps (default 200)
--seed   INT   RNG seed for reproducibility (default 0)
--no-decisiveness-plot   Skip saving the user decisiveness plot

Outputs
-------
  results/dau_v2_efe_timeseries.png      (always)
  results/dau_v2_user_decisiveness.png   (unless --no-decisiveness-plot)
  Terminal: per-agent EFE summary + final user policy entropy
"""

import argparse
import os
import sys
import numpy as np

# Resolve the repo root so the module can be run from any working directory
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dau_v2.config      import SIM_CFG
from dau_v2.agents      import build_designer_agent, build_smartphone_agent, build_user_agent
from dau_v2.env_smartphone import SmartphoneEnvironment
from dau_v2.inference   import policy_entropy
from dau_v2.plotting    import plot_efe_timeseries, plot_user_decisiveness


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

def run(steps: int, seed: int) -> tuple:
    """
    Execute the full DAU v2 simulation.

    Returns (designer, smartphone, user) agent objects with populated logs.
    """
    rng = np.random.default_rng(seed)

    # Build agents
    designer   = build_designer_agent(rng)
    smartphone = build_smartphone_agent(rng)
    user       = build_user_agent(rng)

    # Override steps in sim config (from CLI arg)
    SIM_CFG.steps = steps

    # Create environment
    env = SmartphoneEnvironment(designer, smartphone, user, rng, SIM_CFG)

    print(f"\nDAU v2  |  steps={steps}  seed={seed}")
    print("=" * 50)
    print(f"{'t':>5}  {'D-act':>6}  {'P-act':>6}  {'U-act':>6}  {'γ_user':>7}")
    print("-" * 50)

    for t in range(steps):
        info = env.step()
        # Print a sample every 20 steps to monitor progress
        if t % 20 == 0 or t == steps - 1:
            print(
                f"{info['t']:>5}  "
                f"{designer.cfg.action_labels[info['designer_action']]:>13}  "
                f"{smartphone.cfg.action_labels[info['phone_action']]:>17}  "
                f"{user.cfg.action_labels[info['user_action']]:>14}  "
                f"{info['gamma_user']:>7.3f}"
            )

    print("=" * 50)
    return designer, smartphone, user


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(designer, smartphone, user) -> None:
    print("\n--- EFE Summary (selected action) ---")
    for agt in [designer, smartphone, user]:
        efe = agt.efe_selected
        print(
            f"  {agt.cfg.name:<12}: "
            f"mean={efe.mean():.4f}  median={np.median(efe):.4f}  "
            f"min={efe.min():.4f}  max={efe.max():.4f}"
        )

    # User policy entropy at start vs end
    q_pi_t0   = user.log_q_pi[0]
    q_pi_last = user.log_q_pi[-1]
    H_t0      = policy_entropy(q_pi_t0)
    H_last    = policy_entropy(q_pi_last)
    max_last  = q_pi_last.max()

    print("\n--- User Policy Decisiveness ---")
    print(f"  t=0   entropy H[q(π)] = {H_t0:.4f} nats  (max possible ≈ {np.log(user.cfg.n_actions):.4f})")
    print(f"  t=T-1 entropy H[q(π)] = {H_last:.4f} nats")
    print(f"  t=T-1 max q(π)        = {max_last:.4f}")
    print(f"  Entropy reduction      = {H_t0 - H_last:.4f} nats  "
          f"({'↓ more decisive' if H_last < H_t0 else '↑ less decisive – check γ annealing'})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DAU v2 – Smartphone Triad Active Inference Simulation"
    )
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of simulation timesteps (default 200)")
    parser.add_argument("--seed",  type=int, default=0,
                        help="RNG seed (default 0)")
    parser.add_argument("--no-decisiveness-plot", action="store_true",
                        help="Skip the user decisiveness plot")
    args = parser.parse_args()

    # Run
    designer, smartphone, user = run(args.steps, args.seed)

    # Summary
    print_summary(designer, smartphone, user)

    # Plots
    results_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    efe_path = os.path.join(results_dir, SIM_CFG.efe_plot_filename)
    dec_path = os.path.join(results_dir, SIM_CFG.decisiveness_plot_filename)

    print("\n--- Saving plots ---")
    plot_efe_timeseries(designer, smartphone, user, efe_path)
    if not args.no_decisiveness_plot:
        plot_user_decisiveness(user, dec_path)

    print("\nDone.")
    print(f"  EFE plot:         {efe_path}")
    if not args.no_decisiveness_plot:
        print(f"  Decisiveness plot: {dec_path}")


if __name__ == "__main__":
    main()
