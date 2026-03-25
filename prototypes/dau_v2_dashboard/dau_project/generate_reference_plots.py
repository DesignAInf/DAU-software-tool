"""
Generate matplotlib figures for the same metrics shown in the web dashboard.

The former static ``dashboard_interface*.png`` assets were removed; this script
produces code-based plots (EFE trajectories, user entropy, γ schedule) from a
single simulation run. Output files are written under ``dau_project/results/``.

Usage (from this directory)::

    pip install -r requirements.txt
    python generate_reference_plots.py
    python generate_reference_plots.py --steps 300 --seed 1
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dau_v2.main import run
from dau_v2.plotting import plot_efe_timeseries, plot_user_decisiveness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DAU v2 dashboard-equivalent plots via matplotlib."
    )
    parser.add_argument("--steps", type=int, default=200, help="Simulation length")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    hist, _d, _a, _u = run(steps=args.steps, seed=args.seed)
    plot_efe_timeseries(
        hist["efe_dsg_sel"],
        hist["efe_art_sel"],
        hist["efe_usr_sel"],
        hist["efe_dsg_max"],
        hist["efe_art_max"],
        hist["efe_usr_max"],
        steps=args.steps,
    )
    plot_user_decisiveness(
        hist["usr_entropy"],
        hist["usr_maxprob"],
        hist["usr_gamma"],
        steps=args.steps,
    )
    print("[generate_reference_plots] Wrote PNGs under dau_project/results/")


if __name__ == "__main__":
    main()
