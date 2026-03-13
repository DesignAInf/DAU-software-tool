#!/usr/bin/env python3
"""
run_experiment.py
-----------------
Convenience wrapper: runs the full v5 experiment with recommended parameters
and saves four figures to results/v5/.

Usage:
    python3 run_experiment.py                      # default parameters
    python3 run_experiment.py --relax-em 0.01 \
                              --relax-phonon 0.03  # aggressive cancellation (~40% JRS)
    python3 -m dmbd_joule --help                   # all CLI options
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dmbd_joule.system     import SystemParams
from dmbd_joule.experiment import run_experiment

if __name__ == "__main__":
    params = SystemParams(
        kappa_phonon       = 1.40,
        kappa_em           = 1.10,
        kappa_defect       = 0.95,
        relax_phonon       = 0.08,
        relax_em           = 0.05,
        blanket_strength   = 0.92,
        n_hot_segments     = 20,
        min_Ba_per_hot_seg = 2,
        rng_seed           = 42,
    )
    run_experiment(
        n_warmup   = 200,
        n_baseline = 400,
        n_blanket  = 800,
        params     = params,
        out_dir    = "results/v5",
        save_plots = True,
        verbose    = True,
    )
