#!/usr/bin/env python3
"""run_comparison.py -- four-way comparison for dmbd_joule v5"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dmbd_joule.system     import SystemParams
from dmbd_joule.comparison import run_comparison

params = SystemParams(
    kappa_phonon  = 1.60,
    kappa_em      = 1.40,
    kappa_defect  = 1.10,
    relax_phonon  = 0.03,
    relax_em      = 0.01,
    relax_defect  = 0.04,
    blanket_strength = 0.95,
    n_hot_segments   = 30,
    min_Ba_per_hot_seg = 3,
    rng_seed = 42,
)

run_comparison(
    params    = params,
    n_warmup  = 200,
    n_steps   = 800,
    out_dir   = "results_comparison",
    save_plots= True,
    verbose   = True,
)
