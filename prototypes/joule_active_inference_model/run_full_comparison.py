#!/usr/bin/env python3
"""
run_full_comparison.py
-----------------------
Seven-way comparison with all three methodological fixes:

  FIX 1: E and F use fair role structure (VBEM assigns roles, controller
          decides kappa only -- not role[:]=ACTIVE).
  FIX 2: All conditions share the same warmed-up cable state.
  FIX 3: Condition G = AI with lambda_J=0 (circularity test).

  python3 run_full_comparison.py             # single-seed 7-way
  python3 run_full_comparison.py --robust    # multi-seed (10 seeds)
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dmbd_joule.system     import SystemParams
from dmbd_joule.comparison import run_full_comparison, run_robust_comparison

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robust",  action="store_true")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=800)
    p.add_argument("--out-dir", type=str, default="results/comparison")
    args = p.parse_args()

    params = SystemParams(
        kappa_phonon=1.60, kappa_em=1.40, kappa_defect=1.10,
        relax_phonon=0.03, relax_em=0.01,
        blanket_strength=0.95, n_hot_segments=30, min_Ba_per_hot_seg=3,
        rng_seed=42,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    if args.robust:
        seeds = [42, 123, 7, 999, 314, 2718, 1618, 256, 13, 77][:args.n_seeds]
        run_robust_comparison(
            params=params, n_steps=args.n_steps,
            seeds=seeds, out_dir=args.out_dir,
            save_plots=True, verbose=True,
        )
    else:
        run_full_comparison(
            params=params, n_warmup=200, n_steps=args.n_steps,
            out_dir=args.out_dir, save_plots=True, verbose=True,
        )

if __name__ == "__main__":
    main()
