import argparse, sys, os
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dmbd_joule.system     import SystemParams
    from dmbd_joule.experiment import run_experiment
else:
    from .system     import SystemParams
    from .experiment import run_experiment

def main():
    p = argparse.ArgumentParser(
        description="DMBD Joule v5 -- toward complete Joule elimination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n-warmup",          type=int,   default=200)
    p.add_argument("--n-baseline",        type=int,   default=400)
    p.add_argument("--n-blanket",         type=int,   default=800)
    p.add_argument("--out-dir",           type=str,   default="results")
    p.add_argument("--no-plots",          action="store_true")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--n-electrons",       type=int,   default=240)
    p.add_argument("--phonon-temp",       type=float, default=0.05)
    p.add_argument("--em-sigma",          type=float, default=0.55)
    p.add_argument("--resistivity",       type=float, default=0.40)
    p.add_argument("--blanket-strength",  type=float, default=0.92)
    p.add_argument("--kappa-phonon",      type=float, default=1.40)
    p.add_argument("--kappa-em",          type=float, default=1.10)
    p.add_argument("--kappa-defect",      type=float, default=0.95)
    p.add_argument("--lambda-joule",      type=float, default=8.0)
    p.add_argument("--relax-phonon",      type=float, default=0.08)
    p.add_argument("--relax-em",          type=float, default=0.05)
    p.add_argument("--sigma-min",         type=float, default=0.002)
    p.add_argument("--n-hot-segments",    type=int,   default=20)
    p.add_argument("--min-ba-per-seg",    type=int,   default=2)
    args = p.parse_args()

    params = SystemParams(
        n_electrons         = args.n_electrons,
        phonon_temp         = args.phonon_temp,
        em_sigma            = args.em_sigma,
        base_resistivity    = args.resistivity,
        blanket_strength    = args.blanket_strength,
        kappa_phonon        = args.kappa_phonon,
        kappa_em            = args.kappa_em,
        kappa_defect        = args.kappa_defect,
        lambda_joule_EFE    = args.lambda_joule,
        relax_phonon        = args.relax_phonon,
        relax_em            = args.relax_em,
        sigma_min           = args.sigma_min,
        n_hot_segments      = args.n_hot_segments,
        min_Ba_per_hot_seg  = args.min_ba_per_seg,
        rng_seed            = args.seed,
    )
    run_experiment(
        n_warmup   = args.n_warmup,
        n_baseline = args.n_baseline,
        n_blanket  = args.n_blanket,
        params     = params,
        out_dir    = args.out_dir,
        save_plots = not args.no_plots,
        verbose    = True,
    )

if __name__ == "__main__":
    main()
