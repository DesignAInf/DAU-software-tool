"""
main.py — DAU v3 simulation entry point.

Runs all 4 user profiles in parallel for a single (steps, seed) pair.

Usage
-----
    python -m dau_v3.main --steps 300 --seed 0

Returns
-------
    hist_all : dict mapping profile_name → per-profile history dict
    designer, smartphone : final agent objects
    users    : dict mapping profile_name → UserAgent
"""

import argparse
import numpy as np

from .agents         import DesignerAgent, SmartphoneAgent, UserAgent
from .env_smartphone import SmartphoneEnvironment
from . import config as cfg


def run(steps: int = cfg.DEFAULT_STEPS, seed: int = cfg.DEFAULT_SEED):
    """
    Execute the full simulation for all 4 user profiles simultaneously.
    Designer and Smartphone are shared; each profile has its own UserAgent
    and its own copy of the environment.

    Returns
    -------
    hist_all : dict  {profile_name: hist_dict}
    designer : DesignerAgent
    smartphone : SmartphoneAgent
    users : dict {profile_name: UserAgent}
    """
    np.random.seed(seed)

    profiles = list(cfg.USER_PROFILES.keys())

    # ── Shared agents (one Designer, one Smartphone) ───────────────────────
    designer   = DesignerAgent()
    smartphone = SmartphoneAgent()

    # ── Per-profile agents and environments ───────────────────────────────
    users = {p: UserAgent(profile_name=p) for p in profiles}
    envs  = {p: SmartphoneEnvironment(seed=seed + i)
             for i, p in enumerate(profiles)}

    # Initial observations
    obs_dsg_shared, _, _ = envs[profiles[0]].reset()
    obs_art_shared       = 0
    obs_usr = {}
    for p in profiles:
        _, _, o = envs[p].reset()
        obs_usr[p] = o

    # ── History buffers ────────────────────────────────────────────────────
    keys_shared = ["efe_dsg_sel","efe_dsg_max","efe_art_sel","efe_art_max",
                   "act_dsg","act_art"]
    hist_shared = {k: np.zeros(steps, dtype=float if "efe" in k else int)
                   for k in keys_shared}

    keys_profile = ["efe_usr_sel","efe_usr_max","usr_entropy","usr_maxprob",
                    "usr_gamma","act_usr","user_state"]
    hist_all = {
        p: {k: np.zeros(steps, dtype=float if k!="act_usr" and k!="user_state" else int)
            for k in keys_profile}
        for p in profiles
    }

    # ── Main loop ─────────────────────────────────────────────────────────
    for t in range(steps):

        # Shared agents act once per timestep
        act_dsg, _, G_dsg, efe_d_sel, efe_d_max = \
            designer.step(obs_dsg_shared, t=t, T=steps)
        act_art, _, G_art, efe_a_sel, efe_a_max = \
            smartphone.step(obs_art_shared, t=t, T=steps)

        hist_shared["efe_dsg_sel"][t] = efe_d_sel
        hist_shared["efe_dsg_max"][t] = efe_d_max
        hist_shared["efe_art_sel"][t] = efe_a_sel
        hist_shared["efe_art_max"][t] = efe_a_max
        hist_shared["act_dsg"][t]     = act_dsg
        hist_shared["act_art"][t]     = act_art

        # Each user profile steps independently
        art_obs_accum = []
        dsg_obs_accum = []
        for p in profiles:
            act_usr, _, G_usr, efe_u_sel, efe_u_max = \
                users[p].step(obs_usr[p], t=t, T=steps)
            obs_dsg_p, obs_art_p, obs_usr_p = \
                envs[p].step(act_dsg, act_art, act_usr)

            hist_all[p]["efe_usr_sel"][t] = efe_u_sel
            hist_all[p]["efe_usr_max"][t] = efe_u_max
            hist_all[p]["usr_entropy"][t] = users[p].policy_entropy()
            hist_all[p]["usr_maxprob"][t] = users[p].policy_max_prob()
            hist_all[p]["usr_gamma"][t]   = users[p].gamma
            hist_all[p]["act_usr"][t]     = act_usr
            hist_all[p]["user_state"][t]  = envs[p].user_state

            obs_usr[p] = obs_usr_p
            art_obs_accum.append(obs_art_p)
            dsg_obs_accum.append(obs_dsg_p)

        # Shared agents get mean observation signal (average across profiles)
        obs_art_shared = int(np.round(np.mean(art_obs_accum)))
        obs_dsg_shared = int(np.round(np.mean(dsg_obs_accum)))

    # Merge shared history into each profile dict for convenience
    for p in profiles:
        hist_all[p].update(hist_shared)

    return hist_all, designer, smartphone, users


def print_summary(hist_all: dict, steps: int):
    profiles = list(hist_all.keys())
    print("\n" + "=" * 65)
    print("DAU v3 — Simulation Summary")
    print("=" * 65)

    # Shared agents
    h = hist_all[profiles[0]]
    print(f"\n  Designer  EFE_sel: mean={h['efe_dsg_sel'].mean():.3f}  "
          f"std={h['efe_dsg_sel'].std():.3f}")
    print(f"  Smartphone EFE_sel: mean={h['efe_art_sel'].mean():.3f}  "
          f"std={h['efe_art_sel'].std():.3f}")

    print(f"\n  {'Profile':<12} EFE_mean  Entropy_t0  Entropy_end  MaxProb_end  FinalState")
    print("  " + "-"*62)
    for p in profiles:
        h = hist_all[p]
        print(f"  {p:<12} "
              f"{h['efe_usr_sel'].mean():+.3f}     "
              f"{h['usr_entropy'][0]:.3f}       "
              f"{h['usr_entropy'][-1]:.3f}        "
              f"{h['usr_maxprob'][-1]:.3f}        "
              f"{h['user_state'][-1]}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="DAU v3 — Active Inference persuasive design study"
    )
    parser.add_argument("--steps", type=int, default=cfg.DEFAULT_STEPS)
    parser.add_argument("--seed",  type=int, default=cfg.DEFAULT_SEED)
    args = parser.parse_args()

    print(f"[run] steps={args.steps}  seed={args.seed}")
    hist_all, designer, smartphone, users = run(steps=args.steps, seed=args.seed)
    print_summary(hist_all, args.steps)
    return hist_all


if __name__ == "__main__":
    main()
