"""
self_check.py – Lightweight sanity tests for DAU v2.

Run with:
    python -m dau_v2.self_check

Tests
-----
1. At t=0, user q(π) is near-uniform (high entropy ≥ 95 % of max).
2. After N steps with γ annealing, user q(π) entropy decreases (becomes more peaked).
3. EFE arrays have length == steps and contain only finite values for all agents.
4. Agent matrices are valid stochastic matrices (columns sum to 1).
"""

import sys
import os
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dau_v2.config      import SIM_CFG
from dau_v2.agents      import build_designer_agent, build_smartphone_agent, build_user_agent
from dau_v2.env_smartphone import SmartphoneEnvironment
from dau_v2.inference   import policy_entropy


PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def _run_sim(steps: int = 100, seed: int = 99) -> tuple:
    rng = np.random.default_rng(seed)
    SIM_CFG.steps = steps
    designer   = build_designer_agent(rng)
    smartphone = build_smartphone_agent(rng)
    user       = build_user_agent(rng)
    env = SmartphoneEnvironment(designer, smartphone, user, rng, SIM_CFG)
    for _ in range(steps):
        env.step()
    return designer, smartphone, user


def check_user_initial_near_uniform():
    """Test 1: At t=0, user q(π) has high entropy (≥ 95 % of max possible)."""
    name = "Test 1: User q(π) near-uniform at t=0"
    designer, smartphone, user = _run_sim()

    q_pi_t0 = user.log_q_pi[0]
    H_actual = policy_entropy(q_pi_t0)
    H_max    = np.log(user.cfg.n_actions)   # max entropy for n_actions actions
    ratio    = H_actual / H_max

    passed = ratio >= 0.95
    tag = PASS if passed else FAIL
    print(f"{tag}  {name}")
    print(f"       H(q_pi_t0)={H_actual:.4f}  H_max={H_max:.4f}  ratio={ratio:.4f}")
    return passed


def check_user_entropy_decreases():
    """Test 2: After N steps with annealing, user entropy decreases."""
    name = "Test 2: User q(π) entropy decreases over simulation (annealing)"
    designer, smartphone, user = _run_sim(steps=200)

    H_start = policy_entropy(user.log_q_pi[0])
    H_end   = policy_entropy(user.log_q_pi[-1])

    passed = H_end < H_start
    tag = PASS if passed else FAIL
    print(f"{tag}  {name}")
    print(f"       H(t=0)={H_start:.4f}  H(t=T-1)={H_end:.4f}  "
          f"Δ={'↓' if passed else '↑'}{abs(H_start - H_end):.4f}")
    return passed


def check_efe_arrays_valid():
    """Test 3: EFE arrays have correct length and no NaN/Inf."""
    name = "Test 3: EFE arrays length == steps and fully finite"
    steps = 50
    designer, smartphone, user = _run_sim(steps=steps)

    all_ok = True
    for agt in [designer, smartphone, user]:
        s_ok = len(agt.efe_selected) == steps and np.all(np.isfinite(agt.efe_selected))
        m_ok = len(agt.efe_min) == steps      and np.all(np.isfinite(agt.efe_min))
        if not (s_ok and m_ok):
            all_ok = False
            print(f"       PROBLEM in {agt.cfg.name}: len_sel={len(agt.efe_selected)} "
                  f"finite_sel={np.all(np.isfinite(agt.efe_selected))} "
                  f"len_min={len(agt.efe_min)} finite_min={np.all(np.isfinite(agt.efe_min))}")

    tag = PASS if all_ok else FAIL
    print(f"{tag}  {name}")
    return all_ok


def check_stochastic_matrices():
    """Test 4: A and B matrices for each agent are valid stochastic matrices."""
    name = "Test 4: Agent A and B matrices are valid probability distributions"
    rng  = np.random.default_rng(0)
    agents = [
        build_designer_agent(rng),
        build_smartphone_agent(rng),
        build_user_agent(rng),
    ]

    all_ok = True
    for agt in agents:
        # A columns: each column of A (i.e., A[:, s]) sums to 1
        A_col_sums = agt.A.sum(axis=0)
        if not np.allclose(A_col_sums, 1.0, atol=1e-6):
            print(f"       PROBLEM: {agt.cfg.name} A col sums={A_col_sums}")
            all_ok = False

        # B columns: each column of B[:, s, a] sums to 1
        for a in range(agt.B.shape[2]):
            B_col_sums = agt.B[:, :, a].sum(axis=0)
            if not np.allclose(B_col_sums, 1.0, atol=1e-6):
                print(f"       PROBLEM: {agt.cfg.name} B[:,:,{a}] col sums={B_col_sums}")
                all_ok = False

        # All values non-negative
        if not (np.all(agt.A >= 0) and np.all(agt.B >= 0)):
            print(f"       PROBLEM: {agt.cfg.name} has negative A or B values")
            all_ok = False

    tag = PASS if all_ok else FAIL
    print(f"{tag}  {name}")
    return all_ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 55)
    print("  DAU v2  –  Self-Check Sanity Tests")
    print("=" * 55)

    results = [
        check_stochastic_matrices(),
        check_user_initial_near_uniform(),
        check_user_entropy_decreases(),
        check_efe_arrays_valid(),
    ]

    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print("=" * 55)
    print(f"  Results: {n_pass}/{len(results)} passed", end="")
    if n_fail > 0:
        print(f"  ({n_fail} FAILED)")
        sys.exit(1)
    else:
        print("  ✓ All OK")
        sys.exit(0)


if __name__ == "__main__":
    main()
