"""
self_check.py — DAU v2 sanity tests.

Run with:
    python -m dau_v2.self_check

Three tests:
  1. At t=0, user q_pi is approximately uniform (high entropy).
  2. After N steps, user q_pi entropy decreases (annealing / learning).
  3. EFE arrays have length == steps and contain only finite values.
"""

import sys
import numpy as np
from .main   import run
from .inference import entropy


PASS = "[PASS]"
FAIL = "[FAIL]"


def _check(name: str, condition: bool, detail: str = ""):
    tag = PASS if condition else FAIL
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def run_checks(steps: int = 200, seed: int = 42):
    print("\n" + "=" * 55)
    print("DAU v2 — self_check")
    print("=" * 55)

    hist, _, _, user_agent_final = run(steps=steps, seed=seed)

    all_passed = True

    # ── Test 1: user q_pi near-uniform at t=0 ─────────────────────────────────
    # We re-create a fresh user to inspect its initial state
    from .agents import UserAgent
    fresh_user = UserAgent()
    init_entropy = entropy(fresh_user.q_pi)
    max_entropy  = np.log(fresh_user.q_pi.size)    # entropy of uniform
    # Expect initial entropy ≥ 95% of maximum
    ok1 = init_entropy >= 0.95 * max_entropy
    all_passed &= _check(
        "Initial user q_pi ≈ uniform (entropy ≥ 95% of max)",
        ok1,
        f"H={init_entropy:.4f}  H_max={max_entropy:.4f}",
    )

    # ── Test 2: user entropy decreases over time ───────────────────────────────
    first_quarter  = hist["usr_entropy"][:steps // 4].mean()
    last_quarter   = hist["usr_entropy"][-steps // 4:].mean()
    ok2 = last_quarter < first_quarter
    all_passed &= _check(
        "User entropy decreases (first-quarter mean > last-quarter mean)",
        ok2,
        f"first-Q mean={first_quarter:.4f}  last-Q mean={last_quarter:.4f}",
    )

    # ── Test 3: EFE arrays have correct length and are finite ─────────────────
    for agent_name, key in [("Designer",   "efe_dsg_sel"),
                             ("Smartphone", "efe_art_sel"),
                             ("User",       "efe_usr_sel")]:
        arr = hist[key]
        ok_len    = len(arr) == steps
        ok_finite = bool(np.all(np.isfinite(arr)))
        ok3 = ok_len and ok_finite
        all_passed &= _check(
            f"{agent_name} EFE array: length={steps} and all finite",
            ok3,
            f"len={len(arr)}  finite={ok_finite}",
        )

    print("=" * 55)
    if all_passed:
        print("All checks PASSED.\n")
        return 0
    else:
        print("Some checks FAILED.\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_checks())
