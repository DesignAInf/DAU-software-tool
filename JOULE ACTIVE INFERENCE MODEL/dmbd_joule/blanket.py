"""
blanket.py  --  dmbd_joule v6
==============================
Hard closed-loop Active Inference: role assignment is a genuine
minimization of Expected Free Energy G, not a heuristic score function.

ARCHITECTURE (v6 -- loop closed):
  At every timestep, BlanketSynthesizer solves:

    omega* = argmin_{omega in Omega} G(omega)

  where omega = role assignment {role_i for i=1..N}
        Omega = valid partitions (15% <= f_r <= 55% per role)
        G(omega) = pragmatic + epistemic + joule_risk given omega

  The optimization uses stochastic coordinate descent:
    - Sample K_inner electrons per iteration (default: 20% of N)
    - For each sampled electron i, estimate G under each proposed role
    - Assign role that minimizes G_estimate
    - Repeat for K_cd iterations (default: 5)
    - Project onto Omega (fraction constraints)

  G is estimated analytically from current mean fields without
  re-running the full dynamics (first-order approximation):

    G_I(i):  add electron i to I pool  -> update J_I_est, Q_I_est
    G_S(i):  add electron i to B_s pool -> update S_density, epistemic
    G_A(i):  add electron i to B_a pool -> update A_density, joule_risk

  This makes the EFE genuinely causal: role assignment is driven by
  minimizing G, not tracking it.

COMPARISON WITH v5:
  v5: role = argmax(score_S, score_A, score_I)   [heuristic, G not used]
  v6: role = argmin G(omega)  via coord descent   [G drives policy]

  The score functions are completely removed. The only criterion is G.
"""

import numpy as np
from .system import CableSystem, Role


class BlanketSynthesizer:

    def __init__(self, system: CableSystem,
                 K_cd: int = 5,
                 sample_frac: float = 0.20,
                 tau_switch: int = 4):
        """
        Parameters
        ----------
        K_cd         : coordinate descent iterations per timestep
        sample_frac  : fraction of electrons sampled per CD iteration
        tau_switch   : cooldown steps after a role switch
        """
        self.sys         = system
        self.p           = system.p
        self.K_cd        = K_cd
        self.sample_frac = sample_frac
        self.tau_switch  = tau_switch

        # Lagrange multipliers (updated each step)
        self._lambda1 = 1.0               # flux constraint
        self._lambda2 = self.p.blanket_strength  # CI enforcement
        self._lambda3 = 0.5               # internal Joule
        self._lr      = 0.04

        # Role-switch cooldown
        self._cooldown = np.zeros(self.p.n_electrons, dtype=int)

        # Diagnostics
        self.last_G_improvement = 0.0
        self.last_n_switched    = 0

    # -----------------------------------------------------------------
    # G estimation (first-order, no dynamics re-run)
    # -----------------------------------------------------------------

    def _estimate_G(self, role_vec: np.ndarray) -> float:
        """
        Estimate G = pragmatic + epistemic + joule_risk for a given
        role assignment, using current mean fields (first-order approx).

        Does NOT re-run system dynamics -- uses analytical estimates
        from current electron velocities and mean field arrays.
        """
        e    = self.sys.state.electrons
        mf   = self.sys.state.mf
        p    = self.p
        ns   = p.n_segments
        v    = e.vx

        is_I = (role_vec == Role.INTERNAL)
        is_S = (role_vec == Role.SENSORY)
        is_A = (role_vec == Role.ACTIVE)

        # ── Pragmatic risk: deviation of J_I from target ──────────────
        if is_I.any():
            J_I      = float(v[is_I].mean())
            pragmatic = (J_I - p.target_current) ** 2 * 5.0
        else:
            pragmatic = 25.0   # maximum penalty if no internals

        # ── Epistemic risk: posterior uncertainty * (1 - shield_A) ───
        # Approximate shield_A from A_density implied by role_vec
        seg      = np.clip(e.x.astype(int), 0, ns - 1)
        n_A_seg  = np.zeros(ns)
        if is_A.any():
            np.add.at(n_A_seg, seg[is_A], 1)
        max_d    = max(n_A_seg.max(), 1e-6)
        A_norm   = n_A_seg / max_d
        # Approximate A_response as tanh of normalized density
        A_approx = np.tanh(A_norm * 2.0) * self._lambda2
        shield   = float(A_approx.mean())
        post_var = 1.0 / (max(self._lambda1, 0.1) + 1e-10)
        epistemic = post_var * (1.0 - shield)

        # ── Joule risk: lambda_J * Q_I_estimate ───────────────────────
        # Approximate Q_I from collision probability of internal electrons
        if is_I.any():
            E_eff_local = mf.E_effective[seg[is_I]]
            A_loc_I     = A_approx[seg[is_I]]
            p_coll_I_est = np.clip(
                p.base_resistivity * (np.abs(E_eff_local) * 0.18 + p.phonon_temp * 0.8)
                * (1.0 - A_loc_I * self._lambda2 * 0.80), 0, 0.35)
            # Expected Joule per colliding electron
            v_post_coll  = 0.5 * p.target_current * 0.4
            Q_I_est      = float((p_coll_I_est * v_post_coll ** 2 * p.base_resistivity).mean())
        else:
            Q_I_est = 1.0   # maximum penalty
        joule_risk = p.lambda_joule_EFE * Q_I_est

        return pragmatic + epistemic + joule_risk

    # -----------------------------------------------------------------
    # Per-electron role swap estimate (fast, vectorized per electron)
    # -----------------------------------------------------------------

    def _G_delta_swap(self, i: int, new_role: int,
                      role_vec: np.ndarray,
                      G_current: float) -> float:
        """
        Estimate change in G if electron i switches to new_role.
        Returns G_new (approximated, not full recompute).
        """
        trial = role_vec.copy()
        trial[i] = new_role
        return self._estimate_G(trial)

    # -----------------------------------------------------------------
    # Lagrange multiplier update
    # -----------------------------------------------------------------

    def _update_lagrange(self) -> None:
        sys = self.sys
        self._lambda1 += self._lr * (sys.mean_current_I() - self.p.target_current)
        self._lambda1  = np.clip(self._lambda1, 0.05, 8.0)
        j = sys.joule_by_role()
        self._lambda3 += self._lr * (j["joule_I"] - 0.003)
        self._lambda3  = np.clip(self._lambda3, 0.05, 5.0)

    # -----------------------------------------------------------------
    # Fraction constraint projection
    # -----------------------------------------------------------------

    def _project_constraints(self, role_vec: np.ndarray,
                              G_per_electron: np.ndarray) -> np.ndarray:
        """
        Project role_vec onto the feasible set Omega:
            15% <= f_r <= 55% for each role r
        Electrons are reassigned greedily by G_contribution order.
        """
        p   = self.p
        n_e = p.n_electrons
        f_min = int(0.15 * n_e)
        f_max = int(0.55 * n_e)

        for r in [Role.INTERNAL, Role.SENSORY, Role.ACTIVE]:
            n_r = (role_vec == r).sum()

            if n_r < f_min:
                # Pull in electrons from other roles with smallest G penalty
                others  = np.where(role_vec != r)[0]
                # Sort by how little they contribute to their current role
                # (proxy: their G contribution estimate -- just use random
                #  among non-cooled-down electrons for simplicity)
                deficit = f_min - n_r
                chosen  = others[np.argsort(G_per_electron[others])[:deficit]]
                role_vec[chosen] = r

            elif n_r > f_max:
                # Evict electrons that contribute least to role r
                this    = np.where(role_vec == r)[0]
                excess  = n_r - f_max
                evict   = this[np.argsort(G_per_electron[this])[::-1][:excess]]
                # Assign to the role with lowest G contribution
                for idx in evict:
                    best_G = np.inf
                    best_r = Role.INTERNAL
                    for alt_r in [Role.INTERNAL, Role.SENSORY, Role.ACTIVE]:
                        if alt_r == r: continue
                        trial = role_vec.copy()
                        trial[idx] = alt_r
                        g = self._estimate_G(trial)
                        if g < best_G:
                            best_G = g
                            best_r = alt_r
                    role_vec[idx] = best_r

        return role_vec

    # -----------------------------------------------------------------
    # Main update: stochastic coordinate descent on G
    # -----------------------------------------------------------------

    def update(self, t: float) -> dict:
        """
        Hard closed-loop role assignment via coordinate descent on G.

        For K_cd iterations:
          1. Sample sample_frac electrons (respecting cooldown)
          2. For each, evaluate G under each of 3 roles
          3. Assign role that minimizes G
        Then project onto fraction constraints.
        """
        e   = self.sys.state.electrons
        p   = self.p
        n_e = p.n_electrons
        rng = self.sys.rng

        self._update_lagrange()

        role_vec    = e.role.copy()
        G_before    = self._estimate_G(role_vec)
        n_sample    = max(int(self.sample_frac * n_e), 10)
        can_switch  = (self._cooldown == 0)

        # Per-electron G contribution for constraint projection
        G_per_elec  = np.zeros(n_e)

        for _ in range(self.K_cd):
            # Sample electrons that are allowed to switch
            eligible = np.where(can_switch)[0]
            if len(eligible) == 0:
                break
            chosen = rng.choice(eligible,
                                size=min(n_sample, len(eligible)),
                                replace=False)

            for i in chosen:
                current_r = role_vec[i]
                best_G    = self._estimate_G(role_vec)
                best_r    = current_r

                for new_r in [Role.INTERNAL, Role.SENSORY, Role.ACTIVE]:
                    if new_r == current_r:
                        continue
                    g_new = self._G_delta_swap(i, new_r, role_vec, best_G)
                    if g_new < best_G:
                        best_G = g_new
                        best_r = new_r

                role_vec[i]    = best_r
                G_per_elec[i]  = best_G

        # Project onto feasible set
        role_vec = self._project_constraints(role_vec, G_per_elec)

        # Apply cooldown
        switched  = role_vec != e.role
        self._cooldown[switched]  = self.tau_switch
        self._cooldown[~switched] = np.maximum(0, self._cooldown[~switched] - 1)

        e.role = role_vec
        if switched.any():
            self.sys.notify_roles_changed()

        G_after = self._estimate_G(role_vec)
        self.last_G_improvement = G_before - G_after
        self.last_n_switched    = int(switched.sum())

        counts = self.sys.counts_by_role()
        return {
            "n_switched":      int(switched.sum()),
            "G_before":        G_before,
            "G_after":         G_after,
            "G_improvement":   self.last_G_improvement,
            "lambda1":         self._lambda1,
            "lambda2":         self._lambda2,
            "lambda3":         self._lambda3,
            **counts,
        }

    # -----------------------------------------------------------------
    @property
    def lagrange_multipliers(self) -> dict:
        return {"lambda1_flux":  self._lambda1,
                "lambda2_CI":    self._lambda2,
                "lambda3_joule": self._lambda3}

    def ontological_lagrangian(self) -> dict:
        """Lagrangian of the role assignment problem."""
        sys   = self.sys
        j     = sys.joule_by_role()
        return {
            "log_p0":     0.0,   # prior term (uniform over roles)
            "term_flux":  self._lambda1 * sys.mean_current_I(),
            "term_CI":    self._lambda2 * (1.0 - sys.shield_A()),
            "term_joule": self._lambda3 * j["joule_I"],
            "L_total":    (self._lambda1 * sys.mean_current_I()
                           + self._lambda2 * (1.0 - sys.shield_A())
                           + self._lambda3 * j["joule_I"]),
        }
