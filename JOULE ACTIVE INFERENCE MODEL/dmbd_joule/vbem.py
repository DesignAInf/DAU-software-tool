"""
vbem.py  --  dmbd_joule v6
===========================
Variational Bayes EM inference loop -- CLOSED LOOP.

v6 architecture: EFE G genuinely drives the policy.
-------------------------------------------------------
BlanketSynthesizer.update() now runs stochastic coordinate descent
to find the role assignment omega* that minimizes G(omega). The score
functions from v5 are completely replaced by this optimization.

G = pragmatic_risk + epistemic_risk + joule_risk

  pragmatic:   (J_I - J_0)^2 * 5.0        -- current deviation
  epistemic:   post_var * (1 - shield_A)   -- uncertainty * exposure
  joule_risk:  lambda_J * Q_I_estimate     -- internal dissipation

G is estimated analytically from current mean fields (first-order
approximation) without re-running dynamics. This makes G genuinely
causal: the role assignment is determined by minimizing G, and the
subsequent physics is driven by those roles.

Two interleaved update steps per timestep:

  E-step omega:  coord descent on G  (via BlanketSynthesizer)
                 -- G drives the policy  [CLOSED LOOP]
  E-step I:      Kalman update of posterior q(I) ~ N(mu_I, 1/kappa_I)
  M-step:        Update transition model A_II, A_SI, sigma_I

The expected_free_energy() method now computes G using the ACTUAL
roles chosen by the optimizer, so G_reported is the value that was
minimized (not just tracked). r(EFE, Q_I) should remain high because
we are now genuinely minimizing G, which contains Q_I_estimate.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List
from .system  import CableSystem, Role
from .blanket import BlanketSynthesizer


@dataclass
class VBEMState:
    mu_I:    float = 0.75
    kappa_I: float = 1.0
    mu_S:    float = 0.75
    mu_A:    float = 0.0
    sigma_I: float = 0.05
    A_II:    float = 0.95
    A_SI:    float = 0.05
    elbo_history: List[float] = field(default_factory=list)


class VBEMInference:

    def __init__(self, system: CableSystem, synthesizer: BlanketSynthesizer,
                 lr: float = 0.4):
        self.sys = system
        self.syn = synthesizer
        self.p   = system.p
        self.lr  = lr
        self.v   = VBEMState()

    def _e_step_omega(self) -> dict:
        """
        Role assignment via coordinate descent on G.
        This IS the policy step -- G drives the decision.
        """
        return self.syn.update(self.sys.state.t)

    def _e_step_I(self) -> None:
        """Kalman update of posterior q(I) ~ N(mu_I, 1/kappa_I)."""
        e    = self.sys.state.electrons
        mf   = self.sys.state.mf
        v    = self.v
        is_I = (e.role == Role.INTERNAL)
        if not is_I.any():
            return
        J_obs    = float(e.vx[is_I].mean())
        J_var    = float(e.vx[is_I].var()) / max(is_I.sum(), 1)
        mu_pred  = v.A_II * v.mu_I + v.A_SI * v.mu_S
        var_pred = v.sigma_I ** 2
        K        = var_pred / (var_pred + J_var + 1e-10)
        v.mu_I    = (1 - self.lr)*v.mu_I + self.lr*(mu_pred + K*(J_obs - mu_pred))
        v.kappa_I = 1.0 / (var_pred * (1 - K) + 1e-10)
        v.mu_S    = float(mf.S_velocity.mean())
        v.mu_A    = float(mf.A_response.mean())

    def _m_step(self) -> None:
        """Update transition weights A_II, A_SI and noise sigma_I."""
        e    = self.sys.state.electrons
        v    = self.v
        is_I = (e.role == Role.INTERNAL)
        if not is_I.any():
            return
        J_cur    = float(e.vx[is_I].mean())
        residual = J_cur - (v.A_II * v.mu_I + v.A_SI * v.mu_S)
        v.A_II   = np.clip(v.A_II + self.lr*residual*v.mu_I/(v.sigma_I**2+1e-6), 0.0, 1.0)
        v.A_SI   = np.clip(v.A_SI + self.lr*residual*v.mu_S/(v.sigma_I**2+1e-6), -0.3, 0.3)
        v.sigma_I = np.clip(v.sigma_I + self.lr*(residual**2 - v.sigma_I**2), 0.01, 0.5)

    def free_energy(self) -> dict:
        """Variational free energy F = energy - entropy."""
        v  = self.v
        lm = self.syn.lagrange_multipliers
        energy  = lm["lambda1_flux"] * abs(v.mu_I - self.p.target_current)
        entropy = 0.5 * np.log(2 * np.pi * np.e / (v.kappa_I + 1e-10))
        return {
            "free_energy": float(energy - entropy),
            "energy":      float(energy),
            "entropy":     float(entropy),
        }

    def expected_free_energy(self) -> dict:
        """
        Compute EFE G with the ACTUAL roles chosen by the optimizer.

        In v6, this is NOT just a diagnostic: the optimizer minimized
        G(omega) to choose the current role assignment. The G reported
        here is the value achieved by that optimization.

        It should be lower than the G that would result from random
        or heuristic role assignment -- that is the testable prediction
        of the closed-loop architecture.
        """
        e    = self.sys.state.electrons
        is_I = (e.role == Role.INTERNAL)
        J_I  = float(e.vx[is_I].mean()) if is_I.any() else self.p.target_current

        pragmatic  = float((J_I - self.p.target_current) ** 2) * 5.0
        shield     = self.sys.shield_A()
        post_var   = 1.0 / (self.v.kappa_I + 1e-10)
        epistemic  = post_var * (1.0 - shield)

        Q_I        = self.sys.joule_by_role()["joule_I"]
        joule_risk = self.p.lambda_joule_EFE * Q_I

        G = pragmatic + epistemic + joule_risk
        return {
            "EFE":              G,
            "pragmatic":        pragmatic,
            "epistemic":        epistemic,
            "joule_risk":       joule_risk,
            "blanket_coverage": shield,
            # v6: report G improvement from this step's optimization
            "G_improvement":    self.syn.last_G_improvement,
        }

    def elbo(self) -> float:
        """Evidence Lower Bound."""
        v    = self.v
        e    = self.sys.state.electrons
        is_I = (e.role == Role.INTERNAL)
        y_obs = float(e.vx[is_I].mean()) if is_I.any() else self.p.target_current

        sigma2   = 1.0 / (v.kappa_I + 1e-10) + v.sigma_I ** 2
        log_lik  = (-0.5 * np.log(2 * np.pi * sigma2 + 1e-10)
                    - 0.5 * (y_obs - v.mu_I) ** 2 / (sigma2 + 1e-10))
        log_lik -= self.sys.joule_by_role()["joule_I"] * 14.0

        var_post = 1.0 / (v.kappa_I + 1e-10)
        kl = float(max(0.0, 0.5 * (var_post + (v.mu_I - self.p.target_current)**2
                                   - 1.0 + np.log(1.0/(var_post+1e-10)))))

        n_Bs = (e.role == Role.SENSORY).sum()
        n_Ba = (e.role == Role.ACTIVE).sum()
        blanket_complexity = max(0.0,
            (n_Bs + n_Ba) / self.p.n_electrons * self.p.blanket_strength * 0.15
            - self.sys.cancellation_effectiveness() * 0.20)

        elbo_val = float(log_lik) - kl - blanket_complexity
        v.elbo_history.append(elbo_val)
        return elbo_val

    def step(self) -> tuple:
        """One full VBEM update: omega (coord descent on G) + I + M."""
        synth_out = self._e_step_omega()
        self._e_step_I()
        self._m_step()
        fe   = self.free_energy()
        efe  = self.expected_free_energy()
        el   = self.elbo()
        ontL = self.syn.ontological_lagrangian()
        vbem_out = {**fe, **efe, "ELBO": el,
                    **{f"L_{k}": v for k, v in ontL.items()}}
        return vbem_out, synth_out
