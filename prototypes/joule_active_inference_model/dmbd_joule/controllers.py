"""
controllers.py  --  dmbd_joule v5
==================================
Two classical control baselines for fair comparison with Active Inference.

Conditions:
  E -- LQR (Linear Quadratic Regulator)
       Model-based optimal controller. Has FULL knowledge of the system
       dynamics (A, B matrices). Minimizes a quadratic cost on state
       deviation + control effort. This is the theoretical optimum for
       linear systems -- Active Inference should not beat it on Q_I,
       but matching it without model knowledge would be a strong result.

  F -- Q-learning (tabular epsilon-greedy)
       Model-free RL agent. Learns a policy purely from experience,
       no knowledge of dynamics. Discrete state/action space.
       Fair comparison: same number of steps as Active Inference training.
       If Active Inference matches Q-learning's asymptotic performance
       with faster convergence, that is a publishable result.

Both controllers operate at the same level as Active Inference:
they observe the mean sensory field <S> and decide how much
cancellation to apply to each source, by setting a cancellation
multiplier kappa_ctrl in [0, 2] that scales the physical cancellation.

This is a fair interface: all three methods (AI, LQR, Q-learning) see
the same observations and have the same action space.
"""

import numpy as np
from .system import CableSystem, SystemParams, Role


# ─────────────────────────────────────────────────────────────────────
# Shared observation / action interface
# ─────────────────────────────────────────────────────────────────────

def _observe(sys: CableSystem) -> np.ndarray:
    """
    Compact observation vector for all controllers.
    Dimensions: [J_I_error, S_phonon_mean, S_em_mean, S_defect_mean,
                 cancel_phonon, cancel_em, cancel_defect]  (7-dim)
    """
    mf   = sys.state.mf
    cs   = sys.cancellation_by_source()
    J_I  = sys.mean_current_I()
    return np.array([
        J_I - sys.p.target_current,   # current error
        float(mf.S_phonon.mean()),     # phonon perturbation
        float(mf.S_em.mean()),         # EM perturbation
        float(mf.S_defect.mean()),     # defect perturbation
        cs["cancel_phonon"],           # current cancellation state
        cs["cancel_em"],
        cs["cancel_defect"],
    ], dtype=float)


def _apply_control(sys: CableSystem,
                   kappa_ph: float, kappa_em: float, kappa_df: float) -> None:
    """
    Apply controller-specified cancellation by overriding kappa values
    for one step. Operates on the same physical mechanism as Active Inference.
    """
    c   = sys.state.cable
    mf  = sys.state.mf
    p   = sys.p
    ns  = p.n_segments

    max_d  = max(mf.A_density.max(), 1e-6)
    A_norm = mf.A_density / max_d if max_d > 0 else np.ones(ns) / ns

    # Phonon
    Δph = np.clip(kappa_ph, 0, 3.0) * mf.S_phonon * A_norm
    c.phonon_amplitude = np.maximum(c.phonon_amplitude - Δph, 0.0)

    # EM -- uniform projection (controller doesn't know per-source structure)
    for k in range(p.n_em_sources):
        dx    = np.arange(ns) - c.em_positions[k]
        kern  = np.exp(-0.5 * (dx / 4.0) ** 2)
        kern /= kern.sum() + 1e-10
        cancel_k = np.clip(kappa_em, 0, 3.0) * (mf.S_em * A_norm * kern).sum()
        c.em_sources[k] = max(c.em_sources[k] - cancel_k, 0.0)

    # Defect
    for d in range(p.n_defects):
        dx    = np.arange(ns) - c.defect_positions[d]
        kern  = 1.0 / (1.0 + (dx / 1.5) ** 2)
        kern /= kern.sum() + 1e-10
        cancel_d = np.clip(kappa_df, 0, 3.0) * (mf.S_defect * A_norm * kern).sum()
        c.defect_strength[d] = max(c.defect_strength[d] - cancel_d, 0.0)


# ─────────────────────────────────────────────────────────────────────
# Condition E: LQR
# ─────────────────────────────────────────────────────────────────────

class LQRController:
    """
    Linear Quadratic Regulator with full model knowledge.

    State:  x = [J_error, S_ph, S_em, S_df, c_ph, c_em, c_df]  (7-dim)
    Action: u = [Δkappa_ph, Δkappa_em, Δkappa_df]               (3-dim)

    System linearized around equilibrium:
      x_{t+1} = A x_t + B u_t  (A, B identified from dynamics)

    Cost: J = sum_t (x'Qx + u'Ru)
    Gain: K = (R + B'PB)^{-1} B'PA  (discrete Riccati)
    u_t = -K x_t

    NOTE: LQR has FULL model knowledge -- it knows A and B exactly.
    This is an unfair advantage over Active Inference, which must
    infer them from data. If AI matches LQR, it demonstrates
    model-free optimality.
    """

    def __init__(self, params: SystemParams):
        self.p       = params
        self.n_state = 7
        self.n_act   = 3
        # Cost matrices
        # Q: heavily penalize current error and EM perturbation (dominant source)
        self.Q = np.diag([20.0, 2.0, 8.0, 1.0, 0.5, 0.5, 0.5])
        self.R = np.diag([0.5, 0.3, 0.8])   # control effort cost
        # Linearized system matrices (identified from known dynamics)
        self._build_model()
        self.K  = self._solve_riccati()
        self._kappa = np.array([params.kappa_phonon,
                                params.kappa_em,
                                params.kappa_defect])

    def _build_model(self) -> None:
        """
        Build linearized A, B matrices from known system dynamics.
        A: state transition (how state evolves without control)
        B: control input matrix (how kappa changes affect state)
        """
        p    = self.p
        α    = 0.04    # mean-reversion coefficient (from equations of motion)
        γ_S  = p.gamma_S
        γ_A  = p.gamma_A
        r_ph = p.relax_phonon
        r_em = p.relax_em

        # State: [J_err, S_ph, S_em, S_df, c_ph, c_em, c_df]
        # Approximate linear dynamics around zero perturbation
        A = np.eye(self.n_state)
        A[0, 0] = 1 - α           # J_error decays toward 0
        A[0, 2] = 0.03 * γ_S      # EM perturbation feeds into current error
        A[1, 1] = 1 - γ_S + γ_S * r_ph   # phonon sensory field
        A[2, 2] = 1 - γ_S + γ_S * r_em   # EM sensory field
        A[3, 3] = 1 - γ_S * 0.5           # defect sensory field
        A[4, 4] = 1 - r_ph        # phonon cancellation decays
        A[5, 5] = 1 - r_em        # EM cancellation decays
        A[6, 6] = 0.97            # defect cancellation is quasi-static

        # B: how control (kappa increments) affects cancellation state
        B = np.zeros((self.n_state, self.n_act))
        B[4, 0] = γ_A * 0.8      # kappa_ph -> cancel_ph
        B[5, 1] = γ_A * 0.6      # kappa_em -> cancel_em
        B[6, 2] = γ_A * 0.4      # kappa_df -> cancel_df
        # Cancellation reduces sensory signal
        B[1, 0] = -γ_S * 0.5
        B[2, 1] = -γ_S * 0.4
        B[3, 2] = -γ_S * 0.3

        self.A_mat = A
        self.B_mat = B

    def _solve_riccati(self) -> np.ndarray:
        """Solve discrete-time algebraic Riccati equation iteratively."""
        A, B, Q, R = self.A_mat, self.B_mat, self.Q, self.R
        P = Q.copy()
        for _ in range(500):
            P_new = (Q + A.T @ P @ A
                     - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A)
            if np.max(np.abs(P_new - P)) < 1e-10:
                break
            P = P_new
        K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

    def act(self, obs: np.ndarray) -> tuple:
        """Compute LQR control action u = -K x."""
        u = -self.K @ obs                    # (3,) kappa increments
        self._kappa = np.clip(self._kappa + u * 0.15, 0.1, 3.0)
        return float(self._kappa[0]), float(self._kappa[1]), float(self._kappa[2])


# ─────────────────────────────────────────────────────────────────────
# Condition F: Q-learning
# ─────────────────────────────────────────────────────────────────────

class QLearningController:
    """
    Tabular epsilon-greedy Q-learning.

    State space discretized into a compact grid:
      s = (J_err_bin, S_em_bin, cancel_em_bin)  -- 3 most important dims
      Each dimension: 6 bins -> 216 states total

    Actions: 9 combinations of (kappa_em: low/mid/high) x
                                (kappa_ph: low/mid/high)
    Kappa_df held at default (near-fully cancelled already).

    No model knowledge. Learns purely from reward signal:
      r_t = -(Q_I_t)   -- maximize negative Joule heat = minimize dissipation

    Parameters tuned for 800-step online learning:
      alpha=0.25, gamma=0.95, epsilon: 1.0 -> 0.05 over first 400 steps
    """

    # Discrete kappa values per action dimension
    KAPPA_VALS = [0.6, 1.1, 1.8]   # low / mid / high

    def __init__(self, params: SystemParams, seed: int = 42):
        self.p      = params
        self.rng    = np.random.default_rng(seed + 7)
        self.alpha  = 0.25      # learning rate
        self.gamma  = 0.95      # discount
        self.eps    = 1.0       # exploration rate (annealed)
        self.eps_min = 0.05
        self.n_states  = 6 ** 3   # 216 discrete states
        self.n_actions = 9        # 3x3 kappa combinations
        self.Q_table   = np.zeros((self.n_states, self.n_actions))
        self._prev_state  = None
        self._prev_action = None
        self._step        = 0
        self._total_steps = 800   # set from outside if needed

    def _discretize(self, obs: np.ndarray) -> int:
        """Map continuous observation to discrete state index."""
        # obs = [J_err, S_ph, S_em, S_df, c_ph, c_em, c_df]
        j_bin = int(np.clip((obs[0] + 0.5) / 1.0 * 6, 0, 5))   # J_error
        e_bin = int(np.clip(obs[2] / 0.4 * 6, 0, 5))            # S_em
        c_bin = int(np.clip(obs[5] * 6, 0, 5))                   # cancel_em
        return j_bin * 36 + e_bin * 6 + c_bin

    def _action_to_kappa(self, action: int) -> tuple:
        """Map action index to (kappa_ph, kappa_em) pair."""
        ph_idx = action // 3
        em_idx = action  % 3
        return (self.KAPPA_VALS[ph_idx],
                self.KAPPA_VALS[em_idx],
                self.p.kappa_defect)   # defect fixed

    def act(self, obs: np.ndarray, reward: float = 0.0) -> tuple:
        """Epsilon-greedy action selection + Q-table update."""
        self._step += 1
        # Anneal epsilon linearly over first half of training
        self.eps = max(self.eps_min,
                       1.0 - (1.0 - self.eps_min) * self._step / (self._total_steps * 0.5))

        state = self._discretize(obs)

        # Q-update from previous step
        if self._prev_state is not None:
            s0, a0 = self._prev_state, self._prev_action
            td_target = reward + self.gamma * np.max(self.Q_table[state])
            self.Q_table[s0, a0] += self.alpha * (td_target - self.Q_table[s0, a0])

        # Epsilon-greedy action selection
        if self.rng.random() < self.eps:
            action = int(self.rng.integers(0, self.n_actions))
        else:
            action = int(np.argmax(self.Q_table[state]))

        self._prev_state  = state
        self._prev_action = action
        return self._action_to_kappa(action)
