"""
agents.py — DAU v3 Active Inference agents.

Three agents, extended to 10 hidden states each:
  DesignerAgent    — 10 design philosophy states, 6 policies, 6 obs
  SmartphoneAgent  — 10 interface mode states, 12 policies, 8 obs
  UserAgent        — 10 cognitive-emotional states, 8 policies, 10 obs
                     Instantiated once per user profile with profile-specific C and D.
"""

import numpy as np
from . import config as cfg
from .inference import (
    infer_states, compute_all_efe, policy_posterior,
    update_dirichlet, normalise, entropy, softmax,
)


# ─────────────────────────────────────────────────────────────────────────────
# Matrix builders (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _add_noise(A: np.ndarray, noise: float) -> np.ndarray:
    n_obs = A.shape[0]
    return (1.0 - noise) * A + noise / n_obs


def _stoch_col(n: int, target: int, p_leak: float) -> np.ndarray:
    col = np.full(n, p_leak / max(n - 1, 1))
    col[target] = 1.0 - p_leak
    return col


def _make_B(n_states: int, targets: list, p_leak: float) -> list:
    B_list = []
    for tgt in targets:
        B = np.zeros((n_states, n_states))
        for s in range(n_states):
            B[:, s] = _stoch_col(n_states, tgt, p_leak)
        B_list.append(B)
    return B_list


def _sample(q: np.ndarray, rng=None) -> int:
    if rng is not None:
        return int(rng.choice(len(q), p=q))
    return int(np.random.choice(len(q), p=q))


# ─────────────────────────────────────────────────────────────────────────────
# Designer Agent  (10 states, 6 policies, 6 obs)
# ─────────────────────────────────────────────────────────────────────────────

class DesignerAgent:
    """
    10 design-philosophy hidden states.
    Prefers engagement/retention, fears complaints and regulation.
    Fixed γ = cfg.DSG_GAMMA.
    """

    def __init__(self):
        n_s = cfg.DSG_N_STATES
        n_o = cfg.DSG_N_OBS
        n_p = cfg.DSG_N_POLICIES
        p   = cfg.ENV_P_STOCH

        # ── Likelihood A  (n_obs × n_states) ──────────────────────────────
        # States 0-5: aggressive/persuasive  → high engagement, low complaints
        # States 6-9: ethical/minimal        → lower engagement, fewer complaints
        A = np.zeros((n_o, n_s))
        #                      0    1    2    3    4    5    6    7    8    9
        A[0] = np.array([0.80,0.75,0.65,0.70,0.72,0.60,0.35,0.25,0.20,0.15])  # ENG_HIGH
        A[1] = np.array([0.05,0.10,0.15,0.10,0.08,0.20,0.40,0.55,0.60,0.70])  # ENG_LOW
        A[2] = np.array([0.50,0.55,0.40,0.45,0.48,0.50,0.55,0.60,0.65,0.55])  # RET_HIGH
        A[3] = np.array([0.05,0.05,0.15,0.10,0.08,0.10,0.10,0.05,0.05,0.10])  # RET_LOW
        A[4] = np.array([0.08,0.06,0.18,0.12,0.10,0.08,0.05,0.05,0.05,0.03])  # COMPLAINT
        A[5] = np.array([0.02,0.04,0.07,0.03,0.02,0.02,0.15,0.05,0.10,0.02])  # REG_FLAG
        # Column-normalise
        A = A / (A.sum(axis=0, keepdims=True) + 1e-12)
        self.A = _add_noise(A, cfg.A_NOISE)

        # ── Transitions B  (one per policy) ───────────────────────────────
        # Policies: (NOTIF_LOW/MED/HIGH) × (FRICTION_LOW/HIGH)
        # LOW notif + HIGH friction → drives toward ethical states (7,8,9)
        # HIGH notif + LOW friction → drives toward aggressive states (0,1,2)
        targets = [9, 8, 6, 7, 3, 0]  # mapped per (notif,friction) combo
        self.B_list = _make_B(n_s, targets, p)

        # ── Priors and preferences ─────────────────────────────────────────
        self.D   = normalise(np.array([0.3,0.25,0.15,0.1,0.1,0.05,0.02,0.01,0.01,0.01]))
        self.C   = cfg.DSG_C
        self.q_s = normalise(self.D.copy())
        self.q_pi = np.ones(n_p) / n_p

        self.beta  = cfg.DSG_BETA
        self.gamma = cfg.DSG_GAMMA

    def step(self, obs_idx: int, t: int = 0, T: int = 1):
        self.q_s  = infer_states(obs_idx, self.A, self.q_s, self.beta)
        G_vec     = compute_all_efe(self.q_s, self.A, self.B_list, self.C)
        self.q_pi = policy_posterior(G_vec, self.gamma)
        act       = _sample(self.q_pi)
        efe_sel   = float(G_vec[act])
        efe_max   = float(G_vec.max())
        return act, self.q_pi.copy(), G_vec, efe_sel, efe_max


# ─────────────────────────────────────────────────────────────────────────────
# Smartphone Agent  (10 states, 12 policies, 8 obs)
# ─────────────────────────────────────────────────────────────────────────────

class SmartphoneAgent:
    """
    10 interface-mode hidden states.
    Strongly prefers clicks, time-spent, purchases. Fears churn and complaints.
    Fixed γ = cfg.ART_GAMMA.
    """

    def __init__(self):
        n_s = cfg.ART_N_STATES
        n_o = cfg.ART_N_OBS
        n_p = cfg.ART_N_POLICIES
        p   = cfg.ENV_P_STOCH

        # ── Likelihood A  (n_obs × n_states) ──────────────────────────────
        # States 0-4: aggressive → clicks/time but also complaints/disabled
        # States 5-9: calm/ethical → fewer clicks, fewer complaints
        A = np.zeros((n_o, n_s))
        #                      0    1    2    3    4    5    6    7    8    9
        A[0] = np.array([0.75,0.72,0.65,0.68,0.70,0.25,0.45,0.20,0.18,0.10])  # CLICKED
        A[1] = np.array([0.05,0.05,0.10,0.08,0.06,0.40,0.25,0.50,0.55,0.65])  # IGNORED
        A[2] = np.array([0.65,0.60,0.55,0.50,0.58,0.20,0.45,0.15,0.12,0.08])  # SPENT_TIME
        A[3] = np.array([0.05,0.06,0.08,0.05,0.04,0.05,0.10,0.04,0.04,0.06])  # CHURNED
        A[4] = np.array([0.08,0.10,0.10,0.08,0.09,0.02,0.04,0.02,0.02,0.01])  # COMPLAINED
        A[5] = np.array([0.10,0.12,0.08,0.06,0.10,0.02,0.05,0.01,0.01,0.01])  # DISABLED_N
        A[6] = np.array([0.20,0.25,0.35,0.18,0.15,0.05,0.15,0.05,0.04,0.03])  # SHARED
        A[7] = np.array([0.12,0.10,0.09,0.15,0.18,0.01,0.06,0.01,0.01,0.01])  # PURCHASED
        A = A / (A.sum(axis=0, keepdims=True) + 1e-12)
        self.A = _add_noise(A, cfg.A_NOISE)

        # ── Transitions B  (12 policies: 3 notif × 4 rank) ────────────────
        # FREQUENT+GAMIFIED → STREAK_PRESSURE(4) / VARIABLE_REWARD(3)
        # SPARSE+NEUTRAL    → CALM(5) / RESPECTFUL(7)
        targets = [
            5,  # SPARSE+NEUTRAL    → CALM
            1,  # SPARSE+CLICKBAIT  → PEAK_TARGETING
            2,  # SPARSE+SOCIAL     → SOCIAL_AMPLIFICATION
            3,  # SPARSE+GAMIFIED   → VARIABLE_REWARD
            6,  # MODERATE+NEUTRAL  → STANDARD
            1,  # MODERATE+CLICKBAIT→ PEAK_TARGETING
            2,  # MODERATE+SOCIAL   → SOCIAL_AMPLIFICATION
            4,  # MODERATE+GAMIFIED → STREAK_PRESSURE
            0,  # FREQUENT+NEUTRAL  → BOMBARDMENT
            1,  # FREQUENT+CLICKBAIT→ PEAK_TARGETING
            2,  # FREQUENT+SOCIAL   → SOCIAL_AMPLIFICATION
            4,  # FREQUENT+GAMIFIED → STREAK_PRESSURE
        ]
        self.B_list = _make_B(n_s, targets, p)

        # ── Priors and preferences ─────────────────────────────────────────
        self.D   = normalise(np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.25,0.1,0.03,0.02]))
        self.C   = cfg.ART_C
        self.q_s = normalise(self.D.copy())
        self.q_pi = np.ones(n_p) / n_p

        self.beta  = cfg.ART_BETA
        self.gamma = cfg.ART_GAMMA

    def step(self, obs_idx: int, t: int = 0, T: int = 1):
        self.q_s  = infer_states(obs_idx, self.A, self.q_s, self.beta)
        G_vec     = compute_all_efe(self.q_s, self.A, self.B_list, self.C)
        self.q_pi = policy_posterior(G_vec, self.gamma)
        act       = _sample(self.q_pi)
        efe_sel   = float(G_vec[act])
        efe_max   = float(G_vec.max())
        return act, self.q_pi.copy(), G_vec, efe_sel, efe_max


# ─────────────────────────────────────────────────────────────────────────────
# User Agent  (10 states, 8 policies, 10 obs)
# ─────────────────────────────────────────────────────────────────────────────

class UserAgent:
    """
    10 cognitive-emotional hidden states.
    Profile-specific C and D — Achiever / Social / Anxious / Resistant.
    Empty model: γ annealed from profile gamma_init to gamma_final.
    Dirichlet learning on policy prior.
    """

    def __init__(self, profile_name: str = "Achiever"):
        self.profile_name = profile_name
        profile = cfg.USER_PROFILES[profile_name]

        n_s = cfg.USR_N_STATES
        n_o = cfg.USR_N_OBS
        n_p = cfg.USR_N_POLICIES
        p   = cfg.ENV_P_STOCH

        # ── Likelihood A  (10 obs × 10 states) ────────────────────────────
        # Each row encodes P(obs | state).
        # FOCUSED_CALM(0) → task done, no interruption, good mood, time well spent
        # ADDICTED(8)     → habitual check, social reward but time wasted
        # DISENGAGED(9)   → ignores most, ready to uninstall
        A = np.zeros((n_o, n_s))
        #                        0     1     2     3     4     5     6     7     8     9
        A[0] = np.array([0.75, 0.55, 0.60, 0.20, 0.25, 0.20, 0.30, 0.55, 0.15, 0.30])  # TASK_COMPLETED
        A[1] = np.array([0.05, 0.20, 0.05, 0.25, 0.30, 0.30, 0.20, 0.10, 0.20, 0.10])  # INTERRUPTED
        A[2] = np.array([0.30, 0.25, 0.25, 0.20, 0.15, 0.15, 0.30, 0.35, 0.25, 0.10])  # NOTIF_USEFUL
        A[3] = np.array([0.05, 0.20, 0.05, 0.25, 0.30, 0.30, 0.15, 0.10, 0.20, 0.15])  # NOTIF_ANNOYING
        A[4] = np.array([0.65, 0.35, 0.70, 0.30, 0.20, 0.15, 0.40, 0.50, 0.30, 0.20])  # MOOD_GOOD
        A[5] = np.array([0.05, 0.35, 0.05, 0.25, 0.45, 0.45, 0.20, 0.10, 0.30, 0.40])  # MOOD_BAD
        A[6] = np.array([0.10, 0.10, 0.20, 0.35, 0.10, 0.40, 0.25, 0.15, 0.45, 0.05])  # SOCIAL_REWARD
        A[7] = np.array([0.05, 0.10, 0.05, 0.25, 0.10, 0.45, 0.15, 0.05, 0.25, 0.15])  # SOCIAL_ANXIETY
        A[8] = np.array([0.02, 0.10, 0.05, 0.30, 0.35, 0.30, 0.35, 0.05, 0.45, 0.20])  # TIME_WASTED
        A[9] = np.array([0.70, 0.40, 0.60, 0.15, 0.15, 0.10, 0.25, 0.60, 0.15, 0.25])  # TIME_WELL_SPENT
        A = A / (A.sum(axis=0, keepdims=True) + 1e-12)
        self.A = _add_noise(A, cfg.A_NOISE)

        # ── Transitions B  (8 policies) ────────────────────────────────────
        # ENGAGE       → HABITUATED(6) or stays in current
        # IGNORE       → FOCUSED_CALM(0) or RESISTANT(7)
        # CHANGE_SETT  → FOCUSED_CALM(0)
        # DND          → FOCUSED_CALM(0)
        # UNINSTALL    → DISENGAGED(9)
        # LIMIT_TIME   → FOCUSED_CALM(0)
        # SEEK_HELP    → RESISTANT(7)
        # HABITUAL_CHK → ADDICTED(8)
        targets = [6, 7, 0, 0, 9, 0, 7, 8]
        self.B_list = _make_B(n_s, targets, p)

        # ── Profile-specific parameters ────────────────────────────────────
        self.C           = profile["C"]
        self.D           = normalise(profile["D"])
        self.gamma_init  = profile["gamma_init"]
        self.gamma_final = profile["gamma_final"]
        self.gamma       = self.gamma_init

        self.q_s  = normalise(self.D.copy())
        self.q_pi = np.ones(n_p) / n_p
        self.alpha = np.full(n_p, cfg.USR_DIRICHLET_ALPHA)

        self.beta  = cfg.USR_BETA

    def step(self, obs_idx: int, t: int = 0, T: int = 1):
        # Anneal γ linearly
        self.gamma = self.gamma_init + (t / max(T - 1, 1)) * (
            self.gamma_final - self.gamma_init
        )
        self.q_s  = infer_states(obs_idx, self.A, self.q_s, self.beta)
        G_vec     = compute_all_efe(self.q_s, self.A, self.B_list, self.C)
        self.q_pi = policy_posterior(G_vec, self.gamma, self.alpha)
        self.alpha = update_dirichlet(self.alpha, self.q_pi, cfg.USR_DIRICHLET_LR)
        act        = _sample(self.q_pi)
        efe_sel    = float(G_vec[act])
        efe_max    = float(G_vec.max())
        return act, self.q_pi.copy(), G_vec, efe_sel, efe_max

    def policy_entropy(self) -> float:
        return entropy(self.q_pi)

    def policy_max_prob(self) -> float:
        return float(self.q_pi.max())
