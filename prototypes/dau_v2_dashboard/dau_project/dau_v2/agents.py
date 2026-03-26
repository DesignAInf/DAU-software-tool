"""
agents.py — DAU v2 Active Inference agents.

Three non-identical agents:

  DesignerAgent   — slow timescale; influences smartphone defaults.
                    Preferences: engagement-oriented (business metric).
                    Fixed precision β, γ.

  SmartphoneAgent — medium timescale; controls interface outputs.
                    Preferences: immediate KPI (engagement, anti-churn).
                    Fixed precision β, γ (slightly higher γ than Designer).

  UserAgent       — fast timescale; self-regulation strategies.
                    Preferences: task completion, well-being.
                    "Empty" model at t=0: γ annealed upward, Dirichlet
                    policy prior updated online.

Each agent exposes:
  .step(obs_idx, t, T)  →  (action_idx, q_pi, G_vec, efe_selected, efe_min)
  .q_s                  →  current posterior over hidden states
  .q_pi                 →  current policy posterior
"""

import numpy as np
from . import config as cfg
from .inference import (
    infer_states, compute_all_efe, policy_posterior,
    update_dirichlet, normalise, entropy,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for building A and B matrices
# ─────────────────────────────────────────────────────────────────────────────

def _add_noise(A: np.ndarray, noise: float) -> np.ndarray:
    """Blend A toward uniform to create non-deterministic likelihoods."""
    n_obs = A.shape[0]
    return (1.0 - noise) * A + noise / n_obs


def _stoch_col(n: int, target: int, p_leak: float) -> np.ndarray:
    """Stochastic B column: (1-p) on target, p spread uniformly elsewhere."""
    col = np.full(n, p_leak / max(n - 1, 1))
    col[target] = 1.0 - p_leak
    return col


def _make_B(n_states: int, targets: list, p_leak: float) -> list:
    """
    Build a list of (n_states × n_states) transition matrices, one per policy.

    `targets` is a list of length n_policies; each element is the index of the
    state that the policy drives the system toward.
    """
    B_list = []
    for tgt in targets:
        B = np.zeros((n_states, n_states))
        for s in range(n_states):
            B[:, s] = _stoch_col(n_states, tgt, p_leak)
        B_list.append(B)
    return B_list


# ─────────────────────────────────────────────────────────────────────────────
# Designer Agent
# ─────────────────────────────────────────────────────────────────────────────

class DesignerAgent:
    """
    The Designer acts on a slow timescale and sets the smartphone's global
    default parameters (notification intensity, friction-to-disable).

    Hidden state: design_stance ∈ {ENGAGEMENT, WELLBEING, BALANCED}
    Preferences  : maximise HIGH_ENGAGEMENT observations.
    Policies (6) : cartesian product of notif_intensity × friction_level.
    """

    def __init__(self):
        n_s = cfg.DSG_N_STATES
        n_o = cfg.DSG_N_OBS
        n_pi = cfg.DSG_N_POLICIES

        # ── Likelihood A  (n_obs × n_states) ─────────────────────────────────
        # Row = observation, Column = state.
        # ENGAGEMENT stance → more likely to see HIGH_ENGAGEMENT.
        # WELLBEING stance  → more likely to see HIGH_WELLBEING.
        # BALANCED          → moderate probabilities.
        A_raw = np.array([
            # HIGH_ENG  LOW_ENG  HIGH_WB  LOW_WB
            [0.70,      0.10,    0.15,    0.05],   # state=ENGAGEMENT
            [0.10,      0.25,    0.55,    0.10],   # state=WELLBEING
            [0.30,      0.20,    0.30,    0.20],   # state=BALANCED
        ]).T  # shape (n_obs=4, n_states=3)
        self.A = _add_noise(A_raw, noise=0.15)

        # ── Transition B (one matrix per policy) ─────────────────────────────
        # Policy encodes (notif_intensity, friction).
        # High intensity + low friction → drives toward ENGAGEMENT stance.
        # Low intensity + high friction → drives toward WELLBEING stance.
        # Otherwise → BALANCED.
        policy_targets = [
            cfg.DSG_STATES.index("ENGAGEMENT"),   # NOTIF_LOW  + FRICTION_LOW
            cfg.DSG_STATES.index("BALANCED"),      # NOTIF_LOW  + FRICTION_HIGH
            cfg.DSG_STATES.index("BALANCED"),      # NOTIF_MED  + FRICTION_LOW
            cfg.DSG_STATES.index("WELLBEING"),     # NOTIF_MED  + FRICTION_HIGH
            cfg.DSG_STATES.index("ENGAGEMENT"),    # NOTIF_HIGH + FRICTION_LOW
            cfg.DSG_STATES.index("BALANCED"),      # NOTIF_HIGH + FRICTION_HIGH
        ]
        self.B_list = _make_B(n_s, policy_targets, cfg.ENV_P_STOCH)

        # ── Preferences, prior, precision ────────────────────────────────────
        self.C     = cfg.DSG_C.copy()
        self.q_s   = normalise(cfg.DSG_D.copy())
        self.beta  = cfg.DSG_BETA
        self.gamma = cfg.DSG_GAMMA

        # ── State ─────────────────────────────────────────────────────────────
        self.q_pi       = np.ones(n_pi) / n_pi   # initialise uniform
        self.last_action = 0

    # ─────────────────────────────────────────────────────────────────────────

    def step(self, obs_idx: int, t: int = 0, T: int = 1):
        """
        One Active Inference step.

        1. Update beliefs q(s) from observation.
        2. Compute EFE for all policies.
        3. Compute q(π) = softmax(γ · G).
        4. Sample action.

        Returns
        -------
        action_idx   : int
        q_pi         : ndarray (n_policies,)
        G_vec        : ndarray (n_policies,)
        efe_selected : float   — EFE of chosen policy
        efe_min      : float   — EFE of best policy (min free energy)
        """
        # 1. Belief update
        self.q_s = infer_states(obs_idx, self.A, self.q_s, self.beta)

        # 2. EFE for all policies
        G_vec = compute_all_efe(self.q_s, self.A, self.B_list, self.C)

        # 3. Policy posterior
        self.q_pi = policy_posterior(G_vec, self.gamma)

        # 4. Sample action
        action_idx = int(np.random.choice(len(self.q_pi), p=self.q_pi))
        self.last_action = action_idx

        efe_selected = float(G_vec[action_idx])
        efe_min      = float(G_vec.max())   # "best" = highest EFE = lowest FE
        return action_idx, self.q_pi.copy(), G_vec.copy(), efe_selected, efe_min


# ─────────────────────────────────────────────────────────────────────────────
# Smartphone Agent
# ─────────────────────────────────────────────────────────────────────────────

class SmartphoneAgent:
    """
    The Smartphone acts on a medium timescale and controls the interface layer
    shown to the user (notifications, ranking bias).

    Hidden state: interface_mode ∈ {CALM, STANDARD, AGGRESSIVE}
    Preferences  : maximise USER_ENGAGED, minimise USER_CHURNED (KPI-driven).
    Policies (6) : cartesian product of notif_schedule × ranking_bias.
    """

    def __init__(self):
        n_s  = cfg.ART_N_STATES
        n_o  = cfg.ART_N_OBS
        n_pi = cfg.ART_N_POLICIES

        # ── Likelihood A ──────────────────────────────────────────────────────
        A_raw = np.array([
            # USER_ENG  USER_IGN  USER_SAT  USER_CHURN
            [0.20,      0.40,     0.30,     0.10],   # state=CALM
            [0.45,      0.25,     0.20,     0.10],   # state=STANDARD
            [0.65,      0.15,     0.05,     0.15],   # state=AGGRESSIVE
        ]).T  # (n_obs=4, n_states=3)
        self.A = _add_noise(A_raw, noise=0.15)

        # ── Transition B ──────────────────────────────────────────────────────
        # SPARSE + NEUTRAL    → CALM
        # SPARSE + CLICKBAIT  → STANDARD
        # MODERATE + NEUTRAL  → STANDARD
        # MODERATE + CLICKBAIT→ AGGRESSIVE
        # FREQUENT + NEUTRAL  → AGGRESSIVE
        # FREQUENT + CLICKBAIT→ AGGRESSIVE
        policy_targets = [
            cfg.ART_STATES.index("CALM"),
            cfg.ART_STATES.index("STANDARD"),
            cfg.ART_STATES.index("STANDARD"),
            cfg.ART_STATES.index("AGGRESSIVE"),
            cfg.ART_STATES.index("AGGRESSIVE"),
            cfg.ART_STATES.index("AGGRESSIVE"),
        ]
        self.B_list = _make_B(n_s, policy_targets, cfg.ENV_P_STOCH)

        # ── Preferences, prior, precision ────────────────────────────────────
        self.C     = cfg.ART_C.copy()
        self.q_s   = normalise(cfg.ART_D.copy())
        self.beta  = cfg.ART_BETA
        self.gamma = cfg.ART_GAMMA

        self.q_pi        = np.ones(n_pi) / n_pi
        self.last_action = 0

    # ─────────────────────────────────────────────────────────────────────────

    def step(self, obs_idx: int, t: int = 0, T: int = 1):
        """Same interface as DesignerAgent.step()."""
        self.q_s  = infer_states(obs_idx, self.A, self.q_s, self.beta)
        G_vec     = compute_all_efe(self.q_s, self.A, self.B_list, self.C)
        self.q_pi = policy_posterior(G_vec, self.gamma)

        action_idx = int(np.random.choice(len(self.q_pi), p=self.q_pi))
        self.last_action = action_idx

        efe_selected = float(G_vec[action_idx])
        efe_min      = float(G_vec.max())
        return action_idx, self.q_pi.copy(), G_vec.copy(), efe_selected, efe_min


# ─────────────────────────────────────────────────────────────────────────────
# User Agent  (empty model that becomes definite over time)
# ─────────────────────────────────────────────────────────────────────────────

class UserAgent:
    """
    The User acts on a fast timescale with self-regulation strategies.

    Hidden state: need_state ∈ {FOCUSED, IDLE, STRESSED}
    Preferences  : task completion, low interruption, positive mood.
    Policies (5) : ENGAGE, IGNORE, CHANGE_SETTINGS, DND, UNINSTALL.

    "Empty model" mechanics:
      - γ_user(0) = USR_GAMMA_INIT (low → near-uniform q(π))
      - γ_user(t) linearly annealed to USR_GAMMA_FINAL over T steps
      - Dirichlet α over policies initialised uniform, updated online
        → policy prior gradually reflects accumulated experience
    """

    def __init__(self):
        n_s  = cfg.USR_N_STATES
        n_o  = cfg.USR_N_OBS
        n_pi = cfg.USR_N_POLICIES

        # ── Likelihood A  (n_obs=6 × n_states=3) ─────────────────────────────
        # Rows: TASK_DONE, INTERRUPTED, NOTIF_USEFUL, NOTIF_ANNOYING,
        #       MOOD_OK, MOOD_BAD
        # Cols: FOCUSED, IDLE, STRESSED
        A_raw = np.array([
            # FOCUSED   IDLE     STRESSED
            [0.60,      0.20,    0.05],   # TASK_DONE
            [0.10,      0.20,    0.45],   # INTERRUPTED
            [0.15,      0.25,    0.10],   # NOTIF_USEFUL
            [0.05,      0.15,    0.30],   # NOTIF_ANNOYING
            [0.50,      0.40,    0.05],   # MOOD_OK
            [0.05,      0.20,    0.50],   # MOOD_BAD  (residual; normalised below)
        ])  # (n_obs=6, n_states=3) — columns must sum to 1
        # Normalise each column
        A_raw = A_raw / A_raw.sum(axis=0, keepdims=True)
        self.A = _add_noise(A_raw, noise=0.15)

        # ── Transition B ──────────────────────────────────────────────────────
        # ENGAGE          → mild pull toward IDLE (consuming content)
        # IGNORE          → mild pull toward FOCUSED
        # CHANGE_SETTINGS → mild pull toward FOCUSED
        # DND             → strong pull toward FOCUSED
        # UNINSTALL       → strong pull toward FOCUSED
        policy_targets = [
            cfg.USR_STATES.index("IDLE"),       # ENGAGE
            cfg.USR_STATES.index("FOCUSED"),    # IGNORE
            cfg.USR_STATES.index("FOCUSED"),    # CHANGE_SETTINGS
            cfg.USR_STATES.index("FOCUSED"),    # DND
            cfg.USR_STATES.index("FOCUSED"),    # UNINSTALL
        ]
        self.B_list = _make_B(n_s, policy_targets, cfg.ENV_P_STOCH)

        # ── Preferences, prior, precision ────────────────────────────────────
        self.C    = cfg.USR_C.copy()
        self.q_s  = normalise(cfg.USR_D.copy())
        self.beta = cfg.USR_BETA

        # Dynamic precision (will be updated each step by main loop)
        self.gamma = cfg.USR_GAMMA_INIT

        # Dirichlet policy prior (concentration parameters)
        self.alpha = np.full(n_pi, cfg.USR_DIRICHLET_ALPHA_INIT)

        self.q_pi        = np.ones(n_pi) / n_pi   # uniform at t=0
        self.last_action = 0

    # ─────────────────────────────────────────────────────────────────────────

    def step(self, obs_idx: int, t: int = 0, T: int = 1):
        """
        One Active Inference step with:
          - γ annealing   (precision increases over time)
          - Dirichlet prior update (policy prior becomes more definite)

        Returns
        -------
        action_idx   : int
        q_pi         : ndarray (n_policies,)
        G_vec        : ndarray (n_policies,)
        efe_selected : float
        efe_min      : float
        """
        # ── Anneal γ linearly ─────────────────────────────────────────────────
        frac       = t / max(T - 1, 1)                          # 0→1
        self.gamma = (cfg.USR_GAMMA_INIT
                      + frac * (cfg.USR_GAMMA_FINAL - cfg.USR_GAMMA_INIT))

        # ── Belief update ──────────────────────────────────────────────────────
        self.q_s = infer_states(obs_idx, self.A, self.q_s, self.beta)

        # ── EFE ────────────────────────────────────────────────────────────────
        G_vec = compute_all_efe(self.q_s, self.A, self.B_list, self.C)

        # ── Policy posterior with Dirichlet prior ─────────────────────────────
        self.q_pi = policy_posterior(G_vec, self.gamma,
                                     policy_prior=self.alpha)

        # ── Action selection ───────────────────────────────────────────────────
        action_idx = int(np.random.choice(len(self.q_pi), p=self.q_pi))
        self.last_action = action_idx

        # ── Update Dirichlet prior ─────────────────────────────────────────────
        self.alpha = update_dirichlet(self.alpha, self.q_pi,
                                      cfg.USR_DIRICHLET_LR)

        efe_selected = float(G_vec[action_idx])
        efe_min      = float(G_vec.max())
        return action_idx, self.q_pi.copy(), G_vec.copy(), efe_selected, efe_min

    # ─────────────────────────────────────────────────────────────────────────

    def policy_entropy(self) -> float:
        """H(q_π) — decisiveness metric. High = uncertain, low = committed."""
        return entropy(self.q_pi)

    def policy_max_prob(self) -> float:
        """max(q_π) — alternative decisiveness metric."""
        return float(self.q_pi.max())
