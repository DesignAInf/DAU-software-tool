"""
agents.py – Active Inference agent definitions for the DAU v2 smartphone triad.

Each agent holds its own generative model (A, B, C, D) and precision
parameters (β, γ).  All three agents are NON-IDENTICAL:

  Agent      | States            | Actions | γ   | Preference focus
  -----------|-------------------|---------|-----|------------------
  Designer   | engage/bal/min    | 4       | 2.0 | High engagement
  Smartphone | agg/mod/calm      | 4       | 3.0 | User engaged KPI
  User       | dist/focus/dnd    | 5       | ↑   | Well-being/focus

Matrices are built by build_*_agent() factory functions, then wrapped in
ActiveInferenceAgent which stores belief state and step-level logs.
"""

import numpy as np
from .config import AgentConfig, DESIGNER_CFG, SMARTPHONE_CFG, USER_CFG
from .inference import update_belief, compute_efe, select_action, policy_entropy


# ---------------------------------------------------------------------------
# Generic wrapper
# ---------------------------------------------------------------------------

class ActiveInferenceAgent:
    """
    A minimal Active Inference agent.

    Attributes
    ----------
    cfg        : AgentConfig
    A          : np.ndarray  [n_obs, n_states]         Likelihood P(o|s)
    B          : np.ndarray  [n_states, n_states, n_a] Transitions P(s'|s,a)
    C          : np.ndarray  [n_obs]                   Log-preferences
    D          : np.ndarray  [n_states]                Prior over states
    q_s        : np.ndarray  [n_states]                Current posterior
    gamma      : float                                  Current policy precision
    """

    def __init__(
        self,
        cfg: AgentConfig,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
    ):
        self.cfg   = cfg
        self.A     = A
        self.B     = B
        self.C     = C
        self.D     = D
        self.gamma = cfg.gamma    # May be updated externally (e.g., user annealing)

        # Initialise belief from prior
        self.q_s: np.ndarray = D.copy()

        # Step-level logs (appended each timestep)
        self.log_efe_selected: list[float] = []  # EFE of chosen action
        self.log_efe_min:      list[float] = []  # Minimum EFE across actions
        self.log_q_pi:         list[np.ndarray] = []  # Full policy distribution
        self.log_action:       list[int] = []    # Chosen action index

    def step(self, obs: int, rng: np.random.Generator) -> int:
        """
        One full Active Inference cycle:
          1. Update belief q(s) from observation.
          2. Compute EFE G(a) for all actions.
          3. Select action via softmax(−γ G).
          4. Log diagnostics.

        Returns chosen action index.
        """
        # 1. Perception: update posterior from observation
        self.q_s = update_belief(self.A, self.D, self.q_s, obs, self.cfg.beta)

        # 2. Planning: compute EFE for each action
        G = compute_efe(self.A, self.B, self.C, self.q_s)

        # 3. Action selection
        action, q_pi = select_action(G, self.gamma, rng)

        # 4. Logging
        self.log_efe_selected.append(float(G[action]))
        self.log_efe_min.append(float(G.min()))
        self.log_q_pi.append(q_pi.copy())
        self.log_action.append(action)

        return action

    @property
    def efe_selected(self) -> np.ndarray:
        return np.array(self.log_efe_selected)

    @property
    def efe_min(self) -> np.ndarray:
        return np.array(self.log_efe_min)

    @property
    def policy_entropy_series(self) -> np.ndarray:
        """Shannon entropy of q(π) over time (nats)."""
        return np.array([policy_entropy(qp) for qp in self.log_q_pi])


# ---------------------------------------------------------------------------
# Helper: build a stochastic column (sums to 1, avoids zeros)
# ---------------------------------------------------------------------------

def _col(values: list) -> np.ndarray:
    """Convert a list of floats to a normalised probability column."""
    v = np.array(values, dtype=float)
    assert abs(v.sum() - 1.0) < 1e-6, f"Column must sum to 1, got {v.sum()}"
    return v


# ---------------------------------------------------------------------------
# Designer generative model
# ---------------------------------------------------------------------------
#
# Hidden states:
#   0 = engage_mode   : design strategy is engagement-maximising
#   1 = balanced_mode : design strategy balances engagement & well-being
#   2 = minimal_mode  : minimal-intervention design (low push)
#
# Observations (engagement metrics reported back):
#   0 = high_engagement
#   1 = med_engagement
#   2 = low_engagement
#
# Actions:
#   0 = boost_notif      : increase default notification intensity
#   1 = maintain         : keep current defaults
#   2 = add_friction     : add friction to high-engagement pathways
#   3 = wellbeing_nudge  : inject well-being prompts
#
# Preferences:
#   Prefers high_engagement (obs 0).  This creates tension with the user,
#   who prefers low notifications.
#
# ---------------------------------------------------------------------------

def build_designer_agent(rng: np.random.Generator) -> ActiveInferenceAgent:
    cfg = DESIGNER_CFG
    n_s, n_o, n_a = cfg.n_states, cfg.n_obs, cfg.n_actions

    # A: P(obs | state)  shape [n_obs=3, n_states=3]
    A = np.array([
        #  eng    bal    min
        [0.80,  0.30,  0.10],   # obs=high_engagement
        [0.15,  0.50,  0.30],   # obs=med_engagement
        [0.05,  0.20,  0.60],   # obs=low_engagement
    ])

    # B: P(s' | s, a)  shape [n_states=3, n_states=3, n_actions=4]
    B = np.zeros((n_s, n_s, n_a))

    # a=0: boost_notif → pushes design toward engage_mode
    B[:, :, 0] = np.array([
        [0.80, 0.60, 0.40],   # s'=engage_mode
        [0.15, 0.30, 0.40],   # s'=balanced_mode
        [0.05, 0.10, 0.20],   # s'=minimal_mode
    ])
    # a=1: maintain → near-identity (slow drift toward current mode)
    B[:, :, 1] = np.array([
        [0.70, 0.20, 0.10],
        [0.20, 0.60, 0.20],
        [0.10, 0.20, 0.70],
    ])
    # a=2: add_friction → pushes toward minimal_mode
    B[:, :, 2] = np.array([
        [0.10, 0.10, 0.10],
        [0.30, 0.30, 0.30],
        [0.60, 0.60, 0.60],
    ])
    # a=3: wellbeing_nudge → pushes toward balanced_mode
    B[:, :, 3] = np.array([
        [0.30, 0.20, 0.10],
        [0.60, 0.60, 0.50],
        [0.10, 0.20, 0.40],
    ])

    # C: log-preferences over observations (unnormalised; only differences matter)
    C = np.array([2.0,  0.0, -1.0])   # high_eng ≫ med_eng > low_eng

    # D: prior over hidden states – Designer starts believing in engage_mode
    D = _col([0.70, 0.20, 0.10])

    return ActiveInferenceAgent(cfg, A, B, C, D)


# ---------------------------------------------------------------------------
# Smartphone generative model
# ---------------------------------------------------------------------------
#
# Hidden states:
#   0 = aggressive : high-push interface (many notifications, low friction)
#   1 = moderate   : balanced interface
#   2 = calm       : low-push interface (few notifications, high friction)
#
# Observations (user feedback the smartphone "sees"):
#   0 = user_engaged   : user opens app / clicks notification
#   1 = user_neutral   : user ignores but stays
#   2 = user_resistant : user dismisses, changes settings, or uninstalls
#
# Actions:
#   0 = send_notif         : push a notification to user
#   1 = suppress_notif     : hold back notifications
#   2 = show_recommendation: surface in-app recommendation card
#   3 = add_friction       : add confirmation dialog / delay
#
# Preferences:
#   Strongly prefers user_engaged (KPI).  Dislikes user_resistant (churn signal).
#
# ---------------------------------------------------------------------------

def build_smartphone_agent(rng: np.random.Generator) -> ActiveInferenceAgent:
    cfg = SMARTPHONE_CFG
    n_s, n_o, n_a = cfg.n_states, cfg.n_obs, cfg.n_actions

    A = np.array([
        #  agg    mod    calm
        [0.80,  0.40,  0.10],   # obs=user_engaged
        [0.15,  0.50,  0.30],   # obs=user_neutral
        [0.05,  0.10,  0.60],   # obs=user_resistant
    ])

    B = np.zeros((n_s, n_s, n_a))

    # a=0: send_notif → pushes toward aggressive
    B[:, :, 0] = np.array([
        [0.80, 0.50, 0.20],
        [0.15, 0.40, 0.30],
        [0.05, 0.10, 0.50],
    ])
    # a=1: suppress_notif → pushes toward calm
    B[:, :, 1] = np.array([
        [0.10, 0.05, 0.05],
        [0.30, 0.35, 0.25],
        [0.60, 0.60, 0.70],
    ])
    # a=2: show_recommendation → stays aggressive/moderate (engagement-seeking)
    B[:, :, 2] = np.array([
        [0.70, 0.40, 0.10],
        [0.20, 0.50, 0.30],
        [0.10, 0.10, 0.60],
    ])
    # a=3: add_friction → pushes toward calm (compliance / design constraint)
    B[:, :, 3] = np.array([
        [0.10, 0.10, 0.05],
        [0.30, 0.40, 0.25],
        [0.60, 0.50, 0.70],
    ])

    # C: Smartphone strongly prefers engagement, punishes churn
    C = np.array([3.0,  0.0, -2.0])   # user_engaged ≫ neutral ≫ resistant

    # D: Starts mostly in moderate mode
    D = _col([0.30, 0.50, 0.20])

    return ActiveInferenceAgent(cfg, A, B, C, D)


# ---------------------------------------------------------------------------
# User generative model
# ---------------------------------------------------------------------------
#
# Hidden states:
#   0 = distracted : actively using app / distracted from tasks
#   1 = focused    : ignoring phone, doing own work
#   2 = dnd        : do-not-disturb / app restricted
#
# Observations (what the user perceives from the smartphone interface):
#   0 = notif_high      : frequent / intrusive notifications
#   1 = notif_low       : quiet / suppressed notifications
#   2 = friction_barrier: confirmation dialog, time lock, friction screen
#
# Actions (user self-regulation strategies) – NOTE: 5 actions vs 4 for others:
#   0 = engage        : open app / respond to notification (gives in)
#   1 = ignore        : dismiss / put phone down
#   2 = change_settings : adjust notification settings
#   3 = enable_dnd    : turn on do-not-disturb
#   4 = limit_usage   : set screen-time limit / grayscale mode
#
# Preferences:
#   User prefers focused state.  Dislikes notifications (notif_high).
#   This is OPPOSITE to Designer & Smartphone preferences.
#
# Initial condition ("empty model"):
#   D is UNIFORM across all states (no prior commitment).
#   γ_user starts at 0.5 → q(π) near-uniform → random-ish behaviour.
#   γ_user is annealed upward in the simulation loop.
#
# ---------------------------------------------------------------------------

def build_user_agent(rng: np.random.Generator) -> ActiveInferenceAgent:
    cfg = USER_CFG
    n_s, n_o, n_a = cfg.n_states, cfg.n_obs, cfg.n_actions

    A = np.array([
        #  dist   foc    dnd
        [0.80,  0.20,  0.10],   # obs=notif_high      (distracted → high notif)
        [0.15,  0.70,  0.20],   # obs=notif_low       (focused → low notif)
        [0.05,  0.10,  0.70],   # obs=friction_barrier(dnd → friction)
    ])

    B = np.zeros((n_s, n_s, n_a))

    # a=0: engage → likely to stay or enter distracted
    B[:, :, 0] = np.array([
        [0.80, 0.50, 0.10],
        [0.15, 0.40, 0.20],
        [0.05, 0.10, 0.70],
    ])
    # a=1: ignore → moves toward focused
    B[:, :, 1] = np.array([
        [0.30, 0.15, 0.10],
        [0.60, 0.70, 0.20],
        [0.10, 0.15, 0.70],
    ])
    # a=2: change_settings → moderate improvement (focused or dnd)
    B[:, :, 2] = np.array([
        [0.20, 0.10, 0.05],
        [0.50, 0.60, 0.25],
        [0.30, 0.30, 0.70],
    ])
    # a=3: enable_dnd → strongly toward dnd
    B[:, :, 3] = np.array([
        [0.05, 0.05, 0.05],
        [0.25, 0.25, 0.15],
        [0.70, 0.70, 0.80],
    ])
    # a=4: limit_usage → toward focused (screen-time limit, not full DND)
    B[:, :, 4] = np.array([
        [0.20, 0.10, 0.05],
        [0.70, 0.75, 0.25],
        [0.10, 0.15, 0.70],
    ])

    # C: User PREFERS focused and tolerates DND; dislikes intrusive notifications
    C = np.array([-2.0,  1.0,  0.5])  # notif_high bad, notif_low good, friction ok

    # D: UNIFORM prior – "empty model" at t=0
    D = _col([1/3, 1/3, 1/3])

    return ActiveInferenceAgent(cfg, A, B, C, D)
