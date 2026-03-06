"""
env_smartphone.py – Coupling / environment for the DAU v2 smartphone triad.

Interaction model (one timestep):
──────────────────────────────────
  t-1 state
      │
      ▼
  [Designer]  observes engagement level (from Smartphone's current state)
      │ action: boost_notif / maintain / add_friction / wellbeing_nudge
      │ → modifies smartphone's B-matrix via designer_influence weight
      ▼
  [Smartphone] observes user engagement (from User's last action)
      │ action: send_notif / suppress / show_rec / add_friction
      │ → deterministically maps to a user observation
      ▼
  [User]  observes smartphone interface output
      │ action: engage / ignore / change_settings / enable_dnd / limit_usage
      │ → maps to smartphone observation for next step
      ▼
  t state updated for all agents

Coupling rules
--------------
1. Designer → Smartphone B perturbation
   Each designer action biases which smartphone action is "cheaper" (lower EFE)
   by temporarily blending the smartphone's B matrix toward a target column.

2. Smartphone action → User observation
   Deterministic mapping (with slight noise):
     send_notif         → notif_high  (0)
     suppress_notif     → notif_low   (1)
     show_recommendation→ notif_high  (0)  [treated as intrusive push]
     add_friction       → friction_barrier (2)

3. User action → Smartphone observation
     engage        → user_engaged   (0)
     ignore        → user_neutral   (1)
     change_settings→ user_neutral  (1)  [reducing but not refusing]
     enable_dnd    → user_resistant (2)
     limit_usage   → user_resistant (2)

4. Smartphone state → Designer observation
   argmax of smartphone's q_s maps to:
     aggressive → high_engagement (0)
     moderate   → med_engagement  (1)
     calm       → low_engagement  (2)
"""

import numpy as np
from .agents import ActiveInferenceAgent
from .config import SIM_CFG, EPS


# ---------------------------------------------------------------------------
# Mapping tables (hard-coded coupling logic)
# ---------------------------------------------------------------------------

# Smartphone action → User observation index
_PHONE_ACTION_TO_USER_OBS = {
    0: 0,   # send_notif        → notif_high
    1: 1,   # suppress_notif    → notif_low
    2: 0,   # show_recommendation → notif_high (intrusive)
    3: 2,   # add_friction      → friction_barrier
}

# User action → Smartphone observation index
_USER_ACTION_TO_PHONE_OBS = {
    0: 0,   # engage            → user_engaged
    1: 1,   # ignore            → user_neutral
    2: 1,   # change_settings   → user_neutral
    3: 2,   # enable_dnd        → user_resistant
    4: 2,   # limit_usage       → user_resistant
}

# Designer action → index of the smartphone action it "promotes"
# (i.e., which smartphone action becomes temporarily more attractive)
_DESIGNER_PROMOTES_PHONE_ACTION = {
    0: 0,   # boost_notif    → promotes send_notif
    1: 2,   # maintain       → promotes show_recommendation (status quo engagement)
    2: 3,   # add_friction   → promotes add_friction (phone matches designer intent)
    3: 1,   # wellbeing_nudge→ promotes suppress_notif
}


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class SmartphoneEnvironment:
    """
    Orchestrates one simulation step:
      1. Apply designer influence to smartphone's B.
      2. Step each agent in order: Designer → Smartphone → User.
      3. Route observations between agents via coupling tables.
      4. Update gamma_user according to annealing schedule.
    """

    def __init__(
        self,
        designer: ActiveInferenceAgent,
        smartphone: ActiveInferenceAgent,
        user: ActiveInferenceAgent,
        rng: np.random.Generator,
        sim_cfg=SIM_CFG,
    ):
        self.designer   = designer
        self.smartphone = smartphone
        self.user       = user
        self.rng        = rng
        self.sim_cfg    = sim_cfg

        # Track the smartphone's original B so we can restore per-step perturbation
        self._phone_B_original = smartphone.B.copy()

        # Initialise last actions (for first-step bootstrap)
        self._last_phone_action = 1   # suppress_notif (neutral start)
        self._last_user_action  = 1   # ignore (neutral start)
        self._t = 0                   # current timestep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> dict:
        """
        Run one full triad timestep.

        Returns a dict of per-agent diagnostics for logging.
        """
        t = self._t

        # --- 1. Compute annealed user policy precision ---
        cfg = self.sim_cfg
        frac = t / max(cfg.steps - 1, 1)
        self.user.gamma = cfg.gamma_user_init + (cfg.gamma_user_final - cfg.gamma_user_init) * frac

        # --- 2. Designer step ---
        # Designer observes engagement derived from smartphone's current belief
        designer_obs = self._phone_state_to_designer_obs()
        designer_action = self.designer.step(designer_obs, self.rng)

        # --- 3. Apply designer influence on smartphone's B ---
        self._apply_designer_influence(designer_action)

        # --- 4. Smartphone step ---
        # Smartphone observes user engagement from user's last action
        phone_obs = _USER_ACTION_TO_PHONE_OBS[self._last_user_action]
        phone_action = self.smartphone.step(phone_obs, self.rng)

        # Restore original B (influence is transient – applied fresh each step)
        self.smartphone.B = self._phone_B_original.copy()

        # --- 5. User step ---
        # User observes the smartphone's action output
        user_obs = _PHONE_ACTION_TO_USER_OBS[phone_action]
        user_action = self.user.step(user_obs, self.rng)

        # --- 6. Update cached actions for next step ---
        self._last_phone_action = phone_action
        self._last_user_action  = user_action
        self._t += 1

        return {
            "t":               t,
            "designer_obs":    designer_obs,
            "designer_action": designer_action,
            "phone_obs":       phone_obs,
            "phone_action":    phone_action,
            "user_obs":        user_obs,
            "user_action":     user_action,
            "gamma_user":      self.user.gamma,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _phone_state_to_designer_obs(self) -> int:
        """
        Map smartphone's current state belief to a Designer observation.
        argmax(q_s_phone):
          0 (aggressive) → 0 (high_engagement)
          1 (moderate)   → 1 (med_engagement)
          2 (calm)       → 2 (low_engagement)
        """
        return int(np.argmax(self.smartphone.q_s))

    def _apply_designer_influence(self, designer_action: int) -> None:
        """
        Blend the smartphone's B matrix toward a target column to reflect
        the designer's choice architecture nudge.

        The promoted smartphone action (see _DESIGNER_PROMOTES_PHONE_ACTION)
        gets its transition column shifted toward the "aggressive" state (index 0)
        by blending with the original column:

            B_perturbed[:, :, promoted] = (1 - α) * B_orig[:, :, promoted]
                                         + α * B_engagement

        where B_engagement is a matrix that strongly drives toward aggressive(0).
        α = sim_cfg.designer_influence (default 0.3).

        This is a simple but transparent coupling mechanism.
        """
        promoted = _DESIGNER_PROMOTES_PHONE_ACTION[designer_action]
        alpha    = self.sim_cfg.designer_influence

        # B column that maximally pushes toward the designer-promoted state
        # (here: if designer boosts engagement, phone is nudged toward aggressive)
        n_s = self.smartphone.B.shape[0]

        # Build a "strong push" column toward state 0 (aggressive) for the promoted action
        push_col = np.full((n_s, n_s), (1.0 - 0.8) / (n_s - 1))  # small baseline
        push_col[0, :] = 0.8   # high probability of landing in aggressive state

        B_new = self._phone_B_original.copy()
        B_new[:, :, promoted] = (1.0 - alpha) * B_new[:, :, promoted] + alpha * push_col
        # Re-normalise each column to sum to 1 (preserve stochastic matrix property)
        for a in range(B_new.shape[2]):
            col_sums = B_new[:, :, a].sum(axis=0, keepdims=True)
            B_new[:, :, a] /= np.clip(col_sums, EPS, None)

        self.smartphone.B = B_new
