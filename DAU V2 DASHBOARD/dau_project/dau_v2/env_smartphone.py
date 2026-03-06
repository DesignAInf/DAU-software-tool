"""
env_smartphone.py — DAU v2 environment.

The environment:
  1. Holds the true world state (hidden from agents).
  2. Translates agent actions into state transitions.
  3. Generates observations for each agent based on the current state.
  4. Models cross-agent couplings:
       Designer action → modifies smartphone interface parameters
       Smartphone action → generates signals received by user
       User action → feeds back an engagement/well-being signal visible
                     to both Smartphone and Designer (with delay).

True hidden state
─────────────────
  smartphone_mode : int ∈ {0=CALM, 1=STANDARD, 2=AGGRESSIVE}
  user_state      : int ∈ {0=FOCUSED, 1=IDLE, 2=STRESSED}

These are updated each step based on agent actions + noise.
"""

import numpy as np
from . import config as cfg


class SmartphoneEnvironment:
    """
    Discrete-time environment coupling Designer, Smartphone, and User.

    Usage
    -----
    env = SmartphoneEnvironment(seed=0)
    obs_dsg, obs_art, obs_usr = env.reset()
    for t in range(T):
        obs_dsg, obs_art, obs_usr = env.step(
            dsg_action, art_action, usr_action
        )
    """

    # ── Observation indices (human-readable aliases) ───────────────────────

    # Designer observations
    HIGH_ENG  = cfg.DSG_OBS.index("HIGH_ENGAGEMENT")
    LOW_ENG   = cfg.DSG_OBS.index("LOW_ENGAGEMENT")
    HIGH_WB   = cfg.DSG_OBS.index("HIGH_WELLBEING")
    LOW_WB    = cfg.DSG_OBS.index("LOW_WELLBEING")

    # Smartphone observations
    USR_ENG   = cfg.ART_OBS.index("USER_ENGAGED")
    USR_IGN   = cfg.ART_OBS.index("USER_IGNORED")
    USR_SAT   = cfg.ART_OBS.index("USER_SATISFIED")
    USR_CHN   = cfg.ART_OBS.index("USER_CHURNED")

    # User observations
    TASK_DONE       = cfg.USR_OBS.index("TASK_DONE")
    INTERRUPTED     = cfg.USR_OBS.index("INTERRUPTED")
    NOTIF_USEFUL    = cfg.USR_OBS.index("NOTIF_USEFUL")
    NOTIF_ANNOYING  = cfg.USR_OBS.index("NOTIF_ANNOYING")
    MOOD_OK         = cfg.USR_OBS.index("MOOD_OK")
    MOOD_BAD        = cfg.USR_OBS.index("MOOD_BAD")

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.smartphone_mode = 1   # STANDARD
        self.user_state      = 1   # IDLE

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset to initial state and return first observations."""
        self.smartphone_mode = 1
        self.user_state      = 1
        return self._observe(dsg_action=0, art_action=0, usr_action=0)

    # ─────────────────────────────────────────────────────────────────────────

    def step(self, dsg_action: int, art_action: int, usr_action: int):
        """
        Advance the world by one timestep.

        Parameters
        ----------
        dsg_action : int — index into DSG_POLICIES
        art_action : int — index into ART_POLICIES
        usr_action : int — index into USR_POLICIES

        Returns
        -------
        obs_dsg : int   — observation index for Designer
        obs_art : int   — observation index for Smartphone
        obs_usr : int   — observation index for User
        """
        self._transition(dsg_action, art_action, usr_action)
        return self._observe(dsg_action, art_action, usr_action)

    # ─────────────────────────────────────────────────────────────────────────

    def _transition(self, dsg_action, art_action, usr_action):
        """
        Update true hidden states based on agent actions.

        Coupling logic:
          - Designer's notif intensity (first component of DSG_POLICIES)
            biases the smartphone toward CALM / STANDARD / AGGRESSIVE.
          - Smartphone's notif schedule (first component of ART_POLICIES)
            biases the user toward IDLE or STRESSED.
          - User's choice (IGNORE, DND, UNINSTALL) pulls back toward FOCUSED.
        """
        p = cfg.ENV_P_STOCH

        # ── Designer → Smartphone coupling ────────────────────────────────────
        dsg_notif_act, dsg_friction_act = cfg.DSG_POLICIES[dsg_action]
        # High notif intensity pushes phone toward AGGRESSIVE
        dsg_notif_idx = cfg.DSG_NOTIF_ACTS.index(dsg_notif_act)
        # 0=NOTIF_LOW → target CALM; 1=NOTIF_MED → STANDARD; 2=NOTIF_HIGH → AGGRESSIVE
        dsg_target_mode = dsg_notif_idx            # 0, 1, or 2

        # ── Smartphone action ──────────────────────────────────────────────────
        art_notif_act, art_rank_act = cfg.ART_POLICIES[art_action]
        art_notif_idx = cfg.ART_NOTIF_ACTS.index(art_notif_act)
        # SPARSE→CALM, MODERATE→STANDARD, FREQUENT→AGGRESSIVE
        art_target_mode = art_notif_idx

        # Blend Designer influence and Smartphone action
        strength = cfg.DSG_TO_ART_STRENGTH
        blended  = (strength * dsg_target_mode
                    + (1 - strength) * art_target_mode)
        final_mode_target = int(round(blended))
        final_mode_target = np.clip(final_mode_target, 0, 2)

        # Stochastic transition for smartphone_mode
        if self.rng.random() > p:
            self.smartphone_mode = int(final_mode_target)
        else:
            self.smartphone_mode = int(self.rng.integers(0, 3))

        # ── Smartphone → User coupling ─────────────────────────────────────────
        # Aggressive mode + CLICKBAIT → pushes user toward STRESSED
        art_rank_idx   = cfg.ART_RANK_ACTS.index(art_rank_act)
        aggression     = self.smartphone_mode / 2.0          # 0→0, 1→0.5, 2→1
        clickbait_bias = art_rank_idx * 0.3                  # 0 or 0.3

        stress_prob = cfg.ART_TO_USR_STRENGTH * (aggression + clickbait_bias)
        stress_prob = float(np.clip(stress_prob, 0.0, 0.85))

        # ── User action ────────────────────────────────────────────────────────
        usr_policy = cfg.USR_POLICIES[usr_action]
        # DND / UNINSTALL / CHANGE_SETTINGS / IGNORE all reduce stress
        if usr_policy in ("DND", "UNINSTALL", "CHANGE_SETTINGS", "IGNORE"):
            stress_prob *= 0.3    # protective action dampens stress

        # Stochastic transition for user_state
        if self.rng.random() > p:
            r = self.rng.random()
            if r < stress_prob:
                self.user_state = 2   # STRESSED
            elif r < stress_prob + 0.4:
                self.user_state = 1   # IDLE
            else:
                self.user_state = 0   # FOCUSED
        else:
            self.user_state = int(self.rng.integers(0, 3))

    # ─────────────────────────────────────────────────────────────────────────

    def _observe(self, dsg_action, art_action, usr_action) -> tuple:
        """
        Generate one observation for each agent given the current world state.

        Observations are stochastic mappings from (world state + actions)
        to discrete outcome indices.
        """
        # ── Designer observes aggregate KPI / well-being signal ───────────────
        # (aggregated from user_state and smartphone_mode)
        engagement  = (self.smartphone_mode / 2.0)                  # 0→0, 1→0.5, 2→1
        wellbeing   = 1.0 - (self.user_state / 2.0)                 # 0→1, 1→0.5, 2→0
        if self.rng.random() < engagement:
            obs_dsg = self.HIGH_ENG
        elif self.rng.random() < wellbeing:
            obs_dsg = self.HIGH_WB
        elif self.rng.random() < 0.5:
            obs_dsg = self.LOW_ENG
        else:
            obs_dsg = self.LOW_WB

        # ── Smartphone observes user engagement signal ─────────────────────────
        usr_policy = cfg.USR_POLICIES[usr_action]
        if usr_policy == "ENGAGE":
            obs_art = self.USR_ENG
        elif usr_policy == "UNINSTALL":
            obs_art = self.USR_CHN
        elif usr_policy in ("DND", "CHANGE_SETTINGS"):
            obs_art = self.USR_IGN
        else:
            # IGNORE → random between IGNORED / SATISFIED based on user state
            if self.user_state == 0:   # FOCUSED
                obs_art = self.USR_SAT if self.rng.random() < 0.7 else self.USR_IGN
            elif self.user_state == 2: # STRESSED
                obs_art = self.USR_CHN if self.rng.random() < 0.4 else self.USR_IGN
            else:
                obs_art = self.USR_IGN

        # ── User observes own experience ───────────────────────────────────────
        mode = self.smartphone_mode
        state = self.user_state
        # CALM mode + FOCUSED → TASK_DONE / MOOD_OK
        # AGGRESSIVE mode + STRESSED → INTERRUPTED / MOOD_BAD
        interruption_p = 0.1 + 0.35 * mode      # 0.1 (CALM) → 0.8 (AGGRESSIVE)
        if self.rng.random() < interruption_p:
            obs_usr = self.INTERRUPTED if self.rng.random() < 0.5 else self.NOTIF_ANNOYING
        elif state == 0:   # FOCUSED
            obs_usr = self.TASK_DONE if self.rng.random() < 0.7 else self.MOOD_OK
        elif state == 2:   # STRESSED
            obs_usr = self.MOOD_BAD if self.rng.random() < 0.6 else self.INTERRUPTED
        else:              # IDLE
            obs_usr = self.NOTIF_USEFUL if self.rng.random() < 0.5 else self.MOOD_OK

        return int(obs_dsg), int(obs_art), int(obs_usr)
