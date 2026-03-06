"""
env_smartphone.py — DAU v3 environment.

Generates observations for all three agents based on the world state,
which is influenced by agent actions and cross-agent couplings.

World state vector:
  - smartphone_mode  (0-9, mirrors ART_STATES)
  - user_state       (0-9, mirrors USR_STATES)
  - designer_stance  (0-9, mirrors DSG_STATES)
"""

import numpy as np
from . import config as cfg


class SmartphoneEnvironment:

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.smartphone_mode = 6   # STANDARD
        self.user_state      = 0   # FOCUSED_CALM
        self.designer_stance = 0   # GROWTH_HACKING (default)

    def reset(self):
        self.smartphone_mode = 6
        self.user_state      = 0
        self.designer_stance = 0
        return self._observe()

    def step(self, act_dsg: int, act_art: int, act_usr: int):
        p = cfg.ENV_P_STOCH
        s = cfg.DSG_TO_ART_STRENGTH
        u = cfg.ART_TO_USR_STRENGTH

        # ── Designer → smartphone coupling ────────────────────────────────
        # Designer policies 0-1 (NOTIF_LOW) push phone toward calm (5-7)
        # Designer policies 4-5 (NOTIF_HIGH) push phone toward aggressive (0-4)
        if self.rng.random() < s:
            if act_dsg in [0, 1]:     # NOTIF_LOW
                target_art = 5 if self.rng.random() < 0.6 else 7
            elif act_dsg in [2, 3]:   # NOTIF_MED
                target_art = 6
            else:                     # NOTIF_HIGH
                target_art = self.rng.integers(0, 5)
            if self.rng.random() > p:
                self.smartphone_mode = target_art
            else:
                self.smartphone_mode = int(self.rng.integers(0, cfg.ART_N_STATES))
        else:
            # Smartphone acts on its own logic
            target_art = act_art % cfg.ART_N_STATES
            if self.rng.random() > p:
                self.smartphone_mode = target_art
            else:
                self.smartphone_mode = int(self.rng.integers(0, cfg.ART_N_STATES))

        # ── Smartphone → user coupling ────────────────────────────────────
        # Aggressive modes (0-4) → push user toward stressed/habituated states
        # Calm modes (5-9)       → allow user to stay focused/relaxed
        if self.rng.random() < u:
            mode = self.smartphone_mode
            if mode in [0, 1]:    # BOMBARDMENT, PEAK_TARGETING
                target_usr = int(self.rng.choice([4, 5, 6, 8], p=[0.3, 0.3, 0.25, 0.15]))
            elif mode in [2, 3]:  # SOCIAL_AMP, VARIABLE_REWARD
                target_usr = int(self.rng.choice([5, 6, 8, 3], p=[0.35, 0.3, 0.2, 0.15]))
            elif mode == 4:       # STREAK_PRESSURE
                target_usr = int(self.rng.choice([4, 6, 8, 1], p=[0.4, 0.3, 0.2, 0.1]))
            elif mode in [5, 7]:  # CALM, RESPECTFUL
                target_usr = int(self.rng.choice([0, 2, 7], p=[0.5, 0.3, 0.2]))
            elif mode == 8:       # WELLBEING_MODE
                target_usr = int(self.rng.choice([0, 2, 7], p=[0.4, 0.4, 0.2]))
            else:                 # STANDARD, MINIMAL
                target_usr = int(self.rng.choice([0, 3, 6], p=[0.4, 0.35, 0.25]))

            # User action moderates coupling
            if act_usr in [2, 3, 4, 5]:   # CHANGE_SETTINGS, DND, UNINSTALL, LIMIT
                target_usr = int(self.rng.choice([0, 7, 9], p=[0.5, 0.3, 0.2]))
            elif act_usr == 7:             # HABITUAL_CHECK → reinforces addiction
                target_usr = 8

            if self.rng.random() > p:
                self.user_state = target_usr
            else:
                self.user_state = int(self.rng.integers(0, cfg.USR_N_STATES))
        else:
            if self.rng.random() > p:
                self.user_state = act_usr % cfg.USR_N_STATES
            else:
                self.user_state = int(self.rng.integers(0, cfg.USR_N_STATES))

        # ── Designer stance drifts with its own action ───────────────────
        if self.rng.random() > p:
            self.designer_stance = act_dsg % cfg.DSG_N_STATES
        else:
            self.designer_stance = int(self.rng.integers(0, cfg.DSG_N_STATES))

        return self._observe()

    def _observe(self):
        """Map world state to observation indices for each agent."""
        # Designer sees aggregate engagement/retention
        obs_dsg = self._dsg_obs()
        # Smartphone sees user reaction KPIs
        obs_art = self._art_obs()
        # User sees personal experience
        obs_usr = self._usr_obs()
        return obs_dsg, obs_art, obs_usr

    def _dsg_obs(self) -> int:
        mode = self.smartphone_mode
        # States 0-4 (aggressive) → engagement high, but risk of complaints
        if mode <= 4:
            probs = [0.60, 0.10, 0.12, 0.05, 0.08, 0.05]  # eng_hi, eng_lo, ret_hi, ret_lo, complaint, regflag
        elif mode in [5, 6, 7]:
            probs = [0.35, 0.25, 0.25, 0.08, 0.04, 0.03]
        else:
            probs = [0.20, 0.40, 0.20, 0.10, 0.02, 0.08]
        probs = np.array(probs); probs /= probs.sum()
        return int(self.rng.choice(cfg.DSG_N_OBS, p=probs))

    def _art_obs(self) -> int:
        u = self.user_state
        # States 6,8 (HABITUATED, ADDICTED) → clicks and time spent
        # State 9 (DISENGAGED) → ignored, churn
        # State 7 (RESISTANT) → disabled notifications, no clicks
        if u in [6, 8]:
            probs = [0.55, 0.05, 0.50, 0.03, 0.04, 0.05, 0.20, 0.15]
        elif u == 9:
            probs = [0.05, 0.50, 0.05, 0.25, 0.05, 0.05, 0.02, 0.03]
        elif u == 7:
            probs = [0.10, 0.40, 0.10, 0.05, 0.15, 0.30, 0.05, 0.05]
        elif u in [4, 5]:
            probs = [0.30, 0.10, 0.35, 0.08, 0.10, 0.10, 0.10, 0.05]
        else:
            probs = [0.35, 0.25, 0.30, 0.05, 0.04, 0.04, 0.10, 0.07]
        probs = np.array(probs); probs /= probs.sum()
        return int(self.rng.choice(cfg.ART_N_OBS, p=probs))

    def _usr_obs(self) -> int:
        u = self.user_state
        # Each user state maps to a characteristic observation distribution
        obs_map = {
            0: [0.70, 0.05, 0.25, 0.05, 0.65, 0.05, 0.05, 0.02, 0.02, 0.70],  # FOCUSED_CALM
            1: [0.50, 0.20, 0.20, 0.15, 0.30, 0.25, 0.05, 0.08, 0.10, 0.45],  # FOCUSED_ANXIOUS
            2: [0.55, 0.05, 0.20, 0.08, 0.70, 0.05, 0.15, 0.03, 0.05, 0.65],  # IDLE_RELAXED
            3: [0.20, 0.25, 0.15, 0.20, 0.20, 0.20, 0.30, 0.20, 0.35, 0.20],  # IDLE_BORED
            4: [0.25, 0.30, 0.10, 0.25, 0.15, 0.40, 0.10, 0.10, 0.40, 0.15],  # STRESSED_OVERLOAD
            5: [0.20, 0.25, 0.15, 0.25, 0.15, 0.35, 0.35, 0.40, 0.30, 0.15],  # STRESSED_FOMO
            6: [0.25, 0.20, 0.20, 0.20, 0.30, 0.20, 0.25, 0.15, 0.35, 0.20],  # HABITUATED
            7: [0.55, 0.10, 0.35, 0.08, 0.55, 0.10, 0.10, 0.05, 0.05, 0.60],  # RESISTANT
            8: [0.15, 0.20, 0.10, 0.20, 0.20, 0.25, 0.45, 0.25, 0.50, 0.10],  # ADDICTED
            9: [0.30, 0.15, 0.10, 0.30, 0.25, 0.30, 0.05, 0.10, 0.25, 0.30],  # DISENGAGED
        }
        probs = np.array(obs_map[u]); probs /= probs.sum()
        return int(self.rng.choice(cfg.USR_N_OBS, p=probs))
