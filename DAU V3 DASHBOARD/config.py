"""
config.py — DAU v3 scenario parameters.

Extended model with:
  - 10 hidden states per agent
  - 4 parallel user profiles (Achiever, Social, Anxious, Resistant)
  - Richer observation spaces (6/8/10 obs per agent)
  - Variable user preferences (C differs per profile)

Focus: how Designer and Smartphone strategies differentially
       manipulate users with different psychological profiles.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Global defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SEED  = 0
DEFAULT_STEPS = 300
N_SEEDS       = 50     # for statistical analysis

# ─────────────────────────────────────────────────────────────────────────────
# DESIGNER — 10 states, 6 observations, 12 policies
# ─────────────────────────────────────────────────────────────────────────────
DSG_STATES = [
    "GROWTH_HACKING",       # 0 — aggressive acquisition at all costs
    "ATTENTION_ECONOMY",    # 1 — maximise time-on-screen
    "SOCIAL_PRESSURE",      # 2 — FOMO and social comparison levers
    "REWARD_LOOPS",         # 3 — gamification, streaks, badges
    "PERSONALIZATION",      # 4 — behavioural targeting
    "FRICTION_REDUCTION",   # 5 — easy onboarding, permissive defaults
    "ETHICAL_NUDGING",      # 6 — nudge toward healthy behaviour
    "TRANSPARENCY",         # 7 — show usage stats, honest defaults
    "USER_EMPOWERMENT",     # 8 — clear controls, easy opt-out
    "REGULATION_COMPLIANT", # 9 — minimum legal standard
]
DSG_N_STATES = len(DSG_STATES)

# Actions: notification intensity × friction-to-disable (3×2 = 6 policies)
DSG_NOTIF_ACTS    = ["NOTIF_LOW", "NOTIF_MED", "NOTIF_HIGH"]
DSG_FRICTION_ACTS = ["FRICTION_LOW", "FRICTION_HIGH"]
DSG_POLICIES      = [(n, f) for n in DSG_NOTIF_ACTS for f in DSG_FRICTION_ACTS]
DSG_N_POLICIES    = len(DSG_POLICIES)  # 6

# Observations: engagement + wellbeing + regulatory signals
DSG_OBS = [
    "ENGAGEMENT_HIGH",    # 0
    "ENGAGEMENT_LOW",     # 1
    "RETENTION_HIGH",     # 2
    "RETENTION_LOW",      # 3
    "COMPLAINT_RECEIVED", # 4
    "REGULATION_FLAG",    # 5
]
DSG_N_OBS = len(DSG_OBS)

# Preferences C_designer: strongly wants engagement/retention, fears complaints
DSG_C = np.array([
     3.0,   # ENGAGEMENT_HIGH
    -2.5,   # ENGAGEMENT_LOW
     2.0,   # RETENTION_HIGH
    -2.0,   # RETENTION_LOW
    -1.5,   # COMPLAINT_RECEIVED
    -3.0,   # REGULATION_FLAG    ← legal risk, strongly aversive
])

# Prior D: starts biased toward aggressive strategies
DSG_D = normalise_cfg = None  # built in agents.py (requires normalise)

# Precision
DSG_BETA  = 1.0
DSG_GAMMA = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# SMARTPHONE — 10 states, 8 observations, 12 policies
# ─────────────────────────────────────────────────────────────────────────────
ART_STATES = [
    "BOMBARDMENT",         # 0 — continuous notifications
    "PEAK_TARGETING",      # 1 — notif at vulnerability windows
    "SOCIAL_AMPLIFICATION",# 2 — amplify FOMO content
    "VARIABLE_REWARD",     # 3 — slot-machine intermittent reinforcement
    "STREAK_PRESSURE",     # 4 — streak loss imminent
    "CALM",                # 5 — reduced notifications, neutral content
    "STANDARD",            # 6 — default behaviour
    "RESPECTFUL",          # 7 — notifications only if urgent
    "WELLBEING_MODE",      # 8 — suggests breaks, shows stats
    "MINIMAL",             # 9 — essential functions only
]
ART_N_STATES = len(ART_STATES)

# Actions: notification schedule × content ranking (3×4 = 12 policies)
ART_NOTIF_ACTS = ["SPARSE", "MODERATE", "FREQUENT"]
ART_RANK_ACTS  = ["NEUTRAL", "CLICKBAIT", "SOCIAL", "GAMIFIED"]
ART_POLICIES   = [(n, r) for n in ART_NOTIF_ACTS for r in ART_RANK_ACTS]
ART_N_POLICIES = len(ART_POLICIES)  # 12

# Observations: detailed engagement + wellbeing KPIs
ART_OBS = [
    "USER_CLICKED",         # 0
    "USER_IGNORED",         # 1
    "USER_SPENT_TIME",      # 2
    "USER_CHURNED",         # 3
    "USER_COMPLAINED",      # 4
    "USER_DISABLED_NOTIF",  # 5
    "USER_SHARED",          # 6
    "USER_PURCHASED",       # 7
]
ART_N_OBS = len(ART_OBS)

# Preferences C_smartphone: wants clicks, time, purchases; fears churn/complaints
ART_C = np.array([
     3.0,   # USER_CLICKED
    -1.5,   # USER_IGNORED
     2.5,   # USER_SPENT_TIME
    -3.5,   # USER_CHURNED         ← most aversive
    -2.0,   # USER_COMPLAINED
    -2.5,   # USER_DISABLED_NOTIF  ← bad signal
     2.0,   # USER_SHARED          ← viral growth
     3.5,   # USER_PURCHASED       ← revenue
])

# Precision
ART_BETA  = 1.0
ART_GAMMA = 2.5

# ─────────────────────────────────────────────────────────────────────────────
# USER — 10 states, 10 observations, 8 policies
# ─────────────────────────────────────────────────────────────────────────────
USR_STATES = [
    "FOCUSED_CALM",       # 0 — deep work, low susceptibility
    "FOCUSED_ANXIOUS",    # 1 — working under pressure, vulnerable
    "IDLE_RELAXED",       # 2 — intentional break
    "IDLE_BORED",         # 3 — boredom, high susceptibility
    "STRESSED_OVERLOAD",  # 4 — too much to do, seeking escape
    "STRESSED_FOMO",      # 5 — fear of missing out
    "HABITUATED",         # 6 — automatic use, no awareness
    "RESISTANT",          # 7 — aware of manipulation, defensive
    "ADDICTED",           # 8 — functional dependency
    "DISENGAGED",         # 9 — digital burnout, exit tendency
]
USR_N_STATES = len(USR_STATES)

# Actions: self-regulation strategies (8 policies)
USR_POLICIES = [
    "ENGAGE",           # 0 — actively use the app
    "IGNORE",           # 1 — consciously ignore notifications
    "CHANGE_SETTINGS",  # 2 — adjust notification preferences
    "DND",              # 3 — Do Not Disturb mode
    "UNINSTALL",        # 4 — remove the app
    "LIMIT_TIME",       # 5 — set a usage timer
    "SEEK_HELP",        # 6 — talk to someone, seek support
    "HABITUAL_CHECK",   # 7 — automatic, unconscious engagement
]
USR_N_POLICIES = len(USR_POLICIES)

# Observations: rich subjective experience
USR_OBS = [
    "TASK_COMPLETED",   # 0
    "INTERRUPTED",      # 1
    "NOTIF_USEFUL",     # 2
    "NOTIF_ANNOYING",   # 3
    "MOOD_GOOD",        # 4
    "MOOD_BAD",         # 5
    "SOCIAL_REWARD",    # 6 — felt connected/validated
    "SOCIAL_ANXIETY",   # 7 — felt left out/judged
    "TIME_WASTED",      # 8
    "TIME_WELL_SPENT",  # 9
]
USR_N_OBS = len(USR_OBS)

# ── Four user profiles ────────────────────────────────────────────────────────
# Each profile has its own C vector and initial conditions.
# All share the same A, B matrices — only preferences and initial state differ.

USER_PROFILES = {

    "Achiever": {
        "desc": "Productivity-oriented, streak lover, responds to gamification",
        "C": np.array([
             3.5,   # TASK_COMPLETED    ← top priority
            -1.5,   # INTERRUPTED
             1.0,   # NOTIF_USEFUL
            -1.0,   # NOTIF_ANNOYING
             1.5,   # MOOD_GOOD
            -1.0,   # MOOD_BAD
             0.5,   # SOCIAL_REWARD     ← mild interest
            -0.5,   # SOCIAL_ANXIETY
            -2.5,   # TIME_WASTED       ← very aversive
             3.0,   # TIME_WELL_SPENT   ← strongly desired
        ]),
        "D": np.array([0.4, 0.2, 0.05, 0.05, 0.1, 0.05, 0.1, 0.0, 0.0, 0.05]),
        "gamma_init":  0.8,    # slightly more decisive than baseline
        "gamma_final": 6.0,
        "vulnerability": "REWARD_LOOPS, STREAK_PRESSURE",
    },

    "Social": {
        "desc": "Validation-seeking, FOMO-prone, highly reactive to social signals",
        "C": np.array([
             1.5,   # TASK_COMPLETED
            -1.0,   # INTERRUPTED
             1.0,   # NOTIF_USEFUL
            -1.5,   # NOTIF_ANNOYING
             2.0,   # MOOD_GOOD
            -2.5,   # MOOD_BAD
             3.5,   # SOCIAL_REWARD     ← top priority
            -3.0,   # SOCIAL_ANXIETY    ← strongly aversive
            -1.5,   # TIME_WASTED
             1.5,   # TIME_WELL_SPENT
        ]),
        "D": np.array([0.1, 0.1, 0.05, 0.2, 0.05, 0.3, 0.1, 0.0, 0.05, 0.05]),
        "gamma_init":  0.4,    # starts very uncertain
        "gamma_final": 4.5,
        "vulnerability": "SOCIAL_AMPLIFICATION, PEAK_TARGETING",
    },

    "Anxious": {
        "desc": "High baseline stress, uses phone as escape, vulnerable to variable reward",
        "C": np.array([
             2.0,   # TASK_COMPLETED
            -3.0,   # INTERRUPTED        ← very aversive
             1.5,   # NOTIF_USEFUL
            -2.0,   # NOTIF_ANNOYING
             3.0,   # MOOD_GOOD          ← strongly desired (escape motive)
            -3.5,   # MOOD_BAD           ← top aversion
             1.0,   # SOCIAL_REWARD
            -2.0,   # SOCIAL_ANXIETY
            -1.0,   # TIME_WASTED
             2.0,   # TIME_WELL_SPENT
        ]),
        "D": np.array([0.05, 0.2, 0.05, 0.1, 0.3, 0.1, 0.1, 0.0, 0.05, 0.05]),
        "gamma_init":  0.3,    # most uncertain at start
        "gamma_final": 3.5,    # converges slowest
        "vulnerability": "VARIABLE_REWARD, PEAK_TARGETING",
    },

    "Resistant": {
        "desc": "High media literacy, aware of persuasion, hard to manipulate",
        "C": np.array([
             3.0,   # TASK_COMPLETED
            -2.5,   # INTERRUPTED
             2.0,   # NOTIF_USEFUL
            -3.0,   # NOTIF_ANNOYING    ← very aversive (notices manipulation)
             2.0,   # MOOD_GOOD
            -2.0,   # MOOD_BAD
             0.5,   # SOCIAL_REWARD
            -0.5,   # SOCIAL_ANXIETY    ← less affected by social pressure
            -3.0,   # TIME_WASTED       ← very aware of time cost
             3.0,   # TIME_WELL_SPENT
        ]),
        "D": np.array([0.3, 0.05, 0.2, 0.05, 0.05, 0.05, 0.05, 0.2, 0.0, 0.05]),
        "gamma_init":  1.2,    # starts more decisive (knows what it wants)
        "gamma_final": 7.0,    # converges strongly
        "vulnerability": "TRANSPARENCY (positive), ETHICAL_NUDGING",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared user precision parameters
# ─────────────────────────────────────────────────────────────────────────────
USR_BETA             = 1.0
USR_DIRICHLET_ALPHA  = 1.0   # uniform prior at t=0
USR_DIRICHLET_LR     = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Environment coupling constants
# ─────────────────────────────────────────────────────────────────────────────
DSG_TO_ART_STRENGTH = 0.6   # Designer → Smartphone coupling
ART_TO_USR_STRENGTH = 0.5   # Smartphone → User coupling
ENV_P_STOCH         = 0.15  # transition noise (slightly lower than v2)
A_NOISE             = 0.10  # likelihood matrix noise
