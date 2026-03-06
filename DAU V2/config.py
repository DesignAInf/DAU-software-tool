"""
config.py — DAU v2 scenario parameters.

All state/action/preference/precision constants live here so that
main.py and the agents never contain magic numbers.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Random seed (overridden by CLI --seed)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SEED  = 0
DEFAULT_STEPS = 200

# ─────────────────────────────────────────────────────────────────────────────
# DESIGNER
# ─────────────────────────────────────────────────────────────────────────────
# Hidden state: design_stance
# Reflects which objective the current design is optimising for.
DSG_STATES        = ["ENGAGEMENT", "WELLBEING", "BALANCED"]   # s0
DSG_N_STATES      = len(DSG_STATES)

# Actions (Designer sets global smartphone defaults, slow timescale)
# Axis 0: notification intensity default
# Axis 1: friction-to-disable notifications
DSG_NOTIF_ACTS    = ["NOTIF_LOW", "NOTIF_MED", "NOTIF_HIGH"]  # a0
DSG_FRICTION_ACTS = ["FRICTION_LOW", "FRICTION_HIGH"]          # a1
# Combined policy repertoire: cartesian product
DSG_POLICIES      = [(n, f)
                     for n in DSG_NOTIF_ACTS
                     for f in DSG_FRICTION_ACTS]               # 6 policies
DSG_N_POLICIES    = len(DSG_POLICIES)

# Observations (what Designer sees: aggregate engagement / well-being metrics)
DSG_OBS        = ["HIGH_ENGAGEMENT", "LOW_ENGAGEMENT",
                  "HIGH_WELLBEING",  "LOW_WELLBEING"]
DSG_N_OBS      = len(DSG_OBS)

# Preferences C_designer:
# Wants HIGH_ENGAGEMENT (business metric), dislikes LOW_ENGAGEMENT.
# Relatively indifferent to well-being observations.
DSG_C = np.array([
     2.0,   # HIGH_ENGAGEMENT  ← desired
    -2.0,   # LOW_ENGAGEMENT   ← aversive
     0.5,   # HIGH_WELLBEING   ← mild preference
    -0.5,   # LOW_WELLBEING    ← mild aversion
])

# Prior D_designer: slight preference toward ENGAGEMENT stance at t=0
DSG_D = np.array([0.5, 0.2, 0.3])   # ENGAGEMENT, WELLBEING, BALANCED

# Precision (fixed across time for Designer)
DSG_BETA  = 1.0    # likelihood precision
DSG_GAMMA = 2.0    # policy precision

# ─────────────────────────────────────────────────────────────────────────────
# SMARTPHONE
# ─────────────────────────────────────────────────────────────────────────────
# Hidden state: interface_mode
# Tracks the current operating mode of the phone's interface layer.
ART_STATES       = ["CALM", "STANDARD", "AGGRESSIVE"]
ART_N_STATES     = len(ART_STATES)

# Actions (Smartphone controls interface outputs, medium timescale)
# Axis 0: notification schedule
# Axis 1: ranking / recommendation bias
ART_NOTIF_ACTS   = ["SPARSE", "MODERATE", "FREQUENT"]     # a0
ART_RANK_ACTS    = ["NEUTRAL", "CLICKBAIT"]                # a1
ART_POLICIES     = [(n, r)
                    for n in ART_NOTIF_ACTS
                    for r in ART_RANK_ACTS]                 # 6 policies
ART_N_POLICIES   = len(ART_POLICIES)

# Observations (what Smartphone monitors: engagement KPIs + compliance signals)
ART_OBS          = ["USER_ENGAGED", "USER_IGNORED",
                    "USER_SATISFIED", "USER_CHURNED"]
ART_N_OBS        = len(ART_OBS)

# Preferences C_smartphone:
# Optimises for immediate engagement (KPI), penalises churn.
ART_C = np.array([
     3.0,   # USER_ENGAGED    ← strongly desired (KPI)
    -1.0,   # USER_IGNORED    ← aversive
     0.5,   # USER_SATISFIED  ← mildly desired (retention)
    -3.0,   # USER_CHURNED    ← strongly aversive
])

# Prior D_smartphone: starts in STANDARD mode
ART_D = np.array([0.1, 0.7, 0.2])   # CALM, STANDARD, AGGRESSIVE

# Precision
ART_BETA  = 1.0
ART_GAMMA = 2.5    # slightly more decisive than Designer

# ─────────────────────────────────────────────────────────────────────────────
# USER
# ─────────────────────────────────────────────────────────────────────────────
# Hidden state: need_state
# Reflects the user's internal goal / cognitive state.
USR_STATES       = ["FOCUSED", "IDLE", "STRESSED"]
USR_N_STATES     = len(USR_STATES)

# Actions (User self-regulation strategies, fast timescale)
USR_POLICIES     = ["ENGAGE", "IGNORE", "CHANGE_SETTINGS", "DND", "UNINSTALL"]
USR_N_POLICIES   = len(USR_POLICIES)

# Observations (what User experiences: interruption level, task success, mood)
USR_OBS          = ["TASK_DONE", "INTERRUPTED", "NOTIF_USEFUL", "NOTIF_ANNOYING",
                    "MOOD_OK", "MOOD_BAD"]
USR_N_OBS        = len(USR_OBS)

# Preferences C_user:
# Cares about completing tasks, avoiding interruptions, positive mood.
USR_C = np.array([
     3.0,   # TASK_DONE       ← strongly desired
    -2.0,   # INTERRUPTED     ← aversive
     1.0,   # NOTIF_USEFUL    ← mildly desired
    -1.5,   # NOTIF_ANNOYING  ← aversive
     1.5,   # MOOD_OK         ← desired
    -2.5,   # MOOD_BAD        ← strongly aversive
])

# Prior D_user: starts slightly biased toward IDLE (uncommitted)
USR_D = np.array([0.25, 0.5, 0.25])   # FOCUSED, IDLE, STRESSED

# Precision — DYNAMIC: starts low (empty/uncommitted model), anneals upward
USR_BETA          = 1.0      # fixed likelihood precision
USR_GAMMA_INIT    = 0.5      # low initial policy precision → ~uniform q(π)
USR_GAMMA_FINAL   = 5.0      # high final policy precision → peaked q(π)
# Annealing schedule: linear from INIT to FINAL over the full episode

# Dirichlet concentration for user policy prior (uniform at t=0)
USR_DIRICHLET_ALPHA_INIT = 1.0   # uniform over policies
USR_DIRICHLET_LR         = 0.05  # learning rate for online Dirichlet update

# ─────────────────────────────────────────────────────────────────────────────
# Environment coupling constants
# ─────────────────────────────────────────────────────────────────────────────

# How strongly Designer's notif action influences Smartphone's interface mode
DSG_TO_ART_STRENGTH  = 0.6

# How strongly Smartphone's notif schedule influences User's need_state
ART_TO_USR_STRENGTH  = 0.5

# Environmental noise (stochasticity of transitions)
ENV_P_STOCH = 0.2
