"""
config.py – All simulation parameters for DAU v2.

Every "magic number" lives here, named and explained.
Import this module everywhere instead of scattering raw floats.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Shared numerical constants
# ---------------------------------------------------------------------------

EPS = 1e-16          # Small value to avoid log(0) / division by zero
NORM_EPS = 1e-8      # Floor for probability normalisation


# ---------------------------------------------------------------------------
# Per-agent configuration (dataclass = named, typed, inspectable)
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for a single Active Inference agent."""

    name: str

    # --- State / observation / action space sizes ---
    n_states: int      # Number of hidden states |S|
    n_obs: int         # Number of possible observations |O|
    n_actions: int     # Number of available actions |A|

    # --- Labels (for readability / plotting) ---
    state_labels:  List[str] = field(default_factory=list)
    obs_labels:    List[str] = field(default_factory=list)
    action_labels: List[str] = field(default_factory=list)

    # --- Precision parameters ---
    beta: float = 1.0   # Likelihood precision β: scales log-likelihood in state inference.
                        # β > 1 → agent trusts senses more; β < 1 → relies more on prior.
    gamma: float = 2.0  # Policy precision γ: softmax temperature over -EFE.
                        # γ → 0: uniform (random) policy; γ → ∞: deterministic argmin EFE.

    # --- Planning horizon ---
    horizon: int = 1    # H: number of steps to roll out when computing EFE.
                        # Currently only H=1 is implemented (single-step look-ahead).


# ---------------------------------------------------------------------------
# Concrete agent configurations
# ---------------------------------------------------------------------------

DESIGNER_CFG = AgentConfig(
    name="Designer",
    n_states=3,
    n_obs=3,
    n_actions=4,
    state_labels=["engage_mode", "balanced_mode", "minimal_mode"],
    obs_labels=["high_engagement", "med_engagement", "low_engagement"],
    action_labels=["boost_notif", "maintain", "add_friction", "wellbeing_nudge"],
    beta=1.0,
    gamma=2.0,   # Moderate policy precision – Designer commits slowly
    horizon=1,
)

SMARTPHONE_CFG = AgentConfig(
    name="Smartphone",
    n_states=3,
    n_obs=3,
    n_actions=4,
    state_labels=["aggressive", "moderate", "calm"],
    obs_labels=["user_engaged", "user_neutral", "user_resistant"],
    action_labels=["send_notif", "suppress_notif", "show_recommendation", "add_friction"],
    beta=1.0,
    gamma=3.0,   # Higher precision – Smartphone is more decisive/optimising
    horizon=1,
)

USER_CFG = AgentConfig(
    name="User",
    n_states=3,
    n_obs=3,
    n_actions=5,
    state_labels=["distracted", "focused", "dnd"],
    obs_labels=["notif_high", "notif_low", "friction_barrier"],
    action_labels=["engage", "ignore", "change_settings", "enable_dnd", "limit_usage"],
    beta=1.0,
    # User starts with very LOW policy precision (nearly random / uncommitted)
    # and is annealed upward by the simulation loop.
    gamma=0.5,   # Initial γ_user – will be updated each step by SIM_CFG.gamma_anneal
    horizon=1,
)


# ---------------------------------------------------------------------------
# Simulation-level configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Top-level simulation settings."""

    steps: int = 200       # Total discrete timesteps T
    seed:  int = 0         # RNG seed for reproducibility

    # User policy-precision annealing schedule:
    #   γ_user(t) = gamma_user_init + (gamma_user_final - gamma_user_init) * t / (steps - 1)
    # At t=0 → 0.5 (near-uniform), at t=T-1 → 4.0 (fairly decisive)
    gamma_user_init:  float = 0.5
    gamma_user_final: float = 4.0

    # Designer influence strength: how strongly the designer's chosen action
    # biases the smartphone's B-matrix weights (0=no coupling, 1=full replacement).
    designer_influence: float = 0.3

    # Directories
    results_dir: str = "results"
    efe_plot_filename:         str = "dau_v2_efe_timeseries.png"
    decisiveness_plot_filename: str = "dau_v2_user_decisiveness.png"


SIM_CFG = SimConfig()
