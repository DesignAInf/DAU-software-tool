"""
inference.py – Core Active Inference maths (numpy only).

Functions
---------
normalise(x)        : L1-normalise a non-negative vector (safe).
softmax(x)          : Numerically stable softmax.
log_stable(x)       : Element-wise log with EPS floor.
update_belief(A, D, obs, beta)   : Bayesian belief update → q(s).
compute_efe(A, B, C, q_s)        : Expected Free Energy per action (H=1).
select_action(G, gamma, rng)     : Softmax action selection → (action_idx, q_pi).
policy_entropy(q_pi)             : Shannon entropy of policy distribution.
"""

import numpy as np
from .config import EPS, NORM_EPS


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def normalise(x: np.ndarray) -> np.ndarray:
    """L1-normalise a non-negative vector.  Returns uniform if sum ≈ 0."""
    s = x.sum()
    if s < NORM_EPS:
        return np.ones_like(x, dtype=float) / len(x)
    return x / s


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: subtract max before exponentiation."""
    e = np.exp(x - x.max())
    return e / e.sum()


def log_stable(x: np.ndarray) -> np.ndarray:
    """Element-wise log with a small floor to avoid -inf."""
    return np.log(np.clip(x, EPS, None))


# ---------------------------------------------------------------------------
# Belief update  q(s_t | o_t)
# ---------------------------------------------------------------------------

def update_belief(
    A: np.ndarray,          # Likelihood  P(o|s)  shape [n_obs, n_states]
    D: np.ndarray,          # Prior       P(s)    shape [n_states]
    q_s: np.ndarray,        # Previous belief     shape [n_states]
    obs: int,               # Observed outcome index
    beta: float = 1.0,      # Likelihood precision β
) -> np.ndarray:
    """
    Variational Bayesian belief update (single observation, H=1).

    Derivation (mean-field):
        log q(s) ∝ log D(s) + β · log A(o|s)

    We use the *prior* D as a regulariser rather than the previous posterior,
    so the agent starts fresh each timestep from the prior – a simple but
    principled choice for the sparse observation regime here.  A recurrent
    variant would accumulate evidence; this keeps the maths transparent.

    Returns normalised posterior q(s_t).
    """
    log_q = log_stable(D) + beta * log_stable(A[obs, :])
    # Shift by max for numerical stability before exp
    log_q -= log_q.max()
    q = np.exp(log_q)
    return normalise(q)


# ---------------------------------------------------------------------------
# Expected Free Energy  G(a)   [H = 1 look-ahead]
# ---------------------------------------------------------------------------

def compute_efe(
    A: np.ndarray,   # Likelihood  P(o|s)   shape [n_obs, n_states]
    B: np.ndarray,   # Transitions P(s'|s,a) shape [n_states, n_states, n_actions]
    C: np.ndarray,   # Log-preferences over observations  shape [n_obs]
    q_s: np.ndarray, # Current state belief  shape [n_states]
) -> np.ndarray:
    """
    Compute Expected Free Energy for every action (single-step, H=1).

    Standard decomposition (Friston et al.):
        G(a) = risk(a) + ambiguity(a)

    Risk (pragmatic value – negative expected log preference):
        risk(a) = -E_{q(o'|a)}[ C(o') ]
                = -Σ_o  P(o'|a) · C(o)
        where P(o'|a) = Σ_s' A[o,s'] · q(s'|a)   (predicted observation)

    Ambiguity (epistemic value – expected entropy of the likelihood):
        ambiguity(a) = Σ_s' q(s'|a) · H[ P(o|s') ]
                     = Σ_s' q(s'|a) · (- Σ_o A[o,s'] · log A[o,s'] )

    Lower G → action is *preferred* (better risk + ambiguity trade-off).

    Returns G: np.ndarray of shape [n_actions]
    """
    n_actions = B.shape[2]

    # Pre-compute column-wise entropy of A  (shape [n_states])
    # H[A[:,s']] = -Σ_o A[o,s'] log A[o,s']
    H_A = -np.sum(A * log_stable(A), axis=0)  # [n_states]

    G = np.zeros(n_actions)
    for a in range(n_actions):
        q_s_next = B[:, :, a] @ q_s           # Predicted next state  [n_states]
        q_s_next = normalise(q_s_next)

        o_pred = A @ q_s_next                  # Predicted observation [n_obs]
        o_pred = normalise(o_pred)

        risk      = -np.dot(o_pred, C)         # Scalar – negative expected preference
        ambiguity = np.dot(q_s_next, H_A)     # Scalar – expected epistemic uncertainty

        G[a] = risk + ambiguity

    return G


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def select_action(
    G: np.ndarray,        # EFE per action  shape [n_actions]
    gamma: float,         # Policy precision γ
    rng: np.random.Generator,
) -> tuple:
    """
    Softmax over -EFE gives policy distribution q(π).
    Sample one action from q(π).

    Returns
    -------
    action : int
        Sampled action index.
    q_pi : np.ndarray  [n_actions]
        Full policy distribution (useful for logging / plotting).
    """
    q_pi = softmax(-gamma * G)          # Higher γ → more peaked around argmin G
    action = int(rng.choice(len(q_pi), p=q_pi))
    return action, q_pi


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def policy_entropy(q_pi: np.ndarray) -> float:
    """Shannon entropy of the policy distribution (nats).
    High → uncommitted; Low → decisive."""
    return float(-np.sum(q_pi * log_stable(q_pi)))
