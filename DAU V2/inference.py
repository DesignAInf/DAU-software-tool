"""
inference.py — Core Active Inference routines (numpy only).

Implements:
  - softmax normalisation
  - β-weighted belief update (posterior over hidden states)
  - EFE computation (risk + ambiguity, H=1 horizon)
  - γ-weighted policy posterior (softmax over -EFE)

All functions are stateless and operate on plain numpy arrays so they can
be unit-tested independently of any agent object.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) = -Σ p log p  (nats, base e)."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-16, None)
    return float(-np.sum(p * np.log(p)))


def normalise(v: np.ndarray) -> np.ndarray:
    """L1-normalise a non-negative vector to a probability distribution."""
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s < 1e-16:
        return np.ones_like(v) / len(v)
    return v / s


# ─────────────────────────────────────────────────────────────────────────────
# Belief update  (β-weighted)
# ─────────────────────────────────────────────────────────────────────────────

def infer_states(
    obs_idx: int,
    A: np.ndarray,
    prior: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Compute posterior q(s) given a scalar observation index, likelihood A,
    prior over states, and likelihood precision β.

    Equation:
        log q(s) ∝ log prior(s)  +  β · log A[obs, s]

    Parameters
    ----------
    obs_idx : int
        Index of the observed outcome.
    A : ndarray, shape (n_obs, n_states)
        Likelihood matrix  P(o | s).
    prior : ndarray, shape (n_states,)
        Prior distribution over hidden states q(s).
    beta : float
        Likelihood precision.  β=1 → standard Bayes.
        β≈0 → posterior ≈ prior (agent ignores observations).
        β>1 → likelihood amplified (hypersensitive to evidence).

    Returns
    -------
    q_s : ndarray, shape (n_states,)
        Normalised posterior over hidden states.
    """
    log_prior = np.log(np.clip(prior, 1e-16, None))
    log_like  = beta * np.log(np.clip(A[obs_idx], 1e-16, None))
    return softmax(log_prior + log_like)


# ─────────────────────────────────────────────────────────────────────────────
# Expected Free Energy  (H=1 horizon)
# ─────────────────────────────────────────────────────────────────────────────

def compute_efe(
    q_s: np.ndarray,
    A: np.ndarray,
    B_pi: np.ndarray,
    C: np.ndarray,
) -> float:
    """
    Compute the Expected Free Energy G(π) for a single policy π at H=1.

    Decomposition:
        G(π) = -Σ_o H[P(o|π)]            ← epistemic value (ambiguity)
               + Σ_o P(o|π) · C(o)        ← pragmatic value (risk / preference)

    where:
        P(s'|π) = B_pi @ q_s             (predicted next state)
        P(o|π)  = A @ P(s'|π)            (predicted observations, mean-field)
        H[p]    = -Σ p log p             (entropy = ambiguity)

    Note: pymdp convention uses G_neg = -G; here we return G directly.
    Higher G → more preferred policy (lower free energy).

    Parameters
    ----------
    q_s   : ndarray (n_states,)   — current posterior over states
    A     : ndarray (n_obs, n_states) — likelihood
    B_pi  : ndarray (n_states, n_states) — transition under policy π
    C     : ndarray (n_obs,)      — log prior preferences

    Returns
    -------
    G : float
    """
    # Predicted next state distribution
    q_s_next = B_pi @ q_s                               # (n_states,)
    q_s_next = normalise(q_s_next)

    # Predicted observation distribution
    q_o = A @ q_s_next                                  # (n_obs,)
    q_o = normalise(q_o)

    # Epistemic value: negative entropy of predicted observations
    # (low ambiguity = agent knows what it will see = high epistemic value)
    epistemic = -entropy(q_o)

    # Pragmatic value: expected log preference
    # (how much do predicted observations match what we want?)
    pragmatic = float(q_o @ C)

    return epistemic + pragmatic


def compute_all_efe(
    q_s: np.ndarray,
    A: np.ndarray,
    B_list: list,
    C: np.ndarray,
) -> np.ndarray:
    """
    Compute EFE for every policy in B_list.

    Parameters
    ----------
    B_list : list of ndarray (n_states, n_states)
        One transition matrix per policy.

    Returns
    -------
    G_vec : ndarray (n_policies,)
    """
    return np.array([compute_efe(q_s, A, B_pi, C) for B_pi in B_list])


# ─────────────────────────────────────────────────────────────────────────────
# Policy posterior  (γ-weighted softmax over EFE)
# ─────────────────────────────────────────────────────────────────────────────

def policy_posterior(
    G_vec: np.ndarray,
    gamma: float = 1.0,
    policy_prior: np.ndarray = None,
) -> np.ndarray:
    """
    Compute q(π) = softmax(γ · G(π) + log E(π))

    where E(π) is an optional Dirichlet-derived policy prior (log concentration
    parameter), and G(π) is the EFE for each policy.

    Parameters
    ----------
    G_vec        : ndarray (n_policies,)  — EFE values
    gamma        : float                  — policy precision
    policy_prior : ndarray (n_policies,) or None
        If provided, treated as log-prior over policies (e.g. from Dirichlet α).
        Allows an "empty user model" that can be updated online.

    Returns
    -------
    q_pi : ndarray (n_policies,)  — normalised policy posterior
    """
    log_p = gamma * G_vec
    if policy_prior is not None:
        log_p = log_p + np.log(np.clip(normalise(policy_prior), 1e-16, None))
    return softmax(log_p)


# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet policy prior update  (user "experience accumulation")
# ─────────────────────────────────────────────────────────────────────────────

def update_dirichlet(
    alpha: np.ndarray,
    q_pi: np.ndarray,
    lr: float = 0.05,
) -> np.ndarray:
    """
    Soft online update of Dirichlet concentration parameters.

    Δα = lr · q(π)   →   α ← α + Δα

    This gradually shifts the policy prior toward policies that are
    frequently selected, making the user's model more definite over time.

    Parameters
    ----------
    alpha : ndarray (n_policies,)  — current concentration parameters
    q_pi  : ndarray (n_policies,)  — current policy posterior
    lr    : float                  — learning rate

    Returns
    -------
    alpha_new : ndarray (n_policies,)
    """
    return alpha + lr * q_pi
