"""
inference.py — Core Active Inference routines (numpy only).

Identical mathematical core to DAU v2. Stateless functions that operate
on plain numpy arrays — fully reusable across agent types and profiles.
"""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-16, None)
    return float(-np.sum(p * np.log(p)))


def normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s < 1e-16:
        return np.ones_like(v) / len(v)
    return v / s


def infer_states(obs_idx: int, A: np.ndarray, prior: np.ndarray,
                 beta: float = 1.0) -> np.ndarray:
    log_prior = np.log(np.clip(prior, 1e-16, None))
    log_like  = beta * np.log(np.clip(A[obs_idx], 1e-16, None))
    return softmax(log_prior + log_like)


def compute_efe(q_s: np.ndarray, A: np.ndarray,
                B_pi: np.ndarray, C: np.ndarray) -> float:
    q_s_next  = normalise(B_pi @ q_s)
    q_o       = normalise(A @ q_s_next)
    epistemic = -entropy(q_o)
    pragmatic = float(q_o @ C)
    return epistemic + pragmatic


def compute_all_efe(q_s: np.ndarray, A: np.ndarray,
                    B_list: list, C: np.ndarray) -> np.ndarray:
    return np.array([compute_efe(q_s, A, B_pi, C) for B_pi in B_list])


def policy_posterior(G_vec: np.ndarray, gamma: float = 1.0,
                     policy_prior: np.ndarray = None) -> np.ndarray:
    log_p = gamma * G_vec
    if policy_prior is not None:
        log_p = log_p + np.log(np.clip(normalise(policy_prior), 1e-16, None))
    return softmax(log_p)


def update_dirichlet(alpha: np.ndarray, q_pi: np.ndarray,
                     lr: float = 0.05) -> np.ndarray:
    return alpha + lr * q_pi
