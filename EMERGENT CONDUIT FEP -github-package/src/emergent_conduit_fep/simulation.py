#!/usr/bin/env python3
"""
Inverse Beck-Ramstead emergent conduit simulation with EFE-linked energy payback.

Key improvement:
The designer is no longer rewarded only for lowering late-stage Joule dissipation.
The designer is also penalized for failing to repay any initial energy overspend versus a
passive baseline that evolves from the same initial condition under the same exogenous noise.

This makes the blanket-shaping policy explicitly debt-aware:
- expected free energy contains expected Joule-related risk,
- policy rollouts include a shadow passive baseline,
- a positive energy debt accumulates whenever the designer spends more than the baseline,
- policies are selected to drive that debt back toward zero (break-even / payback).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from matplotlib.animation import FFMpegWriter
    HAS_FFMPEG = True
except Exception:
    HAS_FFMPEG = False

EPS = 1e-9
ROLE_COLORS = {
    "external": "#9aa0a6",
    "boundary": "#fb8c00",
    "internal": "#1f77b4",
}


@dataclass
class Config:
    seed: int = 7
    n_electrons: int = 220
    steps: int = 200
    dt: float = 0.04
    horizon: int = 14
    frame_stride: int = 7

    longitudinal_speed: float = 0.62
    transverse_noise: float = 0.045
    damping: float = 0.52
    domain_y: float = 1.20

    ext_amplitude: float = 1.00
    ext_temporal_freq: float = 0.18
    ext_spatial_freq: float = 2.1

    preferred_center: float = 0.0
    preferred_sigma: float = 0.16
    preferred_boundary_gap: float = 0.08
    preferred_boundary_width: float = 0.07
    preferred_internal_frac: float = 0.66
    preferred_boundary_frac: float = 0.22

    initial_sigma: float = 0.46
    initial_precision: float = 1.0
    attention_gain: float = 3.0
    transition_inertia: float = 1.0
    precision_gain: float = 3.4

    base_kappa: float = 0.88
    passive_kappa: float = 0.16

    # Risk terms inside expected free energy.
    joule_weight: float = 3.2
    particle_joule_weight: float = 1.35
    leakage_weight: float = 1.25
    width_weight: float = 0.95
    role_weight: float = 0.80
    ambiguity_weight: float = 0.16
    epistemic_gain: float = 0.58
    control_cost: float = 0.018
    radial_diffusion_weight: float = 0.22
    leakage_cost: float = 0.95
    sensor_noise: float = 0.10

    # New payback terms.
    debt_path_weight: float = 14.0
    debt_terminal_weight: float = 28.0
    excess_energy_weight: float = 10.0
    payback_bonus: float = 1.8

    fps: int = 16


POLICIES: List[Dict[str, float]] = [
    {"name": "hold", "dkappa": 0.00, "dsigma": 0.00, "dprecision": 0.00},
    {"name": "trim", "dkappa": 0.08, "dsigma": -0.010, "dprecision": 0.12},
    {"name": "focus", "dkappa": 0.14, "dsigma": -0.016, "dprecision": 0.22},
    {"name": "tighten", "dkappa": 0.20, "dsigma": -0.024, "dprecision": 0.30},
    {"name": "squeeze", "dkappa": 0.27, "dsigma": -0.032, "dprecision": 0.38},
    {"name": "coast", "dkappa": -0.06, "dsigma": 0.012, "dprecision": -0.08},
    {"name": "relax", "dkappa": -0.12, "dsigma": 0.022, "dprecision": -0.16},
]


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=axis, keepdims=True) + EPS
    return probs


def entropy_categorical(q: np.ndarray) -> np.ndarray:
    return -np.sum(q * np.log(q + EPS), axis=-1)


def ema(series: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    out = np.zeros_like(series, dtype=float)
    acc = float(series[0])
    for i, val in enumerate(series):
        acc = alpha * float(val) + (1.0 - alpha) * acc
        out[i] = acc
    return out


def external_force(x: np.ndarray, y: np.ndarray, t: int, cfg: Config) -> np.ndarray:
    phase = cfg.ext_temporal_freq * t
    wave = np.sin(2.0 * np.pi * (cfg.ext_spatial_freq * x - phase))
    subwave = 0.45 * np.cos(4.0 * np.pi * x + 0.20 * t)
    envelope = np.exp(-0.5 * (y / 0.92) ** 2)
    skew = 0.16 * np.sign(y) * np.exp(-np.abs(y))
    return cfg.ext_amplitude * envelope * (wave + subwave) + skew


def role_logits(
    y: np.ndarray,
    vy: np.ndarray,
    q_prev: np.ndarray,
    center: float,
    sigma: float,
    precision: float,
    cfg: Config,
) -> np.ndarray:
    d = np.abs(y - center)
    boundary_radius = sigma + cfg.preferred_boundary_gap
    boundary_width = cfg.preferred_boundary_width

    log_prior_external = np.log(0.24 + EPS) + 0.70 * d
    log_prior_boundary = (
        np.log(0.34 + EPS)
        - 0.5 * ((d - boundary_radius) / (1.20 * boundary_width + EPS)) ** 2
        + 0.22
    )
    log_prior_internal = np.log(0.42 + EPS) - 0.5 * (d / (0.92 * sigma + EPS)) ** 2

    attention = cfg.attention_gain * np.exp(
        -0.5 * ((d - boundary_radius) / (boundary_width + EPS)) ** 2
    )
    log_prior_boundary += attention

    log_like_internal = -0.5 * (d / (sigma + EPS)) ** 2 - 0.42 * (vy ** 2)
    log_like_boundary = -0.5 * ((d - boundary_radius) / (boundary_width + EPS)) ** 2 - 0.20 * (vy ** 2)
    log_like_external = -0.5 * ((np.maximum(0.0, 0.35 - d)) / 0.25) ** 2 + 0.55 * d - 0.06 * (vy ** 2)

    transition = cfg.transition_inertia * np.log(q_prev + EPS)

    logits = np.stack(
        [
            log_prior_external + precision * log_like_external + transition[:, 0],
            log_prior_boundary + precision * log_like_boundary + transition[:, 1],
            log_prior_internal + precision * log_like_internal + transition[:, 2],
        ],
        axis=1,
    )
    return logits


def infer_roles(y: np.ndarray, vy: np.ndarray, q_prev: np.ndarray, center: float, sigma: float, precision: float, cfg: Config) -> np.ndarray:
    return softmax(role_logits(y, vy, q_prev, center, sigma, precision, cfg), axis=1)


def update_macrostate(y: np.ndarray, q: np.ndarray, center_prev: float, sigma_prev: float, precision_prev: float, cfg: Config) -> Tuple[float, float, float]:
    w_internal = q[:, 2] + 0.20 * q[:, 1]
    w_internal /= np.sum(w_internal) + EPS
    center = np.sum(w_internal * y)
    variance = np.sum(w_internal * (y - center) ** 2)
    sigma = float(np.clip(np.sqrt(variance + 1e-6), 0.08, 0.60))

    mean_entropy = np.mean(entropy_categorical(q)) / np.log(3.0)
    precision = float(np.clip(1.0 + cfg.precision_gain * (1.0 - mean_entropy), 0.8, 6.0))

    center = 0.54 * center_prev + 0.16 * center + 0.30 * cfg.preferred_center
    sigma = 0.66 * sigma_prev + 0.34 * sigma
    precision = 0.60 * precision_prev + 0.40 * precision
    return center, sigma, precision


def active_field(y: np.ndarray, center: float, sigma: float, kappa: float) -> np.ndarray:
    d = y - center
    radial = np.abs(d)
    soft_pull = -0.80 * kappa * d
    outside = np.maximum(radial - sigma, 0.0)
    outside_pull = -1.20 * kappa * np.sign(d) * outside
    shell_attractor = 0.48 * kappa * np.sign(d) * (sigma - radial) * np.exp(-0.5 * ((radial - sigma) / (0.45 * sigma + 0.03)) ** 2)
    return soft_pull + outside_pull + shell_attractor


def reflect(y: np.ndarray, vy: np.ndarray, domain: float) -> Tuple[np.ndarray, np.ndarray]:
    above = y > domain
    below = y < -domain
    vy = vy.copy()
    y = y.copy()
    vy[above] *= -0.65
    vy[below] *= -0.65
    y[above] = 2 * domain - y[above]
    y[below] = -2 * domain - y[below]
    return y, vy


def compute_vfe(y: np.ndarray, vy: np.ndarray, q: np.ndarray, center: float, sigma: float, precision: float, cfg: Config) -> float:
    logits = role_logits(y, vy, q, center, sigma, precision, cfg)
    expected_energy = -np.mean(np.sum(q * logits, axis=1))
    entropy = np.mean(entropy_categorical(q))
    return float(expected_energy - entropy)


def role_fractions(q: np.ndarray) -> Tuple[float, float, float]:
    return tuple(np.mean(q, axis=0).tolist())


def coupling_proxy(ext: np.ndarray, y: np.ndarray, q: np.ndarray, center: float) -> float:
    d = y - center
    internal_signal = np.sum(q[:, 2] * d) / (np.sum(q[:, 2]) + EPS)
    boundary_signal = np.sum(q[:, 1] * np.abs(d)) / (np.sum(q[:, 1]) + EPS)
    ext_signal = np.sum(q[:, 2] * ext) / (np.sum(q[:, 2]) + EPS)
    return float(np.abs(ext_signal) / (1.0 + np.abs(boundary_signal) + np.abs(internal_signal)))


def joule_terms(vy: np.ndarray, active: np.ndarray, y: np.ndarray, center: float, q: np.ndarray, cfg: Config) -> Tuple[float, float, float, float]:
    radial_diffusion = cfg.radial_diffusion_weight * np.mean((y - center) ** 2)
    particle = cfg.damping * np.mean(vy ** 2) + radial_diffusion
    control = cfg.control_cost * np.mean(active ** 2)
    leakage = cfg.leakage_cost * np.mean(q[:, 0])
    total = particle + control
    return float(total), float(particle), float(control), float(leakage)


def rollout_step(
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    q_prev: np.ndarray,
    center: float,
    sigma: float,
    precision: float,
    kappa: float,
    t: int,
    noise: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray, np.ndarray]:
    ext = external_force(x, y, t, cfg)
    q = infer_roles(y, vy, q_prev, center, sigma, precision, cfg)
    center, sigma, precision = update_macrostate(y, q, center, sigma, precision, cfg)
    active = active_field(y, center, sigma, kappa)

    vy = vy + cfg.dt * (ext + active - cfg.damping * vy) + math.sqrt(cfg.dt) * cfg.transverse_noise * noise
    y = y + cfg.dt * vy
    y, vy = reflect(y, vy, cfg.domain_y)

    vx = 0.90 * vx + 0.10 * cfg.longitudinal_speed
    x = (x + cfg.dt * vx) % 1.0
    return x, y, vx, vy, q, center, sigma, precision, ext, active


def initial_state(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(0.0, 1.0, cfg.n_electrons, endpoint=False)
    y = rng.normal(loc=0.0, scale=0.55, size=cfg.n_electrons)
    vx = np.full(cfg.n_electrons, cfg.longitudinal_speed)
    vy = rng.normal(loc=0.0, scale=0.08, size=cfg.n_electrons)
    q = np.full((cfg.n_electrons, 3), 1.0 / 3.0)
    return x, y, vx, vy, q


def rollout_shadow_passive(
    state: Dict[str, np.ndarray | float],
    t: int,
    noise: np.ndarray,
    cfg: Config,
) -> Tuple[Dict[str, np.ndarray | float], Dict[str, float]]:
    x, y, vx, vy, q = state["x"], state["y"], state["vx"], state["vy"], state["q"]
    center, sigma, precision = float(state["center"]), float(state["sigma"]), float(state["precision"])
    x, y, vx, vy, q, center, sigma, precision, ext, active = rollout_step(
        x.copy(), y.copy(), vx.copy(), vy.copy(), q.copy(), center, sigma, precision,
        cfg.passive_kappa, t, noise, cfg
    )
    total_joule, particle_joule, control_joule, leakage = joule_terms(vy, active, y, center, q, cfg)
    new_state = {"x": x, "y": y, "vx": vx, "vy": vy, "q": q, "center": center, "sigma": sigma, "precision": precision}
    info = {
        "step_joule_total": total_joule,
        "step_joule_particle": particle_joule,
        "step_joule_control": control_joule,
        "step_leakage": leakage,
        "mean_abs_external": float(np.mean(np.abs(ext))),
    }
    return new_state, info


def efe_for_policy(
    policy: Dict[str, float],
    designer_state: Dict[str, np.ndarray | float],
    passive_state: Dict[str, np.ndarray | float],
    energy_debt: float,
    t: int,
    noise_bank: np.ndarray,
    cfg: Config,
) -> Tuple[float, Dict[str, float]]:
    # Deep-ish copies.
    ds = {
        "x": np.array(designer_state["x"], copy=True),
        "y": np.array(designer_state["y"], copy=True),
        "vx": np.array(designer_state["vx"], copy=True),
        "vy": np.array(designer_state["vy"], copy=True),
        "q": np.array(designer_state["q"], copy=True),
        "center": float(designer_state["center"]),
        "sigma": float(designer_state["sigma"]),
        "precision": float(designer_state["precision"]),
    }
    ps = {
        "x": np.array(passive_state["x"], copy=True),
        "y": np.array(passive_state["y"], copy=True),
        "vx": np.array(passive_state["vx"], copy=True),
        "vy": np.array(passive_state["vy"], copy=True),
        "q": np.array(passive_state["q"], copy=True),
        "center": float(passive_state["center"]),
        "sigma": float(passive_state["sigma"]),
        "precision": float(passive_state["precision"]),
    }

    kappa = float(np.clip(cfg.base_kappa + policy["dkappa"], 0.12, 2.60))
    sigma_target = float(np.clip(float(ds["sigma"]) + policy["dsigma"], 0.08, 0.60))
    precision_target = float(np.clip(float(ds["precision"]) + policy["dprecision"], 0.80, 6.20))

    risk = 0.0
    ambiguity = 0.0
    epistemic = 0.0
    predicted_joule = 0.0
    predicted_excess = 0.0
    debt = float(energy_debt)
    path_debt = 0.0
    prev_entropy = np.mean(entropy_categorical(np.array(ds["q"])))

    for h in range(cfg.horizon):
        noise = noise_bank[(t + h) % len(noise_bank)]

        sigma_eff = 0.72 * float(ds["sigma"]) + 0.28 * sigma_target
        precision_eff = 0.60 * float(ds["precision"]) + 0.40 * precision_target
        x, y, vx, vy, q, center, sigma, precision, ext, active = rollout_step(
            np.array(ds["x"]), np.array(ds["y"]), np.array(ds["vx"]), np.array(ds["vy"]), np.array(ds["q"]),
            float(ds["center"]), sigma_eff, precision_eff, kappa, t + h, noise, cfg
        )
        ds = {"x": x, "y": y, "vx": vx, "vy": vy, "q": q, "center": center, "sigma": sigma, "precision": precision}

        ps, p_info = rollout_shadow_passive(ps, t + h, noise, cfg)
        rf_ext, rf_bnd, rf_int = role_fractions(q)
        total_joule, particle_joule, control_joule, leakage = joule_terms(vy, active, y, center, q, cfg)

        excess = (total_joule - p_info["step_joule_total"]) * cfg.dt
        debt = max(0.0, debt + excess)
        path_debt += debt
        predicted_excess += excess
        predicted_joule += total_joule * cfg.dt

        width_error = (sigma - cfg.preferred_sigma) ** 2
        role_error = (rf_int - cfg.preferred_internal_frac) ** 2 + (rf_bnd - cfg.preferred_boundary_frac) ** 2
        risk += (
            cfg.joule_weight * total_joule
            + cfg.particle_joule_weight * particle_joule
            + cfg.leakage_weight * rf_ext
            + cfg.width_weight * width_error
            + cfg.role_weight * role_error
            + cfg.excess_energy_weight * max(excess, 0.0) / cfg.dt
        )

        sensory_noise = cfg.sensor_noise / (precision_eff + EPS)
        ambiguity += cfg.ambiguity_weight * sensory_noise
        new_entropy = np.mean(entropy_categorical(q))
        epistemic += max(prev_entropy - new_entropy, 0.0)
        prev_entropy = new_entropy

    G = (
        risk
        + ambiguity
        + cfg.debt_path_weight * path_debt
        + cfg.debt_terminal_weight * debt
        - cfg.epistemic_gain * epistemic
        - cfg.payback_bonus * max(-predicted_excess, 0.0)
    )
    return float(G), {
        "G": float(G),
        "risk": float(risk),
        "ambiguity": float(ambiguity),
        "epistemic": float(epistemic),
        "predicted_joule_horizon": float(predicted_joule),
        "predicted_excess_energy_horizon": float(predicted_excess),
        "projected_terminal_debt": float(debt),
        "projected_path_debt": float(path_debt),
        "kappa": kappa,
        "sigma_target": sigma_target,
        "precision_target": precision_target,
    }


def choose_policy(designer_state, passive_state, energy_debt, t, noise_bank, cfg: Config):
    best_policy, best_terms, best_G = None, None, np.inf
    scored = []
    for policy in POLICIES:
        G, terms = efe_for_policy(policy, designer_state, passive_state, energy_debt, t, noise_bank, cfg)
        row = {"name": policy["name"], **terms}
        scored.append(row)
        if G < best_G:
            best_G = G
            best_policy = policy
            best_terms = terms
    assert best_policy is not None and best_terms is not None
    return best_policy, best_terms, scored


def simulate(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, np.ndarray]]]:
    x0, y0, vx0, vy0, q0 = initial_state(cfg)
    noise_bank = np.random.default_rng(cfg.seed + 77).normal(size=(cfg.steps + cfg.horizon + 2, cfg.n_electrons))

    designer_state = {"x": x0.copy(), "y": y0.copy(), "vx": vx0.copy(), "vy": vy0.copy(), "q": q0.copy(), "center": 0.0, "sigma": cfg.initial_sigma, "precision": cfg.initial_precision}
    passive_state = {"x": x0.copy(), "y": y0.copy(), "vx": vx0.copy(), "vy": vy0.copy(), "q": q0.copy(), "center": 0.0, "sigma": cfg.initial_sigma, "precision": cfg.initial_precision}

    designer_history: List[Dict[str, float]] = []
    passive_history: List[Dict[str, float]] = []
    frames: List[Dict[str, np.ndarray]] = []

    cumulative_designer = 0.0
    cumulative_passive = 0.0
    cumulative_savings = 0.0
    energy_debt = 0.0
    breakeven_step = np.nan

    for t in range(cfg.steps):
        policy, terms, _ = choose_policy(designer_state, passive_state, energy_debt, t, noise_bank, cfg)
        noise = noise_bank[t]

        x, y, vx, vy, q, center, sigma, precision, ext, active = rollout_step(
            np.array(designer_state["x"]), np.array(designer_state["y"]), np.array(designer_state["vx"]), np.array(designer_state["vy"]), np.array(designer_state["q"]),
            float(designer_state["center"]), terms["sigma_target"], terms["precision_target"], terms["kappa"], t, noise, cfg
        )
        designer_state = {"x": x, "y": y, "vx": vx, "vy": vy, "q": q, "center": center, "sigma": sigma, "precision": precision}

        passive_state, p_info = rollout_shadow_passive(passive_state, t, noise, cfg)

        total_joule, particle_joule, control_joule, leakage = joule_terms(vy, active, y, center, q, cfg)
        cumulative_designer += total_joule * cfg.dt
        cumulative_passive += p_info["step_joule_total"] * cfg.dt
        delta_savings = (p_info["step_joule_total"] - total_joule) * cfg.dt
        cumulative_savings += delta_savings
        energy_debt = max(0.0, energy_debt - delta_savings)
        if np.isnan(breakeven_step) and cumulative_savings >= 0.0 and t > 0:
            breakeven_step = float(t)

        vfe = compute_vfe(y, vy, q, center, sigma, precision, cfg)
        rf_ext, rf_bnd, rf_int = role_fractions(q)
        designer_history.append({
            "step": t,
            "policy": policy["name"],
            "efe": terms["G"],
            "risk": terms["risk"],
            "ambiguity": terms["ambiguity"],
            "epistemic": terms["epistemic"],
            "predicted_joule_horizon": terms["predicted_joule_horizon"],
            "predicted_excess_energy_horizon": terms["predicted_excess_energy_horizon"],
            "projected_terminal_debt": terms["projected_terminal_debt"],
            "projected_path_debt": terms["projected_path_debt"],
            "vfe": vfe,
            "center": center,
            "sigma": sigma,
            "precision": precision,
            "kappa": terms["kappa"],
            "internal_fraction": rf_int,
            "boundary_fraction": rf_bnd,
            "external_fraction": rf_ext,
            "coupling_proxy": coupling_proxy(ext, y, q, center),
            "step_joule_total": total_joule,
            "step_joule_particle": particle_joule,
            "step_joule_control": control_joule,
            "step_leakage": leakage,
            "cumulative_joule_total": cumulative_designer,
            "mean_abs_external": float(np.mean(np.abs(ext))),
            "mean_abs_active": float(np.mean(np.abs(active))),
            "mean_abs_y": float(np.mean(np.abs(y - center))),
            "baseline_step_joule_total": p_info["step_joule_total"],
            "baseline_step_joule_particle": p_info["step_joule_particle"],
            "baseline_step_joule_control": p_info["step_joule_control"],
            "baseline_cumulative_joule_total": cumulative_passive,
            "step_joule_saving_vs_baseline": delta_savings / cfg.dt,
            "cumulative_joule_saving_vs_baseline": cumulative_savings,
            "energy_debt": energy_debt,
            "breakeven_step": breakeven_step,
        })

        passive_history.append({
            "step": t,
            "center": float(passive_state["center"]),
            "sigma": float(passive_state["sigma"]),
            "precision": float(passive_state["precision"]),
            "step_joule_total": p_info["step_joule_total"],
            "step_joule_particle": p_info["step_joule_particle"],
            "step_joule_control": p_info["step_joule_control"],
            "step_leakage": p_info["step_leakage"],
            "cumulative_joule_total": cumulative_passive,
        })

        if (t % cfg.frame_stride == 0) or (t == cfg.steps - 1):
            frames.append({
                "x": x.copy(),
                "y": y.copy(),
                "q": q.copy(),
                "center": np.array(center),
                "sigma": np.array(sigma),
                "step": np.array(float(t)),
            })

    designer_df = pd.DataFrame(designer_history)
    passive_df = pd.DataFrame(passive_history)
    designer_df["step_joule_total_ema"] = ema(designer_df["step_joule_total"].to_numpy(), alpha=0.08)
    designer_df["baseline_step_joule_total_ema"] = ema(designer_df["baseline_step_joule_total"].to_numpy(), alpha=0.08)
    designer_df["energy_debt_ema"] = ema(designer_df["energy_debt"].to_numpy(), alpha=0.08)
    designer_df["cumulative_joule_saving_ema"] = ema(designer_df["cumulative_joule_saving_vs_baseline"].to_numpy(), alpha=0.08)
    designer_df["efe_ema"] = ema(designer_df["efe"].to_numpy(), alpha=0.08)
    passive_df["step_joule_total_ema"] = ema(passive_df["step_joule_total"].to_numpy(), alpha=0.08)
    return designer_df, passive_df, frames


def save_snapshot(history_df: pd.DataFrame, frames: List[Dict[str, np.ndarray]], out_path: Path) -> None:
    final = frames[-1]
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])

    ax0 = fig.add_subplot(gs[:, 0])
    q = final["q"]
    role_idx = np.argmax(q, axis=1)
    labels = ["external", "boundary", "internal"]
    for i, name in enumerate(labels):
        mask = role_idx == i
        ax0.scatter(final["x"][mask], final["y"][mask], s=16, alpha=0.85, c=ROLE_COLORS[name], label=name.title())
    center = float(final["center"])
    sigma = float(final["sigma"])
    ax0.axhspan(center - sigma, center + sigma, color="#1f77b4", alpha=0.10)
    ax0.axhspan(center + sigma, center + sigma + 0.08, color="#fb8c00", alpha=0.15)
    ax0.axhspan(center - sigma - 0.08, center - sigma, color="#fb8c00", alpha=0.15)
    ax0.set_title("Emergent Conduit with Energy-Payback Blanket Shaping")
    ax0.set_xlabel("Longitudinal position")
    ax0.set_ylabel("Transverse position")
    ax0.set_xlim(0.0, 1.0)
    ax0.set_ylim(-1.2, 1.2)
    ax0.legend(loc="upper right", frameon=False)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(history_df["step"], history_df["efe_ema"], label="EFE (EMA)", linewidth=2)
    ax1.plot(history_df["step"], history_df["step_joule_total_ema"], label="Designer Joule / step (EMA)", linewidth=2)
    ax1.plot(history_df["step"], history_df["baseline_step_joule_total_ema"], label="Passive Joule / step (EMA)", linewidth=1.8, linestyle="--")
    ax1.set_title("EFE Reduction and Joule Reduction")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Energy / score")
    ax1.legend(frameon=False, fontsize=9)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(history_df["step"], history_df["cumulative_joule_saving_vs_baseline"], label="Cumulative energy saving vs passive", linewidth=2)
    ax2.plot(history_df["step"], history_df["energy_debt"], label="Energy debt", linewidth=2)
    ax2.axhline(0.0, color="black", linewidth=1)
    finite_be = history_df["breakeven_step"].dropna()
    if len(finite_be):
        be = int(finite_be.iloc[0])
        ax2.axvline(be, color="#2e7d32", linestyle="--", linewidth=1.5, label=f"Break-even step = {be}")
    ax2.set_title("Payback Dynamics")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Joule")
    ax2.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_animation(history_df: pd.DataFrame, frames: List[Dict[str, np.ndarray]], gif_path: Path, mp4_path: Path | None, cfg: Config) -> None:
    sampled_steps = [int(frame["step"]) for frame in frames]
    efe_sampled = history_df.loc[sampled_steps, "efe_ema"].to_numpy()
    designer_j_sampled = history_df.loc[sampled_steps, "step_joule_total_ema"].to_numpy()
    passive_j_sampled = history_df.loc[sampled_steps, "baseline_step_joule_total_ema"].to_numpy()
    debt_sampled = history_df.loc[sampled_steps, "energy_debt_ema"].to_numpy()
    savings_sampled = history_df.loc[sampled_steps, "cumulative_joule_saving_ema"].to_numpy()

    fig = plt.figure(figsize=(13.5, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    finite_be = history_df["breakeven_step"].dropna()
    breakeven = int(finite_be.iloc[0]) if len(finite_be) else None

    def update(i: int):
        ax0.clear(); ax1.clear(); ax2.clear()
        fr = frames[i]
        q = fr["q"]
        role_idx = np.argmax(q, axis=1)
        labels = ["external", "boundary", "internal"]
        for j, name in enumerate(labels):
            mask = role_idx == j
            ax0.scatter(fr["x"][mask], fr["y"][mask], s=16, alpha=0.85, c=ROLE_COLORS[name], label=name.title())
        center = float(fr["center"])
        sigma = float(fr["sigma"])
        ax0.axhspan(center - sigma, center + sigma, color="#1f77b4", alpha=0.10)
        ax0.axhspan(center + sigma, center + sigma + 0.08, color="#fb8c00", alpha=0.15)
        ax0.axhspan(center - sigma - 0.08, center - sigma, color="#fb8c00", alpha=0.15)
        ax0.set_title(f"Emergent conduit — step {int(fr['step'])}")
        ax0.set_xlabel("Longitudinal position")
        ax0.set_ylabel("Transverse position")
        ax0.set_xlim(0.0, 1.0)
        ax0.set_ylim(-1.2, 1.2)
        ax0.legend(loc="upper right", frameon=False, fontsize=9)

        ax1.plot(sampled_steps[: i + 1], efe_sampled[: i + 1], label="EFE (EMA)", linewidth=2)
        ax1.plot(sampled_steps[: i + 1], designer_j_sampled[: i + 1], label="Designer Joule / step (EMA)", linewidth=2)
        ax1.plot(sampled_steps[: i + 1], passive_j_sampled[: i + 1], label="Passive Joule / step (EMA)", linewidth=1.7, linestyle="--")
        ax1.set_title("EFE reduction and Joule reduction")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Energy / score")
        ax1.legend(frameon=False, fontsize=8)

        ax2.plot(sampled_steps[: i + 1], savings_sampled[: i + 1], linewidth=2, label="Cumulative saving (EMA)")
        ax2.plot(sampled_steps[: i + 1], debt_sampled[: i + 1], linewidth=2, label="Energy debt (EMA)")
        ax2.axhline(0.0, color="black", linewidth=1)
        if breakeven is not None:
            ax2.axvline(breakeven, color="#2e7d32", linestyle="--", linewidth=1.5, label=f"Break-even = {breakeven}")
        ax2.set_title("Payback dynamics")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Joule")
        ax2.legend(frameon=False, fontsize=8)
        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / cfg.fps)
    anim.save(gif_path, writer=PillowWriter(fps=cfg.fps))
    if mp4_path is not None and HAS_FFMPEG:
        anim.save(mp4_path, writer=FFMpegWriter(fps=cfg.fps))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Emergent conduit simulation with EFE-linked energy payback.")
    p.add_argument("--out_dir", type=Path, default=Path.cwd() / "emergent_conduit_payback_outputs")
    p.add_argument("--steps", type=int, default=Config.steps)
    p.add_argument("--frame_stride", type=int, default=Config.frame_stride)
    p.add_argument("--n_electrons", type=int, default=Config.n_electrons)
    args = p.parse_args()

    cfg = Config(steps=args.steps, frame_stride=max(1, args.frame_stride), n_electrons=args.n_electrons)
    out_dir = args.out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    designer_df, passive_df, frames = simulate(cfg)
    designer_df.to_csv(out_dir / "history_designer.csv", index=False)
    passive_df.to_csv(out_dir / "history_passive.csv", index=False)

    finite_be = designer_df["breakeven_step"].dropna()
    break_even = int(finite_be.iloc[0]) if len(finite_be) else np.nan
    corr = float(np.corrcoef(designer_df["efe_ema"], designer_df["step_joule_total_ema"])[0, 1])

    summary = pd.DataFrame([
        {
            **asdict(cfg),
            "final_efe": float(designer_df["efe"].iloc[-1]),
            "final_efe_ema": float(designer_df["efe_ema"].iloc[-1]),
            "final_step_joule_total": float(designer_df["step_joule_total"].iloc[-1]),
            "final_step_joule_total_ema": float(designer_df["step_joule_total_ema"].iloc[-1]),
            "final_passive_step_joule_total_ema": float(designer_df["baseline_step_joule_total_ema"].iloc[-1]),
            "final_internal_fraction": float(designer_df["internal_fraction"].iloc[-1]),
            "final_boundary_fraction": float(designer_df["boundary_fraction"].iloc[-1]),
            "final_external_fraction": float(designer_df["external_fraction"].iloc[-1]),
            "final_sigma": float(designer_df["sigma"].iloc[-1]),
            "final_coupling_proxy": float(designer_df["coupling_proxy"].iloc[-1]),
            "designer_cumulative_joule_total": float(designer_df["cumulative_joule_total"].iloc[-1]),
            "passive_cumulative_joule_total": float(designer_df["baseline_cumulative_joule_total"].iloc[-1]),
            "final_cumulative_saving_vs_passive": float(designer_df["cumulative_joule_saving_vs_baseline"].iloc[-1]),
            "final_energy_debt": float(designer_df["energy_debt"].iloc[-1]),
            "breakeven_step": break_even,
            "efe_to_joule_correlation": corr,
        }
    ])
    summary.to_csv(out_dir / "summary.csv", index=False)

    save_snapshot(designer_df, frames, out_dir / "emergent_conduit_payback_snapshot.png")
    gif_path = out_dir / "emergent_conduit_payback.gif"
    mp4_path = out_dir / "emergent_conduit_payback.mp4"
    try:
        save_animation(designer_df, frames, gif_path, mp4_path, cfg)
    except Exception:
        pass

    readme = f"""# Emergent Conduit with Energy Payback

This version adds a new design objective: the blanket must not only reduce late-stage Joule diffusion,
it must also repay any early energy overspend versus a passive baseline.

How payback is implemented:
- a passive shadow system evolves from the same initial condition under the same exogenous noise;
- the designer accumulates **energy debt** whenever its step Joule exceeds the passive baseline;
- policy evaluation over the EFE horizon penalizes path debt and terminal debt;
- a policy gets extra credit if it predicts net energy savings over the horizon.

Recommended command:

```bash
python inverted_beck_ramstead_emergent_conduit_payback.py --steps 5000 --frame_stride 50
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
