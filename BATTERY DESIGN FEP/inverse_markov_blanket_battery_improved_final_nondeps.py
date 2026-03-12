#!/usr/bin/env python3
"""
Rigorous inverse-design demo of a battery as an active-inference agent.

This script is a computational case study inspired by:
- Luca M. Possati, *Design for Entropy*, chapter 4.
- Jeff Beck & Maxwell J. D. Ramstead, *Dynamic Markov Blanket Detection for
  Macroscopic Physics Discovery* (2025).

What it implements
------------------
1) A Beck-Ramstead-style *detection* module:
   - microscopic observations y_i(t)
   - dynamic assignment variables omega_i(t) in {S, B, Z}
   - macroscopic latent variables s(t), b(t), z(t)
   - an AIR/VBEM-like loop: attend -> infer -> repeat

2) The chapter-4 *inversion* of that logic for design:
   - instead of discovering an existing blanket from data,
     we place designer priors on the future blanket and optimize interventions
     so that the desired blanket becomes statistically real.

3) A case study:
   - a theoretical battery designed as an active-inference agent that seeks
     energetic efficiency and longevity.

The model is intentionally pedagogical. It is not an electrochemical design
package and should not be treated as engineering guidance.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    from joblib import Parallel, delayed
    HAVE_JOBLIB = True
except ImportError:
    Parallel = None
    delayed = None
    HAVE_JOBLIB = False

from scipy.optimize import differential_evolution, minimize


# -----------------------------------------------------------------------------
# Paths / utils
# -----------------------------------------------------------------------------


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def clip(x: float | np.ndarray, lo: float, hi: float):
    return np.minimum(np.maximum(x, lo), hi)


def resolve_output_dir() -> Path:
    """
    Deterministic local output directory.

    Always write results into an ``outputs`` folder next to this script, unless
    the user explicitly overrides the destination via BATTERY_OUTPUT_DIR.
    This avoids confusion when the script is launched from another working
    directory (for example from an IDE, notebook, or shell in a different path).
    """
    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
    else:
        script_dir = Path.cwd().resolve()

    out_dir = Path(os.environ["BATTERY_OUTPUT_DIR"]).expanduser().resolve() if os.environ.get("BATTERY_OUTPUT_DIR") else (script_dir / "outputs").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# -----------------------------------------------------------------------------
# Markov blanket structure for the battery case study
# -----------------------------------------------------------------------------

NODE_NAMES = [
    "electrochemical_core",   # internal
    "bms_internal_model",     # internal
    "voltage_sensor",         # sensory blanket
    "temperature_sensor",     # sensory blanket
    "power_gate",             # active blanket
    "cooling_actuator",       # active blanket
    "load_environment",       # external
    "ambient_environment",    # external
    "charger_environment",    # external
]
IDX = {name: i for i, name in enumerate(NODE_NAMES)}
ROLE_NAMES = ["S", "B", "Z"]  # environment, blanket, internal
ROLE_TO_INDEX = {r: i for i, r in enumerate(ROLE_NAMES)}
N_NODES = len(NODE_NAMES)
N_ROLES = len(ROLE_NAMES)

TARGET_ROLE_BY_NODE = {
    "electrochemical_core": "Z",
    "bms_internal_model": "Z",
    "voltage_sensor": "B",
    "temperature_sensor": "B",
    "power_gate": "B",
    "cooling_actuator": "B",
    "load_environment": "S",
    "ambient_environment": "S",
    "charger_environment": "S",
}
TARGET_ROLE_VECTOR = np.array([ROLE_TO_INDEX[TARGET_ROLE_BY_NODE[n]] for n in NODE_NAMES], dtype=int)

GENERATIVE_FACTORIZATION = (
    "p(E,S,A,I) = p(E) p(S|E) p(I|S,A) p(A|I), with E={load, ambient, charger}, "
    "S={voltage_sensor, temperature_sensor}, A={power_gate, cooling_actuator}, "
    "I={electrochemical_core, bms_internal_model}."
)

CONDITIONAL_INDEPENDENCE_TARGETS = [
    "I ⟂ E | (S, A)",
    "S ⟂ (I, A) | E",
    "A ⟂ (E, S) | I",
]


# -----------------------------------------------------------------------------
# Design knobs = chapter 4 mechanisms + structural interventions
# -----------------------------------------------------------------------------

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    # chapter 4 levers
    "precision_crafting": (0.5, 8.0),     # sensory precision / precision weighting
    "policy_precision": (0.8, 10.0),      # decisiveness over policies
    "curiosity_sculpting": (0.0, 2.0),    # epistemic value weight
    "prediction_embedding": (0.0, 3.0),   # strength of preferences / priors
    # structural design interventions
    "separator_strength": (0.2, 3.0),     # suppress direct internal-external couplings
    "control_authority": (0.2, 3.0),      # gate authority
    "cooling_authority": (0.2, 3.0),      # cooling authority
    "thermal_insulation": (0.2, 3.0),     # insulation against ambient flux
    "charge_limit": (0.4, 2.0),           # normalized charge capability
    "discharge_limit": (0.4, 2.5),        # normalized discharge capability
}


@dataclass
class DesignKnobs:
    precision_crafting: float = 1.2
    policy_precision: float = 2.5
    curiosity_sculpting: float = 0.2
    prediction_embedding: float = 1.0
    separator_strength: float = 0.8
    control_authority: float = 0.9
    cooling_authority: float = 0.9
    thermal_insulation: float = 0.9
    charge_limit: float = 0.9
    discharge_limit: float = 1.0

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in PARAM_BOUNDS], dtype=float)

    @classmethod
    def from_vector(cls, x: np.ndarray) -> "DesignKnobs":
        return cls(**{k: float(v) for k, v in zip(PARAM_BOUNDS, x)})

    def clipped(self) -> "DesignKnobs":
        out = {}
        for k, (lo, hi) in PARAM_BOUNDS.items():
            out[k] = float(clip(getattr(self, k), lo, hi))
        return DesignKnobs(**out)


# -----------------------------------------------------------------------------
# Battery graph: microscopic couplings that the designer reshapes
# -----------------------------------------------------------------------------


def base_coupling_graph() -> np.ndarray:
    """Directed weighted graph among microscopic battery components."""
    W = np.zeros((N_NODES, N_NODES), dtype=float)

    core = IDX["electrochemical_core"]
    bms = IDX["bms_internal_model"]
    vs = IDX["voltage_sensor"]
    ts = IDX["temperature_sensor"]
    pg = IDX["power_gate"]
    ca = IDX["cooling_actuator"]
    load = IDX["load_environment"]
    amb = IDX["ambient_environment"]
    chg = IDX["charger_environment"]

    # internal -> sensory
    W[core, vs] = 1.00
    W[core, ts] = 0.90
    W[vs, bms] = 1.00
    W[ts, bms] = 1.00

    # internal -> active blanket
    W[bms, pg] = 1.00
    W[bms, ca] = 0.90

    # active -> environment
    W[pg, load] = 1.00
    W[pg, chg] = 0.90
    W[ca, amb] = 0.80

    # environment -> sensory blanket
    W[load, vs] = 0.70
    W[amb, ts] = 0.95
    W[chg, vs] = 0.55

    # undesirable shortcut couplings violating a clean blanket
    W[core, load] = 0.45
    W[load, core] = 0.30
    W[amb, core] = 0.40
    W[chg, core] = 0.25
    W[core, amb] = 0.20
    W[load, bms] = 0.20
    W[chg, bms] = 0.10
    return W


TARGET_PRIOR_LOGITS = np.full((N_NODES, N_ROLES), -2.2, dtype=float)
for i, node in enumerate(NODE_NAMES):
    TARGET_PRIOR_LOGITS[i, ROLE_TO_INDEX[TARGET_ROLE_BY_NODE[node]]] = 2.2


def apply_design_interventions(knobs: DesignKnobs) -> np.ndarray:
    """
    Inversion step.
    Instead of inferring blanket membership from data alone, the designer
    changes microscopic couplings so that the desired blanket is likely to exist.
    """
    W = base_coupling_graph().copy()

    p = knobs.precision_crafting
    s = knobs.separator_strength
    c = knobs.control_authority
    k = knobs.cooling_authority
    ins = knobs.thermal_insulation

    core = IDX["electrochemical_core"]
    bms = IDX["bms_internal_model"]
    vs = IDX["voltage_sensor"]
    ts = IDX["temperature_sensor"]
    pg = IDX["power_gate"]
    ca = IDX["cooling_actuator"]
    load = IDX["load_environment"]
    amb = IDX["ambient_environment"]
    chg = IDX["charger_environment"]

    # precision crafting: strengthen informative sensory corridor
    for a, b, gain in [
        (core, vs, 0.12),
        (core, ts, 0.10),
        (vs, bms, 0.12),
        (ts, bms, 0.12),
        (load, vs, 0.06),
        (amb, ts, 0.08),
        (chg, vs, 0.06),
    ]:
        W[a, b] *= (0.8 + gain * p)

    # action corridor
    W[bms, pg] *= (0.7 + 0.23 * c)
    W[pg, load] *= (0.7 + 0.23 * c)
    W[pg, chg] *= (0.7 + 0.23 * c)
    W[bms, ca] *= (0.7 + 0.23 * k)
    W[ca, amb] *= (0.7 + 0.23 * k)

    # separator engineering: suppress direct Z <-> S couplings
    for a, b in [
        (core, load),
        (load, core),
        (amb, core),
        (chg, core),
        (core, amb),
        (load, bms),
        (chg, bms),
    ]:
        W[a, b] *= np.exp(-0.75 * s)

    # insulation weakens environmental penetration to the core while preserving sensing
    W[amb, core] *= np.exp(-0.35 * ins)
    W[core, amb] *= np.exp(-0.35 * ins)

    return W


# -----------------------------------------------------------------------------
# Battery physics + active inference controller
# -----------------------------------------------------------------------------

@dataclass
class BatteryPhysicalState:
    soc: float = 0.62      # state of charge [0,1]
    temp: float = 27.5     # core temperature Celsius
    health: float = 0.985  # state of health [0,1]


@dataclass
class BatteryBeliefs:
    mu_soc: float = 0.60
    mu_temp: float = 27.0
    mu_health: float = 0.98
    sigma_soc: float = 0.06
    sigma_temp: float = 1.2
    sigma_health: float = 0.02


@dataclass
class StepRecord:
    t: int
    load: float
    ambient: float
    charger: float
    voltage_obs: float
    temp_obs: float
    gate: float
    cooling: float
    current: float
    soc: float
    temp: float
    health: float
    delivered_power: float
    charge_power: float
    joule_loss: float
    cooling_power: float
    chosen_policy_G: float


@dataclass
class EpisodeMetrics:
    efficiency: float
    service_score: float
    viability_fraction: float
    mean_temp: float
    max_temp: float
    final_health: float
    final_soc: float
    total_delivered_energy: float
    total_charge_energy: float
    total_losses: float
    total_cooling_energy: float


@dataclass
class DetectionMetrics:
    role_posteriors: Dict[str, Dict[str, float]]
    inferred_roles: Dict[str, str]
    blanket_nodes: List[str]
    accuracy_against_target: float
    blanket_clarity: float
    screening_off_score: float
    direct_internal_external_coupling: float
    observability_score: float
    controllability_score: float
    final_elbo: float
    detector_iterations: int


@dataclass
class EvaluationResult:
    design: Dict[str, float]
    objective: float
    energy_score: float
    longevity_score: float
    blanket_score: float
    service_score: float
    metrics: EpisodeMetrics
    detection: DetectionMetrics
    history: List[StepRecord]
    node_traces: Dict[str, List[float]]


def scenario_drivers(T: int, rng: np.random.Generator, scenario: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """External states E = {load, ambient, charger}."""
    t = np.arange(T)
    if scenario == "nominal":
        load = 0.55 + 0.18 * np.sin(2 * np.pi * t / 36.0) + 0.08 * rng.normal(size=T)
        ambient = 26.0 + 2.0 * np.sin(2 * np.pi * t / 70.0) + 0.4 * rng.normal(size=T)
        charger = ((t > 10) & (t < 25)).astype(float) * (0.65 + 0.05 * rng.normal(size=T))
    elif scenario == "hot_peak":
        load = 0.75 + 0.22 * np.sin(2 * np.pi * t / 28.0 + 0.4) + 0.10 * rng.normal(size=T)
        ambient = 31.0 + 2.5 * np.sin(2 * np.pi * t / 55.0) + 0.5 * rng.normal(size=T)
        charger = ((t > 55) & (t < 78)).astype(float) * (0.75 + 0.04 * rng.normal(size=T))
    elif scenario == "cycling":
        load = 0.45 + 0.28 * (np.sin(2 * np.pi * t / 18.0) > 0).astype(float) + 0.08 * rng.normal(size=T)
        ambient = 24.0 + 1.4 * np.sin(2 * np.pi * t / 43.0 + 0.6) + 0.4 * rng.normal(size=T)
        charger = ((t % 34) < 10).astype(float) * (0.82 + 0.05 * rng.normal(size=T))
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return clip(load, 0.05, 1.2), clip(ambient, 18.0, 38.0), clip(charger, 0.0, 1.2)


class BatteryGenerativeModel:
    """Minimal active-inference-style battery generative model."""

    dt = 1.0
    capacity_ah = 1.0

    def __init__(self, knobs: DesignKnobs, rng: np.random.Generator):
        self.knobs = knobs
        self.rng = rng

    @staticmethod
    def ocv(soc: float, health: float) -> float:
        return 3.05 + 1.20 * soc + 0.10 * (health - 0.9)

    @staticmethod
    def internal_resistance(temp: float, health: float) -> float:
        # Higher when health is poor or temperature drifts from nominal.
        return 0.07 + 0.11 * (1.0 - health) + 0.0022 * abs(temp - 27.0)

    def observe(self, state: BatteryPhysicalState) -> Tuple[float, float]:
        # Proxy current for sensor prediction is latent; measurement noise is reduced by precision crafting.
        noise_scale = 1.0 / (0.7 + 0.2 * self.knobs.precision_crafting)
        v = self.ocv(state.soc, state.health) + self.rng.normal(0.0, 0.025 * noise_scale)
        temp = state.temp + self.rng.normal(0.0, 0.16 * noise_scale)
        return v, temp

    def infer(self, beliefs: BatteryBeliefs, voltage_obs: float, temp_obs: float) -> BatteryBeliefs:
        k = self.knobs.precision_crafting
        pred_v = self.ocv(beliefs.mu_soc, beliefs.mu_health)
        pred_t = beliefs.mu_temp
        eps_v = voltage_obs - pred_v
        eps_t = temp_obs - pred_t

        # Precision-weighted updates: chapter 4 precision crafting.
        rate_soc = 0.045 * k
        rate_temp = 0.090 * k
        rate_health = 0.010 * k

        beliefs.mu_soc = float(clip(beliefs.mu_soc + rate_soc * eps_v, 0.02, 0.98))
        beliefs.mu_temp = float(clip(beliefs.mu_temp + rate_temp * eps_t, 15.0, 60.0))
        beliefs.mu_health = float(clip(beliefs.mu_health + rate_health * 0.35 * eps_v, 0.70, 1.00))

        beliefs.sigma_soc = float(clip(0.96 * beliefs.sigma_soc + 0.010 / (0.2 + k), 0.01, 0.20))
        beliefs.sigma_temp = float(clip(0.95 * beliefs.sigma_temp + 0.050 / (0.2 + k), 0.15, 2.50))
        beliefs.sigma_health = float(clip(0.985 * beliefs.sigma_health + 0.003 / (0.2 + k), 0.005, 0.08))
        return beliefs

    def expected_free_energy(
        self,
        beliefs: BatteryBeliefs,
        load: float,
        ambient: float,
        charger: float,
        gate: float,
        cooling: float,
    ) -> float:
        """
        Simple EFE proxy:
        G = risk + ambiguity - epistemic value
        """
        # Predict one step ahead under candidate policy.
        discharge_cap = self.knobs.discharge_limit * gate * load
        charge_cap = self.knobs.charge_limit * gate * charger
        net_current = charge_cap - discharge_cap
        resistance = self.internal_resistance(beliefs.mu_temp, beliefs.mu_health)

        # predicted next hidden states
        soc_next = clip(beliefs.mu_soc + 0.018 * charge_cap - 0.020 * discharge_cap, 0.0, 1.0)
        heat = 0.85 * (abs(net_current) ** 1.7) * resistance
        cooling_effect = 0.80 * self.knobs.cooling_authority * cooling
        ambient_leak = (ambient - beliefs.mu_temp) / (10.0 + 5.0 * self.knobs.thermal_insulation)
        temp_next = beliefs.mu_temp + heat + ambient_leak - cooling_effect
        health_next = beliefs.mu_health - 0.00020 * abs(net_current) - 0.00030 * max(temp_next - 32.0, 0.0)

        # Preferred outcomes = prediction embedding.
        pref = self.knobs.prediction_embedding
        risk = pref * (
            2.2 * (temp_next - 28.0) ** 2 / 25.0
            + 1.8 * (soc_next - 0.65) ** 2 / 0.16
            + 4.0 * max(31.5 - health_next * 32.0, 0.0) / 32.0
        )

        # Ambiguity = expected observation unreliability / state uncertainty.
        ambiguity = 0.30 * beliefs.sigma_soc + 0.10 * beliefs.sigma_temp + 2.0 * beliefs.sigma_health

        # Epistemic value = curiosity sculpting. Policies that expose informative current are slightly valued,
        # but overly noisy, high-thermal actions are discouraged.
        info_signal = abs(net_current) * (0.25 + 0.12 * gate) * np.exp(-max(temp_next - 33.0, 0.0) / 8.0)
        epistemic = self.knobs.curiosity_sculpting * info_signal

        return float(risk + ambiguity - epistemic)

    def select_action(
        self,
        beliefs: BatteryBeliefs,
        load: float,
        ambient: float,
        charger: float,
    ) -> Tuple[float, float, float]:
        candidate_gates = np.array([0.35, 0.55, 0.75, 1.0])
        candidate_cooling = np.array([0.0, 0.35, 0.70, 1.0])
        policies = [(g, c) for g in candidate_gates for c in candidate_cooling]

        G = np.array([
            self.expected_free_energy(beliefs, load, ambient, charger, gate=g, cooling=c)
            for (g, c) in policies
        ])
        p = softmax(-self.knobs.policy_precision * G)
        idx = int(np.argmax(p))
        gate, cooling = policies[idx]
        return float(gate), float(cooling), float(G[idx])

    def evolve(
        self,
        state: BatteryPhysicalState,
        gate: float,
        cooling: float,
        load: float,
        ambient: float,
        charger: float,
    ) -> Tuple[BatteryPhysicalState, Dict[str, float]]:
        resistance = self.internal_resistance(state.temp, state.health)
        discharge_current = self.knobs.discharge_limit * gate * load
        charge_current = self.knobs.charge_limit * gate * charger
        net_current = charge_current - discharge_current

        delivered_power = self.ocv(state.soc, state.health) * discharge_current
        charge_power = self.ocv(state.soc, state.health) * charge_current
        joule_loss = resistance * (abs(net_current) ** 2)
        cooling_power = 0.045 * self.knobs.cooling_authority * (cooling ** 1.4)

        # Battery dynamics
        next_soc = float(clip(state.soc + 0.020 * charge_current - 0.022 * discharge_current, 0.02, 0.98))

        thermal_inflow = 0.95 * joule_loss
        ambient_exchange = (ambient - state.temp) / (11.0 + 6.0 * self.knobs.thermal_insulation)
        cooling_effect = 0.95 * self.knobs.cooling_authority * cooling
        next_temp = float(clip(state.temp + thermal_inflow + ambient_exchange - cooling_effect, 18.0, 60.0))

        cycle_stress = 0.00018 * abs(net_current) ** 1.3
        thermal_stress = 0.00025 * max(next_temp - 31.0, 0.0) ** 1.25
        deep_discharge_stress = 0.00015 * max(0.20 - next_soc, 0.0) * 5.0
        next_health = float(clip(state.health - cycle_stress - thermal_stress - deep_discharge_stress, 0.70, 1.00))

        state = BatteryPhysicalState(soc=next_soc, temp=next_temp, health=next_health)
        aux = {
            "current": float(net_current),
            "delivered_power": float(delivered_power),
            "charge_power": float(charge_power),
            "joule_loss": float(joule_loss),
            "cooling_power": float(cooling_power),
        }
        return state, aux


def micro_node_vector(
    state: BatteryPhysicalState,
    beliefs: BatteryBeliefs,
    voltage_obs: float,
    temp_obs: float,
    gate: float,
    cooling: float,
    load: float,
    ambient: float,
    charger: float,
) -> np.ndarray:
    """Microscopic observations y_i(t) for the detection algorithm."""
    # normalize to comparable scales
    return np.array([
        0.55 * state.soc + 0.22 * (state.health) - 0.015 * (state.temp - 25.0),
        0.50 * beliefs.mu_soc + 0.25 * beliefs.mu_health - 0.012 * (beliefs.mu_temp - 25.0),
        (voltage_obs - 3.2) / 1.2,
        (temp_obs - 25.0) / 10.0,
        gate,
        cooling,
        load,
        (ambient - 20.0) / 15.0,
        charger,
    ], dtype=float)


# -----------------------------------------------------------------------------
# Beck-Ramstead style dynamic blanket detection (simplified AIR/VBEM)
# -----------------------------------------------------------------------------


def allowed_transition_matrix(stickiness: float = 4.0) -> np.ndarray:
    """HMM transitions: S <-> B <-> Z, but no direct S <-> Z."""
    T = np.array([
        [stickiness, 1.0, 0.0],
        [0.8, stickiness, 0.8],
        [0.0, 1.0, stickiness],
    ], dtype=float)
    T = T / T.sum(axis=1, keepdims=True)
    return T


def forward_backward(log_emission: np.ndarray, log_init: np.ndarray, log_T: np.ndarray) -> np.ndarray:
    """Posterior marginals for a 3-state HMM."""
    T_steps, K = log_emission.shape
    alpha = np.zeros((T_steps, K), dtype=float)
    beta = np.zeros((T_steps, K), dtype=float)

    alpha[0] = log_init + log_emission[0]
    alpha[0] -= np.logaddexp.reduce(alpha[0])

    for t in range(1, T_steps):
        for k in range(K):
            alpha[t, k] = log_emission[t, k] + np.logaddexp.reduce(alpha[t - 1] + log_T[:, k])
        alpha[t] -= np.logaddexp.reduce(alpha[t])

    beta[-1] = 0.0
    for t in range(T_steps - 2, -1, -1):
        for k in range(K):
            beta[t, k] = np.logaddexp.reduce(log_T[k] + log_emission[t + 1] + beta[t + 1])
        beta[t] -= np.logaddexp.reduce(beta[t])

    log_gamma = alpha + beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    return np.exp(log_gamma)


class DynamicBlanketDetector:
    def __init__(self, W: np.ndarray, prior_strength: float, max_iter: int = 30, tol: float = 1e-5):
        self.W = W
        self.prior_strength = prior_strength
        self.transition = allowed_transition_matrix(stickiness=4.5)
        self.log_T = np.log(self.transition + 1e-12)
        self.max_iter = max_iter
        self.tol = tol
        self.elbo_history: List[float] = []
        self.last_elbo: float = float('-inf')
        self.n_iter: int = 0

    def structural_logits(self, q_mean: np.ndarray) -> np.ndarray:
        """
        Role bias from graph structure. Boundary nodes should couple to both S and Z;
        internal nodes should couple mostly to B and weakly to S; external vice versa.
        """
        role_mass = q_mean.mean(axis=0)  # (N_nodes, N_roles)
        coupled = np.abs(self.W) + np.abs(self.W.T)
        mass_to_roles = coupled @ role_mass  # (N_nodes, N_roles)
        mS = mass_to_roles[:, ROLE_TO_INDEX["S"]]
        mB = mass_to_roles[:, ROLE_TO_INDEX["B"]]
        mZ = mass_to_roles[:, ROLE_TO_INDEX["Z"]]

        logits = np.zeros((N_NODES, N_ROLES), dtype=float)
        logits[:, ROLE_TO_INDEX["B"]] = 0.90 * np.minimum(mS, mZ) + 0.20 * mB
        logits[:, ROLE_TO_INDEX["Z"]] = 0.85 * mB - 1.10 * mS
        logits[:, ROLE_TO_INDEX["S"]] = 0.85 * mB - 1.10 * mZ
        return logits

    def elbo(self, q: np.ndarray, latents: np.ndarray, Y: np.ndarray, struct: np.ndarray, log_init: np.ndarray) -> float:
        """A variational lower bound for the blanket detector.

        This is a mean-field ELBO over dynamic role assignments. It combines:
        - expected Gaussian reconstruction terms
        - expected transition log-probabilities
        - prior / structural biases
        - posterior entropy
        """
        residual = Y[:, :, None] - latents[:, None, :]
        expected_log_likelihood = np.sum(
            q
            * (
                -0.5 * (residual ** 2) / 0.08
                + 0.18 * struct[None, :, :]
                + self.prior_strength * TARGET_PRIOR_LOGITS[None, :, :] * 0.15
            )
        )
        init_term = np.sum(q[0] * log_init)
        transition_term = np.sum(
            q[:-1, :, :, None] * q[1:, :, None, :] * self.log_T[None, None, :, :]
        )
        entropy = -np.sum(q * np.log(q + 1e-12))
        return float(expected_log_likelihood + init_term + transition_term + entropy)

    def detect(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            q: posterior q_ti(role), shape (T, N_nodes, 3)
            latents: macroscopic latents [s(t), b(t), z(t)] shape (T, 3)
        """
        T_steps, n = Y.shape
        assert n == N_NODES

        q = np.tile(softmax(self.prior_strength * TARGET_PRIOR_LOGITS, axis=1)[None, :, :], (T_steps, 1, 1))
        latents = np.zeros((T_steps, N_ROLES), dtype=float)
        log_init = np.log(softmax(self.prior_strength * TARGET_PRIOR_LOGITS, axis=1) + 1e-12)

        old_elbo = float('-inf')
        self.elbo_history = []

        for it in range(self.max_iter):
            for r in range(N_ROLES):
                weights = q[:, :, r]
                latents[:, r] = np.sum(weights * Y, axis=1) / (np.sum(weights, axis=1) + 1e-9)

            struct = self.structural_logits(q)

            q_new = np.zeros_like(q)
            for i in range(N_NODES):
                log_emission = np.zeros((T_steps, N_ROLES), dtype=float)
                y = Y[:, i]
                for r in range(N_ROLES):
                    residual = y - latents[:, r]
                    log_emission[:, r] = (
                        -0.5 * (residual ** 2) / 0.08
                        + 0.18 * struct[i, r]
                        + self.prior_strength * TARGET_PRIOR_LOGITS[i, r] * 0.15
                    )
                q_new[:, i, :] = forward_backward(log_emission, log_init[i], self.log_T)

            q = 0.60 * q + 0.40 * q_new
            q /= np.sum(q, axis=-1, keepdims=True)

            new_elbo = self.elbo(q, latents, Y, struct, log_init)
            self.elbo_history.append(new_elbo)
            self.n_iter = it + 1

            if it >= 2 and abs(new_elbo - old_elbo) < self.tol:
                old_elbo = new_elbo
                break
            old_elbo = new_elbo

        self.last_elbo = old_elbo
        return q, latents


# -----------------------------------------------------------------------------
# Diagnostics and objective for inversion
# -----------------------------------------------------------------------------


def detection_metrics_from_posteriors(W: np.ndarray, q: np.ndarray, final_elbo: float, detector_iterations: int) -> DetectionMetrics:
    q_mean = q.mean(axis=0)  # (N_nodes, 3)
    inferred = q_mean.argmax(axis=1)
    inferred_roles = {node: ROLE_NAMES[inferred[i]] for i, node in enumerate(NODE_NAMES)}
    role_posteriors = {
        node: {ROLE_NAMES[r]: float(q_mean[i, r]) for r in range(N_ROLES)}
        for i, node in enumerate(NODE_NAMES)
    }

    blanket_nodes = [node for i, node in enumerate(NODE_NAMES) if inferred[i] == ROLE_TO_INDEX["B"]]
    accuracy = float(np.mean(inferred == TARGET_ROLE_VECTOR))
    clarity = float(np.mean(np.max(q_mean, axis=1)))

    # direct internal-external coupling based on inferred roles
    inferred_Z = inferred == ROLE_TO_INDEX["Z"]
    inferred_S = inferred == ROLE_TO_INDEX["S"]
    direct_ie = float(np.sum(np.abs(W[np.ix_(inferred_Z, inferred_S)])) + np.sum(np.abs(W[np.ix_(inferred_S, inferred_Z)])))

    # observability: environment influences sensors; controllability: active blanket influences environment.
    obs_nodes = [IDX["voltage_sensor"], IDX["temperature_sensor"]]
    int_nodes = [IDX["electrochemical_core"], IDX["bms_internal_model"]]
    ext_nodes = [IDX["load_environment"], IDX["ambient_environment"], IDX["charger_environment"]]
    act_nodes = [IDX["power_gate"], IDX["cooling_actuator"]]

    observability = float(
        np.sum(np.abs(W[np.ix_(ext_nodes, obs_nodes)])) + np.sum(np.abs(W[np.ix_(int_nodes, obs_nodes)]))
    )
    controllability = float(
        np.sum(np.abs(W[np.ix_(act_nodes, ext_nodes)])) + np.sum(np.abs(W[np.ix_(int_nodes, act_nodes)]))
    )

    # screening-off score: high when blanket mediates most IE traffic.
    blanket_mask = inferred == ROLE_TO_INDEX["B"]
    mediated = float(
        np.sum(np.abs(W[np.ix_(inferred_Z, blanket_mask)]))
        + np.sum(np.abs(W[np.ix_(blanket_mask, inferred_S)]))
        + np.sum(np.abs(W[np.ix_(inferred_S, blanket_mask)]))
        + np.sum(np.abs(W[np.ix_(blanket_mask, inferred_Z)]))
    )
    screening = float(mediated / (mediated + direct_ie + 1e-9))

    return DetectionMetrics(
        role_posteriors=role_posteriors,
        inferred_roles=inferred_roles,
        blanket_nodes=blanket_nodes,
        accuracy_against_target=accuracy,
        blanket_clarity=clarity,
        screening_off_score=screening,
        direct_internal_external_coupling=direct_ie,
        observability_score=observability,
        controllability_score=controllability,
        final_elbo=float(final_elbo),
        detector_iterations=int(detector_iterations),
    )


def aggregate_episode_metrics(history: List[StepRecord]) -> EpisodeMetrics:
    delivered = np.array([h.delivered_power for h in history])
    charge = np.array([h.charge_power for h in history])
    losses = np.array([h.joule_loss for h in history])
    cooling = np.array([h.cooling_power for h in history])
    temps = np.array([h.temp for h in history])
    soc = np.array([h.soc for h in history])
    health = np.array([h.health for h in history])
    loads = np.array([h.load for h in history])

    delivered_energy = float(delivered.sum())
    charge_energy = float(charge.sum())
    total_losses = float(losses.sum())
    total_cooling = float(cooling.sum())

    # energy efficiency = useful delivery relative to useful + losses + cooling
    eff = delivered_energy / (delivered_energy + total_losses + total_cooling + 1e-9)

    # service score = fraction of requested load effectively served, clipped by gate limitations
    service = float(clip(delivered.mean() / (3.9 * loads.mean() + 1e-9), 0.0, 1.0))

    viability = float(np.mean((temps >= 22.0) & (temps <= 31.5) & (soc >= 0.18) & (soc <= 0.92)))

    return EpisodeMetrics(
        efficiency=float(eff),
        service_score=service,
        viability_fraction=viability,
        mean_temp=float(np.mean(temps)),
        max_temp=float(np.max(temps)),
        final_health=float(health[-1]),
        final_soc=float(soc[-1]),
        total_delivered_energy=delivered_energy,
        total_charge_energy=charge_energy,
        total_losses=total_losses,
        total_cooling_energy=total_cooling,
    )


def run_episode(knobs: DesignKnobs, scenario: str, seed: int) -> Tuple[List[StepRecord], np.ndarray]:
    rng = np.random.default_rng(seed)
    T = 110
    load_seq, ambient_seq, charger_seq = scenario_drivers(T, rng, scenario)

    gm = BatteryGenerativeModel(knobs, rng)
    state = BatteryPhysicalState(
        soc=float(clip(0.58 + 0.04 * rng.normal(), 0.40, 0.75)),
        temp=float(26.5 + 0.8 * rng.normal()),
        health=float(clip(0.985 + 0.003 * rng.normal(), 0.95, 0.995)),
    )
    beliefs = BatteryBeliefs(
        mu_soc=float(clip(state.soc + 0.04 * rng.normal(), 0.35, 0.80)),
        mu_temp=float(state.temp + 0.6 * rng.normal()),
        mu_health=float(clip(state.health + 0.01 * rng.normal(), 0.92, 1.0)),
    )

    history: List[StepRecord] = []
    Y = []

    for t in range(T):
        load = float(load_seq[t])
        ambient = float(ambient_seq[t])
        charger = float(charger_seq[t])

        voltage_obs, temp_obs = gm.observe(state)
        beliefs = gm.infer(beliefs, voltage_obs, temp_obs)
        gate, cooling, G = gm.select_action(beliefs, load, ambient, charger)
        state, aux = gm.evolve(state, gate, cooling, load, ambient, charger)

        Y.append(micro_node_vector(state, beliefs, voltage_obs, temp_obs, gate, cooling, load, ambient, charger))

        history.append(
            StepRecord(
                t=t,
                load=load,
                ambient=ambient,
                charger=charger,
                voltage_obs=voltage_obs,
                temp_obs=temp_obs,
                gate=gate,
                cooling=cooling,
                current=aux["current"],
                soc=state.soc,
                temp=state.temp,
                health=state.health,
                delivered_power=aux["delivered_power"],
                charge_power=aux["charge_power"],
                joule_loss=aux["joule_loss"],
                cooling_power=aux["cooling_power"],
                chosen_policy_G=G,
            )
        )

    return history, np.array(Y, dtype=float)


def evaluate_design(knobs: DesignKnobs, seed: int = 0) -> EvaluationResult:
    W = apply_design_interventions(knobs)
    scenarios = ["nominal", "hot_peak", "cycling"]

    if HAVE_JOBLIB:
        scenario_outputs = Parallel(
            n_jobs=min(3, os.cpu_count() or 1),
            prefer="threads",
        )(
            delayed(run_episode)(knobs, scenario=scenario, seed=seed + 100 * i + 11)
            for i, scenario in enumerate(scenarios)
        )
    else:
        scenario_outputs = [
            run_episode(knobs, scenario=scenario, seed=seed + 100 * i + 11)
            for i, scenario in enumerate(scenarios)
        ]

    all_histories: List[StepRecord] = []
    all_Y = []
    for history, Y in scenario_outputs:
        all_histories.extend(history)
        all_Y.append(Y)
    Y_all = np.concatenate(all_Y, axis=0)

    metrics = aggregate_episode_metrics(all_histories)
    detector = DynamicBlanketDetector(W, prior_strength=0.35 + 0.45 * knobs.prediction_embedding)
    q, latents = detector.detect(Y_all)
    detection = detection_metrics_from_posteriors(
        W,
        q,
        final_elbo=detector.last_elbo,
        detector_iterations=detector.n_iter,
    )

    energy_score = float(0.55 * metrics.efficiency + 0.45 * metrics.viability_fraction)
    longevity_score = float(0.65 * metrics.final_health + 0.35 * (1.0 - max(metrics.max_temp - 31.5, 0.0) / 20.0))
    blanket_score = float(
        0.45 * detection.accuracy_against_target
        + 0.25 * detection.blanket_clarity
        + 0.30 * detection.screening_off_score
    )
    service_score = float(metrics.service_score)

    objective = float(
        -2.2 * blanket_score
        -1.4 * energy_score
        -1.9 * longevity_score
        -1.0 * service_score
        +0.10 * detection.direct_internal_external_coupling
    )

    node_traces = {NODE_NAMES[i]: Y_all[:, i].tolist() for i in range(N_NODES)}
    return EvaluationResult(

        design=asdict(knobs),
        objective=objective,
        energy_score=energy_score,
        longevity_score=longevity_score,
        blanket_score=blanket_score,
        service_score=service_score,
        metrics=metrics,
        detection=detection,
        history=all_histories,
        node_traces=node_traces,
    )


def optimize_design(
    initial: DesignKnobs,
    seed: int = 0,
    optimizer: str = "spsa",
    maxiter: int = 4,
    popsize: int = 6,
) -> Tuple[EvaluationResult, EvaluationResult, Dict[str, float]]:
    baseline = evaluate_design(initial, seed=seed)
    bounds = np.array([PARAM_BOUNDS[k] for k in PARAM_BOUNDS], dtype=float)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    rng = np.random.default_rng(seed)

    cache: Dict[Tuple[float, ...], EvaluationResult] = {}

    def evaluate_vector(x: np.ndarray) -> EvaluationResult:
        x = np.asarray(x, dtype=float)
        x = clip(x, lo, hi)
        knobs = DesignKnobs.from_vector(x).clipped()
        key = tuple(np.round(knobs.to_vector(), 6))
        if key not in cache:
            cache[key] = evaluate_design(knobs, seed=seed)
        return cache[key]

    def objective_vector(x: np.ndarray) -> float:
        return float(evaluate_vector(x).objective)

    if optimizer.lower() == "de":
        result = differential_evolution(
            objective_vector,
            bounds=[tuple(b) for b in bounds],
            seed=seed,
            workers=1,
            updating="immediate",
            tol=1e-4,
            popsize=popsize,
            maxiter=maxiter,
            polish=True,
            disp=False,
        )
        best = evaluate_vector(result.x)
        optimization_info = {
            "optimizer": "scipy.differential_evolution",
            "maxiter": int(maxiter),
            "popsize": int(popsize),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "success": bool(result.success),
            "message": str(result.message),
        }
        return baseline, best, optimization_info

    # Default: low-budget gradient-based optimization using SPSA + Adam-style moments.
    x = initial.to_vector().copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    best = baseline
    nfev = 1

    for k in range(1, maxiter + 1):
        ak = 0.18 / ((k + 2) ** 0.60)
        ck = 0.12 / (k ** 0.20)
        delta = rng.choice([-1.0, 1.0], size=x.shape[0])
        x_plus = clip(x + ck * delta * (hi - lo), lo, hi)
        x_minus = clip(x - ck * delta * (hi - lo), lo, hi)
        f_plus = objective_vector(x_plus)
        f_minus = objective_vector(x_minus)
        nfev += 2

        grad = (f_plus - f_minus) / (2.0 * ck) * (1.0 / delta)
        grad = grad / (hi - lo + 1e-12)

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad ** 2)
        m_hat = m / (1.0 - beta1 ** k)
        v_hat = v / (1.0 - beta2 ** k)
        step = ak * m_hat / (np.sqrt(v_hat) + eps)
        x = clip(x - step * (hi - lo), lo, hi)

        current = evaluate_vector(x)
        nfev += 1
        if current.objective < best.objective:
            best = current

    optimization_info = {
        "optimizer": "gradient-based SPSA+Adam",
        "maxiter": int(maxiter),
        "popsize": 0,
        "nfev": int(nfev),
        "nit": int(maxiter),
        "success": True,
        "message": "Completed fixed-budget SPSA updates.",
    }
    return baseline, best, optimization_info


# -----------------------------------------------------------------------------
# Plotting / report
# -----------------------------------------------------------------------------


def make_plot(baseline: EvaluationResult, best: EvaluationResult, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)

    # history comparisons use first scenario chunk length 110 for legibility
    def arr(res: EvaluationResult, key: str) -> np.ndarray:
        return np.array([getattr(h, key) for h in res.history[:110]], dtype=float)

    axes[0].plot(arr(baseline, "temp"), label="Baseline")
    axes[0].plot(arr(best, "temp"), label="Optimized")
    axes[0].axhline(31.5, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Core temp (C)")
    axes[0].set_title("Battery as active-inference agent: inverse blanket design")
    axes[0].legend()

    axes[1].plot(arr(baseline, "soc"), label="Baseline")
    axes[1].plot(arr(best, "soc"), label="Optimized")
    axes[1].axhline(0.18, linestyle="--", linewidth=1)
    axes[1].axhline(0.92, linestyle="--", linewidth=1)
    axes[1].set_ylabel("State of charge")

    categories = [
        "Blanket\nscore",
        "Energy\nscore",
        "Longevity\nscore",
        "Service\nscore",
        "Clarity",
        "Screening",
    ]
    x = np.arange(len(categories))
    width = 0.36
    base_vals = np.array([
        baseline.blanket_score,
        baseline.energy_score,
        baseline.longevity_score,
        baseline.service_score,
        baseline.detection.blanket_clarity,
        baseline.detection.screening_off_score,
    ])
    best_vals = np.array([
        best.blanket_score,
        best.energy_score,
        best.longevity_score,
        best.service_score,
        best.detection.blanket_clarity,
        best.detection.screening_off_score,
    ])
    axes[2].bar(x - width / 2, base_vals, width=width, label="Baseline")
    axes[2].bar(x + width / 2, best_vals, width=width, label="Optimized")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories)
    axes[2].set_ylim(0, 1.2)
    axes[2].set_ylabel("Normalized score")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def summarize_result(res: EvaluationResult) -> Dict[str, object]:
    return {
        "design": res.design,
        "objective": res.objective,
        "energy_score": res.energy_score,
        "longevity_score": res.longevity_score,
        "blanket_score": res.blanket_score,
        "service_score": res.service_score,
        "metrics": asdict(res.metrics),
        "detection": asdict(res.detection),
    }


def main() -> None:
    out_dir = resolve_output_dir()

    initial = DesignKnobs(
        precision_crafting=1.1,
        policy_precision=2.2,
        curiosity_sculpting=0.15,
        prediction_embedding=0.9,
        separator_strength=0.7,
        control_authority=0.8,
        cooling_authority=0.8,
        thermal_insulation=0.8,
        charge_limit=0.9,
        discharge_limit=1.0,
    )

    baseline, best, optimization_info = optimize_design(initial, seed=7, optimizer=os.environ.get("BATTERY_OPTIMIZER", "spsa"), maxiter=int(os.environ.get("BATTERY_OPT_MAXITER", "4")), popsize=int(os.environ.get("BATTERY_OPT_POPSIZE", "6")))

    plot_path = out_dir / "inverse_markov_blanket_battery_rigorous.png"
    report_path = out_dir / "inverse_markov_blanket_battery_rigorous_report.json"
    code_summary_path = out_dir / "inverse_markov_blanket_battery_rigorous_summary.txt"

    make_plot(baseline, best, plot_path)

    report = {
        "case_study": "Battery designed as an active inference agent via inverse Markov blanket engineering",
        "generative_model_factorization": GENERATIVE_FACTORIZATION,
        "conditional_independence_targets": CONDITIONAL_INDEPENDENCE_TARGETS,
        "optimization": optimization_info,
        "interpretation": {
            "precision_crafting": "sensor and policy precision shaping",
            "curiosity_sculpting": "epistemic value / controlled uncertainty",
            "prediction_embedding": "preferences and target blanket priors",
            "inversion": "designer sets future blanket statistics and optimizes interventions so the boundary becomes real",
            "detector": "VBEM-like detector with ELBO monitoring and early stopping",
        },
        "baseline": summarize_result(baseline),
        "optimized": summarize_result(best),
        "delta": {
            "objective": best.objective - baseline.objective,
            "energy_score": best.energy_score - baseline.energy_score,
            "longevity_score": best.longevity_score - baseline.longevity_score,
            "blanket_score": best.blanket_score - baseline.blanket_score,
            "service_score": best.service_score - baseline.service_score,
            "screening_off_score": best.detection.screening_off_score - baseline.detection.screening_off_score,
            "direct_internal_external_coupling": (
                best.detection.direct_internal_external_coupling
                - baseline.detection.direct_internal_external_coupling
            ),
            "blanket_clarity": best.detection.blanket_clarity - baseline.detection.blanket_clarity,
            "final_health": best.metrics.final_health - baseline.metrics.final_health,
            "max_temp": best.metrics.max_temp - baseline.metrics.max_temp,
            "efficiency": best.metrics.efficiency - baseline.metrics.efficiency,
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary = []
    summary.append("Battery inverse Markov blanket design demo\n")
    summary.append("Generative model:\n")
    summary.append(f"  {GENERATIVE_FACTORIZATION}\n")
    summary.append("Conditional independencies:\n")
    for s in CONDITIONAL_INDEPENDENCE_TARGETS:
        summary.append(f"  - {s}\n")
    summary.append("\nOptimization backend:\n")
    summary.append(f"  {optimization_info['optimizer']} (nit={optimization_info['nit']}, nfev={optimization_info['nfev']}, success={optimization_info['success']})\n")
    summary.append("\nOptimized blanket nodes inferred by AIR detector:\n")
    for node in best.detection.blanket_nodes:
        summary.append(f"  - {node}\n")
    summary.append("\nBaseline vs optimized:\n")
    summary.append(f"  direct internal-external coupling: {baseline.detection.direct_internal_external_coupling:.3f} -> {best.detection.direct_internal_external_coupling:.3f}\n")
    summary.append(f"  blanket score: {baseline.blanket_score:.3f} -> {best.blanket_score:.3f}\n")
    summary.append(f"  blanket clarity: {baseline.detection.blanket_clarity:.3f} -> {best.detection.blanket_clarity:.3f}\n")
    summary.append(f"  screening-off score: {baseline.detection.screening_off_score:.3f} -> {best.detection.screening_off_score:.3f}\n")
    summary.append(f"  detector ELBO: {baseline.detection.final_elbo:.3f} -> {best.detection.final_elbo:.3f}\n")
    summary.append(f"  detector iterations: {baseline.detection.detector_iterations} -> {best.detection.detector_iterations}\n")
    summary.append(f"  efficiency: {baseline.metrics.efficiency:.3f} -> {best.metrics.efficiency:.3f}\n")
    summary.append(f"  final health: {baseline.metrics.final_health:.5f} -> {best.metrics.final_health:.5f}\n")
    summary.append(f"  max temperature: {baseline.metrics.max_temp:.2f}C -> {best.metrics.max_temp:.2f}C\n")
    with open(code_summary_path, "w", encoding="utf-8") as f:
        f.write("".join(summary))

    print(f"Saved plot to: {plot_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved summary to: {code_summary_path}")
    print(f"Optimizer: {optimization_info['optimizer']} | nit={optimization_info['nit']} | nfev={optimization_info['nfev']} | success={optimization_info['success']}")
    print()
    print("=== RESULTS ===")
    print(f"Objective: {baseline.objective:.4f} -> {best.objective:.4f}")
    print(f"Blanket score: {baseline.blanket_score:.4f} -> {best.blanket_score:.4f}")
    print(f"Energy score: {baseline.energy_score:.4f} -> {best.energy_score:.4f}")
    print(f"Longevity score: {baseline.longevity_score:.4f} -> {best.longevity_score:.4f}")
    print(f"Service score: {baseline.service_score:.4f} -> {best.service_score:.4f}")
    print(f"Direct internal-external coupling: {baseline.detection.direct_internal_external_coupling:.4f} -> {best.detection.direct_internal_external_coupling:.4f}")
    print(f"Blanket clarity: {baseline.detection.blanket_clarity:.4f} -> {best.detection.blanket_clarity:.4f}")
    print(f"Screening-off score: {baseline.detection.screening_off_score:.4f} -> {best.detection.screening_off_score:.4f}")
    print(f"Efficiency: {baseline.metrics.efficiency:.4f} -> {best.metrics.efficiency:.4f}")
    print(f"Final health: {baseline.metrics.final_health:.5f} -> {best.metrics.final_health:.5f}")
    print(f"Max temperature: {baseline.metrics.max_temp:.2f}C -> {best.metrics.max_temp:.2f}C")
    print("Inferred optimized blanket nodes:", ", ".join(best.detection.blanket_nodes))


if __name__ == "__main__":
    main()
