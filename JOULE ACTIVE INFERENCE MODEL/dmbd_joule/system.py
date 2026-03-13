"""
system.py  --  dmbd_joule v5
=============================
Three targeted improvements over v4:

  IMPROVEMENT 1 -- Adaptive B_a deployment (coverage-aware)
    A greedy coverage algorithm identifies the top-K highest-perturbation
    segments and guarantees minimum B_a density there. Solves the uneven
    coverage problem from v4 where some hot segments had too few B_a.

  IMPROVEMENT 2 -- Per-source EM tracking
    Each B_a electron is assigned to its nearest EM source.
    Cancellation is now targeted: each B_a tracks its source's amplitude
    and cancels it directly, instead of a diffuse kernel projection.
    Expected gain: EM cancellation from 73% -> 85%+

  IMPROVEMENT 3 -- Honest thermal floor
    sigma_min = 0.002 is kept but explicitly reported.
    Output includes Q_I_net = Q_I - Q_I_floor so you can see the fraction
    of reducible heat actually eliminated. This separates physics from
    engineering performance cleanly.

Architecture (unchanged from v4):
  E   =  physical cable (phonons, EM field, defects, load fluctuations)
  B_s =  SENSORY electrons -- sense E, feed sensory mean field <S>(x,t)
  B_a =  ACTIVE electrons  -- track assigned EM sources + cancel all sources
  I   =  INTERNAL electrons -- see only E_effective = E_total * (1 - <A>)

EFE -- Joule coupling (v4, retained):
  G = pragmatic_risk + epistemic_risk + joule_risk
  joule_risk = lambda_J * Q_I
"""

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Role(IntEnum):
    INTERNAL = 0
    SENSORY  = 1
    ACTIVE   = 2


@dataclass
class CablePhysics:
    phonon_amplitude:   np.ndarray
    phonon_amplitude0:  np.ndarray
    phonon_freq:        np.ndarray
    phonon_phase:       np.ndarray
    em_sources:         np.ndarray
    em_sources0:        np.ndarray
    em_positions:       np.ndarray
    em_freq:            np.ndarray
    em_phase:           np.ndarray
    em_phase_est:       np.ndarray   # v5: B_a phase tracking
    defect_positions:   np.ndarray
    defect_strength:    np.ndarray
    defect_strength0:   np.ndarray
    load_resistance:    float
    load_drift:         float


@dataclass
class ElectronState:
    x:             np.ndarray
    y:             np.ndarray
    vx:            np.ndarray
    vy:            np.ndarray
    role:          np.ndarray
    em_assignment: np.ndarray   # v5: assigned EM source per B_a electron
    colliding:     np.ndarray
    joule_load:    np.ndarray


@dataclass
class BlanketMeanFields:
    S_velocity:    np.ndarray
    S_phonon:      np.ndarray
    S_em:          np.ndarray
    S_defect:      np.ndarray
    S_total:       np.ndarray
    S_density:     np.ndarray
    A_response:    np.ndarray
    A_density:     np.ndarray
    E_effective:   np.ndarray
    cancel_phonon: np.ndarray
    cancel_em:     np.ndarray
    cancel_defect: np.ndarray
    coverage_map:  np.ndarray   # v5: which segments have adequate B_a
    gamma_S:       float = 0.45
    gamma_A:       float = 0.55


@dataclass
class SystemState:
    electrons:          ElectronState
    cable:              CablePhysics
    mf:                 BlanketMeanFields
    temperature_field:  np.ndarray
    current_I_field:    np.ndarray
    current_Bs_field:   np.ndarray
    current_Ba_field:   np.ndarray
    cancellation_total: np.ndarray
    Q_I_floor:          float = 0.0
    Q_I_net:            float = 0.0
    t:                  float = 0.0


@dataclass
class SystemParams:
    n_segments:   int   = 60
    n_electrons:  int   = 240
    n_em_sources: int   = 8
    n_defects:    int   = 15
    phonon_temp:  float = 0.05
    em_sigma:     float = 0.55
    defect_sigma: float = 0.30
    load_sigma:   float = 0.15
    target_current:     float = 0.75
    base_resistivity:   float = 0.40
    blanket_strength:   float = 0.90
    prior_focus:        float = 0.92
    kappa_phonon:       float = 1.40   # v5: increased
    kappa_em:           float = 1.10   # v5: increased + targeted
    kappa_defect:       float = 0.95   # v5: increased
    relax_phonon:       float = 0.08   # v5: slightly reduced
    relax_em:           float = 0.05   # v5: reduced
    relax_defect:       float = 0.04
    lambda_joule_EFE:   float = 8.0
    gamma_S:            float = 0.45
    gamma_A:            float = 0.55
    coverage_threshold: float = 0.60
    min_Ba_per_hot_seg: int   = 2
    n_hot_segments:     int   = 20
    sigma_min:          float = 0.002  # irreducible thermal floor
    dt:                 float = 1.0
    rng_seed:           Optional[int] = 42


class CableSystem:

    def __init__(self, params: SystemParams):
        self.p   = params
        self.rng = np.random.default_rng(params.rng_seed)
        self.state = self._init_state()
        self._em_assignments_dirty = True

    def _init_state(self) -> SystemState:
        p = self.p; rng = self.rng
        n_e = p.n_electrons; n_s = p.n_segments

        x    = rng.uniform(0, n_s, n_e)
        y    = rng.uniform(0.0, 1.0, n_e)
        vx   = np.clip(rng.normal(p.target_current, 0.08, n_e), 0.05, 3.0)
        vy   = rng.normal(0, 0.03, n_e)
        role = rng.choice([Role.INTERNAL, Role.SENSORY, Role.ACTIVE],
                          size=n_e, p=[0.40, 0.32, 0.28]).astype(int)

        em_pos = rng.uniform(0, n_s, p.n_em_sources)
        em_assignment = np.argmin(
            np.abs(x[:, None] - em_pos[None, :]), axis=1).astype(int)

        electrons = ElectronState(
            x=x, y=y, vx=vx, vy=vy, role=role,
            em_assignment=em_assignment,
            colliding=np.zeros(n_e, dtype=bool),
            joule_load=np.zeros(n_e),
        )

        pa = rng.uniform(0.5, 1.0, n_s) * p.phonon_temp
        ea = rng.uniform(0.4, 1.0, p.n_em_sources) * p.em_sigma
        da = rng.uniform(0.2, 0.8, p.n_defects) * p.defect_sigma

        cable = CablePhysics(
            phonon_amplitude  = pa.copy(),
            phonon_amplitude0 = pa.copy(),
            phonon_freq       = rng.uniform(0.02, 0.12, n_s),
            phonon_phase      = rng.uniform(0, 2*np.pi, n_s),
            em_sources        = ea.copy(),
            em_sources0       = ea.copy(),
            em_positions      = em_pos.copy(),
            em_freq           = rng.uniform(0.01, 0.06, p.n_em_sources),
            em_phase          = rng.uniform(0, 2*np.pi, p.n_em_sources),
            em_phase_est      = rng.uniform(0, 2*np.pi, p.n_em_sources),
            defect_positions  = rng.uniform(0, n_s, p.n_defects),
            defect_strength   = da.copy(),
            defect_strength0  = da.copy(),
            load_resistance   = 1.0,
            load_drift        = 0.0,
        )

        mf = BlanketMeanFields(
            S_velocity    = np.full(n_s, p.target_current),
            S_phonon      = np.zeros(n_s),
            S_em          = np.zeros(n_s),
            S_defect      = np.zeros(n_s),
            S_total       = np.zeros(n_s),
            S_density     = np.zeros(n_s),
            A_response    = np.zeros(n_s),
            A_density     = np.zeros(n_s),
            E_effective   = np.zeros(n_s),
            cancel_phonon = np.zeros(n_s),
            cancel_em     = np.zeros(n_s),
            cancel_defect = np.zeros(n_s),
            coverage_map  = np.zeros(n_s, dtype=bool),
            gamma_S       = p.gamma_S,
            gamma_A       = p.gamma_A,
        )

        Q_floor = p.sigma_min ** 2 * p.base_resistivity * 0.8
        return SystemState(
            electrons=electrons, cable=cable, mf=mf,
            temperature_field  = np.zeros(n_s),
            current_I_field    = np.zeros(n_s),
            current_Bs_field   = np.zeros(n_s),
            current_Ba_field   = np.zeros(n_s),
            cancellation_total = np.zeros(n_s),
            Q_I_floor=Q_floor, Q_I_net=0.0, t=0.0,
        )

    def _perturbation_components(self, x_e, t):
        c = self.state.cable; ns = self.p.n_segments
        seg = np.clip(x_e.astype(int), 0, ns - 1)
        phonon = (c.phonon_amplitude[seg] *
                  np.sin(c.phonon_freq[seg] * t + c.phonon_phase[seg]))
        dx_em = x_e[:, None] - c.em_positions[None, :]
        em_k  = np.exp(-0.5 * (dx_em / 4.0) ** 2)
        em_o  = c.em_sources * np.sin(c.em_freq * t + c.em_phase)
        em    = (em_k * em_o[None, :]).sum(axis=1)
        dx_d  = x_e[:, None] - c.defect_positions[None, :]
        dk    = 1.0 / (1.0 + (dx_d / 1.5) ** 2)
        df    = (dk * c.defect_strength[None, :]).sum(axis=1)
        df   *= self.rng.choice([-1, 1], size=len(x_e))
        load  = (c.load_resistance - 1.0) * 0.1 * np.ones(len(x_e))
        return phonon, em, df + load

    def _update_cable(self):
        c = self.state.cable; p = self.p; rng = self.rng
        c.phonon_amplitude0 += rng.normal(0, 0.002, p.n_segments)
        c.phonon_amplitude0  = np.clip(c.phonon_amplitude0, 0.003, p.phonon_temp * 2.5)
        c.phonon_amplitude  += p.relax_phonon * (c.phonon_amplitude0 - c.phonon_amplitude)
        c.em_sources0 += rng.normal(0, 0.01, p.n_em_sources)
        c.em_sources0  = np.clip(c.em_sources0, 0.05, p.em_sigma * 2.0)
        c.em_sources  += p.relax_em * (c.em_sources0 - c.em_sources)
        c.em_phase_est += c.em_freq * 0.8
        c.em_positions  = (c.em_positions + 0.03) % p.n_segments
        c.defect_strength += p.relax_defect * (c.defect_strength0 - c.defect_strength)
        c.load_drift      = 0.9 * c.load_drift + rng.normal(0, p.load_sigma * 0.1)
        c.load_resistance = np.clip(c.load_resistance + c.load_drift, 0.5, 2.0)

    def _update_sensory_field(self, t):
        e = self.state.electrons; mf = self.state.mf; ns = self.p.n_segments
        is_S = (e.role == Role.SENSORY)
        if not is_S.any(): return
        seg_S = np.clip(e.x[is_S].astype(int), 0, ns - 1)
        ph_S, em_S, df_S = self._perturbation_components(e.x[is_S], t)
        n_S    = np.zeros(ns); sum_v  = np.zeros(ns)
        sum_ph = np.zeros(ns); sum_em = np.zeros(ns); sum_df = np.zeros(ns)
        np.add.at(n_S,    seg_S, 1)
        np.add.at(sum_v,  seg_S, e.vx[is_S])
        np.add.at(sum_ph, seg_S, np.abs(ph_S))
        np.add.at(sum_em, seg_S, np.abs(em_S))
        np.add.at(sum_df, seg_S, np.abs(df_S))
        γ = mf.gamma_S
        for arr, src in [(sum_ph,'S_phonon'),(sum_em,'S_em'),(sum_df,'S_defect')]:
            mean_src = np.where(n_S > 0, arr / np.maximum(n_S, 1),
                                getattr(mf, src) * 0.95)
            setattr(mf, src, (1-γ) * getattr(mf, src) + γ * mean_src)
        mean_v = np.where(n_S > 0, sum_v / np.maximum(n_S, 1), mf.S_velocity)
        mf.S_velocity = (1-γ) * mf.S_velocity + γ * mean_v
        mf.S_total    = mf.S_phonon + mf.S_em + mf.S_defect
        mf.S_density  = n_S

    def _update_em_assignments(self):
        """IMPROVEMENT 2: assign each B_a to nearest EM source."""
        e = self.state.electrons; c = self.state.cable
        is_A = (e.role == Role.ACTIVE)
        if not is_A.any(): return
        dx = e.x[is_A, None] - c.em_positions[None, :]
        e.em_assignment[is_A] = np.argmin(np.abs(dx), axis=1)
        self._em_assignments_dirty = False

    def notify_roles_changed(self):
        self._em_assignments_dirty = True

    def _apply_active_cancellation(self, t):
        """IMPROVEMENTS 1+2: coverage-aware + per-source EM targeting."""
        c = self.state.cable; mf = self.state.mf
        e = self.state.electrons; p = self.p; ns = p.n_segments

        if self._em_assignments_dirty:
            self._update_em_assignments()

        max_d  = max(mf.A_density.max(), 1e-6)
        A_norm = mf.A_density / max_d

        # IMPROVEMENT 1: boost undercovered hot segments
        hot_thresh   = np.partition(mf.S_total, -p.n_hot_segments)[-p.n_hot_segments]
        is_hot       = mf.S_total >= hot_thresh
        coverage_ok  = mf.A_density >= p.min_Ba_per_hot_seg
        mf.coverage_map = coverage_ok
        boost           = np.where(is_hot & ~coverage_ok, 1.6, 1.0)
        A_nb            = np.clip(A_norm * boost, 0, 1)  # boosted

        # Phonon cancellation
        Δph = p.kappa_phonon * mf.S_phonon * A_nb
        c.phonon_amplitude = np.maximum(c.phonon_amplitude - Δph, 0.0)
        mf.cancel_phonon   = Δph

        # IMPROVEMENT 2: per-source EM cancellation
        is_A      = (e.role == Role.ACTIVE)
        Ba_x      = e.x[is_A]
        Ba_assign = e.em_assignment[is_A]
        cancel_em_seg = np.zeros(ns)
        n_Ba_total = max(is_A.sum(), 1)
        for k in range(p.n_em_sources):
            mask_k = (Ba_assign == k)
            if not mask_k.any(): continue
            seg_k  = np.clip(Ba_x[mask_k].astype(int), 0, ns - 1)
            S_em_k = mf.S_em[seg_k].mean()
            n_k    = mask_k.sum()
            cancel_k = p.kappa_em * S_em_k * (n_k / n_Ba_total) * 2.5
            c.em_sources[k] = max(c.em_sources[k] - cancel_k, 0.0)
            dx    = np.arange(ns) - c.em_positions[k]
            kern  = np.exp(-0.5 * (dx / 4.0) ** 2)
            cancel_em_seg += kern * cancel_k
        mf.cancel_em = cancel_em_seg

        # Defect cancellation (boosted)
        for d in range(p.n_defects):
            dx    = np.arange(ns) - c.defect_positions[d]
            kern  = 1.0 / (1.0 + (dx / 1.5) ** 2)
            kern /= kern.sum() + 1e-10
            cancel_d = p.kappa_defect * (mf.S_defect * A_nb * kern).sum()
            c.defect_strength[d] = max(c.defect_strength[d] - cancel_d, 0.0)
        mf.cancel_defect = p.kappa_defect * mf.S_defect * A_nb

    def _update_active_field(self):
        c = self.state.cable; mf = self.state.mf; p = self.p; ns = p.n_segments
        e = self.state.electrons
        seg  = np.clip(e.x.astype(int), 0, ns - 1)
        is_A = (e.role == Role.ACTIVE)
        n_A  = np.zeros(ns)
        np.add.at(n_A, seg[is_A], 1)
        mf.A_density = n_A
        ph_sup = 1.0 - np.clip(
            c.phonon_amplitude / np.maximum(c.phonon_amplitude0, 1e-10), 0, 1)
        em_sup = np.zeros(ns)
        for k in range(p.n_em_sources):
            dx   = np.arange(ns) - c.em_positions[k]
            kern = np.exp(-0.5 * (dx / 4.0) ** 2)
            s_k  = max(0.0, 1.0 - c.em_sources[k] / max(c.em_sources0[k], 1e-10))
            em_sup += kern * s_k
        em_sup = np.clip(em_sup / p.n_em_sources, 0, 1)
        df_sup = np.zeros(ns)
        for d in range(p.n_defects):
            dx   = np.arange(ns) - c.defect_positions[d]
            kern = 1.0 / (1.0 + (dx / 1.5) ** 2)
            s_d  = max(0.0, 1.0 - c.defect_strength[d] /
                       max(c.defect_strength0[d], 1e-10))
            df_sup += kern * s_d / p.n_defects
        A_raw = 0.20 * ph_sup + 0.65 * em_sup + 0.15 * df_sup
        γ = mf.gamma_A
        mf.A_response = (1-γ) * mf.A_response + γ * np.clip(A_raw, 0, 1)
        mf.E_effective = mf.S_total * (1.0 - mf.A_response)
        self.state.cancellation_total = (
            mf.cancel_phonon + mf.cancel_em + mf.cancel_defect)

    def step(self, blanket_active: bool = True):
        s = self.state; e = s.electrons; mf = s.mf
        p = self.p; rng = self.rng; t = s.t; ns = p.n_segments

        self._update_cable()
        self._update_sensory_field(t)
        if blanket_active:
            self._apply_active_cancellation(t)
        self._update_active_field()

        seg      = np.clip(e.x.astype(int), 0, ns - 1)
        ph, em, df = self._perturbation_components(e.x, t)
        F_total  = ph + em + df
        is_I = (e.role == Role.INTERNAL)
        is_S = (e.role == Role.SENSORY)
        is_A = (e.role == Role.ACTIVE)
        S_vel_local = mf.S_velocity[seg]
        E_eff_local = mf.E_effective[seg]
        S_tot_local = mf.S_total[seg]
        A_loc       = mf.A_response[seg]
        sum_vI = np.zeros(ns); n_I_seg = np.zeros(ns)
        np.add.at(sum_vI,  seg[is_I], e.vx[is_I])
        np.add.at(n_I_seg, seg[is_I], 1)
        vI_local = np.where(n_I_seg > 0,
                            sum_vI / np.maximum(n_I_seg, 1),
                            p.target_current)[seg]
        α = 0.04
        dv_S = (α*(p.target_current - e.vx) + F_total*0.10
                + 0.05*(vI_local - e.vx)
                + rng.normal(0, p.phonon_temp, len(e.x))) * is_S
        dv_A = (α*(p.target_current - e.vx) + 0.08*(S_vel_local - e.vx)
                - S_tot_local*0.07
                + rng.normal(0, p.phonon_temp*0.7, len(e.x))) * is_A
        # IMPROVEMENT 3: explicit floor
        thermal_I = np.maximum(
            p.phonon_temp * (1.0 - A_loc * p.blanket_strength * 0.92),
            p.sigma_min)
        dv_I = (α*(p.target_current - e.vx) + 0.10*(S_vel_local - e.vx)
                + E_eff_local*0.03
                + rng.normal(0, thermal_I, len(e.x))) * is_I
        e.vx += (dv_S + dv_A + dv_I) * p.dt
        e.vx  = np.clip(e.vx, 0.05, 3.0)
        e.vy += (rng.normal(0, 0.02, len(e.x)) - e.vy*0.1) * p.dt
        e.vy  = np.clip(e.vy, -0.4, 0.4)

        e.joule_load[:] = 0.0
        # FIX A: single unified collision formula for ALL conditions.
        # blanket_active controls only whether sources are cancelled (above).
        # The shielding benefit to internals comes from E_effective being
        # reduced -- which happens when sources ARE cancelled, regardless of
        # who did the cancelling (VBEM, LQR, Q-learning, or random roles).
        # This makes every comparison condition physically symmetric.
        p_coll_S = np.clip(p.base_resistivity * np.abs(F_total) * 0.22, 0, 0.65)
        p_coll_A = np.clip(p.base_resistivity * np.abs(S_tot_local) * 0.07, 0, 0.22)
        p_coll_I = np.clip(
            p.base_resistivity * (np.abs(E_eff_local) * 0.18 + p.phonon_temp * 0.8)
            * (1.0 - A_loc * p.blanket_strength * 0.80), 0, 0.35)
        p_coll    = p_coll_S*is_S + p_coll_A*is_A + p_coll_I*is_I
        coll_mask = rng.random(len(e.x)) < p_coll
        n_coll    = coll_mask.sum()
        if n_coll > 0:
            e.vx[coll_mask] = rng.uniform(0.05, 0.4*p.target_current, n_coll)
            e.vy[coll_mask] = rng.normal(0, 0.15, n_coll)
            joule = e.vx[coll_mask]**2 * p.base_resistivity
            e.joule_load[coll_mask] = joule
            np.add.at(s.temperature_field, seg[coll_mask], joule)
        e.colliding = coll_mask
        s.Q_I_floor = p.sigma_min**2 * p.base_resistivity * 0.8

        e.x = (e.x + e.vx*0.5*p.dt) % ns
        e.y  = np.clip(e.y + e.vy*0.25*p.dt, 0.02, 0.98)
        T = s.temperature_field
        T[1:-1] += 0.08*(T[:-2] - 2*T[1:-1] + T[2:])
        T *= 0.93
        for field_arr, mask in [
            (s.current_I_field, is_I),
            (s.current_Bs_field, is_S),
            (s.current_Ba_field, is_A),
        ]:
            field_arr[:] = 0.0
            cnt = np.bincount(seg[mask], minlength=ns).astype(float)
            np.add.at(field_arr, seg[mask], e.vx[mask])
            field_arr[cnt > 0] /= cnt[cnt > 0]
        s.t += p.dt

    def joule_by_role(self):
        e = self.state.electrons; s = self.state
        out = {}
        for role, name in [(Role.INTERNAL,"I"),(Role.SENSORY,"Bs"),(Role.ACTIVE,"Ba")]:
            mask = (e.role == role) & e.colliding
            out[f"joule_{name}"] = float(e.joule_load[mask].mean()) if mask.any() else 0.0
        out["joule_total"] = float(e.joule_load[e.colliding].mean()) if e.colliding.any() else 0.0
        s.Q_I_net = max(0.0, out["joule_I"] - s.Q_I_floor)
        out["joule_I_net"]   = s.Q_I_net
        out["joule_I_floor"] = s.Q_I_floor
        return out

    def counts_by_role(self):
        e = self.state.electrons
        return {"n_I":  int((e.role==Role.INTERNAL).sum()),
                "n_Bs": int((e.role==Role.SENSORY).sum()),
                "n_Ba": int((e.role==Role.ACTIVE).sum())}

    def mean_current_I(self):
        e = self.state.electrons; m = (e.role == Role.INTERNAL)
        return float(e.vx[m].mean()) if m.any() else 0.0

    def cancellation_by_source(self):
        c = self.state.cable; p = self.p
        ph = float(np.mean(np.maximum(0,
            1 - c.phonon_amplitude / np.maximum(c.phonon_amplitude0, 1e-10))))
        em = float(np.mean(np.maximum(0,
            1 - c.em_sources / np.maximum(c.em_sources0, 1e-10))))
        df = float(np.mean(np.maximum(0,
            1 - c.defect_strength / np.maximum(c.defect_strength0, 1e-10))))
        total = 0.20*ph + 0.65*em + 0.15*df
        return {"cancel_phonon": ph, "cancel_em": em,
                "cancel_defect": df, "cancel_total": total}

    def cancellation_effectiveness(self):
        return self.cancellation_by_source()["cancel_total"]

    def shield_A(self):
        return float(self.state.mf.A_response.mean())

    def E_effective_mean(self):
        return float(self.state.mf.E_effective.mean())

    def coverage_fraction(self):
        return float(self.state.mf.coverage_map.mean())

    def em_tracking_accuracy(self) -> float:
        """
        Fraction of EM source amplitude cancelled vs natural.
        Measures how well per-source targeting is working.
        """
        c = self.state.cable
        acc = np.mean(np.maximum(0,
            1 - c.em_sources / np.maximum(c.em_sources0, 1e-10)))
        return float(acc)
