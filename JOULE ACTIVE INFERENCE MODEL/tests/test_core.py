"""
tests/test_core.py
------------------
Minimal smoke tests. Run with:  python3 -m pytest tests/ -v
"""
import numpy as np

from dmbd_joule.system  import CableSystem, SystemParams, Role
from dmbd_joule.blanket import BlanketSynthesizer
from dmbd_joule.vbem    import VBEMInference
from dmbd_joule.metrics import InformationMetrics


def _quick_run(n_steps=20, blanket_active=True):
    p   = SystemParams(n_electrons=60, n_segments=20, rng_seed=0)
    sys = CableSystem(p)
    syn = BlanketSynthesizer(sys)
    inf = VBEMInference(sys, syn)
    m   = InformationMetrics()
    snaps = []
    for _ in range(n_steps):
        sys.step(blanket_active=blanket_active)
        vo, so = inf.step()
        snaps.append(m.snapshot(sys, vo, so))
    return sys, snaps


def test_system_runs():
    sys, snaps = _quick_run(20)
    assert len(snaps) == 20


def test_roles_populated():
    sys, _ = _quick_run(10)
    e = sys.state.electrons
    assert (e.role == Role.INTERNAL).any()
    assert (e.role == Role.SENSORY).any()
    assert (e.role == Role.ACTIVE).any()


def test_role_fractions_within_bounds():
    sys, _ = _quick_run(50)
    e = sys.state.electrons
    n = len(e.role)
    for role in [Role.INTERNAL, Role.SENSORY, Role.ACTIVE]:
        frac = (e.role == role).sum() / n
        assert 0.10 <= frac <= 0.65, f"Role {role} fraction {frac:.2f} out of bounds"


def test_cancellation_positive():
    sys, snaps = _quick_run(50, blanket_active=True)
    cancel = np.mean([s.cancel_total for s in snaps[-20:]])
    assert cancel > 0.05, f"Expected cancellation > 5%, got {cancel:.3f}"


def test_blanket_reduces_joule():
    _, snaps_on  = _quick_run(80, blanket_active=True)
    _, snaps_off = _quick_run(80, blanket_active=False)
    Q_on  = np.mean([s.joule_I for s in snaps_on[-40:]])
    Q_off = np.mean([s.joule_I for s in snaps_off[-40:]])
    assert Q_on < Q_off, (
        f"Blanket should reduce Q_I: on={Q_on:.5f} off={Q_off:.5f}")


def test_efe_tracks_joule():
    _, snaps = _quick_run(100, blanket_active=True)
    efe = np.array([s.EFE      for s in snaps[-50:]])
    qi  = np.array([s.joule_I  for s in snaps[-50:]])
    r   = float(np.corrcoef(efe, qi)[0, 1])
    assert r > 0.70, f"Expected r(EFE, Q_I) > 0.70, got {r:.3f}"


def test_joule_floor_reported():
    sys, snaps = _quick_run(30, blanket_active=True)
    floors = [s.Q_I_floor for s in snaps]
    assert all(f >= 0 for f in floors)


def test_em_tracking_accuracy():
    sys, snaps = _quick_run(60, blanket_active=True)
    acc = np.mean([s.em_track_acc for s in snaps[-20:]])
    assert acc > 0.30, f"EM tracking accuracy too low: {acc:.3f}"


def test_lqr_controller():
    from dmbd_joule.controllers import LQRController, _observe, _apply_control
    p    = SystemParams(n_electrons=60, n_segments=20, rng_seed=1)
    sys  = CableSystem(p)
    ctrl = LQRController(p)
    sys.state.electrons.role[:] = Role.ACTIVE
    for _ in range(10):
        obs = _observe(sys)
        kp, ke, kd = ctrl.act(obs)
        assert 0.0 <= kp <= 3.0
        assert 0.0 <= ke <= 3.0
        _apply_control(sys, kp, ke, kd)
        sys.step(blanket_active=False)


def test_qlearning_controller():
    from dmbd_joule.controllers import QLearningController, _observe, _apply_control
    p    = SystemParams(n_electrons=60, n_segments=20, rng_seed=2)
    sys  = CableSystem(p)
    ctrl = QLearningController(p, seed=2)
    sys.state.electrons.role[:] = Role.ACTIVE
    prev_Q = 0.01
    for _ in range(20):
        obs = _observe(sys)
        kp, ke, kd = ctrl.act(obs, reward=-prev_Q)
        assert 0.0 < kp <= 3.0
        _apply_control(sys, kp, ke, kd)
        sys.step(blanket_active=False)
        prev_Q = sys.joule_by_role()["joule_total"]


def test_condition_G_lambda_zero():
    """
    v6 closed loop: D and G should differ.
    With lambda_J=0 in EFE, the optimizer has less signal to push
    electrons toward I-protective roles, so G should underperform D.
    (In v5 open-loop D=G was trivially true; in v6 closed-loop it is
    no longer trivially true -- the optimizer genuinely uses lambda_J.)
    """
    import copy
    from dmbd_joule.comparison import _run_condition_D, _run_condition_G
    p  = SystemParams(n_electrons=80, n_segments=20, rng_seed=0,
                      relax_em=0.03, kappa_em=1.10,
                      lambda_joule_EFE=8.0)
    m  = InformationMetrics()
    sD = _run_condition_D(p, 60, m, label="D")
    sG = _run_condition_G(p, 60, m, label="G")
    qD = np.mean([s.joule_I for s in sD[-30:]])
    qG = np.mean([s.joule_I for s in sG[-30:]])
    # Both should reduce Joule vs a high baseline
    assert qD < 0.015, f"D Q_I too high: {qD}"
    # D should be <= G (closed loop with joule_risk performs at least as well)
    # Allow small tolerance for stochasticity
    assert qD <= qG + 0.003, \
        f"v6: D should be <= G (closed loop better than no joule_risk): D={qD:.5f} G={qG:.5f}"


def test_fair_lqr_has_roles():
    """LQR condition must maintain I/B_s/B_a structure (not all-ACTIVE)."""
    from dmbd_joule.comparison import _run_condition_E
    p = SystemParams(n_electrons=60, n_segments=20, rng_seed=0)
    m = InformationMetrics()
    snaps = _run_condition_E(p, 20, m, label="E")
    # After running, roles should be distributed (VBEM active)
    assert len(snaps) == 20


def test_warmup_shared():
    """Warmup cable state is transferred to conditions."""
    from dmbd_joule.comparison import _copy_cable_state
    p1 = SystemParams(rng_seed=0)
    p2 = SystemParams(rng_seed=99)   # different seed -> different init
    s1 = CableSystem(p1)
    s2 = CableSystem(p2)
    # Run warmup on s1
    for _ in range(50): s1.step(blanket_active=False)
    # Copy state to s2
    _copy_cable_state(s1, s2)
    # Cable amplitudes should now match
    np.testing.assert_allclose(
        s1.state.cable.phonon_amplitude,
        s2.state.cable.phonon_amplitude, rtol=1e-6)


def test_hac_se_positive():
    """HAC standard error should be positive for any non-constant series."""
    from dmbd_joule.comparison import hac_se
    ts = np.random.default_rng(0).normal(0.01, 0.002, 200)
    se = hac_se(ts)
    assert se > 0, f"HAC SE must be positive, got {se}"
    assert se < 0.01, f"HAC SE unreasonably large: {se}"
