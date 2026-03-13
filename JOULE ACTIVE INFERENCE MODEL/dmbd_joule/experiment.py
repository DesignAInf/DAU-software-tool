"""experiment.py  --  dmbd_joule v5"""

import time
import numpy as np
from typing import Optional
from .system   import CableSystem, SystemParams
from .blanket  import BlanketSynthesizer
from .vbem     import VBEMInference
from .metrics  import InformationMetrics, MetricSnapshot, run_ablation, analyze_EFE_joule


def _bar(step, total, prefix="", width=38):
    filled = int(width * step / total)
    print(f"\r  {prefix} [{'█'*filled}{'░'*(width-filled)}] {100*step/total:5.1f}%",
          end="", flush=True)
    if step == total:
        print()


def run_experiment(
    n_warmup:   int = 200,
    n_baseline: int = 400,
    n_blanket:  int = 700,
    params: Optional[SystemParams] = None,
    out_dir: str = "results",
    save_plots: bool = True,
    verbose: bool = True,
) -> dict:

    if params is None:
        params = SystemParams()

    t0 = time.time()
    if verbose:
        print("\n" + "="*64)
        print("  DMBD JOULE v5  --  Toward complete Joule elimination")
        print("  Adaptive B_a deployment + per-source EM tracking")
        print("  Honest thermal floor (second-law limit)")
        print("="*64)
        print(f"\n  {params.n_electrons} electrons  .  {params.n_segments} segments")
        print(f"  kappa_ph={params.kappa_phonon}  kappa_em={params.kappa_em}"
              f"  kappa_df={params.kappa_defect}  lambda_J={params.lambda_joule_EFE}")
        print(f"  sigma_min={params.sigma_min} (thermal floor)"
              f"  top_N={params.n_hot_segments} hot segments")

    metrics = InformationMetrics()

    # --- WARMUP ---
    if verbose:
        print(f"\n  Phase 1 -- WARMUP  ({n_warmup} steps)")
    sys_w = CableSystem(params)
    for step in range(n_warmup):
        sys_w.step(blanket_active=False)
        if verbose and step % 10 == 0:
            _bar(step + 1, n_warmup, "warmup  ")
    if verbose:
        _bar(n_warmup, n_warmup, "warmup  ")

    # --- BASELINE ---
    if verbose:
        print(f"\n  Phase 2 -- BASELINE  ({n_baseline} steps)")
    sys_b = CableSystem(params)
    syn_b = BlanketSynthesizer(sys_b)
    inf_b = VBEMInference(sys_b, syn_b)
    baseline_snaps = []
    for step in range(n_baseline):
        sys_b.step(blanket_active=False)
        vbem_out, synth_out = inf_b.step()
        baseline_snaps.append(metrics.snapshot(sys_b, vbem_out, synth_out))
        if verbose and step % 10 == 0:
            _bar(step + 1, n_baseline, "baseline")
    if verbose:
        _bar(n_baseline, n_baseline, "baseline")
        Q_base  = np.mean([s.joule_I     for s in baseline_snaps])
        Q_floor = baseline_snaps[-1].Q_I_floor
        print(f"\n    mean Q_I baseline = {Q_base:.5f}")
        print(f"    thermal floor Q_I = {Q_floor:.5f}  "
              f"(max possible JRS = {max(0,(Q_base-Q_floor)/Q_base*100):.1f}%)")

    # --- BLANKET SYNTHESIS ---
    if verbose:
        print(f"\n  Phase 3 -- BLANKET SYNTHESIS  ({n_blanket} steps)")
        print("    Adaptive B_a in hot segments + per-source EM tracking")
    sys_bl = CableSystem(params)
    syn_bl = BlanketSynthesizer(sys_bl)
    inf_bl = VBEMInference(sys_bl, syn_bl)
    blanket_snaps = []
    for step in range(n_blanket):
        sys_bl.step(blanket_active=True)
        vbem_out, synth_out = inf_bl.step()
        blanket_snaps.append(metrics.snapshot(sys_bl, vbem_out, synth_out))
        if verbose and step % 10 == 0:
            _bar(step + 1, n_blanket, "blanket ")
    if verbose:
        _bar(n_blanket, n_blanket, "blanket ")
        def avg(f): return np.mean([getattr(s, f) for s in blanket_snaps])
        Q_bl    = avg("joule_I")
        Q_net   = avg("joule_I_net")
        Q_floor = avg("Q_I_floor")
        Q_base  = np.mean([s.joule_I for s in baseline_snaps])
        jrs     = max(0, (Q_base - Q_bl) / Q_base * 100)
        max_jrs = max(0, (Q_base - Q_floor) / Q_base * 100)
        print(f"\n    Q_I gross   = {Q_bl:.5f}   (JRS = {jrs:.1f}%)")
        print(f"    Q_I net     = {Q_net:.5f}   (above floor)")
        print(f"    Q_I floor   = {Q_floor:.5f}   (second-law limit)")
        print(f"    max possible JRS = {max_jrs:.1f}%")
        print(f"    attained / max   = {min(100, jrs/max(max_jrs,0.01)*100):.1f}%")
        print(f"    cancellation: phonon={avg('cancel_phonon')*100:.1f}%"
              f"  EM={avg('cancel_em')*100:.1f}%"
              f"  defects={avg('cancel_defect')*100:.1f}%"
              f"  total={avg('cancel_total')*100:.1f}%")
        print(f"    EM tracking accuracy = {avg('em_track_acc')*100:.1f}%")
        print(f"    mean n_I={avg('n_I'):.0f}  n_Bs={avg('n_Bs'):.0f}"
              f"  n_Ba={avg('n_Ba'):.0f}")

    # --- EFE -- JOULE ANALYSIS ---
    efe_analysis = analyze_EFE_joule(blanket_snaps)
    if verbose:
        print(efe_analysis.summary())

    # --- ABLATION VALIDATION ---
    if verbose:
        print("\n  Statistical validation...")
    result = run_ablation(blanket_snaps, baseline_snaps,
                          syn_bl.lagrange_multipliers)
    if verbose:
        print(result.summary())

    # --- FIGURES ---
    if save_plots:
        from .plots import plot_all
        if verbose:
            print(f"\n  Saving figures to {out_dir}/")
        plot_all(baseline_snaps, blanket_snaps, result, sys_bl, efe_analysis, out_dir)

    if verbose:
        print(f"\n  Completed in {time.time() - t0:.1f}s")
        print("=" * 64 + "\n")

    return {
        "baseline_snapshots": baseline_snaps,
        "blanket_snapshots":  blanket_snaps,
        "ablation_result":    result,
        "efe_analysis":       efe_analysis,
        "system":             sys_bl,
        "params":             params,
    }
