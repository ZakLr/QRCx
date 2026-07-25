#!/usr/bin/env python3
"""FINAL SPRINT Phase 1: H200 benchmark (complex128 vs complex64, 12
qubits, 200 steps) + canonical-split decision. Self-contained (reuses
the already-verified SequentialReservoir/extraction engine from
sprint4_v5_gpu_handoff.py, imported as a sibling module -- upload both
files together)."""
import json
import time
from pathlib import Path

import numpy as np

import sprint4_v5_gpu_handoff as v5

OUT_PATH = Path("results_h200_benchmark.json")
N_QUBITS = 12
SAMPLE_STEPS = 200

# Canonical split candidates, per QRCx_FINAL_24H_SUPERPROMPT.md Phase 1
CANONICAL_STEPS = 52584  # train 2019-2022 (~4yr) + val 2023 (1yr) + test 2024 (1yr), ~6yr hourly
BUDGET_COMPLEX128_HOURS = 3.0
BUDGET_COMPLEX64_HOURS = 3.0
FALLBACK_BUDGET_HOURS = 4.0


def bench(dtype_str):
    w_in = np.random.default_rng(123).uniform(0.5, 1.5, size=N_QUBITS)
    res = v5.SequentialReservoir(
        n_qubits=N_QUBITS, tau=1.0, gamma1=0.03, gamma2=0.1, J=1.0, g=1.0,
        input_scaling=0.3, w_in=w_in, multiplexing=1, use_gpu=v5.HAS_CUPY, dtype_str=dtype_str,
    )
    seq = np.random.default_rng(0).uniform(-1, 1, size=(SAMPLE_STEPS, N_QUBITS + 1))
    t0 = time.perf_counter()
    res.drive(seq)
    elapsed = time.perf_counter() - t0
    return elapsed / SAMPLE_STEPS


def main():
    print(f"HAS_CUPY: {v5.HAS_CUPY}")
    results = {"n_qubits": N_QUBITS, "sample_steps": SAMPLE_STEPS, "has_cupy": v5.HAS_CUPY}

    for dtype_str in ("complex128", "complex64"):
        s_per_step = bench(dtype_str)
        proj_hours = CANONICAL_STEPS * s_per_step / 3600
        results[dtype_str] = {"s_per_step": s_per_step, "projected_hours_canonical": proj_hours}
        print(f"  {dtype_str}: {s_per_step:.4f} s/step -> {proj_hours:.2f}h projected "
              f"for {CANONICAL_STEPS} canonical steps")

    # Precision cross-check: complex64 vs complex128 on the same short segment
    w_in = np.random.default_rng(123).uniform(0.5, 1.5, size=N_QUBITS)
    seq = np.random.default_rng(0).uniform(-1, 1, size=(SAMPLE_STEPS, N_QUBITS + 1))
    res128 = v5.SequentialReservoir(n_qubits=N_QUBITS, tau=1.0, gamma1=0.03, gamma2=0.1, J=1.0, g=1.0,
                                     input_scaling=0.3, w_in=w_in, multiplexing=1, use_gpu=v5.HAS_CUPY,
                                     dtype_str="complex128")
    res64 = v5.SequentialReservoir(n_qubits=N_QUBITS, tau=1.0, gamma1=0.03, gamma2=0.1, J=1.0, g=1.0,
                                    input_scaling=0.3, w_in=w_in, multiplexing=1, use_gpu=v5.HAS_CUPY,
                                    dtype_str="complex64")
    f128 = res128.drive(seq)
    f64 = res64.drive(seq)
    rel_err = float(np.max(np.abs(f128 - f64) / (np.abs(f128) + 1e-8)))
    results["complex64_vs_complex128_rel_err"] = rel_err
    print(f"  complex64 vs complex128 max relative error: {rel_err:.2e} (threshold 1e-5)")

    # Decision logic per the superprompt
    if results["complex128"]["projected_hours_canonical"] <= BUDGET_COMPLEX128_HOURS:
        decision = "complex128"
        reason = f"projected {results['complex128']['projected_hours_canonical']:.2f}h <= {BUDGET_COMPLEX128_HOURS}h budget"
    elif results["complex64"]["projected_hours_canonical"] <= BUDGET_COMPLEX64_HOURS:
        decision = "complex64"
        reason = (f"complex128 projected {results['complex128']['projected_hours_canonical']:.2f}h > budget; "
                   f"complex64 projected {results['complex64']['projected_hours_canonical']:.2f}h <= {BUDGET_COMPLEX64_HOURS}h, "
                   f"rel_err={rel_err:.2e} (validated < 1e-5: {rel_err < 1e-5})")
    else:
        decision = "shrink_split"
        reason = (f"even complex64 projected {results['complex64']['projected_hours_canonical']:.2f}h > "
                   f"{FALLBACK_BUDGET_HOURS}h fallback budget -- recommend shrinking to train 2020-2022/val2023/test2024")

    results["decision"] = decision
    results["decision_reason"] = reason
    print(f"\nDECISION: {decision} ({reason})")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    return results


if __name__ == "__main__":
    main()
