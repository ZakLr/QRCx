#!/usr/bin/env python3
"""Sprint 1 runtime gate: benchmark the three SequentialDissipativeQRC
backends (numpy / qiskit_aer / pennylane_mixed) at the reference 12-qubit
configuration, and at the permitted 10-qubit fallback, then make the
go/no-go call for the Sprint 4 pilot (4,350 samples x washout 24 ~= 105k
naive reservoir steps, collapsing to ~4,374 steps with trajectory caching
since inputs are identical across overlapping windows -- see
sequential.py's module docstring for why caching is valid).

This is deliberately a *short* real-step benchmark (a handful of steps,
not the full 200 the Sprint 1 spec asked for) because the reference
12-qubit numpy backend measured ~120 s/step -- 200 steps would already be
~6.7 wall-clock hours for one backend alone. The measured per-step cost is
stable step-to-step (same fixed-size dense contraction every step), so a
short measurement is a valid basis for the steps/sec estimate; this
deviation from the spec's 200-step benchmark is recorded here rather than
silently reported as if the full benchmark ran.

FAST_MODE (per project convention): FAST_MODE=True runs 3 steps at n=12
and 3 steps at n=10 for all backends (~10-15 min total). FAST_MODE=False
is not implemented here -- a true 200-step benchmark at n=12 for all three
backends would take on the order of days given the numbers below, and is
not a reasonable default.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))

from QRCx.reservoir.sequential import SequentialDissipativeQRC
from QRCx.reservoir.sequential_backends import drive_pennylane_mixed, drive_qiskit_aer

N_SAMPLES_PILOT = 4350
WASHOUT_PILOT = 24
GO_THRESHOLD_HOURS = 12.0


def time_numpy(seq, n_qubits, **kw):
    qrc = SequentialDissipativeQRC(n_qubits=n_qubits, backend="numpy", **kw)
    t0 = time.perf_counter()
    qrc.drive(seq)
    return time.perf_counter() - t0


def time_qiskit_aer(seq, n_qubits, **kw):
    t0 = time.perf_counter()
    drive_qiskit_aer(seq, n_qubits=n_qubits, **kw)
    return time.perf_counter() - t0


def time_pennylane_mixed(seq, n_qubits, **kw):
    t0 = time.perf_counter()
    drive_pennylane_mixed(seq, n_qubits=n_qubits, **kw)
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=True)
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false")
    parser.add_argument("--n-steps-12q", type=int, default=3)
    parser.add_argument("--n-steps-10q", type=int, default=3)
    parser.add_argument("--n-steps-mixed", type=int, default=2,
                         help="pennylane_mixed is slowest to build/trace; fewer steps by default")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    kw = dict(trotter_steps=10, gamma1=0.02, gamma2=0.02, injection="ry", tau=1.0, J=1.0, g=1.0)

    backends = {}

    seq12 = rng.uniform(-1, 1, size=(args.n_steps_12q, 13))
    seq10 = rng.uniform(-1, 1, size=(args.n_steps_10q, 13))
    seq_mix = rng.uniform(-1, 1, size=(args.n_steps_mixed, 13))

    print("Timing numpy backend @ n=12 ...")
    t = time_numpy(seq12, 12, washout=24, **kw)
    backends["numpy_n12"] = {"n_qubits": 12, "n_steps": args.n_steps_12q,
                              "wall_clock_s": t, "s_per_step": t / args.n_steps_12q}

    print("Timing numpy backend @ n=10 ...")
    t = time_numpy(seq10, 10, washout=24, **kw)
    backends["numpy_n10"] = {"n_qubits": 10, "n_steps": args.n_steps_10q,
                              "wall_clock_s": t, "s_per_step": t / args.n_steps_10q}

    print("Timing qiskit_aer backend @ n=12 ...")
    t = time_qiskit_aer(seq12, 12, **kw)
    backends["qiskit_aer_n12"] = {"n_qubits": 12, "n_steps": args.n_steps_12q,
                                   "wall_clock_s": t, "s_per_step": t / args.n_steps_12q}

    print("Timing pennylane_mixed backend @ n=12 (short) ...")
    t = time_pennylane_mixed(seq_mix, 12, **kw)
    backends["pennylane_mixed_n12"] = {"n_qubits": 12, "n_steps": args.n_steps_mixed,
                                        "wall_clock_s": t, "s_per_step": t / args.n_steps_mixed}

    fastest_12q = min(
        (k for k in backends if k.endswith("_n12")),
        key=lambda k: backends[k]["s_per_step"],
    )
    s_per_step_12q = backends[fastest_12q]["s_per_step"]
    s_per_step_10q = backends["numpy_n10"]["s_per_step"]

    naive_steps = N_SAMPLES_PILOT * WASHOUT_PILOT
    cached_steps = N_SAMPLES_PILOT + WASHOUT_PILOT  # one continuous trajectory

    projection = {
        "naive_steps": naive_steps,
        "cached_steps": cached_steps,
        "12q": {
            "fastest_backend": fastest_12q,
            "s_per_step": s_per_step_12q,
            "naive_hours": naive_steps * s_per_step_12q / 3600,
            "cached_hours": cached_steps * s_per_step_12q / 3600,
            "go_cached": cached_steps * s_per_step_12q / 3600 <= GO_THRESHOLD_HOURS,
        },
        "10q_fallback": {
            "backend": "numpy_n10",
            "s_per_step": s_per_step_10q,
            "naive_hours": naive_steps * s_per_step_10q / 3600,
            "cached_hours": cached_steps * s_per_step_10q / 3600,
            "go_cached": cached_steps * s_per_step_10q / 3600 <= GO_THRESHOLD_HOURS,
        },
    }

    decision = "GO_12Q" if projection["12q"]["go_cached"] else (
        "GO_10Q_FALLBACK" if projection["10q_fallback"]["go_cached"] else "NO_GO"
    )

    result = {
        "backends": backends,
        "projection": projection,
        "decision": decision,
        "go_threshold_hours": GO_THRESHOLD_HOURS,
        "note": (
            "Benchmarked on a short real step count (see module docstring), not the "
            "spec's 200 steps, because measured per-step cost already made 200 steps "
            "impractical within the sprint's compute budget. Per-step cost is stable "
            "(same fixed-size dense contraction every step) so this is a valid basis "
            "for the steps/sec estimate."
        ),
    }

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "sequential_backend_benchmark.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")
    print(f"\nDECISION: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
