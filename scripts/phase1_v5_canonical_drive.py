#!/usr/bin/env python3
"""FINAL SPRINT Phase 1: launch the canonical-split v5 drive in the
background. Drives train_seq + val_seq + test_seq CONTINUOUSLY (one
recurrent pass, matching deployment semantics -- the reservoir's
recurrent state naturally carries across split boundaries; test labels
are simply never read until the final Phase 5 evaluation, so this does
NOT touch the test-split-scoring guard). Two configs (gamma1 tuned=0.03
vs diagnostic_off=0.0), V=1 (batching across configs as a tensor
dimension was assessed as non-trivial to retrofit into the density-matrix
engine safely within the time available -- run sequentially instead,
per the superprompt's own explicit fallback instruction), complex128
(Phase 1 benchmark's decision), 12-qubit reference config. Checkpointed
every 2000 steps so a stop only costs minutes, not hours."""
import json
import sys
import time
from pathlib import Path

import numpy as np

import sprint4_v5_gpu_handoff as v5

INPUT_NPZ = Path("data/canonical_seq.npz")
OUT_PATH = Path("phase1_canonical_drive_results.json")
N_QUBITS = 12
DTYPE = "complex128"
GAMMA1_VALUES = {"tuned": 0.03, "diagnostic_off": 0.0}
GAMMA2 = 0.1
INPUT_SCALING = 0.3
TAU = 1.0
W_IN_SEED = 123
CKPT_EVERY = 2000


def main():
    npz = np.load(INPUT_NPZ)
    train_seq = npz["train_seq"]
    val_seq = npz["val_seq"]
    test_seq = npz["test_seq"]
    full_seq = np.concatenate([train_seq, val_seq, test_seq], axis=0)
    n_train, n_val, n_test = len(train_seq), len(val_seq), len(test_seq)
    print(f"Loaded canonical sequences: train={n_train} val={n_val} test={n_test} "
          f"total={len(full_seq)}")

    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            results = json.load(f)
        done_configs = {d["gamma1_name"] for d in results.get("drives", [])}
        print(f"Resuming: {len(done_configs)} config(s) already done: {done_configs}")
    else:
        results = {"n_qubits": N_QUBITS, "dtype": DTYPE, "n_train": n_train, "n_val": n_val,
                   "n_test": n_test, "drives": []}
        done_configs = set()

    w_in = np.random.default_rng(W_IN_SEED).uniform(0.5, 1.5, size=N_QUBITS)

    for gamma1_name, gamma1 in GAMMA1_VALUES.items():
        if gamma1_name in done_configs:
            print(f"Skipping {gamma1_name} (already done)")
            continue
        print(f"\n=== Drive: gamma1={gamma1} ({gamma1_name}) ===")
        res = v5.SequentialReservoir(
            n_qubits=N_QUBITS, tau=TAU, gamma1=gamma1, gamma2=GAMMA2, J=1.0, g=1.0,
            input_scaling=INPUT_SCALING, w_in=w_in, multiplexing=1, use_gpu=v5.HAS_CUPY, dtype_str=DTYPE,
        )
        ckpt_path = f"phase1_ckpt_{gamma1_name}.npz"
        t0 = time.perf_counter()
        feats = res.drive(full_seq, ckpt_path=ckpt_path, ckpt_every=CKPT_EVERY)
        elapsed = time.perf_counter() - t0
        feats_path = f"phase1_features_{gamma1_name}.npy"
        np.save(feats_path, feats.astype(np.float32))  # float32 to keep file size manageable
        print(f"  Drive complete: {elapsed/60:.1f} min ({elapsed/len(full_seq):.4f} s/step). "
              f"Features saved to {feats_path}")
        results["drives"].append({"gamma1_name": gamma1_name, "gamma1": gamma1,
                                    "wall_clock_s": elapsed, "s_per_step": elapsed / len(full_seq),
                                    "features_path": feats_path})
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nAll drives complete. Results: {OUT_PATH}")
    return results


if __name__ == "__main__":
    main()
