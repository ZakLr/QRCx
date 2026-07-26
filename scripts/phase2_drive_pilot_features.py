#!/usr/bin/env python3
"""FINAL SPRINT Phase 2 prerequisite: drive real pilot-scale v5 features
locally on CPU (no saved feature arrays exist from Sprint 4 -- only
aggregate metrics were persisted there). Uses N=10 (this project's
documented CPU-feasible fallback) on the real Sprint-4 pilot data
(data/sprint4_pilot_seq.npz, train 2019-2021/eval 2022), both gamma1=0.03
(tuned) and gamma1=0.0 (diagnostic, needed for the feature-variance-vs-
gamma1 diagnostic) so Phase 2's analysis has real features to work with
while Phase 1's canonical-split GPU drive runs in parallel."""
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.reservoir.sequential import SequentialDissipativeQRC

N_QUBITS = 10
N_TRAIN = 3000
N_TEST = 1000
GAMMA1_VALUES = {"tuned": 0.03, "diagnostic_off": 0.0}
OUT_DIR = REPO_ROOT / "results" / "phase2_features"


def main():
    npz = np.load(REPO_ROOT / "data" / "sprint4_pilot_seq.npz")
    train_seq = npz["train_seq"]
    val_seq = npz["val_seq"]
    full_seq = np.concatenate([train_seq[-N_TRAIN:], val_seq[:N_TEST]], axis=0)
    target_col_idx = int(npz["target_col_idx"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, gamma1 in GAMMA1_VALUES.items():
        print(f"Driving gamma1={gamma1} ({name}), N={N_QUBITS}, {len(full_seq)} steps...")
        w_in = np.random.default_rng(123).uniform(0.5, 1.5, size=N_QUBITS)
        qrc = SequentialDissipativeQRC(
            n_qubits=N_QUBITS, tau=1.0, gamma1=gamma1, gamma2=0.1, input_scaling=0.3,
            w_in=w_in, washout=0, seed=42, dtype="complex64",
        )
        t0 = time.perf_counter()
        feats = qrc.drive(full_seq)
        elapsed = time.perf_counter() - t0
        np.save(OUT_DIR / f"v5_features_{name}.npy", feats.astype(np.float32))
        print(f"  done in {elapsed/60:.1f} min ({elapsed/len(full_seq):.3f} s/step)")

    np.save(OUT_DIR / "target_series.npy", full_seq[:, target_col_idx])
    meta = {"n_qubits": N_QUBITS, "n_train": N_TRAIN, "n_test": N_TEST,
            "target_col_idx": target_col_idx}
    import json
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote features + target_series + meta.json to {OUT_DIR}")


if __name__ == "__main__":
    main()
