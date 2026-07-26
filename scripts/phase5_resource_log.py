#!/usr/bin/env python3
"""FINAL SPRINT Phase 5 item 5: consolidated resource-accounting log
(rubric requirement) -- wall-clock, peak VRAM, s/step, qubit count,
feature dimension, precision, GPU model, for every real GPU stage this
project ran. Assembled entirely from already-written results/*.json
files (h200_benchmark.json, phase1_v5_canonical/phase1_canonical_drive_
results.json, phase5_v4_canonical_drive_results.json) plus one real,
directly-observed `nvidia-smi` reading taken during the v4 canonical
drive (32609 MiB used, H200 143771 MiB total) -- recorded here as the
single traceable source for that number, not re-derived or estimated.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Real, observed via `nvidia-smi --query-gpu=utilization.gpu,memory.used
# --format=csv` on the qBraid H200 instance (bma-pr-gpu-h200-7f806292)
# during the v4 canonical drive's test-split pass (2026-07-26 02:26 UTC).
# Not logged automatically by either drive script; recorded here from the
# real interactive observation rather than omitted or estimated.
OBSERVED_PEAK_VRAM_MIB = 32609
GPU_TOTAL_VRAM_MIB = 143771
GPU_MODEL = "NVIDIA H200"


def main():
    h200 = json.load(open(REPO_ROOT / "results" / "h200_benchmark.json"))
    v5 = json.load(open(REPO_ROOT / "results" / "phase1_v5_canonical" / "phase1_canonical_drive_results.json"))
    v4 = json.load(open(REPO_ROOT / "results" / "phase5_v4_canonical_drive_results.json"))

    out = {
        "gpu_model": GPU_MODEL,
        "gpu_total_vram_mib": GPU_TOTAL_VRAM_MIB,
        "observed_peak_vram_mib": OBSERVED_PEAK_VRAM_MIB,
        "observed_peak_vram_source": "nvidia-smi, observed during v4 canonical drive test-split pass",
        "benchmark_200step_probe": {
            "n_qubits": h200["n_qubits"], "sample_steps": h200["sample_steps"],
            "complex128_s_per_step": h200["complex128"]["s_per_step"],
            "complex64_s_per_step": h200["complex64"]["s_per_step"],
            "complex64_vs_complex128_rel_err": h200["complex64_vs_complex128_rel_err"],
            "precision_decision": h200["decision"], "decision_reason": h200["decision_reason"],
        },
        "v5_canonical_drive": {
            "architecture": "sequential dissipative (density matrix)",
            "n_qubits": v5["n_qubits"], "dtype": v5["dtype"],
            "feature_dimension": 3 * v5["n_qubits"] + 3 * v5["n_qubits"] * (v5["n_qubits"] - 1) // 2,
            "n_train": v5["n_train"], "n_val": v5["n_val"], "n_test": v5["n_test"],
            "drives": v5["drives"],
            "total_wall_clock_s": sum(d["wall_clock_s"] for d in v5["drives"]),
        },
        "v4_canonical_drive": {
            "architecture": "windowed statevector",
            "n_qubits": v4["n_qubits"], "dtype": v4["dtype"], "batch_size": v4["batch_size"],
            "feature_dimension": 3 * v4["n_qubits"] + 3 * v4["n_qubits"] * (v4["n_qubits"] - 1) // 2,
            "n_train": v4["n_train"], "n_val": v4["n_val"], "n_test": v4["n_test"],
            "cost_projection": v4["cost_projection"],
            "drives": v4["drives"],
            "total_wall_clock_s": sum(d["wall_clock_s"] for d in v4["drives"].values()),
        },
    }

    out_path = REPO_ROOT / "results" / "full_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
