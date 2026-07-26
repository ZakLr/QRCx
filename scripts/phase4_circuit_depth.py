#!/usr/bin/env python3
"""FINAL SPRINT Phase 4: real circuit depth/gate-count for the v4
windowed architecture (AtmosphericQRC: ZZFeatureMap encoding + TFIM
Trotter evolution), via PennyLane's qml.specs on the actual qnode used
in transform() -- not a hand-derived estimate. Reports both the
paper's reference N=12 config and v4's own N=20 config (the two
qubit counts already in use elsewhere in the project, per
QRCx_FINAL_24H_SUPERPROMPT.md's instruction to state clearly that
v5(12q, density matrix)/v4(20q, statevector) are different
architectures, not a contradiction).

Note: v5 (SequentialDissipativeQRC) uses an exact matrix-exponential
propagator plus Kraus-channel dissipation -- it is simulated as a
open-system density-matrix evolution, not compiled to a literal gate
circuit, so "circuit depth" in the gate sense does not apply to it in
the same way; this script only covers v4."""
import json
import sys
from pathlib import Path

import numpy as np
import pennylane as qml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
from QRCx.reservoir.tfim import AtmosphericQRC

OUT_PATH = REPO_ROOT / "results" / "circuit_depth.json"


def measure(n_qubits, n_layers=3, trotter_steps=10):
    qrc = AtmosphericQRC(n_qubits=n_qubits, n_layers=n_layers, trotter_steps=trotter_steps,
                          use_lightning=False)
    x = np.random.default_rng(0).uniform(-1, 1, size=n_qubits)

    @qml.qnode(qrc._dev)
    def qnode(x):
        qrc.encoder.circuit(x)
        for _ in range(qrc.trotter_steps):
            for i in range(qrc.n_qubits):
                qml.RX(2.0 * qrc.g[i] * qrc.dt, wires=i)
            for i in range(qrc.n_qubits):
                qml.RZ(2.0 * qrc.h[i] * qrc.dt, wires=i)
            for i in range(qrc.n_qubits):
                for j in range(i + 1, qrc.n_qubits):
                    if qrc.J[i, j] != 0:
                        qml.IsingZZ(2.0 * qrc.J[i, j] * qrc.dt, wires=[i, j])
        return qml.state()

    specs = qml.specs(qnode)(x)
    resources = specs["resources"]
    return {
        "n_qubits": n_qubits, "n_layers": n_layers, "trotter_steps": trotter_steps,
        "depth": resources.depth,
        "total_gates": resources.num_gates,
        "gate_types": dict(resources.gate_types),
    }


def main():
    results = {
        "n12_reference": measure(12),
        "n20_v4_native": measure(20),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
