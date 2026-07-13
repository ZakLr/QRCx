# Sprint 1 — Sequential Dissipative Reservoir Core + Runtime Gate

**Status:** Core reservoir built, validated, and benchmarked; runtime gate
decision made. Runtime gate decision: **NO-GO at 12 qubits, GO at the
10-qubit fallback** (with trajectory caching, which is implemented by
default). **NARMA10 micro-validation does NOT meet its target** (best
NMSE 0.945 vs. target <0.4, still behind the linear AR baseline) — see
the NARMA10 section for the full debugging trail and root causes found.
ESP verified in sequential mode at test scale. `pytest tests/ -v` green
(17 tests: 12 from Sprint 0 + 5 new sequential-reservoir tests).

## What was built

- **`QRCx/QRCx/reservoir/sequential.py`** — `SequentialDissipativeQRC`, a
  genuinely recurrent reservoir: a single density matrix `rho` is carried
  forward across the whole input sequence (`drive()`), rather than
  re-encoding a fresh window from `|0><0|` per sample (the v4 windowed
  design in `tfim.py`). Implements:
  - `U(s_k)`: RY(a·x_j) per qubit round-robin injection (default), with
    the 13th feature folded onto qubit 0 as RZ; `injection="zz"` keeps a
    ZZFeatureMap-style alternative for ablation.
  - `e^{-iH tau}`: TFIM with fixed J=1.0, g=1.0, first-order Trotter,
    M=10 steps, tau=1.0 — RX (transverse field) + IsingZZ (coupling),
    reusing the same gate conventions as `reservoir/tfim.py`.
  - `D_{gamma1,gamma2}`: per-qubit amplitude damping (gamma1) + dephasing
    (gamma2) Kraus channels, applied once per step (not per Trotter
    substep).
  - `washout` parameter (default 24) and a rewind-based `transform()` for
    small-scale/testing use; `drive()` is the production single-pass path
    with trajectory caching built in by default (see below).
- **Density-matrix engine** (`NumpyDensityOps` in `sequential.py`): hand
  -rolled reshape+einsum Kraus propagation, never materialising a full
  `2**n x 2**n` Pauli operator. Diagonal Z-basis rotations (RZ, IsingZZ,
  dephasing) are combined into a single elementwise phase/coefficient
  multiply per step rather than applied gate-by-gate.
- **`QRCx/QRCx/readout/correlators.py`** extended with `single_body_dm`,
  `two_body_dm`, `extract_correlators_dm` — density-matrix analogues of
  the existing pure-state `single_body`/`two_body`/`extract_correlators`,
  via efficient partial trace (reshape+einsum) rather than full-operator
  expectation values.
- **`QRCx/QRCx/reservoir/sequential_backends.py`** — `drive_pennylane_mixed`
  (via `qml.device("default.mixed")` + `qml.QubitDensityMatrix` to carry
  state between steps) and `drive_qiskit_aer` (`AerSimulator(method=
  "density_matrix")` + `qiskit_aer`/`qiskit.quantum_info.Kraus` channels),
  implementing the identical per-step update for benchmarking against the
  numpy backend.
- **`QRCx/QRCx/reservoir/esp_sequential.py`** — sequential-mode ESP:
  trace distance (not statevector L2) between a reference trajectory and
  `n_initial_states` random Ginibre-ensemble density matrices driven by
  the same input sequence.
- **`QRCx/QRCx/data/narma.py`** — NARMA10 generator.
- **`scripts/benchmark_sequential_backends.py`**, **`scripts/
  narma10_validation.py`** — reproducible entry points for everything
  below; every number in this report is written to `results/*.json` by
  these scripts (per the project's traceability rule), not hand-typed.

## A real bug found and fixed during validation

Cross-checking the three backends against each other (rather than trusting
the numpy engine in isolation) surfaced two real bugs:

1. **Bra-side Kraus contraction had the K-dagger index order backwards**
   in `NumpyDensityOps.conjugate_1q`. It was masked completely by testing
   only with `RX` first, because `RX = [[c,-is],[-is,c]]` happens to be a
   *symmetric* matrix, so the bug's effect (using `K` where `K^T` was
   needed) was invisible. `RY = [[c,-s],[s,c]]` is not symmetric, and
   applying it immediately broke trace preservation (measured trace
   dropped from 1.0 to ~0.94 after a single gate) and hermiticity. Fixed
   by correcting the einsum subscript from `"ab,xyaz->xybz"` to
   `"ba,xyaz->xybz"`. Verified afterwards: trace and hermiticity preserved
   to float64 precision (~1e-16) over 6+ steps, and the gamma1=gamma2=0,
   `injection="zz"` limit reproduces an independent PennyLane
   `default.qubit` statevector reference to ~1e-16 (see Definition-of-Done
   tests below) — this is the strongest available correctness check, since
   it validates the entire injection+Trotter tensor-contraction pipeline
   against a completely independent simulator.
2. **Qiskit's little-endian qubit ordering** (qubit 0 = least-significant
   tensor factor) versus this project's big-endian convention (qubit 0 =
   most-significant, used by `NumpyDensityOps` and `correlators.py`)
   caused `drive_qiskit_aer`'s output to disagree with the numpy backend
   by ~0.47 (max abs feature difference) at a 4-qubit sanity scale. Fixed
   by flipping the physical qubit index at Qiskit gate-application time
   (`q(i) = n_qubits - 1 - i`) rather than permuting the returned matrix.
   After the fix, `numpy` vs `qiskit_aer` and `numpy` vs `pennylane_mixed`
   agree to ~1e-14/1e-15 (float roundoff) at small qubit counts for both
   `injection="ry"` and `injection="zz"`.

Both bugs were caught *because* three independent backends were built and
cross-checked, not because any one of them was assumed correct — this is
exactly what the spec's "benchmark all three" requirement bought, beyond
just picking a fastest backend.

## Runtime gate: GO/NO-GO decision

<!-- Numbers below match results/sequential_backend_benchmark.json,
     produced by scripts/benchmark_sequential_backends.py. -->

Measured wall-clock per step (all backends implement the identical
per-step update: RY injection + TFIM Trotter M=10 + amplitude damping +
dephasing):

| Backend | n_qubits | steps timed | s/step |
|---|---|---|---|
| numpy | 12 | 3 | ~120.8 |
| numpy | 10 | 3 | ~6.59 |
| qiskit_aer | 12 | 3 | ~87.1 |
| pennylane_mixed | 12 | 1–2 | (slowest of the three; see results file — still substantially slower than numpy/qiskit_aer at n=12 in ad hoc testing) |

**Deviation from spec**: the spec asked for a 200-step benchmark. At
~87–121 s/step for the 12-qubit backends, 200 steps would already cost
5–7 wall-clock hours *per backend* — impractical within this sprint's
compute budget. Per-step cost is a fixed-size dense contraction (same
shape every step regardless of input), so a 1–3 step measurement is a
statistically stable basis for the steps/sec estimate; this is recorded
as a deviation rather than silently presented as if 200 steps ran.

**Projection for the Sprint 4 pilot** (4,350 samples x washout 24 -> 105k
naive reservoir steps; with trajectory caching, implemented by default,
this collapses to one continuous ~4,374-step trajectory — see
`sequential.py`'s module docstring for why this is equivalent to the
rewinding protocol under the ESP assumption):

- **12 qubits, fastest backend (qiskit_aer, ~87.1 s/step): ~106 hours**
  even with caching. **NO-GO** against the 12h threshold.
- **10 qubits, numpy backend (~6.59 s/step): ~8.0 hours** with caching.
  **GO** against the 12h threshold.

**Decision: fall back to 10 qubits for Sprint 4** (contingency (i) in the
spec), keeping trajectory caching enabled by default (contingency (iii),
already implemented rather than conditional). Washout reduction to 12
(contingency (ii)) was not additionally needed given the 10-qubit number
already clears the gate, but remains available if further headroom is
needed later.

This is a real, load-bearing finding, not a placeholder: it means the
12-qubit reference configuration from Sprint 0 cannot be used for the
*sequential* reservoir as currently implemented (it can still be used for
the existing v4 windowed `AtmosphericQRC`, which is a much cheaper
per-sample statevector computation, not a 105k-step density-matrix
trajectory). Sprint 2+ planning should treat 10 qubits as the working
configuration for the sequential/dissipative line of work specifically.

## ESP verification (sequential mode)

`reservoir/esp_sequential.py` drives a reference (vacuum) trajectory and
8 random Ginibre-ensemble initial density matrices through the same input
sequence and tracks mean pairwise trace distance. At 4 qubits (test scale,
`gamma1=0.15`, `gamma2=0.1`, 20 steps), trace distance shrinks
monotonically and drops below the 1e-2 threshold well within `washout=20`
steps (see `tests/test_sequential.py::test_esp_convergence_under_damping`,
passing). A qualitative purity trajectory at 4 qubits (`gamma1=0.1,
gamma2=0.05`) shows purity falling from 1.0 to ~0.87 after step 1 and
settling near ~0.09–0.10 by step 15 — clear fading-memory behavior driven
by the damping channel, not by injection alone (gamma1=gamma2=0 keeps the
state exactly pure, confirmed by the zero-noise unit test).

A dedicated 12-qubit/washout-24 ESP run (the literal Sprint-1-spec scale)
was not performed given the runtime numbers above make even a single
25-step ESP check cost ~50+ minutes per backend; the mechanism is
validated at test scale and the *mechanism itself* (amplitude damping
provably contracts trace distance; verified generically, not qubit-count
-dependent) is what the spec is checking for the go/no-go decision. This
is flagged as a gap, not silently elided.

## NARMA10 micro-validation

**Does not meet the Sprint 1 target (NMSE < 0.4) after three attempts.**
All three numbers are real, in `results/narma10_validation.json` /
`results/narma10_features.npz`, not cherry-picked:

| Attempt | Fix applied | reservoir NMSE | AR-baseline NMSE | beats AR? |
|---|---|---|---|---|
| 1 | none (spec defaults) | 1.211 | 0.146 | no |
| 2 | broadcast scalar to all qubits + `input_scaling=3.0`, 120 steps | 1.026 | 0.146 | no |
| 3 | same, 400 steps + TimeSeriesSplit-CV'd Ridge alpha | 1.117 | 0.099 | no |
| 4 | + fixed random per-qubit input weights (breaks permutation symmetry) | **0.945** | 0.099 | no |

Root causes identified and fixed along the way (each is a real, checkable
finding, not speculation):

1. **Attempt 1**: the scalar NARMA10 input was injected onto a single
   unused feature slot of the 13-dim RY-injection scheme (built for the
   weather feature vector), leaving 9/10 qubits receiving no input at all.
   Fixed by broadcasting the scalar onto every qubit's RY angle (standard
   single-input reservoir-computing practice) and raising `input_scaling`
   to 3.0 (`u` in [0, 0.5] otherwise barely rotates the qubits).
2. **Attempt 2→3**: only 77–91 post-washout training samples against 165
   correlator features (10 qubits) is a severely underdetermined
   regression regardless of reservoir quality. Fixed by increasing to 400
   driven steps (280 train / 120 test) and selecting Ridge's alpha via
   `TimeSeriesSplit` (project convention) over `[1e-2 .. 1e2]` instead of a
   fixed `alpha=1e-3`.
3. **Attempt 3→4**: broadcasting the *identical* scalar to every qubit,
   combined with the TFIM's uniform (qubit-independent) J and g, makes the
   whole reservoir permutation-symmetric under qubit exchange. Inspecting
   `results/narma10_features.npz` directly showed the smoking gun: many of
   the 165 correlator features were numerically identical across time
   (std ~0.03, several columns equal to ~15 significant figures) — the
   165-dim feature vector was carrying only ~6 independent degrees of
   freedom, replicated. Fixed with a fixed random per-qubit input-weight
   vector (`rng.uniform(0.5, 1.5, size=n_qubits)`, seeded), analogous to an
   ESN's random `W_in`, which breaks the qubit-exchange symmetry.

**Still short of target after all four fixes.** Diagnostic on the
attempt-4 features: the single best-correlated individual feature has
`|corr| = 0.42` with the target, while the trivial single-lag correlation
`corr(u[t-1], y[t]) = 0.54` is *higher* — i.e. the reservoir's best
feature is less informative than one raw lagged input sample. This means
the remaining gap is a genuine reservoir-quality/hyperparameter problem
(most likely `gamma1=0.05` erasing memory faster than NARMA10's
lag-10 requirement, and/or `washout=10` interacting badly with per-step
damping), not an evaluation-setup artifact — the setup artifacts (feature
symmetry, sample count, regularization) are now fixed.

**Next steps for Sprint 2** (not done here — each 400-step trial costs
~40–50 minutes at the 10-qubit fallback's measured throughput, and this
is a multi-dimensional hyperparameter search): sweep `gamma1` downward
(e.g. 0.01–0.03) to preserve memory closer to NARMA10's lag-10 depth,
sweep `input_scaling`, and try `injection="zz"` for richer feature
diversity than the RY round-robin gives a scalar input. This report does
not claim `meets_target: true` — `results/narma10_validation.json` says
`false`, and that is what is being reported.

## Definition of Done — verification

- Runtime gate decision: recorded above with steps/sec numbers per
  backend, from `results/sequential_backend_benchmark.json`.
- NARMA10 sanity pass: **not achieved** — best result NMSE=0.945 against a
  target of <0.4, still behind the linear AR baseline (0.099). See
  `results/narma10_validation.json` (current) and the attempt table above
  for the full trail. Reported as a real negative result and carried into
  Sprint 2 as follow-up work, not silently reported as passing.
- ESP verified in sequential mode: yes, at test scale (4 qubits), via
  `tests/test_sequential.py::test_esp_convergence_under_damping` and the
  purity-decay trajectory above; not at the literal 12-qubit/washout-24
  spec scale, for the runtime reasons given above.
- Tests green: `pytest tests/ -v` — 17 passed (12 Sprint-0 tests +
  5 new `tests/test_sequential.py` tests), see below.
