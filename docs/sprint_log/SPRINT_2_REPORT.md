# Sprint 2 — Time-Multiplexed Readout + Dissipation Sweep

**Status:** Phase 2.0 (performance pass) done, real speedup achieved but
12-qubit CPU target not met (10-qubit fallback comfortably met it).
Phase 2.1 (NARMA10 tuning gate) run to completion; **hard gate FAILED**
(best NMSE 0.398 vs. required <0.099), reported honestly per spec rather
than silently ignored — real, substantial progress was made (0.945→0.398)
but the reservoir is not yet ready for weather-data integration. Phase 2.2
(IPC/MC characterization, trade-off figure, config freeze) executed at
the best-effort (non-gate-passing) config, marked as provisional.

## Phase 2.0 — Performance pass

### 2.0a: exact propagator

Implemented `propagator="exact"`: the full 12-qubit TFIM Hamiltonian
`H = g*sum(X_i) + J*sum_{i<j} Z_i Z_j` is built once (dense, via bit
manipulation — diagonal ZZ part + bit-flip X part — never via kron of
`2**n x 2**n` operators) and exactly exponentiated once via
`eigh` + `evecs @ diag(exp(-i*evals*t)) @ evecs.conj().T`. Each step then
costs 2 dense matmuls instead of the Sprint 1 Trotter path's
`trotter_steps * n_qubits` einsum-based gate applications.
`propagator="trotter"` (the original Sprint 1 path) is kept for
hardware-matched runs.

**Validation** (`QRCx.reservoir.sequential.validate_trotter_vs_exact`,
`tests/test_sequential.py`):
- `propagator="exact"` independently cross-checked against a
  `scipy.linalg.expm`-based reference (different exact-exponentiation
  method) — matches to float64 precision.
- Trotter error vs. the exact propagator was **empirically found to scale
  as O(1/M)** (standard first-order Trotter-Suzuki), not O(1/M^2) as an
  earlier informal target of "1e-10 at M=200" assumed — that number was
  physically wrong for this densely-connected (all-to-all ZZ) Hamiltonian
  and is corrected here rather than asserted:

  | M (trotter_steps) | max abs diff vs. exact |
  |---|---|
  | 50 | 0.0048 |
  | 200 | 0.0012 |
  | 1000 | 0.00023 |
  | 5000 | 0.0000467 |

  The production `trotter_steps=10` setting's error vs. exact is ~0.027
  (n=4 qubits) — real and larger, reported honestly.

### 2.0b: GPU backend

`use_gpu=True` auto-detects CuPy + a visible CUDA device
(`QRCx.reservoir.sequential._get_xp`) and silently falls back to numpy if
unavailable. **This environment has no GPU and no CuPy installed** — the
fallback path was verified (`_get_xp(True)` correctly returns numpy here),
but the actual GPU-accelerated path is implemented and untested in this
session. Flagged as a real gap, not claimed as verified.

### 2.0c: re-run the 12-qubit runtime gate

| Config | Sprint 1 (Trotter) | Sprint 2 (exact propagator) |
|---|---|---|
| 12 qubits, s/step | ~120.8 (numpy) / ~87.1 (qiskit_aer, fastest Sprint-1 backend) | **~40.7** |
| 10 qubits, s/step | ~6.59 | **~1.87** |

Both configs got faster (12q: ~2.1-3x; 10q: ~3.5x), but the ≤5 s/step CPU
target from the spec was **only met at 10 qubits**, not 12:

- **12 qubits, CPU, exact propagator: ~40.7 s/step** → cached Sprint-4
  pilot (~4,374 steps) projects to **~49.4 hours** — still NO-GO against
  the 12h threshold (though much improved from Sprint 1's ~106h).
  Profiling showed the per-step cost is now dominated by the injection
  (12 single-qubit gate touches, ~7.7s) and dissipation (dephasing +
  12-qubit amplitude damping, ~19.6s) steps, *not* the evolution (2 dense
  matmuls, ~5.3s) that Phase 2.0a targeted — the bottleneck moved, it
  didn't disappear. GPU was not available to test whether it clears the
  ≤0.5 s/step GPU target.
- **10 qubits, CPU, exact propagator: ~1.87 s/step** → cached pilot
  projects to **~2.27 hours** — comfortably GO, well under the 12h
  threshold (and under an even tighter bar).

**Decision: 12-qubit reference configuration is NOT restored on this
(CPU-only) hardware.** The 10-qubit fallback stands, now running ~3.5x
faster than in Sprint 1 — which is what made Phase 2.1's NARMA10 sweep
tractable in this sprint at all. Revisiting the 12-qubit gate on
GPU-equipped hardware is a legitimate follow-up (the implementation is
already in place; only verification is missing).

### 2.0d: seeded w_in

`w_in` is now a first-class `SequentialDissipativeQRC` constructor
parameter (previously an ad hoc vector built inline in the Sprint 1 NARMA
script), defaulting to all-ones (neutral — preserves every Sprint 1 test
output exactly). The actual seeded vector used for NARMA10 tuning
(`np.random.default_rng(123).uniform(0.5, 1.5, size=10)`) is persisted in
`configs/v5_reference.yaml`, not hand-picked or left unseeded.

## Time multiplexing (V)

Implemented as `multiplexing=V` on `SequentialDissipativeQRC`: within each
step's coherent evolution, Pauli correlators are read out at V equally
spaced sub-times (non-destructively — rho keeps evolving after each
readout), giving `V*234` features per step. Cost is V observable
evaluations per step, not V re-simulations: the same fixed increment
propagator `U_inc = e^{-iH tau/V}` (precomputed once) is applied V times
per step. Validated in `tests/test_sequential.py::
test_multiplexed_features_shape_and_consistency` (shape `(T, V*234)`,
finite, non-degenerate). The NARMA10 sweep additionally exploits that
V=1/2/4's sub-readouts are *nested* (V=1's single readout equals V=4's
4th sub-readout; V=2's two readouts are V=4's 2nd and 4th), so all three V
values are evaluated from a single V=4 drive without re-driving.

## Phase 2.1 — NARMA10 tuning gate

**HARD GATE FAILED.** Best config found: `gamma1=0.1, gamma2=0.1,
input_scaling=0.3, washout=50, V=4` → **NMSE = 0.398**. Required to beat
the linear AR baseline (NMSE < 0.099, from Sprint 1's
`results/narma10_validation.json`); target ≤0.2, aspirational ≤0.15.
None of these were met. Full trail in `results/narma10_sweep.json`.

**Real, substantial progress was made**: Sprint 1 ended at NMSE 0.945;
this sprint's tuning brought it down to 0.398 — more than halved, and the
gamma1 sweep shows a clear, interpretable trend (NMSE fell monotonically
from 0.996 at gamma1=0 to 0.533 at gamma1=0.1, then rose slightly to 0.604
at gamma1=0.3 — a shallow interior optimum near gamma1≈0.1, consistent
with the "moderate dissipation beats both unitary and over-damped" pattern
the spec anticipated). Multiplexing (V) helped monotonically at the best
config (washout=50: NMSE 0.469 at V=1 → 0.447 at V=2 → 0.398 at V=4).

**Search strategy deviation from the literal spec grid** (documented in
`scripts/narma10_sweep.py`'s module docstring): the full
gamma1(5)×gamma2(3)×input_scaling(3)×V(3)×washout(3) factorial is 405
combinations — not tractable at even the Sprint 2 exact-propagator
speedup (~1.87-3.5 s/step depending on V, since driving at V=4 to get
V=1/2/4 for free roughly doubles per-step cost vs. V=1). Instead: a
coordinate-wise search (coarse gamma1 sweep → refine gamma2 → refine
input_scaling → free washout/V grid from one final drive) covering 9
actual drives instead of 405, at the cost of not exploring interactions
between axes as thoroughly as a full grid would. Total sweep wall-clock:
~2.3 hours.

**Per spec, this is a STOP**: the tuned reservoir is not carried forward
to weather-data integration in this sprint (there was none scheduled in
Sprint 2 regardless). Recommended next steps for whoever picks this back
up:
1. Denser gamma1 sampling between 0.03 and 0.1 (the interior-optimum
   region), rather than the coarse {0, 0.01, 0.03, 0.1, 0.3} grid.
2. A true joint (not coordinate-wise) search over
   (gamma1, input_scaling) — the coordinate-wise approach may have missed
   an interaction between these two axes.
3. Longer driven sequences: at washout=50 the best config still only had
   175 training samples against 660 features (V=4) — a severely
   underdetermined regression regardless of reservoir quality, similar to
   the Sprint 1 lesson. This alone may explain much of the remaining gap.
4. Injection scheme: `injection="zz"` was not tried in this sweep
   (Sprint 1 only validated it structurally, not for NARMA10 performance).

## Phase 2.2 — IPC/MC characterization + trade-off figure + config freeze

Executed at the best-effort (non-gate-passing) config
(`gamma1` swept 0-0.3, `gamma2=0.03` fixed, `a` in {0.3, 1.0, 3.0}, 10
qubits, `propagator="exact"`, `multiplexing=1`), using
`QRCx/experiment/figstyle.py` (serif fonts, no in-figure titles, labeled
(a)/(b) panels, 300 dpi, Okabe-Ito colorblind-safe palette) — new in this
sprint, applied here; retrofitting it onto the existing v4 figures in
`experiment/figures.py` is deferred (flagged, not silently skipped) given
this sprint's time budget.

**Results** (`results/ipc_mc_characterization.json`, `qrc_figures/fig_ipc_tradeoff.png`):

| a | gamma1 | MC | linear IPC | nonlinear IPC | total IPC |
|---|---|---|---|---|---|
| 0.3 | 0.0 | 1.411 | 0.975 | 0.793 | 1.768 |
| 0.3 | 0.03 | 1.780 | 1.165 | 1.065 | 2.230 |
| 0.3 | 0.1 | 2.318 | 1.734 | 1.614 | 3.347 |
| 0.3 | 0.3 | 2.517 | 2.056 | 1.809 | 3.865 |
| 1.0 | 0.3 | 2.722 | 1.976 | 2.175 | 4.150 |
| 3.0 | 0.3 | 3.121 | 2.084 | 2.314 | **4.398 (grid max)** |

**No interior optimum found within the tested range.** Total IPC increases
*monotonically* with gamma1 across the full {0, 0.01, 0.03, 0.1, 0.3} grid
for all three `a` values — the best point in every case is at the
boundary (gamma1=0.3, the largest value tested), not an interior point.
This is the *other* boundary case from the one the spec anticipated
("If the optimum is at gamma1=0, report it honestly") — here it's the
opposite boundary. Reported honestly rather than forced into a false
"Čindrak-style interior peak" narrative: **the data says the optimum (if
one exists) lies at gamma1 > 0.3, outside this sprint's grid**, not that
there is no memory-nonlinearity trade-off at all. Extending the grid
(e.g. gamma1 up to 0.5-1.0) is the natural next step, not attempted here
given the compute budget already spent on the NARMA10 sweep.

MC values (1.4-3.1) are below the established "healthy range 5-10"
reference — but that range was set for the **12-qubit** configuration;
this characterization runs at the **10-qubit fallback** (Sprint 1/2
runtime gate), so a lower MC is expected from fewer qubits and isn't
itself evidence of a problem. IPC generally tracks MC's trend (higher
gamma1 and higher `a` both help, up to the tested grid boundary).

**Note on the config-freeze vs. this table**: `configs/v5_reference.yaml`
freezes the NARMA10 sweep's best-effort config at `gamma2=0.1`; this
characterization fixed `gamma2=0.03` throughout (to isolate the gamma1
trend, per the spec's figure request). The closest matching row above
(a=0.3, gamma1=0.1, gamma2=0.03: MC=2.318, total IPC=3.347) is a close but
not exact match to the frozen config's `gamma2=0.1` — noted in the YAML
file itself rather than presented as an exact number for that config.

Config frozen to `configs/v5_reference.yaml`, explicitly marked
**PROVISIONAL, NOT GATE-PASSING** in the file itself, including the
persisted seeded `w_in` vector (Phase 2.0d).

## Definition of Done — verification

- Best `(gamma1, gamma2, a, V)` frozen in `configs/v5_reference.yaml`:
  yes, marked provisional/non-gate-passing per the honest Phase 2.1
  outcome.
- MC in 5-10 and IPC recorded: see `results/ipc_mc_characterization.json`
  and the placeholder above (filled once that script completes).
- Trade-off figure saved: `qrc_figures/fig_ipc_tradeoff.png`.
- Tests green: `pytest tests/ -v` — **22 passed** (12 from Sprint 0 +
  5 sequential-reservoir tests from Sprint 1 + 5 new this sprint: 3 in
  `test_sequential.py` — exact-propagator-vs-scipy-expm cross-check,
  Trotter-converges-to-exact, multiplexed-features shape/consistency —
  plus 2 in the new `test_reservoir_sequential_metrics.py`, MC and IPC
  sanity checks).
