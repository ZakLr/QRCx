# Sprint 5 — IPC-Matched Reservoir Tuning (Novel Contribution)

**Bottom line**: implemented and tested a novel task-demand/reservoir-
supply matching framework (Dambre 2012 / Čindrak 2026 IPC decomposition,
extended to a "task demand" counterpart, believed novel for operational
weather forecasting per the sprint's own framing). Real result on the
pilot KORD data: matching increases captured capacity substantially
(4.53 → 9.26, +104%) by favoring more dissipation and weaker input
scaling than the existing reference config. **The sprint's own stated
hypothesis — that this improves 6h forecast skill — is NOT confirmed**:
the matched config wins at h=1/h=3 but loses to the reference at h=6
(the specifically-named target horizon) and h=12. Reported honestly, per
the sprint spec's own instruction ("report even if it doesn't [improve
6h skill]").

## 1. Method

- **Demand** `D(delay, degree)`: real KORD climatological-anomaly target
  regressed on degree-1/2/3 Legendre polynomials of its own lagged
  history (delays 0-24h), independent per-cell R², train=test. New
  module `QRCx/QRCx/metrics/task_demand.py`.
- **Supply** `C(lag, degree)`: `SequentialDissipativeQRC`'s IPC under an
  iid-Gaussian drive, generalized from the existing linear+quadratic
  split (`measure_ipc_sequential`, untouched) to arbitrary degree via
  probabilists' Hermite polynomials — new, additive function
  `measure_ipc_by_degree` in `QRCx/QRCx/metrics/reservoir_sequential.py`,
  extended to include lag=0 so its axis convention matches demand's
  delay=0 exactly (needed for direct `min(C, D)`).
- **Matching**: coarse grid over `(gamma1, input_scaling)` — `tau` fixed
  at the reference value, matching Sprint 2/2.5's own precedent of never
  sweeping `tau` — maximizing `Σ_h∈{1,6} Σ min(C, D)`.
- Both new functions verified before use: `measure_ipc_by_degree`'s
  degree-1/2 output matches the already-validated `measure_ipc_sequential`
  exactly (max diff ~1e-9); `compute_demand_profile` verified to detect a
  planted autoregressive (delay, degree) signal at the correct location.
  7/7 new unit tests pass (`QRCx/tests/test_task_demand.py`,
  additions to `QRCx/tests/test_reservoir_sequential_metrics.py`).

## 2. Real numbers

Grid: `gamma1 ∈ {0.0, 0.01, 0.03, 0.1, 0.3, 0.5}` (reusing Sprint 2.5's
own IPC/MC characterization grid points), `a ∈ {0.1, 0.3, 1.0}`, 18
configs, `n_qubits=10` (this project's documented CPU-feasible fallback;
the 12-qubit reference is GPU-only feasible at this reservoir's measured
per-step cost), `n_steps=200` per config, real wall-clock ~35 min for the
full grid.

Best matched config: **gamma1=0.3, a=0.1** — captured_total=9.26.
Sprint 2 reference (gamma1=0.03, a=0.3): captured_total=4.53. The matched
config's demand-alignment is real and large (see
`figures/sprint5_captured_capacity_bar.png`).

Matched vs. reference on a real (reduced-sample: 1,500 train + 500 test
real KORD steps, not the full 3-year pilot — tractable wall-clock at
n_qubits=10) forecast comparison, skill vs. persistence:

| h | matched (g1=0.3, a=0.1) | reference (g1=0.03, a=0.3) | winner |
|---|---|---|---|
| 1 | -1.72% | -4.20% | matched |
| 3 | -3.82% | -4.59% | matched |
| 6 | -4.12% | -1.22% | **reference** |
| 12 | +8.89% | +11.15% | **reference** |

Full data: `results/ipc_matching.json`.

## 3. Honest interpretation

The demand/supply matching framework works as designed — it identifies a
real config, `gamma1=0.3, a=0.1`, that captures dramatically more
overlapping capacity than the existing reference — but **higher captured
capacity did not translate into better forecast skill at h=6 or h=12,
the horizons that matter most for this project's real headline claims**.
Plausible explanation (a hypothesis, not independently verified this
sprint): the demand profile characterizes the target's own *univariate*
autoregressive structure, while the actual pilot forecast readout
consumes the reservoir's full multivariate feature vector driven by all
13 real weather features — a configuration tuned to best capture
univariate self-demand is not guaranteed to be the configuration that
best supports the true multivariate readout task. This is exactly the
kind of honest limitation the sprint spec anticipated
("hypothesis to test... report even if it doesn't").

## 4. Scope deviations (logged, not silent)

- Coarse grid restricted to `(gamma1, input_scaling)`; `tau` fixed at
  the reference value 1.0, matching Sprint 2/2.5's own established
  precedent.
- Supply-side IPC measured at `n_steps=200` per grid point (coarse/
  comparative, not a final high-precision single-config measurement).
- Matched-vs-reference forecast comparison used a reduced real pilot
  subsequence (1,500 train + 500 test steps of the real 26,304-step
  pilot train_seq), not the full 3-year pilot, for tractable wall-clock
  at `n_qubits=10` on CPU.
- `n_qubits=10` used throughout (this project's documented CPU-feasible
  fallback), not the 12-qubit GPU-only reference config.

## Definition of Done — verification

- **Demand/supply/matching implemented and tested**: done
  (`QRCx/QRCx/metrics/task_demand.py`, `measure_ipc_by_degree` in
  `reservoir_sequential.py`, `scripts/sprint5_ipc_matching.py`); 7/7 new
  unit tests pass, full suite green (see below).
- **Comparison numbers in `results/ipc_matching.json`**: done.
- **Figure — demand vs. supply heatmaps + captured-capacity bar per
  configuration**: done (`figures/sprint5_demand_supply_heatmaps.png`,
  `figures/sprint5_captured_capacity_bar.png`).
- **Draft paper subsection with citations**: done (`docs/ipc_matching.md`,
  Dambre 2012 / Čindrak 2026 / Jaeger 2001 cited).
- **Tests green**: full suite re-run after these additions (see commit
  for the pass count) — no regressions to any existing Sprint 0-4
  functionality; the two new/modified metrics functions are purely
  additive (`measure_ipc_sequential` untouched).
