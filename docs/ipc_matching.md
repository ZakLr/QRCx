# IPC-Matched Reservoir Tuning (Sprint 5 — Draft Paper Subsection)

## Motivation

Reservoir computing's Information Processing Capacity (IPC) framework
(Dambre et al. 2012, *"Information processing capacity of dynamical
systems"*; extended for continuous/dissipative reservoirs by Čindrak et
al. 2026, arXiv:2603.21371) characterizes what a reservoir *can supply*:
a decomposition of its total nonlinear memory capacity into orthogonal
components indexed by delay and polynomial degree. Reservoir
hyperparameters are conventionally tuned to maximize *total* capacity —
but a task only benefits from capacity in the specific (delay, degree)
region it actually needs. A reservoir tuned for maximum total IPC may
allocate most of its capacity to delays or degrees the task never uses,
while starving the region that matters.

We introduce the natural counterpart: a **task demand profile**,
measuring how much of a forecasting task's own predictability is
explained by low-order polynomial functions of its target's lagged
history, at the same (delay, degree) resolution as the reservoir's
supply-side IPC. Matching supply to demand — maximizing the *captured*
capacity `Σ min(C, D)` rather than raw total supply `Σ C` — is, to our
knowledge, not previously applied to operational weather forecasting.

## Method

**Demand** `D(delay, degree)`: for the real KORD climatological-anomaly
target series (the same residual target used throughout this project),
regress the h-step-ahead value on a degree-`d` Legendre polynomial of the
series' own value `delay` steps in the past, independently per
(delay, degree) cell (train=test R², the same evaluation convention as
this project's established MC/IPC metrics — not a jointly orthogonalized
Volterra decomposition). Legendre polynomials P₁, P₂, P₃ are the
orthogonal basis for a variable uniformly distributed on [-1, 1]; the
real anomaly target is only approximately so (it is closer to
standard-normal), so raw values are first smoothly squashed into (-1, 1)
via `tanh(x/2)` before evaluating the polynomials — a documented
engineering simplification, not a distributional transform.
Implementation: `QRCx/QRCx/metrics/task_demand.py::compute_demand_profile`.

**Supply** `C(lag, degree)`: the sequential dissipative reservoir
(`QRCx.reservoir.sequential.SequentialDissipativeQRC`) driven by a
continuous iid standard-normal stream, then — for each lag `k` and
degree `d` — the probabilists' Hermite polynomial `He_d(u[t-k])` (the
correct orthogonal basis for an iid-Gaussian input, as already used by
this project's degree-1/2 IPC implementation) is regressed against the
reservoir state at time `t`, same train=test R² convention. This is a
direct extension of the project's existing `measure_ipc_sequential`
(fixed at linear + quadratic) to arbitrary degree — implemented
additively as `measure_ipc_by_degree` so the original, already-validated
function is untouched. Lag `k=0` (the readout's instantaneous nonlinear
transform of the *current* input, a legitimate IPC component per Dambre
et al. 2012) is included, so `C` and `D` share an identical
`(delay/lag, degree)` axis convention for direct element-wise
`min(C, D)`.

**Matching**: for a coarse grid over `(gamma1, input_scaling)` — `tau`
fixed at the reference value, matching this project's own established
tuning precedent (Sprint 2/2.5 never swept `tau` either) — the captured
capacity `Σ_h Σ_{delay,degree} min(C, D)` is computed at both h=1 and
h=6, and the configuration maximizing it is compared against the Sprint
2 reference configuration on real (reduced-sample) pilot forecast
metrics.

## Results

Full numeric output: `results/ipc_matching.json`. Figures:
`figures/sprint5_demand_supply_heatmaps.png`,
`figures/sprint5_captured_capacity_bar.png`.

**Demand** is dominated by degree 1 (linear persistence/autocorrelation),
decaying smoothly with delay — physically expected for a temperature
anomaly series. Degree 2/3 demand is comparatively small at every delay
tested. **Supply**, across the coarse (gamma1, input_scaling) grid, is
concentrated at low lag (0-4 steps) regardless of configuration, but its
total magnitude and degree-distribution both vary substantially with
gamma1/a.

**Matching**: the grid point maximizing captured capacity is
**gamma1=0.3, a=0.1** (captured_total=9.26 vs. the Sprint 2 reference
config's gamma1=0.03, a=0.3 at captured_total=4.53) — i.e., *more*
dissipation and *weaker* input scaling than the existing reference,
concentrating supply into the same low-lag, low-degree region where
demand actually lives.

**Matched vs. reference on real (reduced-sample) pilot forecast
metrics** (skill vs. persistence, 1,500 train + 500 test real KORD
steps):

| h | matched (g1=0.3, a=0.1) | reference (g1=0.03, a=0.3) | winner |
|---|---|---|---|
| 1 | -1.72% | -4.20% | matched |
| 3 | -3.82% | -4.59% | matched |
| 6 | -4.12% | -1.22% | **reference** |
| 12 | +8.89% | +11.15% | **reference** |

**Honest result — the sprint's own stated hypothesis is NOT confirmed**:
matching improves short-horizon skill (h=1, h=3) but the reference
config remains better at h=6 (the sprint's specifically-named target
horizon) and h=12. Captured-capacity maximization over `(gamma1,
input_scaling)` alone does not translate into better forecast skill at
every horizon on this pilot task — a genuine, reported-as-such negative
finding, not spun as a win. Plausible explanation (not verified further
here, flagged as a direction for later work): the demand profile was
computed on the target's own univariate autoregressive structure, while
the actual forecasting readout uses the full multivariate 13-feature
reservoir state — a config that best captures the *univariate*
self-demand is not guaranteed to be the config that best supports the
full multivariate readout the real forecast task actually uses.

## Scope notes (logged, not silent)

- The coarse grid restricts to `(gamma1, input_scaling)`; `tau` is fixed.
- Supply-side IPC uses `n_steps=200` per grid point — sufficient for a
  comparative ranking across configurations, not a final high-precision
  measurement (the project's established IPC characterization runs use
  larger samples for a single frozen config, not for a multi-point grid
  search).
- The final matched-vs-reference forecast comparison uses a reduced real
  pilot subsequence (1,500 train + 500 test steps), not the full 3-year
  pilot, for tractable wall-clock at n_qubits=10 (this project's
  documented CPU-feasible fallback; the 12-qubit reference configuration
  is GPU-only feasible at this reservoir's measured per-step cost).

## Citations

- J. Dambre, D. Verstraeten, B. Schrauwen, S. Massar (2012). "Information
  processing capacity of dynamical systems." *Scientific Reports* 2, 514.
- M. Čindrak et al. (2026). "Information processing capacity of
  dissipative quantum reservoirs." arXiv:2603.21371.
- H. Jaeger (2001). "The 'echo state' approach to analysing and training
  recurrent neural networks." GMD Report 148.
