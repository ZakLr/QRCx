# Sprint 3 — Bulletproof Baselines & Fairness Protocol

**Why this sprint exists:** the classical ESN baseline was reporting
skill around -400% to -1400% vs. persistence — a misconfiguration, not a
real comparison. Since beating ESN is what justifies the QRC, this had to
be fixed and locked behind a fairness protocol no judge can attack before
any QRC-vs-classical comparison is meaningful.

**Bottom line**: found and fixed two real bugs, not a tuning problem.
The ESN was feeding reservoirpy flattened 24h windows instead of the
genuine hourly sequence (a target/input-alignment bug — no amount of
hyperparameter tuning fixes what sequence gets fed in) and never tuned
`leak_rate`, silently defaulting to 1.0 (no leaky integration), badly
suited to a task where persistence is a very strong 1h-lag baseline.
ARIMA separately returned the identical forecast array for every
horizon. Both fixed. Tuned ESN (full 384-point grid, validation-selected)
now achieves **+6.2% skill at 1h, +14.6% at 6h** on real KORD data —
**DoD met** (`results/esn_diagnosis.json`). The 13-feature gap flagged in
README's known limitations is closed. A full fairness protocol
(dimension-matched ESN, null-control Ridge/KRR, Residual-ESN/Ridge) and a
new significance-testing module (Diebold-Mariano + moving-block
bootstrap, unit-tested against `statsmodels`) are in place. All 33 tests
pass.

## Phase A — Feature-set gap closed

Ported `pipeline_demo.py`'s `engineer_features()` (13 columns: `T_db,
T_dew, RH, WS, SLP, WD, Wx, Wy, T_dep, hour_sin, hour_cos, doy_sin,
doy_cos`) into the canonical `QRCx/data/preprocessor.py::build_features`,
replacing the previous 10-feature set (`θ, VPD, u, v` dropped, not merged
alongside — the two conventions compute the same wind decomposition
under different names, `u/v` vs. `Wx/Wy`; keeping both would have been a
silent duplicate). `preprocess()` now also returns `train_seq`/
`val_seq`/`test_seq` (the raw, non-windowed per-split scaled anomaly
sequences) and `train_valid_idx`/`val_valid_idx`/`test_valid_idx` (the
window-start positions retained after NaN filtering) — both needed by
Phase B's ESN fix. `QRCx/data/loader.py` was not touched (frozen, per
policy); real 2019-2024 KORD data was already cached locally
(`data/isd/`) and used directly for every diagnosis in this sprint.

## Phase B — ESN diagnosis and fix

Reproduced the failure first, on real data, before changing anything
(`results/esn_diagnosis.json`, stage `"broken"`): a frozen replica of the
pre-Sprint-3 `baselines/esn.py` scores -705.9%/-53.5% skill at h=1/h=6.

**Root cause**: the old code did
`X_flat = X_train.reshape(n, -1)` — flattening each QRC-style 24x13
window into one ~312-dim vector and feeding reservoirpy one such vector
per sample. Consecutive samples share 23/24 hours of overlap, so
reservoirpy's own recurrent state was integrating over a highly
redundant, discontinuous re-encoding of the true hourly series, not a
signal its memory can exploit. `Reservoir`'s leak rate (`lr`) was also
never set, defaulting to 1.0 (no leaky integration) — a second,
independent bug.

**Fix** (`QRCx/baselines/esn.py`, mostly rewritten):
- `prepare_sequential_targets()` feeds the genuine raw hourly sequence
  (`train_seq`/`val_seq`/`test_seq`), aligning input row `i` to target
  `seq[i+h, target_col_idx]` — reservoirpy's native convention.
- `tune_esn()` grid-searches `spectral_radius ∈ {0.7,0.9,0.99,1.1}`,
  `leak_rate ∈ {0.1,0.3,0.6,1.0}`, `input_scaling ∈ {0.1,0.5,1.0}`,
  `ridge α ∈ logspace(-8,-1,8)` (384 points), `washout=100`, selected
  against `val_seq` only — with a `fast_mode` flag (reduced 4-point grid)
  for tractable dev/CI runs, per the project's FAST_MODE convention.
- Predictions are re-aligned to the windowed pipeline's NaN-filtered
  sample set via the new `test_valid_idx` (`pred[valid_idx + W - 1]`),
  so every baseline is scored on the identical target set — real ISD data
  drops roughly a third of 24h windows to missing SLP/WD readings, and a
  naive positional slice would have silently compared mismatched samples.
- Added `residual=True` support (targets `seq[t+h]-seq[t]`, persistence
  added back at predict time) for the Residual-ESN fairness baseline.

**Verification** (`results/esn_diagnosis.json`, real KORD data,
ESN-200):

| Stage | h=1 skill | h=6 skill |
|---|---|---|
| Pre-Sprint-3 (broken) | -705.9% | -53.5% |
| Fixed alignment, fast grid (4 combos) | -58.8% | +8.0% |
| Fixed alignment, full grid (384 combos) | **+6.2%** | **+14.6%** |

The fast-grid stage alone (no tuning improvement beyond the alignment
fix) already moved h=1 from catastrophic to only mildly negative,
confirming the alignment bug — not the missing tuning — was the primary
cause. Full-grid wall clock: 464.3s for one ESN-200 fit+tune.

**ARIMA bug (found while building the fairness harness, fixed
alongside)**: the pre-Sprint-3 `baselines/arima.py::forecast()` computed
one `n_test`-length forecast trajectory and returned the *identical*
array for every requested horizon — h=1 and h=6 predictions were bitwise
equal. Fixed to fit once on the raw chronological sequence, forecast
`len(test_seq) + max(horizons) - 1` steps, and slice `fc[h-1:h-1+n]` per
horizon (the standard multi-step convention, matching `pipeline_demo.py`'s
FSDH-curve code), aligned via the same `valid_idx` mechanism as ESN.
Separately, ARIMA(2,1,2) — order fixed by the spec, not retuned — performs
very poorly on the already-stationary anomaly series (`d=1` over-differences
it); reported honestly in `docs/evaluation_protocol.md` §4 rather than
hidden or silently re-tuned outside the spec.

## Phase C — Fairness protocol

New `QRCx/baselines/fairness.py`:
- `null_control_forecast`: Ridge or KRR fit **directly on the flattened
  raw 24x13 window, no reservoir** — the control that would reveal if the
  reservoir buys nothing.
- `residual_ridge_forecast`: same anomaly-residual target
  `architecture/residual.py::ResidualQRC` predicts, flattened-window
  features instead of reservoir features.
- `dimension_matched_esn_size`: QRC feature-count-matched ESN sizing
  (`AtmosphericQRC.n_features` — new property, `3N + 3*C(N,2)` = 234 at
  the reference 12 qubits; confirms the spec's "ESN-936 for V=4" example
  = 234 x 4 time-multiplexing).

`scripts/generate_baselines.py` runs the full protocol (persistence,
ARIMA, dimension-matched/500/5000 ESN, Residual-ESN, null-control
Ridge/KRR, Residual-Ridge — all sharing the identical 13 features, strict
temporal split, and `StandardScaler`) end-to-end on real KORD data and
writes `results/baselines.json`. FAST_MODE (default) uses the reduced
ESN grid and skips ESN-5000 at the full grid (see "Known scope cuts"
below); `--no-fast-mode` runs the full grid. Null-control KRR's training
set is capped at 3000 samples (`KernelRidge` is O(n²) in samples — a
pre-existing limitation of `KRRReadout` used pipeline-wide, not new here,
just newly load-bearing for this baseline; documented in
`docs/evaluation_protocol.md` §7 rather than silently absorbed).

## Phase D — Significance testing

New `QRCx/metrics/significance.py`:
- `diebold_mariano`: squared-error-loss DM test with Bartlett/Newey-West
  HAC variance correction, `maxlags = h - 1` (standard h-step-ahead
  convention).
- `moving_block_bootstrap` / `skill_difference_ci`: Kunsch (1989) block
  bootstrap (block length `n**(1/3)`) 95% CI for skill differences,
  recomputing the ratio-based skill statistic on each resample.

Unit-tested (`tests/test_significance.py`, 9 tests) against
`statsmodels.OLS(..., cov_type="HAC")` — the DM statistic matches to
`rel=1e-8` at both h=1 (`maxlags=0`) and h=6 (`maxlags=5`). One real bug
caught by this cross-check during development: the HAC long-run-variance
autocovariance terms must be normalized by `n` throughout (not `n - lag`
for lagged terms) to match `statsmodels`' meat-matrix convention — fixed
before the test passed.

## Definition of Done — verification

- **Diagnose current ESN failure first, note the cause in the log**:
  done — root cause is window-flattened input misalignment plus an unset
  leak rate, not a hyperparameter problem (Phase B, verified on real
  data before any fix was applied).
- **ESN via reservoirpy, properly tuned, grid + FAST_MODE, washout=100,
  selected on validation only**: done (`baselines/esn.py::tune_esn`).
- **Fairness protocol — identical features/splits/scaler, residual
  decomposition for Residual-ESN/Residual-Ridge, dimension-matched ESN,
  ESN-500/5000, null control, ARIMA(2,1,2), persistence floor**: done
  (`baselines/fairness.py`, `scripts/generate_baselines.py`). ESN-5000 at
  the full 384-point grid was not executed this session (time budget;
  extrapolated multi-hour run — see `docs/evaluation_protocol.md` §7);
  ESN-5000 at the reduced grid was descoped in favor of the DoD-critical
  dimension-matched/ESN-500 sizes at both grids.
- **Metric harness — RMSE, MAE, NRMSE, skill, FSDH h=1..48, VPT, DM test
  with HAC, moving-block-bootstrap CI**: done. FSDH curve computed to
  h=48 in `generate_baselines.py`; DM test and bootstrap CI run for every
  baseline vs. persistence at every eval horizon, written into
  `results/baselines.json`.
- **`QRCx/metrics/significance.py` with tests against statsmodels/known
  values**: done, 9/9 passing, DM stat matches `statsmodels` HAC OLS
  exactly.
- **Baseline results auto-generated to `results/baselines.json`**: done
  (`scripts/generate_baselines.py`); DoD-verifying ESN full-grid run
  additionally recorded to `results/esn_diagnosis.json`
  (`scripts/esn_diagnosis.py`) since the default FAST_MODE grid alone
  does not clear the DoD bar and the invariant requires every reported
  number to be traceable to a results file, not just diagnostic console
  output.
- **Tuned ESN achieves skill ≥ -5% vs. persistence at 1h on the pilot
  subset**: **met** — +6.2% (full 384-point grid, real KORD data,
  `results/esn_diagnosis.json`). Note: the default FAST_MODE grid
  (4 points) alone does *not* clear this bar (-58.8% at h=1) — the DoD
  is verified at the full grid, reported as such rather than claimed at
  the faster default.
- **Fairness protocol documented in `docs/evaluation_protocol.md`**:
  done.
- **DM test unit-tested**: done, verified against `statsmodels`.
- **Tests green**: `pytest tests/ -v` — **33 passed** (24 carried over from
  Sprint 0-2.5, one shape assertion updated for the 13-feature change,
  plus 9 new in `tests/test_significance.py`).
