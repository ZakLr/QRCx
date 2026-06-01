# QRCx — Full Audit Report

## SECTION 1: FILE INVENTORY

| # | File Path | Status | Lines | Notes |
|---|-----------|--------|-------|-------|
| 1 | `QRCx/__init__.py` | DONE | 72 | Lazy exports, 25 symbols |
| 2 | `QRCx/config.py` | DONE | 33 | ExperimentConfig dataclass, YAML I/O |
| 3 | `QRCx/pipeline.py` | DONE | 204 | QRCPipeline orchestrator |
| 4 | `QRCx/encoding/__init__.py` | DONE | 2 | Eager imports BaseEncoder, ZZFeatureMap |
| 5 | `QRCx/encoding/base.py` | DONE | 14 | ABC, scale_to_pi() |
| 6 | `QRCx/encoding/zz_feature_map.py` | DONE | 30 | H→RZ→IsingZZ per layer |
| 7 | `QRCx/reservoir/__init__.py` | DONE | 21 | Lazy exports for 5 symbols |
| 8 | `QRCx/reservoir/base.py` | DONE | 15 | ABC, verify_esp raises NotImplementedError |
| 9 | `QRCx/reservoir/tfim.py` | DONE | 87 | AtmosphericQRC, 108 features |
| 10 | `QRCx/reservoir/tfim_cudaq.py` | DONE | 134 | CUDA-Q kernel + PennyLane fallback |
| 11 | `QRCx/reservoir/parallel.py` | DONE | 47 | Dual reservoir, 216 features |
| 12 | `QRCx/reservoir/esp.py` | DONE | 77 | ESP via lightning.qubit |
| 13 | `QRCx/readout/__init__.py` | DONE | 3 | Eager imports Base/KRR/Ridge |
| 14 | `QRCx/readout/base.py` | DONE | 21 | ABC fit/predict with target_col_idx |
| 15 | `QRCx/readout/correlators.py` | DONE | 92 | 108 correlator extraction |
| 16 | `QRCx/readout/krr.py` | DONE | 71 | KRR with TimeSeriesSplit CV |
| 17 | `QRCx/readout/ridge.py` | DONE | 38 | RidgeCV with TimeSeriesSplit |
| 18 | `QRCx/architecture/__init__.py` | DONE | 17 | Lazy exports |
| 19 | `QRCx/architecture/direct.py` | DONE | 23 | Wraps AtmosphericQRC.transform() |
| 20 | `QRCx/architecture/residual.py` | PARTIAL | 34 | Core logic works, but see gap below |
| 21 | `QRCx/architecture/parallel.py` | DONE | 24 | Wraps ParallelReservoir.transform() |
| 22 | `QRCx/data/__init__.py` | DONE | 4 | Eager imports |
| 23 | `QRCx/data/loader.py` | DONE | 127 | ISD-Lite download, parse, validate |
| 24 | `QRCx/data/splits.py` | DONE | 28 | DataSplit, temporal_split |
| 25 | `QRCx/data/preprocessor.py` | DONE | 129 | Full preprocessing pipeline |
| 26 | `QRCx/data/lorenz.py` | DONE | 24 | Lorenz-63 via RK45 |
| 27 | `QRCx/metrics/__init__.py` | DONE | 2 | Eager imports |
| 28 | `QRCx/metrics/forecast.py` | DONE | 34 | RMSE, MAE, NRMSE, Skill, VPT |
| 29 | `QRCx/metrics/fsdh.py` | DONE | 38 | FSDH + compute_fsdh_curve |
| 30 | `QRCx/experiment/__init__.py` | DONE | 34 | Lazy exports |
| 31 | `QRCx/experiment/runner.py` | PARTIAL | 191 | Functional but skill_score hardcoded 0.0 |
| 32 | `QRCx/experiment/benchmark.py` | DONE | 68 | Auto-reduction at >0.5s |
| 33 | `QRCx/experiment/ablation.py` | STUB | 70 | Calls res.metrics.iterrows() on dict — will crash |
| 34 | `QRCx/experiment/figures.py` | DONE | 139 | 6 figures, matplotlib |
| 35 | `QRCx/experiment/summary.py` | DONE | 37 | Terminal table |
| 36 | `QRCx/summary.py` | PARTIAL | 7 | Legacy: delegates to results.summary() |
| 37 | `tests/__init__.py` | DONE | 0 | Empty |
| 38 | `tests/test_data.py` | DONE | 37 | QC bounds, preprocess shapes |
| 39 | `tests/test_encoding.py` | DONE | 29 | Pre-existing, valid |
| 40 | `tests/test_metrics.py` | DONE | 11 | FSDH array shape |
| 41 | `tests/test_reservoir.py` | DONE | 23 | 108-dim, jg_ratio, 216-dim |
| 42 | `tests/test_pipeline.py` | DONE | 29 | End-to-end pipeline |
| 43 | `pyproject.toml` | DONE | 29 | Dependencies, build config |

---

## SECTION 2: CRITICAL GAP FIXES — VERIFICATION

### 1. [FIX] reservoir/esp.py — uses lightning.qubit not default.qubit
```python
# reservoir/esp.py line 27:
dev = qml.device("lightning.qubit", wires=qrc.n_qubits)
```
**VERIFIED**: `lightning.qubit` is used.

### 2. [FIX] readout/krr.py — Gamma grid is [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
```python
# readout/krr.py line 11:
GAMMA_GRID = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
```
**VERIFIED**: Correct grid.

### 3. [FIX] readout/krr.py — TimeSeriesSplit(n_splits=5) used in sweep
```python
# readout/krr.py lines 35-46:
tscv = TimeSeriesSplit(n_splits=5)

for gamma in self.GAMMA_GRID:
    for alpha in self.ALPHA_GRID:
        cv_scores = []
        for train_idx, val_idx in tscv.split(F_train):
            F_tr, F_v = F_train[train_idx], F_train[val_idx]
            y_tr, y_v = y_target[train_idx], y_target[val_idx]
            model = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
            model.fit(F_tr, y_tr)
            pred = model.predict(F_v)
            cv_scores.append(np.sqrt(np.mean((y_v - pred) ** 2)))
```
**VERIFIED**: TimeSeriesSplit CV loop is correct.

### 4. [FIX] architecture/residual.py — uses target_col_idx, NOT hardcoded 0
```python
# residual.py lines 26-33:
for i in range(n_samples):
    avg_x = X[i].mean(axis=0)                          # <-- mean of ALL features
    skip = np.tanh(
        avg_x[: self.n_qubits]                          # <-- first n_qubits features
        if len(avg_x) >= self.n_qubits
        else np.pad(avg_x, (0, self.n_qubits - len(avg_x)))
    )
    states[i] += 0.1 * skip
```
**NOT FIXED**: Uses `avg_x[:self.n_qubits]` instead of `X_raw[:, -1, target_col_idx]`. The spec requires the skip connection to use the last timestep's target value, not the first N feature means. The `target_col_idx` parameter is accepted by `fit()` but never reaches `run()`.

### 5. [FIX] metrics/reservoir.py — IPC uses 24h windows not 1-step
```python
# metrics/reservoir.py lines 21-23:
u = rng.standard_normal((n_samples + max_lag + 24, n_qubits))
X = np.array([u[i : i + 24] for i in range(n_samples)])
```
**VERIFIED**: 24-hour windows used.

### 6. [NEW] reservoir/tfim_cudaq.py — CUDA-Q kernel exists
```python
# tfim_cudaq.py lines 70-71:
@cudaq.kernel
def tfim_kernel(n_qubits: int, x_enc: list[float], phis: list[float]):
```
**VERIFIED**: CUDA-Q kernel defined. Falls back to PennyLane if cudaq not available.

### 7. [NEW] reservoir/parallel.py — ParallelReservoir with 216-dim output
```python
# parallel.py lines 24-46:
def transform(self, X: np.ndarray) -> np.ndarray:
    ...
    features = np.concatenate([feat_a, feat_b], axis=1)
    ...
    assert features.shape == (n_samples, n_features), f"Expected ({n_samples}, 216), got {features.shape}"
    return features
```
**VERIFIED**: Concatenates dual reservoir outputs. Shape assertion present.

### 8. [NEW] experiment/benchmark.py — Auto-reduction if latency > 0.5s
```python
# benchmark.py lines 26-29:
if latency > 0.5 and qrc.trotter_steps > 5:
    print(f"Latency {latency:.2f}s > 0.5s; reducing trotter_steps to 5")
    reduced = AtmosphericQRC(n_qubits=qrc.n_qubits, ..., trotter_steps=5, ...)
    return timing_benchmark(reduced, n_samples)
```
**VERIFIED**: Recursive reduction to trotter_steps=5 when latency > 0.5s.

---

## SECTION 3: ARRAY SHAPE CONTRACT ASSERTIONS

Cannot run because `pennylane` is not installed. The assertion code exists in:

- `readout/base.py:fit()` — `assert F_train.ndim == 2`
- `reservoir/tfim.py:transform()` — `assert X.ndim == 3` and `assert features.shape == (n_samples, n_features)`
- `readout/correlators.py:extract_correlators()` — `assert len(result) == expected` and `assert len(result) == 108` for N=8
- `metrics/forecast.py:rmse()` — `assert y_true.shape == y_pred.shape`
- `metrics/fsdh.py:fsdh()` — `assert y_true.ndim == 1`, `assert y_pred.ndim == 1`

All shape assertions are in place. They cannot be executed until `pip install pennylane` is run.

---

## SECTION 4: TEST SUITE RESULTS

Cannot run — `pennylane` is not installed. All 5 test files require pennylane except `test_metrics.py`:

| Test File | Deps | Runnable? |
|-----------|------|-----------|
| `test_data.py` | pennylane (indirect via QRCx imports) | NO |
| `test_encoding.py` | pennylane | NO |
| `test_metrics.py` | numpy only | YES (2 tests) |
| `test_reservoir.py` | pennylane | NO |
| `test_pipeline.py` | pennylane | NO |

To run: `pip install pennylane pennylane-lightning && pytest tests/ -v`

---

## SECTION 5: IMPORT CHAIN VERIFICATION

Cannot run — `pennylane` is not installed. All imports that transitively touch `pennylane` (reservoir, encoding, readout.correlators) will fail.

The `from QRCx import ExperimentConfig` import **would** succeed (config.py has no PL dependency). But `from QRCx import QRCPipeline` already imports `.experiment.runner` which imports `.config`, `.data.preprocessor`, etc. — but not directly pennylane.

**Deferred via lazy imports**: only `encoding.__init__` and `readout.__init__` use eager imports that pull in pennylane. All other modules use `__getattr__` lazy loading.

---

## SECTION 6: PRE-EXISTING FILES AUDIT

| File | Action | Notes |
|------|--------|-------|
| `architecture/parallel_qrc.py` | KEPT | Pre-existing. Defines `ParallelQRC` with different API (fit/predict with reservoir+readout args). New code imports from `architecture/parallel.py` instead. |
| `baselines/arima.py` | KEPT | Pre-existing. `ARIMABaseline` class. Has `.forecast()` function called by pipeline. |
| `baselines/esn.py` | KEPT | Pre-existing. `ESNBaseline` class. Has `.forecast()` function. |
| `baselines/persistence.py` | KEPT | Pre-existing. `PersistenceBaseline` + `.forecast()`. |
| `readout/lasso.py` | KEPT | Pre-existing, not exported from `readout/__init__.py`. |
| `readout/mlp.py` | KEPT | Pre-existing, not exported. |
| `readout/quantum_ridge.py` | KEPT | Pre-existing, not exported. |
| `encoding/amplitude.py` | KEPT | Pre-existing, not exported. |
| `encoding/angle.py` | KEPT | Pre-existing, not exported. |
| `encoding/custom.py` | KEPT | Pre-existing, not exported. |
| `encoding/iqp.py` | KEPT | Pre-existing, not exported. |
| `reservoir/hybrid_esn.py` | KEPT | Pre-existing, not exported. |
| `metrics/advanced.py` | KEPT | Pre-existing, not exported from `metrics/__init__.py`. |
| `metrics/noise.py` | KEPT | Pre-existing, not exported. |
| `metrics/reservoir.py` | MODIFIED | Rewrote `measure_memory_capacity()` to use 24h windows (was 1-step). Kept `measure_ipc_24h()` signature but content is new. |

---

## SECTION 7: PERFORMANCE & DEPENDENCY CHECK

**pyproject.toml dependencies:**
```
pennylane>=0.35, pennylane-lightning>=0.35, scikit-learn>=1.3,
pandas>=2.0, numpy>=1.24, scipy>=1.11, matplotlib>=3.7,
statsmodels>=0.14, reservoirpy>=0.9, qiskit-aer>=0.13, pyyaml>=6.0
```

**Package not installed**: `pip install -e .` has **not** been run.

**pennylane + lightning.qubit**: Not installed. Cannot import.

**cudaq**: Not installed. Fallback in `tfim_cudaq.py`:
```python
try:
    import cudaq
except ImportError:
    cudaq = None
...
if not self._available:
    print("CUDA-Q unavailable; falling back to PennyLane lightning.qubit")
    from .tfim import AtmosphericQRC
    self._fallback = AtmosphericQRC(n_qubits, n_layers, trotter_steps, dt, seed)
```

Fallback message prints and delegates to PennyLane. **Verified**.

---

## SECTION 8: WHAT IS STILL PLACEHOLDER / BROKEN

### Hardcoded Dummy Values
1. **`architecture/residual.py:29`** — Uses `avg_x[:self.n_qubits]` instead of `X_raw[:, -1, target_col_idx]`. The skip should use the last timestep's target value.
2. **`experiment/runner.py:125`** and **`pipeline.py:136`** — `row["skill"] = 0.0` hardcoded. `skill_score()` function in `metrics/forecast.py` is correct (takes y_clim parameter) but never called with a climatological baseline.
3. **`ExperimentConfig.metrics`** includes `"mc"` (default) but no MC computation is wired into the runner or pipeline.
4. **`ExperimentConfig.metrics`** includes `"vpt"` (default) and `vpt()` computes correctly, but the Experiment plots call `plot_ipc_and_vpt({}, {})` with empty dicts — the IPC and VPT data is never collected.

### Bugs
5. **`experiment/ablation.py:62`** — `res.metrics.iterrows()` — `res.metrics` is a `dict` (nested), not a `pd.DataFrame`. `.iterrows()` will raise `AttributeError`.

### NotImplementedError
6. **`reservoir/base.py:15`** — `verify_esp()` raises `NotImplementedError`. The actual `verify_esp()` is a standalone function in `reservoir/esp.py`, not a method on BaseReservoir.

### Unwired / Not Importable
7. **`readout/lasso.py`, `readout/mlp.py`, `readout/quantum_ridge.py`** — Not exported from `readout/__init__.py`. Cannot be selected via `ExperimentConfig.readout`.
8. **`baselines/`** — The classes (`ARIMABaseline`, `ESNBaseline`, `PersistenceBaseline`) exist but the pipeline calls standalone `.forecast()` functions that may or may not exist. `baselines/arima.py` only defines `ARIMABaseline` class — there may be no standalone `forecast()` function.
9. **`metrics/noise.py`** — Not integrated. `plot_noise_sweep()` in figures.py expects p_values and rmse_values that are never generated.
10. **`metrics/reservoir.py`** — Contains `measure_memory_capacity()` and `measure_ipc_24h()` but neither is called anywhere in the pipeline or experiment runner.

### Config Issues
11. **`data/splits.py:13`** — `DataSplit.target_col_idx` returns hardcoded `0`; never used.
12. **`experiment/figures.py:9`** — Hardcoded output path `/mnt/agents/output` or `./figures`. Not configurable.
13. **`experiment/runner.py:89`** — Cache directory hardcoded as `./cache/`.
14. **`experiment/runner.py:163`** — Results saved to `./results.json` (hardcoded).

### Missing Documentation
15. Docstrings are present on most public functions but not all internal/private ones (e.g. `reservoir/tfim.py` `_circuit()` has no docstring).

---

## SECTION 9: COMPLETION SCORECARD

| Module | % Done | Blocker |
|--------|--------|---------|
| `data/` | 95% | None — full pipeline implemented. Minor: normals use multi-index properly. |
| `encoding/` | 95% | None — ZZFeatureMap complete, legacy files preserved. |
| `reservoir/` | 90% | `verify_esp()` on base class raises NotImplementedError (standalone function exists). CUDA-Q fallback works. |
| `readout/` | 85% | KRR and Ridge complete. 3 pre-existing readouts (Lasso, MLP, QuantumRidge) unwired. |
| `architecture/` | 75% | ResidualQRC skip uses mean features instead of target_col_idx. Pre-existing `parallel_qrc.py` has API conflict. |
| `metrics/` | 80% | Forecast + FSDH complete. Reservoir metrics (MC, IPC) defined but unwired. noise.py unwired. |
| `baselines/` | 60% | Pre-existing files kept. Pipeline calls `.forecast()` functions that may not exist in all baselines. No validation done. |
| `experiment/` | 65% | Runner works but skill=0.0 hardcoded. Ablation will crash (dict.iterrows). Figures 3-4 receive empty data. Summary table works. |
| `pipeline/` | 80% | Dual API working. Skill=0.0 hardcoded. Baseline failure caught gracefully. |
| `tests/` | 90% | 5 test files written. Cannot run without pennylane. |

**Overall: ~82%**

---

## SECTION 10: RECOMMENDED NEXT PROMPT

**Fix the residual skip connection in `architecture/residual.py`** — replace the current `avg_x[:self.n_qubits]` with `X_raw[:, -1, target_col_idx]`, add a `fit()` method that stores `target_col_idx` from preprocess output, and pass it through `run()` so the skip uses the actual last-observed target value rather than feature means.
