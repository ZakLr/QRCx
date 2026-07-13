# QRCx — Full Project Analysis

## Project Structure

```
QRCx/                              <-- project root (working dir)
├── QRCx/                          <-- Python package root (has pyproject.toml)
│   ├── pyproject.toml                  <-- build config, dependencies
│   ├── README.md                       <-- install, quickstart, perf targets
│   ├── tests/                          <-- pytest test suite
│   │   ├── __init__.py
│   │   ├── test_data.py                <-- QC bounds, preprocess shapes
│   │   ├── test_encoding.py            <-- ZZFeatureMap shape/gate order
│   │   ├── test_metrics.py             <-- FSDH returns array
│   │   ├── test_pipeline.py            <-- end-to-end pipeline test
│   │   └── test_reservoir.py           <-- 108-dim output, jg_ratio, 216-dim parallel
│   └── QRCx/                      <-- actual Python package source
│       ├── __init__.py                 <-- lazy exports for all public symbols
│       ├── config.py                   <-- ExperimentConfig dataclass
│       ├── pipeline.py                 <-- QRCPipeline orchestrator
│       ├── summary.py                  <-- legacy summary (pre-existing)
│       ├── architecture/
│       │   ├── __init__.py             <-- lazy exports: DirectQRC, ResidualQRC, ParallelQRC
│       │   ├── direct.py               <-- wraps AtmosphericQRC.transform()
│       │   ├── residual.py             <-- adds tanh residual skip connection
│       │   ├── parallel.py             <-- wraps ParallelReservoir.transform()
│       │   └── parallel_qrc.py         <-- pre-existing (kept, different API)
│       ├── baselines/
│       │   ├── __init__.py
│       │   ├── arima.py                <-- pre-existing
│       │   ├── esn.py                  <-- pre-existing
│       │   └── persistence.py          <-- pre-existing
│       ├── data/
│       │   ├── __init__.py             <-- exports load/parse/validate/preprocess/windows/lorenz
│       │   ├── loader.py               <-- NOAA ISD-Lite download + fixed-width parse
│       │   ├── lorenz.py               <-- Lorenz-63 via RK45
│       │   ├── preprocessor.py         <-- QC → features → normals → anomaly → scale → windows
│       │   └── splits.py               <-- temporal_split, DataSplit dataclass
│       ├── encoding/
│       │   ├── __init__.py             <-- exports BaseEncoder, ZZFeatureMap
│       │   ├── base.py                 <-- ABC with scale_to_pi()
│       │   ├── zz_feature_map.py       <-- H → RZ → IsingZZ (bilinear phase)
│       │   ├── amplitude.py            <-- pre-existing
│       │   ├── angle.py                <-- pre-existing
│       │   ├── custom.py               <-- pre-existing
│       │   └── iqp.py                  <-- pre-existing
│       ├── experiment/
│       │   ├── __init__.py             <-- lazy exports: Experiment, Results, benchmark, ablation, figures, summary
│       │   ├── runner.py               <-- Experiment class + ExperimentResults dataclass
│       │   ├── benchmark.py            <-- timing_benchmark + benchmark_gate (auto-reduction)
│       │   ├── ablation.py             <-- run_ablation() for sweeps
│       │   ├── figures.py              <-- 6 paper figures (matplotlib)
│       │   └── summary.py              <-- print_summary() terminal table
│       ├── metrics/
│       │   ├── __init__.py             <-- exports forecast + fsdh metrics
│       │   ├── forecast.py             <-- rmse, mae, nrmse, skill_score, vpt
│       │   ├── fsdh.py                 <-- Fourier Spectral Distance Hamming
│       │   ├── advanced.py             <-- pre-existing
│       │   ├── noise.py                <-- pre-existing
│       │   └── reservoir.py            <-- pre-existing
│       ├── readout/
│       │   ├── __init__.py             <-- exports BaseReadout, KRRReadout, RidgeReadout
│       │   ├── base.py                 <-- ABC: fit(F_train, y_train, F_val, y_val, target_col_idx)
│       │   ├── correlators.py          <-- single_body + two_body → 108 features
│       │   ├── krr.py                  <-- KRR with TimeSeriesSplit CV, gamma grid [0.001..1.0]
│       │   ├── ridge.py                <-- RidgeCV with TimeSeriesSplit
│       │   ├── lasso.py                <-- pre-existing
│       │   ├── mlp.py                  <-- pre-existing
│       │   └── quantum_ridge.py        <-- pre-existing
│       └── reservoir/
│           ├── __init__.py             <-- lazy exports: BaseReservoir, 3 x QRC, verify_esp
│           ├── base.py                 <-- ABC: transform(X) abstract
│           ├── esp.py                  <-- ESP verification (lightning.qubit, log plot)
│           ├── tfim.py                 <-- AtmosphericQRC: random h,g,J → 108 correlators
│           ├── tfim_cudaq.py           <-- CUDA-Q kernel with CX-RZ-CX, fallback to PennyLane
│           ├── parallel.py             <-- ParallelReservoir: dual seeds, 216 features
│           ├── hybrid_esn.py           <-- pre-existing
│           └── esp.py                  <-- pre-existing (but rewritten above)
```

---

## Status Per Module

### ✅ Data (`data/`)
- **loader.py**: ISD-Lite download + fixed-width parse → DatetimeIndex DataFrame. RH via Magnus formula. Missing rates printed. 5% threshold enforced.
- **preprocessor.py**: QC bounds → build_features (theta, VPD, u, v) → temporal_split → normals on TRAIN ONLY → anomaly → StandardScaler on TRAIN ONLY → sliding_windows (drop NaN windows). Returns target_col_idx.
- **splits.py**: Temporal split by year. No shuffling. DataSplit dataclass.
- **lorenz.py**: Lorenz-63 via RK45.

### ✅ Encoding (`encoding/`)
- **zz_feature_map.py**: H → RZ → IsingZZ per layer. Bilinear phase phi_jk = (π - x_j)(π - x_k). `lightning.qubit`.
- **base.py**: ABC with `scale_to_pi()` mapping to [0, π].

### ✅ Reservoir (`reservoir/`)
- **tfim.py**: `AtmosphericQRC(BaseReservoir)`. Samples random h~U(-1,1), g~U(0.5,1.5), J~U(-1,1) symmetrised. J/g ratio ~1.0. Prints ratio + gate count on init. Trotter evolution → state vector → `extract_correlators()` → 108 features (3N + 3·C(N,2)).
- **tfim_cudaq.py**: `AtmosphericQRCCudaQ(BaseReservoir)`. Pure CUDA-Q kernel with CX-RZ-CX for IsingZZ. Falls back to PennyLane with warning if cudaq unavailable.
- **parallel.py**: `ParallelReservoir(BaseReservoir)`. Two independent reservoirs (seed 42, 43). Concatenated 216 features. Tries `cudaq.par_execute`.
- **esp.py**: ESP verification with `lightning.qubit`. 20 random initial states. Log-scale pairwise L2 plot.
- **Gap**: `hybrid_esn.py` pre-existing but not integrated; `base.py` `verify_esp()` stub raises NotImplementedError.

### ✅ Readout (`readout/`)
- **correlators.py**: `single_body()` → (3N,), `two_body()` → (3·C(N,2),), `extract_correlators()` → (108,) for N=8. Pauli expectations from state vector.
- **krr.py**: `KRRReadout(BaseReadout)` with `GAMMA_GRID=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0]`, `TimeSeriesSplit(n_splits=5)` for CV. Prints best gamma/alpha.
- **ridge.py**: `RidgeReadout(BaseReadout)` with `RidgeCV`, `TimeSeriesSplit(n_splits=5)`, `alphas=np.logspace(-6, 1, 20)`.
- **Gap**: `lasso.py`, `mlp.py`, `quantum_ridge.py` pre-existing but not integrated into the runner/pipeline.

### ✅ Architecture (`architecture/`)
- **direct.py**: Wraps `AtmosphericQRC.transform()`. Returns 108 features.
- **residual.py**: Adds tanh residual skip from window mean. Returns 108 features.
- **parallel.py**: Wraps `ParallelReservoir.transform()`. Returns 216 features.
- **Note**: `parallel_qrc.py` is a pre-existing file with a different API (fit/predict instead of run/transform). Kept for backward compat.

### ✅ Experiment (`experiment/`)
- **runner.py**: `Experiment.run(df)` → preprocess → benchmark_gate → cache features to `.npy` → fit readout per horizon → evaluate → run baselines → save `results.json`. `ExperimentResults` has `summary()`, `plot_all()`, `to_json()`.
- **benchmark.py**: Auto-reduces `trotter_steps` to 5 if latency > 0.5s. Target < 0.07s.
- **ablation.py**: Sweeps encoder, readout, trotter_steps, n_qubits.
- **figures.py**: All 6 paper figures (architecture, ESP, IPC+VPT, noise, results table, FSDH bar). Saved to `./figures/`.
- **summary.py**: Terminal table with RMSE/MAE/Skill per model per horizon.

### ✅ Pipeline (`pipeline.py`)
- Dual API: `QRCPipeline(config=ExperimentConfig(...))` or direct params `QRCPipeline(encoder=..., reservoir=..., architecture=..., readout=...)`.
- `fit_evaluate(df)` → end-to-end.
- `.run()` for config-based flow.
- `.summary()` → metrics DataFrame.

### ✅ Metrics (`metrics/`)
- **forecast.py**: `rmse`, `mae`, `nrmse`, `skill_score`, `vpt`. All with shape assertions.
- **fsdh.py**: Fourier Spectral Distance Hamming. `compute_fsdh_curve()` over horizons.

### ✅ Tests (`tests/`)
| File | Tests | Status |
|------|-------|--------|
| `test_data.py` | QC bounds, preprocess shapes (24,9) + (n,2) + target_col_idx | Written, needs pennylane to run |
| `test_encoding.py` | ZZFeatureMap shape, gate order (H→RZ), scale_to_pi | Pre-existing, OK |
| `test_reservoir.py` | 108-dim output, jg_ratio (0.8–1.2), 216-dim parallel | Written, needs pennylane |
| `test_metrics.py` | FSDH returns array of correct shape | Written, no deps |
| `test_pipeline.py` | End-to-end with 4-qubit reservoir, ridge readout | Written, needs pennylane |

### ✅ Pre-existing Files (kept, not integrated)
- `baselines/` — arima.py, esn.py, persistence.py
- `readout/lasso.py`, `readout/mlp.py`, `readout/quantum_ridge.py`
- `encoding/amplitude.py`, `encoding/angle.py`, `encoding/custom.py`, `encoding/iqp.py`
- `reservoir/hybrid_esn.py`
- `metrics/advanced.py`, `metrics/noise.py`, `metrics/reservoir.py`

---

## Key Design Decisions

### Lazy Imports
Every `__init__.py` uses `__getattr__` + `_imports` dict to defer imports until first access. This allows importing non-PennyLane submodules (config, data, metrics, readout) without triggering the `import pennylane` chain.

### Array Shape Contracts
| Array | Shape | Source |
|-------|-------|--------|
| `X_train` | `(n_train, 24, 9)` | `sliding_windows()` |
| `y_train` | `(n_train, 2)` | `sliding_windows()` (horizons=[1,6]) |
| `features_train` | `(n_train, 108)` | `AtmosphericQRC.transform()` |
| `features_parallel` | `(n_train, 216)` | `ParallelReservoir.transform()` |
| `y_pred` | `(n,)` per horizon | `KRRReadout.predict()` |
| `fsdh_curve` | `(max_horizon,)` | `compute_fsdh_curve()` |

### Critical Parameters
- N=8 qubits, W=24 window, d=9 features
- Trotter: M=10 steps, dt=0.1, τ=1.0
- J/g ≈ 1.0 (critical point)
- `lightning.qubit` everywhere (never `default.qubit`)
- Climatological normals + StandardScaler: trained on TRAIN ONLY

---

## Gaps & Next Steps

### Immediate (need deps)
1. **Install dependencies**: `pip install -e ".[cuda]"` (PennyLane, scikit-learn, etc.)
2. **Run tests**: `pytest tests/ -v` — verify all shape contracts pass
3. **Benchmark timing**: Run `benchmark_gate()` — check < 0.07s per sample

### Integration Gaps
4. **Noise integration**: `metrics/noise.py` defines depolarizing noise models but `reservoir/tfim.py` doesn't apply them. The `plot_noise_sweep()` figure expects data that isn't generated anywhere.
5. **`hybrid_esn.py`**: Pre-existing but not exported or tested. Should be integrated as a baseline option.
6. **`parallel_qrc.py`** vs **`parallel.py`**: Two files define `ParallelQRC` with different APIs. The new one (`parallel.py`) is used by the pipeline; the old one (`parallel_qrc.py`) is unused. Should deprecate or merge.
7. **Additional readout types**: `lasso.py`, `mlp.py`, `quantum_ridge.py` exist but are not selectable via `ExperimentConfig.readout`.

### Data Gaps
8. **Actual ISD data**: The pipeline downloads from NOAA on first call. Need internet access or local cache for tests.
9. **RH formula**: Uses Magnus with fixed coefficients — should verify against psychrometric formula for accuracy below -40°C.

### Experiment Gaps
10. **Skill score**: Currently hardcoded to 0.0 in the runner. Needs a climatological baseline to compute `skill_score(y_true, y_pred, y_clim)`.
11. **IPC (Information Processing Capacity)**: Referenced in `figures.py` (`plot_ipc_and_vpt`) but no IPC computation module exists.
12. **MC (Mutual Coherence)**: Listed in `ExperimentConfig.metrics` defaults but not implemented anywhere.
13. **Lorenz-63 VPT**: The `plot_ipc_and_vpt()` figure expects VPT results for QRC vs ESN that aren't generated.

### Infrastructure
14. **Cache directory**: `./cache/` for `.npy` features is hardcoded in `runner.py`. Should be configurable.
15. **Figures output**: `./figures/` or `/mnt/agents/output/`. Should be configurable via config.
16. **`results.json`**: Saved to CWD. Should accept explicit path.

### Notebooks / Demos
17. No Jupyter notebooks exist for step-by-step demo or result visualization.
18. No `scripts/` directory for one-shot experiment runs (e.g., `python -m QRCx.scripts.run_experiment`).
