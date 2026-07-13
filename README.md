# QRCx — Quantum Reservoir Computing for Weather Forecasting

**QRCx Team** · QRC Weather Forecasting Challenge 2026 · Track B

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.45-orange)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com/?gitHubUrl=TODO(repo-url).git)

> No performance claims in this README are asserted without a corresponding
> file in `results/`. See [Performance](#performance) below.

---

## What this is

QRCx is a Quantum Reservoir Computing (QRC) pipeline for short-horizon
(1–12h) temperature forecasting at Chicago O'Hare (KORD), built on NOAA
ISD-Lite hourly observations. A fixed, randomly-initialised Trotterized
Transverse-Field Ising Model (TFIM) reservoir acts as a nonlinear feature
extractor over Pauli-correlator readout features; only a classical KRR/Ridge
readout is trained. This avoids the vanishing-gradient / barren-plateau
problems of variational quantum algorithms while retaining the exponentially
large Hilbert space (2^N) as a feature source.

**Reference configuration: 12 qubits, 234-dimensional base readout**
(36 single-body + 198 two-body Pauli correlators — `3N + 3·C(N,2)` for
N=12). 10 qubits is the only permitted fallback, used only if a
Sprint-1 runtime gate fails; no other qubit count appears in the reference
pipeline. **The Sprint 1 runtime gate did fail at 12 qubits** for the new
sequential dissipative reservoir specifically (not the v4 windowed
reservoir) — see Known Limitations and
`docs/sprint_log/SPRINT_1_REPORT.md`; that reservoir runs at the 10-qubit
fallback.

The prediction target is a **climatological anomaly**:

```
ε_t = (y_t − μ_climo(month, day, hour)) / σ_climo
```

with `μ_climo` / `σ_climo` computed on the TRAIN split only, never on
val/test. This is the single source of truth for the residual definition
across this repo; an earlier first-difference definition (`y_t − y_{t-1}`)
is deprecated and no longer used anywhere in the pipeline.

---

## Architecture

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
│  NOAA    │ → │  Quality  │ → │   ZZ     │ → │   TFIM   │ → │ KRR/Ridge │
│  ISD-    │   │  Control  │   │ Feature  │   │ Reservoir│   │  Readout  │
│  Lite    │   │ + Feature │   │   Map    │   │(234-dim, │   │           │
│  Data    │   │  Engineer │   │  (H→RZ→  │   │ 12 qubit)│   │           │
│          │   │ + Climo   │   │  IsingZZ)│   │ Trotter- │   │           │
│          │   │  Anomaly  │   │          │   │ ized TFIM│   │           │
└──────────┘   └───────────┘   └──────────┘   └──────────┘   └───────────┘
```

- **Encoding**: ZZ Feature Map (Havlíček et al. 2019, *Nature* 567) —
  `H → RZ(x_j) → IsingZZ((π−x_j)(π−x_k))`, data re-uploading over `n_layers`.
- **Reservoir**: TFIM Hamiltonian `H = Σ h_i Z_i + Σ g_i X_i + Σ J_{ij} Z_i Z_j`
  at the critical point (`⟨|J|⟩/⟨g⟩ ≈ 1.0`), Trotterized evolution,
  `lightning.qubit` backend (CUDA-Q kernel available with PennyLane
  fallback). Time-multiplexed reservoir computing follows Fujii & Nakajima
  2017 (*PR Applied* 8, 024030).
- **Readout**: single-body (`⟨X⟩,⟨Y⟩,⟨Z⟩`) + two-body
  (`⟨ZZ⟩,⟨XX⟩,⟨YY⟩`) Pauli expectations, 234-dim at N=12
  (`QRCx/QRCx/readout/correlators.py`).
- **Splits**: strict temporal (train/val/test by year), no shuffling;
  `TimeSeriesSplit` for all cross-validation; explicit `target_col_idx`
  threaded through every stage (`QRCx/QRCx/data/splits.py`).
- **Reservoir-quality protocols**: Memory Capacity follows the Hou et al.
  2026 (arXiv:2508.12383, *PRL* 136, 120602) / Jaeger 2001 protocol —
  iid Gaussian window drive, shape `(N, W, n_feat)`, train=test evaluation
  (healthy range 5–10 at 12 qubits). IPC follows Čindrak et al. 2026
  (arXiv:2603.21371); feature padding scales with `n_qubits`. **The OOS
  70/30 memory-capacity variant is known-wrong (collapses MC to ~0.27) and
  is not used anywhere in this repo.**
- **FSDH** (Forecast Skill Duration Horizon) — the primary operational
  metric — is the first horizon `h` where `skill(h) < 0`, minus 1:
  `max{h : RMSE_model(h) < RMSE_persistence(h)}`.

Related work: Ahmed et al. 2025 (arXiv:2506.22335), Kornjača et al. 2024
(arXiv:2407.02553), Antoncich et al. 2026 (arXiv:2602.14641), Li et al. 2026
(*PR Research* 8, 023028, QRC volatility).

---

## Install

```bash
git clone TODO(repo-url) QRCx
cd QRCx/QRCx
pip install -e .          # PennyLane backend
pip install -e ".[cuda]"  # + CUDA-Q GPU kernel (optional)
```

Dependencies are pinned exactly in `pyproject.toml`; the full resolved
transitive closure (from a clean venv) is recorded in
`requirements-lock.txt`. To verify a clean install:

```bash
python -m venv .venv-check && source .venv-check/Scripts/activate  # or bin/activate on Linux/macOS
pip install -e .
pip install pytest
pytest tests/ -v
```

This was verified in-session (Python 3.13.0, win32): **12 passed in ~20
minutes** (the TFIM state-vector simulation at 12 qubits dominates runtime;
`test_pipeline.py`/`test_reservoir.py` are the slow cases).

---

## Reproduce the headline result

There is no headline result yet — see [Performance](#performance). Once
Sprint 1+ produces `results/*.json`, the reproduce command will be:

```bash
cd QRCx
python -m QRCx --years 2019 2024
python ../scripts/make_readme_tables.py
```

`python -m QRCx` downloads/loads ISD-Lite data via
`QRCx/QRCx/data/loader.py` (frozen — do not modify; wraps it if new data
behavior is needed), preprocesses with climatological-anomaly residuals,
runs the configured architecture, and writes results + figures.
`scripts/make_readme_tables.py` then regenerates the block below from
`results/*.json` — no number below is hand-typed.

### Performance

<!-- RESULTS:BEGIN -->
_Phase 3 results pending. Run the reproduce command in this README to generate `results/*.json`, then re-run this script to populate this section. No performance numbers are reported until they exist in `results/*.json`._
<!-- RESULTS:END -->

---

## Repo map

```
cudaq_qrc/
├── README.md                       # This file
├── scripts/
│   ├── make_readme_tables.py       # Regenerates the results block above from results/*.json
│   ├── narma10_validation.py       # Sprint 1 NARMA10 micro-validation
│   ├── narma10_sweep.py            # Sprint 2 Phase 2.1 NARMA10 tuning gate sweep
│   ├── ipc_mc_characterization.py  # Sprint 2 Phase 2.2 IPC/MC trade-off figure
│   └── benchmark_sequential_backends.py # Runtime-gate backend benchmark
├── configs/v5_reference.yaml       # Frozen v5 reservoir config (Sprint 2; PROVISIONAL, gate not passed)
├── results/                        # results/*.json — the only source of truth for reported numbers
├── qrc_figures/                    # Generated figures, incl. fig_ipc_tradeoff.png (Sprint 2)
├── docs/
│   ├── organizer_question.md       # Drafted question re: Dirac-3 requirement, sent via Aqora/Discord
│   ├── sprint_log/                 # SPRINT_REPORT.md per sprint
│   └── archive/                    # Superseded 8-qubit-prototype-era docs (historical only)
├── pipeline_demo.py                 # Standalone single-file demo (real+synthetic data, all diagnostics)
├── used_baselines.py                # Classical baselines only (Persistence, ARIMA, ESN-500, ESN-5000)
├── Qbraid.py                        # QRC pipeline using only the QRCx package API
├── data/isd/                        # NOAA ISD-Lite raw downloads (gitignored, regenerated by loader.py)
└── QRCx/                            # Installable package root
    ├── pyproject.toml               # Pinned deps, build config
    ├── requirements-lock.txt        # Full transitive freeze from a clean install
    ├── tests/                       # pytest suite, all passing
    └── QRCx/                        # Package source
        ├── config.py                 # ExperimentConfig (n_qubits=12 default)
        ├── pipeline.py                # QRCPipeline orchestrator
        ├── data/
        │   ├── loader.py              # NOAA ISD-Lite download + parse — FROZEN, do not edit
        │   ├── preprocessor.py        # QC → features → climatological anomaly → scale → windows
        │   ├── splits.py              # Strict temporal split, target_col_idx
        │   └── narma.py               # NARMA10 generator (reservoir sanity benchmark)
        ├── encoding/zz_feature_map.py # ZZ Feature Map (H→RZ→IsingZZ)
        ├── reservoir/tfim.py          # AtmosphericQRC — v4 windowed TFIM reservoir (PennyLane)
        ├── reservoir/sequential.py    # SequentialDissipativeQRC — v5 recurrent dissipative reservoir (exact propagator default, Trotter kept for hardware-matched runs; multiplexing=V; see SPRINT_1/2_REPORT.md)
        ├── reservoir/sequential_backends.py # qiskit_aer / pennylane_mixed drivers, benchmarked against the numpy backend above
        ├── reservoir/esp_sequential.py # Trace-distance ESP check for the sequential reservoir
        ├── readout/correlators.py     # Pauli correlator extraction (234-dim at N=12; *_dm variants for density matrices)
        ├── architecture/              # direct.py, residual.py, parallel.py
        ├── baselines/                 # persistence, arima, esn, gfs
        ├── metrics/                   # forecast.py (RMSE/MAE/skill/VPT), fsdh.py, reservoir.py (v4 MC/IPC), reservoir_sequential.py (v5 MC/IPC, Jaeger continuous-drive protocol), noise.py
        └── experiment/                # runner.py, benchmark.py, ablation.py, figures.py, figstyle.py (Sprint 2 academic figure style)
```

---

## Known limitations

- **Feature set gap**: the canonical preprocessor
  (`QRCx/QRCx/data/preprocessor.py`) currently engineers 10 features
  (`T_db, T_dew, SLP, WS, WD, RH, θ, VPD, u, v`), not the full 13-feature
  set (missing `Wx, Wy, T_dep`, cyclical hour/day-of-year encodings). That
  full 13-feature set exists only in the standalone `pipeline_demo.py`
  (`engineer_features()`) and has not been ported into the canonical
  package. Per policy this was **not** silently merged — reconciling the
  two is deferred to a later sprint.
- **Noise injection unintegrated in the canonical package**: the
  `noise_sweep=True` config flag and `QRCx/QRCx/metrics/noise.py` exist,
  but `reservoir/tfim.py` does not apply noise and
  `plot_noise_sweep()` has no wired data source in the canonical pipeline.
  A working noise sweep exists only in `pipeline_demo.py`
  (v4 convention: noise at input-encoding angles). Migrating the useful
  channel to inter-step amplitude damping is planned for a later sprint,
  keeping the v4 hook for ablation. No contradictory noise-sweep prose was
  found elsewhere in this repo during the Sprint 0 audit — nothing else to
  fix here.
- **`FAST_MODE` implemented in Sprint 1/2 scripts, not project-wide**:
  `scripts/narma10_validation.py`, `scripts/narma10_sweep.py`, and
  `scripts/ipc_mc_characterization.py` accept `--fast-mode`/
  `--no-fast-mode`; older experiment entry points (`python -m QRCx`, etc.)
  do not yet. Extending this project-wide is deferred.
- **No results yet**: `results/` is empty; the Performance section above is
  intentionally a pending placeholder rather than a fabricated number.
- **Sequential dissipative reservoir's NARMA10 hard gate FAILED (Sprint 2
  Phase 2.1)**: after a dedicated tuning sweep (`scripts/narma10_sweep.py`)
  over `gamma1, gamma2, input_scaling, washout, multiplexing`, the best
  config found (`gamma1=0.1, gamma2=0.1, a=0.3, washout=50, V=4`) reaches
  NMSE 0.398 — real progress from Sprint 1's 0.945, but still short of the
  required <0.099 (must beat the linear AR baseline) and the ≤0.2/≤0.15
  target/aspirational bars. Per the Sprint 2 spec this is a hard stop:
  the reservoir is not carried forward to weather-data integration until
  this gate passes. See `docs/sprint_log/SPRINT_2_REPORT.md` for the full
  sweep trail, the (shallow, interior) optimum found near `gamma1≈0.1`,
  and concrete next-step recommendations (denser gamma1 sampling, a joint
  rather than coordinate-wise search, longer driven sequences — the best
  config still only had 175 training samples against 660 features at
  V=4).
- **12-qubit sequential reservoir is NO-GO on runtime, even after the
  Sprint 2 performance pass**: Sprint 2 replaced per-step Trotter gates
  with an exact, once-precomputed propagator (eigendecomposition of the
  full 12-qubit TFIM Hamiltonian), giving a real ~2-3.5x speedup (12q:
  ~120.8→~40.7 s/step; 10q: ~6.59→~1.87 s/step) — but the ≤5 s/step CPU
  target was only met at 10 qubits. At 12 qubits the trajectory-cached
  Sprint 4 pilot still projects to ~49.4 hours against the 12h threshold.
  Profiling shows the bottleneck moved from the Trotter evolution (now
  cheap: 2 dense matmuls) to injection + amplitude damping (still
  per-qubit gate touches). A GPU backend (CuPy, auto-detected) is
  implemented but **unverified** — this development environment has no
  GPU/CuPy available. The 10-qubit fallback stands, now ~3.5x faster,
  which is what made Sprint 2's NARMA10 sweep tractable at all. See
  `docs/sprint_log/SPRINT_2_REPORT.md` and
  `results/sequential_backend_benchmark.json`.
- **Reservoir metrics (MC, IPC) exist but are not wired into the
  `QRCPipeline` orchestrator end-to-end** — they're callable directly
  (`QRCx.measure_memory_capacity`, `QRCx.measure_ipc_24h`) but not part of
  `python -m QRCx`'s default run.

---

## LLM-use disclosure

Claude (Anthropic) was used as a coding assistant throughout this project,
including for this Sprint 0 truth-purge: auditing the repo for stale
claims, fixing 8-qubit/108-dim references left over from an early
prototype phase, removing an unpublishable "Winner" claim and fabricated
performance table, resolving a real pytest-collection bug (a stray
duplicate `__init__.py` at the project root shadowing the installed
package), pinning dependencies, and drafting this README. All numeric
claims are gated behind `results/*.json` generated by running the actual
pipeline, not by LLM output.

---

## Citation

```bibtex
@software{QRCx,
  title = {QRCx: Quantum Reservoir Computing for Weather Forecasting},
  author = {QRCx Team},
  year = {2026},
  url = {TODO(repo-url)}
}
```

Key references: Hou et al. 2026 (arXiv:2508.12383, *PRL* 136, 120602) ·
Čindrak et al. 2026 (arXiv:2603.21371) · Ahmed et al. 2025
(arXiv:2506.22335) · Kornjača et al. 2024 (arXiv:2407.02553) · Antoncich
et al. 2026 (arXiv:2602.14641) · Havlíček et al. 2019 (*Nature* 567) ·
Jaeger 2001 · Fujii & Nakajima 2017 (*PR Applied* 8, 024030) · Li et al.
2026 (*PR Research* 8, 023028).

## License

MIT
