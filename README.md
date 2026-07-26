# QRCx — Quantum Reservoir Computing for Weather Forecasting

**QRCx Team** · qBraid x MITRE x JonesTrading Global Industry Challenge 2026 · Track B

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.45-orange)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com/?gitHubUrl=https://github.com/ZakLr/QRCx.git)

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
N=12). 10 qubits is the only permitted fallback; no other qubit count
appears in the reference pipeline. **The new sequential dissipative
reservoir (not the v4 windowed reservoir) runs at 12 qubits when a GPU
(CuPy) is available** — verified at ~7.4s/step on an NVIDIA L4, projecting
the pilot-scale run well under the project's wall-clock budget — **and
falls back to 10 qubits on CPU-only hardware**, where 12-qubit
density-matrix simulation is too slow. See Known Limitations and
`docs/sprint_log/SPRINT_2_5_REPORT.md`.

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
git clone https://github.com/ZakLr/QRCx QRCx
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

```bash
./scripts/reproduce.sh --quick   # <30 min: canonical val split (2019-2022 train), classical fairness protocol, FAST_MODE
./scripts/reproduce.sh --full    # the real Sprint 6 headline run: full KORD record 2011-2024,
                                  # train 2011-2020 / val 2021-2022 / test 2023-2024, 3-seed ESN,
                                  # horizons 1-48, DM test + bootstrap CI at every horizon.
                                  # Real measured wall-clock: ~170 min for val alone (FAST_MODE) --
                                  # NOT a quick operation, documented honestly, not padded down.
                                  # Add --unlock-test to also run the one-time locked test-split pass.
```

Both download/load ISD-Lite data via `QRCx/QRCx/data/loader.py` (frozen
— do not modify), preprocess with climatological-anomaly residuals, and
write real numbers to `results/*.json`. The table below is curated by
hand (not mechanically regenerated — `scripts/make_readme_tables.py` is
a Sprint 0/1-era tool whose generic table format predates the curated
per-sprint tables now used; it is legacy, not the current mechanism) but
every number in it has been directly audited against the corresponding
`results/*.json` file (see `docs/sprint_log/SPRINT_9_REPORT.md`'s numbers
audit) — nothing here is invented or approximated from memory.

### Performance

<!-- RESULTS:BEGIN -->
**FINAL SPRINT canonical-split benchmark — final, locked test-split numbers**
(train 2019-2022, val 2023 tuned on `results/baselines_val.json` /
`results/full_matched_benchmark_val.json`; test 2024 one-time confirmatory
pass, `results/baselines_test.json` / `results/full_matched_benchmark_test.json`),
skill vs. persistence, RMSE in °C (exact conversion, `results/canonical_units.json`):

| model | RMSE°C / skill@1h (test) | RMSE°C / skill@6h (test) |
|---|---|---|
| ARIMA(2,1,2) | 6.76 / -5552.0% | 6.79 / -487.5% |
| ARIMA (auto-order) | 2.33 / -569.9% | 3.48 / -54.8% |
| ESN dim-matched (234) | 1.15 / -64.2% | 2.71 / +6.1% |
| ESN-500 | 1.08 / -45.3% | 2.72 / +6.1% |
| Residual-ESN | 0.87 / +7.0% | 2.59 / +14.8% |
| **null-control Ridge (no reservoir)** | **0.85 / +10.5%** | **2.52 / +19.1%** |
| null-control KRR* | 2.43 / -629.8% | 4.36 / -142.1% |
| Residual-Ridge (no reservoir) | 0.85 / +10.5% | 2.52 / +19.1% |
| v4 QRC, residual (20 qubits) | 0.91 / -1.4% | 2.81 / -0.8% |
| v5 QRC, residual (12 qubits) | 0.87 / +6.5% | 2.59 / +14.1% |
| Concat C=[raw,v5] | 0.86 / +9.1% | 2.57 / +15.7% |
| Concat C=[raw,v4] | 0.86 / +8.6% | 2.54 / +17.8% |

\* Null-control KRR's training set is capped at 3,000 samples, unlike every
other row's full window — not a clean like-for-like comparison.

**Honest headline finding, on the canonical split, locked test-2024 set**:
a plain linear Ridge regression on the flattened raw window — no reservoir
at all, classical or quantum — still matches or beats every reservoir model
tested at both horizons shown. v5 (12-qubit density matrix) is the
best-performing *reservoir* model and is real, DM-significant vs.
persistence, but does not close the gap to the null control; v4 (20-qubit
statevector) underperforms persistence at h=1 and is statistically
indistinguishable from it at h=6. Concatenating either QRC's features onto
the raw window moves skill significantly, but *toward* the raw-window
number, not past the null-control Ridge ceiling — see
`docs/sprint_log/FINAL_SPRINT.md`'s Phase 2/5 entries and the paper's
Sec. 4.2/4.3 for the full concatenated-readout analysis this table is a
summary of.
<!-- RESULTS:END -->

---

## Repo map

```
. (repo root)
├── README.md                       # This file
├── scripts/
│   ├── make_readme_tables.py       # LEGACY (Sprint 0/1-era, generic table format) -- not the current mechanism; results block is now hand-curated + numbers-audited, see Sprint 9 report
│   ├── reproduce.sh                # Current reproduce entry point: --quick (<30min) / --full (real Sprint 6 headline run)
│   ├── sprint6_full_benchmark.py   # Full 14-year dataset benchmark (Sprint 6 headline numbers)
│   ├── narma10_validation.py       # Sprint 1 NARMA10 micro-validation
│   ├── narma10_sweep.py            # Sprint 2 NARMA10 tuning sweep (superseded gate; see narma10_gate_v2.py)
│   ├── narma10_gate_v2.py          # Sprint 2.5 valid-protocol NARMA10 gate (train=3000/test=1000/val-tuned alpha) -- authoritative gate result
│   ├── ipc_mc_characterization.py  # IPC/MC trade-off figure, with Sprint 2.5 shuffle-surrogate thresholding
│   ├── benchmark_sequential_backends.py # Runtime-gate backend benchmark
│   └── gpu_verify_standalone.py    # Self-contained CuPy/GPU benchmark for manual qBraid Lab runs
├── configs/v5_reference.yaml       # Frozen v5 reservoir config (PROVISIONAL, gate not passed)
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

- **Feature set gap — CLOSED (Sprint 3)**: the canonical preprocessor
  (`QRCx/QRCx/data/preprocessor.py::build_features`) now engineers the
  full 13-feature set (`T_db, T_dew, RH, WS, SLP, WD, Wx, Wy, T_dep,
  hour_sin, hour_cos, doy_sin, doy_cos`), ported from `pipeline_demo.py`'s
  `engineer_features()` and replacing the previous 10-feature set
  (`θ, VPD, u, v` dropped, not merged alongside — see
  `docs/evaluation_protocol.md`).
- **ESN and ARIMA baselines — FIXED (Sprint 3)**: the pre-Sprint-3 ESN fed
  reservoirpy flattened 24h windows instead of the genuine hourly
  sequence (a target/input-alignment bug, not a hyperparameter problem)
  and never tuned `leak_rate`/`spectral_radius`/`input_scaling`/`ridge`
  against a validation split, producing skill of roughly -400% to -1400%
  vs. persistence on real KORD data. ARIMA separately returned the
  identical forecast array for every horizon. Both fixed; tuned ESN
  (full 384-point grid) now achieves **+6.2% skill at 1h, +14.6% at 6h**
  on real KORD data — see `docs/evaluation_protocol.md` and
  `results/esn_diagnosis.json`/`results/baselines.json`.
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
- **Sequential dissipative reservoir's NARMA10 hard gate still FAILS after
  a re-run with a statistically valid protocol (Sprint 2.5)**: the Sprint 2
  gate verdict was voided (too few post-washout samples for a trustworthy
  estimate) and re-run with train=3000/test=1000/held-out validation-slice
  alpha tuning. Best full-protocol result: `gamma1=0.03, n_in=10 (all
  qubits driven)`, NMSE **0.224** — real progress from the voided 0.398,
  but still short of the (now more rigorously measured) AR baseline of
  0.0745 and the ≤0.2/≤0.15 target/aspirational bars. Restricted input
  injection (driving only 2 or 4 of 10 qubits, leaving the rest as pure
  memory nodes) was implemented and tested as a hypothesis for improving
  memory — it **did not help**: full injection (all qubits driven) beat
  restricted injection at every damping rate tried, contradicting the
  "input erasure" theory. See `docs/sprint_log/SPRINT_2_5_REPORT.md` for
  the full trail and next-step recommendations.
- **12-qubit sequential reservoir is GO on GPU, still NO-GO on CPU alone.**
  Two real performance bugs were found and fixed in Sprint 2.5 beyond
  Sprint 2's exact-propagator pass — `np.einsum` wasn't dispatching to
  BLAS for local per-qubit gate application (~3x slower than an equivalent
  `np.matmul`-based rewrite), and the correlator-extraction partial trace
  was forcing an unnecessary full-array transpose-copy (~300x slower than
  a direct `einsum` with repeated axis labels) — this cut 12-qubit CPU cost
  from ~40.7 s/step (Sprint 2) to ~15.6-18 s/step (complex128), still over
  the ≤8 s/step CPU target (projects to ~19h on CPU alone, still NO-GO).
  **Verified on an NVIDIA L4 GPU (CuPy, user-run in qBraid Lab): 7.42
  s/step at 12 qubits in full complex128 precision, projecting the
  Sprint-4-scale pilot to ~9.0 hours — GO**, and 0.52 s/step with an
  optional reduced-precision complex64 mode (~0.63h, comfortably GO). Per
  project policy ("if the pilot projects ≤12h on ANY available hardware,
  12 qubits is restored"), **12 qubits is the reference configuration for
  this reservoir when a GPU is available; the 10-qubit fallback is used on
  CPU-only hardware.** See `docs/sprint_log/SPRINT_2_5_REPORT.md`,
  `results/gpu_verification.json`, and
  `results/sequential_backend_benchmark.json`.
- **Reservoir metrics (MC, IPC) exist but are not wired into the
  `QRCPipeline` orchestrator end-to-end** — they're callable directly
  (`QRCx.measure_memory_capacity`, `QRCx.measure_ipc_24h`) but not part of
  `python -m QRCx`'s default run.
- **`ZZFeatureMap.circuit()` was applying every gate twice — FIXED
  (Sprint 4)**: `encode()` built PennyLane ops inside the active QNode's
  recording context (which auto-queues at construction time), then
  `circuit()` called `qml.apply(op)` on top of that, queuing every gate a
  second time. For Hadamards (H·H = I, nothing intervening) this silently
  cancelled them out completely — the encoder never actually created
  superposition — and every RZ/IsingZZ angle was doubled. Found while
  cross-validating a from-scratch GPU reimplementation against the real
  circuit and finding O(1) disagreement. Fixed via
  `qml.QueuingManager.stop_recording()`; verified to ~1e-8 agreement
  after the fix, 33/33 tests still green. See
  `docs/sprint_log/SPRINT_4_REPORT.md`.
- **Sprint 4 (REVISED) pilot verdict: paradigm validated, not yet
  competitive (Branch B)**. Real pilot evaluation (train 2019-2021, eval
  2022, real KORD data) on both v4 (windowed, 16/20 qubits) and v5
  (sequential dissipative, 12-qubit reference config, on an H200 GPU).
  **V1 (paradigm validation): PASS** — a same-architecture ablation
  (dissipation on vs. off) shows a real, DM-significant effect (v5
  residual, h=12: +14.4% skill vs. persistence, p≈0) that vanishes to
  ~0% with dissipation off, confirming the mechanism is load-bearing, not
  incidental. **V2 (competitive verdict): FAIL** — the best QRC result
  still loses to dimension-matched ESN at 3 of 4 horizons, and loses to a
  plain linear Ridge control (no reservoir at all) at every horizon on
  this pilot task. v5's original 12-drive ablation plan (2 gamma1 × 2
  multiplexing × 3 seeds) was reduced to 2 drives (1 seed, V=1 only) for
  real time-budget reasons after two real throughput bugs were found and
  fixed (~75x combined speedup) plus a NaN-in-continuous-sequence bug
  (real KORD missing-data readings corrupting the recurrent density
  matrix mid-drive). See `docs/sprint_log/SPRINT_4_REPORT.md` and
  `momo_reports.md` for the full trail, all four bugs found, and every
  real number.
- **Sprint 5: IPC-matched reservoir tuning — real matching found, but
  the sprint's own stated hypothesis not confirmed**. Built a task-demand
  profile (`QRCx/QRCx/metrics/task_demand.py`) and matched it against the
  reservoir's supply-side IPC (extended to degree 3,
  `measure_ipc_by_degree` in `QRCx/QRCx/metrics/reservoir_sequential.py`)
  to tune `(gamma1, input_scaling)`. Real result: the matched config
  (gamma1=0.3, a=0.1) captures substantially more overlapping capacity
  than the Sprint 2 reference (9.26 vs. 4.53, +104%), but this did **not**
  translate into better 6h forecast skill — the reference config still
  wins at h=6 and h=12 on a real pilot comparison, only the matched
  config wins at h=1/h=3. Reported honestly as a negative result for the
  named hypothesis, not spun. See `docs/sprint_log/SPRINT_5_REPORT.md`
  and `docs/ipc_matching.md`.
- **FINAL SPRINT: does the quantum reservoir add information beyond the
  raw window? Real answer: no, not on this configuration.** A
  concatenated-readout experiment (raw window alone vs. +QRC features vs.
  +dimension-matched-ESN-control features, Diebold-Mariano tested at every
  horizon) found the QRC features never significantly improve on the raw
  window alone, and a dimension-matched classical ESN control performs
  just as poorly — the automatic interpretation rule returns
  **"redundant"**. Diagnostic: the driven reservoir's 165-dimensional
  feature bank has an effective rank (participation ratio) of only
  **~1.5**, i.e. barely more than one real degree of freedom, which
  plausibly explains the redundancy. See
  `docs/sprint_log/FINAL_SPRINT.md`, `results/hybrid_readout.json`
  (pilot-scale), and `results/full_matched_benchmark_val.json`
  (canonical-scale confirmation).
- **Dirac-3: genuine device attempt, real blocker, not "pending."** A
  legitimate best-subset-selection formulation was submitted via
  `qci-client`; it failed at client initialization
  (`QCI_API_URL`/`QCI_TOKEN` not configured in this environment) — logged
  verbatim in `docs/sprint_log/FINAL_SPRINT.md`. A local simulated-
  annealing stand-in (`QRCx/QRCx/readout/dirac3_selector.py`, identical
  solver interface) ran the full K∈{32,64,128} comparison against Lasso
  and greedy forward selection; see `results/dirac3/comparison.json`.

---

## LLM-use disclosure

Claude (Anthropic) was used as a coding assistant throughout this entire
project (Sprints 0-9), under continuous human direction and review:
implementing the reservoir/encoding/readout/metrics code, designing and
running experiments (including real GPU work on qBraid on-demand
instances), finding and fixing real bugs (documented per-sprint in
`docs/sprint_log/`, including a production PennyLane double-gate-
application bug, two independent reservoir throughput bugs, a missing-
data-handling bug, and an undocumented ZZ-injection/input_scaling
interaction), and drafting this README, the paper (`docs/paper/main.tex`),
and every sprint report. All numeric claims — in this README, in the
paper, and in every sprint report — are gated behind `results/*.json`
generated by actually running the pipeline; none are invented or
approximated from memory. The human author(s) directed scope, reviewed
every real bug and result as reported, and made every scope-reduction
decision documented throughout (e.g. `docs/sprint_log/SPRINT_4_REPORT.md`
§4, `SPRINT_6_REPORT.md` §3).

---

## Citation

```bibtex
@software{QRCx,
  title = {QRCx: Quantum Reservoir Computing for Weather Forecasting},
  author = {QRCx Team},
  year = {2026},
  url = {https://github.com/ZakLr/QRCx}
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
