# Sprint 0 — Truth Purge & Repo Hygiene

**Status:** Done. `pytest tests/ -v` green (12 passed), README rewritten,
stale claims purged, dependencies pinned + fresh-venv verified, organizer
question drafted.

## What changed

- **git**: repo had no version control anywhere; initialized at the top
  level (`cudaq_qrc/`) with a baseline commit before any rewrites, so every
  change below is reviewable as a diff.
- **README.md**: rewritten from scratch. Removed the "Winner — QRC Weather
  Forecasting Challenge 2026" claim and the fabricated performance table
  (RMSE 1h = 0.63, FSDH = 6h). Now consistently states 12 qubits / 234-dim
  base readout / climatological-anomaly residual. Performance section is
  gated behind `results/*.json` via `scripts/make_readme_tables.py` and
  currently reads "Phase 3 results pending" — **no `docs/phase2_report.pdf`
  or historical RMSE/FSDH numbers exist anywhere in this repo**, so none
  were carried into the README as the original sprint brief's example
  table suggested; fabricating them would have violated the traceability
  rule they were meant to enforce.
- **scripts/make_readme_tables.py**: reads `results/*.json`, rewrites the
  `<!-- RESULTS:BEGIN -->`/`<!-- RESULTS:END -->` block in README.md.
  `--check` mode added for CI/pre-commit staleness gating.
- **`.pre-commit-config.yaml`** and **`.github/workflows/ci.yml`**: added
  the results-staleness check plus `pytest tests/ -v` as CI.
- **8-qubit/108-dim purge**: `QRCx/tests/test_reservoir.py` (now 12
  qubits, 234-dim / 468-dim parallel), `QRCx/QRCx/readout/correlators.py`
  (removed a dead `n_qubits==8` assert), `QRCx/QRCx/reservoir/parallel.py`
  (default `n_qubits` 8→12), `QRCx/tests/test_encoding.py` (8→4, since
  these are generic encoder unit tests, not architecture claims).
  `analysis.md` and `audit_report.md` were archived (not deleted) to
  `docs/archive/*_8q_prototype.md` with a banner marking them as
  superseded historical snapshots, since they document the retired
  8-qubit/108-dim prototype phase throughout and rewriting them in place
  would misrepresent when that phase existed.
- **winning_solution.py → pipeline_demo.py**: renamed. Fixed the header
  docstring (removed "WINNING SOLUTION" and the OOS-split memory-capacity
  claim — the actual `measure_memory_capacity()` implementation was
  already correct, in-sample/train=test per Jaeger 2001; only the
  comments/header prose were stale). Made the ablation-figure feature-count
  labels dynamic instead of hardcoded "24"/"108". `used_baselines.py`
  updated to import from the renamed module.
- **Real bug found via the fresh-venv verification** (not just a doc
  issue): a stray `QRCx/__init__.py` at the *project root* (sibling to the
  real package in `QRCx/QRCx/`) re-exported `from .QRCx import *` as a
  namespace-package shim. Combined with pytest's package-mode test
  collection (`tests/__init__.py` exists), this caused pytest to resolve
  `QRCx.data.preprocessor` and `QRCx.pipeline` against the wrong directory,
  raising `ModuleNotFoundError` for two of five test files. Root cause
  confirmed by reproducing the exact import path outside pytest; fix was
  deleting the stray shim (the properly `pip install -e`'d package already
  provides `import QRCx` correctly). This is a correctness fix, not a
  policy change — `QRCx/data/loader.py` itself was never touched.
- **pyproject.toml**: added `authors = [{name = "QRCx Team"}]` and a
  `TODO(repo-url)` placeholder under `[project.urls]` (no real GitHub repo
  URL exists yet — flagged rather than fabricated). All dependencies
  pinned to exact versions resolved by a clean install
  (`pennylane==0.45.1`, `pandas==3.0.3`, etc.); full transitive freeze
  written to `QRCx/requirements-lock.txt`.
- **Fresh-venv verification**: created an isolated venv, ran
  `pip install -e .`, `pip install pytest`, `pytest tests/ -v` →
  **12 passed, 1 warning in ~20 minutes** (Python 3.13.0, win32). The one
  warning is a `DeprecationWarning` from `statsmodels`/`numpy` interaction,
  unrelated to this repo's code.
- **docs/organizer_question.md**: drafted the Dirac-3 / "three pipeline
  stages" clarification question for Mouadh to send manually via
  Aqora/Discord.

## What was checked but found not to need a change

- **Noise-sweep prose contradiction**: searched the entire repo (docs, .py
  docstrings, comments) for the "increases monotonically up to p=0.2
  before degrading further"-style claim described in the sprint brief. No
  such prose exists anywhere in this repo. Noted in README's Known
  Limitations that noise injection is unintegrated in the canonical
  package (exists only in `pipeline_demo.py`) rather than inventing a
  contradiction to resolve.
- **Residual definition**: `QRCx/QRCx/data/preprocessor.py`
  (`compute_climatological_normals` / `apply_climatological_anomaly`) was
  already correctly implementing the climatological-anomaly definition
  with train-only normals — only `README.md`'s prose was stale
  (first-difference).
- **12-qubit default**: `QRCx/QRCx/config.py`,
  `QRCx/QRCx/reservoir/tfim.py`, and `pipeline_demo.py`'s `__main__` block
  already defaulted to `n_qubits=12`. `correlators.py`'s formula is
  already fully generic (`3N + 3·C(N,2)`), so no qubit-count-dependent
  logic needed changing — only the dead 8-qubit-specific assert and
  hardcoded labels/tests.
- **`QRCx/data/loader.py`**: not modified, per the standing rule. It was
  read to confirm it already implements USAF=725300/WBAN=94846 and the
  User-Agent header fix.

## Known gaps carried forward (see README § Known limitations)

1. Canonical preprocessor implements 10 of the 13 specified engineered
   features (missing `Wx, Wy, T_dep`, cyclical hour/day-of-year encoding).
2. `FAST_MODE` flag not implemented anywhere yet.
3. `results/` is empty — no Phase 3 numbers exist yet.
4. Reservoir metrics (MC, IPC) are callable but not wired into
   `QRCPipeline`'s default run.

## Definition of Done — verification

- `grep -rniE "winner|108|8.?qubit|yourusername|Your Name"` across
  `*.md`/`*.py`/`*.toml` (excluding `docs/archive/`, which is intentionally
  historical and clearly banner-marked): only legitimate hits remain
  (`SLP.between(870, 1084)`, and this report's/README's own explanatory
  prose about what was fixed).
- Fresh-venv install + `pytest tests/ -v`: green, 12/12.
- `docs/organizer_question.md`: present.
