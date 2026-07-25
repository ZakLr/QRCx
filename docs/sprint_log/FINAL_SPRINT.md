# FINAL 24-HOUR SPRINT — Decision Log

Running log per `QRCx_FINAL_24H_SUPERPROMPT.md`'s instruction: timestamps,
H200 benchmark numbers, canonical split, Phase 2 interpretation rule,
Dirac-3 outcome, every scope cut with its reason.

## Phase 1 — H200 benchmark + canonical-split drive

- **Real mismatch identified and reconciled**: Sprint 6's classical
  baselines used the full 2011-2024 record (train 2011-2020/val
  2021-2022/test 2023-2024), while Sprint 4's QRC results used a
  different pilot split (train 2019-2021/eval 2022). Different data in
  one ranked headline table. Reconciled by re-establishing the
  project's ORIGINAL canonical split (Sprint 3: train 2019-2022/val
  2023/test 2024) as the single split for everything in the final
  headline table. Sprint 6's full-record numbers retained only as a
  supplementary robustness note.
- **H200 benchmark** (`results/h200_benchmark.json`, real, measured):
  complex128 0.1491 s/step, complex64 0.0720 s/step, both at 12 qubits,
  200-step sample. Projected for the canonical drive (~52,584 steps):
  complex128 2.18h, complex64 1.05h.
- **Precision cross-check anomaly, logged not hidden**: complex64 vs.
  complex128 disagree by 5.87% max relative error on this exact config
  — far above the 1e-5 threshold this project normally requires.
  Consistent with a previously-flagged, never-fully-root-caused
  complex64/GPU precision issue from Sprint 2.6 (219% error observed
  there on a different config). Not investigated further this sprint
  (moot for the precision decision below); flagged as a real, open,
  unresolved item.
- **Decision: complex128.** 2.18h projected fits the 3h budget
  comfortably; no need to accept complex64's precision risk.
- **Batching decision**: true tensor-batched multi-config driving
  (stacking seeds/gamma1/V as a batch dimension through the density-
  matrix engine) was assessed as non-trivial to retrofit safely into
  `SequentialReservoir` within the available time — every core method
  (`conjugate_1q`, `conjugate_dense`, `_dissipate`, correlator
  extraction) would need batch-aware reshaping, and this exact code path
  has already produced multiple real, subtle bugs when modified under
  time pressure (Sprint 4's two throughput bugs). Per the superprompt's
  own explicit fallback ("if batching is non-trivial to implement in
  <30 min, run sequentially instead and note it"), ran the 2 configs
  (gamma1 tuned=0.03 vs. diagnostic_off=0.0, V=1, seed=123) **sequentially**
  instead. Logged, not silently absorbed.
- **Drive launched**: `scripts/phase1_v5_canonical_drive.py` on the H200
  (instance `bma-pr-gpu-h200-7f806292`), checkpointed every 2000 steps,
  driving train_seq+val_seq+test_seq continuously (test LABELS are never
  read until Phase 5's one-time evaluation, so this does not touch the
  test-split-scoring guard). Real projected total: ~4.4h for both
  configs sequentially.
