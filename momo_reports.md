# Momo Reports — Sprint 4 (REVISED) Live Log

Running summary of big steps and real results, written as they happen. Every
number here is traceable to a `results/*.json` file or an explicit note if
still in progress.

---

## 2026-07-24 — qBraid H200 direct access set up

Got Claude Code SSHing directly into the qBraid on-demand H200 instance
(`bma-pr-gpu-h200-2de2f5d6`), instead of the paste-results-back workflow.
Real setup issues hit and fixed along the way (all genuine environment bugs,
not guesses):

- qBraid CLI's `ssh setup` writes an `Include` path in Windows
  backslash format, which the Git Bash OpenSSH client can't parse — fixed by
  rewriting to a Unix-style path.
- The `ProxyCommand` itself also used a Windows-style Python path that Git
  Bash mangled — fixed the same way.
- Windows system proxy settings (a local SOCKS proxy at `127.0.0.1:10808`,
  probably a VPN tool) broke the SSH-over-WebSocket bridge — fixed by
  setting `NO_PROXY=*` in the `ProxyCommand` environment.
- Found a real bug in the installed `qbraid_core` package itself: its
  WebSocket bridge (`_run_bridge_with_auth`) uses
  `loop.connect_read_pipe(..., sys.stdin.buffer)`, which requires an
  IOCP-registrable (overlapped) Windows pipe handle. The handle SSH's
  `ProxyCommand` actually hands the child process isn't overlapped, so this
  crashes with `OSError: [WinError 6] The handle is invalid` — reproduced
  identically via both Git Bash and native `cmd.exe`/OpenSSH, so it's not an
  MSYS-specific issue. Worked around with a local shim
  (`C:\Users\there\qbraid_ssh_bridge_shim.py`) that reads stdin in a plain
  blocking background thread instead of through asyncio's pipe transport.
- Learned the hard way that the on-demand instance's container filesystem
  resets pip-installed packages outside `/home/jovyan` on every
  stop/restart (a stop happened once already, killing an in-progress v5
  run before its first checkpoint — real GPU time lost, nothing to recover).
  Fixed by installing with `pip install --user` so packages land inside the
  persistent home directory.

## 2026-07-24 — v4 windowed QRC, real GPU results, 16 qubits

`scripts/sprint4_v4_gpu_handoff.py` run on the H200, full pilot split
(train 2019-2021, eval 2022, all 22,176 windows, no subsampling):

- **Wall clock: ~120 seconds total** (89.7s train + 30.5s val) — dramatically
  faster than either of our pre-run guesses.
- Real, DM-tested result (`results/sprint4_v4_gpu_results_16q.json`):
  `residual` architecture shows a small but statistically significant
  (Diebold-Mariano p < 0.05) skill improvement over persistence at h=1
  (+0.05%), h=3 (+0.17%), h=6 (+0.22%) — not significant at h=12 (p=0.109).
  `direct` architecture is strongly negative at every horizon (expected —
  regressing on raw values overfits).
- Honest read: this is a genuine, small, statistically real effect, not a
  home-run result. Worth reporting as-is in the paper, not spun bigger.
- Given how fast 16q ran, immediately pushed to **20 qubits** (currently
  running on the H200 as of this entry — see next update for results).

## 2026-07-24 — instance auto-stopped mid-run, real GPU time lost

The H200 instance stopped itself (idle/heartbeat policy, not something we
triggered deliberately) while v5's first run was in progress, before it had
written even one checkpoint (v5 checkpoints after each of 12 drives). That
run is gone — no partial numbers to report, real GPU time lost. Container
filesystem also reset pip-installed packages outside `/home/jovyan`
(`scikit-learn`, `cupy` both had to be reinstalled, this time with
`--user` so they survive future stop/restarts). Both v5 and v4 (at 20
qubits) restarted clean afterward.

## 2026-07-24 — v4 at 20 qubits: real, complete result

`results/sprint4_v4_gpu_results_20q.json` — the run actually finished
(46.2 min: 2072.8s train + 701.1s val) right around when the SSH
connection to the instance dropped (the instance auto-stopped again,
~45 min after the previous restart — same pattern as before). Real
numbers, full 22,176-window pilot, no subsampling:

- `residual` architecture: tiny but DM-significant (p<0.05) positive
  skill vs. persistence at h=1 (+0.087%), h=3 (+0.196%), h=6 (+0.250%);
  NOT significant at h=12 (p=0.188, skill +0.16%).
- `direct` architecture: strongly negative at every horizon (same as
  16q — expected, regressing on raw values overfits).
- **Honest read**: this is essentially the SAME tiny effect size as the
  16-qubit run (+0.05% to +0.22% there). The effect is small, real, and
  robust to qubit count — going from 16 to 20 qubits did not meaningfully
  change it. This is useful, interpretable evidence either way: the
  paradigm shows a genuine (if modest) signal that isn't obviously a
  qubit-count-limited effect within this range.
- Cost confirms the earlier bottleneck theory: 20q measured
  0.125 s/window vs 16q's 0.0054 s/window — ~23x slower for a qubit-count
  increase that (dim + more pairs) should only cost ~16-28x more, so the
  scaling is roughly consistent with expectations, not some new blowup.

## 2026-07-24 — real, recurring instance-stop pattern; added resumable checkpointing

The H200 instance has now auto-stopped twice, both times roughly ~45
minutes after a (re)start, regardless of active GPU load (100%
utilization didn't prevent it). Cause not confirmed from the CLI's
available info (`qbraid compute usage`/`sessions` didn't show an obvious
session-length cap), but the pattern is consistent enough to plan around.
Added resumable checkpointing to both GPU scripts so a stop only costs a
few minutes of recompute, not a full restart:

- **v4**: checkpoints the accumulated feature array + progress index
  every 5 batches (embarrassingly parallel — no correctness risk in
  resuming).
- **v5**: checkpoints the full density matrix + step index + features-so-
  far every 500 steps. This one is trickier (ρ is recurrent state, must
  resume exactly) — caught and fixed a real off-by-one bug here during
  local testing (resuming at the checkpointed step itself, instead of the
  step after, would have silently double-applied one step's
  injection+evolution+dissipation onto an already-updated ρ, corrupting
  the whole rest of the trajectory). Verified byte-for-byte identical
  output between an uninterrupted drive and a simulated-crash-and-resume
  drive before redeploying.
- Also fixed a Windows-specific bug where deleting a just-loaded
  `.npz` checkpoint file failed with a file-lock error (`np.load` keeps
  the file handle open until explicitly closed).

## 2026-07-24 — v5 real throughput bug found and fixed (major)

The instance stopped a third time (~26 min in, same recurring pattern),
before v5 hit even one 500-step checkpoint. Root-caused instead of just
retrying blind: `extract_correlators_dm` (correlator readout) AND part of
`_inject` (the 13th-feature-onto-qubit-0 phase, which fires every step
for our real 13-feature/12-qubit data) were both forcing a full 12-qubit
density-matrix GPU->CPU->GPU roundtrip on literally every single step.
Rewrote both to stay entirely on the GPU array module until the very end
(a small feature vector, not the full density matrix, is what actually
needs to leave the GPU). Verified numerically identical to the old
(slow) implementation via direct unit tests before redeploying (max
diff ~3.5e-9, real physics unchanged, just faster).

**Result**: first 500-step checkpoint now lands in ~3 minutes, versus
never completing 500 steps in 45 minutes before — roughly a 15x
throughput improvement, a real and significant fix, not a guess.

**Honest revised timeline**: at this rate, a single full v5 drive
(35,064 steps: train+val) takes on the order of **~3.5 hours**. The
sprint's ablation plan calls for 12 such drives (2 gamma1 x 2 V x 3
seeds) = **~42 hours of GPU compute total**, spread across many
~45-minute sessions given the instance's recurring auto-stop (checkpointing
means each stop only costs a few minutes, not a full restart, but the
total wall-clock to finish all 12 drives is still large). Flagging this
now rather than silently grinding through it — worth deciding whether to
(a) let it run across many restart cycles over the coming hours, or
(b) reduce scope (e.g., fewer seeds, or the "tuned" config only, as a
faster paradigm-validation-only check) to get a verdict sooner.

## 2026-07-24 — second real perf fix, then a real qBraid infra failure

Rewrote correlator extraction again: it was still doing ~700+ individual
small GPU calls per step (a transpose+trace per qubit and per qubit-pair).
Replaced with a handful of large vectorized gather operations (same
math -- Pauli expectations expressed as diagonal sums and off-diagonal
element sums, not partial traces via reshape+transpose). Verified exact
match (rel err ~1e-16) against the already-validated reference before
deploying. Real measured result: **0.36 s/step -> 0.072 s/step, another
~5x**, for a combined ~75x speedup from where this started. At this
rate the full 3-year training data now fits the time budget (confirmed:
the run picked up all 26,304 train steps, not the 1yr fallback) --
2 drives (gamma1 on/off, current reduced scope) finish in **~1.4h total**,
and even the FULL original 12-drive/3-seed/V∈{1,4} ablation would now
cost **~21h instead of ~42-105h**.

Drive 1 got to ~90% complete (partway into the val phase) when the
instance auto-stopped again -- but this time, resuming failed
repeatedly: `qbraid compute server start` reported success (printed a
URL) but the instance's actual status stayed "stopped / Resume failed,
try again" across 6 consecutive attempts over several minutes. This is
a real failure on qBraid's backend, not something fixable from this
side. Also hit (and immediately fixed) a near-miss: `qbraid compute up
gpu-h200` was tried as a workaround and it provisions a **new** instance
rather than resuming the old one -- caught this within seconds and
terminated the accidental duplicate before it ran up double billing.

**Resolution**: abandoned the stuck instance, provisioned a genuinely
fresh on-demand H200 (`bma-pr-gpu-h200-7f806292`), reconfigured SSH,
reinstalled deps, redeployed the scripts, and relaunched v5 from scratch.
Given the ~75x speedup, redoing drive 1's ~90% lost progress costs
~40 minutes, not a real setback.

## 2026-07-24/25 — third real bug: NaN in the continuous sequence

Drive 1 actually completed a full real drive (43.98 min wall-clock for
26,304 train + 8,760 val steps, matching the throughput-fix estimates
exactly) -- then crashed at the readout stage: `sklearn.Ridge: Input X
contains NaN`. Root cause: `train_seq`/`val_seq` (the raw continuous
per-hour sequence) still contain NaN at real KORD ISD's missing SLP/WD
readings (~1-2% of rows: 613/26304 train rows, real numbers). v4's
windowed `X_train`/`X_val` are already NaN-free (the windowing pipeline
drops any window containing one), but v5 drives the raw sequence
CONTINUOUSLY with no windowing -- so a single NaN timestep corrupts the
recurrent density matrix for every step after it, silently, until it
finally surfaces at the Ridge-fit stage 44 minutes later.

Fixed in `scripts/sprint4_export_pilot_seq.py`: forward-fill (per
column, per split, never leaking across the train/val boundary or using
future values) with a backward-fill fallback for the rare case of NaN in
a split's very first rows. Standard, defensible handling for continuous
state-space methods fed real sensor data. Re-ran the export (confirmed
zero NaN in the output), re-uploaded, and relaunched v5 from scratch
(drive 1's previous ~44 minutes had NaN baked into the trajectory from
the start, so it wasn't reusable -- but at the current throughput that's
another ~44 min, not a major setback).

## 2026-07-25 — v5 drive 1 (gamma1=0.03, tuned) real, complete result

First full v5 drive succeeded end-to-end after the three fixes above
(throughput x2, NaN handling). Real numbers, full pilot (26,304 train +
8,760 val steps, 12-qubit reference config), 8 (architecture, horizon)
cells:

| architecture | h | skill vs persistence | DM p-value |
|---|---|---|---|
| direct | 1 | -163.8% | 0.101 |
| direct | 3 | -57.9% | 0.230 |
| direct | 6 | -21.2% | 0.387 |
| direct | 12 | **+8.76%** | **0.0023** |
| residual | 1 | -3.51% | 0.573 |
| residual | 3 | -1.36% | 0.859 |
| residual | 6 | +5.79% | 0.162 |
| residual | 12 | **+14.4%** | **~0.0** |

**This is a materially bigger and more significant effect than v4
showed** (v4's best was +0.25%, tiny but real; v5's residual h=12 here is
+14.4%, DM-significant at essentially p=0). Honest read: v5's recurrent/
continuous architecture appears to capture longer-range temporal
structure (h=12) much better than v4's fresh-window-per-sample
re-encoding -- both architectures get WORSE or flat at short horizons but
v5 clearly pulls ahead at h=12. This is exactly the kind of result that
would make V1 (paradigm validation) a clear pass, if it holds up.
**Caveat**: this is one drive (gamma1=0.03, V=1, one seed) -- the
critical next data point is drive 2 (gamma1=0, dissipation OFF), now
running, which tests whether dissipation is actually responsible for
this or whether an ideal closed-system reservoir does just as well
(the core physics question this ablation exists to answer).

## 2026-07-25 — v5 drive 2 (gamma1=0, diagnostic_off) complete + final verdict

Both v5 drives now done. Drive 2 (dissipation OFF) result: the `residual`
architecture's skill collapses to essentially zero at every horizon
(-0.006% to -0.064%, all indistinguishable from persistence) -- and
`direct` is catastrophically worse (-43% to -603%). Compare to drive 1
(dissipation ON): `residual` reaches +14.4% at h=12, DM-significant at
p≈0. **This is a clean, decisive ablation result: dissipation is what
makes the reservoir carry any predictive signal at all.** Without it, the
architecture is functionally useless on this task; with it, there's a
real, significant, if modest, effect.

Ran `scripts/sprint4_verdict.py` with the real v4 (20q) + v5 + baseline
data. Caught and fixed a real flaw in my own verdict criterion before
trusting it: V2 (competitive-vs-ESN) only checked h=1, which passed
almost automatically because ESN's real skill happens to be
catastrophically bad specifically at h=1 (-90.8%, a genuine finding from
the baselines run) -- meanwhile ESN actually **beats** the best QRC
result at h=3 (+2.1% vs -0.01%), h=6 (+17.9% vs +5.8%), and h=12 (+27.5%
vs +14.4%). Fixed the criterion to require winning a majority of
horizons, not just h=1.

**Final honest verdict**:
- **V1 (paradigm validation): PASS.** Real, DM-significant effect
  (residual, h=12, tuned gamma1: +14.4% skill, p≈0), and the ablation
  confirms it depends on dissipation specifically (vanishes to ~0% when
  gamma1=0).
- **V2 (competitive verdict): FAIL.** Best QRC pairing wins only 1/4
  horizons against dimension-matched ESN -- ESN is still ahead at h=3,
  6, 12 in the real numbers.
- **Branch: B** -- paradigm validated, not yet competitive with classical
  ESN. This is a real, physics-interesting, honestly-reported result,
  not a home run. Reporting it as such.

Full detail: `results/sprint4_verdict.json`, `results/sprint4_v5_results.json`,
`results/sprint4_v4_gpu_results_used_for_verdict.json`,
`results/sprint4_pilot_baselines.json`.

## 2026-07-25 — Sprint 4 closed out; Sprint 5 (IPC-matched tuning) done

Sprint 4 committed (see prior entries for the full result). Per the
master plan, Sprint 5 proceeds regardless of Sprint 4's Branch-B outcome
(the plan explicitly anticipates "sequential QRC doesn't beat the
dimension-matched ESN" as a real possibility, mitigated by rewarding
honest negative results).

**Sprint 5 — IPC-matched reservoir tuning** (the paper's novel-
contribution flag): built a task-demand profile (how much of the real
KORD forecasting task's predictability comes from low-order Legendre-
polynomial functions of its own lagged history) and matched it against
the reservoir's own supply-side IPC (extended the existing degree-1/2
framework to degree 3 via Hermite polynomials, verified against the
already-validated implementation to ~1e-9 before using it). All new code
covered by 7 new unit tests, all passing.

Real numbers, coarse (gamma1, input_scaling) grid at n_qubits=10 (CPU
fallback), 18 configs: best matched config gamma1=0.3, a=0.1 (captured
capacity 9.26, vs. the Sprint 2 reference's 4.53 — a real, +104%
improvement in demand/supply overlap). **Honest finding: this did NOT
translate into better forecast skill at h=6 or h=12** on a real (reduced-
sample) pilot comparison — matched wins at h=1/h=3, reference wins at
h=6/h=12. The sprint spec's own named hypothesis ("matched config
improves 6h skill") is not confirmed. Reported as a genuine negative
result, not spun — plausible explanation logged in
`docs/sprint_log/SPRINT_5_REPORT.md` (demand was computed on the
target's univariate self-history; the real readout uses the full
multivariate 13-feature state, so univariate-optimal isn't guaranteed
multivariate-optimal).

Figures: `figures/sprint5_demand_supply_heatmaps.png`,
`figures/sprint5_captured_capacity_bar.png`. Full data:
`results/ipc_matching.json`. Draft paper subsection:
`docs/ipc_matching.md`.

## 2026-07-25 — Sprint 6 (Full-Dataset Benchmark) real numbers + a real mistake

Full 14-year KORD record (2011-2024) confirmed available with acceptable
QC yield (missingness comparable to the 2019-2024 subset already used).
Locked split: train 2011-2020, val 2021-2022, test 2023-2024 (extended
`QRCx.data.splits.temporal_split`/`preprocess()` to accept val_year/
test_year as either a single int, as before, or a `(start, end)` tuple —
backward compatible, both paths tested).

**Real val-split classical benchmark result** (170 min wall-clock, FAST_MODE,
3-seed ESN, all 48 horizons DM-tested):
`results/full_benchmark_val.json`. Headline confirms Sprint 4's pilot-scale
finding at full 14-year scale: null-control Ridge (no reservoir at all,
+15.2%/+28.0% skill at h1/h6) matches or beats every ESN variant
(dim-matched -15.2%/+28.1%, ESN-500 +6.9%/+30.8%, Residual-ESN
+14.5%/+30.9%) and both QRC architectures. Given real GPU-infrastructure
time cost already spent in Sprint 4/5, decided (with the user) to reuse
Sprint 4's real pilot-scale v4/v5 numbers as the headline QRC entries
rather than launch another multi-hour qBraid campaign — flagged clearly
as pilot-scale, not full-record.

Also built and verified `scripts/reproduce.sh` (`--quick`, <30min,
canonical val split; `--full`, the real Sprint 6 headline run, ~170min
documented honestly).

**Real mistake, caught and fixed**: ran `reproduce.sh --quick` (which does
`pip install -e QRCx`) WHILE the locked test-split confirmatory run was
still executing in the background, in the same shared Python environment.
The pip install modified installed scikit-learn files on disk mid-flight,
corrupting the long-running process's view of the package and crashing it
~3 hours in (`ImportError: cannot import name 'get_tags'` deep inside
sklearn's internal lazy-import chain) — a race condition I caused, not a
real bug in the benchmark code (confirmed: KRR fits fine in a fresh
process afterward). Relaunched the test-split run cleanly, and will not
run anything else that touches the Python environment until it finishes.

## 2026-07-25 — Sprint 6 complete: real, locked test-split headline numbers

Test-split confirmatory run finished clean (139.6 min) after the relaunch.
**Confirmed on the locked, one-time-touched test split**: null-control
Ridge (+14.7%/+27.0% skill at h1/h6) still matches or beats every ESN
variant and both QRC architectures (reused from Sprint 4's real pilot-scale
numbers, +0.087%/+0.25% for v4, -3.51%/+5.79% for v5) — the same
conclusion as val and as Sprint 4's pilot-scale finding. Not an artifact
of tuning-set choice or dataset size.

`scripts/reproduce.sh` built and the `--quick` mode verified for real
(<12 min, real numbers). README results block regenerated with the full
val+test table. Full writeup: `docs/sprint_log/SPRINT_6_REPORT.md`.

Full test suite re-run in progress before committing (checking the
`splits.py`/`preprocess()` val/test-as-range extension for regressions).

## 2026-07-25 — Sprint 8 (Scaling, Shots & Noise Characterization) done

Real data for all four required axes:

- **Qubit scaling** (N=4,6,8,10,12, reference config, real pilot data):
  MC/IPC non-monotonic (dip ~N=8-10, sharp rise at N=12) — a real
  measured pattern, not yet explained further. One forecast-skill data
  point (N=8, h=6: -66.5%) is very likely a small-sample artifact (only
  ~74 effective test points at that sample size) rather than a real
  qubit-count effect — flagged honestly, not smoothed over.
- **Shot noise**: real, positive hardware-readiness finding — skill is
  statistically indistinguishable from exact even at S=1,000 shots (well
  inside the bootstrap CI half-width of 4.55%). Simulated via post-hoc
  binomial sampling of the already-driven exact correlators (no
  redundant re-drives per shot budget).
- **Encoding density**: found a real, previously-undiscovered
  characteristic of the production code —
  `SequentialDissipativeQRC._inject_zz` never references
  `self.input_scaling` at all (confirmed by direct code read), so `zz`
  injection is completely insensitive to that hyperparameter. Not fixed
  (no test coverage asserting the intended behavior, and changing
  established physics this late without a clear signal it's unintentional
  felt riskier than documenting it clearly) — logged as a real finding
  for future deliberate resolution.
- **Noise reconciliation**: resolved Sprint 2's contradictory noise
  paragraph using this project's own real data — v4's encoding-noise
  harm vs. v5's dissipation-as-resource are not in tension, just
  architecture-dependent (backed by Sprint 4's real ablation showing v5
  skill collapses to ~0% with dissipation off).

Four figures written (`figures/sprint8_*.png`), full data in
`results/characterization.json`. Full writeup:
`docs/sprint_log/SPRINT_8_REPORT.md`.

## Next up

- Commit Sprint 8.
- Then: per the master plan sequencing, Sprint 9 (Paper, README,
  Reproducibility Package) is the final sprint (S7 remains pending
  organizer reply, not blocking).
