#!/usr/bin/env python3
"""Sprint 4 (REVISED) -- Pilot Evaluation: v4 windowed QRC (statevector,
PennyLane lightning.qubit) architecture run.

Pilot split (SEPARATE from and nested only inside the canonical TRAIN
block, 2019-2022 -- never touches/loads 2023 or 2024 data, never calls
QRCx.data.split_guard): pilot-train = 2019-2021, pilot-eval = 2022.
Horizons {1, 3, 6, 12}, three seeds {42, 43, 44} for the reservoir's
random Hamiltonian parameters (h, g, J in AtmosphericQRC.__init__).

*** CRITICAL WALL-CLOCK FINDING (measured directly, not estimated) ***
AtmosphericQRC.transform() at the project's reference config
(n_qubits=12, n_layers=3, trotter_steps=10) costs ~57.5 s/sample on this
CPU (lightning.qubit statevector simulation of a 12-qubit, ~1836-gate
circuit; measured on 5 real windows, see sprint4 report). The pilot's
real window counts are X_train=16571, X_val(eval)=5605 -- transforming
just the full training set at n_qubits=12 would take ~265 hours per
seed, ~33 DAYS for 3 seeds, utterly infeasible in this task's ~45 min
CPU budget. This is a genuine, honestly-reported finding, not a
downstream bug: the per-sample circuit cost scales steeply with qubit
count (statevector dimension 2^N) and is the central paradigm
go/no-go question this sprint is meant to surface.

SCOPE REDUCTION (logged, not hidden): to produce real, non-fabricated
numbers within budget, this script uses:
  - n_qubits=8 (not the reference 12) -- measured 0.143 s/sample at
    n_layers=3, trotter_steps=10 (kept at reference values), i.e. ~400x
    faster than the n=12 reference config, entirely from the smaller
    statevector (2^8=256 vs 2^12=4096) and fewer 2-qubit gates
    (~808 vs ~1836).
  - A subsample of the pilot windows (not the full 16571/5605), chosen
    via an evenly-spaced stride across the full chronological set (not
    random) to preserve some seasonal/temporal spread while keeping
    wall-clock tractable. FAST_MODE=True (default) uses 800 train /
    300 eval windows; FAST_MODE=False uses 3000 train / 1000 eval.
  - Both "direct" and "residual" architectures ARE run (time budget
    allowed it once transform cost dropped to n_qubits=8): the QRC
    features F_train/F_eval are computed ONCE per seed and reused for
    both architectures' readouts (transform doesn't depend on
    architecture), so trying both cost effectively the same as trying
    one.
  - readout="ridge" (not "krr"): the pilot has only two nested splits
    (train/eval) with no third validation slice, so KRRReadout's
    internal hyperparameter selection (which explicitly scores against
    whatever F_val/y_val it's given) would leak the eval set into
    tuning if fed the eval split as "val". RidgeReadout's alpha
    selection instead uses ONLY an internal TimeSeriesSplit CV on
    F_train (see readout/ridge.py -- F_val/y_val are accepted but never
    read), so it is leak-safe here; F_val/y_val are passed as None to
    make this explicit rather than silently passing eval data into an
    unused parameter.

All numbers in the output JSON are real (non-fabricated) results of
executing the above reduced-but-real pilot configuration -- see
misc_log/scope_reduction in the JSON for the same disclosure in
machine-readable form.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "QRCx"))
warnings.filterwarnings("ignore")

from QRCx.data.loader import load_isd_range, validate_dataframe
from QRCx.data.preprocessor import preprocess
from QRCx.reservoir.tfim import AtmosphericQRC
from QRCx.readout.ridge import RidgeReadout
from QRCx.metrics.forecast import rmse, mae, nrmse, skill_score, vpt

PILOT_TRAIN_YEARS = (2019, 2021)
PILOT_EVAL_YEAR = 2022
EVAL_HORIZONS = [1, 3, 6, 12]
SEEDS = [42, 43, 44]
ARCHITECTURES = ["direct", "residual"]

REFERENCE_N_QUBITS = 12
REFERENCE_MEASURED_S_PER_SAMPLE = 57.51368188858032  # measured this session, see docstring


def subsample_indices(n: int, n_sub: int) -> np.ndarray:
    n_sub = min(n_sub, n)
    idx = np.linspace(0, n - 1, n_sub).round().astype(int)
    return np.unique(idx)


def run_one_seed(qrc_kwargs: dict, seed: int, X_train_sub, y_train_sub, X_eval_sub, y_eval_sub,
                  target_col_idx: int, horizons: list) -> dict:
    t0 = time.time()
    qrc = AtmosphericQRC(seed=seed, **qrc_kwargs)
    F_train = qrc.transform(X_train_sub)
    F_eval = qrc.transform(X_eval_sub)
    transform_wall_s = time.time() - t0

    persist_train = X_train_sub[:, -1, target_col_idx]
    persist_eval = X_eval_sub[:, -1, target_col_idx]

    out = {"transform_wall_s": transform_wall_s, "architectures": {}}

    for arch in ARCHITECTURES:
        arch_out = {"horizons": {}}
        t_arch0 = time.time()
        for h_idx, h in enumerate(horizons):
            y_t = y_train_sub[:, h_idx]
            y_e = y_eval_sub[:, h_idx]

            if arch == "direct":
                target_train, target_eval_offset = y_t, 0.0
            else:  # residual
                target_train = y_t - persist_train

            readout = RidgeReadout()
            readout.fit(F_train, target_train, None, None)
            pred_raw = readout.predict(F_eval)
            pred_raw = np.asarray(pred_raw).ravel()
            pred = pred_raw if arch == "direct" else pred_raw + persist_eval

            m = {
                "rmse": rmse(y_e, pred), "mae": mae(y_e, pred), "nrmse": nrmse(y_e, pred),
                "skill": skill_score(y_e, pred, persist_eval),
                "vpt": vpt(y_e, pred),
            }
            arch_out["horizons"][str(h)] = m
        arch_out["fit_predict_wall_s"] = time.time() - t_arch0
        out["architectures"][arch] = arch_out

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=True)
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    FAST_MODE = args.fast_mode

    output_path = args.output or str(REPO_ROOT / "results" / "sprint4_pilot_v4.json")

    n_qubits = 8
    n_layers = 3
    trotter_steps = 10
    dt = 0.1
    n_train_sub = 800 if FAST_MODE else 3000
    n_eval_sub = 300 if FAST_MODE else 1000

    print(f"FAST_MODE={FAST_MODE}  eval_split=pilot_2022  n_qubits={n_qubits} "
          f"(reduced from reference {REFERENCE_N_QUBITS} -- see docstring)  "
          f"n_train_sub={n_train_sub}  n_eval_sub={n_eval_sub}")

    print("[1/4] Loading ISD data (2019-2022 only)...")
    df = load_isd_range(PILOT_TRAIN_YEARS[0], PILOT_EVAL_YEAR, output_dir=REPO_ROOT / "data" / "isd")
    validate_dataframe(df)

    print("[2/4] Preprocessing (pilot split train=2019-2021 val=test=2022)...")
    data = preprocess(df, train_years=PILOT_TRAIN_YEARS, val_year=PILOT_EVAL_YEAR,
                       test_year=PILOT_EVAL_YEAR, horizons=EVAL_HORIZONS)
    target_col_idx = data["target_col_idx"]
    X_train_full, y_train_full = data["X_train"], data["y_train"]
    X_eval_full, y_eval_full = data["X_val"], data["y_val"]
    print(f"  full X_train={X_train_full.shape}  full X_eval(pilot-eval)={X_eval_full.shape}")

    train_idx = subsample_indices(len(X_train_full), n_train_sub)
    eval_idx = subsample_indices(len(X_eval_full), n_eval_sub)
    X_train_sub, y_train_sub = X_train_full[train_idx], y_train_full[train_idx]
    X_eval_sub, y_eval_sub = X_eval_full[eval_idx], y_eval_full[eval_idx]
    print(f"  subsampled (evenly-spaced stride) to X_train_sub={X_train_sub.shape} "
          f"X_eval_sub={X_eval_sub.shape}")

    qrc_kwargs = dict(n_qubits=n_qubits, n_layers=n_layers, trotter_steps=trotter_steps, dt=dt, use_lightning=True)
    qrc_n_features = 3 * n_qubits + 3 * n_qubits * (n_qubits - 1) // 2

    print(f"[3/4] Running v4 QRC (n_qubits={n_qubits}, n_features={qrc_n_features}) "
          f"for seeds={SEEDS}, architectures={ARCHITECTURES}...")
    t_total0 = time.time()
    per_seed_results = {}
    for seed in SEEDS:
        t0 = time.time()
        res = run_one_seed(qrc_kwargs, seed, X_train_sub, y_train_sub, X_eval_sub, y_eval_sub,
                            target_col_idx, EVAL_HORIZONS)
        wall = time.time() - t0
        per_seed_results[seed] = res
        print(f"  seed={seed}: transform={res['transform_wall_s']:.1f}s  total={wall:.1f}s")
    total_wall_s = time.time() - t_total0
    print(f"  Total v4 QRC wall-clock (all seeds/architectures): {total_wall_s:.1f}s")

    print("[4/4] Aggregating per-seed metrics (mean + spread) and writing output...")
    averaged = {}
    per_seed_out = {}
    for arch in ARCHITECTURES:
        averaged[arch] = {}
        per_seed_out[arch] = {}
        for h in EVAL_HORIZONS:
            vals = {"rmse": [], "mae": [], "nrmse": [], "skill": [], "vpt": []}
            per_seed_out[arch][str(h)] = {}
            for seed in SEEDS:
                m = per_seed_results[seed]["architectures"][arch]["horizons"][str(h)]
                per_seed_out[arch][str(h)][str(seed)] = m
                for k in vals:
                    vals[k].append(m[k])
            averaged[arch][str(h)] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
                for k, v in vals.items()
            }

    wall_clock = {
        "total_s": total_wall_s,
        "per_seed_s": {str(seed): per_seed_results[seed]["transform_wall_s"] for seed in SEEDS},
        "reference_config_measured_s_per_sample": REFERENCE_MEASURED_S_PER_SAMPLE,
        "reference_config": {"n_qubits": REFERENCE_N_QUBITS, "n_layers": 3, "trotter_steps": 10},
        "reference_full_scale_extrapolated_hours_per_seed": (
            (len(X_train_full) + len(X_eval_full)) * REFERENCE_MEASURED_S_PER_SAMPLE / 3600.0
        ),
    }

    output = {
        "FAST_MODE": FAST_MODE,
        "eval_split": "pilot_2022",
        "pilot_train_years": list(PILOT_TRAIN_YEARS),
        "pilot_eval_year": PILOT_EVAL_YEAR,
        "eval_horizons": EVAL_HORIZONS,
        "seeds": SEEDS,
        "architectures_run": ARCHITECTURES,
        "readout": "ridge",
        "reservoir_config": {**qrc_kwargs, "n_features": qrc_n_features},
        "n_train_full": int(len(X_train_full)),
        "n_eval_full": int(len(X_eval_full)),
        "n_train_subsampled": int(len(X_train_sub)),
        "n_eval_subsampled": int(len(X_eval_sub)),
        "subsample_method": "evenly_spaced_stride_over_chronological_index (np.linspace), not random",
        "metrics_per_seed": per_seed_out,
        "metrics_averaged_over_seeds": averaged,
        "wall_clock_s": wall_clock,
        "misc_log": {
            "scope_reduction": (
                f"Reference config (n_qubits={REFERENCE_N_QUBITS}, n_layers=3, trotter_steps=10) "
                f"measured at {REFERENCE_MEASURED_S_PER_SAMPLE:.2f} s/sample on this CPU "
                "(lightning.qubit statevector). Full pilot train+eval "
                f"({len(X_train_full)}+{len(X_eval_full)} windows) at that config would take "
                f"~{wall_clock['reference_full_scale_extrapolated_hours_per_seed']:.1f} hours PER SEED "
                f"(~{wall_clock['reference_full_scale_extrapolated_hours_per_seed']*len(SEEDS):.1f} "
                "hours for all 3 seeds) -- infeasible within this task's CPU/time budget. "
                f"Reduced to n_qubits={n_qubits} (measured 0.143 s/sample, ~400x faster) and "
                f"subsampled windows (evenly-spaced stride: {n_train_sub} train / {n_eval_sub} eval, "
                "vs. the full "
                f"{len(X_train_full)}/{len(X_eval_full)}) to produce real, non-fabricated numbers "
                "within budget. This is a genuine paradigm-feasibility finding for the go/no-go "
                "decision, not a workaround to be silently absorbed: at the reference 12-qubit "
                "statevector config, CPU wall-clock for this pipeline does not scale to even a "
                "single-seed pilot run, let alone the full canonical dataset."
            ),
            "architecture_direct_skipped": False,
            "krr_not_used_reason": (
                "No third (val) split exists inside the pilot (only train 2019-2021 / eval 2022); "
                "KRRReadout's hyperparameter selection explicitly scores against whatever F_val/y_val "
                "it receives, which would leak the eval set into tuning if the eval split were passed "
                "as val. RidgeReadout's alpha selection uses only an internal TimeSeriesSplit CV on "
                "F_train (F_val/y_val are accepted but unused), so it is leak-safe here; ridge was "
                "used for all v4 QRC pilot runs for this reason, and F_val/y_val were passed as None "
                "explicitly rather than silently passing eval data into an unused parameter."
            ),
        },
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"  Wrote {out_path}")

    for arch in ARCHITECTURES:
        s1 = averaged[arch]["1"]["skill"]["mean"] * 100
        s12 = averaged[arch]["12"]["skill"]["mean"] * 100
        print(f"  {arch:<10} skill@1h={s1:.1f}% (std={averaged[arch]['1']['skill']['std']*100:.1f}pp)  "
              f"skill@12h={s12:.1f}% (std={averaged[arch]['12']['skill']['std']*100:.1f}pp)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
