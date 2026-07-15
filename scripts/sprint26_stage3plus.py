"""Sprint 2.6 continuation: safe Stage 3+ (12q verification, gate,
IPC/MC), picking up from sprint26_pipeline.py's Stage 2 checkpoint.

Fixes the flaw in sprint26_pipeline.py's original Stage 3: that version's
very first move was to time ONE full (~4200-step) 12-qubit config, with no
way to know in advance whether that "probe" itself would take minutes or
>24 hours. This version instead:
  1. Loads the Stage 2 checkpoint (sprint26_results.json) and takes its
     ranked "topk_refine_complex128" list.
  2. For each candidate (best first), drives a SHORT sample (SAMPLE_STEPS,
     default 200) at 12 qubits with that exact config, measures real
     s/step, and extrapolates to the full protocol's total step count
     BEFORE committing to a full run. Prints the projected hours.
  3. Only runs the full 12q protocol for a candidate if its projected cost
     fits within the remaining time budget (REMAINING_BUDGET_HOURS below
     -- set this to your actual remaining time before running). Skips
     (does not run) candidates that don't fit, logging why.
  4. Proceeds to the gate-stability check (3 w_in seeds) only for
     candidates that were actually run, each also cost-projected first.
  5. IPC/MC characterization at the winning config last (cheaper: V=1,
     200 iid steps per gamma1 point).

If NO candidate fits the remaining budget at 12 qubits, this is reported
honestly as "12q verification not completed within the time-box" using
the 10q full-protocol numbers (already real, valid data) as the best
available evidence -- per the sprint's own time-boxing rule ("if this
hasn't finished in ~2 days, whatever completed so far should be reported
as-is").

Usage in qBraid Lab (same directory/kernel as sprint26_pipeline.py and
its sprint26_results.json checkpoint):
    !python sprint26_stage3plus.py
    # or paste into a cell; set REMAINING_BUDGET_HOURS first to whatever
    # time you actually have left in your 2-day window.
"""
import json
import time

import numpy as np

import sprint26_pipeline as s26

CHECKPOINT_PATH = "sprint26_results.json"
OUT_PATH = "sprint26_stage3plus_results.json"
SAMPLE_STEPS = 200
REMAINING_BUDGET_HOURS = 24.0  # <-- SET THIS to your actual remaining time before running
GATE_BUDGET_HOURS = 6.0  # separate, smaller budget reserved for the 3-seed gate-stability check


def project_12q_cost(cfg, protocol=None):
    """Drive SAMPLE_STEPS at 12 qubits with `cfg` and extrapolate to the
    full protocol's total step count. Returns (projected_hours, s_per_step)."""
    protocol = protocol or s26.FULL_PROTOCOL
    washout = cfg["washout"]
    effective_start = max(washout, s26.AR_TAPS_ORDER)
    total_steps_full = protocol["n_train"] + protocol["n_test"] + effective_start

    w_in = np.random.default_rng(cfg["w_in_seed"]).uniform(0.5, 1.5, size=12)
    g = 1.0
    J = cfg["jg_ratio"] * g
    res = s26.SequentialReservoir(
        n_qubits=12, tau=cfg["tau"], trotter_steps=10, gamma1=cfg["gamma1"],
        gamma2=cfg["gamma2"], J=J, g=g, input_scaling=cfg["a"], w_in=w_in,
        multiplexing=cfg["V"], use_gpu=s26.HAS_CUPY, dtype_str="complex128",
    )
    sample_seq = np.random.default_rng(0).uniform(-1, 1, size=(SAMPLE_STEPS, 12)) * w_in[None, :]
    t0 = time.perf_counter()
    res.drive(sample_seq)
    elapsed = time.perf_counter() - t0
    s_per_step = elapsed / SAMPLE_STEPS
    projected_hours = total_steps_full * s_per_step / 3600
    return projected_hours, s_per_step, total_steps_full


def main():
    with open(CHECKPOINT_PATH) as f:
        checkpoint = json.load(f)

    stages = checkpoint.get("stages", checkpoint)  # tolerate either shape
    refined = stages.get("topk_refine_complex128") or stages.get("top10_refine_complex128")
    if not refined:
        raise RuntimeError(
            f"No 'topk_refine_complex128'/'top10_refine_complex128' key found in {CHECKPOINT_PATH}. "
            "Make sure Stage 2 has finished and the checkpoint file is up to date."
        )
    refined = [r for r in refined if "error" not in r]
    refined.sort(key=lambda r: r["track_nmse"])
    candidates = refined[:3]
    print(f"Loaded {len(refined)} Stage-2 configs; considering top {len(candidates)} for 12q verification.")

    remaining_budget = REMAINING_BUDGET_HOURS
    verified12q = []
    skipped = []

    print(f"\n=== STAGE 3 (safe): cost-projecting top candidates before running at 12q ===")
    for cfg in candidates:
        c = {k: cfg[k] for k in s26.CFG_KEYS}
        proj_hours, s_per_step, total_steps = project_12q_cost(c)
        print(f"Candidate (gamma1={c['gamma1']:.4f}, V={c['V']}, washout={c['washout']}): "
              f"sample gave {s_per_step:.3f} s/step -> projected {proj_hours:.2f}h for "
              f"{total_steps} full-protocol steps. Remaining budget: {remaining_budget:.2f}h.")
        if proj_hours > remaining_budget:
            print(f"  -> SKIPPING (projected {proj_hours:.2f}h exceeds remaining budget "
                  f"{remaining_budget:.2f}h). Not run.")
            skipped.append({**c, "projected_hours": proj_hours})
            continue
        print(f"  -> Running (fits budget).")
        t0 = time.perf_counter()
        entry = s26.run_one_config(c, n_qubits=12, dtype_str="complex128", use_gpu=s26.HAS_CUPY,
                                    protocol=s26.FULL_PROTOCOL)
        actual_hours = (time.perf_counter() - t0) / 3600
        remaining_budget -= actual_hours
        verified12q.append(entry)
        print(json.dumps(entry))
        print(f"  Actual time: {actual_hours:.2f}h. Remaining budget now: {remaining_budget:.2f}h.")

    result = {
        "stage3_candidates_considered": len(candidates),
        "stage3_verified_12q": verified12q,
        "stage3_skipped": skipped,
    }

    if not verified12q:
        print("\nNo candidate fit the remaining time budget at 12 qubits. "
              "Reporting Stage 2's 10-qubit full-protocol results as the best "
              "available evidence, per the sprint's time-boxing rule.")
        result["gate_check"] = {
            "gate_pass": False,
            "reason": "12q verification not completed within the time-box; no 10q->12q "
                      "extrapolation is asserted as a pass/fail. See stage3_skipped for "
                      "projected costs.",
            "best_10q_full_protocol_result": candidates[0] if candidates else None,
        }
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWritten to {OUT_PATH}. Please share this file's contents back.")
        return result

    print("\n=== STAGE 4 (safe): gate-stability check across 3 w_in seeds, cost-projected first ===")
    winner = min(verified12q, key=lambda r: r["track_nmse"])
    winner_cfg = {k: winner[k] for k in s26.CFG_KEYS if k != "w_in_seed"}
    proj_hours, s_per_step, total_steps = project_12q_cost({**winner_cfg, "w_in_seed": 1})
    print(f"Winner's config: projected {proj_hours:.2f}h per seed x 3 seeds "
          f"= {3*proj_hours:.2f}h total. Remaining budget: {remaining_budget:.2f}h.")

    gate_runs = []
    if 3 * proj_hours > max(remaining_budget, GATE_BUDGET_HOURS):
        print("Gate-stability check does not fit remaining/gate budget -- SKIPPING. "
              "Reporting the single verified 12q result without seed-stability confirmation.")
        result["gate_check"] = {
            "gate_pass": False,
            "reason": "Gate-stability check (3 seeds) skipped -- did not fit the time budget. "
                      "A single 12q verified result exists (see stage3_verified_12q) but "
                      "'stable across 3 seeds' per the spec's gate criterion was not confirmed.",
            "single_verified_result": winner,
        }
    else:
        for seed in [1, 2, 3]:
            c = {**winner_cfg, "w_in_seed": seed}
            entry = s26.run_one_config(c, n_qubits=12, dtype_str="complex128", use_gpu=s26.HAS_CUPY,
                                        protocol=s26.FULL_PROTOCOL)
            gate_runs.append(entry)
            print(json.dumps(entry))
        gate_pass = all(r["delta_nmse_relative"] >= 0.15 for r in gate_runs)
        result["gate_check"] = {
            "winner_config": winner_cfg, "runs_across_seeds": gate_runs, "gate_pass": gate_pass,
            "nmse_trajectory": {
                "sprint1_best": 0.945, "sprint2_voided": 0.398, "sprint2_5_valid_protocol": 0.224,
                "sprint2_6_final_track_nmse": winner["track_nmse"],
            },
        }
        print(f"\nGATE {'PASSED' if gate_pass else 'FAILED'}")

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== STAGE 5: IPC/MC characterization at winning 12q config (gamma1 grid) ===")
    w_in = np.random.default_rng(winner["w_in_seed"]).uniform(0.5, 1.5, size=12)
    ipc_records = []
    for gamma1 in [0.0, 0.01, 0.03, 0.1, 0.3, 0.5]:
        g = 1.0
        J = winner["jg_ratio"] * g
        res = s26.SequentialReservoir(
            n_qubits=12, tau=winner["tau"], trotter_steps=10, gamma1=gamma1,
            gamma2=winner["gamma2"], J=J, g=g, input_scaling=winner["a"], w_in=w_in,
            multiplexing=1, use_gpu=s26.HAS_CUPY, dtype_str="complex128",
        )
        rng_u = np.random.default_rng(0)
        u_iid = rng_u.standard_normal(200)
        seq = u_iid[:, None] * w_in[None, :]
        feats = res.drive(seq)
        mc_ipc = s26.measure_mc_ipc(u_iid, feats)
        record = {"gamma1": gamma1, **mc_ipc}
        ipc_records.append(record)
        print(json.dumps(record))
    result["ipc_mc_characterization_12q"] = {
        "winner_config": {k: winner[k] for k in ["a", "gamma2", "tau", "jg_ratio", "w_in_seed"]},
        "records": ipc_records,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nAll done. Full results in {OUT_PATH} -- please share this file's contents back.")
    return result


if __name__ == "__main__":
    main()
