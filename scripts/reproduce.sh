#!/bin/bash
# reproduce.sh: clean checkout -> data download -> headline table.
#
# CANONICAL SPLIT (train 2019-2022 / val 2023 / test 2024) is the paper's
# headline split as of the FINAL SPRINT reconciliation
# (docs/sprint_log/FINAL_SPRINT.md, Phase 1) -- it fixes a real earlier
# inconsistency where classical baselines were scored on the full
# 2011-2024 record while the QRC rows used a different pilot split.
#
# Two variants:
#   ./scripts/reproduce.sh --quick   (default) real classical-baseline
#       numbers on the canonical val split (train 2019-2022, val 2023),
#       FAST_MODE tuning grid -- measured ~7 minutes wall-clock (see
#       docs/sprint_log/SPRINT_3_REPORT.md). Demonstrates the full
#       real-data pipeline end-to-end quickly for judges. Does NOT include
#       the QRC rows (v4/v5), which require GPU access -- see below.
#   ./scripts/reproduce.sh --full    the full-record (2011-2024) classical
#       benchmark, train 2011-2020 / val 2021-2022 / test 2023-2024,
#       3-seed ESN, horizons 1-48, DM test + bootstrap CI at every horizon.
#       Retained as a SUPPLEMENTARY robustness check (same qualitative
#       conclusion holds at this larger, differently-split scale) -- it is
#       NOT the paper's headline table anymore (see canonical split above).
#       Measured real wall-clock: val split 169.9 minutes (2.83h); test
#       split touched once already (results/full_benchmark_test.json).
#
# The QRC rows (v4 20-qubit statevector, v5 12-qubit density matrix) on
# the canonical split require GPU access (measured on an H200: v5 ~1.05-
# 2.18h/config depending on precision, v4 ~1.14h for all three splits) --
# see scripts/phase1_v5_canonical_drive.py, scripts/phase5_v4_canonical_drive.py,
# and scripts/phase5_final_benchmark.py, which combines their output with
# the classical baselines above into results/full_matched_benchmark_{val,test}.json.
# This is not a <30min judge-friendly operation and is documented as such,
# not minimized; the already-driven feature arrays are not redistributed
# in this repo (large, regeneratable) but the summary JSONs are.
#
# --full additionally requires --unlock-test to actually score against
# the locked test split (same split-ledger discipline as
# scripts/sprint6_full_benchmark.py itself); without it, --full scores
# against val only.
set -e
cd "$(dirname "$0")/.."

MODE="quick"
UNLOCK_TEST=""
for arg in "$@"; do
    case "$arg" in
        --quick) MODE="quick" ;;
        --full) MODE="full" ;;
        --unlock-test) UNLOCK_TEST="--unlock-test" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

echo "=== Sprint 6 reproduce.sh (mode=$MODE) ==="
echo "[1/3] Installing dependencies..."
pip install -q -e QRCx 2>&1 | tail -5 || pip install -q -r requirements.txt 2>&1 | tail -5 || true

if [ "$MODE" = "quick" ]; then
    echo "[2/3] Downloading data (canonical range 2019-2024, cached under data/isd/)..."
    echo "[3/3] Running classical fairness protocol on the canonical val split (FAST_MODE, ~7 min real measured wall-clock)..."
    python scripts/generate_baselines.py --fast-mode --split val
    echo "Done. Headline table (not the full-record paper numbers -- see --full): results/baselines_val.json"
else
    echo "[2/3] Downloading full data (2011-2024, cached under data/isd/)..."
    echo "[3/3] Running Sprint 6 full-dataset benchmark (FAST_MODE, real measured wall-clock ~170 min for val alone -- this is NOT a quick operation)..."
    python scripts/sprint6_full_benchmark.py --fast-mode --split val
    if [ -n "$UNLOCK_TEST" ]; then
        echo "Running the ONE-TIME locked test-split confirmatory pass (--unlock-test was passed)..."
        python scripts/sprint6_full_benchmark.py --fast-mode --split test --unlock-test
    fi
    echo "Done. Headline table(s): results/full_benchmark_val.json (results/full_benchmark_test.json if --unlock-test)"
fi
