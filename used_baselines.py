#!/usr/bin/env python3
"""
used_baselines.py — Classical baselines with full FSDH curve (1..48h).

Same ISD data pipeline as winning_solution.py (13-feature engineering,
80/10/10 split, climatological deseasonalization, StandardScaler).

Baselines: Persistence, ARIMA(2,1,2), ESN-500, ESN-5000
Metrics:   RMSE, MAE, Skill, FSDH (full 48h curve), VPT
Output:    Terminal table + FSDH curve figure

Usage:
    python used_baselines.py
"""

import sys, os, warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.preprocessing import StandardScaler
from winning_solution import (
    load_isd_range, engineer_features,
    generate_synthetic_weather, build_features_from_array,
    rmse, mae, nrmse, skill_score, compute_vpt_curve,
    PersistenceBaseline, ARIMABaseline, ESNBaseline,
    ExperimentConfig, HAS_PANDAS,
)


@dataclass
class BaselineConfig:
    """Configuration for the classical baselines experiment.

    All fields have sensible defaults.  Override only what you need::

        cfg = BaselineConfig(fsdh_max=24, esn_nodes=[500, 2000])
        main(cfg)
    """
    seed: int = 42
    """Random seed."""
    fsdh_max: int = 48
    """Maximum forecast horizon for the FSDH curve."""
    eval_h: List[int] = field(default_factory=lambda: [1, 6])
    """Horizons (hours) for the evaluation table."""
    window: int = 24
    """Input window length in hours."""
    target_col: int = 0
    """Target column index (0 = T_db)."""
    esn_nodes: List[int] = field(default_factory=lambda: [500, 5000])
    """List of ESN sizes to evaluate."""
    arima_order: Tuple[int, int, int] = (2, 1, 2)
    """ARIMA (p, d, q) order."""

    @property
    def all_h(self) -> List[int]:
        return list(range(1, self.fsdh_max + 1))


def make_win(arr, h_list, window=24, target_col=0):
    W, mx, tci = window, max(h_list), target_col
    xl, yl = [], []
    for i in range(W, len(arr) - mx + 1):
        w = arr[i - W:i]
        t = np.array([arr[i + h - 1, tci] for h in h_list])
        if not (np.isnan(w).any() or np.isnan(t).any()):
            xl.append(w); yl.append(t)
    return np.array(xl), np.array(yl)


def main(cfg: Optional[BaselineConfig] = None):
    if cfg is None:
        cfg = BaselineConfig()
    c = cfg

    FSDH_MAX = c.fsdh_max
    EVAL_H = c.eval_h
    ALL_H = c.all_h

    print("=" * 70)
    print(f"  CLASSICAL BASELINES — Persistence | ARIMA | ESN-{c.esn_nodes}")
    print(f"  FSDH curve: 1..{FSDH_MAX}h")
    print("=" * 70)

    # ── 1. DATA ──
    print("\n[1/4] Loading ISD KORD data...")
    if HAS_PANDAS:
        df = load_isd_range(years=[2019])
        if df is not None and len(df) > 1000:
            df = engineer_features(df); data_arr = df.to_numpy()
        else:
            n_syn = 2000; data_arr = generate_synthetic_weather(n_syn, seed=c.seed)
            data_arr = build_features_from_array(data_arr)
    else:
        data_arr = generate_synthetic_weather(2000, seed=c.seed)
        data_arr = build_features_from_array(data_arr)

    # ── 2. PREPROCESS ──
    print("\n[2/4] Preprocessing (80/10/10)...")
    N = len(data_arr); n_test = int(N * 0.10); n_val = int(N * 0.10)
    train_raw = data_arr[:N - n_val - n_test]
    test_raw  = data_arr[N - n_test:]

    h24 = np.arange(len(train_raw)) % 24
    nm = np.array([train_raw[h24 == h].mean(axis=0) for h in range(24)])
    ns = np.maximum(np.array([train_raw[h24 == h].std(axis=0) for h in range(24)]), 1e-6)

    def deseason(arr, offset=0):
        o = arr.copy()
        for i in range(len(arr)):
            h = (offset + i) % 24
            o[i] = (arr[i] - nm[h]) / ns[h]
        return o

    scaler = StandardScaler()
    train_sc = scaler.fit_transform(deseason(train_raw, offset=0))
    test_sc = scaler.transform(deseason(test_raw, offset=N - n_test))
    X_tr, y_tr = make_win(train_sc, ALL_H, window=c.window, target_col=c.target_col)
    X_te, y_te = make_win(test_sc, ALL_H, window=c.window, target_col=c.target_col)

    ei = [ALL_H.index(h) for h in EVAL_H]
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}  FSDH range: 1..{FSDH_MAX}h")

    # ── 3. BASELINES ──
    print("\n[3/4] Training baselines...")

    esn_models = {
        f"ESN-{n}": ESNBaseline(nodes=n, seed=c.seed) for n in c.esn_nodes
    }

    preds_full = {}
    preds_eval = {}

    print("  Persistence...")
    pers = PersistenceBaseline(target_col_idx=c.target_col)
    pers.fit(X_tr, y_tr)
    preds_full["Persistence"] = pers.predict(X_te)
    preds_eval["Persistence"] = preds_full["Persistence"][:, ei]

    arima_name = f"ARIMA{c.arima_order}"
    print(f"  {arima_name}...")
    arima = ARIMABaseline(order=c.arima_order)
    arima.fit(X_tr, y_tr[:, ei])
    preds_arima_eval = arima.predict(X_te)
    preds_eval[arima_name] = preds_arima_eval

    for name, m in esn_models.items():
        print(f"  {name}...")
        m.fit(X_tr, y_tr)
        preds_full[name] = m.predict(X_te)
        preds_eval[name] = preds_full[name][:, ei]

    # ── 4. EVALUATION TABLE ──
    print("\n[4/4] Evaluation\n")
    for h_idx, h_name in enumerate([f"{h}h" for h in EVAL_H]):
        print(f"  Horizon {h_name}")
        print(f"  {'Model':<18} {'RMSE':>8} {'MAE':>8} {'Skill%':>8}")
        print(f"  {'-' * 46}")
        gt = y_te[:, ei[h_idx]]
        for name, preds in preds_eval.items():
            p = preds[:, h_idx]
            r = rmse(gt, p); m_v = mae(gt, p)
            s = skill_score(gt, p, preds_eval["Persistence"][:, h_idx])
            mk = " *" if "5000" in name else ""
            print(f"  {name:<18} {r:>8.4f} {m_v:>8.4f} {s * 100:>7.1f}%{mk}")
        print()

    # ── FSDH CURVE ──
    print(f"  FSDH curve (1..{FSDH_MAX}h):")
    y_persist_all = preds_full["Persistence"]

    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    y_arima_all = np.zeros((X_te.shape[0], FSDH_MAX))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_fsdh = _ARIMA(y_tr[:, 0], order=c.arima_order).fit()
        fc = arima_fsdh.forecast(steps=X_te.shape[0] + FSDH_MAX - 1)
        for h in range(FSDH_MAX):
            y_arima_all[:, h] = fc[h:h + X_te.shape[0]]
    except Exception as e:
        print(f"  [ARIMA FSDH] failed: {e}")

    curves = {"Persistence": np.array([rmse(y_te[:, h], y_persist_all[:, h]) for h in range(FSDH_MAX)])}
    for name, preds in preds_full.items():
        curves[name] = np.array([rmse(y_te[:, h], preds[:, h]) for h in range(FSDH_MAX)])
    curves[arima_name] = np.array([rmse(y_te[:, h], y_arima_all[:, h]) for h in range(FSDH_MAX)])

    p_curve = curves["Persistence"]

    print(f"\n  {'Model':<18} {'FSDH':>6}   {'VPT':>6} (hours)")
    print(f"  {'-' * 48}")
    model_preds_all = {**preds_full, arima_name: y_arima_all}
    for name, preds in model_preds_all.items():
        c = curves[name]; fsdh_v = 0
        for h in range(FSDH_MAX):
            if c[h] < p_curve[h]:
                fsdh_v = h + 1
            else:
                break
        vpt_v = compute_vpt_curve(y_te, preds, threshold=0.5)
        print(f"  {name:<18} {fsdh_v:>4}h  {vpt_v:>4}h")

    # ── FSDH FIGURE ──
    try:
        import matplotlib.pyplot as plt
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"FSDH Curve — RMSE vs Horizon (1–{FSDH_MAX}h)", fontsize=13, color="white")
        palette = ["#f87171", "#fbbf24", "#38bdf8", "#4ade80", "#a78bfa"]
        markers = ["s", "^", "o", "D", "v"]
        plot_names = list(curves.keys())
        for i, name in enumerate(plot_names):
            c = curves[name]
            ax.plot(ALL_H, c, f'{markers[i % len(markers)]}-', color=palette[i % len(palette)],
                    label=name, linewidth=2, markersize=4)
            if name != "Persistence":
                fv = 0
                for h in range(FSDH_MAX):
                    if c[h] < p_curve[h]:
                        fv = h + 1
                    else:
                        break
                if fv > 0:
                    ax.axvline(fv, color=palette[i % len(palette)], linestyle='--', alpha=0.4)
                    ax.annotate(f'FSDH={fv}h', xy=(fv, c[fv - 1]),
                                fontsize=9, color=palette[i % len(palette)], fontweight='bold')
                preds = model_preds_all.get(name)
                if preds is not None:
                    vv = compute_vpt_curve(y_te, preds, threshold=0.5)
                    if vv > 0:
                        ax.annotate(f'VPT={vv}h', xy=(vv, c[vv - 1]),
                                    fontsize=8, color=palette[i % len(palette)], alpha=0.7,
                                    va='top')
        ax.set_xlabel("Forecast horizon (hours)", color="#94a3b8")
        ax.set_ylabel("RMSE", color="#94a3b8")
        ax.legend(fontsize=10); ax.grid(True, alpha=0.2)
        ax.set_facecolor("#111418"); fig.patch.set_facecolor("#0b0d11")
        plt.tight_layout()
        fig.savefig("./qrc_figures/fig6_fsdh_baselines.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [Figure] Saved -> ./qrc_figures/fig6_fsdh_baselines.png")
    except Exception as e:
        print(f"  [Figure] failed: {e}")

    print("\n" + "=" * 70)
    print("  V Baselines complete. Same 13-feature ISD pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main(BaselineConfig(
        fsdh_max=48,
        esn_nodes=[500, 5000],
        eval_h=[1, 6],
    ))
