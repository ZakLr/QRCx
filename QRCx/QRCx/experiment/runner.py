from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import ExperimentConfig


@dataclass
class ExperimentResults:
    metrics: dict
    fsdh: dict
    predictions: dict
    config: dict
    baselines: dict

    def summary(self) -> None:
        from .summary import print_summary
        print_summary(self.metrics)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for model, horizons in self.metrics.items():
            for h, h_metrics in horizons.items():
                if isinstance(h_metrics, dict):
                    rows.append({"model": model, "horizon": h, **h_metrics})
        return pd.DataFrame(rows)

    def plot_all(self) -> None:
        from .figures import (
            plot_architecture, plot_esp_convergence, plot_ipc_and_vpt,
            plot_noise_sweep, plot_results_table, plot_fsdh_bar,
        )
        plot_architecture()
        plot_results_table(self.metrics)
        plot_fsdh_bar(self.fsdh)

    def to_json(self, path: str) -> None:
        import json
        payload = {
            "metrics": {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in self.metrics.items()},
            "fsdh": {str(k): v for k, v in self.fsdh.items()},
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in self.config.items()},
            "baselines": {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in self.baselines.items()},
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: Optional[ExperimentResults] = None

    def run(self, df: pd.DataFrame) -> ExperimentResults:
        from ..data.preprocessor import preprocess
        from ..experiment.benchmark import benchmark_gate
        from ..architecture import DirectQRC, ResidualQRC, ParallelQRC
        from ..metrics.forecast import rmse, mae, nrmse, skill_score, vpt
        from ..metrics.fsdh import fsdh, compute_fsdh_curve

        cfg = self.config
        data = preprocess(
            df,
            target_col=cfg.target_col,
            train_years=cfg.train_years,
            val_year=cfg.val_year,
            test_year=cfg.test_year,
            W=24,
            horizons=cfg.horizons,
        )

        target_col_idx = data["target_col_idx"]

        model_results = {}
        all_predictions = {}
        all_fsdh = {}
        all_baselines = {}

        for model_name in ["direct", "residual", "parallel"]:
            print(f"\n--- Running {model_name} QRC ---")

            if model_name == "residual":
                from ..reservoir.tfim import AtmosphericQRC
                reservoir = AtmosphericQRC(
                    n_qubits=cfg.n_qubits, n_layers=cfg.n_layers,
                    trotter_steps=cfg.trotter_steps, seed=cfg.seed,
                )
                from ..readout import get_readout
                arch = ResidualQRC(reservoir, get_readout(cfg.readout), target_col_idx=target_col_idx)

                arch.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
                all_preds = arch.predict(data["X_test"])
                if all_preds.ndim == 1:
                    all_preds = all_preds[:, np.newaxis]

                model_preds = {}
                for h_idx, h in enumerate(cfg.horizons):
                    pred_t = all_preds[:, h_idx]
                    model_preds[h] = pred_t

                    y_te = data["y_test"][:, target_col_idx] if data["y_test"].ndim == 1 else data["y_test"][:, h_idx]
                    y_persist = data["X_test"][:, -1, target_col_idx]

                    row = {"horizon": h}
                    for metric_name in cfg.metrics:
                        if metric_name == "rmse":
                            row["rmse"] = rmse(y_te, pred_t)
                        elif metric_name == "mae":
                            row["mae"] = mae(y_te, pred_t)
                        elif metric_name == "nrmse":
                            row["nrmse"] = nrmse(y_te, pred_t)
                        elif metric_name == "skill":
                            row["skill"] = skill_score(y_te, pred_t, y_persist)
                        elif metric_name == "fsdh":
                            row["fsdh"] = fsdh(y_te, pred_t, y_persist)
                        elif metric_name == "vpt":
                            row["vpt"] = vpt(y_te, pred_t)
                    model_results.setdefault(model_name, {})[h] = row

                all_predictions[model_name] = model_preds
                y_true_mh = data["y_test"]
                y_model_mh = np.column_stack([model_preds[h] for h in cfg.horizons])
                y_persist_mh = np.column_stack([data["X_test"][:, -1, target_col_idx] for _ in cfg.horizons])
                all_fsdh[model_name] = compute_fsdh_curve(y_true_mh, y_model_mh, y_persist_mh)

                if "mc" in cfg.metrics:
                    from ..metrics.reservoir import measure_memory_capacity
                    model_results.setdefault(model_name, {})["mc"] = measure_memory_capacity(reservoir)
                if "ipc" in cfg.metrics:
                    from ..metrics.reservoir import measure_ipc_24h
                    model_results.setdefault(model_name, {})["ipc"] = measure_ipc_24h(reservoir, data["X_test"])
                if cfg.noise_sweep:
                    from ..metrics.noise import depolarising_sweep
                    noise_res = depolarising_sweep(reservoir, data["X_train"], data["y_train"], data["X_val"], data["y_val"], cfg.noise_p_values)
                    model_results.setdefault(model_name, {})["noise"] = noise_res

                bench = benchmark_gate(reservoir, threshold=0.5)
                if not bench["passed"]:
                    print(f"  WARNING: latency {bench['latency']:.3f}s exceeds 0.5s threshold")
                continue

            if model_name == "direct":
                arch = DirectQRC(n_qubits=cfg.n_qubits, trotter_steps=cfg.trotter_steps, n_layers_enc=cfg.n_layers, seed=cfg.seed)
            elif model_name == "parallel":
                arch = ParallelQRC(n_qubits=cfg.n_qubits, trotter_steps=cfg.trotter_steps, n_layers_enc=cfg.n_layers, seed=cfg.seed)

            bench = benchmark_gate(arch.reservoir if hasattr(arch, "reservoir") else arch, threshold=0.5)
            if not bench["passed"]:
                print(f"  WARNING: latency {bench['latency']:.3f}s exceeds 0.5s threshold")

            cache_dir = Path(cfg.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            F_train = self._cache_features(data["X_train"], cache_dir / f"F_train_{model_name}.npy", arch)
            F_val = self._cache_features(data["X_val"], cache_dir / f"F_val_{model_name}.npy", arch)
            F_test = self._cache_features(data["X_test"], cache_dir / f"F_test_{model_name}.npy", arch)

            from ..readout import get_readout
            readout = get_readout(cfg.readout)

            model_metrics = {}
            model_preds = {}

            for h_idx, h in enumerate(cfg.horizons):
                y_t = data["y_train"][:, target_col_idx] if data["y_train"].ndim == 1 else data["y_train"][:, h_idx]
                y_v = data["y_val"][:, target_col_idx] if data["y_val"].ndim == 1 else data["y_val"][:, h_idx]
                y_te = data["y_test"][:, target_col_idx] if data["y_test"].ndim == 1 else data["y_test"][:, h_idx]

                readout.fit(F_train, y_t.reshape(-1, 1) if y_t.ndim == 1 else y_t, F_val, y_v.reshape(-1, 1) if y_v.ndim == 1 else y_v)
                pred_t = readout.predict(F_test)
                if pred_t.ndim > 1 and pred_t.shape[1] == 1:
                    pred_t = pred_t.ravel()

                model_preds[h] = pred_t

                y_persist = data["X_test"][:, -1, target_col_idx]
                row = {"horizon": h}
                for metric_name in cfg.metrics:
                    if metric_name == "rmse":
                        row["rmse"] = rmse(y_te, pred_t)
                    elif metric_name == "mae":
                        row["mae"] = mae(y_te, pred_t)
                    elif metric_name == "nrmse":
                        row["nrmse"] = nrmse(y_te, pred_t)
                    elif metric_name == "skill":
                        row["skill"] = skill_score(y_te, pred_t, y_persist)
                    elif metric_name == "fsdh":
                        row["fsdh"] = fsdh(y_te, pred_t, y_persist)
                model_metrics[h] = row

            model_results[model_name] = model_metrics
            all_predictions[model_name] = model_preds

            y_true_mh = data["y_test"]
            y_model_mh = np.column_stack([model_preds[h] for h in cfg.horizons])
            y_persist_mh = np.column_stack([data["X_test"][:, -1, target_col_idx] for _ in cfg.horizons])
            all_fsdh[model_name] = compute_fsdh_curve(y_true_mh, y_model_mh, y_persist_mh)

            reservoir_for_metrics = arch.reservoir if hasattr(arch, "reservoir") else arch
            if "mc" in cfg.metrics:
                from ..metrics.reservoir import measure_memory_capacity
                model_results.setdefault(model_name, {})["mc"] = measure_memory_capacity(reservoir_for_metrics)
            if "ipc" in cfg.metrics:
                from ..metrics.reservoir import measure_ipc_24h
                model_results.setdefault(model_name, {})["ipc"] = measure_ipc_24h(reservoir_for_metrics, data["X_test"])
            if cfg.noise_sweep:
                from ..metrics.noise import depolarising_sweep
                noise_res = depolarising_sweep(reservoir_for_metrics, data["X_train"], data["y_train"], data["X_val"], data["y_val"], cfg.noise_p_values)
                model_results.setdefault(model_name, {})["noise"] = noise_res

        print("\n--- Running Baselines ---")
        for baseline_name in cfg.baselines:
            try:
                bl_preds = self._run_baseline(baseline_name, data)
                all_baselines[baseline_name] = {"predictions": bl_preds}
                bl_metrics = {}
                for h_idx, h in enumerate(cfg.horizons):
                    y_te = data["y_test"][:, target_col_idx] if data["y_test"].ndim == 1 else data["y_test"][:, h_idx]
                    bl_metrics[h] = {"rmse": rmse(y_te, bl_preds[h])}
                model_results[baseline_name] = bl_metrics
            except Exception as e:
                print(f"  Baseline {baseline_name} failed: {e}")

        self.results = ExperimentResults(
            metrics=model_results,
            fsdh=all_fsdh,
            predictions=all_predictions,
            config=cfg.__dict__,
            baselines=all_baselines,
        )
        self.results.to_json(cfg.results_path)
        return self.results

    def _cache_features(self, X: np.ndarray, path: Path, arch) -> np.ndarray:
        if path.exists():
            return np.load(path)
        F = arch.run(X)
        np.save(path, F)
        return F

    def _run_baseline(self, name: str, data: dict) -> dict:
        y_train = data["y_train"]
        y_test = data["y_test"]
        target_col_idx = data["target_col_idx"]
        if y_train.ndim == 2:
            y_train = y_train[:, 0]
        if y_test.ndim == 2:
            y_test = y_test[:, 0]

        if name == "persistence":
            from ..baselines import persistence
            return persistence.forecast(data["X_test"], y_test, horizons=self.config.horizons, target_col_idx=target_col_idx)
        elif name == "arima":
            from ..baselines import arima
            return arima.forecast(y_train, y_test, horizons=self.config.horizons)
        elif "esn" in name:
            from ..baselines import esn
            size = int(name.replace("esn", ""))
            return esn.forecast(data["X_train"], y_train, data["X_test"], y_test, reservoir_size=size, horizons=self.config.horizons)
        raise ValueError(f"Unknown baseline: {name}")
