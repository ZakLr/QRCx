from typing import Optional

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data.loader import load_isd_range, validate_dataframe
from .data.preprocessor import preprocess
from .data.lorenz import generate_lorenz63


class QRCPipeline:
    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        encoder=None,
        reservoir=None,
        architecture: str = "residual",
        readout: str = "krr",
        metrics: Optional[list[str]] = None,
        baselines: Optional[list[str]] = None,
        horizons: Optional[list[int]] = None,
    ):
        self.config = config or ExperimentConfig()
        self.encoder = encoder
        self.reservoir = reservoir
        self.architecture = architecture
        self.readout = readout
        self.metrics = metrics or ["rmse", "mae", "skill", "fsdh"]
        self.baselines = baselines or ["persistence", "arima", "esn500", "esn5000"]
        self.horizons = horizons or [1, 6]
        self.data: Optional[dict] = None
        self.results: Optional["ExperimentResults"] = None

    def load_data(
        self, start_year: Optional[int] = None, end_year: Optional[int] = None
    ) -> pd.DataFrame:
        sy = start_year or self.config.train_years[0]
        ey = end_year or self.config.test_year
        df = load_isd_range(sy, ey)
        validate_dataframe(df)
        return df

    def generate_synthetic(self, n_steps: int = 5000) -> np.ndarray:
        return generate_lorenz63(n_steps=n_steps, seed=self.config.seed)

    def preprocess(self, df: pd.DataFrame) -> dict:
        cfg = self.config
        self.data = preprocess(
            df,
            target_col=cfg.target_col,
            train_years=cfg.train_years,
            val_year=cfg.val_year,
            test_year=cfg.test_year,
            W=24,
            horizons=cfg.horizons,
        )
        return self.data

    def fit_evaluate(self, df: pd.DataFrame, target_col: str = "T_db") -> "ExperimentResults":
        from .experiment.runner import Experiment, ExperimentResults
        from .experiment.benchmark import benchmark_gate

        self.config.target_col = target_col
        data = preprocess(
            df,
            target_col=target_col,
            train_years=self.config.train_years,
            val_year=self.config.val_year,
            test_year=self.config.test_year,
            W=24,
            horizons=self.horizons,
        )
        self.data = data

        if self.reservoir is not None:
            bench = benchmark_gate(self.reservoir, threshold=0.5)
            if not bench["passed"]:
                print(f"Benchmark: latency {bench['latency']:.3f}s")

        from .architecture import DirectQRC, ResidualQRC, ParallelQRC
        from .metrics.forecast import rmse, mae, nrmse, skill_score, vpt
        from .metrics.fsdh import fsdh, compute_fsdh_curve

        target_col_idx = data["target_col_idx"]
        model_results = {}
        all_predictions = {}
        all_fsdh = {}
        all_baselines = {}

        model_name = self.architecture

        if self.reservoir is not None and model_name == "residual":
            from .readout import get_readout
            arch = ResidualQRC(self.reservoir, get_readout(self.readout), target_col_idx=target_col_idx)
            arch.fit(data["X_train"], data["y_train"], data["X_val"], data["y_val"])
            all_preds = arch.predict(data["X_test"])
            if all_preds.ndim == 1:
                all_preds = all_preds[:, np.newaxis]

            model_preds = {}
            for h_idx, h in enumerate(self.horizons):
                y_te = data["y_test"][:, h_idx]
                pred_t = all_preds[:, h_idx]
                y_persist = data["X_test"][:, -1, target_col_idx]
                model_preds[h] = pred_t
                row = {"horizon": h}
                for metric_name in self.metrics:
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
            y_model_mh = np.column_stack([model_preds[h] for h in self.horizons])
            y_persist_mh = np.column_stack([data["X_test"][:, -1, target_col_idx] for _ in self.horizons])
            all_fsdh[model_name] = compute_fsdh_curve(y_true_mh, y_model_mh, y_persist_mh)

        elif self.reservoir is not None:
            if model_name == "direct":
                arch = DirectQRC(n_qubits=self.reservoir.n_qubits, trotter_steps=self.reservoir.trotter_steps, n_layers_enc=self.config.n_layers, seed=self.config.seed)
            elif model_name == "parallel":
                arch = ParallelQRC(n_qubits=self.reservoir.n_qubits, trotter_steps=self.reservoir.trotter_steps, n_layers_enc=self.config.n_layers, seed=self.config.seed)
            else:
                raise ValueError(f"Unknown architecture: {model_name}")

            F_train = arch.run(data["X_train"])
            F_val = arch.run(data["X_val"])
            F_test = arch.run(data["X_test"])

            from .readout import get_readout
            readout = get_readout(self.readout)

            model_preds = {}
            for h_idx, h in enumerate(self.horizons):
                y_t = data["y_train"][:, h_idx]
                y_v = data["y_val"][:, h_idx]
                y_te = data["y_test"][:, h_idx]
                readout.fit(F_train, y_t, F_val, y_v)
                pred_t = readout.predict(F_test)
                y_persist = data["X_test"][:, -1, target_col_idx]
                model_preds[h] = pred_t
                row = {"horizon": h}
                for metric_name in self.metrics:
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
            y_model_mh = np.column_stack([model_preds[h] for h in self.horizons])
            y_persist_mh = np.column_stack([data["X_test"][:, -1, target_col_idx] for _ in self.horizons])
            all_fsdh[model_name] = compute_fsdh_curve(y_true_mh, y_model_mh, y_persist_mh)
        else:
            from .experiment.runner import Experiment as Exp
            exp = Exp(self.config)
            results = exp.run(df)
            self.results = results
            return results

        for baseline_name in self.baselines:
            try:
                bl_preds = self._run_baseline(baseline_name, data)
                all_baselines[baseline_name] = {"predictions": bl_preds}
                bl_metrics = {}
                for h_idx, h in enumerate(self.horizons):
                    y_te = data["y_test"][:, h_idx]
                    bl_metrics[h] = {"rmse": rmse(y_te, bl_preds[h])}
                model_results[baseline_name] = bl_metrics
            except Exception as e:
                print(f"Baseline {baseline_name} failed: {e}")

        self.results = ExperimentResults(
            metrics=model_results,
            fsdh=all_fsdh,
            predictions=all_predictions,
            config=self.config.__dict__,
            baselines=all_baselines,
        )
        return self.results

    def _run_baseline(self, name: str, data: dict) -> dict:
        y_test = data["y_test"][:, 0]
        target_col_idx = data["target_col_idx"]
        if name == "persistence":
            from .baselines import persistence
            return persistence.forecast(data["X_test"], y_test, horizons=self.horizons, target_col_idx=target_col_idx)
        elif name == "arima":
            from .baselines import arima
            return arima.forecast(
                data["train_seq"], data["test_seq"], target_col_idx, self.horizons,
                valid_idx=data["test_valid_idx"],
            )
        elif "esn" in name:
            from .baselines import esn
            size = int(name.replace("esn", ""))
            return esn.forecast(
                data["train_seq"], data["test_seq"], target_col_idx, self.horizons,
                reservoir_size=size, val_seq=data["val_seq"],
                seed=self.config.seed, fast_mode=self.config.fast_mode,
                valid_idx=data["test_valid_idx"],
            )
        raise ValueError(f"Unknown baseline: {name}")

    def run(self) -> "ExperimentResults":
        from .experiment.runner import Experiment, ExperimentResults
        if self.data is None:
            raise ValueError("Call preprocess() or fit_evaluate() before run()")
        experiment = Experiment(self.config)
        self.results = experiment.run(
            self.data["X_train"],
            self.data["y_train"],
            self.data["X_val"],
            self.data["y_val"],
            self.data["X_test"],
            self.data["y_test"],
        )
        return self.results

    def summary(self) -> pd.DataFrame:
        if self.results is None:
            raise ValueError("Call run() or fit_evaluate() before summary()")
        return self.results.metrics
