from typing import Optional

from ..config import ExperimentConfig
from ..experiment.runner import Experiment


def run_ablation(
    config: ExperimentConfig,
    df,
    sweeps: Optional[dict] = None,
) -> dict:
    if sweeps is None:
        sweeps = {
            "encoder": ["zz_feature_map"],
            "readout": ["krr", "ridge"],
            "trotter_steps": [5, 10],
            "n_qubits": [4, 6, 8],
        }

    results = {}

    for param_name, values in sweeps.items():
        results[param_name] = {}
        for val in values:
            label = f"{param_name}={val}"
            print(f"\n=== Ablation: {label} ===")

            cfg = ExperimentConfig(
                n_qubits=config.n_qubits,
                trotter_steps=config.trotter_steps,
                readout=config.readout,
                encoder=config.encoder,
                architecture=config.architecture,
                seed=config.seed,
                train_years=config.train_years,
                val_year=config.val_year,
                test_year=config.test_year,
            )
            if param_name == "encoder":
                cfg.encoder = val
            elif param_name == "readout":
                cfg.readout = val
            elif param_name == "trotter_steps":
                cfg.trotter_steps = val
            elif param_name == "n_qubits":
                cfg.n_qubits = val
            elif param_name == "architecture":
                cfg.architecture = val

            try:
                exp = Experiment(cfg)
                res = exp.run(df)
                metrics = {}
                for model_name, horizons in res.metrics.items():
                    if not isinstance(horizons, dict):
                        continue
                    for horizon, metric_dict in horizons.items():
                        if not isinstance(metric_dict, dict):
                            continue
                        for metric_name, metric_value in metric_dict.items():
                            key = f"{metric_name}_{horizon}h_{model_name}"
                            metrics[key] = metric_value
                results[param_name][label] = metrics
            except Exception as e:
                print(f"  FAILED: {e}")
                results[param_name][label] = {"error": str(e)}

    return results
