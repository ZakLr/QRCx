"""QRCx — Quantum Reservoir Computing for Weather Forecasting."""

__version__ = "0.1.0"

# Lazy imports to avoid triggering heavy dependencies on simple imports
_IMPORTS = {
    # Top-level
    "QRCPipeline": ".pipeline",
    "Experiment": ".experiment.runner",
    "ExperimentResults": ".experiment.runner",
    "ExperimentConfig": ".config",
    
    # Encoding
    "ZZFeatureMap": ".encoding.zz_feature_map",
    "BaseEncoder": ".encoding.base",
    
    # Reservoir
    "AtmosphericQRC": ".reservoir.tfim",
    "AtmosphericQRCCudaQ": ".reservoir.tfim_cudaq",
    "ParallelReservoir": ".reservoir.parallel",
    "BaseReservoir": ".reservoir.base",
    "verify_esp": ".reservoir.esp",
    "SequentialDissipativeQRC": ".reservoir.sequential",
    "verify_esp_sequential": ".reservoir.esp_sequential",
    
    # Architecture
    "DirectQRC": ".architecture.direct",
    "ResidualQRC": ".architecture.residual",
    "ParallelQRC": ".architecture.parallel",
    
    # Readout
    "KRRReadout": ".readout.krr",
    "RidgeReadout": ".readout.ridge",
    "BaseReadout": ".readout.base",
    "extract_correlators": ".readout.correlators",
    "extract_correlators_dm": ".readout.correlators",
    
    # Metrics
    "rmse": ".metrics.forecast",
    "mae": ".metrics.forecast",
    "nrmse": ".metrics.forecast",
    "skill_score": ".metrics.forecast",
    "vpt": ".metrics.forecast",
    "compute_vpt_curve": ".metrics.forecast",
    "fsdh": ".metrics.fsdh",
    "compute_fsdh_curve": ".metrics.fsdh",
    
    # Data
    "load_isd_range": ".data.loader",
    "preprocess": ".data.preprocessor",
    "generate_lorenz63": ".data.lorenz",
    "generate_narma10": ".data.narma",
    "DataSplit": ".data.splits",
    
    # Reservoir metrics
    "measure_memory_capacity": ".metrics.reservoir",
    "measure_ipc_24h": ".metrics.reservoir",

    # Experiment
    "timing_benchmark": ".experiment.benchmark",
    "run_ablation": ".experiment.ablation",
    "print_summary": ".experiment.summary",
}


def __getattr__(name: str):
    if name in _IMPORTS:
        module_path = _IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_IMPORTS.keys())