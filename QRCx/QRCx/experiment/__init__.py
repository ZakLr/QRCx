__all__ = [
    "Experiment", "ExperimentResults",
    "timing_benchmark",
    "run_ablation",
    "plot_architecture", "plot_esp_convergence", "plot_ipc_and_vpt",
    "plot_noise_sweep", "plot_results_table", "plot_fsdh_bar",
    "print_summary",
    "apply_style", "PALETTE", "label_panel",
]

_imports = {
    "Experiment": ".runner",
    "ExperimentResults": ".runner",
    "timing_benchmark": ".benchmark",
    "run_ablation": ".ablation",
    "plot_architecture": ".figures",
    "plot_esp_convergence": ".figures",
    "plot_ipc_and_vpt": ".figures",
    "plot_noise_sweep": ".figures",
    "plot_results_table": ".figures",
    "plot_fsdh_bar": ".figures",
    "print_summary": ".summary",
    "apply_style": ".figstyle",
    "PALETTE": ".figstyle",
    "label_panel": ".figstyle",
}


def __getattr__(name):
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
