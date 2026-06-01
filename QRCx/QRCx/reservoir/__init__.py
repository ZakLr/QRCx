__all__ = ["BaseReservoir", "AtmosphericQRC", "AtmosphericQRCCudaQ", "ParallelReservoir", "verify_esp"]

_imports = {
    "BaseReservoir": ".base",
    "AtmosphericQRC": ".tfim",
    "AtmosphericQRCCudaQ": ".tfim_cudaq",
    "ParallelReservoir": ".parallel",
    "verify_esp": ".esp",
}


def __getattr__(name):
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
