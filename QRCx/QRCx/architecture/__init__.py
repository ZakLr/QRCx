__all__ = ["ResidualQRC", "DirectQRC", "ParallelQRC"]

_imports = {
    "ResidualQRC": ".residual",
    "DirectQRC": ".direct",
    "ParallelQRC": ".parallel",
}

def __getattr__(name):
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return __all__
