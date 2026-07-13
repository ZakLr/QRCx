from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    encoder: str = "zz_feature_map"
    n_qubits: int = 12
    n_layers: int = 3
    trotter_steps: int = 10
    reservoir_type: str = "pennylane"
    architecture: str = "residual"
    readout: str = "krr"
    horizons: list[int] = field(default_factory=lambda: [1, 6])
    baselines: list[str] = field(default_factory=lambda: ["persistence", "arima", "esn500", "esn5000"])
    metrics: list[str] = field(default_factory=lambda: ["rmse", "mae", "skill", "fsdh", "vpt", "mc"])
    target_col: str = "T_db"
    train_years: tuple[int, int] = (2019, 2022)
    val_year: int = 2023
    test_year: int = 2024
    seed: int = 42
    cache_dir: str = "./cache"
    figures_dir: str = "./figures"
    results_path: str = "./results.json"
    noise_sweep: bool = False
    noise_p_values: list[float] = field(default_factory=lambda: [0.0, 1e-3, 5e-3, 1e-2, 5e-2])

    def to_yaml(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
