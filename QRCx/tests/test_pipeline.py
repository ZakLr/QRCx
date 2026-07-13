import numpy as np
import pandas as pd
from QRCx.pipeline import QRCPipeline
from QRCx.encoding.zz_feature_map import ZZFeatureMap
from QRCx.reservoir.tfim import AtmosphericQRC


def test_pipeline_end_to_end():
    dates = pd.date_range("2019-01-01", "2024-01-15", freq="h")
    df = pd.DataFrame(
        index=dates,
        data={
            "T_db": np.random.randn(len(dates)),
            "T_dew": np.random.randn(len(dates)),
            "SLP": 1000 + np.random.randn(len(dates)),
            "WS": np.abs(np.random.randn(len(dates))),
            "WD": np.random.rand(len(dates)) * 360,
            "RH": np.random.rand(len(dates)) * 100,
        },
    )
    pipe = QRCPipeline(
        encoder=ZZFeatureMap(n_qubits=4, n_layers=2),
        reservoir=AtmosphericQRC(n_qubits=4, trotter_steps=3, seed=42),
        architecture="residual",
        readout="ridge",
        horizons=[1, 6],
    )
    results = pipe.fit_evaluate(df, target_col="T_db")
    assert hasattr(results, "metrics")
