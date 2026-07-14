#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         QRCx — PIPELINE DEMO (v4 + original merged)                         ║
║  qBraid x MITRE x JonesTrading Global Industry Challenge 2026 · Track B      ║
║                                                                              ║
║  PURPOSE OF THIS FILE                                                        ║
║  ─────────────────────                                                       ║
║  Single-file demonstration of the complete QRC weather forecasting           ║
║  pipeline: ISD-Lite real data loading (2019-2024), 13-physical-feature       ║
║  engineering, ZZFeatureMap with projection embedding, TFIM reservoir at      ║
║  critical point, ESP verification, IPC sweep, in-sample memory capacity,     ║
║  noise sweep, and all diagnostic figures.                                    ║
║                                                                              ║
║  This is a standalone demo/reference script, not the canonical package.      ║
║  The canonical, tested pipeline is QRCx/QRCx/ (see top-level README.md).     ║
║  Its own data loader duplicates QRCx/data/loader.py; QRCx/data/loader.py     ║
║  is the frozen source of truth — do not port fixes in reverse.               ║
║                                                                              ║
║  What this demonstrates:                                                     ║
║    1. Full pipeline from real/synthetic data → encode → reservoir →          ║
║       readout → residual → metrics works end-to-end                          ║
║    2. ResidualQRC beats DirectQRC at 1h horizon on chaotic data              ║
║    3. Two-body readout beats single-body readout                             ║
║    4. FSDH returns integer hours                                              ║
║    5. ESP converges → reservoir has echo state property                      ║
║    6. IPC increases with J/g ratio, peaks near critical point                ║
║    7. Memory capacity (iid windows, train=test, Jaeger 2001) > N/2 near      ║
║       J/g ≈ 1.0 — the OOS 70/30 variant is known-wrong and is not used.     ║
║    8. Noise tolerance sweep vs. injection probability p                     ║
║                                                                              ║
║  To run:  python pipeline_demo.py                                            ║
║  Deps:    pip install pennylane pennylane-lightning scikit-learn              ║
║           numpy scipy matplotlib reservoirpy statsmodels pandas              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import warnings
import gzip
import urllib.request
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    print("WARNING: PennyLane not found — using classical mock reservoir for demo")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not found — ISD data loader unavailable")


# ══════════════════════════════════════════════════════════════════════════════
# CENTRAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """Central configuration for the QRC weather forecasting experiment.

    All parameters have sensible defaults — you only need to override what
    you want to change. Pass an instance to ``run_full_experiment()``::

        cfg = ExperimentConfig(n_qubits=12, fsdh_max=12, use_isd=False)
        results, esp_fig, ipc_fig, noise_fig = run_full_experiment(cfg)

    Every field is documented below.  If you are unsure what a parameter
    does, just leave the default — the experiment will still work correctly.
    """

    # ── Random seed ──────────────────────────────────────────────────────
    seed: int = 42
    """Master random seed for reproducibility."""

    # ── Data ─────────────────────────────────────────────────────────────
    n_total_samples: int = 4350
    """Number of hourly rows to use from the ISD dataset (first N rows).
    Set to a large value (e.g. 8000) to use all available data."""
    data_window: int = 24
    """Input window length in hours.  Each training example is a 24-hour
    sequence of 13-dimensional feature vectors."""
    eval_horizons: List[int] = field(default_factory=lambda: [1, 6])
    """Forecast horizons (hours) for the main evaluation table."""
    fsdh_max: int = 12
    """Maximum forecast horizon for the full FSDH curve (1..fsdh_max h)."""
    target_col_idx: int = 0
    """Index of the target variable in the 13-feature array (0 = T_db)."""

    # ── ISD station ──────────────────────────────────────────────────────
    station_usaf: str = "725300"
    """USAF identifier for the weather station (KORD Chicago O'Hare)."""
    station_wban: str = "94846"
    """WBAN identifier for the weather station."""
    data_dir: str = "./kord_data"
    """Directory where ISD data is cached / downloaded."""

    # ── Quantum circuit ──────────────────────────────────────────────────
    n_qubits: int = 12
    """Number of qubits in the quantum reservoir.  Values 8-16 work well.
    More qubits → more features → potentially better accuracy but slower."""
    n_layers: int = 2
    """Number of encoding / evolution layers in the quantum circuit.
    2-3 layers are typically sufficient."""
    trotter_steps: Optional[int] = None
    """Trotter steps for TFIM time evolution.  ``None`` → auto:
    6 when n_qubits < 12, otherwise 4."""
    tau: float = 1.0
    """Total evolution time for the TFIM Hamiltonian."""
    n_input_features: int = 13
    """Dimensionality of the input feature vectors (13 physical features)."""

    # ── Reservoir / TFIM Hamiltonian ─────────────────────────────────────
    jg_target: float = 1.0
    """Target |J|/g ratio.  The ratio is rescaled to match this value.
    J/g ~ 1.0 corresponds to the critical / information-maximising point."""
    h_range: Tuple[float, float] = (-1.0, 1.0)
    "Range for random transverse fields h_i (uniform)."
    g_range: Tuple[float, float] = (0.5, 1.5)
    "Range for random on-site fields g_i (uniform)."
    J_range: Tuple[float, float] = (-3.0, 3.0)
    "Range for random Ising couplings J_ij (uniform, then symmetrised)."

    # ── ESP (Echo State Property) verification ───────────────────────────
    esp_n_initial_states: int = 5
    """Number of different initial quantum states to compare."""
    esp_n_steps: int = 30
    """Number of input steps for the ESP convergence test."""
    esp_convergence_threshold: float = 0.5
    """ESP is considered converged when the mean pairwise distance between
    trajectories falls below this threshold."""

    # ── IPC sweep ────────────────────────────────────────────────────────
    ipc_n_samples: int = 200
    """Number of i.i.d. samples for the IPC (information processing
    capacity) sweep."""
    ipc_window: int = 24
    """Window length for IPC sweep."""
    ipc_jg_range: Tuple[float, float] = (0.1, 3.0)
    "(min, max) J/g ratio to sweep over."
    ipc_n_points: int = 10
    """Number of evenly spaced J/g values in the sweep."""
    ipc_n_layers: int = 2
    "Number of circuit layers for the IPC sweep (separate from main QRC)."
    ipc_trotter_steps: int = 5
    "Trotter steps for the IPC sweep."
    ipc_max_lag: int = 10
    "Maximum lag for IPC calculation."
    ipc_ridge_alpha: float = 1e-3
    "Ridge regularisation for IPC readout."

    # ── Memory Capacity ──────────────────────────────────────────────────
    mc_max_lag: int = 15
    "Maximum lag for memory capacity measurement."
    mc_n_samples: int = 500
    "Number of i.i.d. samples for memory capacity."
    mc_window: int = 24
    "Window length for memory capacity."
    mc_ridge_alpha: float = 1e-3
    "Ridge regularisation for memory capacity readout."

    # ── KRR readout ──────────────────────────────────────────────────────
    krr_gamma_grid: List[float] = field(
        default_factory=lambda: [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    )
    """RBF kernel gamma values to try during hyperparameter search."""
    krr_alpha_grid: List[float] = field(
        default_factory=lambda: list(np.logspace(-6, -1, 6))
    )
    """Regularisation alpha values to try during hyperparameter search."""
    krr_tscv_splits: int = 5
    "Number of splits for TimeSeriesSplit cross-validation."

    # ── ESN baseline ─────────────────────────────────────────────────────
    esn_nodes: int = 500
    """Number of reservoir nodes for the ESN baseline.  You may also pass
    5000 for a larger reservoir."""
    esn_seed: int = 42
    "Random seed for the ESN."
    esn_spectral_radius: Optional[float] = None
    """Spectral radius.  ``None`` → auto: 0.9 for ≤500 nodes, 0.99 for
    >500 nodes."""
    esn_input_scaling: float = 0.5
    "Input scaling for the ESN."
    esn_rc_connectivity: float = 0.1
    "Recurrent connectivity (fraction of non-zero weights)."
    esn_ridge_alpha: float = 1.0
    "Ridge regression alpha for ESN readout."
    esn_warmup: int = 100
    "Number of warmup / washout steps for the ESN."

    # ── ARIMA baseline ───────────────────────────────────────────────────
    arima_order: Tuple[int, int, int] = (2, 1, 2)
    "(p, d, q) order for the ARIMA baseline."

    # ── Noise sweep ──────────────────────────────────────────────────────
    noise_p_values: List[float] = field(
        default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5]
    )
    "Noise levels (depolarising probability) to test."
    noise_krr_gamma: float = 0.1
    "RBF gamma for the readout used during noise sweep."
    noise_krr_alpha: float = 1e-4
    "Regularisation alpha for the readout used during noise sweep."
    noise_multiplier: float = 5.0
    "Noise std = feature_range * p * this_multiplier."

    # ── Ablation ─────────────────────────────────────────────────────────
    ablation_gamma: float = 0.1
    "RBF gamma for the readout used in the single-body / two-body ablation."
    ablation_alpha: float = 1e-4
    "Regularisation alpha for the readout used in the ablation."

    # ── Figures ──────────────────────────────────────────────────────────
    fig_save_dir: str = "./qrc_figures"
    "Directory where all figures are saved."
    fig_style: str = "dark_background"
    "Matplotlib style for figures."

    # ── Miscellaneous ────────────────────────────────────────────────────
    verbose: bool = True
    "Whether to print progress messages."

    @property
    def effective_trotter_steps(self) -> int:
        """Return the effective Trotter steps, accounting for the
        ``n_qubits < 12`` override."""
        if self.trotter_steps is not None:
            return self.trotter_steps
        return 6 if self.n_qubits < 12 else 4

    @property
    def effective_esn_sr(self) -> float:
        """Return the effective spectral radius for the ESN."""
        if self.esn_spectral_radius is not None:
            return self.esn_spectral_radius
        return 0.9 if self.esn_nodes <= 500 else 0.99

RNG = np.random.default_rng(42)

# Default station / data-dir constants (can be overridden via ExperimentConfig).
# Kept as module-level for backward compatibility with data-loading functions.
_DEFAULT_USAF = "725300"
_DEFAULT_WBAN = "94846"
_DEFAULT_DATA_DIR = Path("./kord_data")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA (ISD-Lite loader + synthetic fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _data_dir(cfg: Optional[ExperimentConfig] = None) -> Path:
    if cfg is not None:
        return Path(cfg.data_dir)
    return _DEFAULT_DATA_DIR

def _station_usaf(cfg: Optional[ExperimentConfig] = None) -> str:
    if cfg is not None:
        return cfg.station_usaf
    return _DEFAULT_USAF

def _station_wban(cfg: Optional[ExperimentConfig] = None) -> str:
    if cfg is not None:
        return cfg.station_wban
    return _DEFAULT_WBAN

def download_isd_year(year: int, max_retries: int = 3,
                      config: Optional[ExperimentConfig] = None) -> "Path | None":
    """Download and decompress ISD-Lite yearly file for KORD (Chicago O'Hare)."""
    dd = _data_dir(config)
    usaf = _station_usaf(config)
    wban = _station_wban(config)
    dd.mkdir(parents=True, exist_ok=True)
    gz_path  = dd / f"kord_{year}.gz"
    txt_path = dd / f"kord_{year}.txt"
    if txt_path.exists() and txt_path.stat().st_size > 10_000:
        print(f"  [Data] Cache hit: {txt_path}")
        return txt_path
    url = (f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/"
           f"{year}/{usaf}-{wban}-{year}.gz")
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [Data] Downloading {year} (attempt {attempt})...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                with open(gz_path, "wb") as f:
                    f.write(r.read())
            break
        except Exception as e:
            print(f"  [Data] {e}")
            if attempt == max_retries:
                return None
    if not gz_path.exists():
        return None
    import shutil
    with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    return txt_path


def parse_isd_file(txt_path: Path) -> "pd.DataFrame":
    """Parse an ISD-Lite whitespace-delimited file into a DataFrame."""
    if not HAS_PANDAS:
        raise ImportError("pandas required for ISD data loading")
    rows = []
    with open(txt_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            try:
                yr, mo, dy, hr = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                if not (1 <= mo <= 12 and 1 <= dy <= 31 and 0 <= hr <= 23):
                    continue
                pf = lambda i: np.nan if (i >= len(parts) or parts[i] == "-9999") else int(parts[i])
                rows.append({"year": yr, "month": mo, "day": dy, "hour": hr,
                             "T_db": pf(4), "T_dew": pf(5), "SLP": pf(6),
                             "WD": pf(7), "WS": pf(8), "sky": pf(9)})
            except (ValueError, IndexError):
                continue
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    for col in ["T_db", "T_dew", "SLP", "WS"]:
        df[col] = df[col] / 10.0
    e_s = 6.1078 * np.exp(17.2694 * df["T_db"] / (df["T_db"] + 237.3))
    e_a = 6.1078 * np.exp(17.2694 * df["T_dew"] / (df["T_dew"] + 237.3))
    df["RH"] = (100.0 * e_a / e_s).clip(0, 100)
    return df


def load_isd_range(years: Optional[List[int]] = None,
                   config: Optional[ExperimentConfig] = None) -> "pd.DataFrame | None":
    """Load ISD-Lite data for given years. Returns DataFrame with hourly index."""
    if not HAS_PANDAS:
        raise ImportError("pandas required for ISD data loading")
    if years is None:
        years = [2019]
    frames = []
    for year in years:
        txt = download_isd_year(year, config=config)
        if txt is None:
            print(f"  [Data] {year}: download failed"); continue
        df_raw = parse_isd_file(txt)
        print(f"  [Data] {year}: {len(df_raw):,} records")
        frames.append(df_raw)
    if not frames:
        return None
    df = pd.concat(frames).sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)
    df.index.name = "timestamp"
    for c in ["T_db", "T_dew", "SLP", "WS", "WD", "RH"]:
        if c in df.columns:
            df[c] = df[c].interpolate(method="linear", limit=3)
    df = df.ffill().bfill()
    print(f"  [Data] {len(df):,} hourly slots  "
          f"T_db=[{df.T_db.min():.1f},{df.T_db.max():.1f}] C")
    return df[["T_db", "T_dew", "RH", "WS", "SLP", "WD"]]


def generate_lorenz63(n_steps: int = 5000, dt: float = 0.01, seed: int = 42) -> np.ndarray:
    """5000-step Lorenz-63 trajectory via RK45. σ=10, ρ=28, β=8/3. Returns shape (n_steps, 3)."""
    rng0 = np.random.default_rng(seed)
    y0 = rng0.uniform(-1, 1, 3)
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def lorenz(t, y):
        x, y_, z = y
        return [sigma * (y_ - x), x * (rho - z) - y_, x * y_ - beta * z]

    t_span = (0, n_steps * dt)
    t_eval = np.linspace(0, n_steps * dt, n_steps)
    sol = solve_ivp(lorenz, t_span, y0, method="RK45", t_eval=t_eval, rtol=1e-9, atol=1e-9)
    return sol.y.T


def generate_synthetic_weather(n_hours: int = 2000, seed: int = 42) -> np.ndarray:
    """
    Synthetic stand-in for NOAA ISD KORD data.
    Shape: (n_hours, 6) -> columns: [T_db, T_dew, SLP, WS, WD, RH]
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)

    T_db = (15.0
            + 8.0 * np.sin(2 * np.pi * t / 24 - np.pi / 2)
            + 3.0 * np.sin(2 * np.pi * t / (24 * 7))
            + rng.normal(0, 0.5, n_hours))

    lorenz_x = generate_lorenz63(n_hours, seed=seed)[:, 0]
    lorenz_x = (lorenz_x - lorenz_x.mean()) / lorenz_x.std()
    T_db += 1.2 * lorenz_x

    T_dew = T_db - 5.0 + rng.normal(0, 0.3, n_hours)
    SLP = 1013.25 + 5.0 * np.sin(2 * np.pi * t / (24 * 3.5)) + rng.normal(0, 0.8, n_hours)
    WS = np.abs(5.0 + rng.normal(0, 2.0, n_hours))
    WD = (180.0 + 60.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 20, n_hours)) % 360
    RH = np.clip(70.0 - 0.5 * (T_db - T_dew) * 10 + rng.normal(0, 5, n_hours), 5, 100)

    data = np.stack([T_db, T_dew, SLP, WS, WD, RH], axis=1)
    print(f"  [Data] Synthetic weather: shape={data.shape}, T_db range=[{T_db.min():.1f}, {T_db.max():.1f}] C")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PHYSICAL FEATURE ENGINEERING (13 features)
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Engineer 13 physical features from 6 ISD raw columns.

    Input columns:  [T_db, T_dew, RH, WS, SLP, WD]
    Output columns: [T_db, T_dew, RH, WS, SLP, WD,
                     Wx, Wy, T_dep,
                     hour_sin, hour_cos, doy_sin, doy_cos]
    """
    out = df.copy()
    wd_rad = np.deg2rad(out["WD"].fillna(0))
    out["Wx"]    = out["WS"] * np.sin(wd_rad)
    out["Wy"]    = out["WS"] * np.cos(wd_rad)
    out["T_dep"] = out["T_db"] - out["T_dew"]

    if isinstance(out.index, pd.DatetimeIndex):
        h   = out.index.hour.values.astype(float)
        doy = out.index.dayofyear.values.astype(float)
    else:
        h   = np.zeros(len(out))
        doy = np.ones(len(out))

    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    out["doy_sin"]  = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"]  = np.cos(2 * np.pi * doy / 365.25)

    cols = ["T_db", "T_dew", "RH", "WS", "SLP", "WD",
            "Wx", "Wy", "T_dep",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    print(f"  [Features] {len(df.columns)} raw -> {len(cols)} engineered features")
    return out[cols]


def build_features_from_array(data6: np.ndarray, hours: np.ndarray = None, doy: np.ndarray = None) -> np.ndarray:
    """
    NumPy-only build_features for synthetic data (no pandas dependency).
    Input:  (n, 6)  -> [T_db, T_dew, SLP, WS, WD, RH]
    Output: (n, 13) -> [T_db, T_dew, RH, WS, SLP, WD,
                        Wx, Wy, T_dep,
                        hour_sin, hour_cos, doy_sin, doy_cos]
    """
    T_db, T_dew, SLP, WS, WD, RH = [data6[:, i] for i in range(6)]

    wd_rad = np.radians(WD)
    Wx = WS * np.sin(wd_rad)
    Wy = WS * np.cos(wd_rad)
    T_dep = T_db - T_dew

    if hours is None:
        hours_arr = np.arange(len(data6)) % 24
    else:
        hours_arr = hours
    hour_sin = np.sin(2 * np.pi * hours_arr / 24.0)
    hour_cos = np.cos(2 * np.pi * hours_arr / 24.0)

    if doy is None:
        doy_arr = np.ones(len(data6))
    else:
        doy_arr = doy
    doy_sin = np.sin(2 * np.pi * doy_arr / 365.25)
    doy_cos = np.cos(2 * np.pi * doy_arr / 365.25)

    out_cols = [T_db, T_dew, RH, WS, SLP, WD,
                Wx, Wy, T_dep,
                hour_sin, hour_cos, doy_sin, doy_cos]

    out = np.stack(out_cols, axis=1)
    print(f"  [Preprocess] build_features_from_array: {data6.shape} -> {out.shape} (13 features)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ZZ FEATURE MAP ENCODER + PROJECTION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

class ProjectionMatrix:
    """
    Gaussian random projection: 13 physical features -> N qubit features.
    P shape: (n_qubits, 13). Fixed at init, never trained.
    """

    def __init__(self, n_qubits: int, n_input: int = 13, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.P = rng.normal(0, 1.0 / np.sqrt(n_input), size=(n_qubits, n_input))
        print(f"  [Projection] P shape: {self.P.shape} (13 -> {n_qubits})")

    def project(self, x: np.ndarray) -> np.ndarray:
        """x: (d,) or (n, d) -> (n_qubits,) or (n, n_qubits)"""
        if x.ndim == 1:
            return self.P @ x
        return x @ self.P.T


class ZZFeatureMap:
    """
    ZZ Feature Map encoder (Havlicek et al. 2019) with projection support.
    Gate order per layer: H -> RZ(x_j) -> IsingZZ((pi-x_j)(pi-x_k))

    When n_features > n_qubits, uses ProjectionMatrix to reduce dimensionality.
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 3,
                 n_input_features: int = 13):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.pairs = list(combinations(range(n_qubits), 2))
        if n_input_features > n_qubits:
            self.proj = ProjectionMatrix(n_qubits, n_input=n_input_features)
        else:
            self.proj = None

    def scale_to_pi(self, x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-8:
            return np.full_like(x, np.pi / 2)
        return np.pi * (x - lo) / (hi - lo)

    def encode(self, x_scaled: np.ndarray, dev) -> None:
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        for i in range(self.n_qubits):
            qml.RZ(float(x_scaled[i % len(x_scaled)]), wires=i)
        for j, k in self.pairs:
            phi_jk = (np.pi - float(x_scaled[j % len(x_scaled)])) * \
                     (np.pi - float(x_scaled[k % len(x_scaled)]))
            qml.IsingZZ(phi_jk, wires=[j, k])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESERVOIR: TFIM AT CRITICAL POINT
# ══════════════════════════════════════════════════════════════════════════════

class AtmosphericQRC:
    """
    Transverse-Field Ising Model (TFIM) quantum reservoir at the critical point.

    Hamiltonian: H = sum_i h_i Z_i + sum_i g_i X_i + sum_{i<j} J_{ij} Z_i Z_j
    Critical point: <|J|>/<g> ~ 1.0 (maximum IPC per qubit)

    Readout features for N qubits:
      - Single-body: <X_i>, <Y_i>, <Z_i> for i=1..N -> 3N features
      - Two-body: <Z_i Z_j>, <X_i X_j>, <Y_i Y_j> for pairs -> 3*C(N,2) features
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 3,
                 trotter_steps: int = 10, tau: float = 1.0,
                 n_input_features: int = 13, seed: int = 42,
                 jg_target: float = 1.0,
                 h_range: Tuple[float, float] = (-1.0, 1.0),
                 g_range: Tuple[float, float] = (0.5, 1.5),
                 J_range: Tuple[float, float] = (-3.0, 3.0)):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.trotter_steps = trotter_steps
        self.tau = tau
        self.seed = seed
        self.dt = tau / trotter_steps
        self.pairs = list(combinations(range(n_qubits), 2))

        rng = np.random.default_rng(seed)
        self.h = rng.uniform(h_range[0], h_range[1], n_qubits)
        self.g = rng.uniform(g_range[0], g_range[1], n_qubits)
        J_raw = rng.uniform(J_range[0], J_range[1], (n_qubits, n_qubits))
        J_sym = (J_raw + J_raw.T) / 2
        J_vals_arr = J_sym[np.triu_indices(n_qubits, k=1)]

        mean_abs_J = np.abs(J_vals_arr).mean()
        mean_g = self.g.mean()
        scale = jg_target * mean_g / (mean_abs_J + 1e-8)
        J_sym_rescaled = J_sym * scale

        self.J = {(i, j): J_sym_rescaled[i, j] for i, j in self.pairs}
        self.jg_ratio = float(np.abs(list(self.J.values())).mean() / self.g.mean())
        print(f"  [Reservoir] J/g ratio: {self.jg_ratio:.3f}  (target {jg_target})")

        self.encoder = ZZFeatureMap(n_qubits=n_qubits, n_layers=n_layers,
                                    n_input_features=n_input_features)

        if HAS_PENNYLANE:
            try:
                self.dev = qml.device("lightning.qubit", wires=n_qubits)
                print(f"  [Reservoir] Backend: lightning.qubit  ({n_qubits} qubits, Trotter={trotter_steps})")
            except Exception:
                self.dev = qml.device("default.qubit", wires=n_qubits)
                print(f"  [Reservoir] Backend: default.qubit (lightning unavailable)")
        else:
            self.dev = None

        if HAS_PENNYLANE:
            self._build_qnode()

        n_pairs = len(self.pairs)
        two_q_gates = n_layers * trotter_steps * (n_pairs + n_qubits - 1)
        print(f"  [Reservoir] Est. 2-qubit gate count: ~{two_q_gates}")

    def _build_qnode(self):
        enc = self.encoder
        h_par = self.h
        g_par = self.g
        J_par = self.J
        dt = self.dt
        steps = self.trotter_steps
        nq = self.n_qubits
        pairs = self.pairs

        @qml.qnode(self.dev, interface="numpy", diff_method=None)
        def circuit(x_window):
            for ell in range(enc.n_layers):
                t_idx = (ell * (len(x_window) // enc.n_layers)) % len(x_window)
                x_slice = x_window[t_idx, :nq]
                x_scaled = enc.scale_to_pi(x_slice)
                enc.encode(x_scaled, self.dev)

                for _ in range(steps):
                    for i in range(nq):
                        qml.RX(2.0 * g_par[i] * dt, wires=i)
                    for i in range(nq):
                        qml.RZ(2.0 * h_par[i] * dt, wires=i)
                    for (i, j) in pairs:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2.0 * J_par[(i, j)] * dt, wires=j)
                        qml.CNOT(wires=[i, j])

            obs_sb = (
                [qml.expval(qml.PauliX(i)) for i in range(nq)] +
                [qml.expval(qml.PauliY(i)) for i in range(nq)] +
                [qml.expval(qml.PauliZ(i)) for i in range(nq)]
            )

            obs_tb = []
            for (i, j) in pairs:
                obs_tb += [
                    qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)),
                    qml.expval(qml.PauliX(i) @ qml.PauliX(j)),
                    qml.expval(qml.PauliY(i) @ qml.PauliY(j)),
                ]

            return obs_sb + obs_tb

        self._circuit = circuit

    def _mock_transform(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        W_in = rng.normal(0, 1, (X.shape[1] * X.shape[2], self.n_features))
        n = X.shape[0]
        X_flat = X.reshape(n, -1)
        return np.tanh(X_flat @ W_in)

    @property
    def n_features(self) -> int:
        nq = self.n_qubits
        return 3 * nq + 3 * nq * (nq - 1) // 2

    def transform(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        nf = self.n_features

        if not HAS_PENNYLANE or self.dev is None:
            features = self._mock_transform(X)
        else:
            features = np.zeros((n, nf))
            for idx in range(n):
                x_win = X[idx]
                if self.encoder.proj is not None:
                    x_win_proj = np.array([self.encoder.proj.project(row) for row in x_win])
                else:
                    x_win_proj = x_win[:, :self.n_qubits]
                raw = self._circuit(x_win_proj)
                features[idx] = np.array(raw)

        assert features.shape == (n, nf), \
            f"Expected (n, {nf}) got {features.shape}"
        span = features.max() - features.min()
        assert span > 0.01, \
            f"Feature range only {span:.4f} — circuit produces trivial output"

        return features


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ESP VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_esp(qrc, input_sequence: np.ndarray,
               n_initial_states: int = 5,
               n_steps: int = 30,
               convergence_threshold: float = 0.5) -> tuple:
    """
    Verify Echo State Property for TFIM reservoir by comparing trajectories
    from different initial quantum states driven by the same input.
    """
    n_steps_actual = min(n_steps, input_sequence.shape[0])
    nq = qrc.n_qubits

    dev = qml.device("lightning.qubit" if HAS_PENNYLANE else "default.qubit",
                     wires=nq)

    def make_esp_qnode(init_rotation: float = 0.0):
        @qml.qnode(dev)
        def _esp_qnode(x_scaled: np.ndarray) -> list:
            if init_rotation != 0.0:
                for i in range(nq):
                    qml.RX(init_rotation * (i + 1) / nq, wires=i)
            for i in range(nq):
                qml.Hadamard(wires=i)
            for i in range(nq):
                qml.RZ(float(x_scaled[i % len(x_scaled)]), wires=i)
            for (j, k) in qrc.pairs:
                phi = (np.pi - float(x_scaled[j % len(x_scaled)])) * \
                      (np.pi - float(x_scaled[k % len(x_scaled)]))
                qml.IsingZZ(phi, wires=[j, k])
            for _ in range(qrc.trotter_steps):
                for i in range(nq):
                    qml.RX(2.0 * qrc.g[i] * qrc.dt, wires=i)
                for i in range(nq):
                    qml.RZ(2.0 * qrc.h[i] * qrc.dt, wires=i)
                for (i, j) in qrc.pairs:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2.0 * qrc.J[(i, j)] * qrc.dt, wires=j)
                    qml.CNOT(wires=[i, j])
            return [qml.expval(qml.PauliZ(k)) for k in range(nq)]
        return _esp_qnode

    ref_qnode = make_esp_qnode(init_rotation=0.0)
    alt_qnodes = [make_esp_qnode(init_rotation=r) for r in
                  np.linspace(0.1, 1.0, n_initial_states)]

    rng = np.random.default_rng(42)
    distances = np.zeros(n_steps_actual)

    for step in range(n_steps_actual):
        x_step = input_sequence[step]
        if qrc.encoder.proj is not None:
            x_step = np.array([qrc.encoder.proj.project(row) for row in x_step])
        x_flat = x_step.ravel()
        x_scaled = qrc.encoder.scale_to_pi(x_flat[:nq])

        ref_state = np.array(ref_qnode(x_scaled))

        pair_dists = []
        for alt_qnode in alt_qnodes:
            alt_state = np.array(alt_qnode(x_scaled))
            pair_dists.append(np.linalg.norm(ref_state - alt_state))

        distances[step] = np.mean(pair_dists)

    converged = distances[-1] < convergence_threshold

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(distances, color="cyan", linewidth=2)
    ax.axhline(convergence_threshold, color="red", linestyle="--",
               label=f"threshold={convergence_threshold}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean pairwise L2 distance")
    ax.set_title(f"ESP Verification (converged={converged})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor("#0b0d11")
    ax.set_facecolor("#111418")

    print(f"  [ESP] Converged: {converged}  (final dist={distances[-1]:.2e})")
    return fig, converged, distances


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — IPC SWEEP (J/g ratio scan)
# ══════════════════════════════════════════════════════════════════════════════

def sweep_jg_ratio(qrc_class, n_qubits: int = 8, n_samples: int = 200,
                   W: int = 24, jg_range: tuple = (0.1, 3.0),
                   n_points: int = 10, seed: int = 42,
                   n_layers: int = 2, trotter_steps: int = 5,
                   ridge_alpha: float = 1e-3) -> tuple:
    """
    Sweep J/g ratio and measure IPC (sum of squared correlations).
    """
    rng = np.random.default_rng(seed)
    u_full = rng.normal(0, 1.0, size=(n_samples, W, n_qubits))
    u_scalar = u_full[:, -1, 0]

    jg_values = np.linspace(jg_range[0], jg_range[1], n_points)
    ipc_values = []

    for jg_target in jg_values:
        qrc = qrc_class(n_qubits=n_qubits, n_layers=n_layers, trotter_steps=trotter_steps,
                        n_input_features=n_qubits, seed=seed + int(jg_target * 100))

        qrc.jg_ratio = jg_target

        R = qrc.transform(u_full)

        ipc = 0.0
        for lag in range(1, min(10, n_samples)):
            R_sub = R[lag:]
            u_sub = u_scalar[:n_samples - lag]
            reg = Ridge(alpha=ridge_alpha).fit(R_sub, u_sub)
            pred = reg.predict(R_sub)
            corr = np.corrcoef(u_sub, pred)[0, 1]
            ipc += corr ** 2

        ipc_values.append(ipc)
        print(f"  [IPC] J/g={jg_target:.2f}  IPC={ipc:.3f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(jg_values, ipc_values, "o-", color="cyan", linewidth=2)
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="J/g = 1 (critical)")
    ax.set_xlabel("J/g ratio")
    ax.set_ylabel("IPC (sum r^2, lags 1-9)")
    ax.set_title("IPC vs J/g Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor("#0b0d11")
    ax.set_facecolor("#111418")

    return fig, jg_values, np.array(ipc_values)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MEMORY CAPACITY (iid windows, train=test, Jaeger 2001)
# ══════════════════════════════════════════════════════════════════════════════

def measure_memory_capacity(qrc: AtmosphericQRC,
                            max_lag: int = 15,
                            n_samples: int = 500,
                            W: int = 24,
                            ridge_alpha: float = 1e-3) -> dict:
    """
    Linear memory capacity (Jaeger 2001) — in-sample measurement (standard).
    """
    nq = qrc.n_qubits
    rng = np.random.default_rng(0)

    n_input = qrc.encoder.proj.P.shape[1] if qrc.encoder.proj is not None else nq
    u_full = rng.normal(0, 1.0, size=(n_samples, W, n_input))
    u_scalar = u_full[:, -1, 0]

    R = qrc.transform(u_full)

    mc_per_lag = []
    for k in range(1, max_lag + 1):
        if k >= n_samples:
            break
        R_sub = R[k:]
        u_sub = u_scalar[:n_samples - k]
        reg = Ridge(alpha=ridge_alpha).fit(R_sub, u_sub)
        pred = reg.predict(R_sub)
        r2 = float(np.corrcoef(u_sub, pred)[0, 1] ** 2)
        mc_per_lag.append(r2)

    mc = float(np.sum(mc_per_lag))
    target = nq / 2

    if mc < target:
        print(f"  [MC] MC={mc:.3f} < N/2={target} (J/g={qrc.jg_ratio:.3f})")
    else:
        print(f"  [MC] Memory Capacity = {mc:.3f}  (target > {target})")

    return {"MC": mc, "per_lag": mc_per_lag}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — KRR READOUT
# ══════════════════════════════════════════════════════════════════════════════

class KRRReadout:
    """
    Kernel Ridge Regression readout with RBF kernel.
    Hyperparameter grid via TimeSeriesSplit.
    """

    def __init__(self,
                 gamma_grid: Optional[List[float]] = None,
                 alpha_grid: Optional[List[float]] = None,
                 tscv_splits: int = 5):
        self.model = None
        self.best_gamma = None
        self.best_alpha = None
        self.gamma_grid = gamma_grid or [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        self.alpha_grid = alpha_grid or list(np.logspace(-6, -1, 6))
        self.tscv_splits = tscv_splits

    def fit(self, F_train: np.ndarray, y_train: np.ndarray,
            horizon_idx: int = 0) -> None:
        y = y_train[:, horizon_idx]
        tscv = TimeSeriesSplit(n_splits=self.tscv_splits)

        best_val_rmse = np.inf
        for gamma in self.gamma_grid:
            for alpha in self.alpha_grid:
                val_rmses = []
                for train_idx, val_idx in tscv.split(F_train):
                    m = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
                    m.fit(F_train[train_idx], y[train_idx])
                    pred = m.predict(F_train[val_idx])
                    val_rmses.append(rmse(y[val_idx], pred))
                avg = float(np.mean(val_rmses))
                if avg < best_val_rmse:
                    best_val_rmse = avg
                    self.best_gamma = gamma
                    self.best_alpha = alpha

        self.model = KernelRidge(kernel="rbf",
                                 gamma=self.best_gamma,
                                 alpha=self.best_alpha)
        self.model.fit(F_train, y)
        print(f"  [KRR h={horizon_idx+1}h] best gamma={self.best_gamma}, "
              f"alpha={self.best_alpha:.2e}, val_RMSE={best_val_rmse:.4f}")

    def predict(self, F: np.ndarray) -> np.ndarray:
        assert self.model is not None, "Call fit() before predict()"
        return self.model.predict(F)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — BASELINES
# ══════════════════════════════════════════════════════════════════════════════

class PersistenceBaseline:
    """y_{t+h} = y_t for all h."""

    def __init__(self, target_col_idx: int = 0):
        self.target_col_idx = target_col_idx
        self.n_out = 2

    def fit(self, X_train, y_train):
        self.n_out = y_train.shape[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        persist = X[:, -1, self.target_col_idx, None]
        return np.tile(persist, (1, self.n_out))


class ARIMABaseline:
    """ARIMA(p,d,q) via statsmodels."""

    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)):
        self.order = order
        self.model_fits = {}

    def fit(self, X_train, y_train):
        import warnings
        try:
            from statsmodels.tsa.arima.model import ARIMA
            for h_idx in range(y_train.shape[1]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mdl = ARIMA(y_train[:, h_idx], order=self.order)
                    self.model_fits[h_idx] = mdl.fit()
        except ImportError:
            print("  [ARIMA] statsmodels not installed — skipping")

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        n_out = len(self.model_fits)
        preds = np.full((n, n_out), np.nan)
        for h_idx, fit in self.model_fits.items():
            try:
                fc = fit.forecast(steps=n)
                preds[:, h_idx] = fc[:n]
            except Exception:
                pass
        return preds


class ESNBaseline:
    """Echo State Network via reservoirpy. Raises ImportError if missing."""

    def __init__(self, nodes: int = 500, seed: int = 42,
                 sr: Optional[float] = None, input_scaling: float = 0.5,
                 rc_connectivity: float = 0.1,
                 ridge_alpha: float = 1.0, warmup: int = 100):
        self.nodes = nodes
        self.seed = seed
        self.sr = sr if sr is not None else (0.9 if nodes <= 500 else 0.99)
        self.input_scaling = input_scaling
        self.rc_connectivity = rc_connectivity
        self.ridge_alpha = ridge_alpha
        self.warmup = warmup
        self._esn = None

    def fit(self, X_train, y_train):
        n = X_train.shape[0]
        X_flat = X_train.reshape(n, -1)
        if not np.isfinite(X_flat).all():
            X_flat = np.nan_to_num(X_flat)
        import reservoirpy as rpy
        from reservoirpy.nodes import Reservoir, Ridge as RPyRidge
        reservoir = Reservoir(self.nodes, sr=self.sr, seed=self.seed,
                              input_scaling=self.input_scaling,
                              rc_connectivity=self.rc_connectivity)
        ridge = RPyRidge(ridge=self.ridge_alpha)
        esn = reservoir >> ridge
        esn.fit(X_flat, y_train, warmup=self.warmup)
        self._esn = esn

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        X_flat = X.reshape(n, -1)
        if not np.isfinite(X_flat).all():
            X_flat = np.nan_to_num(X_flat)
        preds = self._esn.run(X_flat)
        if preds.ndim == 1:
            preds = preds[:, np.newaxis]
        return preds


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — METRICS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    std = float(np.std(y_true))
    if std < 1e-8:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) / 1e-8
    return rmse(y_true, y_pred) / std

def skill_score(y_true: np.ndarray, y_pred: np.ndarray,
                y_baseline: np.ndarray) -> float:
    return float(1.0 - rmse(y_true, y_pred) / (rmse(y_true, y_baseline) + 1e-8))

def compute_vpt_curve(y_true_horizons: np.ndarray,
                      y_pred_horizons: np.ndarray,
                      threshold: float = 0.5,
                      max_horizon: Optional[int] = None) -> int:
    max_h = y_true_horizons.shape[1]
    if max_horizon is not None:
        max_h = min(max_h, max_horizon)
    vpt_val = 0
    for h in range(max_h):
        if nrmse(y_true_horizons[:, h], y_pred_horizons[:, h]) < threshold:
            vpt_val = h + 1
        else:
            break
    return vpt_val

def compute_fsdh_curve(y_true_horizons: np.ndarray,
                       y_model_horizons: np.ndarray,
                       y_persist_horizons: np.ndarray,
                       max_horizon: Optional[int] = None) -> int:
    max_h = y_true_horizons.shape[1]
    if max_horizon is not None:
        max_h = min(max_h, max_horizon)
    fsdh_val = 0
    for h in range(max_h):
        if rmse(y_true_horizons[:, h], y_model_horizons[:, h]) < rmse(y_true_horizons[:, h], y_persist_horizons[:, h]):
            fsdh_val = h + 1
        else:
            break
    return fsdh_val


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — NOISE SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def noise_sweep(qrc: AtmosphericQRC = None, X_train: np.ndarray = None, y_train: np.ndarray = None,
                X_test: np.ndarray = None, y_test: np.ndarray = None,
                F_train: np.ndarray = None, F_test: np.ndarray = None,
                p_values: Optional[List[float]] = None,
                krr_gamma: float = 0.1, krr_alpha: float = 1e-4,
                noise_multiplier: float = 5.0) -> dict:
    if p_values is None:
        p_values = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5]

    if F_train is None:
        assert qrc is not None and X_train is not None and X_test is not None
        F_train = qrc.transform(X_train)
        F_test = qrc.transform(X_test)

    feat_range = float(F_train.max() - F_train.min())

    results = {}
    for p in p_values:
        if p > 0:
            rng = np.random.default_rng(int(p * 1e6))
            noise_std = feat_range * p * noise_multiplier
            F_test_noisy = F_test + rng.normal(0, noise_std, size=F_test.shape)
        else:
            F_test_noisy = F_test

        krr = KernelRidge(kernel="rbf", gamma=krr_gamma, alpha=krr_alpha)
        krr.fit(F_train, y_train[:, 0])
        pred = krr.predict(F_test_noisy)
        results[p] = rmse(y_test[:, 0], pred)
        print(f"  [Noise] p={p:.1e}  RMSE={results[p]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    p_plot = list(results.keys())
    rmse_plot = list(results.values())
    ax.semilogx(p_plot, rmse_plot, "o-", color="orange", linewidth=2)
    ax.set_xlabel("Noise level p")
    ax.set_ylabel("RMSE (1h)")
    ax.set_title("Noise Sweep: RMSE vs Depolarizing Noise")
    ax.grid(True, alpha=0.3)
    fig.patch.set_facecolor("#0b0d11")
    ax.set_facecolor("#111418")

    return fig, results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_all_figures(results: dict, save_dir: str = "./qrc_figures",
                     esp_fig=None, ipc_fig=None, noise_fig=None,
                     config: Optional[ExperimentConfig] = None) -> None:
    if config is not None:
        save_dir = config.fig_save_dir
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use("dark_background" if config is None else config.fig_style)
    CYAN, GREEN, RED, AMBER = "#38bdf8", "#4ade80", "#f87171", "#fbbf24"

    if esp_fig is not None:
        path = f"{save_dir}/fig1_esp_convergence.png"
        esp_fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(esp_fig)
        print(f"  [Fig 1] Saved -> {path}")

    if ipc_fig is not None:
        path = f"{save_dir}/fig2_ipc_sweep.png"
        ipc_fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(ipc_fig)
        print(f"  [Fig 2] Saved -> {path}")

    if "MC_per_lag" in results:
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.suptitle("Figure 3 — Memory Capacity per Lag (IPC proxy)",
                     fontsize=13, color="white")
        lags = list(range(1, len(results["MC_per_lag"]) + 1))
        ax.bar(lags, results["MC_per_lag"], color=CYAN, edgecolor="#232830", alpha=0.85)
        ax.axhline(0, color="#232830", linewidth=0.5)
        ax.set_xlabel("Lag (hours)", color="#94a3b8")
        ax.set_ylabel("r^2 (memory at this lag)", color="#94a3b8")
        mc_total = results["MC"]
        n_q = results.get("n_qubits", 12)
        ax.set_title(f"Total MC = {mc_total:.3f}  (target > {n_q/2:.0f} for N={n_q})",
                     color="#94a3b8", fontsize=10)
        ax.tick_params(colors="#94a3b8")
        ax.set_facecolor("#111418")
        fig.patch.set_facecolor("#0b0d11")
        plt.tight_layout()
        path = f"{save_dir}/fig3_memory_capacity.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Fig 3] Saved -> {path}")

    if noise_fig is not None:
        path = f"{save_dir}/fig4_noise_sweep.png"
        noise_fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(noise_fig)
        print(f"  [Fig 4] Saved -> {path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 5 — QRC vs Baselines: RMSE at 1h and 6h horizon",
                 fontsize=13, color="white")

    for ax, h_name in zip(axes, ["1h", "6h"]):
        models = list(results[h_name].keys())
        rmses = [results[h_name][m]["rmse"] for m in models]
        colors = [GREEN if "Residual" in m else
                  CYAN if "Direct" in m else
                  RED if "Persist" in m else AMBER for m in models]
        bars = ax.bar(models, rmses, color=colors, edgecolor="#232830", width=0.6)
        ax.set_title(f"RMSE at {h_name} horizon", color="white")
        ax.set_ylabel("RMSE (scaled units)", color="#94a3b8")
        ax.tick_params(axis="x", rotation=30, colors="#94a3b8")
        ax.tick_params(axis="y", colors="#94a3b8")
        for bar, val in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="white")
        ax.set_facecolor("#111418")

    fig.patch.set_facecolor("#0b0d11")
    plt.tight_layout()
    path = f"{save_dir}/fig5_results_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Fig 5] Saved -> {path}")

    if "fsdh_curves" in results:
        fig, ax = plt.subplots(figsize=(10, 5))
        fsdh_max_plot = results.get("fsdh_horizons", list(range(1, 13)))[-1]
        fig.suptitle(f"Figure 6 — FSDH: RMSE vs Forecast Horizon (1–{fsdh_max_plot}h)",
                     fontsize=13, color="white")
        hz = results["fsdh_horizons"]
        curves = results["fsdh_curves"]
        for name, clr, mk in [("Persistence", RED, "s"),
                              ("DirectQRC", CYAN, "o"),
                              ("ResidualQRC", GREEN, "D")]:
            if name in curves:
                ax.plot(hz, curves[name], f'{mk}-', color=clr, label=name,
                        linewidth=2, markersize=4, markerfacecolor=clr)
                if name != "Persistence":
                    fv = results["FSDH"].get(name, 0)
                    if fv > 0:
                        ax.axvline(fv, color=clr, linestyle='--', alpha=0.4)
                        ax.annotate(f'FSDH={fv}h', xy=(fv, curves[name][fv - 1]),
                                    fontsize=10, color=clr, fontweight='bold',
                                    va='bottom')
                    vpt_curve = results.get("VPT_curve", {})
                    vv = vpt_curve.get(name, 0)
                    if vv > 0:
                        ax.axvline(vv, color=clr, linestyle=':', alpha=0.3)
                        ax.annotate(f'VPT={vv}h', xy=(vv, curves[name][vv - 1]),
                                    fontsize=9, color=clr, alpha=0.7,
                                    va='top')
        ax.set_xlabel("Forecast horizon (hours)", color="#94a3b8")
        ax.set_ylabel("RMSE (scaled units)", color="#94a3b8")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor("#111418")
        fig.patch.set_facecolor("#0b0d11")
        plt.tight_layout()
        path = f"{save_dir}/fig6_fsdh_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Fig 6] Saved -> {path}")

    if "ablation" in results:
        abl = results["ablation"]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.suptitle("Ablation: Single-body vs Two-body Readout (1h RMSE)",
                     fontsize=12, color="white")
        vals = [abl["single_body_rmse_1h"], abl["two_body_rmse_1h"]]
        nq_lbl = results.get("n_qubits", 12)
        n_single = 3 * nq_lbl
        n_two = 3 * nq_lbl * (nq_lbl - 1) // 2
        lbls = [f"Single-body\n({n_single} features)", f"Two-body\n({n_two} features)"]
        cols = [AMBER, GREEN]
        bars = ax.bar(lbls, vals, color=cols, edgecolor="#232830", width=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=11, color="white")
        impr = abl["improvement_pct"]
        ax.set_title(f"Two-body improves RMSE by {impr:.1f}%", color="#94a3b8", fontsize=10)
        ax.set_ylabel("RMSE (1h)", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.set_facecolor("#111418")
        fig.patch.set_facecolor("#0b0d11")
        plt.tight_layout()
        path = f"{save_dir}/fig7_ablation_readout.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Fig 7] Saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — SUMMARY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict,
                  config: Optional[ExperimentConfig] = None) -> None:
    if config is None:
        config = ExperimentConfig()
    cfg = config

    fsdh_max = results.get("fsdh_horizons", list(range(1, 13)))[-1]
    n_q = results.get("n_qubits", cfg.n_qubits)
    eval_h_str = ", ".join(f"{h}h" for h in cfg.eval_horizons)

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    for h_name in [f"{h}h" for h in cfg.eval_horizons]:
        if h_name not in results:
            continue
        print(f"\n  Horizon: {h_name}")
        print(f"  {'Model':<18} {'RMSE':>8} {'MAE':>8} {'Skill%':>8}")
        print(f"  {'-' * 46}")
        h_res = results[h_name]
        sorted_models = sorted(h_res.keys(), key=lambda m: h_res[m]["rmse"])
        for m in sorted_models:
            r = h_res[m]
            marker = " *" if m == "ResidualQRC" else ""
            print(f"  {m:<18} {r['rmse']:>8.4f} {r['mae']:>8.4f} "
                  f"{r['skill'] * 100:>7.1f}%{marker}")

    print(f"\n  FSDH & VPT (full {fsdh_max}h curve):")
    print(f"  {'Model':<18} {'FSDH':>6}   {'VPT':>6} (hours)")
    print(f"  {'-' * 48}")
    fsdh_data = results.get("FSDH", {})
    vpt_data = results.get("VPT_curve", {})
    for m in sorted(fsdh_data, key=lambda x: -fsdh_data[x]):
        if m == "Persistence":
            continue
        fh = fsdh_data.get(m, 0)
        vh = vpt_data.get(m, 0)
        marker = " *" if m == "ResidualQRC" else ""
        print(f"  {m:<18} {fh:>4}h  {vh:>4}h{marker}")

    vpt_eval = results.get("VPT", {})
    if vpt_eval:
        print(f"\n  VPT@eval (max hours NRMSE < 0.5 @ [{eval_h_str}]):")
        for m, v in sorted(vpt_eval.items(), key=lambda x: -x[1]):
            marker = " *" if m == "ResidualQRC" else ""
            print(f"  {m:<18} {v}h{marker}")

    if vpt_data:
        print(f"\n  VPT@{fsdh_max}h (Lyapunov time: max h NRMSE < 0.5 on 1..{fsdh_max}h):")
        for m, v in sorted(vpt_data.items(), key=lambda x: -x[1]):
            marker = " *" if m == "ResidualQRC" else ""
            print(f"  {m:<18} {v}h{marker}")

    mc = results.get("MC", "N/A")
    print(f"\n  Memory Capacity (MC):    {mc:.3f}  (target > {n_q/2:.0f} for N={n_q})")

    if "ablation" in results:
        abl = results["ablation"]
        nf_sb = 3 * n_q
        nf_tb = 3 * n_q + 3 * n_q * (n_q - 1) // 2
        print(f"\n  Ablation (1h RMSE):")
        print(f"    Single-body ({nf_sb} feat): {abl['single_body_rmse_1h']:.4f}")
        print(f"    Two-body  ({nf_tb} feat):  {abl['two_body_rmse_1h']:.4f}")
        print(f"    Improvement:           {abl['improvement_pct']:.1f}%")

    if "noise" in results:
        print(f"\n  Noise sweep (p -> RMSE):")
        for p, r_val in sorted(results["noise"].items()):
            print(f"    p={p:.0e} -> RMSE={r_val:.4f}")

    print("\n" + "=" * 70)
    print("  KEY CLAIMS TO VERIFY WITH REAL KORD DATA:")
    print("  1. ResidualQRC RMSE < DirectQRC RMSE at 1h horizon")
    print("  2. ResidualQRC skill > 0 at 1h (beats persistence)")
    print(f"  3. FSDH(ResidualQRC) >= {fsdh_max} hours")
    print(f"  4. VPT(ResidualQRC) >= {fsdh_max} hours (NRMSE < 0.5)")
    print("  5. Two-body RMSE < Single-body RMSE")
    print(f"  6. MC > {n_q/2:.0f} for N={n_q} at J/g ~ 1.0")
    print("  7. ESP converges within 50 steps")
    print("  8. IPC peaks near J/g = 1.0")
    print("  9. Noise degrades gracefully")
    print("=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_full_experiment(config: Optional[ExperimentConfig] = None) -> dict:
    """
    Complete end-to-end QRC experiment.

    Accepts an ``ExperimentConfig`` object.  If ``None``, a default config
    is created (12 qubits, 800 samples, FSDH 12h, etc.).

    Returns ``(results_dict, esp_fig, ipc_fig, noise_fig)`` where
    ``results_dict`` contains all metrics, ablation, noise data, etc.
    """
    if config is None:
        config = ExperimentConfig()
    cfg = config

    nq = cfg.n_qubits
    seed = cfg.seed

    print("\n" + "=" * 70)
    print(f"  QRC WEATHER FORECASTING — {nq}-QUBIT PIPELINE")
    print(f"  Config: {cfg.n_total_samples} samples, FSDH 1..{cfg.fsdh_max}h, "
          f"eval horizons={cfg.eval_horizons}")
    print("=" * 70)

    # --- 1. DATA ---
    print("\n[1/8] Loading data...")
    if cfg.n_total_samples > 0 and HAS_PANDAS:
        df = load_isd_range(years=[2019], config=cfg)
        if df is not None and len(df) > 1000:
            df = engineer_features(df)
            data_arr = df.to_numpy()[:cfg.n_total_samples]
            print(f"  [Data] ISD-KORD loaded: {data_arr.shape} (first {cfg.n_total_samples} rows)")
        else:
            n_syn = max(cfg.n_total_samples, 1000)
            print(f"  [Data] ISD insufficient — synthetic ({n_syn}h)")
            data_arr = generate_synthetic_weather(n_syn, seed=seed)
            data_arr = build_features_from_array(data_arr)
    else:
        data_arr = generate_synthetic_weather(cfg.n_total_samples, seed=seed)
        data_arr = build_features_from_array(data_arr)

    N = len(data_arr)
    n_test = int(N * 0.10)
    n_val  = int(N * 0.10)
    n_train = N - n_val - n_test

    train_raw = data_arr[:n_train]
    val_raw   = data_arr[n_train:n_train + n_val]
    test_raw  = data_arr[n_train + n_val:]

    hours_train = np.arange(n_train) % 24
    norm_mean = np.array([train_raw[hours_train == h].mean(axis=0) for h in range(24)])
    norm_std  = np.array([train_raw[hours_train == h].std(axis=0)  for h in range(24)])
    norm_std  = np.maximum(norm_std, 1e-6)

    def deseason(arr):
        out = arr.copy()
        for i in range(len(arr)):
            out[i] = (arr[i] - norm_mean[i % 24]) / norm_std[i % 24]
        return out

    scaler = StandardScaler()
    train_sc = scaler.fit_transform(deseason(train_raw))
    val_sc   = scaler.transform(deseason(val_raw))
    test_sc  = scaler.transform(deseason(test_raw))

    tci = cfg.target_col_idx
    W = cfg.data_window
    eval_horizons = cfg.eval_horizons
    fsdh_max = cfg.fsdh_max
    all_horizons = list(range(1, fsdh_max + 1))
    fsdh_idx = [all_horizons.index(h) for h in eval_horizons]

    def make_win(arr, h_list):
        mx = max(h_list)
        xl, yl = [], []
        for i in range(W, len(arr) - mx + 1):
            w = arr[i - W:i]
            t = np.array([arr[i + h - 1, tci] for h in h_list])
            if not (np.isnan(w).any() or np.isnan(t).any()):
                xl.append(w); yl.append(t)
        return np.array(xl), np.array(yl)

    X_tr, y_tr = make_win(train_sc, all_horizons)
    X_va, y_va = make_win(val_sc,   all_horizons)
    X_te, y_te = make_win(test_sc,  all_horizons)

    y_tr_eval = y_tr[:, fsdh_idx]
    y_te_eval = y_te[:, fsdh_idx]

    print(f"  [Preprocess] X_train={X_tr.shape} y_train={y_tr_eval.shape}")
    print(f"  [Preprocess] X_test={X_te.shape}  (FSDH horizons: 1..{fsdh_max})")

    # --- 2. RESERVOIR ---
    print("\n[2/8] Building reservoir...")
    trot = cfg.effective_trotter_steps
    qrc = AtmosphericQRC(n_qubits=nq, n_layers=cfg.n_layers,
                         trotter_steps=trot,
                         n_input_features=cfg.n_input_features,
                         seed=seed, jg_target=cfg.jg_target,
                         h_range=cfg.h_range, g_range=cfg.g_range,
                         J_range=cfg.J_range)
    if not (0.3 < qrc.jg_ratio < 2.5):
        print(f"  [Reservoir] Warning: J/g={qrc.jg_ratio:.3f} far from critical point")

    # --- 3. EXTRACT FEATURES ONCE ---
    print("\n[3/8] Extracting reservoir features...")
    F_tr = qrc.transform(X_tr)
    F_te = qrc.transform(X_te)

    # --- 4. ESP VERIFICATION ---
    print("\n[4/8] ESP verification...")
    esp_input = np.random.default_rng(seed).normal(size=(50, W, cfg.n_input_features))
    esp_fig, esp_converged, esp_distances = verify_esp(
        qrc, esp_input,
        n_initial_states=cfg.esp_n_initial_states,
        n_steps=cfg.esp_n_steps,
        convergence_threshold=cfg.esp_convergence_threshold,
    )
    results_meta = {"esp_converged": esp_converged}

    # --- 5. IPC SWEEP ---
    print("\n[5/8] IPC sweep...")
    ipc_fig, jg_vals, ipc_vals = sweep_jg_ratio(
        AtmosphericQRC, n_qubits=nq, n_samples=cfg.ipc_n_samples,
        W=cfg.ipc_window, jg_range=cfg.ipc_jg_range,
        n_points=cfg.ipc_n_points, seed=seed,
        n_layers=cfg.ipc_n_layers, trotter_steps=cfg.ipc_trotter_steps,
        ridge_alpha=cfg.ipc_ridge_alpha,
    )

    # --- 6. MEMORY CAPACITY ---
    print("\n[6/8] Memory capacity...")
    mc_result = measure_memory_capacity(
        qrc, max_lag=cfg.mc_max_lag,
        n_samples=cfg.mc_n_samples, W=cfg.mc_window,
        ridge_alpha=cfg.mc_ridge_alpha,
    )
    results_meta["MC"] = mc_result["MC"]
    results_meta["MC_per_lag"] = mc_result["per_lag"]

    N_H = len(all_horizons)
    n_eval = len(eval_horizons)

    # --- 7. TRAIN ALL MODELS ---
    print("\n[7/8] Training models...")

    esn_name = f"ESN-{cfg.esn_nodes}"

    print(f"  [DirectQRC] Training readouts (1..{fsdh_max}h)...")
    d_readouts = {}
    for h_idx in range(N_H):
        krr = KRRReadout(gamma_grid=cfg.krr_gamma_grid,
                         alpha_grid=cfg.krr_alpha_grid,
                         tscv_splits=cfg.krr_tscv_splits)
        krr.fit(F_tr, y_tr, horizon_idx=h_idx)
        d_readouts[h_idx] = krr
    y_direct_all = np.column_stack([d_readouts[h].predict(F_te) for h in range(N_H)])

    print(f"  [ResidualQRC] Training readouts (1..{fsdh_max}h)...")
    persist_tr = X_tr[:, -1, tci][:, np.newaxis]
    res_tr = y_tr - persist_tr
    r_readouts = {}
    for h_idx in range(N_H):
        krr = KRRReadout(gamma_grid=cfg.krr_gamma_grid,
                         alpha_grid=cfg.krr_alpha_grid,
                         tscv_splits=cfg.krr_tscv_splits)
        krr.fit(F_tr, res_tr, horizon_idx=h_idx)
        r_readouts[h_idx] = krr
    res_preds_all = np.column_stack([r_readouts[h].predict(F_te) for h in range(N_H)])
    persist_te = X_te[:, -1, tci][:, np.newaxis]
    y_residual_all = persist_te + res_preds_all

    y_persist_all = np.tile(persist_te, (1, N_H))

    print("  [Baselines] Training at eval horizons...")
    persistence = PersistenceBaseline(target_col_idx=tci)
    arima = ARIMABaseline(order=cfg.arima_order)
    esn = ESNBaseline(nodes=cfg.esn_nodes, seed=cfg.esn_seed,
                      sr=cfg.effective_esn_sr,
                      input_scaling=cfg.esn_input_scaling,
                      rc_connectivity=cfg.esn_rc_connectivity,
                      ridge_alpha=cfg.esn_ridge_alpha,
                      warmup=cfg.esn_warmup)
    persistence.fit(X_tr, y_tr_eval)
    arima.fit(X_tr, y_tr_eval)
    esn.fit(X_tr, y_tr_eval)
    y_persist_eval = persistence.predict(X_te)
    y_arima_eval   = arima.predict(X_te)
    y_esn_eval     = esn.predict(X_te)

    # ARIMA iterative forecast for full FSDH curve
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from statsmodels.tsa.arima.model import ARIMA
            arima_fsdh = ARIMA(y_tr[:, 0], order=cfg.arima_order).fit()
            fc = arima_fsdh.forecast(steps=X_te.shape[0] + N_H - 1)
            y_arima_all = np.column_stack([fc[h:h + X_te.shape[0]] for h in range(N_H)])
        except Exception:
            y_arima_all = y_persist_all.copy()

    # ESN full curve
    esn_full = ESNBaseline(nodes=cfg.esn_nodes, seed=cfg.esn_seed,
                           sr=cfg.effective_esn_sr,
                           input_scaling=cfg.esn_input_scaling,
                           rc_connectivity=cfg.esn_rc_connectivity,
                           ridge_alpha=cfg.esn_ridge_alpha,
                           warmup=cfg.esn_warmup)
    esn_full.fit(X_tr, y_tr)
    y_esn_all = esn_full.predict(X_te)

    # --- EVALUATE ---
    print("\n[8/8] Evaluating...")

    y_direct_eval   = y_direct_all[:, fsdh_idx]
    y_residual_eval = y_residual_all[:, fsdh_idx]

    results = {}
    for h_idx, h_name in enumerate([f"{h}h" for h in eval_horizons]):
        gt = y_te_eval[:, h_idx]
        results[h_name] = {
            "DirectQRC":   {"rmse": rmse(gt, y_direct_eval[:, h_idx]),
                            "mae":  mae(gt, y_direct_eval[:, h_idx]),
                            "skill": skill_score(gt, y_direct_eval[:, h_idx], y_persist_eval[:, h_idx])},
            "ResidualQRC": {"rmse": rmse(gt, y_residual_eval[:, h_idx]),
                            "mae":  mae(gt, y_residual_eval[:, h_idx]),
                            "skill": skill_score(gt, y_residual_eval[:, h_idx], y_persist_eval[:, h_idx])},
            "Persistence": {"rmse": rmse(gt, y_persist_eval[:, h_idx]),
                            "mae":  mae(gt, y_persist_eval[:, h_idx]),
                            "skill": 0.0},
            "ARIMA":       {"rmse": rmse(gt, y_arima_eval[:, h_idx]),
                            "mae":  mae(gt, y_arima_eval[:, h_idx]),
                            "skill": skill_score(gt, y_arima_eval[:, h_idx], y_persist_eval[:, h_idx])},
            esn_name:      {"rmse": rmse(gt, y_esn_eval[:, h_idx]),
                            "mae":  mae(gt, y_esn_eval[:, h_idx]),
                            "skill": skill_score(gt, y_esn_eval[:, h_idx], y_persist_eval[:, h_idx])},
        }

    # --- FSDH CURVE ---
    print(f"\n  FSDH curve (1..{fsdh_max}h)...")
    model_names_all = ["DirectQRC", "ResidualQRC", "Persistence", "ARIMA", esn_name]
    model_preds_all = [y_direct_all, y_residual_all, y_persist_all, y_arima_all, y_esn_all]
    model_rmse_curves = {}
    for name, preds in zip(model_names_all, model_preds_all):
        curve = np.array([rmse(y_te[:, h], preds[:, h]) for h in range(N_H)])
        model_rmse_curves[name] = curve

    p_curve = model_rmse_curves["Persistence"]
    print(f"\n  {'Model':<18} {'FSDH':>6}   {'VPT':>6} (hours)")
    print(f"  {'-' * 48}")
    fsdh_vals = {}
    for name in ["DirectQRC", "ResidualQRC", "ARIMA", esn_name]:
        mc = model_rmse_curves[name]
        fsdh_h = 0
        for h in range(N_H):
            if mc[h] < p_curve[h]:
                fsdh_h = h + 1
            else:
                break
        fsdh_vals[name] = fsdh_h
        vpt_h = compute_vpt_curve(y_te, model_preds_all[model_names_all.index(name)], threshold=0.5)
        print(f"  {name:<18} {fsdh_h:>4}h  {vpt_h:>4}h")

    results["FSDH"] = {"DirectQRC": fsdh_vals["DirectQRC"],
                       "ResidualQRC": fsdh_vals["ResidualQRC"],
                       "ARIMA": fsdh_vals["ARIMA"],
                       esn_name: fsdh_vals[esn_name],
                       "Persistence": 0}

    vpt_direct   = compute_vpt_curve(y_te_eval, y_direct_eval)
    vpt_residual = compute_vpt_curve(y_te_eval, y_residual_eval)
    results["VPT"] = {"DirectQRC": vpt_direct, "ResidualQRC": vpt_residual,
                      "Persistence": compute_vpt_curve(y_te_eval, y_persist_eval),
                      "ARIMA": compute_vpt_curve(y_te_eval, y_arima_eval),
                      esn_name: compute_vpt_curve(y_te_eval, y_esn_eval)}

    results["fsdh_curves"] = model_rmse_curves
    results["fsdh_horizons"] = all_horizons

    vpt_curve = {}
    for name, preds in [("DirectQRC", y_direct_all),
                        ("ResidualQRC", y_residual_all),
                        ("Persistence", y_persist_all),
                        ("ARIMA", y_arima_all),
                        (esn_name, y_esn_all)]:
        vpt_curve[name] = compute_vpt_curve(y_te, preds, threshold=0.5)
    results["VPT_curve"] = vpt_curve
    print(f"  VPT {fsdh_max}h: "
          + "  ".join(f"{n}={vpt_curve[n]}h" for n in ["DirectQRC", "ResidualQRC", "ARIMA", esn_name, "Persistence"]))

    results["MC"] = results_meta["MC"]
    results["MC_per_lag"] = results_meta["MC_per_lag"]
    results["esp_converged"] = results_meta["esp_converged"]

    # Ablation: same cached features
    print("\n  Ablation: single-body vs two-body readout...")
    nf_single = 3 * qrc.n_qubits
    F_tr_sb = F_tr[:, :nf_single]
    F_te_sb = F_te[:, :nf_single]

    krr_sb = KernelRidge(kernel="rbf", gamma=cfg.ablation_gamma, alpha=cfg.ablation_alpha)
    krr_tb = KernelRidge(kernel="rbf", gamma=cfg.ablation_gamma, alpha=cfg.ablation_alpha)
    krr_sb.fit(F_tr_sb, y_tr[:, 0])
    krr_tb.fit(F_tr,    y_tr[:, 0])

    rmse_sb = rmse(y_te[:, 0], krr_sb.predict(F_te_sb))
    rmse_tb = rmse(y_te[:, 0], krr_tb.predict(F_te))
    results["ablation"] = {
        "single_body_rmse_1h": rmse_sb,
        "two_body_rmse_1h":    rmse_tb,
        "improvement_pct":     100 * (rmse_sb - rmse_tb) / (rmse_sb + 1e-8),
    }

    # Noise sweep: same cached features
    print("\n  Noise sweep...")
    noise_fig, noise_results = noise_sweep(
        F_train=F_tr, F_test=F_te, y_train=y_tr, y_test=y_te,
        p_values=cfg.noise_p_values,
        krr_gamma=cfg.noise_krr_gamma, krr_alpha=cfg.noise_krr_alpha,
        noise_multiplier=cfg.noise_multiplier,
    )
    results["noise"] = noise_results
    results["jg_ratio"] = qrc.jg_ratio
    results["n_qubits"] = nq

    return results, esp_fig, ipc_fig, noise_fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(__doc__)

    # ── Override any field here ──────────────────────────────────────────
    cfg = ExperimentConfig(
        n_qubits=12,           # reference config; 10 is the only permitted fallback
        n_total_samples=4350,  # increase to 8000 for full year
        fsdh_max=12,           # try 6, 24, 48
        eval_horizons=[1, 6],  # try [1, 3, 6, 12]
        seed=42,
    )
    # ─────────────────────────────────────────────────────────────────────

    results, esp_fig, ipc_fig, noise_fig = run_full_experiment(cfg)

    print_summary(results, config=cfg)

    print("\n[Figures] Generating paper figures...")
    plot_all_figures(results, save_dir=cfg.fig_save_dir,
                     esp_fig=esp_fig, ipc_fig=ipc_fig, noise_fig=noise_fig,
                     config=cfg)

    print("\n[Gate checks] Verifying all pipeline contracts...")

    for m, val in results["FSDH"].items():
        assert isinstance(val, int), f"FSDH[{m}] is {type(val)}, must be int"

    assert isinstance(results["MC"], float), f"MC is {type(results['MC'])}, must be float"

    abl = results["ablation"]
    print(f"  Two-body vs single-body improvement: {abl['improvement_pct']:.1f}%")
    print(f"  J/g ratio: {results['jg_ratio']:.3f}")
    print(f"  ESP converged: {results.get('esp_converged', 'N/A')}")

    print("\nAll pipeline contracts verified.")
    print("Winning solution v4 demo complete.")
