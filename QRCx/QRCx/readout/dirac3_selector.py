"""FINAL SPRINT Phase 3: readout feature sparsification via QCi's Dirac-3
(polynomial/QUBO-style continuous optimizer), with a local simulated-
annealing stand-in behind the identical interface so development and CI
never block on device access.

Formulation (best-subset selection of reservoir readout features):
    minimize over z in [0,1]^n:  E(z) = -2 c^T z + z^T Q z
    subject to sum(z) = K
  where c = feature-target covariances (train split), Q = feature Gram
  matrix (train split), z = (relaxed) selection variables. Quadratic --
  within Dirac-3's stated 5th-order polynomial capability. The top-K
  entries of z (by value, after enforcing the sum constraint) are taken
  as the selected feature indices.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SelectionResult:
    solver: str
    K: int
    selected_idx: np.ndarray
    energy: float
    device_params: dict = field(default_factory=dict)
    wall_clock_s: float = 0.0
    error: Optional[str] = None


def build_problem(F_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """c = feature-target covariance vector, Q = feature Gram matrix,
    both computed on the train split only (never val/test)."""
    y = y_train if y_train.ndim == 1 else y_train.mean(axis=1)
    Fc = F_train - F_train.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    c = Fc.T @ yc / len(y)
    Q = (Fc.T @ Fc) / len(y)
    return c, Q


def _sum_constrained_round(z: np.ndarray, K: int) -> np.ndarray:
    """Project a continuous relaxation onto exactly-K selected indices:
    take the K largest entries of z. This is the standard rounding step
    for a sum(z)=K box-constrained relaxation."""
    idx = np.argsort(-z)[:K]
    return np.sort(idx)


def select_sa(c: np.ndarray, Q: np.ndarray, K: int, seed: int = 0,
              n_restarts: int = 8, n_iters: int = 4000) -> SelectionResult:
    """Local simulated-annealing stand-in for the Dirac-3 formulation
    above, operating directly on the discrete K-subset (not the
    continuous relaxation) via swap moves -- behind the identical
    `SelectionResult` interface as the real device call."""
    t0 = time.perf_counter()
    n = len(c)
    rng = np.random.default_rng(seed)

    def energy(idx):
        z = np.zeros(n)
        z[idx] = 1.0
        return float(-2.0 * c @ z + z @ Q @ z)

    best_idx, best_e = None, np.inf
    for restart in range(n_restarts):
        idx = set(rng.choice(n, size=K, replace=False).tolist())
        e = energy(np.array(sorted(idx)))
        T0, T1 = 1.0, 1e-3
        for it in range(n_iters):
            T = T0 * (T1 / T0) ** (it / n_iters)
            out_elem = rng.choice(list(idx))
            in_elem = rng.integers(0, n)
            if in_elem in idx:
                continue
            new_idx = (idx - {out_elem}) | {in_elem}
            e_new = energy(np.array(sorted(new_idx)))
            if e_new < e or rng.random() < np.exp(-(e_new - e) / max(T, 1e-9)):
                idx, e = new_idx, e_new
        if e < best_e:
            best_idx, best_e = idx, e

    return SelectionResult(
        solver="sa", K=K, selected_idx=np.array(sorted(best_idx)), energy=best_e,
        device_params={"n_restarts": n_restarts, "n_iters": n_iters, "seed": seed},
        wall_clock_s=time.perf_counter() - t0,
    )


def select_dirac3(c: np.ndarray, Q: np.ndarray, K: int, num_samples: int = 20,
                   relaxation_schedule: int = 1, cache_dir: Optional[Path] = None,
                   max_retries: int = 3) -> SelectionResult:
    """Real Dirac-3 submission via qci-client. Requires QCI_API_URL /
    QCI_TOKEN env vars (falls back to qci_client's own env lookup if
    api_token/url not passed explicitly). Caches the raw device
    response to cache_dir so a poll timeout doesn't lose a paid job."""
    t0 = time.perf_counter()
    try:
        import qci_client
    except ImportError as e:
        return SelectionResult(solver="dirac3", K=K, selected_idx=np.array([]),
                                energy=np.nan, error=f"qci_client not installed: {e}",
                                wall_clock_s=time.perf_counter() - t0)

    try:
        client = qci_client.QciClient()
    except Exception as e:
        return SelectionResult(solver="dirac3", K=K, selected_idx=np.array([]),
                                energy=np.nan,
                                error=f"QciClient() init failed (no QCI_API_URL/QCI_TOKEN "
                                      f"credentials configured in this environment): {e}",
                                wall_clock_s=time.perf_counter() - t0)

    n = len(c)
    poly_terms = []
    for i in range(n):
        poly_terms.append({"idx": [i + 1], "val": float(-2.0 * c[i])})
    for i in range(n):
        for j in range(i, n):
            val = float(Q[i, j] * (1.0 if i == j else 2.0))
            if val != 0.0:
                poly_terms.append({"idx": [i + 1, j + 1], "val": val})

    last_err = None
    for attempt in range(max_retries):
        try:
            poly_file = {
                "file_name": f"dirac3_selector_K{K}.json",
                "file_config": {"polynomial": {"min_degree": 1, "max_degree": 2,
                                                "num_variables": n, "data": poly_terms}},
            }
            file_id = client.upload_file(file=poly_file)["file_id"]
            job_body = client.build_job_body(
                job_type="sample-hamiltonian", polynomial_file_id=file_id,
                job_params={"device_type": "dirac-3", "num_samples": num_samples,
                            "relaxation_schedule": relaxation_schedule,
                            "sum_constraint": K},
            )
            job_response = client.process_job(job_body=job_body)
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_dir / f"dirac3_response_K{K}.json", "w") as f:
                    json.dump(job_response, f, indent=2, default=str)
            samples = job_response["results"]["solutions"]
            energies = job_response["results"]["energies"]
            best = int(np.argmin(energies))
            z = np.array(samples[best])
            selected_idx = _sum_constrained_round(z, K)
            return SelectionResult(
                solver="dirac3", K=K, selected_idx=selected_idx, energy=float(energies[best]),
                device_params={"num_samples": num_samples,
                                "relaxation_schedule": relaxation_schedule,
                                "allocation_seconds": job_response.get("job_metrics", {})
                                                                    .get("total_time", None)},
                wall_clock_s=time.perf_counter() - t0,
            )
        except Exception as e:
            last_err = str(e)
            time.sleep(1.0)

    return SelectionResult(solver="dirac3", K=K, selected_idx=np.array([]), energy=np.nan,
                            error=f"failed after {max_retries} attempts: {last_err}",
                            wall_clock_s=time.perf_counter() - t0)


def select(solver: str, c: np.ndarray, Q: np.ndarray, K: int, **kwargs) -> SelectionResult:
    if solver == "sa":
        return select_sa(c, Q, K, **kwargs)
    if solver == "dirac3":
        return select_dirac3(c, Q, K, **kwargs)
    raise ValueError(f"Unknown solver {solver!r}, expected 'sa' or 'dirac3'")
