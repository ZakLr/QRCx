"""NARMA10 synthetic benchmark (Jaeger 2001 / Fujii & Nakajima 2017 standard
reservoir-computing sanity check): a nonlinear autoregressive-moving-average
task of order 10. Used in Sprint 1 as a fast micro-validation that the
sequential dissipative reservoir beats a linear AR baseline before
committing to the full ISD pipeline.
"""
import numpy as np


def generate_narma10(n_steps: int = 2000, seed: int = 42) -> tuple:
    """Generate a NARMA10 sequence.

    Returns:
        u: input sequence, shape (n_steps,), iid Uniform(0, 0.5).
        y: target sequence, shape (n_steps,), y[t] for t < 10 is 0.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=n_steps)
    y = np.zeros(n_steps)
    for t in range(10, n_steps):
        y[t] = (
            0.3 * y[t - 1]
            + 0.05 * y[t - 1] * np.sum(y[t - 10 : t])
            + 1.5 * u[t - 10] * u[t - 1]
            + 0.1
        )
    return u, y
