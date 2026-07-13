import numpy as np
from scipy.integrate import solve_ivp


def generate_lorenz63(
    n_steps: int = 5000, dt: float = 0.01, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    def lorenz(t, state):
        x, y, z = state
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t0 = rng.uniform(-15, 15, size=3)
    t_span = (0.0, dt * n_steps)
    t_eval = np.linspace(0.0, dt * n_steps, n_steps)
    sol = solve_ivp(lorenz, t_span, t0, method="RK45", t_eval=t_eval, rtol=1e-8, atol=1e-10)
    return sol.y.T
