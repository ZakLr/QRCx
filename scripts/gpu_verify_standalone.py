"""Standalone GPU benchmark for the Sprint 2.5 SequentialDissipativeQRC
performance pass. Self-contained (no QRCx import needed) so it can be
pasted directly into a qBraid Lab GPU notebook cell.

Usage in a qBraid Lab GPU notebook:
    !pip install cupy-cuda12x   # match the notebook's CUDA version if needed
    # paste this whole file's contents into a cell and run
    # then paste the printed output back

Requires: numpy, and cupy (cupy-cuda11x or cupy-cuda12x depending on the
qBraid GPU environment's CUDA version) for the GPU path. Falls back to
numpy (CPU) if cupy isn't available, so it's safe to run anywhere.
"""
import time

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    HAS_CUPY = False

print(f"HAS_CUPY (GPU visible): {HAS_CUPY}")
if HAS_CUPY:
    print(f"GPU device: {cp.cuda.runtime.getDeviceProperties(0)['name']}")


def build_tfim_hamiltonian(n_qubits, J, g):
    dim = 2 ** n_qubits
    idx = np.arange(dim)
    z = [1 - 2 * ((idx >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)]
    diag = np.zeros(dim, dtype=np.float64)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            diag += J * z[i] * z[j]
    H = np.diag(diag).astype(np.complex128)
    for q in range(n_qubits):
        bitpos = n_qubits - 1 - q
        flipped = idx ^ (1 << bitpos)
        H[idx, flipped] += g
    return H, z


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def conjugate_1q(xp, rho, K, qubit, n_qubits, dim):
    K = xp.asarray(K, dtype=rho.dtype)
    P, S = 2 ** qubit, 2 ** (n_qubits - qubit - 1)
    r = rho.reshape(P, 2, S * dim)
    r = xp.matmul(K, r)
    r = r.reshape(dim, dim)
    Kc = xp.conj(K)
    r = r.reshape(dim * P, 2, S)
    r = xp.matmul(Kc, r)
    return r.reshape(dim, dim)


def benchmark(n_qubits, use_gpu, dtype_str, n_steps=5, gamma1=0.02, gamma2=0.02, seed=0):
    xp = cp if (use_gpu and HAS_CUPY) else np
    dtype = getattr(xp, dtype_str)
    float_dtype = xp.float64 if dtype_str == "complex128" else xp.float32
    dim = 2 ** n_qubits

    H, z_np = build_tfim_hamiltonian(n_qubits, J=1.0, g=1.0)
    evals, evecs = np.linalg.eigh(H)
    U_full = (evecs * np.exp(-1j * evals * 1.0)) @ evecs.conj().T
    U_full = xp.asarray(U_full, dtype=dtype)

    z = [xp.asarray(zz) for zz in z_np]

    # dephasing coeff (precomputed once, like the CPU fix)
    c2 = float(np.sqrt(max(0.0, 1.0 - gamma2)))
    dephasing_coeff = xp.ones((dim, dim), dtype=float_dtype)
    for q in range(n_qubits):
        same_bit = (z[q][:, None] * z[q][None, :]) > 0
        dephasing_coeff = dephasing_coeff * xp.where(same_bit, 1.0, c2).astype(float_dtype)

    c1 = float(np.sqrt(max(0.0, 1.0 - gamma1)))
    damping_d = [xp.where(z[q] > 0, 1.0, c1).astype(float_dtype) for q in range(n_qubits)]
    damping_e1 = xp.asarray([[0, np.sqrt(gamma1)], [0, 0]], dtype=dtype)

    rng = np.random.default_rng(seed)
    seq = rng.uniform(-1, 1, size=(n_steps, n_qubits))

    rho = xp.zeros((dim, dim), dtype=dtype)
    rho[0, 0] = 1.0

    if use_gpu and HAS_CUPY:
        cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for k in range(n_steps):
        x = seq[k]
        for j in range(n_qubits):
            rho = conjugate_1q(xp, rho, ry(x[j]), j, n_qubits, dim)
        rho = U_full @ rho @ U_full.conj().T
        rho = rho * dephasing_coeff
        for j in range(n_qubits):
            d = damping_d[j]
            term0 = rho * d[:, None] * d[None, :]
            term1 = conjugate_1q(xp, rho, damping_e1, j, n_qubits, dim)
            rho = term0 + term1
    if use_gpu and HAS_CUPY:
        cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / n_steps


print("\n--- CPU (numpy) ---")
for n in [12, 10]:
    s = benchmark(n, use_gpu=False, dtype_str="complex128", n_steps=3)
    print(f"CPU n={n} complex128: {s:.4f} s/step")

if HAS_CUPY:
    print("\n--- GPU (cupy) ---")
    for n in [12, 10]:
        # warmup (first call compiles CUDA kernels)
        benchmark(n, use_gpu=True, dtype_str="complex128", n_steps=1)
        s128 = benchmark(n, use_gpu=True, dtype_str="complex128", n_steps=5)
        s64 = benchmark(n, use_gpu=True, dtype_str="complex64", n_steps=5)
        print(f"GPU n={n} complex128: {s128:.4f} s/step")
        print(f"GPU n={n} complex64:  {s64:.4f} s/step")
else:
    print("\nNo GPU/cupy detected -- install cupy-cuda12x (or cuda11x) and re-run in a GPU-enabled qBraid environment.")
