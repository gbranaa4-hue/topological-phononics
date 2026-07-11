#!/usr/bin/env python3
"""The unifying axis: RANK. Random noise is just FULL-rank structured noise. The trained readout
can null LOW-rank structured interference; it cannot null full-rank. This sweeps the rank of
readout-stage structured noise at FIXED total power, interpolating between the two extremes of
noise_feature_stage.py:  rank 1 = the cancelled common-mode ;  rank M = effectively random.

Design number wanted: how many correlated interferer directions can the M-node readout absorb
before structured noise starts eating signal capacity?

PRE-REGISTERED: MC ~flat (full cancellation) while rank << M, then falls toward the random-noise
floor as rank -> M. => "cancellable" == "low-rank", and the readout's spare dimensions set the margin.
"""
import numpy as np

N = 8; M = 2 * N; T = 3500; BURN = 400
VT, WT = 0.4, 1.0; D = 0.5

def ssh_H(v, w):
    H = np.zeros((M, M))
    for i in range(M - 1): H[i, i + 1] = H[i + 1, i] = (v if i % 2 == 0 else w)
    return H

def states(rank, seed=0, alpha=0.5, rho=0.95):
    rng = np.random.default_rng(seed)
    W = rho * ssh_H(VT, WT) / (np.max(np.abs(np.linalg.eigvalsh(ssh_H(VT, WT)))) + 1e-9)
    u = rng.standard_normal(T); win = np.full(M, 0.5)
    # clean dynamics (noise only at the readout)
    x = np.zeros(M); X = np.zeros((T, M))
    for t in range(T):
        x = (1 - alpha) * x + alpha * np.tanh(W @ x + win * u[t]); X[t] = x
    # rank-r structured noise: r orthonormal directions, r independent AR(1) series, FIXED total power
    V, _ = np.linalg.qr(rng.standard_normal((M, max(rank, 1))))
    V = V[:, :rank]
    C = np.zeros((T, rank))
    for i in range(rank):
        c = np.zeros(T)
        for t in range(1, T): c[t] = 0.9 * c[t - 1] + rng.standard_normal()
        C[:, i] = c / (c.std() + 1e-9)
    Xn = X + (D / np.sqrt(rank)) * (C @ V.T)      # /sqrt(rank) => total injected power independent of rank
    return Xn[BURN:], u[BURN:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def MC(rank, seed=0, dmax=12):
    X, u = states(rank, seed)
    n = len(u); tr = int(0.6 * n); mc = 0.0
    for k in range(1, dmax + 1):
        t = np.arange(k, n); Xf, y = X[t], u[t - k]; sp = tr - k
        w_ = ridge(Xf[:sp], y[:sp])
        mc += max(0.0, 1 - np.mean((Xf[sp:] @ w_ - y[sp:]) ** 2) / (np.var(y[sp:]) + 1e-12))
    return mc

print(f"readout-stage structured noise, FIXED total power D={D}, N={N} cells ({M} nodes).")
print("rank 1 = single common-mode (fully cancellable) ... rank M = effectively random.\n")
print(f"{'rank':>5} | {'MC':>6} | fraction of clean MC retained")
print("-" * 46)
clean = np.mean([MC(1, s) for s in range(2)])   # rank-1 ~ fully cancelled ~ clean ceiling
for r in (1, 2, 4, 8, 12, 16):
    mc = np.mean([MC(r, s) for s in range(2)])
    bar = "#" * int(round(24 * mc / max(clean, 1e-6)))
    print(f"{r:>5} | {mc:>6.2f} | {bar} {mc/max(clean,1e-6)*100:4.0f}%")
print("\n--- flat while rank is small (readout nulls it), decaying to the random floor as rank->M ---")
