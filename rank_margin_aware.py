#!/usr/bin/env python3
"""Is the structured-noise cancellation MAXIMIZED? rank_margin.py used PLAIN ridge (blind uniform
shrinkage) -- not the optimal linear noise-canceller. The linear CEILING is a readout that knows
the noise subspace V and projects it out: X_perp = X (I - V V^T), then regress. This measures the
gap = headroom left by plain ridge, and exposes the true ceiling (nulling r dirs also costs the
signal that lived in them, so even the oracle falls as rank->M).

PRE-REGISTERED: aware >= plain at every rank; gap is ~0 at rank 1 (plain already perfect) and opens
at intermediate rank (plain leaves noise in); both decay as rank->M (oracle sacrifices signal dims).
If the gap is large, plain ridge was NOT maximized; if ~0, it already was.
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
    x = np.zeros(M); X = np.zeros((T, M))
    for t in range(T):
        x = (1 - alpha) * x + alpha * np.tanh(W @ x + win * u[t]); X[t] = x
    V, _ = np.linalg.qr(rng.standard_normal((M, max(rank, 1)))); V = V[:, :rank]
    C = np.zeros((T, rank))
    for i in range(rank):
        c = np.zeros(T)
        for t in range(1, T): c[t] = 0.9 * c[t - 1] + rng.standard_normal()
        C[:, i] = c / (c.std() + 1e-9)
    Xn = X + (D / np.sqrt(rank)) * (C @ V.T)
    return Xn[BURN:], u[BURN:], V

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def MC(rank, aware, seed=0, dmax=12):
    X, u, V = states(rank, seed)
    if aware: X = X - (X @ V) @ V.T                 # project out the known noise subspace (linear ceiling)
    n = len(u); tr = int(0.6 * n); mc = 0.0
    for k in range(1, dmax + 1):
        t = np.arange(k, n); Xf, y = X[t], u[t - k]; sp = tr - k
        w_ = ridge(Xf[:sp], y[:sp])
        mc += max(0.0, 1 - np.mean((Xf[sp:] @ w_ - y[sp:]) ** 2) / (np.var(y[sp:]) + 1e-12))
    return mc

am = lambda rank, aware: np.mean([MC(rank, aware, s) for s in range(2)])

print(f"structured noise, D={D}, {M} nodes. PLAIN ridge vs NOISE-AWARE (subspace-projected) readout.\n")
print(f"{'rank':>5} | {'plain MC':>8} | {'aware MC':>8} | {'aware gain':>10} | headroom left by plain ridge")
print("-" * 66)
for r in (1, 2, 4, 8, 12, 16):
    p, a = am(r, False), am(r, True)
    bar = "#" * int(round(30 * max(a - p, 0) / max(a, 1e-6)))
    print(f"{r:>5} | {p:>8.2f} | {a:>8.2f} | {a-p:>+9.2f} | {bar} {100*max(a-p,0)/max(a,1e-6):3.0f}%")
print("\n--- gap ~0  => plain ridge already ~maximized ;  gap large => headroom via a noise-aware readout")
print("    (even the oracle falls as rank->M: nulling r directions also costs the signal in them) ---")
