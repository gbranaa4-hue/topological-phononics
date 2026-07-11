#!/usr/bin/env python3
"""Noise as a feature: can a reservoir + TRAINED readout reject STRUCTURED (correlated) noise
while RANDOM (thermal-like) noise gets through? Both injected at MATCHED per-node power.
  structured = common-mode, temporally-correlated (AR1) -> rank-1 -> the linear readout can NULL it
  random     = full-rank white per node                 -> fills all dims -> readout CANNOT null it
Metric: linear memory capacity (MC). Leaky-integrator SSH ESN (leak alpha = cavity Q).

PRE-REGISTERED
  E1  at matched power, STRUCTURED noise preserves MC (readout cancels it) while RANDOM degrades
      it -> "the reservoir eats structured noise; only the random floor survives."
  E2  under RANDOM noise, HIGH-Q retains MC better than LOW-Q at high noise (averaging crossover).
  E3  topology barely changes noise rejection (topology is for DEFECTS, not noise) -- honest check.
"""
import numpy as np

N = 8; M = 2 * N; T = 3500; BURN = 400
VT, WT = 0.4, 1.0; VR, WR = 1.0, 0.4

def ssh_H(v, w):
    H = np.zeros((M, M))
    for i in range(M - 1): H[i, i + 1] = H[i + 1, i] = (v if i % 2 == 0 else w)
    return H

def states(v, w, alpha, noise, D, seed=0, rho=0.95):
    rng = np.random.default_rng(seed)
    W = rho * ssh_H(v, w) / (np.max(np.abs(np.linalg.eigvalsh(ssh_H(v, w)))) + 1e-9)
    u = rng.standard_normal(T); win = np.full(M, 0.5)
    s = np.zeros(T)                                          # AR(1) common-mode structured noise
    for t in range(1, T): s[t] = 0.9 * s[t - 1] + rng.standard_normal()
    s = s / (s.std() + 1e-9)                                 # unit variance -> matched power vs random
    x = np.zeros(M); X = np.zeros((T, M))
    for t in range(T):
        pre = np.tanh(W @ x + win * u[t])
        if noise == "structured": nz = D * s[t] * np.ones(M)          # rank-1, correlated
        elif noise == "random":   nz = D * rng.standard_normal(M)     # full-rank, white
        else:                     nz = 0.0
        x = (1 - alpha) * x + alpha * (pre + nz)            # noise inside the leak (so high-Q averages it)
        X[t] = x
    return X[BURN:], u[BURN:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def MC(v, w, alpha, noise, D, seed=0, dmax=12):
    X, u = states(v, w, alpha, noise, D, seed)
    n = len(u); tr = int(0.6 * n); mc = 0.0
    for k in range(1, dmax + 1):
        t = np.arange(k, n); Xf, y = X[t], u[t - k]; s = tr - k
        w_ = ridge(Xf[:s], y[:s])
        r2 = 1 - np.mean((Xf[s:] @ w_ - y[s:]) ** 2) / (np.var(y[s:]) + 1e-12)
        mc += max(0.0, r2)
    return mc

def amc(seeds=2, **kw): return np.mean([MC(seed=s, **kw) for s in range(seeds)])

print("E1 -- readout rejection: MC vs noise power, STRUCTURED vs RANDOM (topological, alpha=0.5):\n")
print(f"{'noise D':>8} | {'STRUCTURED MC':>13} | {'RANDOM MC':>9} | structured advantage")
print("-" * 60)
for D in (0.0, 0.1, 0.3, 0.6):
    st = amc(v=VT, w=WT, alpha=0.5, noise="structured", D=D)
    rn = amc(v=VT, w=WT, alpha=0.5, noise="random", D=D)
    print(f"{D:>8.2f} | {st:>13.2f} | {rn:>9.2f} | {st/max(rn,1e-6):>6.1f}x")

print("\nE2 -- RANDOM noise, LOW-Q (a=1.0) vs HIGH-Q (a=0.3). MC:\n")
print(f"{'noise D':>8} | {'LOW-Q MC':>9} | {'HIGH-Q MC':>9}")
print("-" * 32)
for D in (0.0, 0.1, 0.4):
    print(f"{D:>8.2f} | {amc(v=VT, w=WT, alpha=1.0, noise='random', D=D):>9.2f} | "
          f"{amc(v=VT, w=WT, alpha=0.3, noise='random', D=D):>9.2f}")

print("\nE3 -- honest check: does topology change noise rejection? (alpha=0.5, D=0.3) MC:\n")
print(f"{'noise':>12} | {'topological':>11} | {'trivial':>8}")
print("-" * 36)
for nz in ("structured", "random"):
    print(f"{nz:>12} | {amc(v=VT, w=WT, alpha=0.5, noise=nz, D=0.3):>11.2f} | "
          f"{amc(v=VR, w=WR, alpha=0.5, noise=nz, D=0.3):>8.2f}")
print("\n--- read tables: E1 = the headline (readout cancels structured, not random) ---")
