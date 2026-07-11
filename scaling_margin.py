#!/usr/bin/env python3
"""Verify the ONE untested lever: does the readout's structured-noise cancellation MARGIN scale
with node count M? rank_margin_aware.py showed plain ridge is already at the linear ceiling at
M=16 (knee ~ rank M/2). Claim: spare dimensions = margin, so margin grows ~linearly with M.

Metric = RETENTION = MC(noise) / MC(clean at same M), to divide out raw-capacity differences.
Two views, readout-stage structured noise, plain ridge (already-maximal), D=0.5:
  ABSOLUTE : fixed 4 interferer directions      -> retention should RISE toward 100% as M grows
  RELATIVE : rank = M/2 interferer directions   -> retention should be ~CONSTANT if margin ~ M

PRE-REGISTERED: ABS retention increases with M; REL retention ~flat across M => margin scales
linearly with node count (more nodes = proportionally more correlated interferers absorbed).
"""
import numpy as np

T = 3500; BURN = 400; D = 0.5; VT, WT = 0.4, 1.0

def ssh_H(M, v, w):
    H = np.zeros((M, M))
    for i in range(M - 1): H[i, i + 1] = H[i + 1, i] = (v if i % 2 == 0 else w)
    return H

def states(M, rank, seed=0, alpha=0.5, rho=0.95):
    rng = np.random.default_rng(seed)
    W = rho * ssh_H(M, VT, WT) / (np.max(np.abs(np.linalg.eigvalsh(ssh_H(M, VT, WT)))) + 1e-9)
    u = rng.standard_normal(T); win = np.full(M, 0.5)
    x = np.zeros(M); X = np.zeros((T, M))
    for t in range(T):
        x = (1 - alpha) * x + alpha * np.tanh(W @ x + win * u[t]); X[t] = x
    if rank > 0:
        V, _ = np.linalg.qr(rng.standard_normal((M, rank))); V = V[:, :rank]
        C = np.zeros((T, rank))
        for i in range(rank):
            c = np.zeros(T)
            for t in range(1, T): c[t] = 0.9 * c[t - 1] + rng.standard_normal()
            C[:, i] = c / (c.std() + 1e-9)
        X = X + (D / np.sqrt(rank)) * (C @ V.T)
    return X[BURN:], u[BURN:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def MC(M, rank, seed=0, dmax=12):
    X, u = states(M, rank, seed)
    n = len(u); tr = int(0.6 * n); mc = 0.0
    for k in range(1, dmax + 1):
        t = np.arange(k, n); Xf, y = X[t], u[t - k]; sp = tr - k
        w_ = ridge(Xf[:sp], y[:sp])
        mc += max(0.0, 1 - np.mean((Xf[sp:] @ w_ - y[sp:]) ** 2) / (np.var(y[sp:]) + 1e-12))
    return mc

am = lambda M, rank: np.mean([MC(M, rank, s) for s in range(2)])

print(f"cancellation margin vs node count. structured noise D={D}, plain ridge (already-maximal).\n")
print(f"{'N':>3} | {'M':>3} | {'clean MC':>8} | {'ret @ 4 interferers':>19} | {'ret @ M/2 interferers':>21}")
print("-" * 66)
rows = []
for N in (4, 8, 12, 16, 24):
    M = 2 * N
    clean = am(M, 0)
    abs_ret = am(M, 4) / max(clean, 1e-9)
    rel_ret = am(M, M // 2) / max(clean, 1e-9)
    rows.append((M, abs_ret, rel_ret))
    print(f"{N:>3} | {M:>3} | {clean:>8.2f} | {'':4}{abs_ret*100:5.0f}% (4/{M}){'':4} | {'':5}{rel_ret*100:5.0f}% ({M//2}/{M}){'':4}")

print("\n--- verdict ---")
abs_trend = rows[-1][1] - rows[0][1]
rel_vals = [r[2] for r in rows]
rel_spread = max(rel_vals) - min(rel_vals)
print(f"ABS (fixed 4 interferers): retention {rows[0][1]*100:.0f}% -> {rows[-1][1]*100:.0f}% as M grows "
      f"({'RISES' if abs_trend > 0.05 else 'flat'}).")
print(f"REL (rank M/2): retention spread across M = {rel_spread*100:.0f} pts "
      f"({'~CONSTANT' if rel_spread < 0.12 else 'not flat'}).")
if abs_trend > 0.05 and rel_spread < 0.12:
    print("VERIFIED: margin scales ~linearly with node count -- more nodes absorb proportionally "
          "more correlated interferers; a fixed few get easier. The scaling lever is real.")
else:
    print("Read the table -- scaling not as clean as pre-registered.")
