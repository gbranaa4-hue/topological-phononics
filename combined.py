#!/usr/bin/env python3
"""Piece 3: do the two robustness mechanisms COMPOSE?
  A) with a DEAD resonator present, does the trained readout still CANCEL structured noise? (MC metric)
  B) does the topological defect-tolerance (frozen-readout penalty, topo vs trivial) survive added noise?
"""
import numpy as np
from phononic_methods import reservoir_states

N = 8; M = 2 * N; T = 2500; BURN = 300; SITE = M // 2
ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

def mc(g, remove=None, noise=None, D=0.0, rank=1, seed=0, dmax=10):
    X, u = reservoir_states(N, g, T=T, remove=remove, alpha=0.5, noise=noise, D=D, rank=rank,
                            stage="readout", seed=seed)
    X, u = X[BURN:], u[BURN:]; n = len(u); tr = int(0.6 * n); m = 0.0
    for k in range(1, dmax + 1):
        t = np.arange(k, n); Xf, yv = X[t], u[t - k]; sp = tr - k
        w = ridge(Xf[:sp], yv[:sp])
        m += max(0.0, 1 - np.mean((Xf[sp:] @ w - yv[sp:]) ** 2) / (np.var(yv[sp:]) + 1e-12))
    return m
amc = lambda **kw: np.mean([mc(seed=s, **kw) for s in range(3)])

print("=== Piece 3: do defect-tolerance and noise-cancellation COEXIST? ===\n")
print("A) with a DEAD resonator, does the readout still cancel structured noise? (MC, defect at mid-chain)\n")
for g, lab in ((+0.6, "topological"), (-0.6, "trivial")):
    base = amc(g=g, remove=SITE)
    st = amc(g=g, remove=SITE, noise="structured", D=0.6, rank=1)
    rn = amc(g=g, remove=SITE, noise="random", D=0.6)
    print(f"  {lab:12}: defect MC {base:.2f} | +structured {st:.2f} ({st/max(base,1e-9)*100:3.0f}% kept)"
          f" | +random {rn:.2f} ({rn/max(base,1e-9)*100:3.0f}% kept)")

def add_noise(X, frac, seed):
    rng = np.random.default_rng(9000 + seed); return X + frac * X.std() * rng.standard_normal(X.shape)
def penalty(g, frac, seed=0):
    Xin, u = reservoir_states(N, g, T=T, win_scale=0.12, seed=seed); Xin = Xin[BURN:]
    y = np.zeros(T); y[2:] = u[:-2]; yb = y[BURN:]; w = ridge(Xin, yb)
    Xd, _ = reservoir_states(N, g, T=T, win_scale=0.12, remove=SITE, seed=seed); Xd = add_noise(Xd[BURN:], frac, seed)
    return nmse(Xd @ w, yb) - nmse(Xd @ ridge(Xd, yb), yb)

print("\nB) does topological defect-tolerance survive added random noise? (frozen-readout recall penalty)\n")
print(f"  {'noise frac':>10} | {'topological':>11} | {'trivial':>8} | ratio")
print("  " + "-" * 42)
for frac in (0.0, 0.2, 0.5):
    pt = np.mean([penalty(+0.6, frac, s) for s in range(3)])
    pr = np.mean([penalty(-0.6, frac, s) for s in range(3)])
    print(f"  {frac:>10.1f} | {pt:>11.2f} | {pr:>8.2f} | {pr/max(pt,1e-6):.1f}x")
print("\n--- A: structured stays high with a defect = cancellation composes. "
      "B: ratio > 1 as noise rises = topo advantage survives noise. ---")
