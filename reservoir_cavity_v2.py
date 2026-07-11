#!/usr/bin/env python3
"""High-Q cavity reservoir, proper metric: linear MEMORY CAPACITY (MC = sum over delays of how
well u[t-k] is recalled) -- doesn't privilege short or long memory, so it's fair across Q.
Leaky-integrator ESN with SSH structure; leak alpha = cavity time constant (small alpha = HIGH
Q = long memory, low-passes noise, since noise enters inside the leak). N=8 cells -> MC<=16.

Questions, pre-registered:
  E0  capability + Q<->memory: MC is substantial at D=0; higher Q shifts memory to longer delays.
  E1  noise: does HIGH-Q retain MC better than LOW-Q as noise grows? (the cavity/noise claim)
  E2  topology: at fixed Q under noise, does topology protect MC vs COUPLING disorder + defect
      (chiral) but NOT vs ON-SITE disorder (symmetry-breaking)? (the honest complementarity test)
"""
import numpy as np

N = 8; M = 2 * N; T = 3500; BURN = 400
VT, WT = 0.5, 1.0; VR, WR = 1.0, 0.5

def ssh_H(v, w, cdis=0.0, odis=0.0, defect=None, rng=None):
    H = np.zeros((M, M))
    for i in range(M - 1):
        c = (v if i % 2 == 0 else w)
        if cdis and rng is not None: c *= (1 + cdis * 2 * (rng.random() - 0.5))
        H[i, i + 1] = H[i + 1, i] = c
    if odis and rng is not None:
        for i in range(M): H[i, i] += odis * 2 * (rng.random() - 0.5)
    if defect is not None:
        for k in np.atleast_1d(defect): H[k, :] = 0; H[:, k] = 0
    return H

def states(v, w, alpha, D, cdis=0.0, odis=0.0, defect=None, seed=0, rho=0.95):
    rng = np.random.default_rng(seed)
    H = ssh_H(v, w, cdis, odis, defect, rng)
    W = rho * H / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-9)
    u = rng.standard_normal(T); win = np.full(M, 0.5)
    x = np.zeros(M); X = np.zeros((T, M))
    for t in range(T):
        pre = np.tanh(W @ x + win * u[t]) + D * rng.standard_normal(M)
        x = (1 - alpha) * x + alpha * pre
        X[t] = x
    return X[BURN:], u[BURN:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def MC(dmax=16, **kw):
    X, u = states(**kw)
    n = len(u); tr = int(0.6 * n); mc = 0.0; per = []
    for k in range(1, dmax + 1):
        t = np.arange(k, n); Xf, y = X[t], u[t - k]; s = tr - k
        w = ridge(Xf[:s], y[:s]); pred = Xf[s:] @ w
        r2 = 1 - np.mean((pred - y[s:]) ** 2) / (np.var(y[s:]) + 1e-12)
        mc += max(0.0, r2); per.append(max(0.0, r2))
    return mc, per

def avg_mc(seeds=2, **kw): return np.mean([MC(**kw, seed=s)[0] for s in range(seeds)])

print("E0 -- capability + Q<->memory (D=0, topological). MC (max 16) and where the memory sits:\n")
for a in (1.0, 0.5, 0.2):
    mc, per = MC(alpha=a, D=0.0, v=VT, w=WT, seed=0)
    peak = int(np.argmax(per)) + 1
    print(f"  alpha={a:4.2f} (Q~{1/a:4.1f})  MC={mc:5.2f}  peak-recall delay={peak}  "
          f"per-delay R2[1..6]={' '.join(f'{p:.2f}' for p in per[:6])}")

print("\nE1 -- noise-robustness: MC vs noise D, LOW-Q (a=1.0) vs HIGH-Q (a=0.3). avg 2 seeds:\n")
print(f"{'noise D':>8} | {'LOW-Q MC':>9} | {'HIGH-Q MC':>9} | who keeps more")
print("-" * 48)
for D in (0.0, 0.05, 0.15, 0.35):
    lo = avg_mc(alpha=1.0, D=D, v=VT, w=WT); hi = avg_mc(alpha=0.3, D=D, v=VT, w=WT)
    print(f"{D:>8.2f} | {lo:>9.2f} | {hi:>9.2f} | {'HIGH-Q' if hi > lo else 'low-Q'}")

print("\nE2 -- topology at a=0.5, noise D=0.1 (avg 3 seeds). MC, higher=better:\n")
print(f"{'perturbation':>22} | {'topological':>11} | {'trivial':>8} | topo helps?")
print("-" * 60)
def cmp(label, **kw):
    t = np.mean([MC(alpha=0.5, D=0.1, v=VT, w=WT, seed=s, **kw)[0] for s in range(3)])
    r = np.mean([MC(alpha=0.5, D=0.1, v=VR, w=WR, seed=s, **kw)[0] for s in range(3)])
    print(f"{label:>22} | {t:>11.2f} | {r:>8.2f} | {'YES' if t > r + 0.3 else 'no'}")
cmp("clean (noise only)")
cmp("COUPLING disorder 0.4", cdis=0.4)
cmp("ON-SITE disorder 0.4", odis=0.4)
cmp("one dead cavity", defect=M // 2)
print("\n--- read the raw tables honestly (E0 capability first; then E1 noise; then E2 topology) ---")
