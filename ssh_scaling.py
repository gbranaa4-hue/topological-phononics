#!/usr/bin/env python3
"""SCALING (tightened): does the topological defect-tolerance advantage grow/hold/shrink with
chain length? Confound-free uniform drive, full-state frozen-readout penalty, recall task.
Per N: 16 random single-defect placements x 2 input seeds = 32 samples. Report mean penalties,
the advantage ratio, and the WIN-RATE (bounded, robust to small-penalty blow-ups).

PRE-REGISTERED: absent/weak below a minimum size (no bulk), then win-rate high and advantage
non-diminishing as N grows -> a BULK topological property, not a small-system fluke.
"""
import numpy as np

T = 1200; BURN = 200; NPLACE = 16; NSEED = 2
VT, WT = 0.4, 1.0; VR, WR = 1.0, 0.4

def ssh_H(N, v, w, remove=None):
    M = 2 * N; H = np.zeros((M, M))
    for i in range(M - 1):
        H[i, i + 1] = H[i + 1, i] = (v if i % 2 == 0 else w)
    if remove is not None:
        for k in np.atleast_1d(remove): H[k, :] = 0; H[:, k] = 0
    return H

def states(N, v, w, u, remove=None, rho=0.95):
    M = 2 * N; H = ssh_H(N, v, w, remove)
    W = rho * H / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-9)
    win = np.full(M, 0.12); x = np.zeros(M); X = np.zeros((len(u), M))
    for t in range(len(u)):
        x = np.tanh(W @ x + win * u[t]); X[t] = x
    return X[BURN:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

def penalty(X, wout, yy):
    return nmse(X @ wout, yy) - nmse(X @ ridge(X, yy), yy)

print(f"tightened scaling: {NPLACE} defect placements x {NSEED} seeds per N\n")
print(f"{'N':>3} | {'nodes':>5} | {'TOPO pen (mean±sd)':>20} | {'TRIV pen (mean±sd)':>20} | {'adv':>5} | {'topo win%':>8}")
print("-" * 78)
rows = []
for N in (4, 6, 8, 12, 16, 24, 32):
    M = 2 * N; pT, pR, wins = [], [], 0
    for s in range(NSEED):
        rng = np.random.default_rng(s)
        u = rng.standard_normal(T); y = np.zeros(T); y[2:] = u[:-2]; y = y[BURN:]
        wT = ridge(states(N, VT, WT, u), y); wR = ridge(states(N, VR, WR, u), y)
        dr = np.random.default_rng(100 + s + N)
        for _ in range(NPLACE):
            k = int(dr.integers(2, M - 2))
            a = penalty(states(N, VT, WT, u, remove=k), wT, y)
            b = penalty(states(N, VR, WR, u, remove=k), wR, y)
            pT.append(a); pR.append(b); wins += (a < b)
    pT, pR = np.array(pT), np.array(pR); adv = pR.mean() / max(pT.mean(), 1e-6)
    wr = int(100 * wins / len(pT))
    rows.append((N, pT.mean(), pR.mean(), adv, wr))
    print(f"{N:>3} | {M:>5} | {pT.mean():>9.2f} ± {pT.std():>7.2f} | {pR.mean():>9.2f} ± {pR.std():>7.2f} | "
          f"{adv:>4.1f}x | {wr:>7}%")

print("\n--- verdict ---")
big = [r for r in rows if r[0] >= 8]
wr_big = np.mean([r[4] for r in big]); adv_big = np.mean([r[3] for r in big])
small_wr = rows[0][4]
print(f"N=4 (no bulk): topo win {small_wr}%.  N>=8: mean win-rate {wr_big:.0f}%, mean advantage {adv_big:.1f}x, "
      f"advantage at N=32 = {rows[-1][3]:.1f}x.")
if small_wr < 60 and wr_big > 75 and rows[-1][3] >= rows[2][3] * 0.8:
    print("BULK PROPERTY: weak with no bulk (N=4), then robustly present and NON-DIMINISHING for N>=8. "
          "Not a small-system fluke.")
else:
    print("Read the table -- trend not as clean as pre-registered.")
