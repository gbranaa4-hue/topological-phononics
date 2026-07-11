#!/usr/bin/env python3
"""STRESS TEST: does the topological defect-tolerance survive MULTIPLE simultaneous dead
resonators, or does the chain fragment and the advantage collapse? Confound-free uniform
drive (from the fixed position sweep). For n = 1..4 random interior removals, measure the
frozen-readout penalty (transfer NMSE - oracle) for topological vs trivial, averaged over
many random placements.

PRE-REGISTERED: topological penalty < trivial at every n, and the advantage persists (ideally
grows -- more defects = more boundaries, where topology helps). DISCONFIRM: advantage
collapses as n grows (fragmentation destroys the protected structure) -> honest limit.
"""
import numpy as np

N = 20; M = 2 * N; T = 2000; BURN = 200; R = 25
VT, WT = 0.4, 1.0; VR, WR = 1.0, 0.4

def ssh_H(v, w, remove=None):
    H = np.zeros((M, M))
    for i in range(M - 1):
        H[i, i + 1] = H[i + 1, i] = (v if i % 2 == 0 else w)
    if remove is not None:
        for k in np.atleast_1d(remove):
            H[k, :] = 0; H[:, k] = 0
    return H

def states(H, u, rho=0.95):
    W = rho * H / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-9)
    win = np.full(M, 0.12)                                  # uniform drive (confound-free)
    x = np.zeros(M); X = np.zeros((len(u), M))
    for t in range(len(u)):
        x = np.tanh(W @ x + win * u[t]); X[t] = x
    return X[BURN:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

rng0 = np.random.default_rng(1); u = rng0.standard_normal(T)
y = np.zeros(T); y[2:] = u[:-2]; y = y[BURN:]

woutT = ridge(states(ssh_H(VT, WT), u), y)                 # frozen readouts (trained on clean chains)
woutR = ridge(states(ssh_H(VR, WR), u), y)
cand = np.arange(2, M - 2)

def pen(v, w, wout, rem):
    Xd = states(ssh_H(v, w, remove=list(rem)), u)
    return nmse(Xd @ wout, y) - nmse(Xd @ ridge(Xd, y), y)

print(f"multiple-defect stress test ({R} random placements per n; penalty = frozen transfer - oracle)\n")
print(f"{'#defects':>8} | {'TOPO penalty':>13} | {'TRIV penalty':>13} | {'advantage':>9} | topo-wins")
print("-" * 62)
for n in (1, 2, 3, 4):
    rng = np.random.default_rng(10 + n)
    pT, pR, wins = [], [], 0
    for _ in range(R):
        rem = rng.choice(cand, n, replace=False)
        a = pen(VT, WT, woutT, rem); b = pen(VR, WR, woutR, rem)
        pT.append(a); pR.append(b); wins += (a < b)
    mT, mR = np.mean(pT), np.mean(pR)
    print(f"{n:>8} | {mT:>13.3f} | {mR:>13.3f} | {mR/max(mT,1e-6):>8.1f}x | {int(wins/R*100):>3d}%")

print("\n--- verdict ---")
print("If topological penalty stays below trivial and the advantage does not collapse as n grows,")
print("graceful degradation is robust to multiple dead elements -- the write-up sentence.")
