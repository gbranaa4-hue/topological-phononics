#!/usr/bin/env python3
"""RIGOR TEST (v2): does defect-tolerance track the topological invariant -- separating
CAPABILITY from ROBUSTNESS. v1 mistake: measured raw transfer NMSE, but transfer==oracle at
every g, so it was capability vs g, not defect-robustness. Fix: report BOTH
  oracle   = retrained-per-chip NMSE   (CAPABILITY: how well the channel does the task)
  penalty  = transfer - oracle         (ROBUSTNESS: how much a FROZEN readout loses to the defect)
An interior removed resonator barely challenges the edge-tap (penalty ~0 either way), so the
honest robustness test is the FULL-state readout, where the defect actually costs something.

Sweep v=1-g, w=1+g. g<0 trivial (0 edge modes), g>0 topological (2 edge modes).
PRE-REGISTERED: if defect-robustness is topological, the full-state PENALTY should DROP as g
crosses 0 into the topological region (edge modes present). If the penalty is flat / doesn't
track the transition -> the earlier 'topological defect-tolerance' was CAPABILITY, not
robustness; say so and retract.
"""
import numpy as np

N = 20; M = 2 * N; T = 2000; BURN = 200; NTEST = 8
EDGE = list(range(0, M // 8)) + list(range(M - M // 8, M)); FULL = list(range(M))

def ssh_H(v, w, remove=None):
    H = np.zeros((M, M))
    for i in range(M - 1):
        H[i, i + 1] = H[i + 1, i] = (v if i % 2 == 0 else w)
    if remove is not None:
        H[remove, :] = 0; H[:, remove] = 0
    return H

def n_edge_modes(H, tol=0.05, ew=0.5):
    E, V = np.linalg.eigh(H); k = max(2, M // 8)
    return sum(1 for j in range(M) if abs(E[j]) < tol and (V[:, j] ** 2)[:k].sum() + (V[:, j] ** 2)[-k:].sum() > ew)

def states(H, u, rho=0.95):
    W = rho * H / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-9)
    win = np.zeros(M); win[0] = 0.6; win[M // 2] = 0.6
    x = np.zeros(M); X = np.zeros((len(u), M))
    for t in range(len(u)):
        x = np.tanh(W @ x + win * u[t]); X[t] = x
    return X[BURN:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

rng0 = np.random.default_rng(1); u = rng0.standard_normal(T)
y = np.zeros(T); y[2:] = u[:-2]; y = y[BURN:]

def transfer(v, w, cols):
    wout = ridge(states(ssh_H(v, w), u)[:, cols], y)         # train on clean chip, freeze
    t, o = [], []
    for s in range(NTEST):
        rem = int(np.random.default_rng(300 + s).integers(M // 4, 3 * M // 4))
        X = states(ssh_H(v, w, remove=rem), u)[:, cols]
        t.append(nmse(X @ wout, y)); o.append(nmse(X @ ridge(X, y), y))
    return np.mean(t), np.mean(o)

print("sweep v=1-g, w=1+g.  penalty = frozen-readout transfer - retrained oracle (isolates robustness)\n")
print(f"{'g':>6} | {'edge':>4} | {'FULL oracle':>11} {'FULL penalty':>12} | {'EDGE oracle':>11} {'EDGE penalty':>12}")
print("-" * 74)
rows = []
for g in np.round(np.linspace(-0.6, 0.6, 13), 2):
    v, w = 1 - g, 1 + g
    ne = n_edge_modes(ssh_H(v, w))
    ft, fo = transfer(v, w, FULL); et, eo = transfer(v, w, EDGE)
    rows.append((g, ne, fo, ft - fo, eo, et - eo))
    print(f"{g:+6.2f} | {ne:>4} | {fo:11.3f} {ft-fo:+12.3f} | {eo:11.3f} {et-eo:+12.3f}")

print("\n--- verdict (read the PENALTY column, not raw transfer) ---")
fp_triv = np.mean([p for g, ne, fo, p, eo, ep in rows if g <= -0.2])
fp_topo = np.mean([p for g, ne, fo, p, eo, ep in rows if g >= 0.2])
print(f"FULL-state defect penalty  trivial (g<=-0.2) = {fp_triv:+.3f}   topological (g>=0.2) = {fp_topo:+.3f}")
if fp_topo < fp_triv - 0.03:
    print("ROBUSTNESS is topological: the frozen-readout defect penalty is smaller in the "
          "topological phase -- the chain heals around the defect. Claim stands (as robustness).")
else:
    print("NULL on robustness: the defect penalty does NOT drop with topology. The earlier "
          "'defect-tolerance' was CAPABILITY (the edge channel computes better), not robustness. Retract.")
