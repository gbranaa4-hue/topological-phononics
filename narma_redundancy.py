#!/usr/bin/env python3
"""Degree-matched control: is the random ESN's defect-tolerance from being RANDOM, or just DENSER
(more redundant) than the sparse 1D SSH chain? Decompose the two axes on NARMA10 (M=16):
  STRUCTURE: SSH-topological / SSH-trivial / random ;  REDUNDANCY: edge count (15=tree ... 60=dense).
The SSH chain is a PATH (15 edges, no redundancy) -- removing an interior node CUTS it. A random
graph with the SAME 15 edges is also a tree (also cuts); adding edges gives redundant paths.

PRE-REGISTERED: if defect penalty tracks EDGE COUNT (falls as redundancy rises) roughly regardless
of structure, and SSH-topological ~ random-15-edges, then the ESN's advantage was REDUNDANCY, not
randomness -- and topology is a modest within-fixed-connectivity effect. If SSH-topological beats
random-15-edges at matched sparsity, topology adds robustness beyond mere connectivity.
"""
import numpy as np
from phononic_methods import ssh_H

T = 3000; BURN = 200; NDEF = 15; NSEED = 3; M = 16

def narma10(T, seed):
    rng = np.random.default_rng(seed); u = rng.uniform(0.0, 0.5, T); y = np.zeros(T)
    for t in range(9, T - 1):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 9:t + 1]) + 1.5 * u[t - 9] * u[t] + 0.1
    return u, y

def random_reservoir(M, n_edges, seed):
    rng = np.random.default_rng(2000 + seed); A = np.zeros((M, M))
    nodes = rng.permutation(M)                                     # random spanning tree -> connected
    for i in range(1, M):
        j = nodes[rng.integers(0, i)]; a, b = nodes[i], j
        w = rng.standard_normal(); A[a, b] = A[b, a] = w
    edges = M - 1
    while edges < n_edges:                                         # add redundant edges (cycles)
        a, b = rng.integers(0, M), rng.integers(0, M)
        if a != b and A[a, b] == 0:
            A[a, b] = A[b, a] = rng.standard_normal(); edges += 1
    return A, rng.uniform(-1, 1, M)

def run(W, win, u, alpha=0.3, rho=0.95, remove=None):
    W = rho * W / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-9)
    if remove is not None:
        W = W.copy(); W[remove, :] = 0.0; W[:, remove] = 0.0
    x = np.zeros(W.shape[0]); X = np.zeros((len(u), W.shape[0]))
    for t in range(len(u)):
        x = (1 - alpha) * x + alpha * np.tanh(W @ x + win * u[t]); X[t] = x
    return X

def aug(X): return np.hstack([X, np.ones((len(X), 1))])
def ridge(X, y, lam=1e-2):
    Xb = aug(X); return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)
def nmse(p, y): return float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

def build(kind, seed):
    if kind == "SSH topological": return ssh_H(M // 2, +0.6), np.full(M, 0.3), 15
    if kind == "SSH trivial":     return ssh_H(M // 2, -0.6), np.full(M, 0.3), 15
    e = int(kind.split()[1]); A, win = random_reservoir(M, e, seed); return A, win, e

kinds = ["SSH topological", "SSH trivial", "random 15", "random 24", "random 40", "random 60"]
print(f"NARMA10, M={M}, {NSEED} seeds x {NDEF} placements, lam=1e-2. Isolating structure vs redundancy.\n")
print(f"{'reservoir':>16} | {'edges':>5} | {'clean NMSE':>10} | {'MEDIAN defect penalty':>21}")
print("-" * 62)
res = {}
for kind in kinds:
    cl, pn, edges = [], [], None
    for s in range(NSEED):
        u, y = narma10(T, s); yb = y[BURN:]
        A, win, edges = build(kind, s)
        Xin = run(A, win, u)[BURN:]; wout = ridge(Xin, yb); cl.append(nmse(aug(Xin) @ wout, yb))
        dr = np.random.default_rng(50 + s)
        for _ in range(NDEF):
            k = int(dr.integers(1, M - 1))
            Xd = run(A, win, u, remove=k)[BURN:]
            pn.append(nmse(aug(Xd) @ wout, yb) - nmse(aug(Xd) @ ridge(Xd, yb), yb))
    res[kind] = (edges, np.mean(cl), np.median(pn))
    print(f"{kind:>16} | {edges:>5} | {np.mean(cl):>10.3f} | {np.median(pn):>21.3f}")

print("\n--- read honestly ---")
rt = res["SSH topological"][2]; r15 = res["random 15"][2]
r60 = res["random 60"][2]; falls = res["random 15"][2] > res["random 40"][2] > res["random 60"][2]
print(f"redundancy axis (random): 15e {res['random 15'][2]:.2f} -> 40e {res['random 40'][2]:.2f} "
      f"-> 60e {res['random 60'][2]:.2f}   ({'penalty FALLS with edges' if falls else 'no clean trend'})")
print(f"structure at matched sparsity: SSH-topological {rt:.2f}  vs  random-15-edges {r15:.2f}")
if falls and abs(rt - r15) < max(rt, r15) * 0.5:
    print("VERDICT: redundancy (edge count) drives defect-tolerance; SSH-topological ~ random-15 at matched")
    print("sparsity -> the ESN's edge was REDUNDANCY, not randomness. Topology = modest within-connectivity effect.")
elif falls and rt < r15:
    print("VERDICT: redundancy helps, BUT SSH-topological beats random at matched sparsity -> topology adds")
    print("robustness beyond mere connectivity.")
else:
    print("Read the table -- trend not as clean as pre-registered.")
