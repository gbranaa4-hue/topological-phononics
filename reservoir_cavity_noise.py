#!/usr/bin/env python3
"""Topological HIGH-Q cavity reservoir: does raising Q buy noise-robustness, and does topology
cover high-Q's tuning fragility? Honest test of the complementarity claim -- INCLUDING the
catch that SSH protection is chiral (protects COUPLING disorder, not ON-SITE frequency drift).

Model: a network of damped oscillators (each a cavity), stiffness K with SSH-dimerized
couplings, damping gamma = w0/Q (high Q = narrowband cavity). Driven by input + process noise.
Linear (recall task). Metric = recall NMSE (lower better).

PRE-REGISTERED
  E1 noise-robustness: at fixed noise, higher Q -> lower NMSE (the cavity averages/filters noise).
  E2 complementarity: at high Q, topology should protect against COUPLING disorder + a defect
     (chiral-protected) but NOT against ON-SITE frequency disorder (symmetry-breaking). If
     topology helps on-site too -> pitch fully holds; if only coupling -> pitch is PARTIAL
     (report the correction honestly).
"""
import numpy as np

N = 8; M = 2 * N; W0 = 1.0; DT = 0.08; STEPS = 6000; BURN = 1000
VT, WT = 0.5, 1.0; VR, WR = 1.0, 0.5

def Kmat(v, w, cdis=0.0, odis=0.0, defect=None, rng=None):
    K = np.zeros((M, M))
    for i in range(M - 1):
        c = (v if i % 2 == 0 else w)
        if cdis and rng is not None: c *= (1 + cdis * 2 * (rng.random() - 0.5))   # COUPLING disorder
        K[i, i + 1] = K[i + 1, i] = -c
    for i in range(M):
        on = W0 ** 2
        if odis and rng is not None: on *= (1 + odis * 2 * (rng.random() - 0.5))   # ON-SITE freq disorder
        K[i, i] = on + sum(-K[i, j] for j in range(M) if j != i)
    if defect is not None:
        for k in np.atleast_1d(defect):
            K[k, :] = 0; K[:, k] = 0; K[k, k] = W0 ** 2
    return K

MACRO, SUB = 1200, 15                                        # input held SUB integration steps per macro-step
def states(v, w, Q, D, cdis=0.0, odis=0.0, defect=None, seed=0):
    rng = np.random.default_rng(seed)
    K = Kmat(v, w, cdis, odis, defect, rng)
    gamma = W0 / Q
    u = rng.standard_normal(MACRO); win = np.full(M, 0.3)
    x = np.zeros(M); xd = np.zeros(M); X = np.zeros((MACRO, M))
    for k in range(MACRO):
        for _ in range(SUB):                                 # hold input u[k], let the cavities respond
            f = -K @ x - gamma * xd + win * u[k] + rng.normal(0, D, M)
            xd = xd + DT * f; x = x + DT * xd
        if not np.all(np.isfinite(x)): return None, None
        X[k] = x
    b = MACRO // 5
    return X[b:], u[b:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

def recall_nmse(**kw):
    X, u = states(**kw)
    if X is None: return 1.5
    y = np.zeros(len(u)); y[2:] = u[:-2]                      # recall u[k-2] (macro-steps)
    n = len(u); tr = int(0.6 * n)
    w = ridge(X[:tr], y[:tr]); return nmse(X[tr:] @ w, y[tr:])

print("E1 -- noise-robustness vs Q (topological, no disorder). recall NMSE, avg 2 seeds:\n")
print(f"{'noise D':>8} | " + " | ".join(f'Q={q:<5}' for q in (3, 40)))
print("-" * 30)
for D in (0.0, 0.1, 0.4):
    row = [np.mean([recall_nmse(v=VT, w=WT, Q=q, D=D, seed=s) for s in range(2)]) for q in (3, 40)]
    print(f"{D:>8.2f} | " + " | ".join(f'{r:5.3f} ' for r in row))

print("\nE2 -- complementarity at HIGH Q=40, noise D=0.1 (avg 3 seeds). NMSE:\n")
print(f"{'perturbation':>22} | {'topological':>11} | {'trivial':>8} | topo helps?")
print("-" * 60)
def cmp(label, **kw):
    t = np.mean([recall_nmse(v=VT, w=WT, Q=40, D=0.1, seed=s, **kw) for s in range(3)])
    r = np.mean([recall_nmse(v=VR, w=WR, Q=40, D=0.1, seed=s, **kw) for s in range(3)])
    print(f"{label:>22} | {t:>11.3f} | {r:>8.3f} | {'YES' if t < r - 0.03 else 'no'}")
    return t, r
cmp("clean (noise only)")
cmp("COUPLING disorder 0.3", cdis=0.3)
cmp("ON-SITE disorder 0.3", odis=0.3)
cmp("one dead cavity", defect=M // 2)
print("\n--- read the raw table: does topology protect coupling+defect but NOT on-site? ---")
