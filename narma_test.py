#!/usr/bin/env python3
"""KEYSTONE missing piece: does topological defect-tolerance survive a genuinely NONLINEAR task,
and does topology beat a GENERIC random reservoir? Task = NARMA10 (the standard nonlinear reservoir
benchmark: 10-step nonlinear memory, product terms). Three reservoirs, matched node count M and
spectral radius: topological SSH (g=+0.6), trivial SSH (g=-0.6), random echo-state network (ESN).

PRE-REGISTERED
  CAPABILITY BAR first: each reservoir must do NARMA10 non-vacuously (clean NMSE well below 1).
    Honest risk: the structured SSH reservoir may be a WEAKER computer than a random ESN -- if so,
    that is a real capability/robustness trade-off, reported not hidden.
  DEFECT: with the capability bar cleared, is the frozen-readout penalty (one dead node) lower for
    topological than trivial ON THE NONLINEAR TASK (does ~5x hold), and how does it compare to the
    random ESN? topo < triv would upgrade the claim from 'protects memory' to 'protects computation';
    topo <= ESN would show topology is a GOOD robustness route, not just generic structure.
"""
import numpy as np
from phononic_methods import ssh_H

T = 3000; BURN = 200; NDEF = 12; NSEED = 3

def narma10(T, seed):
    rng = np.random.default_rng(seed); u = rng.uniform(0.0, 0.5, T); y = np.zeros(T)
    for t in range(9, T - 1):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 9:t + 1]) + 1.5 * u[t - 9] * u[t] + 0.1
    return u, y

def esn_W(M, seed, density=0.2):
    rng = np.random.default_rng(1000 + seed)
    A = rng.standard_normal((M, M)) * (rng.random((M, M)) < density)
    return A, rng.uniform(-1, 1, M)

def run(W, win, u, alpha=0.3, rho=0.95, remove=None):
    W = rho * W / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-9)
    if remove is not None:
        W = W.copy(); W[remove, :] = 0.0; W[:, remove] = 0.0
    M = W.shape[0]; x = np.zeros(M); X = np.zeros((len(u), M))
    for t in range(len(u)):
        x = (1 - alpha) * x + alpha * np.tanh(W @ x + win * u[t]); X[t] = x
    return X

def aug(X): return np.hstack([X, np.ones((len(X), 1))])            # affine readout (bias column)
def ridge(X, y, lam=1e-2):                                         # regularized (matches the linear-task readout)
    Xb = aug(X); return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)
def nmse(p, y): return float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

def reservoir(kind, M, seed):
    if kind == "topological": return ssh_H(M // 2, +0.6), np.full(M, 0.3)
    if kind == "trivial":     return ssh_H(M // 2, -0.6), np.full(M, 0.3)
    return esn_W(M, seed)                                            # random ESN

M = 16; clean = {k: [] for k in ("topological", "trivial", "random-ESN")}
pen = {k: [] for k in clean}
paired = 0; paired_n = 0                                            # topological vs trivial, SAME seed+site
for s in range(NSEED):
    u, y = narma10(T, s); yb = y[BURN:]
    res = {k: reservoir(k, M, s) for k in clean}
    wout = {}
    for k, (A, win) in res.items():
        Xin = run(A, win, u)[BURN:]; wout[k] = ridge(Xin, yb); clean[k].append(nmse(aug(Xin) @ wout[k], yb))
    dr = np.random.default_rng(50 + s)
    for _ in range(NDEF):
        kdef = int(dr.integers(2, M - 2)); p_here = {}
        for k, (A, win) in res.items():
            Xd = run(A, win, u, remove=kdef)[BURN:]
            p_here[k] = nmse(aug(Xd) @ wout[k], yb) - nmse(aug(Xd) @ ridge(Xd, yb), yb)
            pen[k].append(p_here[k])
        paired += (p_here["topological"] < p_here["trivial"]); paired_n += 1

print(f"NARMA10 (nonlinear), M={M}, {NSEED} seeds x {NDEF} placements, regularized readout (lam=1e-2).\n")
print(f"{'reservoir':>12} | {'clean NMSE':>10} | {'MEDIAN defect penalty':>21} | {'mean':>9}")
print("-" * 62)
for k in ("topological", "trivial", "random-ESN"):
    print(f"{k:>12} | {np.mean(clean[k]):>10.3f} | {np.median(pen[k]):>21.3f} | {np.mean(pen[k]):>9.2f}")

print("\n--- read honestly (median = robust to catastrophic-placement tail) ---")
ct, cr, ce = (np.mean(clean[k]) for k in ("topological", "trivial", "random-ESN"))
mt, mr, me = (np.median(pen[k]) for k in ("topological", "trivial", "random-ESN"))
print(f"capability: topo {ct:.3f}, trivial {cr:.3f}, ESN {ce:.3f}  -> "
      f"{'PASS' if max(ct,cr,ce) < 0.6 else 'CHECK (near-vacuous)'}")
print(f"paired topo-vs-trivial win-rate (same seed+site): {100*paired/paired_n:.0f}%  "
      f"(median penalty topo {mt:.3f} vs trivial {mr:.3f})")
print(f"vs generic random ESN median penalty: {me:.3f}")
if paired/paired_n > 0.6 and mt < mr:
    print("NONLINEAR defect-tolerance HOLDS: protects COMPUTATION, not just memory. "
          f"Topology {'beats' if mt < me else 'does NOT beat'} a random ESN.")
else:
    print("HONEST NULL/REVERSAL: the topological defect advantage does NOT clearly transfer to the "
          "nonlinear task -- it may be specific to linear memory. Reported, not hidden.")
