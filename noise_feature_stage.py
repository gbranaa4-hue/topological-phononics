#!/usr/bin/env python3
"""WHERE does structured noise enter -- and does that decide whether the readout can cancel it?
Follow-up to noise_feature.py, which found: structured noise injected INSIDE the recurrent dynamics
is rejected at low amplitude but REVERSES at high amplitude (a big common-mode drive saturates the
nonlinearity and wrecks the signal representation -- not linearly nullable).

Mechanism hypothesis: the linear readout can cancel structured noise ONLY when it sees it LINEARLY.
  stage='dynamics' : noise enters inside the leak, fed back through tanh(W x) -> nonlinearly mangled
  stage='readout'  : clean dynamics, noise added to the MEASURED states only (never fed back) -> linear

PRE-REGISTERED:
  READOUT-stage STRUCTURED (rank-1 common-mode on the measurements) is cancelled at ALL amplitudes
    -> MC ~flat. READOUT-stage RANDOM (full-rank) still degrades. DYNAMICS-stage structured reverses
    (reproduces noise_feature.py). => rejectability is set by linearity-at-the-readout, not by
    "structured vs random" per se.
"""
import numpy as np

N = 8; M = 2 * N; T = 3500; BURN = 400
VT, WT = 0.4, 1.0

def ssh_H(v, w):
    H = np.zeros((M, M))
    for i in range(M - 1): H[i, i + 1] = H[i + 1, i] = (v if i % 2 == 0 else w)
    return H

def states(v, w, alpha, noise, D, stage, seed=0, rho=0.95):
    rng = np.random.default_rng(seed)
    W = rho * ssh_H(v, w) / (np.max(np.abs(np.linalg.eigvalsh(ssh_H(v, w)))) + 1e-9)
    u = rng.standard_normal(T); win = np.full(M, 0.5)
    s = np.zeros(T)
    for t in range(1, T): s[t] = 0.9 * s[t - 1] + rng.standard_normal()
    s = s / (s.std() + 1e-9)
    x = np.zeros(M); X = np.zeros((T, M))
    for t in range(T):
        pre = np.tanh(W @ x + win * u[t])
        nz = 0.0
        if stage == "dynamics":                                     # noise fed back through dynamics
            if noise == "structured": nz = D * s[t] * np.ones(M)
            elif noise == "random":   nz = D * rng.standard_normal(M)
        x = (1 - alpha) * x + alpha * (pre + nz)
        X[t] = x
    if stage == "readout":                                          # noise only on the measurement
        if noise == "structured": X = X + D * s[:, None] * np.ones((1, M))
        elif noise == "random":   X = X + D * rng.standard_normal((T, M))
    return X[BURN:], u[BURN:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def MC(stage, noise, D, seed=0, dmax=12):
    X, u = states(VT, WT, 0.5, noise, D, stage, seed)
    n = len(u); tr = int(0.6 * n); mc = 0.0
    for k in range(1, dmax + 1):
        t = np.arange(k, n); Xf, y = X[t], u[t - k]; sp = tr - k
        w_ = ridge(Xf[:sp], y[:sp])
        r2 = 1 - np.mean((Xf[sp:] @ w_ - y[sp:]) ** 2) / (np.var(y[sp:]) + 1e-12)
        mc += max(0.0, r2)
    return mc

def amc(**kw): return np.mean([MC(seed=s, **kw) for s in range(2)])

for stage in ("readout", "dynamics"):
    where = "MEASUREMENT only (linear)" if stage == "readout" else "INSIDE dynamics (nonlinear feedback)"
    print(f"\n=== structured noise entering at the {stage.upper()} stage -- {where} ===\n")
    print(f"{'noise D':>8} | {'STRUCTURED MC':>13} | {'RANDOM MC':>9} | structured advantage")
    print("-" * 60)
    for D in (0.0, 0.1, 0.3, 0.6, 1.0):
        st = amc(stage=stage, noise="structured", D=D)
        rn = amc(stage=stage, noise="random", D=D)
        tag = "  <- cancelled" if (stage == "readout" and st > rn + 0.3) else ""
        print(f"{D:>8.2f} | {st:>13.2f} | {rn:>9.2f} | {st/max(rn,1e-6):>6.1f}x{tag}")

print("\n--- if READOUT-stage structured stays flat/high at all D while DYNAMICS-stage reverses,")
print("    the mechanism is confirmed: readout cancels structure iff it sees it LINEARLY. ---")
