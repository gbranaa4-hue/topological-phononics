#!/usr/bin/env python3
"""FIRM UP the one loose end: is the compensated (chiral-preserved) Duffing model's topological
defect-advantage a REAL effect, or noise? The prior run had only ~12 samples (1.1x, 64%). Here:
6 seeds x all 13 interior sites = 78 paired samples per model, with a PRE-REGISTERED decision rule.

PRE-REGISTERED (per the gbranaa4-hue method: pre-register, read the raw numbers, report the null):
  H1  compensated model = REAL topological positive  <=>  paired topo-vs-trivial win-rate 95%-CI
      LOWER BOUND > 50%  AND  median penalty topo < trivial.
  Otherwise report inconclusive/null. Naive (chiral-broken) model expected <= 50% (reversal).
"""
import numpy as np

N = 8; M = 2 * N; T = 900; BURN = 150
WIN = np.random.default_rng(123).uniform(-1, 1, M) * 0.5

def coupling(topo, remove, mode):
    ki, ko = (0.2, 0.8) if topo else (0.8, 0.2); A = np.zeros((M, M))
    for i in range(M - 1): A[i, i + 1] = A[i + 1, i] = (ki if i % 2 == 0 else ko)
    if remove is not None:
        for k in np.atleast_1d(remove): A[k, :] = 0.0; A[:, k] = 0.0
    if mode == "naive": return np.diag(A.sum(1)) - A, 1.0
    return -A, 2.0

def reservoir(u, topo, remove=None, mode="naive", gamma=0.4, beta=0.3, dt=0.05, hold=30):
    K, w0sq = coupling(topo, remove, mode); x = np.zeros(M); v = np.zeros(M); Xs = np.zeros((len(u), 2 * M))
    acc = lambda x, v, f: -gamma * v - w0sq * x - beta * x ** 3 - K @ x + f
    for t in range(len(u)):
        f = WIN * u[t]
        for _ in range(hold):
            a1 = acc(x, v, f); a2 = acc(x + 0.5 * dt * v, v + 0.5 * dt * a1, f)
            a3 = acc(x + 0.5 * dt * (v + 0.5 * dt * a1), v + 0.5 * dt * a2, f)
            a4 = acc(x + dt * (v + 0.5 * dt * a2), v + dt * a3, f)
            x = x + dt * (v + dt / 6 * (a1 + a2 + a3)); v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        Xs[t] = np.concatenate([x, v])
    return Xs

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))
def data(seed):
    u = np.random.default_rng(seed).standard_normal(T); y = np.zeros(T); y[2:] = u[:-2]; return u, y[BURN:]

SEEDS = 6; SITES = list(range(2, M - 1))
print(f"FIRM-UP: {SEEDS} seeds x {len(SITES)} interior sites = {SEEDS*len(SITES)} paired samples per model.")
print("pre-registered: compensated is a REAL positive only if win-rate 95%-CI lower bound > 50%.\n")
print(f"{'model':>12} | {'cap NMSE':>8} | {'topo win-rate (95% CI)':>24} | {'median pen topo/triv':>20} | verdict")
print("-" * 92)
for mode in ("naive", "compensated"):
    u0, yb0 = data(0); Xc = reservoir(u0, True, mode=mode)[BURN:]; cap = nmse(Xc @ ridge(Xc, yb0), yb0)
    pT, pR, wins = [], [], 0
    for s in range(SEEDS):
        u, yb = data(s)
        wT = ridge(reservoir(u, True, mode=mode)[BURN:], yb); wR = ridge(reservoir(u, False, mode=mode)[BURN:], yb)
        for site in SITES:
            XdT = reservoir(u, True, remove=site, mode=mode)[BURN:]; XdR = reservoir(u, False, remove=site, mode=mode)[BURN:]
            a = nmse(XdT @ wT, yb) - nmse(XdT @ ridge(XdT, yb), yb)
            b = nmse(XdR @ wR, yb) - nmse(XdR @ ridge(XdR, yb), yb)
            pT.append(a); pR.append(b); wins += (a < b)
    n = len(pT); wr = wins / n; se = np.sqrt(wr * (1 - wr) / n); lo, hi = wr - 1.96 * se, wr + 1.96 * se
    mT, mR = np.median(pT), np.median(pR)
    verdict = ("REAL positive" if (lo > 0.5 and mT < mR) else
               "reversal" if hi < 0.5 else "inconclusive (CI spans 50%)")
    print(f"{mode:>12} | {cap:>8.3f} | {wr*100:>6.0f}% ({lo*100:>3.0f}-{hi*100:>3.0f}%){'':>6} | "
          f"{mT:>8.2f} / {mR:<8.2f}  | {verdict}")
print("\n--- pre-registered read: compensated REAL positive iff CI lower bound clears 50% ---")
