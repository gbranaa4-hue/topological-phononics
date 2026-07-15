#!/usr/bin/env python3
"""Is the non-monotone wobble (1.70x @ N=2048 -> 1.88x @ N=4096, confirmed real, CIs don't
overlap) a TRUE scale effect, or a TRUNCATION-PHASE artifact? Prior N values (16,32,...,4096,
powers of 2) truncate the Fibonacci word mid-generation -- N=2048 lands ~79% through generation
16, N=4096 lands ~98% through generation 17, different phases of the same quasi-periodic pattern.

This uses N = EXACT Fibonacci-word length at each generation (NO truncation at all) -- if the
trajectory is smooth/monotone here, truncation phase was the source of the earlier wobble.
"""
import numpy as np

T = 1500; BURN = 200
V, W = 0.4, 1.0
N_SEEDS = 6

def fib_word(g):
    w = "A"
    for _ in range(g):
        w = w.replace("B", "0").replace("A", "AB").replace("0", "A")
    return w

def fib_coupling_K_exact(g, v=V, w=W, rho=0.95):
    """N = len(fib_word(g)) + 1 nodes, N-1 bonds = the FULL untruncated word -- no boundary cut."""
    word = fib_word(g)
    n = len(word) + 1
    K = np.zeros((n, n))
    for i, sym in enumerate(word):
        c = v if sym == "A" else w
        K[i, i + 1] = K[i + 1, i] = c
    return n, rho * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def random_degree2_K(n, seed=0, rho=0.95):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    vals = rng.choice([V, W], size=n - 1)
    K = np.zeros((n, n))
    for i in range(n - 1):
        a, b = perm[i], perm[i + 1]
        K[a, b] = K[b, a] = vals[i]
    return rho * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def reservoir_states(K, u, win_scale=0.5):
    n = K.shape[0]; win = np.full(n, win_scale); x = np.zeros(n); X = np.zeros((len(u), n))
    for t in range(len(u)):
        x = np.tanh(K @ x + win * u[t]); X[t] = x
    return X

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

def recall_nmse(K, seed=0, delay=2):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(T)
    y = np.zeros(T); y[delay:] = u[:-delay]; yb = y[BURN:]
    X = reservoir_states(K, u)[BURN:]
    return nmse(X @ ridge(X, yb), yb)

print("EXACT Fibonacci-word-length N (no truncation) -- isolating truncation-phase from true scale\n")
print(f"{'gen g':>5} | {'N':>6} | {'Fibonacci':>10} | {'random-order':>13} | {'ratio':>8}")
print("-" * 55)
rows = []
for g in range(6, 17):     # generations giving N from a few dozen up past 4000
    n, fib_K = fib_coupling_K_exact(g)
    fib_scores = np.array([recall_nmse(fib_K, seed=s) for s in range(N_SEEDS)])
    rnd_scores = np.array([recall_nmse(random_degree2_K(n, seed=s), seed=s) for s in range(N_SEEDS)])
    ratio = fib_scores.mean() / max(rnd_scores.mean(), 1e-9)
    rows.append((g, n, fib_scores.mean(), rnd_scores.mean(), ratio))
    print(f"{g:>5} | {n:>6} | {fib_scores.mean():>10.4f} | {rnd_scores.mean():>13.4f} | {ratio:>7.3f}x")

print("\n--- verdict ---")
ratios = [r[4] for r in rows]
diffs = [ratios[i+1] - ratios[i] for i in range(len(ratios) - 1)]
print(f"trajectory: {' -> '.join(f'{r:.2f}x' for r in ratios)}")
print(f"step diffs: {[f'{d:+.2f}' for d in diffs]}")
n_reversals = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
if n_reversals <= 1:
    print(f"SMOOTH/MONOTONE-ish at exact word boundaries ({n_reversals} sign reversal(s)) -- supports")
    print("the truncation-phase-artifact explanation for the earlier wobble.")
else:
    print(f"STILL WOBBLES even at exact boundaries ({n_reversals} sign reversals) -- truncation phase")
    print("was NOT the (sole) explanation; something else drives the non-monotonicity. Report this.")
