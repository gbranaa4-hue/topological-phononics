#!/usr/bin/env python3
"""Does the confirmed degree-matched gap-vs-N trend (1.40x @ N=50 -> 1.95x @ N=10000, linear
recall) hold on the NONLINEAR task (NARMA10), which was only checked at a single N=64 earlier
tonight? Same clean prefix-of-one-fixed-word design (no construction-parity confound).
"""
import sys
import time
import numpy as np

def progress(i, total, label, t0):
    filled = int(30 * (i / total))
    bar = "#" * filled + "-" * (30 - filled)
    print(f"[{bar}] {i}/{total}  {label}  ({time.time()-t0:5.1f}s elapsed)", flush=True)

T = 1500; BURN = 200
V, W = 0.4, 1.0
N_SEEDS = 5
LONG_WORD_GEN = 20

def fib_word(g):
    w = "A"
    for _ in range(g):
        w = w.replace("B", "0").replace("A", "AB").replace("0", "A")
    return w

LONG_WORD = fib_word(LONG_WORD_GEN)

def fib_coupling_K_prefix(n, v=V, w=W, rho=0.95):
    word = LONG_WORD[: n - 1]
    K = np.zeros((n, n))
    for i, sym in enumerate(word):
        c = v if sym == "A" else w
        K[i, i + 1] = K[i + 1, i] = c
    return rho * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

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

def narma10(seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, T); y = np.zeros(T)
    for t in range(9, T - 1):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 9:t + 1]) + 1.5 * u[t - 9] * u[t] + 0.1
    return u, y

def narma_nmse(K, seed=0):
    u, y = narma10(seed); yb = y[BURN:]
    X = reservoir_states(K, u)[BURN:]
    return nmse(X @ ridge(X, yb), yb)

print(f"NARMA10 scaling: clean prefix sweep (fixed word, generation {LONG_WORD_GEN}), {N_SEEDS} seeds\n")
print(f"{'N':>6} | {'Fibonacci':>10} | {'random-order':>13} | {'ratio':>8}")
print("-" * 48)
Ns = [50, 100, 200, 400, 600, 900, 1300, 1800, 2400, 3100, 4000, 5000, 6500]
rows = []
t0 = time.time()
for i, n in enumerate(Ns):
    progress(i, len(Ns), f"starting N={n}", t0)
    fib_K = fib_coupling_K_prefix(n)
    fib_scores = np.array([narma_nmse(fib_K, seed=s) for s in range(N_SEEDS)])
    rnd_scores = np.array([narma_nmse(random_degree2_K(n, seed=s), seed=s) for s in range(N_SEEDS)])
    ratio = fib_scores.mean() / max(rnd_scores.mean(), 1e-9)
    rows.append((n, ratio))
    print(f"{n:>6} | {fib_scores.mean():>10.4f} | {rnd_scores.mean():>13.4f} | {ratio:>7.3f}x", flush=True)
progress(len(Ns), len(Ns), "done", t0)

print("\n--- verdict ---")
ratios = [r[1] for r in rows]
print(f"trajectory: {' -> '.join(f'{r:.2f}x' for r in ratios)}")
print(f"NARMA10 range: {min(ratios):.2f}x - {max(ratios):.2f}x   "
      f"(compare linear-recall range: 1.40x - 1.95x)")
