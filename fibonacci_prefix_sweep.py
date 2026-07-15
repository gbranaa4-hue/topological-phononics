#!/usr/bin/env python3
"""Corrected design: earlier tests conflated 'generation index g' with 'chain length N', so
every N used a DIFFERENT, idiosyncratically-terminated word construction -- confounding true
N-dependence with which generation's construction happened to be used. Fibonacci words nest as
clean prefixes (verified: fib_word(g) is a prefix of fib_word(20) for all g<20). This sweeps N
as arbitrary PREFIX LENGTHS of ONE fixed long word -- the clean version of the same question.
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

LONG_WORD = fib_word(LONG_WORD_GEN)   # length 17711 -- one fixed sequence for every N below

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

def recall_nmse(K, seed=0, delay=2):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(T)
    y = np.zeros(T); y[delay:] = u[:-delay]; yb = y[BURN:]
    X = reservoir_states(K, u)[BURN:]
    return nmse(X @ ridge(X, yb), yb)

print(f"CLEAN prefix sweep: all N use prefixes of ONE fixed word (generation {LONG_WORD_GEN}), "
      f"{N_SEEDS} seeds\n")
print(f"{'N':>6} | {'last symbol':>11} | {'Fibonacci':>10} | {'random-order':>13} | {'ratio':>8}")
print("-" * 62)
Ns = [50, 100, 200, 400, 600, 900, 1300, 1800, 2400, 3100, 4000, 5000, 6500, 8000, 10000]
rows = []
t0 = time.time()
for i, n in enumerate(Ns):
    progress(i, len(Ns), f"starting N={n}", t0)
    end_sym = LONG_WORD[n - 2]   # the symbol occupying the last BOND (index n-2 in the word)
    fib_K = fib_coupling_K_prefix(n)
    fib_scores = np.array([recall_nmse(fib_K, seed=s) for s in range(N_SEEDS)])
    rnd_scores = np.array([recall_nmse(random_degree2_K(n, seed=s), seed=s) for s in range(N_SEEDS)])
    ratio = fib_scores.mean() / max(rnd_scores.mean(), 1e-9)
    rows.append((n, end_sym, ratio))
    print(f"{n:>6} | {end_sym:>11} | {fib_scores.mean():>10.4f} | {rnd_scores.mean():>13.4f} | {ratio:>7.3f}x", flush=True)
progress(len(Ns), len(Ns), "done", t0)

print("\n--- verdict ---")
ratios = [r[2] for r in rows]
diffs = [ratios[i+1] - ratios[i] for i in range(len(ratios) - 1)]
n_reversals = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
print(f"trajectory: {' -> '.join(f'{r:.2f}x' for r in ratios)}")
print(f"step diffs: {[f'{d:+.2f}' for d in diffs]}  ({n_reversals} sign reversals out of {len(diffs)-1} pairs)")
if n_reversals <= 2:
    print("SMOOTH with the construction artifact removed -- the earlier wobble WAS a generation-")
    print("construction confound, not a true property of the quasiperiodic sequence itself.")
else:
    print("STILL wobbles even with a single fixed underlying sequence -- this is now a genuine,")
    print("confirmed property of prefix length within the true Fibonacci sequence, not an artifact.")
