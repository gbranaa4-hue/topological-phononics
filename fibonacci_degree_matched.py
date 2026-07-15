#!/usr/bin/env python3
"""Degree-matched control for fibonacci_scaling_narma.py's Extension 1. The widening gap there
was confounded: density=0.15 random gets MORE neighbors/node as N grows, while the Fibonacci/SSH
chain stays fixed at ~2 neighbors/node (nearest-neighbor only). This isolates the RULE from the
DEGREE by comparing against a random graph with the SAME fixed degree (~2 edges/node) at every N
-- a random graph built from a random 1D permutation (still degree~2, just non-Fibonacci order).

PRE-REGISTERED: with degree matched, the gap should be much smaller and roughly FLAT across N
(not widening) -- if the earlier widening trend was purely the degree confound, this control
should show it. Report whatever it actually shows.
"""
import numpy as np

T = 1500; BURN = 200
V, W = 0.4, 1.0

def fib_word(g):
    w = "A"
    for _ in range(g):
        w = w.replace("B", "0").replace("A", "AB").replace("0", "A")
    return w

def fib_gen_for_length(n_min):
    g = 0
    while len(fib_word(g)) < n_min:
        g += 1
    return g

def fib_coupling_K(n, v=V, w=W, rho=0.95):
    word = fib_word(fib_gen_for_length(n - 1))[: n - 1]
    K = np.zeros((n, n))
    for i, sym in enumerate(word):
        c = v if sym == "A" else w
        K[i, i + 1] = K[i + 1, i] = c
    return rho * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def random_degree2_K(n, seed=0, rho=0.95):
    """Same degree as the chain (~2 edges/node, a random PERMUTATION chain instead of a
    Fibonacci-ordered one) -- isolates the ORDERING RULE from the CONNECTION COUNT."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)                      # random node ORDER, still a simple chain
    vals = rng.choice([V, W], size=n - 1)           # random coupling values, same v/w vocabulary
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

print("DEGREE-MATCHED control: Fibonacci-order chain vs RANDOM-order chain, same degree (~2/node)\n")
print(f"{'N':>5} | {'Fibonacci':>10} | {'random-order chain':>19} | {'ratio rand/fib':>15}")
print("-" * 58)
rows = []
for N in (16, 32, 64, 128, 256, 512):
    fib = np.mean([recall_nmse(fib_coupling_K(N), seed=s) for s in range(3)])
    rnd = np.mean([recall_nmse(random_degree2_K(N, seed=s), seed=s) for s in range(3)])
    ratio = fib / max(rnd, 1e-9)
    rows.append((N, fib, rnd, ratio))
    print(f"{N:>5} | {fib:>10.3f} | {rnd:>19.3f} | {ratio:>14.2f}x")

print("\n--- verdict ---")
ratios = [r[3] for r in rows]
spread = max(ratios) - min(ratios)
trend = ratios[-1] - ratios[0]
print(f"ratio range across N: {min(ratios):.2f}x - {max(ratios):.2f}x  (spread={spread:.2f}, "
      f"first->last trend={trend:+.2f})")
if spread < 0.5:
    print("FLAT with degree matched -- the earlier widening trend WAS the degree confound, not the")
    print("Fibonacci rule itself. The rule vs random-order-at-same-degree gap is small and stable.")
else:
    print("STILL widens/shrinks even degree-matched -- the earlier trend was NOT purely a degree")
    print("confound; the Fibonacci ORDERING itself interacts with scale. Report this, it's real.")
