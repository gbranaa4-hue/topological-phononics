#!/usr/bin/env python3
"""Push fibonacci_degree_matched.py further: does the residual Fibonacci-vs-random-order gap
(1.43x at N=16 -> 1.76x at N=512, degree-matched) keep growing at N=1024/2048/4096, or saturate?
Same construction as fibonacci_degree_matched.py, extended.
"""
import time
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

print("PUSHING FURTHER: degree-matched Fibonacci vs random-order chain, N up to 4096\n")
print("(prior result: 1.43x @ N=16 -> 1.58x @ N=64 -> 1.76x @ N=512 -- does it keep climbing?)\n")
print(f"{'N':>5} | {'Fibonacci':>10} | {'random-order chain':>19} | {'ratio rand/fib':>15} | {'sec':>6}")
print("-" * 68)
rows = []
for N in (512, 1024, 2048, 4096):
    t0 = time.time()
    fib = np.mean([recall_nmse(fib_coupling_K(N), seed=s) for s in range(3)])
    rnd = np.mean([recall_nmse(random_degree2_K(N, seed=s), seed=s) for s in range(3)])
    ratio = fib / max(rnd, 1e-9)
    dt = time.time() - t0
    rows.append((N, fib, rnd, ratio))
    print(f"{N:>5} | {fib:>10.3f} | {rnd:>19.3f} | {ratio:>14.2f}x | {dt:>6.1f}")

print("\n--- verdict ---")
ratios = [r[3] for r in rows]
print(f"trajectory: {' -> '.join(f'{r:.2f}x' for r in ratios)}")
diffs = [ratios[i+1] - ratios[i] for i in range(len(ratios)-1)]
print(f"step-to-step change: {[f'{d:+.2f}' for d in diffs]}")
if all(d < 0.03 for d in diffs[-2:]):
    print("SATURATES -- the residual gap flattens out at large N, does not grow without bound.")
elif all(d > 0 for d in diffs):
    print("KEEPS CLIMBING through N=4096 -- the residual scale-dependence has not saturated yet.")
else:
    print("non-monotone -- read the raw trajectory above, don't trust a one-line summary.")
