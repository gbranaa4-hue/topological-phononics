#!/usr/bin/env python3
"""Is the N=4096 uptick (1.72x @ N=2048 -> 1.90x @ N=4096) real, or noise from only 3 seeds?
Re-run both N with more seeds and report mean +/- std / SEM so the jump can actually be judged.
"""
import numpy as np

T = 1500; BURN = 200
V, W = 0.4, 1.0
N_SEEDS = 10

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

print(f"variance check: {N_SEEDS} seeds each at N=2048 and N=4096 (fixed Fibonacci K; random K reseeded)\n")
for N in (2048, 4096):
    fib_K = fib_coupling_K(N)   # the Fibonacci chain itself is deterministic -- only the INPUT seed varies
    fib_scores = np.array([recall_nmse(fib_K, seed=s) for s in range(N_SEEDS)])
    rnd_scores = np.array([recall_nmse(random_degree2_K(N, seed=s), seed=s) for s in range(N_SEEDS)])
    ratios = fib_scores / np.maximum(rnd_scores, 1e-9)
    print(f"N={N}:")
    print(f"  Fibonacci NMSE: mean={fib_scores.mean():.4f} std={fib_scores.std():.4f}  "
          f"[{', '.join(f'{v:.3f}' for v in fib_scores)}]")
    print(f"  random    NMSE: mean={rnd_scores.mean():.4f} std={rnd_scores.std():.4f}  "
          f"[{', '.join(f'{v:.3f}' for v in rnd_scores)}]")
    print(f"  per-seed ratio: mean={ratios.mean():.3f}  std={ratios.std():.3f}  "
          f"SEM={ratios.std()/np.sqrt(N_SEEDS):.3f}  95% CI=[{ratios.mean()-1.96*ratios.std()/np.sqrt(N_SEEDS):.3f}, "
          f"{ratios.mean()+1.96*ratios.std()/np.sqrt(N_SEEDS):.3f}]\n")

print("--- verdict: do the two 95% CIs overlap? If yes, the '1.72->1.90' jump was likely noise")
print("    from too few seeds, not a real scale effect. If no overlap, it's a real jump. ---")
