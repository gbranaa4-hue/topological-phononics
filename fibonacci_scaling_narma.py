#!/usr/bin/env python3
"""Extends fibonacci_connectivity.py: does the Fibonacci-vs-random capability gap narrow or
widen with N, and does it hold on a nonlinear task (NARMA10)? Same reservoir machinery as
fibonacci_connectivity.py (tanh reservoir, ridge readout) for a clean, apples-to-apples check.
"""
import numpy as np

T = 1500; BURN = 200
V, W = 0.4, 1.0; DENSITY = 0.15

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

def ssh_periodic_K(n, v=V, w=W, rho=0.95):
    K = np.zeros((n, n))
    for i in range(n - 1):
        K[i, i + 1] = K[i + 1, i] = (v if i % 2 == 0 else w)
    return rho * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def random_K(n, density=DENSITY, seed=0, rho=0.95):
    rng = np.random.default_rng(seed)
    A = rng.normal(0, 1, (n, n))
    mask = rng.random((n, n)) < density
    mask = np.triu(mask, 1); A = A * mask; A = A + A.T
    return rho * A / (np.max(np.abs(np.linalg.eigvalsh(A))) + 1e-9)

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

# ---------------------------------------------------------------------------------
print("EXTENSION 1 -- does the fib-vs-random gap narrow or widen with N? (recall task, 3 seeds)\n")
print(f"{'N':>5} | {'Fibonacci':>10} | {'SSH periodic':>13} | {'random':>10} | {'ratio rand/fib':>15}")
print("-" * 65)
for N in (16, 32, 64, 128, 256, 512):
    fib = np.mean([recall_nmse(fib_coupling_K(N), seed=s) for s in range(3)])
    ssh = np.mean([recall_nmse(ssh_periodic_K(N), seed=s) for s in range(3)])
    rnd = np.mean([recall_nmse(random_K(N), seed=s) for s in range(3)])
    print(f"{N:>5} | {fib:>10.3f} | {ssh:>13.3f} | {rnd:>10.3f} | {fib/max(rnd,1e-9):>14.2f}x")

# ---------------------------------------------------------------------------------
print("\nEXTENSION 2 -- NARMA10 (nonlinear task), N=64, 3 seeds\n")
print("PRE-REGISTERED: same qualitative pattern as the linear task -- Fibonacci/SSH clear a")
print("non-vacuous bar, random computes better; report the actual gap, don't assume it matches.\n")
N = 64
results = {}
for label, K in (
    ("Fibonacci (3 params)", fib_coupling_K(N)),
    ("SSH periodic (3 params)", ssh_periodic_K(N)),
    (f"random ({int(round(N*N*DENSITY))} params)", random_K(N)),
):
    scores = [narma_nmse(K, seed=s) for s in range(3)]
    results[label] = np.mean(scores)
    print(f"  {label:<26} NARMA10 NMSE = {np.mean(scores):.3f}  (seeds: {['%.3f'%s for s in scores]})")

fib_n = results["Fibonacci (3 params)"]; rnd_n = results[f"random ({int(round(N*N*DENSITY))} params)"]
print(f"\n--- read: linear-task gap was {0.204/0.092:.2f}x (from the prior run); NARMA10 gap is "
      f"{fib_n/max(rnd_n,1e-9):.2f}x ---")
if max(results.values()) < 0.7:
    print("all three clear the nonlinear capability bar -- the efficiency claim holds on a harder task too.")
else:
    print("at least one is vacuous on NARMA10 -- read the table, don't assume the linear-task result transfers.")
