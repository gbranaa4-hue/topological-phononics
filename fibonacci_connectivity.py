#!/usr/bin/env python3
"""Generation-efficiency test: can a recursive self-similar rule (a genuine quasicrystal
substitution -- the 1D Fibonacci word, the same inflation-rule family as Penrose tilings)
specify a reservoir's connectivity with a SHORT description, instead of storing a full random
matrix -- and does the resulting reservoir still compute?

This is NOT "does self-similar structure compute better than random" -- that class of question
was already tested tonight (ssh topology vs random connectivity, narma_redundancy.py) and lost:
random redundancy beat structured topology on defect-tolerance. This is a DIFFERENT axis:
description length / regenerability, independent of whether performance is better.

THE RULE: Fibonacci substitution, sigma: A->AB, B->A. Starting from "A" and iterating g times
produces an aperiodic word of length F(g+2) (a Fibonacci number) -- the same combinatorial
object underlying 1D quasicrystals and Penrose-tiling inflation. Map each symbol to a coupling
strength (A->v, B->w) along a 1D chain, exactly generalizing tonight's SSH period-2 alternation
(v,w,v,w,...) to a genuine aperiodic self-similar sequence.

PART A (deterministic, no randomness): parameter count needed to SPECIFY connectivity of an
  N-node chain -- full random sparse matrix (must store every weight) vs the Fibonacci rule
  (store only: generation depth g, values v and w -- 3 numbers, regenerates any N).
PART B (measured, pre-registered): does a Fibonacci-coupled reservoir clear the SAME capability
  bar (linear recall, matching tonight's metric) as a matched-density random reservoir and the
  standard SSH-periodic reservoir? PREDICTION: comparable, non-vacuous performance (not
  necessarily better) -- if it can't compute, the efficiency property is worthless regardless
  of how cheap it is to generate. Report whatever the data says.
"""
import numpy as np

# ---------------------------------------------------------------------------------
# PART A: description-length scaling (exact, no randomness needed)
# ---------------------------------------------------------------------------------
def fib_word(g):
    """Fibonacci substitution sigma: A->AB, B->A, starting from 'A', g generations."""
    w = "A"
    for _ in range(g):
        w = w.replace("B", "0").replace("A", "AB").replace("0", "A")
    return w

def fib_gen_for_length(n_min):
    """Smallest generation g such that the Fibonacci word has length >= n_min."""
    g = 0
    while len(fib_word(g)) < n_min:
        g += 1
    return g

print("PART A -- parameter count to SPECIFY an N-node chain's connectivity\n")
print(f"{'N':>6} | {'random sparse (density~0.15)':>30} | {'Fibonacci rule (g, v, w)':>26}")
print("-" * 70)
DENSITY = 0.15
for N in (8, 16, 32, 64, 128, 256, 512, 1024, 4096):
    random_params = int(round(N * N * DENSITY))          # must store every nonzero weight
    g = fib_gen_for_length(N)
    fib_params = 3                                          # (generation depth g, v, w) -- CONSTANT
    print(f"{N:>6} | {random_params:>30,} | {fib_params:>5} numbers (g={g:>2}, regenerates any N)")

print("\n--- read: random storage grows as N^2; the Fibonacci rule's parameter count is CONSTANT")
print("    (only the generation depth index g grows, and only as O(log N)) ---\n")


# ---------------------------------------------------------------------------------
# PART B: capability bar -- does the efficiently-generated reservoir still compute?
# ---------------------------------------------------------------------------------
N = 64            # chain length (nodes), matches the scaling range already validated tonight
T = 1500; BURN = 200
V, W = 0.4, 1.0    # same coupling magnitudes used throughout tonight's SSH work

def fib_coupling_K(n, v=V, w=W, rho=0.95):
    """Chain coupling from the Fibonacci word: symbol A -> coupling v, symbol B -> coupling w,
    applied to successive BONDS (edges) along the chain -- the direct aperiodic generalization
    of tonight's SSH period-2 alternation (which is the g=0-ish periodic special case)."""
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

def capability(K, seed=0, delay=2):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(T)
    y = np.zeros(T); y[delay:] = u[:-delay]; yb = y[BURN:]
    X = reservoir_states(K, u)[BURN:]
    return nmse(X @ ridge(X, yb), yb)

print(f"PART B -- capability bar (recall u[t-2], N={N}, avg 3 seeds, lower NMSE = better)\n")
print("PRE-REGISTERED: Fibonacci-coupled reservoir clears the same non-vacuous bar (NMSE well")
print("below 1.0) as matched-density random and standard periodic SSH -- comparable, not")
print("necessarily better. If it's vacuous, the efficiency property doesn't matter.\n")

results = {}
for label, K in (
    ("Fibonacci (3 params)", fib_coupling_K(N)),
    ("SSH periodic (3 params)", ssh_periodic_K(N)),
    (f"random ({int(round(N*N*DENSITY))} params)", random_K(N)),
):
    scores = [capability(K, seed=s) for s in range(3)]
    results[label] = np.mean(scores)
    print(f"  {label:<26} NMSE = {np.mean(scores):.3f}  (seeds: {['%.3f'%s for s in scores]})")

print("\n--- verdict ---")
fib_score = results["Fibonacci (3 params)"]
others_ok = all(v < 0.7 for k, v in results.items())
if fib_score < 0.7 and others_ok:
    print(f"Fibonacci-coupled reservoir CLEARS the capability bar (NMSE={fib_score:.3f}, non-vacuous),")
    print("comparable to the other two -- the 3-parameter description is not just cheap, it still works.")
elif fib_score >= 0.7:
    print(f"Fibonacci-coupled reservoir is VACUOUS or near-vacuous (NMSE={fib_score:.3f}) -- the cheap")
    print("description does NOT compute usefully here. Honest negative on the efficiency claim's value.")
else:
    print("Read the table -- one of the controls didn't clear the bar either; check before trusting this.")
