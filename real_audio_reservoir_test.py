#!/usr/bin/env python3
"""Every reservoir-connectivity comparison run tonight (random vs SSH-topological
vs Fibonacci vs the real de Bruijn quasicrystal) used a SYNTHETIC drive signal --
i.i.d. white noise (rng.standard_normal) or a clean sine tone. Real audio has
strong temporal/spectral correlation that idealized white noise does not --
a genuine, mechanistic reason the "random beats structured" pattern found all
night on synthetic signals might not hold on a real one. This is the first
test tonight using an actual recorded signal (6s of live microphone audio,
real_audio_clip.npy) instead of a synthetic one.

K-generation formulas are the EXACT ones already validated tonight in
debruijn_quasicrystal_reservoir.py (V=0.4, W=1.0, RHO=0.95, same fib_word/
ssh dimerization/random-matched-edges constructions) -- not reconstructed
from a description, copied from the same working night's code.

Task: delay-5 recall on the real recorded signal (same task convention used
in reservoir_from_scratch.py and every fibonacci/quasicrystal script tonight).

PRE-REGISTERED: no confident prior on which architecture wins here -- real
audio's autocorrelation structure is different enough from white noise that
the synthetic-signal finding (random beats structured) might or might not
transfer. Report whichever way it lands, capability bar first.
"""
import numpy as np

V, W = 0.4, 1.0
RHO = 0.95

def ssh_K(N, g):
    v, w = 1.0 - g, 1.0 + g
    K = np.zeros((N, N))
    for i in range(N - 1):
        K[i, i + 1] = K[i + 1, i] = v if i % 2 == 0 else w
    return RHO * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def fib_word(g):
    w = "A"
    for _ in range(g):
        w = w.replace("B", "0").replace("A", "AB").replace("0", "A")
    return w

def fib_K(N, v=V, w=W, rho=RHO):
    word = fib_word(20)[: N - 1]
    K = np.zeros((N, N))
    for i, sym in enumerate(word):
        c = v if sym == "A" else w
        K[i, i + 1] = K[i + 1, i] = c
    return rho * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def random_matched_K(N, n_edges, seed=0, coupling=V, rho=RHO):
    rng = np.random.default_rng(seed)
    edges = set(); attempts = 0
    while len(edges) < n_edges and attempts < n_edges * 50:
        a, b = rng.integers(0, N, size=2)
        if a != b:
            edges.add((min(int(a), int(b)), max(int(a), int(b))))
        attempts += 1
    K = np.zeros((N, N))
    for a, b in edges:
        K[a, b] = K[b, a] = coupling
    return rho * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def reservoir_states(K, u, win_scale=0.5):
    n = K.shape[0]; win = np.full(n, win_scale); x = np.zeros(n); X = np.zeros((len(u), n))
    for t in range(len(u)):
        x = np.tanh(K @ x + win * u[t]); X[t] = x
    return X

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

# --- load REAL recorded audio, not synthetic ---
raw = np.load("real_audio_clip.npy")
T = 20000   # ~0.9s of the 6s clip -- bounded for speed, still real data, not synthetic
u = raw[:T]
u = u / (np.std(u) + 1e-9)   # normalize to unit variance -- raw mic amplitude was
                              # tiny (max abs 0.016), rescaled so the reservoir's tanh
                              # nonlinearity actually operates in a meaningful range
                              # instead of staying in the near-linear regime the whole time
print(f"Real audio clip loaded: T={T} samples, normalized std={np.std(u):.3f}, "
      f"raw max abs (pre-normalize)={np.max(np.abs(raw[:T])):.4f}")

DELAY = 5
y = np.zeros(T); y[DELAY:] = u[:-DELAY]
BURN = 200
split = T - 3000

N = 64
architectures = {
    "random (degree-matched)": lambda: random_matched_K(N, N - 1, seed=0),
    "SSH topological (g=+0.6)": lambda: ssh_K(N, +0.6),
    "SSH trivial (g=-0.6)":     lambda: ssh_K(N, -0.6),
    "Fibonacci":                 lambda: fib_K(N),
}

print(f"\nReal-audio delay-{DELAY} recall, N={N}, T={T}\n")
results = {}
for name, Kfn in architectures.items():
    K = Kfn()
    X = reservoir_states(K, u)
    Xtr, ytr = X[BURN:split], y[BURN:split]
    Xte, yte = X[split:], y[split:]
    w = ridge(Xtr, ytr)
    score = nmse(Xte @ w, yte)
    results[name] = score
    print(f"  {name:<28} NMSE = {score:.4f}")

print("\n--- verdict ---")
caps_ok = all(v < 0.9 for v in results.values())
print(f"capability (all non-vacuous, NMSE<0.9): {'PASS' if caps_ok else 'CHECK -- read raw numbers'}")
if caps_ok:
    best = min(results, key=results.get)
    worst = max(results, key=results.get)
    print(f"best: {best} ({results[best]:.4f})   worst: {worst} ({results[worst]:.4f})   "
          f"ratio worst/best: {results[worst]/max(results[best],1e-9):.3f}x")
    rand_score = results["random (degree-matched)"]
    structured_best = min(results[k] for k in results if k != "random (degree-matched)")
    if structured_best < rand_score * 0.9:
        print("On REAL audio, a structured architecture beats random -- this would be a genuine")
        print("DIVERGENCE from every synthetic-signal test run tonight. Real, worth confirming further.")
    elif rand_score < structured_best * 0.9:
        print("Random still beats every structured architecture -- the synthetic-signal finding")
        print("REPLICATES on real audio too. Report as a real, now-broader-scoped confirmation.")
    else:
        print("No clean separation between random and the best structured architecture on this")
        print("real signal -- honest tie within this test's resolution, report as such.")
