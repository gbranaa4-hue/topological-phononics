#!/usr/bin/env python3
"""Does a REAL 2D quasicrystal (de Bruijn cut-and-project, the actual math behind Penrose
tilings) make a better/worse/different reservoir than a random graph or my 1D Fibonacci chain,
at matched edge count? The de Bruijn generator is copied VERBATIM from
quasicrystal-mems-reservoir/phononic_symmetry_grading_sim.py (gbranaa4-hue's own repo,
debruijn_quasicrystal_points(), read directly from the cloned source -- not reconstructed from
a description) -- that repo already validated this as a real quasiperiodic tiling; it has never
been wired into a reservoir's connectivity before tonight.

Design: take the REAL 2D point positions, build a k-nearest-neighbor graph (k=2, matching the
degree convention used throughout tonight's Fibonacci/SSH work), MEASURE the actual resulting
mean degree (kNN is not reciprocal, so real degree can exceed k -- verify, don't assume), then
build a random graph with the SAME actual edge count for a fair comparison. Compare against the
1D Fibonacci-word chain at matched N.

PRE-REGISTERED: capability bar first (all three non-vacuous). Then report the three-way
comparison honestly -- no prediction about which wins, given tonight's whole pattern has been
"structure doesn't reliably beat random, but description length differs."
"""
import sys
import time
import numpy as np

def progress(i, total, label, t0):
    filled = int(30 * (i / total)) if total else 0
    bar = "#" * filled + "-" * (30 - filled)
    print(f"[{bar}] {i}/{total}  {label}  ({time.time()-t0:5.1f}s elapsed)", flush=True)

# ---------------------------------------------------------------------------------
# debruijn_quasicrystal_points -- copied VERBATIM from gbranaa4-hue/quasicrystal-mems-reservoir
# phononic_symmetry_grading_sim.py (cloned and read directly, not reconstructed)
# ---------------------------------------------------------------------------------
def debruijn_quasicrystal_points(n_fold, window_radius, grid_index_range=8,
                                  offset_seed=0.0, seed=42):
    j_idx = np.arange(n_fold)
    dirs = np.stack([np.cos(2 * np.pi * j_idx / n_fold),
                      np.sin(2 * np.pi * j_idx / n_fold)], axis=1)
    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0.1, 0.9, size=n_fold) + offset_seed
    gammas -= gammas.mean()

    m_range = np.arange(-grid_index_range, grid_index_range + 1)
    points = []
    for j in range(n_fold):
        for k in range(j + 1, n_fold):
            ej, ek = dirs[j], dirs[k]
            det = ej[0] * ek[1] - ej[1] * ek[0]
            if abs(det) < 1e-9:
                continue
            for m in m_range:
                for p in m_range:
                    rhs0 = m + gammas[j]
                    rhs1 = p + gammas[k]
                    x0 = (rhs0 * ek[1] - rhs1 * ej[1]) / det
                    x1 = (ej[0] * rhs1 - ek[0] * rhs0) / det
                    x = np.array([x0, x1])
                    if np.hypot(x0, x1) > window_radius * 1.5:
                        continue
                    idx = np.ceil(x @ dirs.T - gammas).astype(int)
                    vertex = idx @ dirs
                    points.append(vertex)
    if not points:
        return np.zeros((0, 2))
    pts = np.array(points)
    pts = np.round(pts, 6)
    pts = np.unique(pts, axis=0)
    dist = np.hypot(pts[:, 0], pts[:, 1])
    return pts[dist <= window_radius]

# ---------------------------------------------------------------------------------
V, W = 0.4, 1.0; RHO = 0.95
T = 1500; BURN = 200

def knn_edges(points, k=2):
    N = len(points)
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    edges = set()
    for i in range(N):
        for j in np.argsort(d[i])[:k]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return edges

def K_from_edges(N, edges, coupling=V, rho=RHO):
    K = np.zeros((N, N))
    for a, b in edges:
        K[a, b] = K[b, a] = coupling
    return rho * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def random_matched_K(N, n_edges, seed=0, coupling=V, rho=RHO):
    rng = np.random.default_rng(seed)
    edges = set()
    attempts = 0
    while len(edges) < n_edges and attempts < n_edges * 50:
        a, b = rng.integers(0, N, size=2)
        if a != b:
            edges.add((min(int(a), int(b)), max(int(a), int(b))))
        attempts += 1
    return K_from_edges(N, edges, coupling, rho)

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

# ---------------------------------------------------------------------------------
print("Generating REAL de Bruijn quasicrystal point set (n_fold=5, the genuine Penrose symmetry)...\n")
pts = debruijn_quasicrystal_points(n_fold=5, window_radius=6, grid_index_range=6)
N = len(pts)
edges = knn_edges(pts, k=2)
n_edges = len(edges)
degrees = np.zeros(N)
for a, b in edges:
    degrees[a] += 1; degrees[b] += 1
print(f"points: N={N}  |  kNN(k=2) edges: {n_edges}  |  ACTUAL mean degree: {degrees.mean():.2f} "
      f"(measured, not assumed -- kNN is not reciprocal)\n")

qc_K = K_from_edges(N, edges)
fibN_K = fib_K(N)
N_SEEDS = 8
t0 = time.time()

print(f"Three-way comparison, N={N}, edge count matched to {n_edges}, {N_SEEDS} seeds, linear recall\n")
results = {}
labels = ["real quasicrystal (de Bruijn)", "1D Fibonacci chain", f"random (matched, {n_edges} edges)"]
Ks = {
    labels[0]: lambda s: qc_K,
    labels[1]: lambda s: fibN_K,
    labels[2]: lambda s: random_matched_K(N, n_edges, seed=1000 + s),
}
for i, label in enumerate(labels):
    progress(i, len(labels), label, t0)
    scores = [recall_nmse(Ks[label](s), seed=s) for s in range(N_SEEDS)]
    results[label] = (np.mean(scores), np.std(scores))
    print(f"  {label:<32} NMSE = {np.mean(scores):.4f} (std {np.std(scores):.4f})", flush=True)
progress(len(labels), len(labels), "done", t0)

print("\n--- verdict ---")
caps_ok = all(v[0] < 0.9 for v in results.values())
print(f"capability (all non-vacuous, NMSE<0.9): {'PASS' if caps_ok else 'CHECK -- read raw numbers'}")
qc, fb, rd = results[labels[0]][0], results[labels[1]][0], results[labels[2]][0]
print(f"quasicrystal vs random: {qc/max(rd,1e-9):.3f}x   |   quasicrystal vs Fibonacci: {qc/max(fb,1e-9):.3f}x   "
      f"|   Fibonacci vs random: {fb/max(rd,1e-9):.3f}x")
print("(ratio > 1 = worse than the comparison; < 1 = better. Report whatever this says, no spin.)")
