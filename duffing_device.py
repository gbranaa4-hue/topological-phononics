#!/usr/bin/env python3
"""Piece 4: sim-to-device bridge. Damped, driven, DUFFING-nonlinear coupled oscillators (RK4 ODEs),
SSH-dimerized coupling. Reservoir state = sampled [x, v] per input step. Does topological
defect-tolerance survive real physics -- and does it depend on preserving the CHIRAL symmetry?

Two mechanical models:
  NAIVE spring chain: coupling = graph Laplacian (D - A). Removing a resonator lowers its neighbors'
     on-site stiffness -> an ON-SITE (symmetry-BREAKING) perturbation, which SSH topology does NOT
     protect (see the chiral caveat). The dimerization also sits on the diagonal -> chiral broken.
  COMPENSATED chain: grounding springs keep on-site stiffness UNIFORM; only the OFF-DIAGONAL coupling
     alternates (like tight-binding) -> chiral symmetry preserved. (= the mechanical version of the
     grounding-compensation caps circuit_predict.py already flagged as REQUIRED.)

PRE-REGISTERED: naive model loses/reverses the topological advantage (dead element -> on-site
disorder); compensated model RECOVERS it -> the advantage transfers to hardware ONLY IF the chiral
symmetry is engineered. (Reported honestly either way -- no p-hacking toward the positive.)
"""
import numpy as np

N = 8; M = 2 * N; T = 1200; BURN = 200
WIN = np.random.default_rng(123).uniform(-1, 1, M) * 0.5          # fixed random drive, shared everywhere

def coupling(topological, remove, mode):
    ki, ko = (0.2, 0.8) if topological else (0.8, 0.2)
    A = np.zeros((M, M))
    for i in range(M - 1): A[i, i + 1] = A[i + 1, i] = (ki if i % 2 == 0 else ko)
    if remove is not None:
        for k in np.atleast_1d(remove): A[k, :] = 0.0; A[:, k] = 0.0
    if mode == "naive":  return np.diag(A.sum(1)) - A, 1.0        # Laplacian; on-site alternates (chiral broken)
    return -A, 2.0                                                # off-diagonal only; uniform on-site (chiral kept)

def reservoir(u, topological, remove=None, mode="naive", gamma=0.4, beta=0.3, dt=0.05, hold=40):
    K, w0sq = coupling(topological, remove, mode); x = np.zeros(M); v = np.zeros(M)
    Xs = np.zeros((len(u), 2 * M))
    acc = lambda x, v, f: -gamma * v - w0sq * x - beta * x ** 3 - K @ x + f
    for t in range(len(u)):
        f = WIN * u[t]
        for _ in range(hold):
            a1 = acc(x, v, f)
            a2 = acc(x + 0.5 * dt * v, v + 0.5 * dt * a1, f)
            a3 = acc(x + 0.5 * dt * (v + 0.5 * dt * a1), v + 0.5 * dt * a2, f)
            a4 = acc(x + dt * (v + 0.5 * dt * a2), v + dt * a3, f)
            x = x + dt * (v + dt / 6 * (a1 + a2 + a3)); v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        Xs[t] = np.concatenate([x, v])
    return Xs

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))
def data(seed):
    u = np.random.default_rng(seed).standard_normal(T); y = np.zeros(T); y[2:] = u[:-2]; return u, y[BURN:]

print("Piece 4: damped/driven Duffing SSH reservoir -- does topological defect-tolerance survive real physics?\n")
for mode in ("naive", "compensated"):
    tag = "Laplacian, chiral BROKEN" if mode == "naive" else "grounded, chiral PRESERVED"
    ct = np.mean([nmse(reservoir(data(s)[0], True, mode=mode)[BURN:] @
                       ridge(reservoir(data(s)[0], True, mode=mode)[BURN:], data(s)[1]), data(s)[1]) for s in range(2)])
    cr = np.mean([nmse(reservoir(data(s)[0], False, mode=mode)[BURN:] @
                       ridge(reservoir(data(s)[0], False, mode=mode)[BURN:], data(s)[1]), data(s)[1]) for s in range(2)])
    if max(ct, cr) >= 0.6:
        print(f"[{mode:11}] ({tag})  capability topo {ct:.3f}/triv {cr:.3f}  -> VACUOUS, skip defect test\n"); continue
    pT, pR, wins, npair = [], [], 0, 0
    for s in range(2):
        u, yb = data(s)
        wT = ridge(reservoir(u, True, mode=mode)[BURN:], yb); wR = ridge(reservoir(u, False, mode=mode)[BURN:], yb)
        for site in range(2, M - 1, 2):
            XdT = reservoir(u, True, remove=site, mode=mode)[BURN:]; XdR = reservoir(u, False, remove=site, mode=mode)[BURN:]
            a = nmse(XdT @ wT, yb) - nmse(XdT @ ridge(XdT, yb), yb)
            b = nmse(XdR @ wR, yb) - nmse(XdR @ ridge(XdR, yb), yb)
            pT.append(a); pR.append(b); wins += (a < b); npair += 1
    pT, pR = np.array(pT), np.array(pR); r = np.median(pR) / max(np.median(pT), 1e-6)
    verdict = "topology HELPS" if (np.median(pT) < np.median(pR) and wins / npair > 0.6) else "topology does NOT help"
    print(f"[{mode:11}] ({tag})  capability topo {ct:.3f}/triv {cr:.3f}")
    print(f"{'':13} defect penalty  topo {np.median(pT):.2f}  vs trivial {np.median(pR):.2f}  "
          f"({r:.1f}x)  topo wins {100*wins/npair:.0f}%  -> {verdict}\n")

print("--- if naive=NOT help but compensated=HELPS: the advantage transfers to hardware ONLY with")
print("    engineered chiral symmetry (grounding compensation). That is the honest device conclusion. ---")
