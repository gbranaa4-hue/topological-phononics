#!/usr/bin/env python3
"""TIGHTENED cross-device transfer: (1) magnitude-match the two disorder types by equal RMS
spectral perturbation so the symmetry-boundary question is fair; (2) add a REAL defect (a
removed resonator) -- the harsh test fabrication imposes. Task = linear recall u[t-2] (the
tanh reservoir clears the bar on it; even-order was unlearnable -- tanh is odd).

PRE-REGISTERED
  boundary  with matched disorder, if topological transfers BETTER under HOPPING than under
            ON-SITE, the protection is symmetry-specific (the pinned edge mode). If ~equal,
            the advantage comes from the band GAP, not the perfectly-pinned mode -- report which.
  defect    under a removed resonator, the topological edge-tap readout transfers better than
            the trivial one (new internal boundaries host new protected modes; outer edge
            modes survive). DISCONFIRM: no defect-robustness edge -> topology doesn't help the
            harsh case; the application is weak -- say so before any fabrication.
"""
import numpy as np

N = 20; M = 2 * N; T = 2500; BURN = 200; NTEST = 10
VT, WT = 0.4, 1.0; VR, WR = 1.0, 0.4
EDGE = list(range(0, M // 8)) + list(range(M - M // 8, M))
FULL = list(range(M))

def ssh_H(v, w, onsite=0.0, hop=0.0, rng=None, remove=None):
    H = np.zeros((M, M))
    for i in range(M - 1):
        base = (v if i % 2 == 0 else w)
        if hop and rng is not None:
            base *= (1 + hop * 2 * (rng.random() - 0.5))
        H[i, i + 1] = H[i + 1, i] = base
    if onsite and rng is not None:
        H[np.diag_indices(M)] += onsite * 2 * (rng.random(M) - 0.5)
    if remove is not None:
        H[remove, :] = 0; H[:, remove] = 0                 # removed resonator = decoupled site
    return H

def states(H, u, rho=0.95):
    W = rho * H / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-9)
    win = np.zeros(M); win[0] = 0.6; win[M // 2] = 0.6
    x = np.zeros(M); X = np.zeros((len(u), M))
    for t in range(len(u)):
        x = np.tanh(W @ x + win * u[t]); X[t] = x
    return X[BURN:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

rng0 = np.random.default_rng(1); u = rng0.standard_normal(T)
y = np.zeros(T); y[2:] = u[:-2]; y = y[BURN:]              # recall u[t-2]

# ---- (1) calibrate disorder strengths to equal RMS spectral shift ----
def rms_shift(kind, s, reps=25):
    E0 = np.linalg.eigvalsh(ssh_H(VT, WT))
    kw = lambda r: ({"onsite": s} if kind == "onsite" else {"hop": s})
    return np.mean([np.sqrt(np.mean((np.linalg.eigvalsh(ssh_H(VT, WT, rng=np.random.default_rng(700 + r), **kw(r))) - E0) ** 2))
                    for r in range(reps)])
TARGET = 0.15
STR_HOP = 0.30 * TARGET / rms_shift("hopping", 0.30)
STR_ON = 0.30 * TARGET / rms_shift("onsite", 0.30)
print(f"[calibration] matched to RMS spectral shift {TARGET}: STR_hop={STR_HOP:.3f} "
      f"(check {rms_shift('hopping', STR_HOP):.3f}), STR_on={STR_ON:.3f} (check {rms_shift('onsite', STR_ON):.3f})\n")

def transfer(v, w, cols, kind=None, strength=0.0, defect=False):
    HA = ssh_H(v, w, rng=np.random.default_rng(100),
               **({"onsite": strength} if kind == "onsite" else {"hop": strength}) if kind else {})
    wout = ridge(states(HA, u)[:, cols], y)
    t, o = [], []
    for s in range(NTEST):
        rng = np.random.default_rng(200 + s)
        kw = ({"onsite": strength} if kind == "onsite" else {"hop": strength}) if kind else {}
        rem = int(rng.integers(M // 4, 3 * M // 4)) if defect else None
        X = states(ssh_H(v, w, rng=rng, remove=rem, **kw), u)[:, cols]
        t.append(nmse(X @ wout, y)); o.append(nmse(X @ ridge(X, y), y))
    return np.mean(t), np.mean(o)

print("=== (2) matched-disorder boundary test (transfer NMSE; lower=survives device swap) ===")
print(f"{'phase':>12} | {'disorder':>8} | {'readout':>5} | {'oracle':>6} | {'transfer':>8}")
print("-" * 56)
B = {}
for phase, v, w in [("topological", VT, WT), ("trivial", VR, WR)]:
    for kind, strn in [("hopping", STR_HOP), ("onsite", STR_ON)]:
        for tag, cols in [("edge", EDGE), ("full", FULL)]:
            o, t = transfer(v, w, cols, kind, strn)[::-1]
            B[(phase, kind, tag)] = t
            print(f"{phase:>12} | {kind:>8} | {tag:>5} | {o:6.3f} | {t:8.3f}")

print("\n=== (3) real-defect test: one resonator removed on each unseen chip ===")
print(f"{'phase':>12} | {'readout':>5} | {'oracle':>6} | {'transfer':>8}")
print("-" * 44)
D = {}
for phase, v, w in [("topological", VT, WT), ("trivial", VR, WR)]:
    for tag, cols in [("edge", EDGE), ("full", FULL)]:
        o, t = transfer(v, w, cols, defect=True)[::-1]
        D[(phase, tag)] = t
        print(f"{phase:>12} | {tag:>5} | {o:6.3f} | {t:8.3f}")

print("\n--- verdicts vs pre-registration ---")
th, to = B[("topological", "hopping", "edge")], B[("topological", "onsite", "edge")]
print(f"boundary: topological edge transfer  hopping={th:.3f}  on-site={to:.3f}  -> "
      + ("symmetry-SPECIFIC (hopping clearly better)" if th < to - 0.05 else
         "GAP-driven, not the pinned mode (both similar) -- protection is broader than the edge state"))
dt, dr = D[("topological", "edge")], D[("trivial", "edge")]
print(f"defect:   edge-tap transfer under a removed resonator  topological={dt:.3f}  trivial={dr:.3f}  -> "
      + ("topology SURVIVES defects better" if dt < dr - 0.05 else "no defect-robustness edge (null)"))
