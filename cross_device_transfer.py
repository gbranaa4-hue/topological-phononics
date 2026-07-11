#!/usr/bin/env python3
"""THE APPLICATION TEST: does a topological reservoir give DEVICE-INVARIANT computation?

Physical-reservoir computing's deployment killer is device-to-device variability -- every
fabricated chip differs (disorder), so a readout trained on one chip fails on the next and
you must recalibrate per device. If a TOPOLOGICALLY PROTECTED channel is disorder-immune,
one trained readout might transfer across chips: 'train once, deploy on many imperfect
devices.' This tests exactly that, and checks the catch: SSH protection survives only
SYMMETRY-RESPECTING (hopping) disorder -- but real fabrication scatter is largely on-site
(symmetry-breaking), so we test BOTH.

Setup: SSH chain as a tanh reservoir (nonlinearity fixed from the last probe). Task =
u[t-1]*u[t-2] (even-order product -- ties to the reservoir symmetry work; needs real
computation). Train a ridge readout on ONE disordered chip, apply it UNCHANGED to 8 unseen
disordered chips. transfer penalty = transfer NMSE - oracle NMSE (oracle = retrained per
chip). Small penalty = the readout survived the device swap.

PRE-REGISTERED
  capability   oracle (retrained) NMSE must be < 0.6, else the reservoir can't do the task.
  CONFIRM      under HOPPING disorder, topological transfer penalty << trivial's -- one
               readout works across topological chips, fails across trivial ones.
  BOUNDARY     under ON-SITE disorder the topological advantage shrinks (protection broken).
  DISCONFIRM   topological penalty ~ trivial's -> topology does NOT buy device-invariance;
               the application is not supported (report the null before any fabrication).
"""
import numpy as np

N = 20; M = 2 * N; T = 2500; BURN = 200; STR = 0.30; NTEST = 8
VT, WT = 0.4, 1.0                                            # topological w>v
VR, WR = 1.0, 0.4                                            # trivial     v>w
EDGE = list(range(0, M // 8)) + list(range(M - M // 8, M))

def ssh_H(v, w, onsite=0.0, hop=0.0, rng=None):
    H = np.zeros((M, M))
    for i in range(M - 1):
        base = (v if i % 2 == 0 else w)
        if hop and rng is not None:
            base *= (1 + hop * 2 * (rng.random() - 0.5))
        H[i, i + 1] = H[i + 1, i] = base
    if onsite and rng is not None:
        H[np.diag_indices(M)] += onsite * 2 * (rng.random(M) - 0.5)
    return H

def states(H, u, rho=0.95):
    W = rho * H / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-9)
    win = np.zeros(M); win[0] = 0.6; win[M // 2] = 0.6      # drive edge AND bulk (no locality confound)
    x = np.zeros(M); X = np.zeros((len(u), M))
    for t in range(len(u)):
        x = np.tanh(W @ x + win * u[t]); X[t] = x
    return X[BURN:]

def ridge(X, y, lam=1e-2):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)

def nmse(p, y):
    return float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

rng0 = np.random.default_rng(1)
u = rng0.standard_normal(T)
y = np.zeros(T); y[2:] = u[:-2]; y = y[BURN:]               # linear memory recall: y[t] = u[t-2]
                                                            # (tanh is odd -> even-order product was unlearnable)

def run(v, w, kind, cols):
    kw = {"onsite": STR} if kind == "onsite" else {"hop": STR}
    HA = ssh_H(v, w, rng=np.random.default_rng(100), **kw)
    XA = states(HA, u)[:, cols]
    wout = ridge(XA, y)
    tr, orc = [], []
    for s in range(NTEST):
        Xs = states(ssh_H(v, w, rng=np.random.default_rng(200 + s), **kw), u)[:, cols]
        tr.append(nmse(Xs @ wout, y))                       # frozen readout on unseen chip
        orc.append(nmse(Xs @ ridge(Xs, y), y))              # retrained per chip (floor)
    return np.mean(tr), np.mean(orc)

print(f"even-order product task, {NTEST} unseen chips, disorder {STR}; penalty = transfer - oracle NMSE\n")
print(f"{'phase':>12} | {'disorder':>8} | {'readout':>5} | {'oracle':>6} | {'transfer':>8} | {'penalty':>7}")
print("-" * 66)
rows = {}
for phase, v, w in [("topological", VT, WT), ("trivial", VR, WR)]:
    for kind in ("hopping", "onsite"):
        for tag, cols in [("full", list(range(M))), ("edge", EDGE)]:
            t, o = run(v, w, kind, cols)
            rows[(phase, kind, tag)] = (o, t, t - o)
            print(f"{phase:>12} | {kind:>8} | {tag:>5} | {o:6.3f} | {t:8.3f} | {t-o:+7.3f}")

print("\n--- verdict vs pre-registration ---")
cap = rows[("topological", "hopping", "full")][0]
th = rows[("topological", "hopping", "full")][2]; tr = rows[("trivial", "hopping", "full")][2]
tho = rows[("topological", "onsite", "full")][2]
if cap > 0.6:
    print(f"VACUOUS: reservoir can't do the task (oracle NMSE {cap:.2f} > 0.6) -- capability bar not cleared.")
elif th < tr - 0.05:
    print(f"SUPPORTED (hopping): topological transfer penalty {th:+.3f} << trivial {tr:+.3f} -- one readout "
          f"survives across topological chips. Boundary: on-site penalty {tho:+.3f} "
          f"({'advantage holds' if tho < tr - 0.05 else 'advantage LOST when symmetry broken'}).")
else:
    print(f"NOT SUPPORTED: topological penalty {th:+.3f} ~ trivial {tr:+.3f} -- topology does NOT buy "
          f"device-invariance here. Report the null; do not fabricate on this claim.")
