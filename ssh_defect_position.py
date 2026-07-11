#!/usr/bin/env python3
"""MECHANISM probe of the one surviving positive: topological reservoirs degrade more
gracefully under a dead resonator (full-state frozen-readout penalty ~2-7x lower than
trivial at matched dimerization). WHY? Hypothesis: the topological chain 'heals' around a
defect (a removed interior site spawns new protected boundary modes). If so, defect-
robustness should depend on WHERE the resonator dies and WHICH bond it breaks.

Sweep the removed-site position k across the chain; measure the frozen-readout defect
PENALTY = transfer NMSE - oracle NMSE (train readout on the clean chain, freeze, test on the
chain with site k removed). Topological (v<w) vs trivial (v>w). Recall task, full-state.

PRE-REGISTERED
  A  topological penalty < trivial penalty across MOST interior positions (robustness is broad).
  B  mechanism signature: the advantage varies with position -- either weaker near the EDGE
     (the protected mode's home) or an even/odd PARITY pattern (breaking a weak vs strong bond).
  DISCONFIRM  penalty flat in k / no topological gap -> robustness isn't position-structured;
     the 'healing' picture is wrong (report honestly).
"""
import numpy as np

N = 20; M = 2 * N; T = 2000; BURN = 200
VT, WT = 0.4, 1.0; VR, WR = 1.0, 0.4
FULL = list(range(M))

def ssh_H(v, w, remove=None):
    H = np.zeros((M, M))
    for i in range(M - 1):
        H[i, i + 1] = H[i + 1, i] = (v if i % 2 == 0 else w)
    if remove is not None:
        H[remove, :] = 0; H[:, remove] = 0
    return H

def states(H, u, rho=0.95):
    W = rho * H / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-9)
    win = np.full(M, 0.12)                                  # UNIFORM drive (confound-free: no privileged site)
    x = np.zeros(M); X = np.zeros((len(u), M))
    for t in range(len(u)):
        x = np.tanh(W @ x + win * u[t]); X[t] = x
    return X[BURN:]

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

rng0 = np.random.default_rng(1); u = rng0.standard_normal(T)
y = np.zeros(T); y[2:] = u[:-2]; y = y[BURN:]

def penalty(v, w, k, wout):
    Xd = states(ssh_H(v, w, remove=k), u)
    return nmse(Xd @ wout, y) - nmse(Xd @ ridge(Xd, y), y)

woutT = ridge(states(ssh_H(VT, WT), u), y)                 # frozen readouts (trained clean)
woutR = ridge(states(ssh_H(VR, WR), u), y)

print("defect-position sweep: penalty = frozen-readout transfer NMSE - oracle (per removed site k)\n")
print(f"{'k':>3} | {'dist-to-edge':>12} | {'bond':>6} | {'TOPO pen':>9} | {'TRIV pen':>9} | topo<triv?")
print("-" * 66)
rows = []
for k in range(2, M - 2):
    pT = penalty(VT, WT, k, woutT); pR = penalty(VR, WR, k, woutR)
    d = min(k, M - 1 - k)
    bond = "weak" if k % 2 == 0 else "strong"              # site k parity ~ which bond-type it centers
    rows.append((k, d, bond, pT, pR))
    if k % 2 == 0 or abs(pT - pR) > 0.3:                   # thin the printout
        flag = "yes" if pT < pR - 0.02 else ("~" if abs(pT - pR) <= 0.02 else "NO")
        print(f"{k:>3} | {d:>12} | {bond:>6} | {pT:>9.3f} | {pR:>9.3f} | {flag}")

pT_all = np.array([r[3] for r in rows]); pR_all = np.array([r[4] for r in rows])
edge = np.array([r[1] < 6 for r in rows]); even = np.array([r[0] % 2 == 0 for r in rows])
print(f"\nmean penalty   topological={pT_all.mean():+.3f}   trivial={pR_all.mean():+.3f}   "
      f"(topological wins {int((pT_all < pR_all).mean()*100)}% of positions)")
print(f"by region      EDGE(d<6): topo {pT_all[edge].mean():+.3f} vs triv {pR_all[edge].mean():+.3f} | "
      f"CENTER: topo {pT_all[~edge].mean():+.3f} vs triv {pR_all[~edge].mean():+.3f}")
print(f"by bond broken WEAK(k even): topo {pT_all[even].mean():+.3f} | STRONG(k odd): topo {pT_all[~even].mean():+.3f}")

print("\n--- verdict vs pre-registration ---")
broad = (pT_all < pR_all).mean() > 0.6 and pT_all.mean() < pR_all.mean() - 0.05
edge_effect = abs(pT_all[edge].mean() - pT_all[~edge].mean()) > 0.1
parity_effect = abs(pT_all[even].mean() - pT_all[~even].mean()) > 0.1
if broad and (edge_effect or parity_effect):
    sig = ("EDGE-dependent (weaker/different near the boundary -> the protected-mode home matters)"
           if edge_effect else "PARITY-dependent (weak vs strong broken bond matters -> topological bond structure)")
    print(f"MECHANISM FOUND: topological robustness is broad AND position-structured -- {sig}. "
          f"Supports the 'heals around the defect' picture.")
elif broad:
    print("Robust but position-FLAT: topological wins broadly but no position/parity structure -> "
          "the advantage is global, not local 'healing'. Report honestly.")
else:
    print("NO clean topological advantage across positions -> the earlier result may be fragile; read the table.")
