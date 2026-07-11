#!/usr/bin/env python3
"""FIRST probe of the topology thread: do topologically protected modes exist in a phononic
resonator chain (the MEMS/acoustic substrate), and do they buy DISORDER-ROBUST computation?

SSH chain: 2N coupled resonators, hoppings alternating v (intra-cell) and w (inter-cell);
tight-binding dynamical matrix H (mode frequencies = eigenvalues).
  topological  w > v  -> a mid-gap EDGE mode localized at the boundary
  trivial      v > w  -> gapped, no edge mode
SSH edge modes are protected by CHIRAL (sublattice) symmetry, so they survive HOPPING
disorder (symmetry-respecting) but NOT on-site disorder (symmetry-breaking) -- a 'when does
the structure protect?' boundary, in the spirit of the reservoir symmetry-selection rule.

PRE-REGISTERED
  A1 physics    topological chain has a mode with |E|<0.1 localized at an end (edge weight
                >0.5); trivial chain does not.
  A2 protection under HOPPING disorder the edge |E| stays pinned (<0.1); under ON-SITE
                disorder it drifts up -- protection is CONDITIONAL on the symmetry.
  B  computation as a linear reservoir, an EDGE-tap readout keeps short-term memory better
                under hopping disorder (topological) than a bulk-tap or the trivial chain.
                DISCONFIRM: no robustness edge -> topology doesn't transfer to computation
                here (report the null, like the target-net test).
"""
import numpy as np

def ssh_H(N, v, w, onsite=0.0, hop=0.0, rng=None):
    M = 2 * N
    H = np.zeros((M, M))
    for i in range(M - 1):
        base = v if i % 2 == 0 else w                       # v within cell, w between cells
        if hop and rng is not None:
            base *= (1 + hop * 2 * (rng.random() - 0.5))    # symmetry-RESPECTING (hopping) disorder
        H[i, i + 1] = H[i + 1, i] = base
    if onsite and rng is not None:
        H[np.diag_indices(M)] += onsite * 2 * (rng.random(M) - 0.5)   # symmetry-BREAKING (on-site)
    return H

def edge_mode(H):
    E, V = np.linalg.eigh(H)
    j = int(np.argmin(np.abs(E)))                           # closest-to-zero mode
    psi2 = V[:, j] ** 2
    k = max(2, len(psi2) // 8)
    return abs(E[j]), float(psi2[:k].sum() + psi2[-k:].sum())

def reservoir_MC(H, rng, T=3000, rho=0.95, burn=200, dmax=15, cols=None):
    M = H.shape[0]
    Hn = H / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-9)
    u = rng.standard_normal(T + burn)
    x = np.zeros(M); X = np.zeros((T, M))
    for t in range(T + burn):
        x = rho * (Hn @ x); x[0] += u[t]
        if t >= burn:
            X[t - burn] = x
    if cols is not None:
        X = X[:, cols]
    n = T; tr = int(n * 0.6); mc = 0.0
    for d in range(1, dmax + 1):
        t = np.arange(d, n); Xf, y = X[t], u[t - d + burn]
        s = tr - d
        Xtr, ytr, Xte, yte = Xf[:s], y[:s], Xf[s:], y[s:]
        A = Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1])
        wout = np.linalg.solve(A, Xtr.T @ ytr)
        pred = Xte @ wout
        r2 = 1 - np.sum((yte - pred) ** 2) / (np.sum((yte - yte.mean()) ** 2) + 1e-12)
        mc += max(0.0, r2)
    return mc

N = 20; VT, WT = 0.4, 1.0                                   # topological: w>v
VR, WR = 1.0, 0.4                                            # trivial:     v>w
R = 40                                                       # disorder realizations
rng = np.random.default_rng(0)

print("=== A1: does a protected edge mode exist? (clean chains) ===")
e_t, w_t = edge_mode(ssh_H(N, VT, WT))
e_r, w_r = edge_mode(ssh_H(N, VR, WR))
print(f"  topological: min|E|={e_t:.3f}  edge-weight={w_t:.2f}   {'EDGE MODE' if e_t<0.1 and w_t>0.5 else 'none'}")
print(f"  trivial:     min|E|={e_r:.3f}  edge-weight={w_r:.2f}   {'EDGE MODE' if e_r<0.1 and w_r>0.5 else 'none'}")

print("\n=== A2: is it protected? (edge |E| over 40 disorder realizations, strength 0.35) ===")
def sweep(v, w, onsite, hop):
    es = [edge_mode(ssh_H(N, v, w, onsite=onsite, hop=hop, rng=rng))[0] for _ in range(R)]
    return np.mean(es), np.std(es)
hop_m, hop_s = sweep(VT, WT, 0.0, 0.35)
ons_m, ons_s = sweep(VT, WT, 0.35, 0.0)
print(f"  topological + HOPPING disorder (symmetry-respecting): edge |E| = {hop_m:.3f} +/- {hop_s:.3f}  "
      f"{'PINNED (protected)' if hop_m<0.1 else 'drifts'}")
print(f"  topological + ON-SITE  disorder (symmetry-breaking):  edge |E| = {ons_m:.3f} +/- {ons_s:.3f}  "
      f"{'pinned' if ons_m<0.1 else 'DRIFTS (protection broken)'}")

print("\n=== B: does the edge channel give disorder-robust reservoir memory? ===")
M = 2 * N; edge_cols = list(range(0, M // 8)) + list(range(M - M // 8, M)); bulk_cols = list(range(M//2 - M//8, M//2 + M//8))
def mc_robust(v, w, cols, hop):
    clean = np.mean([reservoir_MC(ssh_H(N, v, w), rng, cols=cols) for _ in range(3)])
    dis = np.mean([reservoir_MC(ssh_H(N, v, w, hop=hop, rng=rng), rng, cols=cols) for _ in range(R // 4)])
    return clean, dis, dis / (clean + 1e-9)
for label, v, w, cols in [("topological edge-tap", VT, WT, edge_cols),
                          ("topological bulk-tap", VT, WT, bulk_cols),
                          ("trivial edge-tap", VR, WR, edge_cols)]:
    c, d, keep = mc_robust(v, w, cols, 0.35)
    print(f"  {label:<22}: MC clean={c:5.2f}  disordered={d:5.2f}  retained={keep*100:4.0f}%")

print("\n--- read the raw numbers above against the pre-registration; report honestly ---")
