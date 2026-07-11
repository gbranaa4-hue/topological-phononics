#!/usr/bin/env python3
"""Real methods from the topological-phononics work, consolidated for reuse (dashboard + tests).
Formulas copied VERBATIM from the trusted, pre-registered scripts -- this module is the single
source of truth so the interactive dashboard runs the SAME code, not a re-derivation:
  ssh_H / edge mode ............ ssh_topological_reservoir.py, phase_transition_defect.py
  reservoir + ridge + penalty .. ssh_scaling.py, cross_device_transfer_tight.py
  memory capacity + noise ...... reservoir_cavity_v2.py, noise_feature*.py, rank_margin*.py

Run `python phononic_methods.py` to self-test that these reproduce the reported numbers
(edge mode ~0 topological, defect penalty several-x lower topological, structured noise cancelled).
"""
import numpy as np

# ---------- SSH lattice (matched-dimerization parametrization, phase_transition_defect.py) ----------
def ssh_H(N, g, remove=None):
    """2N-site SSH chain. Dimerization g: v = 1-g (intra-cell), w = 1+g (inter-cell).
       g > 0 -> topological (w>v, edge modes); g < 0 -> trivial. remove = dead resonator site(s)."""
    M = 2 * N; v, w = 1.0 - g, 1.0 + g
    H = np.zeros((M, M))
    for i in range(M - 1):
        H[i, i + 1] = H[i + 1, i] = (v if i % 2 == 0 else w)
    if remove is not None:
        for k in np.atleast_1d(remove): H[k, :] = 0.0; H[:, k] = 0.0
    return H

def edge_mode(N, g):
    """Spectrum + the eigenmode nearest E=0. Returns (eigenvalues, psi_edge, minE, edge_weight)."""
    H = ssh_H(N, g); vals, vecs = np.linalg.eigh(H)
    j = int(np.argmin(np.abs(vals)))
    psi = vecs[:, j]; w2 = psi ** 2
    M = 2 * N; e = max(1, M // 8)                       # outer 1/8 of each end
    return vals, psi, float(abs(vals[j])), float(w2[:e].sum() + w2[-e:].sum())

# ---------- leaky-integrator SSH reservoir (verbatim dynamics from the trusted scripts) ----------
def reservoir_states(N, g, u=None, T=1500, remove=None, alpha=1.0, rho=0.95, win_scale=0.5,
                     noise=None, D=0.0, rank=1, stage="readout", seed=0):
    """Returns (X, u). If u is None it is drawn first from rng(seed) (matching the originals),
       then any noise structures are drawn AFTER u from the same stream (so noise is independent).
       win_scale: uniform drive strength -- 0.12 for the defect scripts, 0.5 for the noise scripts."""
    rng = np.random.default_rng(seed)
    if u is None: u = rng.standard_normal(T)
    M = 2 * N; H = ssh_H(N, g, remove)
    W = rho * H / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-9)
    win = np.full(M, win_scale); x = np.zeros(M); X = np.zeros((len(u), M))
    if noise == "structured":
        V, _ = np.linalg.qr(rng.standard_normal((M, rank))); V = V[:, :rank]
        C = np.zeros((len(u), rank))
        for i in range(rank):
            c = np.zeros(len(u))
            for t in range(1, len(u)): c[t] = 0.9 * c[t - 1] + rng.standard_normal()
            C[:, i] = c / (c.std() + 1e-9)
    for t in range(len(u)):
        pre = np.tanh(W @ x + win * u[t]); nz = 0.0
        if stage == "dynamics" and noise == "structured": nz = (D / np.sqrt(rank)) * (C[t] @ V.T)
        elif stage == "dynamics" and noise == "random":   nz = D * rng.standard_normal(M)
        x = (1 - alpha) * x + alpha * (pre + nz); X[t] = x
    if stage == "readout" and noise == "structured": X = X + (D / np.sqrt(rank)) * (C @ V.T)
    elif stage == "readout" and noise == "random":   X = X + D * rng.standard_normal((len(u), M))
    return X, u

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

# ---------- defect-tolerance: frozen readout, topological(+g) vs trivial(-g), one dead resonator ----------
def defect_demo(N, absg, remove_site, seed=0, T=1500, BURN=200):
    rng = np.random.default_rng(seed); u = rng.standard_normal(T)
    y = np.zeros(T); y[2:] = u[:-2]; yb = y[BURN:]
    out = {}
    for label, g in (("topological", +absg), ("trivial", -absg)):
        Xin = reservoir_states(N, g, u=u, win_scale=0.12)[0][BURN:]; wout = ridge(Xin, yb)
        Xdef = reservoir_states(N, g, u=u, win_scale=0.12, remove=remove_site)[0][BURN:]
        pen = nmse(Xdef @ wout, yb) - nmse(Xdef @ ridge(Xdef, yb), yb)
        out[label] = dict(g=g, penalty=pen, target=yb,
                          pred_intact=Xin @ wout, pred_defect=Xdef @ wout)
    return out

# ---------- memory capacity (fair-across-Q metric) + noise-as-a-feature ----------
def memory_capacity(N, g, D=0.0, rank=1, noise=None, stage="readout", alpha=0.5,
                    seed=0, T=3500, BURN=400, dmax=12, return_per=False):
    X, u = reservoir_states(N, g, T=T, alpha=alpha, noise=noise, D=D, rank=rank, stage=stage, seed=seed)
    X, u = X[BURN:], u[BURN:]; n = len(u); tr = int(0.6 * n); mc = 0.0; per = []
    for k in range(1, dmax + 1):
        t = np.arange(k, n); Xf, yv = X[t], u[t - k]; sp = tr - k
        w_ = ridge(Xf[:sp], yv[:sp])
        r2 = max(0.0, 1 - np.mean((Xf[sp:] @ w_ - yv[sp:]) ** 2) / (np.var(yv[sp:]) + 1e-12))
        mc += r2; per.append(r2)
    return (mc, per) if return_per else mc

# ------------------------------------ self-test / honesty cross-check ------------------------------------
if __name__ == "__main__":
    print("SELF-TEST -- does this module reproduce the reported behaviour?\n")

    print("[1] EDGE MODE (N=8):")
    for lbl, g in (("topological (+g)", 0.6), ("trivial (-g)", -0.6)):
        _, _, minE, ew = edge_mode(8, g)
        print(f"    {lbl:18} min|E| = {minE:.4f}   edge-weight = {ew:.2f}")
    print("    expect: topological min|E|~0 & edge-weight high ; trivial min|E| large & edge-weight low\n")

    print("[2] DEFECT TOLERANCE (N=8, uniform drive, avg 3 seeds):")
    def adv(site):
        pT = np.mean([defect_demo(8, 0.6, site, seed=s)["topological"]["penalty"] for s in range(3)])
        pR = np.mean([defect_demo(8, 0.6, site, seed=s)["trivial"]["penalty"]     for s in range(3)])
        return pT, pR
    for name, site in (("bulk (mid-chain, site 8)", 8), ("near boundary (site 3)", 3)):
        pT, pR = adv(site)
        print(f"    {name:26} topo {pT:7.3f}  vs  trivial {pR:7.3f}   -> {pR/max(pT,1e-6):5.1f}x")
    interior = [adv(k) for k in range(2, 14)]
    mT = np.mean([p[0] for p in interior]); mR = np.mean([p[1] for p in interior])
    wins = np.mean([p[0] < p[1] for p in interior])
    print(f"    position-AVERAGE (sites 2..13)   topo {mT:7.3f}  vs  trivial {mR:7.3f}   -> {mR/max(mT,1e-6):5.1f}x"
          f"   (topo wins {wins*100:.0f}% of sites)")
    print("    expect: bulk ~2x, boundary larger, position-average several-x, topo wins most sites\n")

    print("[3] NOISE AS A FEATURE (N=8, readout-stage, avg 2 seeds):")
    clean = np.mean([memory_capacity(8, 0.6, D=0.0, seed=s) for s in range(2)])
    st = np.mean([memory_capacity(8, 0.6, D=0.6, rank=1, noise="structured", seed=s) for s in range(2)])
    rn = np.mean([memory_capacity(8, 0.6, D=0.6, noise="random", seed=s) for s in range(2)])
    print(f"    clean MC = {clean:.2f}   structured@D=0.6 = {st:.2f} (should stay ~clean)   "
          f"random@D=0.6 = {rn:.2f} (should drop)")
    print("    rank sweep (structured, D=0.5, fixed total power):")
    for r in (1, 4, 8, 16):
        mc = np.mean([memory_capacity(8, 0.6, D=0.5, rank=r, noise="structured", seed=s) for s in range(2)])
        print(f"        rank {r:>2}: MC = {mc:.2f}  ({mc/max(clean,1e-6)*100:3.0f}% of clean)")
    print("\n(numbers should match the trusted scripts' behaviour -- this is the fidelity check.)")
