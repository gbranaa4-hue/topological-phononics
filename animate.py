#!/usr/bin/env python3
"""Auto-playing animations of the real methods (shareable GIFs -- no Python needed to view).
Reuses phononic_methods.py (the self-tested, verbatim code). Three short loops:
  edge_transition.gif  -- sweep dimerization; the edge mode forms + localizes at the ends
  defect_sweep.gif     -- sweep the dead-resonator position; topological penalty stays below trivial
  noise_sweep.gif      -- sweep noise amplitude; structured MC stays flat while random collapses
Run: python animate.py   (needs Pillow, which ships with matplotlib)
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os, phononic_methods as pm

OUT = os.path.dirname(os.path.abspath(__file__))
FOOT = "simulation of a tight-binding SSH model — a principle demonstrator, not a device."
TOPO, TRIV = "#2b6cb0", "#c53030"

def _foot(fig): fig.text(0.5, 0.015, FOOT, ha="center", fontsize=8, style="italic", color="#666")

# ---------- 1. edge mode forming as dimerization sweeps ----------
def edge_transition():
    N = 8; M = 2 * N
    gs = np.concatenate([np.linspace(-0.8, 0.8, 45), np.linspace(0.8, -0.8, 45)])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=80)
    fig.subplots_adjust(bottom=0.18, top=0.84, wspace=0.28); _foot(fig)
    fig.suptitle("Topological transition: the edge mode forms as dimerization sweeps",
                 fontsize=12, weight="bold")
    def frame(i):
        g = gs[i]; ax1.clear(); ax2.clear()
        vals, psi, minE, ew = pm.edge_mode(N, g); col = TOPO if g > 0 else TRIV
        ax1.axhline(0, color="#bbb", lw=0.8); ax1.plot(np.sort(vals), "o-", color=col, ms=4)
        ax1.set_ylim(-2.2, 2.2); ax1.set_xlabel("mode index"); ax1.set_ylabel("energy E")
        ax1.set_title(f"g = {g:+.2f}   ({'topological' if g > 0 else 'trivial'})   min|E| = {minE:.3f}",
                      fontsize=10)
        ax2.bar(range(M), psi ** 2, color=col); ax2.set_ylim(0, 0.55)
        ax2.set_xlabel("resonator site")
        ax2.set_title(f"|ψ|² of the mode nearest E=0    edge-weight = {ew:.2f}", fontsize=10)
    FuncAnimation(fig, frame, frames=len(gs), interval=70).save(
        OUT + "/edge_transition.gif", writer=PillowWriter(fps=14))
    frame(35); fig.savefig(OUT + "/_frame_edge.png"); plt.close(fig)

# ---------- 2. defect position sweep ----------
def defect_sweep():
    N = 8; M = 2 * N; sites = list(range(2, M - 1)); pT, pR = [], []
    for s in sites:                                       # avg 3 seeds, one defect_demo per (site,seed)
        r = [pm.defect_demo(N, 0.6, s, seed=sd, T=1000, BURN=150) for sd in range(3)]
        pT.append(np.mean([x["topological"]["penalty"] for x in r]))
        pR.append(np.mean([x["trivial"]["penalty"] for x in r]))
    pT, pR = np.array(pT), np.array(pR)
    winrate = 100 * np.mean(pT < pR); ratio = pR.mean() / max(pT.mean(), 1e-6)
    summ = f"topological wins {winrate:.0f}% of positions · {ratio:.1f}× lower penalty on average"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=80)
    fig.subplots_adjust(bottom=0.18, top=0.84, wspace=0.28); _foot(fig)
    fig.suptitle("Defect tolerance: a dead resonator hurts the trivial chain far more",
                 fontsize=12, weight="bold")
    def frame(i):
        ax1.clear(); ax2.clear(); s = sites[i]; won = pT[i] < pR[i]
        ax1.bar(["topological", "trivial"], [max(pT[i], 1e-3), max(pR[i], 1e-3)], color=[TOPO, TRIV])
        ax1.set_yscale("log"); ax1.set_ylabel("defect penalty (log)")
        ax1.set_title(f"kill resonator #{s}    →   topological "
                      f"{'wins '+format(pR[i]/max(pT[i],1e-6),'.1f')+'×' if won else 'loses (a boundary site)'}",
                      fontsize=10)
        ax2.plot(sites, pT, "o-", color=TOPO, ms=4, label="topological", alpha=0.35)
        ax2.plot(sites, pR, "s-", color=TRIV, ms=4, label="trivial", alpha=0.35)
        ax2.plot(sites[i], pT[i], "o", color=TOPO, ms=10); ax2.plot(sites[i], pR[i], "s", color=TRIV, ms=10)
        ax2.set_xlim(1, M - 1); ax2.set_ylim(0, max(pR) * 1.12); ax2.legend(fontsize=8, loc="upper right")
        ax2.set_xlabel("dead-resonator position"); ax2.set_ylabel("penalty (avg 3 seeds)")
        ax2.set_title(summ, fontsize=9)
    FuncAnimation(fig, frame, frames=len(sites), interval=350).save(
        OUT + "/defect_sweep.gif", writer=PillowWriter(fps=3))
    frame(len(sites) // 2); fig.savefig(OUT + "/_frame_defect.png"); plt.close(fig)

# ---------- 3. noise amplitude sweep ----------
def noise_sweep():
    N = 8; g = 0.6; Ds = np.linspace(0.0, 1.0, 21)
    clean = pm.memory_capacity(N, g, D=0.0, T=1500, dmax=8)
    st = [pm.memory_capacity(N, g, D=D, rank=1, noise="structured", T=1500, dmax=8) for D in Ds]
    rn = [pm.memory_capacity(N, g, D=D, noise="random", T=1500, dmax=8) for D in Ds]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=80)
    fig.subplots_adjust(bottom=0.18, top=0.84, wspace=0.28); _foot(fig)
    fig.suptitle("Noise as a feature: the readout cancels structured noise; random sets the floor",
                 fontsize=12, weight="bold")
    def frame(i):
        ax1.clear(); ax2.clear()
        ax1.bar(["clean", "structured", "random"], [clean, st[i], rn[i]], color=["#999", TOPO, TRIV])
        ax1.set_ylim(0, clean * 1.15); ax1.set_ylabel("memory capacity")
        ax1.set_title(f"noise amplitude D = {Ds[i]:.2f}", fontsize=10)
        ax2.plot(Ds[:i + 1], st[:i + 1], "o-", color=TOPO, ms=4, label="structured (cancelled)")
        ax2.plot(Ds[:i + 1], rn[:i + 1], "s-", color=TRIV, ms=4, label="random (irreducible)")
        ax2.set_xlim(0, 1); ax2.set_ylim(0, clean * 1.15); ax2.legend(fontsize=8)
        ax2.set_xlabel("noise amplitude D"); ax2.set_ylabel("memory capacity")
        ax2.set_title("MC vs noise (structured stays flat, random collapses)", fontsize=10)
    FuncAnimation(fig, frame, frames=len(Ds), interval=180).save(
        OUT + "/noise_sweep.gif", writer=PillowWriter(fps=7))
    frame(len(Ds) - 1); fig.savefig(OUT + "/_frame_noise.png"); plt.close(fig)

if __name__ == "__main__":
    for name, fn in (("edge_transition", edge_transition), ("defect_sweep", defect_sweep),
                     ("noise_sweep", noise_sweep)):
        fn(); sz = os.path.getsize(OUT + f"/{name}.gif") / 1024
        print(f"wrote {name}.gif  ({sz:.0f} KB)")
    print("done -- 3 shareable GIFs + verification frames in", OUT)
