#!/usr/bin/env python3
"""Interactive dashboard for the topological-phononic reservoir -- runs the REAL methods live
(imports phononic_methods.py, the same verbatim code that reproduces the paper numbers; run
`python phononic_methods.py` to see the fidelity check). No new dependencies: numpy + matplotlib.

    python dashboard.py

Three views (radio, top-left):
  Edge mode ......... drag dimerization g / size N -> spectrum + edge-mode localization, live
  Defect tolerance .. click-drag the dead-resonator site -> frozen-readout recall, topo vs trivial
  Noise as a feature. drag noise amplitude/rank, toggle structured|random & readout|dynamics -> MC

Everything on screen is computed from the actual algorithm. Honest scope: simulation of a
tight-binding model, a linear-memory reservoir task -- a principle demonstrator, not a device.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
import phononic_methods as pm

TOPO, TRIV, TGT = "#2b6cb0", "#c53030", "#222222"     # blue / red / black
_clean_cache = {}

def build_dashboard():
    fig = plt.figure(figsize=(12.5, 7.2))
    fig.canvas.manager.set_window_title("Topological phononic reservoir -- live methods")
    fig.text(0.35, 0.965, "Topological phononic reservoir  —  every curve computed live from the real method",
             fontsize=12, weight="bold")
    fig.text(0.35, 0.02, "simulation of a tight-binding SSH model, linear-memory reservoir task — "
             "a principle demonstrator, not a device.", fontsize=8, style="italic", color="#666")

    ax1 = fig.add_axes([0.35, 0.55, 0.60, 0.36])
    ax2 = fig.add_axes([0.35, 0.10, 0.60, 0.35])

    # ---- controls (left column) ----
    ax_view = fig.add_axes([0.03, 0.73, 0.22, 0.20]); ax_view.set_title("view", fontsize=9, loc="left")
    r_view = RadioButtons(ax_view, ("Edge mode", "Defect tolerance", "Noise as a feature"))
    def mk(y, lo, hi, init, step, label):
        a = fig.add_axes([0.08, y, 0.15, 0.02])
        return Slider(a, label, lo, hi, valinit=init, valstep=step)
    s_N    = mk(0.68, 4, 14, 8, 1, "N cells")
    s_g    = mk(0.635, 0.10, 0.90, 0.6, 0.05, "dimer g")
    s_site = mk(0.59, 1, 26, 8, 1, "defect site")
    s_D    = mk(0.545, 0.0, 1.0, 0.5, 0.05, "noise D")
    s_rank = mk(0.50, 1, 28, 1, 1, "noise rank")
    ax_nt = fig.add_axes([0.03, 0.37, 0.13, 0.10]); ax_nt.set_title("noise", fontsize=8, loc="left")
    r_nt = RadioButtons(ax_nt, ("structured", "random"))
    ax_st = fig.add_axes([0.17, 0.37, 0.10, 0.10]); ax_st.set_title("enters at", fontsize=8, loc="left")
    r_st = RadioButtons(ax_st, ("readout", "dynamics"))
    ax_btn = fig.add_axes([0.08, 0.30, 0.15, 0.04]); b_go = Button(ax_btn, "Recompute")
    fig.text(0.03, 0.25, "Edge: uses g, N\nDefect: g, N, site\nNoise: g, N, D, rank,\n           noise, enters-at",
             fontsize=7.5, color="#555", va="top")

    def clean_mc(N, g):
        key = (N, round(g, 3))
        if key not in _clean_cache:
            _clean_cache[key] = pm.memory_capacity(N, g, D=0.0, T=1500, dmax=8, return_per=True)
        return _clean_cache[key]

    def redraw(_=None):
        view = r_view.value_selected
        N = int(s_N.val); g = float(s_g.val); M = 2 * N
        ax1.clear(); ax2.clear()

        if view == "Edge mode":
            vals, psi, minE, ew = pm.edge_mode(N, +g)
            valsT, psiT, minET, ewT = pm.edge_mode(N, -g)
            ax1.axhline(0, color="#aaa", lw=0.8)
            ax1.plot(np.sort(vals), "o-", color=TOPO, ms=4, label="topological (+g)")
            ax1.plot(np.sort(valsT), "s-", color=TRIV, ms=3, label="trivial (−g)")
            ax1.set_title(f"spectrum   |   topological min|E|={minE:.3f}, edge-wt={ew:.2f}   "
                          f"vs   trivial min|E|={minET:.3f}, edge-wt={ewT:.2f}", fontsize=9)
            ax1.set_xlabel("mode index (sorted)"); ax1.set_ylabel("energy E"); ax1.legend(fontsize=8)
            ax2.bar(np.arange(M) - 0.18, psi ** 2, width=0.36, color=TOPO, label="topological")
            ax2.bar(np.arange(M) + 0.18, psiT ** 2, width=0.36, color=TRIV, alpha=0.7, label="trivial")
            ax2.set_title("mode nearest E=0: probability weight |ψ|² per site "
                          "(topological pins to the ends)", fontsize=9)
            ax2.set_xlabel("resonator site"); ax2.set_ylabel("|ψ|²"); ax2.legend(fontsize=8)

        elif view == "Defect tolerance":
            site = min(int(s_site.val), M - 2)
            res = pm.defect_demo(N, g, site, seed=0, T=1200, BURN=150)
            pT, pR = res["topological"]["penalty"], res["trivial"]["penalty"]
            tgt = res["topological"]["target"]; w = slice(0, 120)
            ax1.plot(tgt[w], color=TGT, lw=1.6, label="target  u[t−2]")
            ax1.plot(res["topological"]["pred_defect"][w], color=TOPO, lw=1.2, label="topological (dead site)")
            ax1.plot(res["trivial"]["pred_defect"][w], color=TRIV, lw=1.2, alpha=0.8, label="trivial (dead site)")
            ax1.axvline(0, color="none")
            ax1.set_title(f"frozen readout after killing resonator #{site}   |   "
                          f"penalty  topo {pT:.2f}  vs  trivial {pR:.2f}   ({pR/max(pT,1e-6):.1f}× better)",
                          fontsize=9)
            ax1.set_xlabel("time step"); ax1.set_ylabel("recall"); ax1.legend(fontsize=8, ncol=3, loc="upper right")
            ax2.bar(["topological", "trivial"], [max(pT, 1e-3), max(pR, 1e-3)], color=[TOPO, TRIV])
            ax2.set_yscale("log"); ax2.set_ylabel("defect penalty (log)")
            ax2.set_title("robustness penalty = frozen-readout error above the retrained oracle "
                          "(lower = more defect-tolerant)", fontsize=9)

        else:  # Noise as a feature
            rank = min(int(s_rank.val), M); D = float(s_D.val)
            nt, st = r_nt.value_selected, r_st.value_selected
            cur, per_cur = pm.memory_capacity(N, +g, D=D, rank=rank, noise=nt, stage=st,
                                              T=1500, dmax=8, return_per=True)
            clean, per_clean = clean_mc(N, +g)
            col = TOPO if nt == "structured" else TRIV
            ax1.bar(["clean", "with noise"], [clean, cur], color=["#999", col])
            ax1.set_ylabel("memory capacity")
            ax1.set_title(f"{nt} noise @ {st}, D={D:.2f}, rank={rank}  →  MC retained "
                          f"{cur/max(clean,1e-6)*100:.0f}%   (structured@readout stays ~flat; "
                          f"random / dynamics-stage falls)", fontsize=9)
            dl = np.arange(1, len(per_clean) + 1)
            ax2.plot(dl, per_clean, "o-", color="#999", label="clean")
            ax2.plot(dl, per_cur, "o-", color=col, label=f"{nt} / {st}")
            ax2.set_ylim(-0.02, 1.02)
            ax2.set_title("where the memory lives: recall R² per delay (noise eats the tail first)", fontsize=9)
            ax2.set_xlabel("delay k (recall u[t−k])"); ax2.set_ylabel("R²"); ax2.legend(fontsize=8)

        fig.canvas.draw_idle()

    for w in (s_N, s_g, s_site, s_D, s_rank):
        w.on_changed(redraw)
    r_view.on_clicked(redraw); r_nt.on_clicked(redraw); r_st.on_clicked(redraw); b_go.on_clicked(redraw)
    redraw()
    # keep refs alive
    fig._widgets = (r_view, s_N, s_g, s_site, s_D, s_rank, r_nt, r_st, b_go)
    return fig

if __name__ == "__main__":
    build_dashboard()
    plt.show()
