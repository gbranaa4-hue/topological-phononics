#!/usr/bin/env python3
"""HONEST prediction tool: compute what the circuit SHOULD do from the REAL component values,
so you check measurements against real predictions instead of inventing them.

v2 -- fixes two problems the first pass (and the hyped shopping list) missed:
  (1) coupling caps must be COMPARABLE to the site term or the SSH gap is a few Hz (invisible
      under real Q/tolerance). Use a large dimerization ratio C1/C2.
  (2) CHIRAL SYMMETRY (what protects the SSH edge mode) requires every node to see the SAME
      total capacitance to ground. Edge nodes have one coupling cap, interior nodes two -- so
      you MUST add a grounding-compensation cap to each node to equalize them. Omit this and
      the 'edge mode' drifts and the protection is gone. (The hype text left these out.)

Model (topolectrical SSH, Lee et al.): capacitance Laplacian C_L; each eigenvalue lambda ->
resonance f = 1/(2*pi*sqrt(L*lambda)); eigenvector = mode shape.
"""
import numpy as np

# --- component values (EDIT to what you buy; these give a clean, audible, measurable gap) ---
C1 = 8.2e-9          # STRONG coupling cap, 8.2 nF
C2 = 1.0e-9          # WEAK   coupling cap, 1.0 nF   (dimerization ratio 8.2x -> wide clean gap)
Rg = 15e3            # gyrator resistors, 15 k
Cg = 12e-9           # gyrator capacitor, 12 nF
L = Cg * Rg * Rg     # synthesized inductance
NCELL = 4; M = 2 * NCELL

def laplacian(topological=True, compensate=True):
    C = np.zeros((M, M))
    for i in range(M - 1):
        weak = (i % 2 == 0) if topological else (i % 2 == 1)   # topological: outer bond WEAK
        c = C2 if weak else C1
        C[i, i + 1] = C[i + 1, i] = -c
    row_coupling = [sum(-C[i, j] for j in range(M) if j != i) for i in range(M)]
    target = C1 + C2                                # every node should total this to ground
    for i in range(M):
        gnd = (target - row_coupling[i]) if compensate else 0.0   # grounding-compensation cap
        C[i, i] = row_coupling[i] + gnd
    return C

def analyze(name, topological, compensate=True):
    C = laplacian(topological, compensate)
    lam, V = np.linalg.eigh(C)
    f = 1.0 / (2 * np.pi * np.sqrt(L * np.clip(lam, 1e-18, None)))
    o = np.argsort(f); f, V = f[o], V[:, o]
    k = max(1, M // 8)
    ew = np.array([(V[:k, j] ** 2).sum() + (V[-k:, j] ** 2).sum() for j in range(M)])
    j = int(np.argmax(ew))
    fmid = 1.0 / (2 * np.pi * np.sqrt(L * (C1 + C2)))
    lo = f[f < fmid - 1]; hi = f[f > fmid + 1]
    gap = (hi.max() if len(hi) else fmid) - (lo.min() if len(lo) else fmid)  # rough band span
    print(f"\n=== {name} ===")
    print(f"  band-center f_mid = {fmid:.0f} Hz;  modes (Hz): {', '.join(f'{v:.0f}' for v in f)}")
    print(f"  edge-localized mode: f = {f[j]:.0f} Hz, edge-weight = {ew[j]:.2f}  "
          f"({'CLEAN EDGE MODE' if ew[j] > 0.85 else 'weak/none'})")
    print(f"  |amp| by node: {', '.join(f'{a:.2f}' for a in np.abs(V[:, j])/np.abs(V[:, j]).max())}")
    return f[j], ew[j]

print(f"L = {L:.2f} H (gyrator Cg*Rg^2)")
fT, ewT = analyze("TOPOLOGICAL (outer bond weak) + compensation", True)
analyze("TRIVIAL (outer bond strong) + compensation", False)
print("\n--- sanity: WITHOUT grounding compensation (shows why it's needed) ---")
analyze("TOPOLOGICAL, NO compensation", True, compensate=False)

print("\n--- corrected build spec + what to check ---")
print(f"  site freq / band center ~ {1/(2*np.pi*np.sqrt(L*(C1+C2))):.0f} Hz (audible, sound-card measurable)")
print(f"  predicted TOPOLOGICAL edge mode: f = {fT:.0f} Hz, edge-weight {ewT:.2f} (should localize at the driven end)")
print("  REQUIRED extra parts the hype omitted: one grounding-compensation cap per node")
print("  (value = (C1+C2) - [coupling caps on that node]; edge nodes get ~C1, interior nodes ~0).")
print("  These are LOSSLESS predictions. Record ACTUAL measured numbers; never reuse these as data.")
