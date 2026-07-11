#!/usr/bin/env python3
"""Data acquisition + readout training + defect test for the topological reservoir circuit.

Two modes:
  --sim  : linear-SSH proxy of the circuit; lets you develop & test the WHOLE pipeline today
           (train -> freeze -> break a node -> re-measure -> penalty) before the parts arrive.
  --hw   : read REAL node voltages from your ADC. Fill in record_hw() for your hardware.

HONEST HARDWARE NOTE: the reservoir readout needs ALL node voltages sampled SIMULTANEOUSLY.
A 2-channel USB sound card CANNOT do that -> it's fine for the edge-mode localization sweep
(drive at f_edge, probe one node at a time) but NOT for the 'computation survives' demo. For
that you need a multi-channel simultaneous ADC (e.g., Arduino Due 8ch, or a DAQ). Don't skip
this -- it's the difference between demonstrating the mode and demonstrating the computation.

Output is a table of MEASURED numbers. Never hand-write the numbers; let this script produce
them from real data.
"""
import sys, numpy as np

N = 20; M = 2 * N; T = 2000; BURN = 200
VT, WT = 0.4, 1.0; VR, WR = 1.0, 0.4                        # topological / trivial dimerization

# ---------- SIM proxy of the circuit (for pipeline development only) ----------
def ssh_H(v, w, remove=None):
    H = np.zeros((M, M))
    for i in range(M - 1):
        H[i, i + 1] = H[i + 1, i] = (v if i % 2 == 0 else w)
    if remove is not None:
        for k in np.atleast_1d(remove):
            H[k, :] = 0; H[:, k] = 0
    return H

def record_sim(u, v, w, remove=None, rho=0.95):
    H = ssh_H(v, w, remove); W = rho * H / (np.max(np.abs(np.linalg.eigvalsh(H))) + 1e-9)
    win = np.full(M, 0.12); x = np.zeros(M); X = np.zeros((len(u), M))
    for t in range(len(u)):
        x = np.tanh(W @ x + win * u[t]); X[t] = x
    return X[BURN:]

# ---------- HW acquisition (FILL THIS IN for your ADC) ----------
def record_hw(u, remove=None):
    """Return an (len(u)-BURN, n_nodes) array of node voltages while driving with signal u.
    Arduino Due sketch: stream 8 analog reads per input sample over serial; read here with
    pyserial. Sound card (localization only): use sounddevice to play u and record.
    'remove' = which node you physically opened with the switch (for your own logging)."""
    raise NotImplementedError("wire up your ADC here: pyserial (Arduino Due) or sounddevice (2ch)")

# ---------- task, readout, metric ----------
rng = np.random.default_rng(1); u = rng.standard_normal(T)
y = np.zeros(T); y[2:] = u[:-2]; y = y[BURN:]              # recall task y[t]=u[t-2]
ridge = lambda X, yy, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ yy)
nmse = lambda p, yy: float(np.mean((p - yy) ** 2) / (np.var(yy) + 1e-12))

def run(record, label, defect_node=M // 2):
    print(f"\n[{label}] recording clean -> training readout -> freezing -> breaking node "
          f"{defect_node} -> re-measuring")
    Xc = record(u, remove=None)
    wout = ridge(Xc, y)
    clean_err = nmse(Xc @ wout, y)
    Xd = record(u, remove=defect_node)
    frozen_err = nmse(Xd @ wout, y)                        # SAME weights, broken node
    oracle_err = nmse(Xd @ ridge(Xd, y), y)               # retrained on broken board (floor)
    print(f"    clean NMSE = {clean_err:.3f} | after defect (frozen) = {frozen_err:.3f} | "
          f"oracle (retrained) = {oracle_err:.3f} | penalty = {frozen_err - oracle_err:+.3f}")
    return clean_err, frozen_err, oracle_err

if __name__ == "__main__":
    hw = "--hw" in sys.argv
    rec = (lambda uu, remove=None: record_hw(uu, remove)) if hw else None
    print("MODE:", "HARDWARE" if hw else "SIM (pipeline dry-run; swap to --hw with the board)")
    print("=" * 70)
    if hw:
        run(lambda uu, remove=None: record_hw(uu, remove), "TOPOLOGICAL board")
        print("  (rebuild/flip to trivial wiring, rerun) ")
    else:
        run(lambda uu, remove=None: record_sim(uu, VT, WT, remove), "TOPOLOGICAL (sim)")
        run(lambda uu, remove=None: record_sim(uu, VR, WR, remove), "TRIVIAL (sim)")
    print("\nFill RESULTS_TEMPLATE.md with the MEASURED numbers this prints on --hw. "
          "The --sim numbers are a pipeline check, NOT results.")
