# Topological reservoir circuit — measurement log (TEMPLATE)

**Rule of this file:** every number in a "measured" column comes from `measure_and_train.py --hw`
or a meter reading. The "predicted" column comes from `circuit_predict.py` (lossless model).
**Do not type a number into a measured cell that you did not measure.** Empty is honest; invented is not.

Build date: ____  ·  Board: 8-node breadboard  ·  Components: C1=8.2nF, C2=1.0nF, gyrator Cg=12nF/Rg=15k (L≈2.7H), grounding-compensation caps per node.

---

## A. Edge mode exists and localizes (drive one end, probe each node)

Drive frequency: predicted band center **~1010 Hz**. Measured resonance: ______ Hz.

| node | predicted \|amp\| (topological) | **measured (topological)** | predicted (trivial) | **measured (trivial)** |
|---|---|---|---|---|
| 1 (edge) | 1.00 | | ~0.5 | |
| 2 | 0.00 | | ~0.5 | |
| 3 | 0.12 | | ~0.5 | |
| 4 | 0.01 | | ~0.5 | |
| 5 | 0.01 | | ~0.5 | |
| 6 | 0.12 | | ~0.5 | |
| 7 | 0.00 | | ~0.5 | |
| 8 (edge) | 1.00 | | ~0.5 | |

Pass criterion: topological amplitude **decays from the driven end** (localized); trivial does **not**.
Note the real Q / peak width: ______  · frequency shift from tolerance: ______ %.

---

## B. Defect tolerance (train → freeze readout → open a node → re-measure)

Task: recall `u[t-2]`. Defect: open node ____ with the switch.

| config | clean NMSE | **NMSE after defect (frozen weights)** | oracle (retrained) | **penalty = frozen − oracle** |
|---|---|---|---|---|
| **topological** | | | | |
| **trivial** | | | | |

Advantage = (trivial penalty) / (topological penalty) = ______ ×.
(Sim proxy gave ~2× at a single center defect, up to ~5× averaged / near boundaries — **your hardware number replaces this**.)

Repeat for a few defect positions and 2–3 simultaneous defects; log each:

| # defects / positions | topological penalty | trivial penalty | advantage |
|---|---|---|---|
| 1 @ node __ | | | |
| 2 @ nodes __,__ | | | |
| 3 @ nodes __,__,__ | | | |

---

## C. Honest scope (fill in after measuring)

- Q factor / damping (real, not in the model): ______
- Component tolerance spread you observed: ______ %
- Did the topological advantage survive real noise/tolerance? ______
- Where it FAILED or surprised you: ______

---

## D. Claims ledger — what the data lets you say

Tick only if the measurement supports it. If unticked, **you may not write it in the paper.**

- [ ] "The circuit has a resonance at ~____ Hz" — needs A.
- [ ] "A localized edge mode exists in the topological wiring, absent in the trivial" — needs A, topological amp decaying, trivial flat.
- [ ] "Under a dead node with a frozen readout, the topological circuit's error rose ____× less than the trivial one" — needs B, with your measured numbers.
- [ ] "The advantage persists under N simultaneous defects" — needs the multi-defect rows.

**You may NOT claim** (no data supports these — from the honesty pass):
- ~~"first / born fault-tolerant / cannot crash"~~ — prior art exists (Nature Nanotech 2024); protection is conditional and finite.
- ~~any winding-number / grant / ASIC / MNIST / keyword-spotting / medical / autopilot number~~ — not measured here.

Honest one-liner you *can* earn: *"An accessible ~$60 circuit demonstration that topological wiring gives a measured ____× more graceful degradation under component failure in a linear reservoir task, consistent with simulation and building on prior topological-neuromorphic work."*
