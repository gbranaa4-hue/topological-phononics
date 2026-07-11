# Topological protection as a defect-tolerant reservoir primitive: a pre-registered simulation study of when it holds

**gbranaa4-hue** (independent researcher) — *[add full name / ORCID before submission]*
Contact: gbranaa4@gmail.com · Code and data: https://github.com/gbranaa4-hue/topological-phononics
Preprint, July 2026.

---

## Abstract

Physical reservoir computing turns a fixed nonlinear dynamical system into a computational substrate by training only a linear readout. Because analog substrates are imperfect, a natural hope is that *topological* structure -- a robust, quantized invariant -- could make such a reservoir tolerant to component failure. We test this hope on the Su-Schrieffer-Heeger (SSH) resonator chain, the minimal one-dimensional lattice with a topological invariant, and follow the claim down a full scoping ladder under a strict pre-registration discipline: state the test and its pass/fail rule before looking, validate every metric on known cases, split every average, and report the negative as carefully as the positive. Five results follow. (1) The protected zero-energy edge mode is real and its protection is conditional on chiral symmetry. (2) In the idealized tight-binding model on a linear memory task, a frozen readout degrades several-fold more gracefully on the topological chain than a trivial one; the effect is a bulk property with a size threshold and holds to 64 nodes. (3) It transfers to nonlinear computation (NARMA10) but weakens (~2x vs ~5x) and erodes under noise. (4) Topology is not the best route to defect-tolerance -- a generic reservoir with redundant connectivity is more robust; a pre-registered redundancy prediction was refuted and is retracted. (5) In a physically faithful damped, driven, Duffing-nonlinear model the advantage does not transfer in magnitude: a naive spring chain reverses it, and only engineered chiral symmetry removes the reversal -- a pre-registered firm-up lands the compensated case at inconclusive (win-rate 58%, 95% CI 47-69%) while confirming the naive reversal (29%, CI 19-40%). Separately, the trained readout cancels structured (low-rank) noise as a subspace-rejection property orthogonal to topology; only the full-rank thermal floor is irreducible. The earned conclusion is a conditional, not a headline: topological structure protects a reservoir's computation against a dead element exactly when chiral symmetry holds -- cleanly in the idealized model, marginally and unconfirmedly in a realistic device. We argue this is the honest, useful form of the result and discuss why the idealized magnitude does not survive real physics.

## 1. Introduction

Physical reservoir computing (PRC) exploits the intrinsic dynamics of a physical system -- photonic, spintronic, mechanical, or otherwise -- as a fixed, high-dimensional nonlinear map, training only a linear output layer [Jaeger 2001; Maass et al. 2002; Tanaka et al. 2019]. Its appeal is efficiency and the reuse of native physics; its obstacle is that real substrates are noisy, imprecise, and prone to failed elements.

Topological band structures offer quantized invariants that are robust to disorder [Hasan and Kane 2010], and the SSH model [Su, Schrieffer, Heeger 1979] is the minimal system exhibiting a protected zero-energy edge mode. This suggests an attractive hypothesis: a topologically structured reservoir might compute robustly despite a dead or degraded element, giving a route to fault-tolerant analog hardware. Adjacent published work on topologically protected states for neuromorphic computing indicates the idea is in the air. *(A complete related-work review remains to be added before formal submission; the present study is an independent simulation and does not claim priority.)*

We treat the hypothesis not as "does it work?" but as "under precisely what conditions?" -- and we hold ourselves to a measurement discipline in which a result is worth exactly as much as the test that could have broken it and did not. Every experiment is pre-registered with an explicit pass/fail rule; every metric is validated on cases whose answer is known; averages are split to expose worst-case behavior; and results that a sharper test demotes are retracted by name. The complete, reproducible code is public.

## 2. Methods

**Reservoir.** A chain of `2N` coupled resonators with tight-binding Hamiltonian `H` whose hoppings alternate between intra-cell `v` and inter-cell `w`. The topological phase is `w > v`; the trivial phase is `v > w`. We use a matched-dimerization control: parametrize `v = 1 - g`, `w = 1 + g`, so flipping the sign of `g` changes only the topology while holding the perturbation magnitude fixed. Reservoir states evolve as `x_{t+1} = tanh(rho * H_hat * x_t + w_in * u_t)` with `H_hat` spectrally normalized and `rho = 0.95`, or a leaky-integrator variant for the noise and cavity experiments. Drive is uniform across sites unless noted; the readout is a full-state ridge regression.

**Robustness metric.** A removed resonator is a decoupled site. We isolate robustness from raw capability with the PENALTY = (frozen-readout-transfer error) - (retrained-oracle error), where the frozen readout is trained on the intact chain and applied to the damaged one. Penalty ~0 means a defect is absorbed without recalibration; large penalty means it is not.

**Pre-registration.** Each script states its confirming and disconfirming outcomes before execution. Automated verdicts were cross-checked against the raw tables and corrected by hand where wrong; two such corrections (a regularization-induced blow-up and a metric that measured capability rather than robustness) are reported below rather than hidden.

**Physical model (Section 3.7).** To test transfer beyond the abstract map we also simulate a chain of damped, driven, Duffing-nonlinear mechanical oscillators, `x'' + gamma x' + w0^2 x + beta x^3 = -[K x] + w_in u`, integrated by RK4, with the reservoir state sampled from position and velocity. Two coupling models are compared: a naive spring chain (coupling = graph Laplacian, which places the dimerization on the diagonal and breaks chiral symmetry) and a grounding-compensated chain (uniform on-site stiffness, chiral symmetry preserved).

## 3. Results

### 3.1 The edge mode is real, and its protection is conditional

The topological chain has a mode at `|E| = 0.000` localized entirely at the ends (edge-weight 1.00); the trivial chain has none (`|E| = 0.607`, edge-weight 0.05). Over 40 disorder realizations at strength 0.35, hopping (chiral-symmetry-preserving) disorder leaves the edge mode exactly pinned (`|E| = 0.000 +/- 0.000`), whereas on-site (symmetry-breaking) disorder drifts it (`|E| = 0.085 +/- 0.072`). Protection holds only for symmetry-preserving disorder. This chiral conditionality returns as the governing boundary in Section 3.7.

### 3.2 Defect-tolerance in the idealized model (linear task)

Under the matched-dimerization control, flipping the sign of the dimerization flips the robustness with it: the full-set defect penalty is ~7x lower in the topological phase at equal dimerization strength. A confound-free single-defect position sweep (uniform drive) finds the topological chain wins 94% of removal positions, mean penalty +4.6 versus +26.2 (~5.7x), perfectly left-right symmetric (a validity check), and largest at the boundaries.

### 3.3 A bulk property with a size threshold

With one to four simultaneous dead resonators the topological penalty stays 4-5x below trivial (92-100% of placements). Sweeping chain length from 8 to 64 nodes, the advantage requires a bulk: it is absent at N=4 (0% win-rate, no interior), crosses over at N=6 (56%), and for N>=8 wins 87-100% of placements (mean 95%) with 5-8x lower penalty, non-diminishing through 64 nodes. So the effect is not a small-system artifact.

### 3.4 Noise as a feature: the readout cancels structured noise

Injecting structured (correlated, low-rank) versus random (full-rank white) noise at matched power, the trained readout cancels structured noise if and only if it sees it linearly. Structured noise at the readout stage is cancelled perfectly at any amplitude (memory capacity flat even when the noise exceeds the signal); the same noise injected inside the recurrent dynamics is cancelled only while small and reverses at large amplitude. Rank is the unifying axis -- random noise is full-rank structured noise; an M-node readout nulls low-rank interference (100% at rank 1, ~88% at rank M/2) and fails at full rank (~36%). Plain ridge is already at the linear ceiling: a noise-aware oracle that projects out the known noise subspace buys <=3%, and at full rank is worse, because nulling a full-rank subspace annihilates the signal too. The cancellation margin scales approximately linearly with node count. This is a readout/subspace property, orthogonal to topology, and it composes with the defect result: with a dead resonator present the readout still cancels structured noise (99% of memory capacity kept versus 24% for random). The core fact -- a linear readout performs subspace rejection of low-rank noise but cannot touch full-rank -- is established signal-processing knowledge; what is new here is the clean measurement in this reservoir, the dynamics-stage reversal, and the node-scaling of the margin.

### 3.5 Nonlinear task: the advantage transfers, but weakens

On NARMA10, the standard nonlinear reservoir benchmark, all reservoirs clear the capability bar (topological NMSE 0.55, trivial 0.56, random echo-state network 0.40 -- the random network is the better computer). A first pass with weak regularization produced blow-up penalties; this was diagnosed as an instrument fault (large readout weights amplifying an operating-point shift) and corrected with proper regularization and robust statistics. With that correction the topological chain still beats trivial: 67% paired win-rate, ~2.3x lower median penalty. The claim upgrades from "protects memory" to "protects computation" -- but weaker than the linear task (67%/2.3x versus 94%/5x), and it erodes under noise, the penalty ratio falling from 10.1x (no noise) to 1.9x as random noise rises.

### 3.6 Architecture control: topology is not the best route

Against a random echo-state network at matched node count, the topological chain is not the most defect-tolerant (median penalty 9.5 versus 2.7 for the random network). A pre-registered prediction -- that penalty would fall with redundancy and that the SSH chain would match a random graph at equal sparsity -- was refuted and is retracted by name. The data show penalty rising with connectivity, and the SSH chain (a path, in which every interior site is a cut vertex) is uniquely fragile relative to a random graph of equal edge count. We flag a confound: the frozen-readout penalty partly rewards decoupling, so cross-architecture comparisons are not clean. The only fully controlled comparison -- topological versus trivial within the SSH chain -- is where topology wins consistently (~2.8x); comparisons across architectures fail or are confounded.

### 3.7 Physical device model: the boundary transfers, the magnitude does not

Replacing the abstract map with the damped/Duffing oscillator model, capability required tuning (uniform drive was vacuous; random per-site drive with stronger damping cleared it -- counter-intuitively, more damping recalls a short delay better, because light damping lets past inputs superimpose and blur). Two mechanical models were run as a designed-to-kill test with a damage-matched control. In the naive spring chain (Laplacian coupling), a dead resonator lowers its neighbors' on-site stiffness -- an on-site, symmetry-breaking perturbation, exactly the kind Section 3.1 shows topology does not protect -- and the advantage reverses. In the grounding-compensated chain (uniform on-site stiffness, chiral symmetry preserved), the reversal is removed.

A pre-registered firm-up (6 seeds x 13 interior sites = 78 paired samples; rule fixed before looking: a real positive requires the win-rate 95% CI lower bound above 50%) gives:

| model | chiral symmetry | topo win-rate (95% CI) | median penalty topo / trivial | verdict |
|---|---|---|---|---|
| naive | broken | 29% (19-40%) | 6.21 / 3.19 | reversal, confirmed |
| compensated | preserved | 58% (47-69%) | 3.37 / 5.25 | inconclusive (CI spans 50%) |

The compensated case does not clear the pre-registered bar and is reported as inconclusive, not upgraded. But the two rows together confirm the boundary: flipping the chiral condition flips the sign of the effect at equal cost -- break it and topology loses; preserve it and topology leans to win. Same cost, opposite outcome. Chiral symmetry is confirmed as the governing condition; the idealized ~5x magnitude does not transfer to a realistic device.

## 4. Discussion

The result is a conditional: topological structure protects a reservoir's computation against a dead element exactly when chiral symmetry holds -- several-fold in the idealized model, weaker under nonlinearity and noise, beaten by generic redundancy, and only marginally (and not significantly) recovered in a chiral-engineered physical model. Two observations organize why the big claim shrinks.

First, the protected object is a topological *invariant* (the edge-mode energy), which is one step removed from the *computation*. Robust protection of the invariant does not automatically confer robust computation; the computational benefit is a separate, weaker, conditional matter. This mirrors a recurring pattern in physical reservoir studies where a protected dynamical feature only half-transfers to the trained readout.

Second, the transfer to hardware is governed by symmetry preservation. A mechanical realization must actively engineer chiral symmetry (uniform on-site stiffness, e.g. grounding compensation) merely to avoid a *reversal* of the effect; without it, a dead element manifests as on-site disorder, precisely the perturbation class the protection excludes. This is a concrete and, we think, useful design constraint: a naive build would show no benefit, and even a carefully compensated one should expect a modest effect, not the idealized several-fold.

We therefore present a candidate primitive with a stated, tested boundary, not a universal claim. A rule with a boundary is knowledge; a rule sold as universal is marketing.

## 5. What did not hold (ledger)

- A pre-registered redundancy prediction (Section 3.6) was refuted (penalty rises with connectivity; the SSH path is uniquely fragile) and is retracted.
- The compensated-chiral physical advantage (Section 3.7) is inconclusive (58%, CI 47-69%), not a demonstrated positive; reported as such, not spun from the favorable median.
- An earlier "edge-tap immunity" was capability, not robustness (frozen-readout penalty ~0 for both phases); retracted in favor of the full-state penalty.
- A "near-edge reversal" and a "weak/strong-bond parity mechanism" were artifacts of single-site drive; both vanish under uniform drive.
- A proposed topology x high-Q cavity complementarity was tested and not supported.

## 6. Limitations and scope

This is a simulation study. Sections 3.1-3.6 use a tight-binding or leaky-ESN model; Section 3.7 adds damping and Duffing nonlinearity but remains a simulation of an idealized lattice, not a fabricated device. Tasks are linear memory and NARMA10. The central result is a fully-scoped conditional, and the honest device implication is that a hardware build would need deliberate chiral-symmetry engineering just to avoid a reversal, with a modest expected effect. This is not a quantum computer and makes no quantum claim.

## 7. Data and code availability

All code, the extended lab report, the interactive dashboard, and the animations are public and reproducible at https://github.com/gbranaa4-hue/topological-phononics. Each experiment script is pre-registered and self-testing; `phononic_methods.py` includes a self-test that reproduces the reported numbers.

## References

- Su WP, Schrieffer JR, Heeger AJ (1979). Solitons in polyacetylene. *Physical Review Letters* 42, 1698.
- Hasan MZ, Kane CL (2010). Colloquium: Topological insulators. *Reviews of Modern Physics* 82, 3045.
- Jaeger H (2001). The "echo state" approach to analysing and training recurrent neural networks. *GMD Report* 148.
- Maass W, Natschlager T, Markram H (2002). Real-time computing without stable states. *Neural Computation* 14(11), 2531-2560.
- Tanaka G, Yamane T, Heroux JB, et al. (2019). Recent advances in physical reservoir computing: A review. *Neural Networks* 115, 100-123.
- Atiya AF, Parlos AG (2000). New results on recurrent network training. *IEEE Transactions on Neural Networks* 11(3), 697-709. *(NARMA benchmark.)*

*Conducted under the gbranaa4-hue method: pre-register the test, validate the instrument, split the average, find the boundary, and write the "no" as carefully as the "yes."*
