# Topological protection as a defect-tolerant reservoir primitive — and the exact conditions under which it holds

**gbranaa4-hue — draft v0.2, 2026-07-10** (prepared with an AI research assistant; every number reproducible from the cited scripts). Conducted under the [gbranaa4-hue method](../gbranaa4-hue-method.pdf): pre-register the test, validate the instrument, split the average, find the boundary, and write the "no" as carefully as the "yes."

## Abstract

An SSH resonator chain — the minimal one-dimensional lattice with a topological invariant — is used as a physical reservoir, and the question *"does topological structure make an analog reservoir tolerate a dead element?"* is pursued down the full scoping ladder until it stops giving. Five things are established. (1) The protected zero-energy **edge mode is real** and its protection is **conditional on chiral symmetry** (pinned under coupling disorder, drifts under on-site disorder). (2) In the **idealized tight-binding model** on a **linear** memory task, a *frozen* readout degrades **several-fold more gracefully** on the topological chain than a trivial one — controlled by flipping the dimerization sign, broad (94% of defect positions), robust to multiple defects, and a **bulk property with a size threshold** (absent below N≈6, then non-diminishing through 64 nodes). (3) The advantage **transfers to nonlinear computation** (NARMA10) but **weakens** (~2× / 67% vs ~5× / 94%), and **erodes under noise**. (4) Topology is **not the best route to defect-tolerance** — a generic reservoir with redundant connectivity is more robust; the SSH chain is a *path*, uniquely fragile because every interior site is a cut vertex. A pre-registered redundancy prediction was **refuted and is retracted**. (5) In a **physically faithful damped/driven Duffing model**, the advantage **does not transfer in magnitude**: a naive spring chain **reverses** it (a dead element becomes symmetry-breaking on-site disorder), and only **engineered chiral symmetry** (grounding compensation) removes the reversal — a **pre-registered firm-up** lands the compensated case at *inconclusive* (win-rate 58%, 95% CI 47–69%), while confirming the naive **reversal** (29%, CI 19–40%). Separately, the trained readout **cancels structured noise** (a readout/subspace property, orthogonal to topology; only the full-rank thermal floor is irreducible), and the two mechanisms **compose** (cancellation survives a defect). The earned conditional: *topological structure protects a reservoir's computation against a dead element **exactly when chiral symmetry holds** — cleanly in the idealized model, marginally and unconfirmedly in a realistic device.* Several sub-claims were tested and **retracted** (§ledger).

## 1. Setup

Chain of `2N` coupled resonators; hoppings alternate `v` (intra-cell) / `w` (inter-cell); tight-binding matrix `H`, mode frequencies = eigenvalues. **Topological** `w>v`; **trivial** `v>w` (matched-dimerization control: flip the sign of `g` in `v=1−g, w=1+g`, changing only the topology). Reservoir `x_{t+1}=tanh(ρ·Ĥ·x_t + w_in·u_t)` (or a leaky variant), `Ĥ` spectrally normalized, **uniform drive**, full-state ridge readout. **Robustness metric — PENALTY** = frozen-readout-transfer NMSE − retrained-oracle NMSE — isolates *robustness* from *capability*. A removed resonator = a decoupled site. Every claim was pre-registered per script; auto-verdicts were cross-checked against the raw tables and corrected by hand.

## 2. The edge mode is real, and its protection is conditional

`ssh_topological_reservoir.py`: topological chain has a mode at `|E|=0.000` localized entirely at the ends (edge-weight `1.00`); trivial has none (`|E|=0.607`, edge-weight `0.05`). Over 40 disorder realizations (strength 0.35): **hopping** (chiral-preserving) disorder leaves the edge mode `|E|=0.000±0.000` — exactly pinned; **on-site** (symmetry-breaking) disorder drifts it to `0.085±0.072`. **Protection holds only for symmetry-preserving disorder.** This chiral conditionality returns as the governing boundary in §9.

## 3. Defect-tolerance is topological (idealized model, linear task)

Matched-dimerization control (`phase_transition_defect.py`, `cross_device_transfer_tight.py`): flipping the sign of the dimerization flips the robustness with it — full-set defect penalty ~7× lower in the topological phase at equal dimerization strength. Confound-free single-defect sweep with uniform drive (`ssh_defect_position.py`): the topological chain **wins 94% of removal positions**, mean penalty **+4.6 vs +26.2 (~5.7×)**, perfectly left–right symmetric (a validity check), and **largest at the boundaries** (up to ~95× at near-edge positions where a defect catastrophically wrecks the trivial spectrum).

## 4. Robust to multiple defects, and a bulk property with a size threshold

`ssh_multiple_defects.py`: with 1–4 simultaneous dead resonators the topological penalty stays **4–5× below trivial**, winning 92–100% of placements. `ssh_scaling.py` (8→64 nodes): the advantage **requires a bulk** — absent at N=4 (0% win-rate, no interior), crossing over at N=6 (56%), and for **N≥8 winning 87–100% (mean 95%) with 5–8× lower penalty, non-diminishing through 64 nodes.** So ~5× is not a small-system artifact; it is a bulk topological property above a size threshold.

## 5. Mechanism (honest)

The advantage tracks the topological dimerization and is largest at boundaries — but the phase-transition sweep shows it is **not** a clean edge-mode-only step (penalty minimizes at the gapless transition, non-monotone inside the phase). Honest reading: **a mix of edge-mode support and bulk dimerization.** Topology protects boundaries, so near-boundary defects — exactly where the trivial chain fails worst — are where the protection pays off most.

## 6. High-Q cavities and noise (secondary)

`reservoir_cavity_v2.py`, leaky-integrator reservoir (leak = cavity time constant, noise inside the leak), metric = memory capacity (MC). **A clean noise/capacity crossover:** at zero noise low-Q has more raw memory (MC 3.6 vs 2.0) but *collapses* under noise (→0.5), while high-Q degrades slowly (→0.7); **above noise ≈0.1 the high-Q reservoir retains more usable memory.** High-Q trades raw capacity for noise-robustness. Orthogonal to §3–4; a proposed topology×high-Q *complementarity* was tested and **not supported** (§ledger).

## 6b. Noise as a feature — the readout cancels structured noise

`noise_feature.py`, `noise_feature_stage.py`, `rank_margin.py`, `rank_margin_aware.py`, `scaling_margin.py`. Injecting **structured** (correlated, low-rank) vs **random** (full-rank white) noise at matched power, the trained readout **cancels structured noise if and only if it sees it linearly**:
- **Readout-stage structured noise is cancelled perfectly, at any amplitude** (MC flat at 2.32 even when noise exceeds the signal; advantage grows to 6.2× over random). The *same* structured noise injected **inside the recurrent dynamics** is cancelled only while small and **reverses** at large amplitude (a big common-mode drive saturates the nonlinearity irreversibly).
- **Rank is the unifying axis** — random noise *is* full-rank structured noise. An M-node readout nulls low-rank interference (100% at rank-1, ~88% at rank M/2) and fails at full rank (~36%).
- **Plain ridge is already at the linear ceiling** — a noise-aware oracle that projects out the known noise subspace buys ≤3%, and at full rank is *worse* (nulling a full-rank subspace annihilates the signal too). The limit is subspace geometry, not readout cleverness — the same wall as the physical thermal floor, from the algebra side.
- **The cancellation margin scales ~linearly with node count** (`scaling_margin.py`): a fixed 4 interferers go from 82% retained (M=8) to 99% (M=48); a proportional M/2 interferers stay ~90%.

**This is a readout/subspace property, orthogonal to topology**, and it **composes** with the defect result: with a dead resonator present the readout still cancels structured noise (99% of MC kept vs 24% for random; `combined.py`). The honest core fact — a linear readout performs subspace rejection of low-rank noise but cannot touch full-rank — is established signal-processing knowledge; what is new here is the clean measurement in this reservoir, the dynamics-stage reversal, the rank margin, and its node-scaling.

## 7. Nonlinear task — the advantage transfers, but weakens

`narma_test.py`. On **NARMA10** (the standard nonlinear reservoir benchmark; 10-step nonlinear memory, product terms), all reservoirs clear the capability bar (topological NMSE 0.55, trivial 0.56, random ESN 0.40 — the ESN is the better *computer*). With a regularized readout and robust statistics (a first pass with `λ=1e-6` produced blow-up penalties and was corrected — the instrument, not the world), the topological chain still beats trivial: **67% paired win-rate, ~2.3× lower median penalty.** The claim upgrades from "protects memory" to "protects **computation**" — but the advantage is **weaker than on the linear task** (67%/2.3× vs 94%/5×). It **erodes under noise** (`combined.py`): the frozen-penalty ratio falls 10.1× (no noise) → 4.2× → 1.9× as random noise rises.

## 8. Architecture control — topology is not the best route (a retracted prediction)

`narma_redundancy.py`. Against a **random echo-state network** at matched node count, the topological SSH chain is **not** the most defect-tolerant (median penalty topological 9.5 vs random ESN 2.7). **Pre-registered prediction — that penalty would fall with redundancy and SSH ≈ random at matched sparsity — was refuted** and is retracted by name. The data: penalty *rises* with connectivity (random 15e→3.5, 60e→12.9), and SSH-topological (10.5) is worse than a random 15-edge graph (3.5) at matched edge count. Corrected reading: the SSH chain is a **path**, uniquely fragile because every interior site is a **cut vertex** (removing one splits the lattice); a random tree has low-impact leaves. **Caveat (stated, not hidden):** the frozen-readout penalty partly *rewards decoupling* (sparser → more independent nodes → losing one matters less), so all cross-architecture comparisons are confounded. The **only** unconfounded comparison — same structure, edges, and task — is **topological vs trivial within the SSH chain**, and there topology wins consistently (10.5 vs 29.4, ~2.8×). Everything beyond that fails or is confounded.

## 9. Physical device model — the boundary transfers, the magnitude does not

`duffing_device.py`, `firm_chiral.py`. Replacing the abstract tight-binding map with a physically faithful chain of **damped, driven, Duffing-nonlinear coupled oscillators** (2nd-order ODEs, RK4; state = sampled position+velocity). Capability required tuning (uniform drive was vacuous; random per-site drive + higher damping cleared it — counter-intuitively, *more* damping recalls a short delay better, since light damping blurs past inputs). Two mechanical models, run as a **designed-to-kill test with a damage-matched control**:

- **Naive spring chain** (coupling = graph Laplacian): a dead resonator lowers its neighbors' on-site stiffness — an **on-site (symmetry-breaking) perturbation**, exactly what §2 shows topology does *not* protect; the dimerization also sits on the diagonal, breaking chiral symmetry outright. **The advantage reverses.**
- **Compensated chain** (grounding springs hold on-site stiffness uniform; only off-diagonal coupling alternates — chiral symmetry preserved; the mechanical twin of the grounding-compensation caps `circuit_predict.py` flagged as *required*). **The reversal is removed.**

**Pre-registered firm-up** (`firm_chiral.py`, 6 seeds × 13 interior sites = 78 paired samples; rule set before looking: real positive *only if* win-rate 95%-CI lower bound > 50%):

| model | chiral symmetry | topo win-rate (95% CI) | median penalty topo/triv | verdict |
|---|---|---|---|---|
| naive | broken | **29% (19–40%)** | 6.21 / 3.19 | **reversal — confirmed** |
| compensated | preserved | **58% (47–69%)** | 3.37 / 5.25 | **inconclusive** (CI spans 50%) |

The compensated case **does not clear the pre-registered bar** — reported as *inconclusive*, not upgraded. But the two rows together **confirm the boundary**: flipping the chiral condition flips the *sign* of the effect at equal cost (break it → topo loses, 29%, medians topo-worse; preserve it → topo leans to win, 58%, medians topo-better). *Same cost, opposite outcome* — the signature of a true boundary. **Chiral symmetry is confirmed as the governing condition; the idealized 5× magnitude does not transfer to a realistic device.**

## Honest ledger — what did NOT hold

- **A pre-registered redundancy prediction (§8)** — that defect-penalty falls with connectivity and SSH ≈ random at matched sparsity — was **refuted** (penalty *rises* with edges; SSH-path is uniquely fragile) and is retracted by name.
- **The compensated-chiral physical advantage (§9)** is **inconclusive** (58%, CI 47–69%), not a demonstrated positive — reported as such, not spun from the favorable median.
- **Edge-tap "immunity" was capability, not robustness** — the frozen-readout penalty is ~0 for both phases; retracted in favor of the full-state penalty.
- **A "near-edge reversal"** and a **"weak/strong-bond parity mechanism"** were artifacts of single-site drive; both vanish under uniform drive.
- **Topology × high-Q *complementarity*** was tested and **not supported** — topological and trivial MC are within noise under coupling disorder, on-site disorder, and a defect (topology's benefit is frozen-readout transfer, not raw capacity; and SSH is chiral, guarding coupling not on-site drift).
- **"Device-invariant / calibration-free reservoir"** framing is adjacent to published topological-neuromorphic work and is not claimed.

## Scope and limits

Simulation throughout. §2–8 use a tight-binding (or leaky-ESN) model; §9 adds damping and Duffing nonlinearity but remains a simulation of an idealized lattice, not a fabricated device. Tasks are linear memory and NARMA10. The central result is a **fully-scoped conditional**, not a universal claim: *topological structure protects a reservoir's computation against a dead element exactly when chiral symmetry holds — several-fold in the idealized model, weaker under nonlinearity and noise, beaten by generic redundancy, and only marginally (and not significantly) recovered in a chiral-engineered physical model.* A hardware build would need deliberate chiral-symmetry engineering merely to avoid a reversal, and should expect a modest effect, not 5×.

## Reproducibility

`ssh_topological_reservoir.py` (edge mode + symmetry-conditional protection) · `phase_transition_defect.py`, `cross_device_transfer_tight.py` (advantage tracks the invariant, matched control) · `ssh_defect_position.py` (position/mechanism, confound-fixed) · `ssh_multiple_defects.py`, `ssh_scaling.py` (multi-defect + bulk scaling 8→64) · `reservoir_cavity_v2.py` (high-Q noise/MC crossover; the topology×cavity null) · `noise_feature.py`, `noise_feature_stage.py`, `rank_margin.py`, `rank_margin_aware.py`, `scaling_margin.py` (§6b noise-as-a-feature) · `narma_test.py` (§7 nonlinear) · `narma_redundancy.py` (§8 architecture control) · `combined.py` (composition + erosion under noise) · `duffing_device.py`, `firm_chiral.py` (§9 physical model + pre-registered chiral firm-up) · `phononic_methods.py` (verbatim methods + self-test), `dashboard.py`, `animate.py` (interactive + shareable demos). All pre-registered; corrected auto-verdicts and retractions noted inline and in the ledger.
