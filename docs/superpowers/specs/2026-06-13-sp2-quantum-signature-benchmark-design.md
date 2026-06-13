# SP2 — Predicative Falsification Benchmark for Quantum Signatures in LMs — Design Spec

**Status:** design, pending implementation plan
**Date:** 2026-06-13
**Sub-project:** SP2 of the "usable quantum language models" programme (SP1 = quantum backend, done — PR #146; SP3 = QLM⇄vindex compiler/distiller).

## Objective

Experimentally test — **by attempting to falsify** — whether trained LLM attention heads carry quantum structure *beyond what a generic random matrix already exhibits*. The benchmark does **not** seek to confirm "LLMs are quantum"; it seeks **conclusive negatives** (ruling hypotheses out, having ruled out experimental error). It also re-adjudicates the earlier, null-free "raw SmolLM2 attention is not quantum-compressible" claim, which this design shows was under-determined.

## Methodological principles (these govern the whole design)

1. **Predicativity.** No comparison may be circular. Every empirical source is pushed through the *identical* `coupling → from_matrix → partial-trace → witness` pipe; controls are constructed *independently* of the property being tested. The signal is **real − null**, not "real vs a baseline defined by the absence of the effect."

2. **Dual testing of dualizable criteria.** For each dualizable axis (classical↔quantum, independent↔dependent, separable↔entangled, …) neither pole is assumed. Two **analytically-known poles** bracket the axis; every empirical source is *placed on the axis by measurement*. If you assume one pole you must run the test that detects its dual — so the assumption is falsifiable.

3. **Falsification-first; conclusive negatives are the product.** Positive results (a real head "violating" a bound) are confirmation-bias-prone and, given the embedding confound, cheap — reported only as *"not ruled out; needs a stronger null."* The valued outputs are **conclusive negatives** (e.g. "this reduction is provably separable," or "real is statistically indistinguishable from the random null"), each **licensed by positive controls** that prove the witness can detect the effect and does not false-positive.

## The confound this design exists to defeat

`from_matrix(C)` normalizes any real matrix and reads it as a bipartite pure state; a *generic* matrix so read is highly entangled (for 2-qubit pure states max-CHSH and negativity are monotone in concurrence). Consequences:
- A real attention coupling read this way will *generically* "look entangled / violate CHSH" — **not** because of training, but because non-degenerate matrices are generically entangled under the embedding.
- The earlier `entanglement_entropy_bipartition(from_matrix(C))` equals the spectral entropy of `C`'s singular values; a Gaussian `C` has a near-maximal (Marchenko–Pastur) singular spectrum → also reads "not quantum-compressible." **The prior finding had no null and almost certainly does not distinguish SmolLM2 from a random matrix.**

The fix is principle 1: the random-matrix null passes through the identical pipe, and the analytic poles (principle 2) calibrate the witnesses.

## Architecture

A battery of **exact, deterministic witnesses** (functions of a density matrix — no sampling noise) on a shared pipe, evaluated over **two analytic poles** (positive-control gates) and **two empirical sources** (placed by measurement).

### Shared pipe
`source → C (head_dim×head_dim, or a synthesized state) → from_matrix → partial trace to a pre-registered 2-qubit pair → ρ₂ (4×4 Hermitian) → witnesses`.

### Witnesses (v1)
| Axis | Witness | Conclusive direction | Bounds |
|---|---|---|---|
| separable ↔ entangled | **Negativity** (PPT / Peres–Horodecki) — *the lead* | negativity = 0 ⟺ **provably separable** (necessary & sufficient for 2-qubit) — a conclusive negative | 0 (separable) … 0.5 (max) |
| classical ↔ quantum | **CHSH via Horodecki** `2√M`, `M` = sum of top-2 eigenvalues of `TᵀT`, `T_ij = Tr[ρ₂ σ_i⊗σ_j]` | non-violation inconclusive; violation confound-prone (reported, not claimed) | 2 (classical) … 2√2 |
| independent ↔ dependent | **Mutual information** `S(ρ_A)+S(ρ_B)−S(ρ₂)` | MI = 0 ⟺ **product / independent** — a conclusive negative | 0 … 2 bits |

All three reuse `larql_hilbert::eig::hermitian_eigenvalues` (partial transpose eigenvalues for negativity; subsystem entropies for MI).

### Positive-control gates (analytic poles — must pass or the witness is void)
- **Classical/independent/separable pole:** product & dephased QLM states → negativity 0, MI 0, CHSH ≤ 2.
- **Quantum/dependent/entangled pole:** Bell/GHZ/Dicke QLM states → negativity > 0, MI > 0, CHSH = 2√2 (Bell).

These rule out experimental error: they prove each witness detects the effect when present and does not false-positive. A negative on real data is interpretable only after its witness passes these gates.

### Empirical sources (placed by measurement — neither assumed)
- **Random-matrix null:** shape/scale-matched Gaussian `W_Q,W_K → C`, same pipe. An *unknown* (the advisor's point: it likely lands on the quantum side via the embedding — that is data, not a flaw).
- **Real SmolLM2 heads:** the 480 per-head couplings.

### Pre-registration (rules out a hidden second maximization)
The 2-qubit reduction is **fixed before looking at data**: partial-trace the `from_matrix` state down to **qubits {0, 1}** (the two most-significant row bits, big-endian) — one deterministic pair, applied identically to poles, null, and real heads. **No max-over-pairs** (that would stack a maximization on Horodecki's own and inflate violation rates / create multiple-comparison bias). A sensitivity sweep over *all* pairs, if ever run, is reported as the full null-calibrated distribution over pairs, never as the most-violating pair.

## What the benchmark reports

Per witness, over the head population: the **real − null** distribution and where real and null fall relative to **both** analytic poles. The headline outputs are the **conclusive negatives**:
- *"Real heads are statistically indistinguishable from the random null on witness W"* (real − null ≈ 0, with a stated effect-size/power floor and the random ensemble as the null distribution) → falsifies "training induces quantum structure beyond generic-matrix-ness" for W.
- *"Real head h's 2-qubit reduction is provably separable"* (negativity 0) → conclusive non-entanglement for that head.

A real-≠-null result leaning quantum is reported as **"not ruled out — requires the singular-value-matched null and replication before any confirmation,"** never as a positive finding.

This directly re-adjudicates "not quantum-compressible": the same entanglement/gap quantities, now with the random null, answer *is SmolLM2 distinguishable from random?* rather than assuming the bare number is meaningful.

## Components

- **larql-hilbert (pure, additive):** `partial_trace_2q(state, pair) -> Array2<Complex64>` (4×4 ρ₂); `negativity(rho2) -> f64` (PPT via partial transpose + `hermitian_eigenvalues`); `correlation_matrix(rho2) -> Array2<f64>` + `chsh_max(rho2) -> f64` (Horodecki); `mutual_information(rho2) -> f64`. Positive-control unit tests (Bell / product / GHZ exact values).
- **larql-cli (bridges + real weights):** a benchmark runner — for each head, compute the battery on real `C` and a shape-matched random null; aggregate real−null over the head population; emit a report (JSON + summary). Pre-registered reduction. Optionally a `larql quantum-signature <vindex>` command mirroring `larql entanglement`.
- **Harness:** demo (the analytic poles + a small synthetic real-vs-null illustration, CI-safe) + integration tests (positive-control gates; real-vs-null on the SmolLM2 vindex when present, gated like the existing real-vindex tests).

## Out of scope (v1)

- The compressible↔incompressible axis as a *fourth* dual witness (the re-adjudication reuses the existing entanglement/gap, so the axis-4 formalization is a follow-on).
- The **singular-value-matched null** (controls for the spectrum, isolating structure beyond it) — a stronger null, added once v1's Gaussian-null result is in hand.
- Any physical-nonlocality *claim*. This is a structural/spectral comparison of weight matrices read as states; "CHSH violation" here is a property of the embedded coupling, calibrated against poles and the null — never a certification that the transformer is a quantum device.
- SP3 (QLM⇄vindex compiler/distiller).

## Testing strategy

TDD. The witnesses are exact, so unit tests assert against analytic values (Bell negativity = 0.5, CHSH = 2√2, MI = 2 bits; product/dephased → 0, 0, ≤2). The first implementation checkpoint runs **the advisor's verification**: confirm a Gaussian-random `C` violates via the embedding — if it does (expected), the random null is *mandatory* and the diagonal-classical baseline is formally disqualified, validating the whole predicative design. Real-vindex tests are `LARQL_TEST_VINDEX`-gated; the synthetic poles + null run always in CI.
