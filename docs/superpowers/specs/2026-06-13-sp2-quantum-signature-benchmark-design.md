# SP2 — Predicative, Over-Constrained Falsification Apparatus for Quantum Signatures in LMs — Design Spec

**Status:** design, pending implementation plan
**Date:** 2026-06-13
**Sub-project:** SP2 of the "usable quantum language models" programme (SP1 = quantum backend, done — PR #146; SP3 = QLM⇄vindex compiler/distiller).

## Objective

Experimentally test — **by attempting to falsify** — whether trained LLM attention heads carry quantum structure *beyond what a generic random matrix already exhibits*, with an apparatus that is **formally well-posed**: enough independent constraints that the classical/quantum verdict is uniquely (over-)determined rather than inconclusive, inconsistent, under-determined, over-determined, or incomplete. It re-adjudicates the earlier, null-free "raw SmolLM2 attention is not quantum-compressible" claim, which this design shows was under-determined.

## Governing principles (non-negotiable, in priority order)

0. **Formal correctness over engineering.** Where formal correctness of the theoretical/experimental apparatus conflicts with software-engineering optimization, scope narrowing, or convenience, we side **unambiguously with formal correctness.** Scope is determined by what makes the experiment well-posed, not by build cost. (Implementation may be *staged*; conclusions may not be drawn from a partial apparatus.)
1. **Predicativity.** No circular comparisons. Every empirical source flows through the *identical* `coupling → embed → reduce → witness` pipe; controls are constructed *independently* of the property tested. The signal is **real − null**, never "vs a baseline defined by the effect's absence."
2. **Dual testing of dualizable criteria.** Neither pole of any dualizable axis is assumed; two *analytic* poles bracket the axis and every empirical source is *placed by measurement*. The random null is an *unknown to be placed*, not the classical anchor.
3. **Falsification-first; conclusive negatives are the product.** Positive results are confirmation-bias-prone and (under the embedding confound) cheap → reported only as "not ruled out." Valued outputs are **conclusive negatives**, each **licensed by positive controls** that prove the witness detects the effect and does not false-positive.
4. **Over-constraint via multi-dimensional independent checks.** A non-trivial classical/quantum experiment needs *multiple constraints on its degrees of freedom* to be statable and solvable with a unique/robust answer. We therefore require a **battery of independent witnesses** and **multiple independent nulls**, plus the **logical-implication lattice among witnesses** as apparatus self-checks. Maximizing the number of independent refutation surfaces is the design goal; a single elementary test is rejected as inconclusive by construction.

## The confound this apparatus defeats

`from_matrix(C)` normalizes any real matrix and reads it as a bipartite pure state; a *generic* matrix so read is highly entangled. So a real coupling will *generically* "look entangled / violate CHSH" — not from training but from the embedding. And `entanglement_entropy_bipartition(from_matrix(C))` equals the spectral entropy of `C`'s singular values, which is near-maximal for a Gaussian (Marchenko–Pastur) matrix → the prior "not quantum-compressible" finding **had no null and does not distinguish SmolLM2 from a random matrix.** The apparatus below removes this by passing independently-constructed nulls through the identical pipe and cross-checking multiple witnesses.

## The degrees of freedom that must be constrained

For the verdict to be well-posed, each DOF is pinned and cross-checked:
| DOF | Unconstrained failure | Constraint |
|---|---|---|
| Embedding (row/col→qubit grouping) | artifactual entanglement | multiple nulls through the *same* embedding |
| Reduction (which 2-qubit pair) | hidden max-over-pairs, multiple comparisons | **pre-registered** pair (qubits {0,1}, top-2 row bits); a pair-sweep, if run, is reported as the full null-calibrated distribution, never the max |
| Null model | one null hides confounds it doesn't control | **multiple independent nulls** (below) |
| Semantics axis (classical/quantum) | one witness is under-determined | **independent witness battery** (below) + implication lattice |

## Witness battery (independent probes — full set required)

All operate on the 2-qubit reduced state ρ₂ (and the full pure state where noted); all are exact, deterministic functions of a density matrix (no sampling noise). All reuse `larql_hilbert::eig::hermitian_eigenvalues`.

| # | Witness | Axis | Conclusive direction | Range / bound |
|---|---|---|---|---|
| W1 | **Mutual information** `I(A:B)=S(ρ_A)+S(ρ_B)−S(ρ₂)` | independent ↔ dependent | `I=0` ⟺ **product/independent** (conclusive) | 0 … 2 bits |
| W2 | **Negativity** (PPT / Peres–Horodecki) | separable ↔ entangled | `N=0` ⟺ **provably separable** (nec.&suff. for 2-qubit; conclusive) | 0 … 0.5 |
| W3 | **CHSH via Horodecki** `2√M` | local ↔ nonlocal | non-violation inconclusive; violation confound-prone | 2 … 2√2 |
| W4 | **Entanglement entropy** `S(ρ_A)` of the full pure state across the cut | low-Schmidt ↔ high-Schmidt | bipartite entanglement of the whole state | 0 … log₂(min dim) |
| W5 | **Compressibility gap** `H−S` (Shannon of flattened \|C\|² minus W4) | compressible ↔ incompressible | the prior metric, now null-controlled | ≥ 0 |
| W6 | **Hilbertian residual** `‖[C,J]‖/‖C‖` (split-half J) | real-linear ↔ complex-linear | *independent of correlation* — does the coupling admit a unitary/coherent (quantum) reading | 0 … 2 |

These are **not equivalent**: correlation (W1) ⊋ entanglement (W2,W4) ⊋ nonlocality (W3) is a strict hierarchy for mixed states, and coherence/complex-structure (W6) is an orthogonal axis to correlation entirely. Measuring all six **over-constrains** the verdict and *locates* each source within the hierarchy (e.g. "real heads: correlated but separable, local, and not complex-linear" is a specific, well-determined classical placement).

## Implication lattice (apparatus self-checks — refutation surfaces for the experiment itself)

The witnesses satisfy logical implications that **must** hold; checking them on every source (and exactly on the analytic poles) is a multi-dimensional consistency constraint. A violation indicts the *apparatus*, not the hypothesis:
- nonlocal ⟹ entangled ⟹ correlated: `M>1 ⟹ N>0 ⟹ I>0`.
- product ⟹ separable ⟹ local: `I=0 ⟹ N=0 ⟹ M≤1`.
- Werner-type gap: states with `N>0, M≤1` (entangled, not nonlocal) must be representable — confirms W2 and W3 are genuinely independent, not redundant.

These checks are first-class tests, not afterthoughts: they are how we rule out experimental error in the apparatus.

## Multiple independent nulls (each removes a different confound)

Real data must exceed **every applicable null** on a witness to claim structure on that axis (over-constraint across nulls):
- **N0 Gaussian** — shape/scale-matched `W_Q,W_K → C`; controls matrix dimensions/scale (MP spectrum).
- **N1 singular-value-matched** — real `C`'s singular spectrum with Haar-random singular vectors; controls *for the spectrum*, isolating singular-vector/structural content.
- **N2 sign/phase-randomized** — real `C` with randomized entry signs (magnitudes preserved); controls sign structure.

## Analytic poles (positive-control validity gates — exact)

- **Classical pole** — product & dephased QLM states → `I=0, N=0, M≤2, W6` real-linear (residual 0 for the appropriate real coupling).
- **Quantum pole** — Bell/GHZ/Dicke QLM states → Bell: `I=2, N=0.5, M=2 (CHSH 2√2)`; complex-structured couplings → high W6.
Each witness must reproduce its pole values exactly, or it is void.

## Well-posedness criterion (when a verdict may be stated)

A classical/quantum verdict for a source is admissible **only** when: (a) all six witnesses pass their analytic-pole gates; (b) the implication lattice holds on that source; (c) the source is compared against **all** applicable nulls; and (d) the witnesses **agree** (over-determined) — or, where they disagree, the disagreement is itself located in the hierarchy (e.g. entangled-but-local) and reported as the verdict. A result from a partial apparatus (missing witnesses or nulls) is **inconclusive by construction** and may not be reported as a finding.

## Reporting

Per source, the full witness×null table, the pole placements, and the implication-lattice check. Headline outputs are **conclusive negatives** ("real is statistically indistinguishable from null Nk on witness Wj, with effect-size/power floor"; "head h's reduction is provably separable, N=0"). A real-≠-null result leaning quantum is reported as "not ruled out — survives nulls N0/N1/N2? needs replication," never as a positive finding. This re-adjudicates "not quantum-compressible" by construction.

## Components

- **larql-hilbert (pure, additive):** `partial_trace_2q`; `mutual_information`; `negativity` (partial transpose + `hermitian_eigenvalues`); `correlation_matrix` + `chsh_max` (Horodecki); reuse `entanglement_entropy`, the gap, and `commutator_residual`/`split_half_j` (W6). Density-matrix utilities as needed.
- **Implication-lattice + pole tests:** exact unit tests asserting pole values *and* the lattice implications (incl. a Werner state for the N>0,M≤1 cell).
- **Null generators (larql-cli or a test-support module):** N0/N1/N2 through the identical pipe.
- **Runner (larql-cli):** per head, the full witness×null battery on real `C`; aggregate over the head population; emit the report. Pre-registered reduction. Optionally `larql quantum-signature <vindex>`.
- **Harness:** demo (poles + lattice + a small real-vs-null illustration, CI-safe) + integration tests (pole gates, lattice, real-vs-null on the SmolLM2 vindex when present).

## Implementation staging (engineering only — does not narrow the apparatus)

The build may proceed witness-by-witness and null-by-null, but **no experimental verdict is claimed until the full apparatus (all six witnesses, all three nulls, the lattice, the poles) is in place** — because a partial apparatus is under-determined. First checkpoint: run the verification that a Gaussian-random `C` violates via the embedding; if it does (expected), the random nulls are mandatory and the diagonal baseline is formally disqualified.

## Out of scope

- Physical-nonlocality *claims*: this is a structural/spectral comparison of weight matrices read as states, calibrated against poles and nulls — never a certification that a transformer is a quantum device.
- SP3 (QLM⇄vindex compiler/distiller).

## Testing strategy

TDD. Witnesses are exact → unit tests assert analytic values (Bell N=0.5, CHSH 2√2, I=2; product → 0,0,≤2) and the implication lattice (including a Werner-state cell). Real-vindex tests are `LARQL_TEST_VINDEX`-gated; poles, lattice, and nulls run always in CI.
