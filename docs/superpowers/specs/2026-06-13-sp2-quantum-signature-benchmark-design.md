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

## Formal foundations: the trilemma ≅ fire-triangle ≅ sheaf global-section

The classical/quantum no-go is one ternary incompatibility expressed three ways:
the **quantum trilemma** {locality, tractability, objectivity}; the type-theoretic
**fire triangle** {effects, substitution, dependent elimination} (Pédrot–Tabareau,
POPL 2020 — you can't consistently combine all three); and the **sheaf-theoretic
structure** of Abramsky–Brandenburger ([arXiv:1102.0264](https://arxiv.org/abs/1102.0264)),
where **contextuality = obstruction to a global section** and **Bell-nonlocality
is the special case**, with a strength hierarchy (Bell < Hardy < GHZ).

| Quantum trilemma | Fire triangle | Sheaf / semantics | LM / WASM |
|---|---|---|---|
| Locality | drop **Effects** (pure ⟹ closed) | empirical model on a closed cover | closed call-graph (no I/O) |
| Objectivity / non-contextual | **Dependent elimination** | **global section exists (extensional)** ↔ gluing obstruction (intensional) | extensional vindex-as-function ↔ intensional process |
| Tractability | **Substitution** (β = substitutivity) | **compositional** gluing local→global (DisCoCat) | context-free ↔ context-sensitive |

**Consequences for the witnesses (a consolidation, not an expansion):**
- The **objectivity axis is the sheaf global-section test**: a source's empirical
  model (measurement cover + outcome distributions) either admits a **global
  section** — a context-independent deterministic value-assignment marginalizing
  to the observed statistics (= non-contextual = objective = *extensional* =
  *compositional*) — or has a **gluing obstruction** (= contextual = *intensional*).
  This is computable by the Abramsky–Brandenburger linear-algebra method (and a
  Čech-cohomological obstruction for the logical/possibilistic case).
- **W3 (CHSH/Horodecki) and W8 (Peres–Mermin/KS) are facets of this one witness**
  — different measurement *covers* (Bell cover vs KS cover) in the strength
  hierarchy. They remain as named, exactly-calibrated special cases; the
  sheaf-theoretic global-section/obstruction is their common generalization.
- **Intensional/extensional** is the measurement↔computation correspondence:
  extensional = the global section (denotation/I-O function, the vindex-as-table);
  intensional = the irreducibly context-dependent process (the dependent-elimination
  vertex). Contextuality = no extensional reduction.
- **Effects ↔ non-locality**: monadic interaction with an environment = calls
  outside the closed graph = the WASM/operational non-locality. Pure = local.

Sources: [Sheaf structure of non-locality & contextuality](https://arxiv.org/abs/1102.0264);
[Peres–Mermin noncontextuality inequalities](https://arxiv.org/abs/1704.01153);
[Fire Triangle](https://dl.acm.org/doi/10.1145/3371126);
[extensional/intensional categorical models](https://arxiv.org/abs/2408.07058);
[semantic unification (sheaf NLP)](https://link.springer.com/chapter/10.1007/978-3-642-54789-8_1).

## The confound this apparatus defeats

`from_matrix(C)` normalizes any real matrix and reads it as a bipartite pure state; a *generic* matrix so read is highly entangled. So a real coupling will *generically* "look entangled / violate CHSH" — not from training but from the embedding. And `entanglement_entropy_bipartition(from_matrix(C))` equals the spectral entropy of `C`'s singular values, which is near-maximal for a Gaussian (Marchenko–Pastur) matrix → the prior "not quantum-compressible" finding **had no null and does not distinguish SmolLM2 from a random matrix.** The apparatus below removes this by passing independently-constructed nulls through the identical pipe and cross-checking multiple witnesses.

## Randomness regime (formalization): the determinative refutation

Genuine quantum randomness is certifiable **only device-independently**, via a loophole-free Bell violation, which certifies the outputs are *not pre-determined* ([Nature DIQRNG, 2018](https://www.nature.com/articles/s41586-018-0559-3)). Algorithmically, genuine randomness is Kolmogorov-**incompressible**; pseudo-randomness is **compressible** — a deterministic function of a short seed (the seed is the hidden variable) ([Randomness: quantum vs classical](https://arxiv.org/pdf/1512.08852)). And under **efficiently-computable measures, quantum randomness and pseudo-randomness are indistinguishable** ([arXiv:2309.11117](https://arxiv.org/pdf/2309.11117)); classical simulation reproduces Born statistics exactly.

**Determinative consequences (these bound every other witness):**
1. The LLM **and our simulated `NQubitLM`** are *both* pseudo-random — reproducible under fixed seed, Kolmogorov-compressible, classically computed. On the randomness axis, **genuine quantum randomness is determinatively refuted for the entire apparatus, including the QLM "quantum" pole.** (Refuting quantum randomness is the determinative test; reproducibility refutes it.)
2. Therefore the CHSH/Horodecki witness is **not** a Bell/nonlocality/randomness certification — there is no spacelike separation, no free measurement choice, no loophole-free experiment, only one classical computation of a density matrix. Per Principle 0 it is labeled exactly what it is: a **structural** test (does the correlation matrix exceed the classical bound). The same caveat applies to all witnesses: they test *structure of pseudo-random outputs*, never randomness-as-such.
3. The apparatus can confirm **structural** quantum signatures (Bell-violating correlation *structure*, entanglement *structure*) and can **conclusively refute genuine quantum randomness** (a real negative) — it cannot positively certify quantum randomness (impossible by device-independence + the indistinguishability bound).

This is recorded as a first-class **determinative axis W7** (below), whose conclusive negative — "reproducible ⟹ pseudo-random ⟹ not quantum-random" — holds for every computable source (LLM, QLM-sim, all nulls) and frames the interpretation of W1–W6.

## The degrees of freedom that must be constrained

For the verdict to be well-posed, each DOF is pinned and cross-checked:
| DOF | Unconstrained failure | Constraint |
|---|---|---|
| Embedding (row/col→qubit grouping) | artifactual entanglement | multiple nulls through the *same* embedding |
| Reduction (which 2-qubit pair) | hidden max-over-pairs, multiple comparisons | **pre-registered** pair `{0, n/2}` — **one row qubit + one column qubit** (top row & column bits of the Choi state; for 2×2 ⇒ the full state). Must straddle the row\|column cut (two row qubits trace out the column → separable). No max-over-pairs; a sweep, if run, reports the full null-calibrated distribution, never the max |
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
| W7 | **Reproducibility / compressibility** of the generative output stream (fixed-seed determinism + a lossless-compression proxy for Kolmogorov complexity) | pseudo-random ↔ quantum-random | **reproducible/compressible ⟹ pseudo-random ⟹ not quantum-random** (conclusive, determinative) — see Randomness regime | n/a (boolean + ratio) |

W1–W6 operate on the state/coupling (structure); **W7 operates on the generative process** (the output stream), and is **determinative**: per the Randomness regime, *every* computable source (LLM, QLM-sim, all nulls) is reproducible/compressible and so conclusively pseudo-random. W7 does not discriminate LLM from QLM-sim (both fail it) — its role is the meta-level conclusive negative that **bounds the interpretation of W1–W6 to structural signatures only**, never randomness-as-such.

W1–W6 are **not equivalent**: correlation (W1) ⊋ entanglement (W2,W4) ⊋ nonlocality-*structure* (W3) is a strict hierarchy for mixed states, and coherence/complex-structure (W6) is an orthogonal axis to correlation entirely. Measuring all of them **over-constrains** the verdict and *locates* each source within the hierarchy (e.g. "real heads: correlated but separable, local-structure, not complex-linear, and pseudo-random" is a specific, well-determined classical placement).

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

A classical/quantum verdict for a source is admissible **only** when: (a) all structural witnesses (W1–W6) pass their analytic-pole gates and the determinative randomness axis (W7) is recorded; (b) the implication lattice holds on that source; (c) the source is compared against **all** applicable nulls; and (d) the witnesses **agree** (over-determined) — or, where they disagree, the disagreement is itself located in the hierarchy (e.g. entangled-but-local) and reported as the verdict. A result from a partial apparatus (missing witnesses or nulls) is **inconclusive by construction** and may not be reported as a finding.

## Reporting

Per source, the full witness×null table, the pole placements, and the implication-lattice check. Headline outputs are **conclusive negatives** ("real is statistically indistinguishable from null Nk on witness Wj, with effect-size/power floor"; "head h's reduction is provably separable, N=0"). A real-≠-null result leaning quantum is reported as "not ruled out — survives nulls N0/N1/N2? needs replication," never as a positive finding. This re-adjudicates "not quantum-compressible" by construction.

## Components

- **larql-hilbert (pure, additive):** `partial_trace_2q`; `mutual_information`; `negativity` (partial transpose + `hermitian_eigenvalues`); `correlation_matrix` + `chsh_max` (Horodecki); reuse `entanglement_entropy`, the gap, and `commutator_residual`/`split_half_j` (W6). Density-matrix utilities as needed.
- **Implication-lattice + pole tests:** exact unit tests asserting pole values *and* the lattice implications (incl. a Werner state for the N>0,M≤1 cell).
- **Null generators (larql-cli or a test-support module):** N0/N1/N2 through the identical pipe.
- **Runner (larql-cli):** per head, the full witness×null battery on real `C`; aggregate over the head population; emit the report. Pre-registered reduction. Optionally `larql quantum-signature <vindex>`.
- **Harness:** demo (poles + lattice + a small real-vs-null illustration, CI-safe) + integration tests (pole gates, lattice, real-vs-null on the SmolLM2 vindex when present).

## Implementation staging (engineering only — does not narrow the apparatus)

The build may proceed witness-by-witness and null-by-null, but **no experimental verdict is claimed until the full apparatus (all seven witnesses W1–W7, all three nulls, the lattice, the poles) is in place** — because a partial apparatus is under-determined. First checkpoint: run the verification that a Gaussian-random `C` violates via the embedding; if it does (expected), the random nulls are mandatory and the diagonal baseline is formally disqualified.

## Out of scope

- Physical-nonlocality *claims*: this is a structural/spectral comparison of weight matrices read as states, calibrated against poles and nulls — never a certification that a transformer is a quantum device.
- SP3 (QLM⇄vindex compiler/distiller).

## Testing strategy

TDD. Witnesses are exact → unit tests assert analytic values (Bell N=0.5, CHSH 2√2, I=2; product → 0,0,≤2) and the implication lattice (including a Werner-state cell). Real-vindex tests are `LARQL_TEST_VINDEX`-gated; poles, lattice, and nulls run always in CI.
