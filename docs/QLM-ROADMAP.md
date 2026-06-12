<!--
SPDX-License-Identifier: CC-BY-SA-4.0
Copyright © 2026 Ian Douglas Lawrence Norman McLean.
Licensed under the Creative Commons Attribution-ShareAlike 4.0 International
License (CC BY-SA 4.0): https://creativecommons.org/licenses/by-sa/4.0/

This is the QLM (Quantum Language Model) roadmap. It is a SEPARATE work from the
LARQL project roadmap (`ROADMAP.md`), which has its own authorship and license
and is left unmodified.
-->

# QLM Roadmap — Quantum Language Models for LARQL

> **Copyright © 2026 Ian Douglas Lawrence Norman McLean · CC BY-SA 4.0.**
> Distinct from the upstream `ROADMAP.md` (unmodified). Per-phase code lives in
> the `larql-hilbert` crate and is cross-referenced to its PRs/issues below.

This roadmap tracks the **quantum language model** line: the Hilbert-space
formalization of language models, the minimal canonical quantum LM, and the
classical-vs-quantum compressibility programme — built as the pure-Rust leaf
crate `larql-hilbert` (deps: `ndarray` + `num-complex`; no BLAS), kept portable
toward the `wasm32v1-none` minimal-LM target.

---

## Purpose (load-bearing — read first)

A **vindex** is a queryable weight database (the object language / the data); an
**LQL** operation is a process over it (the metalanguage / the query). The QLM
line asks what these become at the Hilbert-space limit:

- **State** = a (projective) Hilbert-space point — a qubit on the Bloch sphere
  (`ℂP¹`) at the minimal scale.
- **Readout** = the Born rule — a *measurement*, which (per the constructive
  reading) is the **elimination rule** of the linear, no-cloning fragment, not a
  new primitive.
- **Evolution** = unitary (SU(2)/SU(2ⁿ)) gates — norm-preserving, the
  conservative idealization of the residual-stream update.
- **Canonical form** = the gauge-fixed representative (the dagger/metric, then
  the projective quotient) — the same canonicalization the `larql canonicalize`
  pipeline performs, at minimal scale.

The minimal quantum LM is the **single qubit**; the place classical structure
fails is the **Bell point** (two qubits stop factoring, `A⊗B ≠ A×B`); the
asymptotic regime is **GHZ / W** (3+ qubits). The unifying claim is that the
**classical/quantum compressibility gap of a vindex is denominated in ebits**,
with **superdense coding** as the operational unit and the **entanglement
entropy** as the measured quantity.

## Theoretical frame (the four pillars)

1. **Categorical quantum mechanics / the dagger.** Canonicalization is choosing
   the dagger (the inner product), lifting `FdVect → FdHilb`; it reduces the
   gauge group `GL(n) → U(n)/O(n)` (why cross-model alignment is Procrustes).
   The token vocabulary is the classical structure; the hidden space is the
   basis-free quantum object. (Baez & Stay, *Rosetta Stone*; Abramsky & Coecke.)
2. **Zizzi quantum metalanguage.** Assertions carry **complex assertion degrees**
   (amplitudes); Sambin's reflection turns metalinguistic bonds into the
   object-language **superposition connective** (logic `Lq`). Maps onto
   LQL (metalanguage) / vindex (object language); the single-qubit LM is an
   operational model of `Lq`; entanglement is the non-factorizing bond.
   (Zizzi, arXiv:1003.5976.)
3. **Rosko admissibility.** Admissible measurement = finite extraction ⊆
   **Σ⁰₁ ∪ Π⁰₂** (Δ₀-decidable core), via realizability over Heyting Arithmetic.
   This is the *finiteness filter on Zizzi's metalanguage* — the
   `admissibility` module (Δ₀/Σ⁰₁/Π⁰₂) is its implementation; the ⊥ outcome
   (`project → None`, `score → −∞`) is the degree-0 / uninhabited assertion.
   (Rosko, arXiv:2511.21296.)
4. **Superdense coding & compressibility.** One ebit ↔ the 2-bit dense-coding
   gain; the classical/quantum vindex compressibility ratio is set by the
   entanglement entropy in ebits. Weight-matrix spectra are heavy-tailed
   (Martin–Mahoney HTSR), which is why *lossy ≫ lossless* and the on-shell Zipf
   head carries the function. (Bennett–Wiesner; Schumacher.)

---

## Phases (critical path)

Status legend: ✅ done (PR open in `metavacua/larql-to-sparql`), 🔄 in progress,
⬜ planned, 🔬 research.

### Phase 0 — Hilbert-space foundation ✅ (PR #137)
Complex structure `J` (`J²=−I`), commutator residual, the unifying theorem
`commutator_residual = 2·antilinear_fraction`, the real↔complex (`realify` /
`complex_parts`) bridge, and the **Hilbertian residual** — a per-attention-head,
weights-only score of complex-linearity. *Result:* on SmolLM2, residuals
0.45–1.46 (discriminating); the metric doubles as a **quantum-compressibility
meter** (residual 0 ⇒ 2× complex-compressible). Built on the canonical pipeline
(PR #133).

### Phase 1 — Single-qubit Bloch LM ✅ (PR #138)
`Qubit` + Bloch coordinates (`ℂP¹`), SU(2) gates, Born readout, `SingleQubitLM`
(2-token vocab, collapse-then-unitary, autoregressive `score`/`generate`).
With projective collapse it reduces to a first-order Markov chain — the boundary
where quantum = classical, hence the right *minimal* object.

### Phase 2 — Constructive measurement + admissibility ✅ (PR #139)
`measurement::project` (elimination rule; `None` = ⊥), `LinearQubit` (no-cloning
via Rust move semantics), and `admissibility` (Δ₀ `is_realizable`, Σ⁰₁
`exists_continuation`, Π⁰₂ `uniformly_stable`). Implements the Rosko filter on
the Zizzi metalanguage; the integration test pins `⊥ ≅ −∞`.

### Phase 3 — Two-qubit LM + Bell ✅ (PR #140)
`TwoQubit` (ℂ⁴), tensor product, `is_product` (entanglement = nonzero 2×2
determinant), partial measurement, 4×4 gate algebra, `cnot`, `bell` (Φ⁺), and
`TwoQubitLM`. *Result:* the Bell-init LM forbids the anti-correlated tokens
01/10 (`−∞`) — the place the single-qubit Markov reduction provably breaks.

### Phase 4 — Entanglement-entropy / compressibility meter 🔄 (branch `feat/entanglement-entropy`)
`spectral_entropy(&[f64])` (quantum compressibility in ebits) ✅. Next:
`entanglement_entropy(&Array2<f64>)` via a pure-Rust cyclic-Jacobi symmetric
eigensolver (plan: `docs/superpowers/plans/2026-06-11-matrix-entanglement-entropy.md`).
Turns `is_product` (yes/no) into "how many ebits."

### Phase 5 — GHZ / W (3+ qubits) ⬜
`ℂ^{2ⁿ}` states, n-qubit gate algebra, the GHZ and W entangling operations,
multi-cut entanglement. The asymptotic regime where the classical/quantum size
ratio (`Vⁿ` vs bond dimension `χ ≈ 2^S`) is a real, growing gap.

### Phase 6 — Superdense coding + quantum LQL ⬜
Superdense coding as a protocol on the Bell machinery (Pauli-gated Bell states +
Bell-basis readout) — the first *protocol* (not just state) and the literal
quantum factor-of-2. Then `MEASURE` (Born) and `STEP` (collapse+unitary) as
quantum LQL operations under a linear/affine resource discipline (no-cloning at
the type level). Generation ↔ communication as dual readouts.

### Phase 7 — Vindex compressibility analysis 🔬 (depends on Phase 4)
A CLI/analysis running `entanglement_entropy` over weight matrices: **on-shell
vs full**, **canonical vs raw**, **von Neumann (complex) vs Shannon (real)**
coherence defect, and a **Zipfian/HTSR** power-law check. Tests the conjecture:
*the quantum compressibility advantage is concentrated in the canonicalized
on-shell subspace and invisible in the raw syntactic whole.* (Prior from the
Hilbertian data: small in the raw basis — the open question is whether
canonicalization recovers it.)

### Phase 8 — `wasm32v1-none` minimal quantum metalanguage 🔬 (epic #132)
The constructively-admissible (Zizzi ∩ Rosko) fragment — finitely-extractable,
computable-amplitude, sub-Turing (`¬L ∧ ¬M`) — is the minimal LM. The
larql-hilbert primitives are the certification target: alloc-free / no_std /
bounded-stack versions of the kernels (the Hilbertian 64×64 kernel first).
Numerical-backend portability tracked in #135.

---

## Cross-references

- **PR stack** (all in `metavacua/larql-to-sparql`, merge in order): #133
  canonical → #137 hilbertian → #138 qlm-hilbert → #139 constructive-measurement
  → #140 two-qubit-bell → (Phase 4) entanglement-entropy.
- **Issues:** #132 (wasm32v1-none minimal LM), #134 (regime-classifier finding),
  #135 (numerical-backend portability / BLAS-free), #136 (hilbertian
  refinements).
- **Plans:** `docs/superpowers/plans/` — `2026-06-10-canonical-extraction-pipeline`,
  `2026-06-11-attention-hilbertian-residual`, `2026-06-11-qlm-hilbert-foundation`,
  `2026-06-11-constructive-measurement-layer`, `2026-06-11-two-qubit-bell`,
  `2026-06-11-matrix-entanglement-entropy`.

## Bibliography

- P. Zizzi, *From Quantum Metalanguage to the Logic of Qubits*, arXiv:1003.5976 (2010).
- M. Rosko, *A Constructive Fragment of Physical Propositions*, arXiv:2511.21296 (2025).
- J. Baez & M. Stay, *Physics, Topology, Logic and Computation: A Rosetta Stone* (2009).
- S. Abramsky & B. Coecke, *Categorical Quantum Mechanics*.
- *Towards a Computational Quantum Logic* (the `Lᶜ` calculus), arXiv:2504.07609 (2025).
- C. Bennett & S. Wiesner, *Communication via one- and two-particle operators on EPR states* (1992) — superdense coding.
- B. Schumacher, *Quantum coding* (1995) — quantum data compression.
- C. Martin & M. Mahoney, *Heavy-Tailed Self-Regularization in Deep Neural Networks* — weight-spectrum power laws.
