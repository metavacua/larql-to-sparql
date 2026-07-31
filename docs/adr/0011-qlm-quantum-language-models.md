# ADR 0011 — QLM: Quantum Language Models as a Pure-Rust Leaf Crate

**Status:** Accepted
**Date:** 2026-06-11
**Depends on:** `larql-hilbert` (new), `larql-vindex` (canonical pipeline)

---

## Context

The QLM line (see `docs/QLM-ROADMAP.md`) formalizes language models as
Hilbert-space objects: qubit states, unitary evolution, Born-rule readout, and
the classical-vs-quantum compressibility programme. It is built as a new crate,
`larql-hilbert`, across PRs #137–#140 (+ the entanglement-entropy work). Several
non-obvious architectural choices were made that are not legible from the code
alone; this ADR records them and their rationale.

## Decision

1. **`larql-hilbert` is a pure-Rust leaf crate.** Its only dependencies are
   `ndarray` (real matrices) and `num-complex` (the qubit `Complex64` algebra).
   It depends on **no other `larql-*` crate** and links **no BLAS/LAPACK**.

2. **Measurement is an elimination rule, not a new primitive.** Projective
   measurement is `project(&Qubit, outcome) -> Option<Qubit>`, with `None` the
   uninhabited outcome ⊥ (coinciding with `SingleQubitLM::score` returning −∞).
   The no-cloning discipline is enforced by the type system: `LinearQubit` is
   `!Copy`/`!Clone`, so `measure(self, …)` consumes it (Rust move semantics =
   linear-logic's prohibition on contraction).

3. **Admissibility is a first-class layer.** Bounded extraction over the LM is
   classified by arithmetical fragment — `is_realizable` (Δ₀), `exists_continuation`
   (Σ⁰₁), `uniformly_stable` (Π⁰₂) — implementing Rosko's finite-extraction
   criterion (admissible measurement ⊆ Σ⁰₁ ∪ Π⁰₂) as the quantifier shape of
   bounded procedures.

4. **Numerics are hand-written and BLAS-free.** Single/two-qubit gates are
   fixed-size (`[[Complex64;2];2]`, `[[Complex64;4];4]`) with hand-written
   `mat_mul`/`dagger`/`apply`; the entanglement-entropy meter uses a pure-Rust
   cyclic-Jacobi symmetric eigensolver, not `ndarray-linalg`/LAPACK.

5. **Born readout presupposes the dagger installed by canonicalization.** The
   Born rule `|⟨t|ψ⟩|²` is only well-defined given an inner product; the
   `larql canonicalize` whitening *is* the act of choosing that dagger/metric.
   The QLM is therefore conceptually downstream of the canonical pipeline.

## Rationale

- **Portability toward `wasm32v1-none` (#132) and BLAS-free numerics (#135).**
  A leaf crate with no native linkage and alloc-light, fixed-size kernels is the
  certification target for the minimal-LM epic. Linking BLAS would make the
  crate impossible on the bare wasm target; depending on `larql-compute` would
  drag the whole inference stack.
- **Constructive/categorical correctness.** Measurement-as-elimination and
  no-cloning-as-move follow the Curry-Howard-Lambek / categorical-quantum-
  mechanics reading; encoding linearity in the type system makes the no-cloning
  theorem mechanical rather than a runtime convention.
- **Isolation for an exploratory subproject.** The QLM is a research line with
  its own roadmap and (separate) authorship/licence; keeping it a leaf avoids
  coupling the main vindex/inference path to in-flux quantum-LM code.

## Consequences

- **Positive:** the crate builds anywhere `ndarray` + `num-complex` build; it is
  the natural no_std/wasm certification target; the type system enforces the
  linear discipline; the qubit core is small and auditable.
- **Negative / accepted cost:** some numerics are reimplemented rather than
  reused (a symmetric eigensolver, fixed-size complex matmul) — justified by the
  no-BLAS constraint; a future shared math crate could consolidate (see #135).
- **Reuse is deliberately forgone:** an entropy helper already exists in
  `larql-cli` (`ov_rd/metrics.rs`), but reusing it would invert the dependency
  hierarchy onto a leaf crate; the re-implementation is intentional (`/simplify`
  review confirmed this is the right call).

## Cross-references

PRs #133 (canonical), #137 (hilbertian residual), #138 (single-qubit LM), #139
(constructive measurement + admissibility), #140 (two-qubit/Bell). Issues #132
(wasm32v1-none minimal LM), #135 (numerical-backend portability). Roadmap:
`docs/QLM-ROADMAP.md`.
