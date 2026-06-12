# Proposal: Quantum Language Model Foundation

## Status: in-flight

## Why

The QLM line (see `docs/QLM-ROADMAP.md`) formalizes language models as
Hilbert-space objects — qubit states, unitary evolution, Born-rule readout — and
grounds the classical-vs-quantum compressibility programme. The foundation is
the `larql-hilbert` crate (PRs #137–#140 + the entanglement-entropy work). This
change captures its requirements as testable contracts, mirroring the
`canonical-extraction` change.

## What

Specify the `quantum-language-model` capability provided by `larql-hilbert`:

1. The complex-structure core and the unifying theorem
   `commutator_residual = 2·antilinear_fraction`, plus the real↔complex bridge.
2. The single-qubit Bloch-sphere LM: state, SU(2) gates, Born readout,
   autoregressive scoring/generation.
3. Constructive measurement (elimination rule, ⊥) with no-cloning enforced by
   the type system, and the bounded-extraction admissibility layer (Δ₀/Σ⁰₁/Π⁰₂).
4. The two-qubit LM: tensor product, entanglement (non-factorization), CNOT and
   the Bell operation, partial measurement and its correlation.
5. The entanglement-entropy meter (quantum compressibility, in ebits).

## Non-goals (this change)

- GHZ / W (3+ qubit) states — Phase 5.
- Superdense coding protocol and quantum LQL verbs (`MEASURE`/`STEP`) — Phase 6.
- Vindex-wide compressibility analysis CLI — Phase 7.
- `wasm32v1-none` / `no_std` certification of the kernels — Phase 8 (epic #132).

## Related

- ADR: `docs/adr/0011-qlm-quantum-language-models.md`.
- Roadmap: `docs/QLM-ROADMAP.md`.
- Depends-on: the canonical pipeline (`canonical-extraction` change) — Born
  readout presupposes the dagger that whitening installs.

## Risk

Low. `larql-hilbert` is an additive pure-Rust leaf crate; nothing in the
inference/vindex path depends on it.
