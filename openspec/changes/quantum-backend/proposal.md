# Proposal: Quantum Backend (usable QLM via larql)

## Status: in-flight

## Why

The QLM infrastructure (`larql-hilbert`: `NQubitLM`, `NRegister`, entanglement)
is mathematically complete but not *usable* — it has never been driven as a
language model through larql's real interfaces. This change makes a quantum
model a real on-disk **quantum vindex** that the existing `larql-lql` `Session`
can `USE` and `INFER`, producing ranked next-token predictions. It is SP1 of the
"usable quantum language models" programme (SP2 = quantum-vs-(semi)classical
benchmark; SP3 = QLM⇄vindex compiler/distiller).

Design: `docs/superpowers/specs/2026-06-12-quantum-backend-design.md`.

## What Changes

1. `NQubit::dicke(n, k)` — the missing Dicke (angular-momentum `(n,k)`) state
   constructor (`w` = `dicke(n,1)`).
2. The **Dicke quantum vindex** artifact: a vindex dir with `index.json`
   (`family: "quantum"`) + `qlm.json` carrying *only the quantum numbers*
   `(n, state-class, k)`. The state is reconstructed, never stored — "completely
   specified by its quantum numbers."
3. `Backend::Quantum` in `larql-lql`: `USE` branches on `family == "quantum"`;
   `INFER` returns the Born next-token distribution rendered as ranked tokens;
   `STATS`/`SHOW`/`DESCRIBE` report the quantum numbers.
4. The **classicalization seam**: every non-native statement routes through one
   extension point (`classical_view()` → a dephased `ClassicalRegister`) and
   returns a single uniform error, so a later sub-project can serve the classical
   vindex operations by measuring the QLM without touching dispatch.

## Non-goals (this change)

- `L > 0` autoregressive generation dynamics (per-step re-mixing unitaries).
- Family B (hypergraph + feature-basis readout) and distillation — SP3.
- The benchmark harness comparing quantum vs (semi)classical — SP2.
- Implementing the classical ops via classicalization — a later sub-project
  (this change only provides the seam).

## Related

- Design spec: `docs/superpowers/specs/2026-06-12-quantum-backend-design.md`.
- Builds on: the `quantum-language-model` capability (`qlm-foundation` change) —
  `NQubitLM`, `NRegister`, `entanglement_entropy_bipartition`.
- Theory: graph/hypergraph states; DisCoCat / categorical quantum NLP (the
  functor G: graph-DB → FdHilb); LLM-as-hypergraph-database.

## Risk

Low–moderate. Additive: a new `Backend` variant and a new vindex `family`;
nothing in the classical inference path changes. The one cross-crate edge is
`larql-lql → larql-hilbert` (new). Ground truth is the analytic Dicke Born
distribution, so all assertions are exact (no model download).
