# Proposal: Quantum-Signature Falsification Apparatus (SP2)

## Status: in-flight

## Why

The earlier "raw SmolLM2 attention is not quantum-compressible" finding was
**under-determined**: it had no null. `from_matrix(C)` reads any real matrix as a
bipartite pure state, and a *generic* matrix so read is near-maximally entangled
(Marchenko–Pastur), so a trained coupling will "look entangled / violate CHSH"
from the embedding alone. This change re-adjudicates the question with a
**predicative, over-constrained falsification apparatus**: every source flows
through the *identical* `coupling → embed → reduce → witness` pipe, and the
reported quantity is the signal **real − null**, never a bare baseline. It is SP2
of the "usable quantum language models" programme (SP1 = quantum backend, done;
SP3 = QLM⇄vindex compiler/distiller).

Design: `docs/superpowers/specs/2026-06-13-sp2-quantum-signature-benchmark-design.md`.

## What

1. A **witness battery** on the 2-qubit reduced density matrix and the coupling:
   W1 mutual information, W2 negativity (PPT), W3 CHSH (Horodecki), W4
   entanglement entropy, W5 compressibility gap, W6 Hilbertian residual, plus a
   boolean **implication lattice** as an apparatus self-check — each calibrated to
   **exact analytic poles** (Bell N=0.5 / CHSH 2√2 / MI 2; product 0/0/≤2; a
   Werner entangled-but-local cell).
2. The **sheaf contextual fraction** (Abramsky–Barbosa–Mansfield) as the common
   generalization of W3 and W8 (Peres–Mermin KS): CF=0 ⟺ a global section exists
   ⟺ non-contextual (a conclusive negative); CF=√2−1 on the Bell cover.
3. **Three independent nulls** through the identical pipe — Gaussian,
   singular-value-matched (spectrum-controlling), sign-randomized.
4. The **canonical (whitened) metric dual** — the dagger as inner-product choice:
   raw Euclidean vs Cholesky-whitened (`canonical_meta`), `C_canon = (W_Q M)(W_K M)ᵀ`.
5. The **determinative W7** (reproducibility ⟹ pseudo-random ⟹ not quantum-random),
   which is the operational **reflexivity test** (classical self-identity under
   re-seeding) and bounds every other witness to *structure* only.
6. The **`larql quantum-signature`** runner: per head it evaluates the battery on
   the real coupling and the three nulls under both metrics, computes the
   contextual fraction, and emits the predicative `real − null` report to
   `quantum_signature_meta.json`.

## Non-goals (this change)

- A classical/quantum **verdict** on any specific model: admissible only from the
  *full* apparatus on a real model run (gated behind `LARQL_TEST_VINDEX`); a
  partial apparatus is inconclusive by construction.
- Positive certification of quantum **randomness** — impossible by
  device-independence + the indistinguishability bound (W7 is a conclusive
  negative only).
- A **fully quantum (non-Tarskian) metalanguage** for the truth/identity layer
  (complex assertion-degrees, Zizzi's QML connectives): the apparatus is
  deliberately *semi-classical* in its identity layer; the quantum metalanguage
  is recorded as future scope (cf. SP3).

## Related

- Design spec: `docs/superpowers/specs/2026-06-13-sp2-quantum-signature-benchmark-design.md`.
- Builds on: `quantum-backend` (SP1) — `Backend::Quantum`, `NQubitLM`.
- Theory: Abramsky–Brandenburger sheaf contextuality; Choi–Jamiołkowski
  map-state duality; Peres–Mermin KS; the quantum trilemma ≅ fire triangle ≅
  sheaf global-section obstruction; Zizzi's non-Tarskian logic of qubits.

## Risk

Low. Purely additive: new pure primitives in `larql-hilbert`, new
`larql-cli` extraction modules, and one new `quantum-signature` subcommand;
nothing in existing inference or extraction paths changes. Every witness is
calibrated to an exact analytic pole, so the apparatus's own correctness is
checked without any model download.
