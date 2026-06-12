# Roadmap — larql-hilbert

`larql-hilbert` is the Hilbert-space formalization crate: complex structures,
the qubit / Bloch-sphere language models, constructive measurement, and the
entanglement-entropy / compressibility meters. It is a pure-Rust leaf crate
(deps: `ndarray` + `num-complex`; no BLAS) — see
[ADR 0011](../../docs/adr/0011-qlm-quantum-language-models.md).

The full phased plan, theory, and bibliography live in the QLM roadmap:

➡️ **[`docs/QLM-ROADMAP.md`](../../docs/QLM-ROADMAP.md)** — Copyright © 2026
Ian Douglas Lawrence Norman McLean, CC BY-SA 4.0.

## At a glance

- ✅ Phase 0–3: Hilbert foundation + Hilbertian residual (PR #137), single-qubit
  Bloch LM (#138), constructive measurement + admissibility (#139), two-qubit
  LM + Bell (#140).
- 🔄 Phase 4: entanglement-entropy / compressibility meter
  (`spectral_entropy` landed; `entanglement_entropy(matrix)` + pure-Rust
  eigensolver per `docs/superpowers/plans/2026-06-11-matrix-entanglement-entropy.md`).
- ⬜ Phase 5–6: GHZ/W (3+ qubits), superdense coding + quantum LQL.
- 🔬 Phase 7–8: vindex compressibility analysis; `wasm32v1-none` minimal LM
  (epic #132).
