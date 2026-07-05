# ADR-009: Extract is a typed Input→Output relation that runs confined and constructively witnesses totality

**Status**: Proposed
**Date**: 2026-06-16
**Context**: `larql extract` is two positional procedures (`build_vindex`, `build_vindex_streaming`) returning `Result<(), VindexError>` — no output value, with strategy routing scattered in the CLI. Two defects share a root: (1) the host can be CPU-saturated (no thread governance: OpenBLAS + rayon uncapped); (2) tensors can be silently dropped, yielding a hollow vindex with exit 0 (#147/#153/#158). Defect (2) is a non-constructivity: the code *assumes* every source tensor is handled (`∀t. written(t) ∨ skipped(t)`) without *constructing* the witness, so dropped tensors fall into a truth-value gap.

## Decision

Model extraction as a single typed relation, not a procedure:

```rust
fn extract(input: ExtractInput, io: &mut dyn IndexBuildCallbacks)
    -> Result<ExtractOutput, VindexError>;
```

with `ExtractInput` carrying the source, level, shaping, and `resources` (governance), and `ExtractOutput` carrying `components`, `skipped`, `completeness ∈ {Complete, Partial, Hollow, Unverified}`, and `resources` used. The relation `R ⊆ ExtractInput × ExtractOutput` has four clauses: **Confinement** (run ≤ `resources` threads), **Totality** (`components ⊎ skipped` partitions source tensors at `level`), **Level-completeness** (`Complete ⟺` no level-required tensor skipped), **Faithfulness** (each component decodes isomorphic to source modulo shaping).

Two principles bind the implementation:
- **Confined execution**: one thread budget `N` set before any matmul, applied to *both* OpenBLAS (`openblas_set_num_threads` + `OMP_NUM_THREADS`) and rayon, default never all-cores. Working memory is bounded *independent of model size* — `O(largest tensor + fixed scratch)` via streaming, or a refusal where a path cannot stream — so a model's extractability is determined by **disk** (source + output must fit), not RAM, and the host never crashes. In-memory fallback paths whose footprint cannot be bounded **hard-error** (constructive refusal) rather than estimate.
- **Constructive totality**: `completeness` is a *constructed* value. Code that does not enumerate-and-decide each source tensor reports `Unverified` — it must **not assert `Complete`**. `skipped` is the constructive witness of the partition.

## Consequences

- **Good**: governance and completeness become *post-conditions of one relation* the unit can be verified against, rather than ad-hoc command bodies.
- **Good**: silently-hollow vindexes become impossible to *claim* — absent a built witness the output is `Unverified`, not `Complete`.
- **Good**: bounded working set makes peak RAM sublinear in model size (evidence: 360M → 0.88 GB, 2B → ~2.9 GB peak under a 4-thread cap), so any model that fits on disk is extractable given time, on memory-constrained hosts, without crashing.
- **Good**: CLI strategy routing (`should_stream_*`) moves into `extract()`, shrinking the command to input-construction.
- **Good**: `down_meta` becomes an optional, confined stage (skippable at inference level since inference does not consume it; thread-capped and peak-bounded when run) — so the verification run completes without the 98%-wall-time projection that is the crash-adjacent hot stage.
- **Trade-off**: an implementation that does not build the partition emits `Unverified`; `Complete`/`Partial` population may be added later without changing the signature.
- **Trade-off**: the joint thread budget assumes extract's BLAS and rayon phases are disjoint; nested BLAS-in-rayon must scope BLAS to 1 (invariant).
- **Trade-off**: `rayon::build_global()` is process-set-once (first call wins); REPL/server apply once at startup.
