# ADR-0013 — LQL ↔ wasm32v1-none Correspondence

**Status:** Accepted  
**Date:** 2026-06-02  
**Depends on:** `larql-lql`, `larql-wasm-certify`, `larql-lql-wasm32v1-none`

---

## Context

The LARQL system has two parallel concerns that are formally related:

1. **LQL statement semantics** — each `Statement` variant in `crates/larql-lql/src/ast.rs`
   defines an operation over a vindex (a graph database of transformer weights).

2. **wasm32v1-none deployability** — the `larql-wasm-certify --strict` tool verifies that
   a compiled `.wasm` module is in the ¬L∧¬M class: no recursion (¬L) and no unbounded
   heap growth (¬M). This class is sub-Turing: provably terminating, formally decidable.

The question is: which LQL statements have evaluators that satisfy the ¬L∧¬M property,
and what are the precise blockers for those that do not?

This matters for three reasons:

- **Browser deployment.** A ¬L∧¬M-certified evaluator can run in a browser WASM sandbox
  without a trusted execution environment. Users can query a vindex client-side with formal
  termination guarantees.
- **Distillation target.** The LQL synthesis toolchain (`lql_distill.py`) distills LQL
  knowledge into a vindex. The certified subset defines which queries the distilled model
  can answer in a browser context.
- **Formal boundary documentation.** The attention parallelizability seam (ADR-0014) and the
  Born Rule replacement (ADR-0015) both require knowing exactly which statements cross the
  ¬L∧¬M boundary and why.

---

## The Correspondence

For each LQL `Statement` variant `S`:

```
certified(S)  ⟺  evaluator(S) compiles to wasm32v1-none
              ∧   larql-wasm-certify --strict evaluator(S).wasm exits 0
```

Compilation to wasm32v1-none is the **ceiling gate** — necessary but not sufficient.
`--strict` certification is the **floor gate** — it verifies the sub-Turing properties
(call-graph acyclicity for ¬L; absence of `memory.grow` for ¬M).

---

## Statement Classification

### Certified subset — browser LQL dialect

These statements have ¬L∧¬M evaluators. Their evaluators compile to wasm32v1-none and
pass `larql-wasm-certify --strict`.

| Statement | Category | ¬L | ¬M | Notes |
|---|---|:---:|:---:|---|
| `WALK` | query | ✅ | ✅ | Gate KNN + sparse down: bounded loops over fixed feature arrays, zero allocation |
| `DESCRIBE` | query | ✅ | ✅ | Feature scan: bounded KNN query over mmap'd gate vectors |
| `SELECT` | query | ✅ | ✅ | Predicate scan over fixed-size edge/feature/entity tables |
| `SHOW RELATIONS` | introspection | ✅ | ✅ | Scan over static layer metadata |
| `SHOW LAYERS` | introspection | ✅ | ✅ | Bounded iteration over layer range |
| `SHOW FEATURES` | introspection | ✅ | ✅ | Bounded scan with filter |
| `SHOW ENTITIES` | introspection | ✅ | ✅ | Bounded scan |
| `SHOW MODELS` | introspection | ✅ | ✅ | Metadata read |
| `STATS` | introspection | ✅ | ✅ | Fixed-size aggregate over vindex metadata |
| `INSERT` (KNN mode) | mutation | ✅ | ✅ | Writes to a fixed feature slot at bounded layer range; `find_free_feature` is O(features_per_layer) bounded scan |
| `DELETE` | mutation | ✅ | ✅ | Predicate scan + fixed-size slot clear |
| `UPDATE` | mutation | ✅ | ✅ | Predicate scan + in-place field write |

Note: `INSERT` in **Compose mode** (`install_compiled_slot`) involves a more complex
gate/up/down overlay write that warrants separate certification analysis before
including in the certified subset. KNN mode is the default and is certified.

### Not certified — native context only

These statements have evaluators that violate ¬L, ¬M, or both. They require the native
`larql` binary and cannot run in a browser WASM sandbox without architectural changes.

| Statement | Category | Blocker | Path to certification |
|---|---|---|---|
| `INFER` | query | **¬L∧¬M violation**: softmax attention requires a global L1 reduction (all scores before any output). This is the Phase 3 seam — see ADR-0014. | Born Rule attention (L2 normalization, ADR-0015) removes the global reduction. |
| `EXPLAIN INFER` | query | Same as `INFER` (runs full forward pass) | Same as `INFER` |
| `TRACE` | trace | **¬M violation**: captures per-layer residual vectors (`Vec<f32>` per layer), unbounded by sequence length and layer count | Could be certified for a bounded-layer, single-position variant |
| `COMPILE` | lifecycle | **¬M violation**: bakes patches by reading and rewriting weight matrices — allocation proportional to model size | Structural: COMPILE is inherently an offline/build-time operation |
| `EXTRACT` | lifecycle | **¬M violation + OS imports**: downloads and parses safetensors (GBs of allocation) + file I/O host imports | Structural: EXTRACT is a build-time operation; no WASM path intended |
| `DIFF` | lifecycle | **¬M**: allocates a diff result proportional to vindex size | Bounded variant (single-layer diff) could be certified |
| `REBALANCE` | mutation | **¬L**: iterative fixed-point loop; number of iterations data-dependent | Bounded variant (`MAX N`) with static iteration count could be certified |
| `COMPACT MAJOR` | introspection | **¬M**: full vindex rewrite | Structural: offline operation |
| `INSERT` (Compose mode) | mutation | Requires deeper analysis of `install_compiled_slot` | Pending certification audit |
| `PIPE` | composition | Inherits the blocker of the most-restricted component | `WALK \| SELECT` would be certified; `WALK \| INFER` would not |

### Conditionally certified

These statements are certified in a wasm32v1-none context but have their OS-dependent
operations compiled out (gated by `target_os = "none"`). The gate works correctly because
the WASM runtime has no OS to provide these operations anyway.

| Statement | Gate | WASM behavior |
|---|---|---|
| `BEGIN PATCH` | file I/O compiled out | No-op in browser; patch state held in linear memory |
| `SAVE PATCH` | file I/O compiled out | No-op in browser; caller must retrieve patch via WASM memory API |
| `APPLY PATCH` | file I/O compiled out | Patch data passed as argument, not read from file |
| `SHOW PATCHES` | file I/O compiled out | Returns in-memory patch list only |
| `REMOVE PATCH` | file I/O compiled out | Clears in-memory patch only |
| `COMPACT MINOR` | bounded rewrite | Certified for the fixed-slot compaction variant |
| `USE` (vindex) | file I/O compiled out | vindex is passed as a pre-loaded memory buffer in WASM context |

---

## The Browser LQL Dialect

The certified subset forms a self-contained query language for browser-side vindex access:

```sql
-- Read
WALK "The capital of France is" TOP 10;
DESCRIBE "France";
SELECT * FROM EDGES WHERE relation = "capital" LIMIT 20;
EXPLAIN WALK "The speed of light is";
SHOW RELATIONS;
SHOW LAYERS;
STATS;

-- Mutate (KNN mode only)
INSERT INTO EDGES (entity, relation, target) VALUES ("Atlantis", "capital-of", "Poseidon");
DELETE WHERE entity = "Atlantis" AND relation = "capital-of";
UPDATE EDGES SET confidence = 0.9 WHERE layer = 26;
```

This is a read/write graph query surface over a vindex with no full inference — sufficient
for knowledge browsing, edge insertion, and retrieval, but not for autoregressive generation.
`INFER` requires either native execution or the Born Rule attention replacement (ADR-0015).

---

## Certification Procedure

The certifier at `crates/larql-wasm/larql-wasm-certify/src/main.rs` checks:
1. Import-free (no host imports)
2. `call_indirect`-free (no indirect calls / ACE surface)
3. `memory.grow`-free (¬M gate, requires `--strict`)
4. Call-graph acyclicity (¬L gate, requires `--strict`)

To certify the browser dialect evaluator:

```bash
# Build the LQL wasm32v1-none crate
cargo build \
  --manifest-path crates/larql-wasm/Cargo.toml \
  -p larql-lql-wasm32v1-none \
  --target wasm32v1-none \
  --release

# Certify
cargo run \
  --manifest-path crates/larql-wasm/Cargo.toml \
  -p larql-wasm-certify \
  -- --strict \
  crates/larql-wasm/target/wasm32v1-none/release/larql_lql_wasm32v1_none.wasm
```

Expected: `WASM-SAFE [¬L∧¬M]: import-free + call_indirect-free + memory.grow-free + recursion-free`

This should be added as a CI gate once `larql-lql-wasm32v1-none` is sufficiently complete
to compile the certified-subset evaluator paths. Currently the crate exists in the wasm
workspace but the evaluator stubs are incomplete — the CI gate is deferred until the stubs
cover at least WALK, DESCRIBE, SELECT, and INSERT (KNN mode).

---

## Consequences

**Positive.**

- The browser dialect is formally defined and machine-checkable. Any future statement added
  to LQL has a clear certification path and a clear reason if it fails.
- The distillation toolchain (`lql_distill.py`) targets first-token prediction for all
  statement types, but only the certified subset can be *evaluated* in a browser context.
  This defines the scope of the browser-deployable LQL model.
- The ¬L∧¬M boundary for LQL mirrors the ¬L∧¬M boundary for compute kernels
  (see `crates/larql-compute/docs/compute-inventory.md`). The same certifier, the same
  formal class, applied one abstraction level higher.

**Negative / cost.**

- `INFER` is not in the certified subset. A user querying a browser-deployed vindex cannot
  run autoregressive inference without downloading the native binary or waiting for ADR-0015.
- The conditionally-certified patch statements need a WASM host API for passing patch data
  as memory buffers rather than file paths — a small but non-trivial integration point.

**Not in scope.**

- REBALANCE certification (bounded variant) — deferred; the fixed-point loop analysis
  requires a separate convergence proof.
- DIFF certification (single-layer variant) — deferred; lower priority.
- INSERT Compose mode certification — requires auditing `install_compiled_slot` for
  allocator usage.
