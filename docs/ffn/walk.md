# WalkFfn — Vindex Gate KNN with Sparse FFN

**File:** `crates/larql-inference/src/vindex/walk_ffn/` (module directory — routing in `mod.rs`, one file per execution path)
**Status:** Production as the *instrumentable/editable execution layer* and the CPU sparse path. Raw decode throughput is served by the Q4K GPU decode path (~88 tok/s, Gemma 3 4B on M3 Max) — the CPU INFER walk runs ~1.9 tok/s (517 ms/tok); see the repo README "Benchmarks" tables.
**Accuracy:** Walk-vs-dense numerical parity is pinned by the parity suite (2026-07-30 review, item 20). Historical result (2026-04-03, [`docs/walk-boundary-sweep.md`](../walk-boundary-sweep.md)): identical top-1 to all-dense at every layer boundary L0–L34 at gate-KNN top-K = 8092 — that is the sweep harness's literal constant (79% of Gemma 3 4B's 10,240 features, almost certainly a typo for 8192 that got baked into the harness; the same 8092 appears in `docs/ffn/sparse.md` and the remote-codec tests). The companion "97.91% on France→Paris" figure is the 2026-04 LQL-spec INFER example run (with attention), not the boundary sweep's 80.47% all-dense ground truth.

Note on K: with the current dispatch, a requested K at or above 80% of a
layer's feature count is rewritten to the exact full-K gemv fast path
(`walk_ffn/thresholds.rs`, `FULL_K_DENSITY_NUM/DEN`) — on Gemma 3 4B
that is K ≥ 8192, so K=8092 is a genuinely sparse walk while K=8192
would execute densely (and exactly) unless `force_walk` is set.

## Description

The FFN backend that replaces dense matmul with vindex lookups: vindex gate KNN for feature
selection, then sparse FFN computation on only the selected features. Feature selection on the
hot path is the **exact** `gate_walk` batched gemv — `enable_hnsw()` never affects walk numerics
(2026-07-30 review, item 13). Captures a walk trace showing which features activated and what
they mean.

This is the backend used by the LQL `INFER` statement.

## Architecture

```
Input x (post-attention residual)
  │
  ├─► GateIndex::gate_walk(layer, x_last, top_k)  →  feature selection (exact gemv)
  │     Falls back to gate_knn only where exactness is unavailable
  │     (Q4K-interleaved-only gates, override/tombstone overlay layers).
  │     Uses VectorIndex or PatchedVindex (both implement GateIndex)
  │
  └─► sparse_ffn_forward(weights, layer, x, features)  →  sparse FFN output
        Only computes gate/up/down for selected features
```

The `GateIndex` trait abstracts over both `VectorIndex` (base, readonly) and `PatchedVindex`
(with overlay). This means INSERT/DELETE/UPDATE to the vindex immediately affect inference
output — patched gate vectors are used for feature selection.

## Walk Trace

Since 2026-07-31 (review item 17) the trace is **emitted at runtime from
the executed path**: `take_trace` reports the features the walk actually
EXECUTED — selector, route-pool and cell-router selections included —
not a post-hoc `gate_knn` re-run. Tracing requires `with_trace` /
`new_with_trace` (it upgrades every call to `Observe::Record`).

Each layer's trace contains:
- **Feature ID** — which FFN feature activated
- **Gate score** — how strongly it activated
- **Down meta** — what token this feature predicts (from the vindex)

Example for "The capital of France is":
```
L27: F9515  gate=+9.247  hears="Paris"   c=0.05
L26: F5040  gate=+7.880  hears="French"  c=0.08
L28: F8200  gate=-5.297  hears="France"  c=0.08
```

## Usage

```rust
use larql_inference::vindex::WalkFfn;

// Works with VectorIndex (unpatched)
let walk_ffn = WalkFfn::new(weights, &index, top_k);

// Works with PatchedVindex (mutations visible). Patched layers first
// try the exact base+delta dense path (review item 16) and fall back
// to the sparse walk only when its preconditions decline.
let walk_ffn = WalkFfn::new(weights, &patched, top_k);

// Tracing needs the trace-enabled constructor (runtime emission).
let walk_ffn = WalkFfn::new_with_trace(weights, &index, top_k);
let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);
let trace = walk_ffn.take_trace(); // features the executed walk actually used
```

## LQL

```sql
INFER "The capital of France is" TOP 5;
EXPLAIN INFER "The capital of France is" TOP 5;
```
