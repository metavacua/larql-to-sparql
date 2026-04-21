# ADR-0002 — Three-Tier FFN Activation Cache

**Status:** Implemented (Phase 1 L1, Phase 2 L2). Phase 3 (CDN) deferred.

---

## Context

The LARQL vindex walk FFN replaces dense matmul with sparse KNN lookup: a residual vector (2560 floats) is projected against gate vectors, the top-K matching features are selected, and their down-projection vectors are gathered and summed. This computation is:

- **Deterministic** — same residual always produces the same output
- **Discrete at the feature boundary** — gate KNN maps a continuous residual to a discrete sparse feature index set
- **Stateless** — no side effects, no session dependency

These three properties make the FFN walk output a natural candidate for aggressive caching. Dense matmul has none of these properties; the vindex architecture uniquely enables this approach.

### The Paraphrase Collapse Observation

Mechanistic interpretability work on Gemma 3 4B shows that the residual stream is effectively 1-dimensional (1 PC ≈ 90–95% of variance). Similar prompts — paraphrases, synonymous phrasings, structurally equivalent queries — collapse to nearly identical residuals at L10 (cosine similarity 0.98–0.99). This means the cache hit radius is much larger than the syntactic diversity of inputs would suggest.

---

## Cache Key Design

Cache on the **sparse feature index set** returned by gate KNN — not the raw residual.

```
residual: [f32; 2560]
    → gate_knn(residual, layer=N) → [(feature_id, gate_score); K]
    → sort feature IDs (make key order-independent)
    → hash(sorted_ids) → cache_key: u64
```

**Why not the raw residual?** Floating-point residuals are sensitive to context length, quantisation noise, and prompt phrasing. Two residuals with cosine similarity 0.99 will activate the same sparse feature set — hashing the feature set captures the equivalence class that raw residual hashing would miss.

**Key normalisation:** Feature IDs are sorted before hashing so gate-score order doesn't affect the key. Two residuals that activate the same features in different score order hit the same cache entry.

---

## Architecture

```
larql-cli (inference session)
  └─ WalkFfn::walk_ffn_sparse()
       └─ L1: FfnL1Cache  (HashMap per layer, RefCell, session-scoped)

larql-server (process lifetime)
  └─ routes/walk_ffn.rs :: run_full_output()
       └─ L2: FfnL2Cache  (RwLock<HashMap> per layer, Arc values, cross-client)

[Phase 3, not yet implemented]
  └─ CDN / Workers KV — global, pre-seeded from labelled features
```

---

## Tier Specifications

### L1 — In-Process Cache (`larql-inference`)

| Property | Value |
|---|---|
| Location | `WalkFfn` heap, `RefCell<HashMap<u64, Vec<f32>>>` per layer |
| Scope | Single `WalkFfn` instance (one inference session or one HTTP request) |
| Default capacity | 4096 entries per layer |
| Eviction | None (bounded by `max_entries`; new entries dropped when full) |
| Activation | `WalkFfn::new(...).with_l1_cache(num_layers)` |
| Stats | `walk_ffn.l1_cache_stats()` → `(hits, misses)` |
| Path | Only fires on `walk_ffn_sparse` (bounded top-k < intermediate/2) |

### L2 — Server Process Cache (`larql-server`)

| Property | Value |
|---|---|
| Location | `LoadedModel.ffn_l2_cache`, `RwLock<HashMap<u64, Arc<Vec<f32>>>>` per layer |
| Scope | Server process lifetime, shared across all clients |
| Default capacity | 4096 entries per layer |
| Eviction | None (bounded by `max_entries`) |
| Stats | `model.ffn_l2_cache.hits()`, `.misses()`, `.stats()` |
| Path | `run_full_output`, single-position requests only (`seq_len == 1`) |

### L3 — CDN / Distributed KV (not yet implemented)

Pre-seeded from 1,923 labelled features × 34 layers. Write-back from server on miss. Globally persistent across server restarts.

---

## Patch Safety

**Problem:** The cache key is derived from gate KNN feature IDs only. A patch that changes a down/up vector without changing the gate vector would produce the same feature IDs → same cache key → stale cached output.

**Fix:** Both L1 and L2 skip the cache entirely when `index.has_overrides_at(layer)` returns true. This means:

- Clean model (no INSERT) → cache is active
- Patched session (after INSERT) → cache bypassed for that layer
- Cost: a cache miss on every call to a patched layer — which is correct, since the output changes with the patch

This is tested explicitly in `examples/ffn_cache_demo.rs` (Scenario 3).

---

## Miss Propagation

```
L1 miss → L2 lookup → HIT: populate L1, return
                     → MISS: compute walk_ffn_sparse, populate L1 + L2
```

The L2 gate-KNN call in `run_full_output` uses the request's `top_k` to derive the cache key, then calls `walk_ffn.forward()` which does gate KNN again internally on miss. This double gate-KNN on miss is accepted as the cost of simplicity — gate KNN is fast relative to the full FFN computation it replaces.

---

## Expected Hit Rates

| Query type | L1 (within session) | L2 (cross-client, warmed) |
|---|---|---|
| Repeated token in generation | 30–40% | — |
| Common factual (capitals, numbers) | 10–20% | 60–80% |
| Novel entity / unusual prompt | 5–10% | 20–30% |

---

## Implementation Files

| File | Role |
|---|---|
| `crates/larql-inference/src/vindex/l1_cache.rs` | `FfnL1Cache` struct + unit tests |
| `crates/larql-inference/src/vindex/walk_ffn.rs` | L1 wired into `walk_ffn_sparse` |
| `crates/larql-server/src/ffn_l2_cache.rs` | `FfnL2Cache` struct + unit tests |
| `crates/larql-server/src/state.rs` | `LoadedModel.ffn_l2_cache` field |
| `crates/larql-server/src/routes/walk_ffn.rs` | L2 wired into `run_full_output` |
| `crates/larql-inference/examples/ffn_cache_demo.rs` | Demo: hit rates + patch safety |
| `crates/larql-inference/examples/bench_ffn_cache.rs` | Benchmark: latency delta |
| `docs/ffn-cache.md` | User-facing guide |

---

## Open Questions

1. **Optimal K for key stability.** Smaller K → more collisions (higher hit rate, lower precision). Larger K → more specific. Profile on benchmark queries to find the sweet spot.

2. **Compression of cache values.** 10KB per entry is manageable. The FFN output vector is sparse in practice — int8 or delta-compressed storage could reduce to 2–3KB.

3. **Cache invalidation on vindex patch.** The current bypass (`has_overrides_at`) is conservative — it skips the cache for the whole layer. A per-feature invalidation scheme could recover hits for unaffected features.

4. **L2 LRU eviction.** Current implementation drops new entries when full. LRU would improve hit rates for workloads with many unique residuals.

5. **Hit rate measurement.** Add `GET /v1/cache-stats` endpoint to expose `FfnL2Cache::stats()`.
