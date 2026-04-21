# FFN Activation Cache

The LARQL vindex walk FFN is deterministic and stateless: the same residual vector always produces the same output. The FFN activation cache exploits this to skip recomputation when the same feature set is activated again.

See [ADR-0002](adr/0002-ffn-activation-cache.md) for design rationale and architecture decisions.

---

## How It Works

Gate KNN maps a continuous residual to a discrete set of feature IDs. The cache key is a hash of those sorted IDs — not the raw residual. Two residuals that activate the same K features (even at different gate scores or in different order) share a cache key and share a cache entry.

This works because of **paraphrase collapse**: on Gemma 3 4B, residuals from semantically equivalent prompts have cosine similarity 0.98–0.99, and the sparse feature activation set is identical.

```
residual [f32; 2560]
    └─ gate_knn(layer) → [(feature_id, score); K]
         └─ sort IDs → hash → u64 cache key
              └─ hit:  return cached [f32; 2560]
              └─ miss: compute sparse walk, store, return
```

---

## Tiers

### L1 — In-Process (per WalkFfn instance)

Enabled per-instance with `.with_l1_cache(num_layers)`. Persists for the lifetime of the `WalkFfn` — one inference session in the CLI, one HTTP request in the server.

```rust
use larql_inference::vindex::WalkFfn;

let walk = WalkFfn::new(weights, &index, top_k)
    .with_l1_cache(num_layers);

// ... run inference layers ...

// Check stats at end of session
if let Some((hits, misses)) = walk.l1_cache_stats() {
    let total = hits + misses;
    println!("L1 hit rate: {:.1}%", 100.0 * hits as f64 / total.max(1) as f64);
}
```

**When it fires:** Only on the `walk_ffn_sparse` path, which requires `top_k * 2 < intermediate_size`. For Gemma 3 4B (intermediate=16384), this means `top_k < 8192`. The default bench top-k of 8092 meets this threshold.

**When it does NOT fire:**
- `top_k >= intermediate_size / 2` → interleaved or full-mmap path (no sparse KNN)
- `seq_len > 1` → prefill phase (multi-position, not cached)
- `index.has_overrides_at(layer)` → INSERT session active (see Patch Safety below)

### L2 — Server Process (shared across all clients)

Wired automatically into `POST /v1/walk-ffn` with `full_output: true` for single-position requests. No configuration required — present in every `larql-server` process.

Access stats from the server process via `model.ffn_l2_cache.stats()`. A `/v1/cache-stats` endpoint is planned (see ADR-0002 open questions).

**L2 warms automatically** across clients. Once any client has computed the output for a given feature set, every subsequent client gets the cached result. Common factual activations (major capitals, numbers, common verbs) stabilise after the first few hundred queries.

---

## Patch Safety

INSERT patches may change down/up vectors without changing the gate vector. If the gate is unchanged, the cache key is the same — but the output would differ. To prevent stale reads:

**Both L1 and L2 are bypassed when `index.has_overrides_at(layer)` is true.**

This means:
- A clean model (no INSERT) → cache is active for all layers
- An INSERT session → cache bypassed for layers that have overrides; active for layers without
- The override check is per-layer, not per-session, so a session that only patches L10 still gets cache hits at L0–L9 and L11–L33

This is validated in `examples/ffn_cache_demo.rs` (Scenario 3) and is the correct behaviour: correctness over hit rate for live-patched layers.

---

## Expected Hit Rates

| Scenario | L1 | L2 (warmed) |
|---|---|---|
| Repeated identical residual (same token) | ~100% | — |
| Paraphrase collapse (cos ≈ 0.99) | 60–90% | — |
| Common factual queries | 10–20% | 60–80% |
| Novel entities / unusual prompts | 5–10% | 20–30% |

---

## Benchmarking

```bash
cargo run --release -p larql-inference --example bench_ffn_cache -- \
  --model google/gemma-3-4b-it \
  --vindex path/to/gemma3-4b.vindex \
  --top-k 8092 \
  --iters 200
```

This prints baseline (no cache), cold-cache, warm-cache (100% hit), and rotating-residual hit rates + latency per call.

---

## Capacity and Eviction

Both L1 and L2 use a simple capacity cap per layer: once `max_entries` is reached, new entries are silently dropped. There is no LRU eviction in the current implementation.

| Tier | Default capacity | Approximate memory |
|---|---|---|
| L1 | 4096 per layer | ≤1.3GB total (34 layers × 4096 × 10KB) |
| L2 | 4096 per layer | ≤1.3GB total |

For most inference sessions the working set is far smaller — typical generation sessions see 10–200 unique feature sets per layer.

Custom capacity:

```rust
use larql_inference::FfnL1Cache;

// 512 entries per layer — reduces memory for edge deployment
let cache = FfnL1Cache::with_max_entries(num_layers, 512);
let walk = WalkFfn::new(weights, &index, top_k);
// (direct field access; or use the builder if you add a new constructor)
```

---

## Cache Key Stability

The cache key is a `DefaultHasher` hash of sorted feature IDs. This means:

- **Order-independent** — gate-score ranking doesn't affect the key
- **Stable within a process** — `DefaultHasher` is deterministic per-run but not cross-process (intentional: no cross-process cache poisoning)
- **Not cross-tier portable** — L1 and L2 use the same algorithm, but a key from one process cannot be assumed valid in another

For the L3 CDN tier (planned), keys would need to be serialised alongside the model version counter to survive server restarts.
