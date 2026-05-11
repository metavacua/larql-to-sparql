# ADR-009: Feature-Major Q4_K Down

**Status**: Accepted
**Date**: 2026-04-25
**Context**: The down-projection cache (`q4k_ffn_layer`) was the only
remaining heap-side cache on the FFN data path. It capped at ~840 MB
on Gemma 4B and required a Mutex on first access; on multi-shard
grid servers and MoE workloads the cache never amortised because
each shard touched each layer once or twice.

## Decision

Emit down weights twice when `Q4kWriteOptions::feature_major_down=true`:
- Once in `interleaved_q4k.bin` at `[hidden, intermediate]`
  orientation (the existing slot — preserved for full-K matmul).
- Once in a new file `down_features_q4k.bin` at
  `[intermediate, hidden]` orientation, Q4_K/Q6_K-encoded with the
  same precision as the interleaved down slot.

Per-feature down decode (`ffn_row_scaled_add` for `component == 2`)
prefers the feature-major file when present — a single row dequant
replaces the whole-layer dequant + transpose. Falls back to the
legacy cache for vindexes extracted before this landed.

## On-disk layout

```
model.vindex/
├── interleaved_q4k.bin              [hidden, intermediate] down (existing)
├── down_features_q4k.bin            [intermediate, hidden] down (W2)
└── down_features_q4k_manifest.json  per-layer (offset, length, format, shape)
```

The manifest entry shape is `Q4kManifestEntry` shared with
`interleaved_q4k_manifest.json` and `attn_weights_q4k_manifest.json`
(see `format/weights/manifest.rs`). Loaders deserialise into the
typed struct rather than poking `serde_json::Value` with string keys.

## Trade-offs

| | Cache (legacy) | Feature-major (W2) |
|---|---|---|
| Disk overhead | 0 (data shared with interleaved) | ~14 MB / layer at Gemma 4B (~500 MB / 34 layers) |
| Heap ceiling | up to ~840 MB / VectorIndex on Gemma 4B | 0 — straight mmap |
| First-access decode (K=100) | 77.6 ms | 31.8 µs (2440×) |
| First-access decode (full K) | 82.9 ms | 3.24 ms (25×) |
| Warm-cache decode | scaled-add only (fast) | scaled-add only (fast) |
| Lock contention | Mutex on cache | none |

## When the new path actually fires

The W2 dispatch lives inside `ffn_row_scaled_add` for `component == 2`,
which is called by `walk_ffn_sparse`. Sparse walk runs when at least
one of:

- the layer has overrides (post-INSERT patches),
- `WalkFfnConfig::is_sparse(layer)` is true (explicit sparse-K),
- the vindex has FP4 storage (FP4 always routes through sparse).

The default dense Q4K walk (`walk_ffn_q4k_dequant`) does an inline
full-layer dequant + dense matmul instead — it bypasses both the
legacy `q4k_ffn_layer` cache *and* the W2 feature-major path. For
pure-dense Q4K traffic the cache stays at 0 slots either way; the
value of W2 there is the *capability* — you can hot-attach a patch or
switch on sparse mode and still hit the per-feature path without
lighting up an unbounded cache.

Production Metal full-K decode goes through `q4k_matmul_transb` and
also bypasses both paths.

## When to enable

- **Yes**: CPU sparse walk, interpretability pipelines, multi-shard
  grid servers running INSERT-heavy workloads, MoE experts (Kimi,
  DeepSeek-V3+) — anywhere the cache *would* fire and the RSS bound
  matters.
- **Yes (defensive)**: pure-dense Q4K grid servers where you might
  later add patches or sparse-K. The disk overhead is the price of
  preserving the cache-bounded RSS guarantee.
- **No**: Metal-only decode farms with no patch traffic. The disk
  overhead buys nothing today.

Default is **off**. CLI flag `--feature-major-down` on
`larql extract-index` and `larql convert quantize q4k`. Live status:
`GET /v1/stats` → `q4k_ffn.feature_major_down`.

## Why not delete the legacy cache?

Two reasons. (1) Vindexes extracted before W2 landed don't have the
file; the cache stays as the fallback so old artefacts keep
working. (2) The cache is correct in its own right — feature-major
is faster on first access and avoids the heap ceiling, but the
cache is the right answer for warm decode of a tight layer-set.
A future round can revisit deleting the cache once feature-major
is the norm.

## References

- W2 in `ROADMAP.md`
- `format/weights/write_q4k/feature_major_down.rs` — emit
- `index/storage/ffn_store/mod.rs::load_down_features_q4k` — load
- `index/compute/q4k_dispatch.rs::q4k_down_feature_scaled_add` — dispatch
- `tests/test_vindex_to_q4k.rs::q4k_feature_major_down_round_trip` — round-trip
- `benches/q4k_cache.rs::bench_down_cache_vs_feature_major` — perf
