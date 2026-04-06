# CachedFfn — Precomputed FFN Outputs

**File:** `crates/larql-inference/src/ffn/experimental/cached.rs`
**Status:** Experimental (test harness)
**Speed:** 1us/layer (4160x faster than dense)
**Accuracy:** 100% bit-identical

## Description

Runs one dense calibration forward pass per prompt, caches the FFN output at every layer as
an `ArcArray2`. At inference time, returns the cached output (refcount bump) instead of
computing. Zero matmuls.

## Why it's experimental

Not scalable: requires one cache per (prompt, entity) pair. Each cache is ~2MB (34 layers x
seq_len x 2560 x 4 bytes). For 1000 entities that's 2GB. For 100K entities, 200GB.

The cache is only valid for the exact token sequence. Different phrasing of the same query
produces a different cache.

## Throughput

| Method | tok/s |
|--------|-------|
| clone (current trait) | 32K |
| memcpy (pre-alloc) | 171K |
| arc clone (refcount) | 3.2M |

## Value

Useful as a test oracle: calibrate once, verify other backends match. Also proves the
theoretical maximum FFN performance — attention is the remaining bottleneck at 588ms.

## Usage

```rust
let cached = CachedFfn::calibrate(weights, &token_ids);
let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &cached);
cached.save(&path)?;
let loaded = CachedFfn::load(&path)?;
```
