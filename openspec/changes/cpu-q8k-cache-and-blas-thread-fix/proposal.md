## Why

While profiling the post-#144 decode loop on Gemma 3 4B, two issues surfaced
that compound when both are present:

1. **`with_q8k_for` cache aliasing**
   (`crates/larql-models/src/quant/lazy.rs:470`) keys its Q8_K-quantised
   activation cache by `(x.as_ptr(), x.len())`. The MoE path that motivated
   the cache keeps `x` alive across sibling expert matvecs, so the cache
   pays off and the pointer never gets reused mid-flight.
   PR #144's lm_head path doesn't — `last_2d.row(0).to_owned()` allocates
   a fresh `Array1<f32>` per decode step and drops it immediately; the
   allocator routinely reuses the same heap slot for the next step's
   `x`. The cache check sees `(ptr, len)` matching the previous step's
   value, returns the STALE Q8_K bytes, and lm_head produces logits for
   a token that was last fed several steps ago. Output: non-deterministic
   gibberish ("Okay, let's dive into the wonderful world of cats cats! …
   complex,,, fascinating creatures with a long and and and rich rich
   history…"). Two consecutive runs of the same prompt diverge.

2. **OpenBLAS multi-threading contends with rayon**
   `gqa_attention_decode_step` calls `ndarray::Array2::dot` for the
   per-head Q·K^T and softmax·V dots. With OpenBLAS default thread count
   (= core count, 48 on the EPYC host), each tiny 150×256 dot dispatches
   through OpenBLAS's 48-thread fork-join — but the rayon-parallel
   matvec from #143/#144 is already using all 48 cores in the
   surrounding work. The BLAS dispatch overhead piles up to ~160ms per
   decode step at 150-token cache. Total decode-step time grows from
   89ms (cache_len=10) to 252ms (cache_len=150), almost all of it in
   BLAS overhead, not actual compute.

Both fixes are needed for the post-#144 path:
- Cache fix alone (multi-thread BLAS): output is coherent but decode is
  bandwidth-OK but BLAS-overhead-bound at 4.18 tok/s.
- BLAS pin alone (cached lm_head path): output is gibberish.
- **Both together**: coherent output, 9.81 tok/s.

## What This Change Ships

**`crates/larql-models/src/quant/lazy.rs`** — `with_q8k_for` cache key extended
to include f32 fingerprints of `x[0]` and `x[len-1]`. The MoE pattern (`x`
alive across sibling matvecs) still hits the cache trivially — both
fingerprints match. The lm_head pattern (per-step Vec, allocator reuse)
gets a fingerprint mismatch and re-quantises. No perf loss on either path;
correctness restored.

**`crates/larql-server/src/main.rs`** — pin OpenBLAS / OpenMP / MKL thread
count to 1 at process start, via both env var (catches lazy-init) and
the `openblas_set_num_threads(1)` FFI call (catches eager-init). Respect
user-supplied env vars so explicit overrides (`OPENBLAS_NUM_THREADS=N`)
still work.

## Bench (Gemma 3 4B Q4_K_M, 48-thread EPYC host)

| Variant | Output | tok/s @ 150 tok |
|---|---|---:|
| BLAS default + stale cache | gibberish | 1.79–4.18 |
| BLAS=1 + stale cache | gibberish | 9.89 (broken) |
| BLAS default + cache fix | **coherent** | 4.18 |
| **BLAS=1 + cache fix (this PR)** | **coherent** | **9.81** |
| llama.cpp CPU reference | coherent | 14.1 |

End-to-end Gemma 3 4B: **9.81 tok/s** = **83.9×** the 0.117 baseline,
**70% of llama.cpp**.

## Capability Deltas

- Under `compute-backend-traits/`: `with_q8k_for` content-aware cache key.
- Under `server-vindex-loading/`: BLAS thread pinning at server startup.

## Why one PR

The two fixes are tightly coupled. The cache hazard is independently a
correctness bug (latent before #144 because the cache caller pattern in
MoE/Qwen kept `x` alive). PR #144's lm_head path exposed it. The BLAS
fix is the next-biggest perf lever, but lands as gibberish unless the
cache hazard is fixed first. Shipping them together is cleaner than
relying on bisection to surface the pairing.
