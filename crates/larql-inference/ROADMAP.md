# Roadmap — larql-inference

## Current: 4.9 tok/s honest (real model) | 59 tok/s GPU synthetic | Ollama: 97 tok/s

## P0: Close Ollama Gap

### Fix GPU prefill for post-norm models (Gemma3)
**Impact**: 203ms → ~17ms honest with GPU prefill  
**Effort**: Medium  
**Status**: In progress — activation fix done, post-norm wiring incomplete

The GPU `prefill_q4` path produces wrong output for Gemma3 post-norm architecture.
Root cause: `prefill.rs` doesn't mirror `full_pipeline.rs`'s post-norm handling.
CPU fallback is correct. See larql-compute ADR-009.

### Wire KV-cached decode into honest path
**Impact**: 4.9 tok/s → 59+ tok/s decode  
**Effort**: Low  
**Status**: Infrastructure ready

After prefill populates KV cache, subsequent decode_token calls at seq=1 should
give 59 tok/s (measured in compute benchmarks). Need to wire the prefill → decode
loop in predict_honest or a new `generate()` function.

### Merge per-layer dispatches
**Impact**: ~30% speedup on GPU path  
**Effort**: Medium  
**Status**: Identified in compute component profiling

Currently 7 encoders per layer. Merging norm+QKV+attend+O+FFN into fewer encoders
would save ~8ms on the 34-layer GPU path.

## P1: Production Hardening

### Clean up experimental FFN backends
**Effort**: Low  
**Status**: Not started

6 experimental FFN backends in `ffn/experimental/` (CachedFfn, ClusteredFfn, etc.).
Should be moved to a research module or removed if superseded by WalkFfn.

### Example reorganization
**Effort**: Low  
**Status**: Not started

22 examples need prefix-based organization like larql-compute:
`demo_`, `compare_`, `profile_`, `bench_`, `test_`

### Add doc tests
**Effort**: Low  
**Status**: 0 doc tests currently

Add examples to `attention.rs`, `forward.rs`, `layer_graph/mod.rs`.

## P2: Research

### Template-guided walk (restrict feature universe)
Pre-compute per-template feature sets. Only score features in the template's universe.
Reduces gate KNN work for known entity types.

### Multi-token generation loop
`generate(prompt, max_tokens)` → prefill once, decode in loop with KV cache.
Currently predict_honest does one prediction. Need streaming generation.

## Completed

| Item | Date | Impact |
|------|------|--------|
| Forward pass (CPU BLAS) | 2026-03 | Foundation |
| BLAS-fused attention | 2026-04-03 | Online softmax, O(seq) memory |
| WalkFfn (sparse FFN via vindex) | 2026-04-03 | Gate KNN + top-K |
| CachedLayerGraph | 2026-04-04 | Skip L0-12, 0.999 cosine |
| LayerGraph trait | 2026-04-04 | Pluggable per-layer routing |
| predict_honest | 2026-04-06 | Production path, GPU+CPU hybrid |
| GPU prefill pipeline | 2026-04-06 | seq>1 on GPU (pre-norm models) |
| Q4_K FFN format wiring | 2026-04-07 | Vindex Q4_K FFN → FullPipelineLayer |
| GELU-tanh activation | 2026-04-07 | Gemma3 correct on GPU |
| Post-norm guard | 2026-04-07 | Gemma3 falls to CPU correctly |
| Zero warnings | 2026-04-07 | Clean build |
| PERFORMANCE.md | 2026-04-07 | Benchmark data documented |
