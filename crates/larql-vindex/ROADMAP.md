# Roadmap — larql-vindex

## Current State

- 146 tests passing, 0 build warnings
- 3 storage formats: f32, Q8, Q4_K/Q6_K (Ollama-compatible)
- Mmap zero-copy with adaptive residency
- HNSW graph index for sub-linear KNN
- Patch system for editable knowledge

## P0: Support Cached Layer Decode

### Store pre-computed residuals for template-fixed layers (L0-12)
**Impact**: Enables 155+ tok/s decode (skip 13 of 21 layers)  
**Effort**: Medium  
**Status**: Not started (infrastructure ready — CachedLayerGraph in larql-inference)

The vindex needs to store cached residuals per template. During extraction, run one forward pass per template through L0-12 and save the output residual. At decode time, look up the cached residual instead of computing 13 layers.

### Wire Q4_K FFN consumption (interleaved_q4k.bin) — DONE
**Impact**: Match Ollama's exact FFN quantization  
**Effort**: Medium  
**Status**: ✅ Complete (2026-04-07)

Added `load_interleaved_q4k()`, `has_interleaved_q4k()`, `interleaved_q4k_mmap_ref()` to vindex.
Inference `predict_honest` now prefers Q4_K FFN (`interleaved_q4k.bin`) over Q4_0.
Format tag (`ffn_format`) passed through `FullPipelineLayer` to compute for shader dispatch.

### GGUF Q4_K format option (144 bytes vs 148 bytes)
**Impact**: Direct compatibility with llama.cpp weight files  
**Effort**: Low  
**Status**: Quantizer ready in larql-compute (`quantize_q4_k_gguf`)

Add option to store attention weights in GGUF-canonical 144-byte Q4_K format (packed scales+mins in 12 bytes) instead of our 148-byte format.

## P1: Production Hardening

### HuggingFace resolution in Vindexfile
**Effort**: Medium  
**Status**: TODO in `vindexfile/mod.rs:162`

FROM directive in Vindexfile should resolve `hf://user/repo` paths.

### Streaming extraction checkpoints
**Effort**: Medium  
**Status**: Not started

Save extraction progress between layers so interrupted builds can resume.

### Q4_K FFN in vindex
**Effort**: Low  
**Status**: Not started (Q4_0 interleaved exists)

Currently FFN gate/up/down stored as Q4_0. Switch to Q4_K (matching Ollama) for better precision at similar size.

## P2: Research

### Multi-model vindex
Store features from multiple models in one vindex. Compare representations across architectures.

### Incremental extraction
Add new layers/features to an existing vindex without full rebuild.

## Completed

| Item | Date | Impact |
|------|------|--------|
| Core VectorIndex with mmap | 2026-03 | Foundation |
| Gate KNN (brute-force + BLAS) | 2026-03 | Walk engine |
| Walk FFN (per-feature down/up vectors) | 2026-03 | Sparse inference |
| Binary down_meta format | 2026-03 | 5x compression vs JSONL |
| F16 storage + decode cache | 2026-03 | 2x smaller gate vectors |
| Interleaved layout (gate\|up\|down packed) | 2026-04 | Reduced TLB thrash |
| Q4_0 gate vectors + interleaved | 2026-04 | 7x smaller gates |
| HNSW graph index | 2026-04 | Sub-linear KNN |
| Adaptive residency (pin/evict) | 2026-04 | Memory budget management |
| Patch system (PatchedVindex) | 2026-04 | Editable knowledge |
| MoE expert routing | 2026-04 | Mixtral/DeepSeek support |
| Q4_K/Q6_K attention weights | 2026-04 | Ollama-compatible |
| Q8 attention weights | 2026-04 | Higher precision option |
| Streaming extraction (mmap, per-layer) | 2026-04 | ~2 GB peak RAM |
| Safety doc for mmap_optimized | 2026-04-07 | Clippy compliance |
| VindexPatch::is_empty() | 2026-04-07 | API completeness |
| Q4_K FFN loader + wiring | 2026-04-07 | `interleaved_q4k.bin` end-to-end |
| Quantizer single source of truth | 2026-04-07 | Builder uses larql-compute (ADR-008) |
| Example cleanup (13→11) | 2026-04-07 | Removed Q4_0 attn + Q4_0 interleaved |
| 8 ADRs documented | 2026-04-07 | All major decisions recorded |
| PERFORMANCE.md + format alignment | 2026-04-07 | Fresh benchmarks, verified pipeline |
