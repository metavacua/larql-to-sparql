# ADR-007: Interleaved Weight Layout

**Status**: Accepted  
**Date**: 2026-04  
**Context**: FFN inference reads gate, up, and down weights for the same layer. Three separate mmap regions cause TLB thrashing.

## Decision

Pack gate|up|down weights contiguously per layer in a single file:

```
interleaved.bin:
  Layer 0: [gate_vectors][up_vectors][down_vectors]
  Layer 1: [gate_vectors][up_vectors][down_vectors]
  ...

interleaved_q4.bin: same layout, Q4_0 quantized
interleaved_q4k.bin: same layout, Q4_K (gate/up) + Q6_K (down)
```

## Why

When processing layer L, the GPU reads:
1. gate[L] for activation selection
2. up[L] for up-projection
3. down[L] for down-projection

With separate files, these are at different virtual addresses → different TLB entries → page table thrashing at large model sizes.

Interleaved: all three matrices for layer L are adjacent → same TLB page → one page fault brings all three.

## Origin

Original LARQL design. Standard technique in GPU inference (llama.cpp also uses contiguous per-layer layout in GGUF).

## Consequences

- Single mmap for all FFN weights per layer
- `prefetch_interleaved_layer(L+1)` while processing L → OS pre-pages next layer
- Build pipeline: `build_interleaved` packs from separate files
- 3 format variants: f32, Q4_0, Q4_K/Q6_K
