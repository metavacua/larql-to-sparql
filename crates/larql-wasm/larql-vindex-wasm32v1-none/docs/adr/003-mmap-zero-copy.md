# ADR-003: Mmap Zero-Copy Architecture

**Status**: Accepted  
**Date**: 2026-03  
**Context**: Gate vectors for large models (70B) are 5+ GB. Cannot heap-allocate.

## Decision

Memory-map all weight files. Gate vectors, down features, attention weights, and FFN interleaved data are accessed via `memmap2::Mmap` with zero-copy slicing.

## Implementation

```rust
// Gate vectors: return &[f32] view directly from mmap — no allocation
pub fn gate_vectors_f32(&self, layer: usize) -> Option<&[f32]>

// Metal GPU: newBufferWithBytesNoCopy for mmap'd data
// Unified memory: GPU reads directly from the mmap'd page
```

### Access Patterns

| Data | Access | Hint |
|------|--------|------|
| Gate vectors | Sequential per query | `MADV_SEQUENTIAL` |
| Down features | Random per selected feature | `MADV_WILLNEED` on next layer |
| Attention weights | Sequential per layer | `MADV_SEQUENTIAL` |
| Interleaved FFN | Sequential per layer | Prefetch next layer |

### Adaptive Residency

```rust
pub enum LayerState {
    Cold,       // Not loaded, page on demand
    MmapQ4,     // Paged from Q4 mmap (~0.53ms cold penalty)
    Pinned,     // Pre-loaded to heap for zero page faults
}
```

ResidencyManager pins hot layers based on access frequency and memory budget.

## Origin

Original LARQL design. Mmap is standard for large file access, but the combination with:
- OS advisory hints (madvise)
- Metal zero-copy GPU buffers (newBufferWithBytesNoCopy)  
- Adaptive residency (frequency-based pin/evict)
is specific to LARQL.

## Consequences

- **Good**: 70B model browseable in 4.9GB resident
- **Good**: Zero-copy Metal GPU access on Apple Silicon unified memory
- **Good**: Smooth performance gradient (pinned > mmap > cold)
- **Trade-off**: Requires page-aligned data for zero-copy GPU access
- **Trade-off**: First access to cold layers incurs page fault (~0.5ms)
