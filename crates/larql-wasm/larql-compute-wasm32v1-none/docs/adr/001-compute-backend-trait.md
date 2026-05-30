# ADR-001: ComputeBackend Trait as Single Abstraction

**Status**: Accepted  
**Date**: 2026-04-03  
**Context**: Need to support CPU, Metal GPU, and future CUDA without callers knowing the implementation.

## Decision

All compute goes through `ComputeBackend` trait. Callers never touch Metal/CUDA APIs directly.

```rust
pub trait ComputeBackend: Send + Sync {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32>;
    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32>;
    fn q4_matvec(&self, ...) -> Option<Vec<f32>>;
    fn q4k_matvec(&self, ...) -> Option<Vec<f32>>;
    fn full_pipeline_q4(&self, ...) -> Option<Vec<f32>>;
    fn decode_token(&self, ...) -> Option<Vec<f32>>;
    // ... 15 methods total
}
```

## Consequences

- **Good**: Inference crate is backend-agnostic. Can swap CPU/Metal with zero code changes.
- **Good**: Optional methods return `Option<Vec<f32>>` — caller falls back gracefully.
- **Good**: `default_backend()` auto-detects best available hardware.
- **Trade-off**: Trait methods have many parameters (up to 13). Accepted because FullPipelineLayer struct bundles per-layer data.
- **Trade-off**: Can't return references (must copy GPU → CPU). Accepted because decode produces one hidden vector per call.
