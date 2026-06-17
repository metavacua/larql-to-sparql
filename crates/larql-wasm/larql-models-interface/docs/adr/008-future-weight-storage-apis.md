# ADR-008: Future Lazy and Quantized Weight Storage APIs

**Status**: Proposed  
**Date**: 2026-04-26  
**Context**: `ModelWeights` is intentionally simple: retained dense tensors are f32 `ArcArray2`s, with selected packed expert tensors exposed through byte slices backed by retained mmap ranges or small in-memory fallback buffers. This works for current extraction and inference flows, but two roadmap items need broader ownership: lazy safetensors loading and GGUF quantized inference without f32 dequantization.

## Decision

Do not overload `ModelWeights` with lazy or quantized variants. Add explicit
storage types when these features are implemented:

```rust
pub enum LoadedWeights {
    Dense(ModelWeights),
    Lazy(LazyModelWeights),
    Quantized(QuantizedModelWeights),
}
```

`ModelWeights` remains the eager f32 representation used by existing callers,
with `get_packed_bytes()` as the compatibility path for packed expert blobs.
Future APIs should be additive:

```rust
load_model_dir_lazy(path) -> Result<LazyModelWeights, ModelError>
load_gguf_quantized(path) -> Result<QuantizedModelWeights, ModelError>
```

Validated variants should mirror eager loading:

```rust
load_model_dir_lazy_validated(path) -> Result<LazyModelWeights, ModelError>
load_gguf_quantized_validated(path) -> Result<QuantizedModelWeights, ModelError>
```

## Lazy Safetensors Shape

`LazyModelWeights` should keep shard mmaps alive and store tensor descriptors:

```rust
pub struct LazyTensor {
    pub key: String,
    pub dtype: StorageDtype,
    pub shape: Vec<usize>,
    pub shard_id: usize,
    pub byte_range: (usize, usize),
}
```

Accessors can decode one tensor or layer at a time. This avoids converting all
retained tensors into f32 at load time and gives downstream crates control over
when memory is materialized.

## Quantized GGUF Shape

`QuantizedModelWeights` should preserve GGUF tensor bytes and GGML type ids:

```rust
pub struct QuantizedTensor {
    pub key: String,
    pub ggml_type: u32,
    pub shape: Vec<usize>,
    pub byte_range: (usize, usize),
}
```

Compute crates can then call Q4_K/Q6_K row kernels directly instead of receiving
eager f32 arrays. Unsupported GGML types should remain explicit
`UnsupportedDtype` errors unless a downstream kernel exists.

## Consequences

- **Good**: Existing eager f32 loading remains stable and simple.
- **Good**: Lazy and quantized ownership models are explicit in type signatures.
- **Good**: Validated and permissive entry points stay symmetrical.
- **Trade-off**: Downstream crates must handle more than one weight representation.
- **Trade-off**: This requires API design across `larql-models`, `larql-compute`, and callers before implementation.
