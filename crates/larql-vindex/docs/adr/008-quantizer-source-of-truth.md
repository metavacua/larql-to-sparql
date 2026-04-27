# ADR-008: Single Source of Truth for Quantizers

**Status**: Accepted  
**Date**: 2026-04-07  
**Context**: Found critical bug — vindex `build_q4k_weights` reimplemented Q4_K quantization with different formulas than `larql_compute::cpu::ops::q4_common::quantize_q4_k()`. Weights built by vindex would not dequantize correctly through compute shaders.

## Decision

**All quantization functions live in `larql_compute::cpu::ops::q4_common`.** Vindex build tools import from compute, never reimplement.

```rust
// In vindex build_q4k_weights.rs:
use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};

// NOT: fn quantize_q4_k(data: &[f32]) -> Vec<u8> { ... } // local reimplementation
```

## Available Quantizers (all in q4_common.rs)

| Function | Output Format | Block Size |
|----------|--------------|------------|
| `quantize_q4_0` | Q4_0 (18B/32vals) | 32 |
| `quantize_q4_k` | Q4_K (148B/256vals) | 256 |
| `quantize_q4_k_gguf` | GGUF Q4_K (144B/256vals) | 256 |
| `quantize_q4_kf` | Q4_KF (160B/256vals) | 256 |
| `quantize_q6_k` | Q6_K (210B/256vals) | 256 |
| `quantize_to_q8` | Q8_0 (int8 + f32 scales) | 32 |

## The Bug

```rust
// Old vindex builder (WRONG):
let d = global_max / 15.0 / 63.0;      // = global_max / 945
let dmin = -global_min / 15.0 / 15.0;   // = -global_min / 225

// Correct (larql-compute):
let d = global_max_range / 63.0;        // range, not max
let dmin = -global_min / 15.0;          // no double division
```

Different scale factors → different quantized values → wrong dequantization in shaders.

## Consequences

- Vindex build tools are thinner (just I/O, no math)
- Shader correctness tested once in compute, validated by cross-backend tests
- Format changes only need updating in one place
- Build pipeline: vindex examples depend on `larql-compute` (already a dependency)
