# ADR-006: Norm Weight Offset (Gemma 2/3 vs Gemma 4)

**Status**: Accepted  
**Date**: 2026-04  
**Context**: Different Gemma generations store norm weights differently. The forward pass computes `output = (weight + offset) * rms_norm(input)`. The offset must match how weights were saved.

## Decision

Two separate trait methods control norm weight offsets:

```rust
/// Weight offset for layer norms (input_layernorm, post_attention_layernorm, etc.)
fn norm_weight_offset(&self) -> f32 { 0.0 }  // default

/// Weight offset for QK norms (q_norm, k_norm)
fn qk_norm_weight_offset(&self) -> f32 { 0.0 }  // default
```

### Per-architecture values

| Architecture | `norm_weight_offset` | `qk_norm_weight_offset` | Explanation |
|-------------|---------------------|------------------------|-------------|
| Gemma 2 | 1.0 | 1.0 | HF saves `weight - 1`; runtime adds 1 back |
| Gemma 3 | 1.0 | 1.0 | Same convention as Gemma 2 |
| Gemma 4 | 0.0 | 0.0 | `Gemma4RMSNorm` applies weight directly |
| Llama | 0.0 | N/A (no QK norm) | Standard RMSNorm |
| Others | 0.0 | 0.0 | Standard convention |

## Rationale

Gemma 2/3 HuggingFace checkpoints store norm weights as `learned_delta` where runtime weight = 1.0 + learned_delta. This is a HuggingFace saving convention, not a mathematical requirement. Gemma 4 changed to storing the full weight directly.

Separating layer norm offset from QK norm offset allows architectures where these differ (though currently they match within each model family).

## Consequences

- **Good**: Forward pass uses `(weight + arch.norm_weight_offset())` — no per-architecture branching.
- **Good**: Separate QK norm offset catches the Gemma 3 bug (was missing before ADR).
- **Trade-off**: Two methods instead of one. Justified by the Gemma 3 omission that this ADR fixes.
