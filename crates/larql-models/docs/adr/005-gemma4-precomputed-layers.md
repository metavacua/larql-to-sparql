# ADR-005: Gemma 4 Precomputed Layer Tables

**Status**: Accepted  
**Date**: 2026-04  
**Context**: Gemma 4 has per-layer attention geometry: different head_dim, KV heads, rotary fraction, and RoPE base for sliding vs global layers. KV sharing adds another per-layer lookup. These are queried on every token for every layer.

## Decision

Precompute lookup vectors in `Gemma4Arch::from_config` rather than computing per call:

```rust
pub struct Gemma4Arch {
    config: ModelConfig,
    global_layers: Vec<bool>,          // from layer_types or sliding_window_pattern
    kv_sources: Vec<Option<usize>>,    // KV sharing: Some(source) or None
}
```

Trait methods use simple vector indexing:

```rust
fn is_global_layer(&self, layer: usize) -> bool {
    self.global_layers.get(layer).copied().unwrap_or(false)
}

fn kv_shared_source_layer(&self, layer: usize) -> Option<usize> {
    self.kv_sources.get(layer).copied().flatten()
}
```

## Source Priority for Layer Types

1. Explicit `layer_types` array in config.json (Gemma 4 provides this)
2. `sliding_window_pattern` field (every Nth layer is full)
3. Default pattern of 6

## KV Sharing Logic

For `num_kv_shared_layers = 20` with 35 layers:
- Layers 0-14: non-shared, compute own KV
- Layers 15-34: shared, reuse from last non-shared layer of same type
  - Shared sliding → last non-shared sliding layer
  - Shared global → last non-shared global layer

## Consequences

- **Good**: O(1) per-layer queries — no conditionals, no pattern arithmetic.
- **Good**: KV sharing sources computed once, correctly handling mixed sliding/global.
- **Good**: Out-of-bounds access returns safe default (false / None).
- **Trade-off**: O(num_layers) allocation at construction. Negligible — 35 bools + 35 Options.
