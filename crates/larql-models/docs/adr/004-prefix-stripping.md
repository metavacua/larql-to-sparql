# ADR-004: Key Prefix Stripping at Load Time

**Status**: Accepted  
**Date**: 2026-03  
**Context**: HuggingFace safetensors files store tensor keys with varying prefixes depending on the model wrapper. The same weight might be stored as `model.layers.0.self_attn.q_proj.weight`, `language_model.model.layers.0.self_attn.q_proj.weight`, or just `layers.0.self_attn.q_proj.weight`.

## Decision

Each architecture declares which prefixes to strip via `key_prefixes_to_strip()`:

```rust
// Default (most models):
fn key_prefixes_to_strip(&self) -> &[&str] {
    &["language_model.model.", "model."]
}

// Gemma 4 (deeper multimodal nesting):
fn key_prefixes_to_strip(&self) -> &[&str] {
    &["model.language_model.model.", "model.language_model.", 
      "language_model.model.", "model."]
}
```

The loader tries each prefix in order; first match wins. After stripping, all architectures use the same canonical key format: `layers.{N}.self_attn.q_proj.weight`.

## Consequences

- **Good**: Architecture-specific key patterns centralized in one method.
- **Good**: Loader is architecture-agnostic — just calls `key_prefixes_to_strip()`.
- **Good**: Order matters: longer prefixes tried first, preventing partial matches.
- **Trade-off**: If a new wrapper nesting is encountered, must add a prefix. Low risk — prefixes are model-family-level, not per-checkpoint.
