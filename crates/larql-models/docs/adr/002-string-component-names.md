# ADR-002: String Component Names (No Domain-Specific Enums)

**Status**: Accepted  
**Date**: 2026-03  
**Context**: Vector extraction and vindex operations reference model components (FFN down, attention OV, embeddings, etc.). Need a naming scheme.

## Decision

Use string constants (`&str`) for component names, not an enum:

```rust
pub const COMPONENT_FFN_DOWN: &str = "ffn_down";
pub const COMPONENT_FFN_GATE: &str = "ffn_gate";
pub const COMPONENT_FFN_UP: &str = "ffn_up";
pub const COMPONENT_ATTN_OV: &str = "attn_ov";
pub const COMPONENT_ATTN_QK: &str = "attn_qk";
pub const COMPONENT_EMBEDDINGS: &str = "embeddings";
```

## Rationale

The engine must be generic with no domain-specific enums or defaults baked into code. String constants:
- Allow new components without modifying a central enum
- Work naturally with file paths, JSON keys, and CLI arguments
- Avoid forcing downstream crates to match on variants they don't care about

## Consequences

- **Good**: Adding new component types is non-breaking.
- **Good**: No enum exhaustiveness issues in downstream match statements.
- **Trade-off**: No compile-time checking of component names. Mitigated by using constants rather than raw string literals.
