# ADR-001: Trait-Based Architecture Descriptions

**Status**: Accepted  
**Date**: 2026-03  
**Context**: Need a unified way to describe model architectures (tensor keys, norms, scaling, attention patterns) that works across all LARQL crates without introducing compute dependencies.

## Decision

Define a `ModelArchitecture` trait with ~80 methods, all with default implementations. Each model family implements this trait, overriding only what differs.

```rust
pub trait ModelArchitecture: Send + Sync {
    fn family(&self) -> &str;
    fn config(&self) -> &ModelConfig;
    
    // ~80 methods with defaults covering:
    // tensor keys, norms, attention, FFN, MoE, MLA, scaling, softcapping
}
```

## Consequences

- **Good**: New architectures require minimal code (only override differences).
- **Good**: Adding new trait methods never breaks existing architectures.
- **Good**: Zero compute dependency — `larql-models` has no BLAS, Metal, or math imports.
- **Good**: `Box<dyn ModelArchitecture>` enables runtime architecture dispatch.
- **Trade-off**: Large trait surface (~80 methods). Accepted because most have one-line defaults and are logically grouped.
- **Trade-off**: `ModelConfig` struct grows with each new architecture's fields. Accepted for now — fields are flat and documented.
