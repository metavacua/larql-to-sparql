# ADR-007: Explicit Config Validation

**Status**: Accepted  
**Date**: 2026-04-26  
**Context**: `detect_from_json` historically accepted malformed configs and filled missing fields with defaults. That is useful for inspection tools, but invalid values could fail later in inference or extraction with less actionable errors.

## Decision

Keep architecture detection permissive and add an explicit validation step:

```rust
let arch = detect_from_json_validated(&config_json)?;
let weights = load_model_dir_validated(path)?;
```

`ModelArchitecture::validate()` returns `Result<(), Vec<ConfigValidationError>>`.
Each error has a centralized field identifier from `validation.rs` plus a
human-readable message.

Validation checks:
- Core dimensions are positive
- `head_dim` divides `hidden_size`
- KV heads do not exceed Q heads, and Q heads divide evenly by KV heads
- RoPE bases, scaling factors, partial rotary fractions, and scalar multipliers are finite and valid
- Explicit `layer_types` length matches `num_layers`
- KV sharing leaves at least one source layer
- MoE configs include both expert count and experts-per-token, and top-k does not exceed total experts
- Hybrid MoE configs include `moe_intermediate_size`

## Consequences

- **Good**: Inspection and conversion tools can still parse partial configs.
- **Good**: Inference/extraction callers can fail fast with structured diagnostics.
- **Good**: Diagnostic field names are constants, not scattered string literals.
- **Good**: Architecture constructors must tolerate malformed-but-parseable configs and leave rejection to validation.
- **Trade-off**: Callers must choose the permissive or validated API explicitly.
