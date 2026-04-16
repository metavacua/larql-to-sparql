# ADR-003: Multimodal Config Parsing (text_config Nesting)

**Status**: Accepted  
**Date**: 2026-03  
**Context**: Multimodal models (Gemma 3/4 with vision) nest their text model config under a `text_config` key in config.json. The same architecture (e.g., gemma3_text) needs to work whether loaded from a text-only or multimodal checkpoint.

## Decision

`parse_model_config` always checks for `text_config` first:

```rust
let text_config = config.get("text_config").unwrap_or(config);

// model_type from text_config takes precedence
let model_type = text_config["model_type"]
    .as_str()
    .or_else(|| config["model_type"].as_str())
    .unwrap_or("")
    .to_string();
```

All dimension fields (hidden_size, num_layers, etc.) read from `text_config`. Only `model_type` falls back to the top level.

## Consequences

- **Good**: Same architecture code works for both text-only and multimodal checkpoints.
- **Good**: No special "multimodal wrapper" architectures needed.
- **Good**: Detection logic (`detect_from_json`) is format-agnostic.
- **Trade-off**: Vision-specific config fields (image encoder, patch size) are ignored. Accepted because `larql-models` only handles the text model.
