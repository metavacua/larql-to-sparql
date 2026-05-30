# ADR-004: Q4_K / Q6_K Ollama-Compatible Quantization

**Status**: Accepted  
**Date**: 2026-04-06  
**Context**: Ollama uses Q4_K for attention Q/K/O, Q6_K for V and FFN down. Need matching formats.

## Decision

Implement Q4_K and Q6_K Metal shaders and CPU reference implementations that read the same superblock format as Ollama/llama.cpp (GGUF specification).

## Origin

- **Format specification**: GGUF standard (llama.cpp, MIT license). Q4_K = 148 bytes per 256 values (our layout: 2B d + 2B dmin + 12B scales + 4B mins + 128B nibbles). GGUF canonical is 144 bytes (scales+mins packed in 12 bytes).
- **Metal shaders** (`q4k_matvec`, `q6k_matvec`): Original LARQL implementation matching the GGUF dequantization formula.
- **CPU reference** (`q4k_matvec.rs`, `q6k_matvec.rs`): Original LARQL implementation for cross-backend testing. Scalar code mirroring the Metal shader logic.
- **Quantizers** (`quantize_q4_k`, `quantize_q6_k`, `quantize_q4_k_gguf`): Original LARQL implementation. `quantize_q4_k_gguf` produces exact GGUF 144-byte blocks with packed 12-byte scales+mins.

## Q4_K Block Layouts

| Field | LARQL (148B) | GGUF (144B) |
|-------|-------------|-------------|
| d, dmin | 2+2 bytes (f16) | 2+2 bytes (half) |
| Scales | 12 bytes (8×6-bit) | 12 bytes (8×6-bit scale + 8×6-bit min packed) |
| Mins | 4 bytes (8×4-bit) | (packed into scales) |
| Nibbles | 128 bytes | 128 bytes |

## Consequences

- Vindex stores Q4_K attention weights in our 148-byte format
- GGUF 144-byte format available via `quantize_q4_k_gguf` for compatibility testing
- Cross-backend tests verify Metal vs CPU with tolerance (Q4_K: 0.5, Q6_K: 0.3)
- Q4_K data is 1.73x smaller than Q8 per layer (7.6MB vs 13.1MB)
