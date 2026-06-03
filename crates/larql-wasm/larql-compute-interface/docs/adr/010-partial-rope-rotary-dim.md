# ADR-010: Partial RoPE via rotary_dim Parameter

**Status**: Accepted
**Date**: 2026-04-07
**Context**: Gemma 4 (and potentially other architectures) uses partial rotary position embeddings — only 25% of head dimensions get RoPE rotation on global attention layers. The existing shaders assumed full rotation.

## Decision

Add a `rotary_dim` parameter to both RoPE-related shaders:

### rope_apply (standalone)
- Buffer 3: `rotary_dim` (uint). 0 = use full `dim` (backward compatible).
- Grid dispatch changes from `(dim/2, seq_len)` to `(rotary_dim/2, seq_len)` when partial.
- Inv-freq computed over `rotary_dim`, not `dim`. Dimensions beyond `rotary_dim` pass through.

### fused_attention (inline RoPE)
- Buffer 13: `rotary_dim` (uint). 0 = use full `head_dim` (backward compatible).
- Q and K rotation guarded by `tid < rdim` / `d < rdim` checks.
- Dimensions beyond `rotary_dim` contribute to Q-K dot product but without rotation.

## Design choices

1. **0 means full rotation** — all existing dispatch sites pass 0, preserving behavior without code changes. The shader resolves `rdim = (rotary_dim == 0) ? dim : min(rotary_dim, dim)`.

2. **Per-layer dispatch, not per-model** — `rotary_dim` is a dispatch-time parameter, not baked into the shader. This supports architectures where different layers use different rotation fractions (Gemma 4: sliding layers get full rotation, global layers get 25%).

3. **CPU path already complete** — `apply_rope_partial()` in larql-inference handles partial rotation on CPU. The Metal shaders match this behavior for GPU acceleration.

## Consequences

- All existing dispatch sites pass `rotary_dim=0` — zero behavior change for current models
- Gemma 4 global attention layers can pass `rotary_dim = head_dim * 0.25` for correct partial rotation
- Branch is coherent (all threads in a threadgroup take the same path since rotary_dim is uniform)
- Shader binary is marginally larger (one extra comparison per thread)
