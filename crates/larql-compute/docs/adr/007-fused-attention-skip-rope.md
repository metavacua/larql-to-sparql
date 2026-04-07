# ADR-007: Fused Attention Shader with skip_rope Flag

**Status**: Accepted  
**Date**: 2026-04-06  
**Context**: The fused_attention shader applies RoPE internally. For prefill with KV cache, we need to apply RoPE separately to K (for cache population) then run attention without re-applying RoPE.

## Decision

Add `skip_rope` parameter (buffer 12, uint) to fused_attention shader:
- `skip_rope == 0`: Apply RoPE to Q and K in-shader (default, used by decode)
- `skip_rope == 1`: Skip RoPE on both Q and K (caller pre-applied)

## Origin

Original LARQL design. The flag enables the prefill pipeline to:
1. Apply RoPE to K separately via `rope_apply` shader
2. Store post-RoPE K in KV cache
3. Run fused_attention with skip_rope=1 (K already has RoPE)

## Consequences

- 3-line shader change (wrap RoPE blocks in `if (skip_rope == 0)`)
- All existing dispatch sites pass `skip_rope=0` (no behavior change)
- Prefill pipeline uses `skip_rope=1` when pre-applying RoPE for KV cache
- Shader binary is slightly larger but branch is coherent (all threads take same path)
