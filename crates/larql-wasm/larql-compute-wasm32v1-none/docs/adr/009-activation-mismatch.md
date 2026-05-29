# ADR-009: Activation Function Mismatch (geglu_silu vs gelu_tanh)

**Status**: Open — needs fix  
**Date**: 2026-04-07  
**Context**: Real vindex benchmark with Gemma3 4B produces "peregr" instead of "Paris" through the GPU honest path.

## Root Cause

The `geglu_silu` Metal shader uses SiLU activation: `out = silu(gate) × up = (gate × sigmoid(gate)) × up`.

Gemma3 uses GELU-tanh activation: `out = gelu_tanh(gate) × up = (gate × 0.5 × (1 + tanh(√(2/π) × (gate + 0.044715 × gate³)))) × up`.

The `full_pipeline_q4` and `decode_token` paths hardcode `geglu_silu` — they don't check `arch.activation()`.

## Evidence

```
Dense (CPU, correct activation):     "Paris" (80.47%)   546ms
Honest (GPU, SiLU instead of GELU):  "peregr" (20.00%)  329ms
Prefill logits (CPU path):           "Paris" (88.41%)    4.8ms
```

The CPU path in `larql-inference/src/forward.rs` correctly dispatches based on `arch.activation()`. The GPU path in `larql-compute/src/metal/decode.rs` and `full_pipeline.rs` always uses `geglu_silu`.

## Fixes Applied (2026-04-07)

1. ✅ Added `geglu_gelu_tanh` Metal shader alongside `geglu_silu`
2. ✅ Added `use_gelu_tanh: bool` field to `FullPipelineLayer`
3. ✅ `decode_token` selects activation per-layer from `layers[l].use_gelu_tanh`
4. ✅ `full_pipeline_q4` and `prefill_q4` select activation from first layer
5. ✅ Inference sets `use_gelu_tanh: arch.activation() == Activation::GeluTanh`

## Remaining Issue

After the activation AND post-norm fixes, honest path still produces "peregr". Confirmed fixes applied:
- ✅ `use_gelu_tanh=true` flag set correctly
- ✅ Pre-FFN norm uses `pre_feedforward_layernorm` (not post_attn_norm)
- ✅ Post-FFN norm applied before residual add

The remaining issue is likely in the **prefill.rs attention residual path** — the interplay between post-attention norm, residual multiplier, and the fused attention shader's handling of Gemma3's architecture. The `full_pipeline.rs` handles all these correctly (produces correct output for seq=1). The `prefill.rs` was a simplified port that doesn't replicate all edge cases.

The **CPU path produces correct output** ("Paris" at 224ms). The **GPU full_pipeline** also produces "Paris" for seq=1 through decode_token.

## Recommendation

1. Short term: for Gemma3, use CPU path for prefill (correct), GPU decode_token for seq=1 (correct with activation fix)
2. Long term: refactor prefill.rs to exactly mirror full_pipeline.rs's post-norm handling

## Impact

This is the reason the real model benchmark shows wrong output on the GPU path. The synthetic benchmarks in `compare_ollama` don't hit this because they use random weights where the activation function doesn't affect correctness of timing measurements.

## Models Affected

| Model | Activation | GPU Path |
|-------|-----------|----------|
| Gemma 2/3 | GELU-tanh | ❌ Wrong |
| Llama 2/3 | SiLU | ✅ Correct |
| Mistral | SiLU | ✅ Correct |
| Qwen2 | SiLU | ✅ Correct |
| Phi | GELU-approx | ❌ Wrong |
| GPT-2 | GELU | ❌ Wrong |
