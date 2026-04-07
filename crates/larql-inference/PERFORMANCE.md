# Performance — larql-inference

Machine: M3 Max, macOS. Gemma 3 4B (34 layers, hidden=2560, vocab=262K).

## Production Benchmark: "The capital of France is"

Real vindex (`output/gemma3-4b-v2.vindex`), 6-token prompt.

| Strategy | Output | Time | tok/s | Notes |
|----------|--------|------|-------|-------|
| Dense (baseline) | Paris (80.47%) | 552ms | 1.8 | CPU BLAS, all 34 layers |
| Full pipe (CPU) | Paris | 224ms | 4.5 | Cached L0-12 + WalkFfn L13-33 |
| **Honest (production)** | **Paris (88.41%)** | **203ms** | **4.9** | **Cached L0-12, CPU L13-33, GPU logits** |
| Split cached | Paris (88.41%) | 3ms | 311 | Pre-computed residuals (one-time build) |
| Prefill logits | Paris (88.41%) | 4.0ms | — | Logits only (from prefilled hidden state) |
| Ollama | Paris | 144ms + 8.5ms/tok | 117 | Full GPU pipeline |

## Honest Path Breakdown

```
predict_honest("The capital of France is"):
  Phase 0 (L0-12): CachedLayerGraph          ~5ms  (template-fixed, 0.999 cosine)
  Phase 1 (L13-33): CPU attention + WalkFfn  ~195ms (GELU-tanh activation, post-norms)
  Phase 2: GPU logits KNN                     ~4ms  (vindex lm_head Q4 via Metal)
  Total:                                     ~203ms = 4.9 tok/s
```

## GPU Decode Path (synthetic, seq=1)

From `compare_ollama` benchmark (larql-compute):

| Engine | ms/tok | tok/s |
|--------|--------|-------|
| LARQL Q4_K decode (21L, KV) | 16.9ms | 59 |
| LARQL Q8 decode (21L, KV) | 24.0ms | 42 |
| Ollama (34L) | 9.7ms | 103 |

## Layer Graph Strategies

| Strategy | What it does | When used |
|----------|-------------|-----------|
| CachedLayerGraph | Returns pre-computed residual | L0-12 (template-fixed) |
| DenseLayerGraph | Matmul attention + pluggable FFN | Baseline/fallback |
| WalkLayerGraph | Dense attention + sparse WalkFfn | CPU walk path |
| PipelinedLayerGraph | CPU attention + Metal Q4 FFN | GPU acceleration |
| PerLayerGraph | Per-layer strategy selection | Adaptive routing |

## Activation Function Support

| Model | Activation | GPU Path | CPU Path |
|-------|-----------|----------|----------|
| Llama 2/3 | SiLU | ✅ geglu_silu | ✅ |
| Gemma 2/3 | GELU-tanh | ✅ geglu_gelu_tanh | ✅ |
| Mistral | SiLU | ✅ | ✅ |
| Qwen2 | SiLU | ✅ | ✅ |
| GPT-2 | GELU | ✅ geglu_gelu_tanh | ✅ |

## Post-Norm Architecture

Gemma3 uses post-norms (norm after attention/FFN, not before):
- CPU path: fully correct (tested, "Paris" output)
- GPU decode_token: correct (activation + post-norm handled)
- GPU prefill_q4: **not yet correct** for post-norm models → falls to CPU
- See larql-compute ADR-009

## Connection to Compute and Vindex

```
larql-inference orchestrates:
  predict_honest()
    → CachedLayerGraph (pre-computed residuals from vindex)
    → FullPipelineLayer (weights from vindex, format tags from vindex)
    → ComputeBackend.decode_token() (GPU Metal kernels)
    → finalize_logits() (vindex lm_head KNN via backend.q4_matvec)
```

Quantization format flows: vindex Q4_K bytes → FullPipelineLayer.format → compute shader dispatch.
