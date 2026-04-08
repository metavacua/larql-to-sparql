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

| Engine | ms/tok | tok/s | Notes |
|--------|--------|-------|-------|
| LARQL Q4_K decode (21L, KV) | 17.5ms | 57 | 3 encoders/layer (merged) |
| LARQL Q8 decode (21L, KV) | 24.5ms | 41 | |
| LARQL Q4_K decode (34L, KV) | 28.0ms | 36 | |
| Ollama (34L) | 10.2ms | 98 | |
| **Projected cached (8L)** | **~5ms** | **~200** | Cache L0-12, compute 8 layers |

## Layer Graph Strategies

| Strategy | What it does | When used |
|----------|-------------|-----------|
| CachedLayerGraph | Returns pre-computed residual | L0-12 (template-fixed) |
| DenseLayerGraph | Matmul attention + pluggable FFN | Baseline/fallback |
| WalkLayerGraph | Dense attention + sparse WalkFfn | CPU walk path |
| PipelinedLayerGraph | CPU attention + Metal Q4 FFN | GPU acceleration |
| PerLayerGraph | Per-layer strategy selection | Adaptive routing |

## Component Breakdown (CPU BLAS, seq=6, Gemma 3 4B, `bench_components`)

| Component | µs/layer | % | Notes |
|-----------|---------|---|-------|
| FFN gate+up (2× BLAS) | 6,008 | 44.5% | Dominant cost |
| FFN down (BLAS) | 3,475 | 25.7% | |
| QKV projection (3× BLAS) | 2,896 | 21.4% | |
| O projection (BLAS) | 789 | 5.8% | |
| Attention (scores+softmax+V) | 143 | 1.1% | Small at seq=6 |
| GEGLU SiLU | 105 | 0.8% | Element-wise |
| RoPE | 56 | 0.4% | |
| RMSNorm (×2) | 30 | 0.2% | |
| Residual add (×2) | 3 | 0.0% | |
| **Layer total** | **13,504** | | |
| **34-layer model** | **513ms** | | **2 tok/s CPU** |

97% of time is BLAS matmul. GPU Q4_K pipeline replaces these: 513ms → 17.5ms (29x speedup).

### Norm comparison

| Norm | µs (seq=6, hidden=2560) | vs RMSNorm |
|------|------------------------|-----------|
| RMSNorm | 14.9µs | baseline |
| LayerNorm | 28.4µs | 1.91x |

### RoPE comparison

| Variant | µs (8 heads) | Notes |
|---------|-------------|-------|
| Full (hd=256) | 56.0µs | Standard |
| Partial 25% (hd=512) | 16.8µs | Gemma 4 global, 3.3x faster |

## Activation Function Support

| Model | Activation | FFN Type | GPU Path | CPU Path |
|-------|-----------|----------|----------|----------|
| Llama 2/3 | SiLU | Gated | geglu_silu | ✅ |
| Gemma 2/3/4 | GELU-tanh | Gated | geglu_gelu_tanh | ✅ |
| Mistral | SiLU | Gated | geglu_silu | ✅ |
| Qwen 2/3 | SiLU | Gated | geglu_silu | ✅ |
| StarCoder2 | GELU-tanh | Standard | gelu_tanh (standalone) | ✅ |
| GPT-2 | GELU | Standard | gelu_tanh (standalone) | ✅ |
| Granite | SiLU | Gated | geglu_silu | ✅ |

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

## Cross-Crate Performance Comparison

All measurements on M3 Max, Gemma 3 4B (34 layers, hidden=2560).

| Path | Component | Crate | Time | Notes |
|------|-----------|-------|------|-------|
| **CPU forward** | Matmul (BLAS) | inference | 13.5ms/layer | 97% of layer time |
| **CPU forward** | Attention | inference | 0.14ms/layer | 1.1% — negligible |
| **CPU forward** | RMSNorm + GEGLU + RoPE | inference | 0.19ms/layer | 1.4% — element-wise |
| **GPU decode** | Q4_K QKV (fused) | compute | 0.044ms/layer | 6.3x faster than Ollama's layer |
| **GPU decode** | Q4 FFN (gate+up+geglu+down) | compute | 0.38ms/layer | 36% of GPU time |
| **GPU decode** | KV cache attend | compute | 0.31ms/layer | 29% of GPU time |
| **GPU decode** | Norms | compute | 0.16ms/layer | Actual GPU compute |
| **Vindex** | Gate KNN (f32 BLAS) | vindex | 3.0ms/layer | Production dims |
| **Vindex** | Gate KNN (Q4 CPU) | vindex | 1.0ms/layer | 3x faster |
| **Vindex** | Gate KNN (Q4 Metal) | vindex | 0.5ms/layer | 6x faster |
| **Vindex** | Walk (14 layers) | vindex | 14ms | Mmap zero-copy |
| **Ollama** | Full layer | external | 0.30ms/layer | Metal GPU, merged dispatches |
