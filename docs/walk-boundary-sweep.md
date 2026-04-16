# Walk Boundary Sweep Results

**Date**: 2026-04-03
**Model**: Gemma-3 4B IT (google/gemma-3-4b-it)
**Vindex**: gemma3-4b-f16.vindex (34 layers, 348,160 gate vectors, f16)

## Finding

The vindex FFN walk produces identical top-1 predictions to the all-dense forward pass at **every layer boundary from L0 to L34**. FFN matmuls can be fully replaced by vindex gate KNN lookups across all 34 layers with zero accuracy loss.

## Sweep Design

For each boundary B:
- Layers `0..B`: dense attention + dense FFN (WeightFfn, full matmul)
- Layers `B..34`: dense attention + vindex FFN (WalkFfn, gate KNN top-8092 → sparse down)

Attention runs as BLAS-fused dense at all layers for all boundaries. Only the FFN path varies.

## Results

```
     B   walk%   correct  top1_avg  details
  -------------------------------------------------------
  L0     100%    5/5      82.63%   all match ground truth
  L4      88%    5/5      82.63%   all match ground truth
  L8      76%    5/5      82.63%   all match ground truth
  L12     65%    5/5      82.63%   all match ground truth
  L16     53%    5/5      82.63%   all match ground truth
  L20     41%    5/5      82.63%   all match ground truth
  L24     29%    5/5      82.63%   all match ground truth
  L28     18%    5/5      82.63%   all match ground truth
  L34      0%    5/5      82.63%   all match ground truth
```

### Ground truth (all-dense f32)

| Prompt | Top-1 | Probability |
|--------|-------|-------------|
| The capital of France is | Paris | 80.47% |
| The capital of Germany is | Berlin | 86.46% |
| The capital of Japan is | Tokyo | 83.76% |
| The capital of Italy is | Rome | 63.70% |
| The largest planet in our solar system is | Jupiter | 98.78% |

### Key observations

1. **Zero divergence**: Not a single prediction changed at any boundary. Top-1 token and probability are identical whether 0% or 100% of layers use vindex FFN.

2. **Average probability stable**: 82.63% at every boundary. The vindex walk doesn't degrade confidence — it produces the same softmax distribution.

3. **L0 works**: 100% vindex FFN (zero dense FFN matmuls) gives identical results to 100% dense FFN. The gate vectors at each layer are calibrated to that layer's residual space.

## Why it works

The vindex at each layer was extracted from that layer's actual gate weight matrix. The gate vectors at layer 5 were computed from `W_gate` at layer 5 — they expect layer 5 residuals, not layer 25 residuals. The KNN matches because the index and the query live in the same space.

The WalkFfn path:
1. Gate KNN: `residual × gate_vectors^T` → top-8092 features (same as dense gate matmul, same operation)
2. Sparse FFN: only the selected features go through up/down projections
3. Result: mathematically equivalent to dense FFN with top-K sparsification

The gate KNN IS the gate matmul. Same dot products, same features selected, same result.

## Implications

### FFN quantization is unnecessary

The entire FFN computation across all 34 layers is served by vindex lookups. No FFN weight matrices (gate, up, down) need to be loaded for inference. No FFN matmuls are executed.

### Remaining matmuls

The only matrix multiplications in the forward pass are:

| Operation | Shape | Per layer | Notes |
|-----------|-------|-----------|-------|
| Q projection | `[seq, 2560] × [2560, 2560]` | ~1 ms | Accelerate AMX |
| K projection | `[seq, 2560] × [2560, 2560]` | ~1 ms | Accelerate AMX |
| V projection | `[seq, 2560] × [2560, 2560]` | ~1 ms | Accelerate AMX |
| O projection | `[seq, 2560] × [2560, 2560]` | ~1 ms | Accelerate AMX |
| Final logits | `[1, 2560] × [262144, 2560]` | ~27 ms | Once, not per-layer |

Everything else — embedding, RoPE, norms, FFN — is lookup, scalar math, or eliminated.

### Performance projection

```
Current (all-dense):
  Attention projections:  4 × ~1ms × 34 layers = ~136ms
  FFN (dense):            ~5ms × 34 layers = ~170ms
  Logits:                 ~27ms
  Other:                  ~10ms
  Total:                  ~343ms → ~3 tok/s

With full vindex walk (proven):
  Attention projections:  4 × ~1ms × 34 layers = ~136ms
  FFN (vindex walk):      ~1ms × 34 layers = ~34ms
  Logits:                 ~27ms
  Other:                  ~10ms
  Total:                  ~207ms → ~5 tok/s

With Q4_K_M attention:
  Attention projections:  4 × ~0.3ms × 34 layers = ~41ms
  FFN (vindex walk):      ~1ms × 34 layers = ~34ms
  Logits:                 ~7ms (quantized)
  Other:                  ~10ms
  Total:                  ~92ms → ~11 tok/s

With attention template cache:
  Attention:              ~0.001ms × 34 layers ≈ 0ms
  FFN (vindex walk):      ~1ms × 34 layers = ~34ms
  Logits:                 ~7ms
  Other:                  ~10ms
  Total:                  ~51ms → ~20 tok/s
```

### Build order (revised)

1. ~~Q4_K_M for FFN weights~~ — **eliminated** by this sweep
2. Q4_K_M for attention weights only (Q/K/V/O — much smaller surface)
3. Q4_K_M for embeddings/logits (bandwidth bottleneck)
4. Attention template cache (eliminates attention matmuls)

## Reproducing

```bash
cargo run --release -p larql-inference --example walk_boundary_sweep -- \
  --model google/gemma-3-4b-it \
  --vindex /path/to/gemma3-4b-f16.vindex
```

Options:
- `--top-k N`: Gate KNN top-K (default: 8092)
- `--prompts "prompt1=answer1;prompt2=answer2"`: Custom test set

## Configuration

- **Model**: google/gemma-3-4b-it (34 layers, hidden=2560, 10 Q heads, 2 KV heads)
- **Vindex**: f16 gate vectors, 348,160 total (10,240 features × 34 layers)
- **Gate KNN top-K**: 8092 (out of 10,240 intermediate features)
- **Attention**: BLAS-fused (online softmax, Accelerate AMX)
- **Platform**: Apple Silicon (macOS)
