# KV Cache Benchmark Spec v3

## Standard KV vs TurboQuant vs Markov RS vs Hybrid RS + Cracked Attention vs RS Graph Walk

### Purpose

Five-way comparison showing progression from compression to elimination to replacement of the forward pass:

1. **Standard KV** -- baseline
2. **TurboQuant** -- best possible compression (Shannon ceiling)
3. **Markov RS** -- eliminate the cache, keep the forward pass
4. **Hybrid RS + Cracked Attention** -- eliminate MOST of the cache without solving attention fully
5. **RS Graph Walk** -- eliminate the forward pass, keep the graph

Strategy 4 is the near-term demonstrable win. 95.5% of attention heads are cacheable. FFN already solved. Most KV cache disappears now, with remaining ~4.5% dynamic heads using tiny residual KV cache.

---

### The Five Contenders

#### 1. Standard KV Cache (Baseline)

- FP16 keys + values, per-layer, per-head, per-token
- Memory: `seq_len x layers x 2 x kv_heads x head_dim x 2 bytes`
- Gemma 3-4B at 4K: ~544 MB, at 370K: ~56 GB
- Full forward pass (34 layers, attention + FFN)

#### 2. TurboQuant (Google, ICLR 2026)

- Compresses KV cache to 3-4 bits per coordinate
- Algorithm: WHT rotation -> Lloyd-Max scalar quantization
- Compression: 4-6x at Shannon limit
- Still grows O(context_length), still full forward pass
- Community: MSE-only beats MSE+QJL after softmax

#### 3. Markov Residual Stream (LARQL)

- Eliminates KV cache entirely
- Bounded window of residuals + cold-tier token IDs (4 bytes each)
- Proven Markov property, KL = 0.0 (bit-perfect)
- Does NOT grow O(context_length)
- 135-1,012x compression

#### 4. Hybrid RS + Cracked Attention (LARQL near-term)

- Key insight: don't need to SOLVE attention to ELIMINATE most of KV cache
- 95.5% of attention heads cacheable (cosine 0.942+)
- FFN already solved (vindex walk, 34 layers validated, zero accuracy loss)
- Cache static heads, tiny KV only for ~4.5% dynamic heads
- Memory at 4K: ~20-37 MB (vs 544 MB standard) = 15-27x reduction
- Memory at 370K: ~150-300 MB = 180-370x compression
- No new research needed -- engineering assembly task

#### 5. RS Graph Walk (LARQL endgame)

- Eliminates forward pass itself
- FFN graph (348K features), Attention graph (11K edges), Residual graph
- Vindex walk: 34 layers validated, zero accuracy loss
- 41x speedup on FFN, 10.7 GB weights dropped

---

### What We Measure

Seven axes of comparison:

1. **Memory Scaling** -- how cache size grows with context length
2. **Multi-Turn Wall Clock** -- latency across conversation turns
3. **Accuracy/Fidelity** -- token-level match, KL divergence, downstream task accuracy
4. **First Token Latency** -- time to first token on cold start and warm start
5. **Compute Backend** -- CPU, Metal, CUDA; which backends each strategy supports
6. **Cold Storage/Distributed** -- cost to persist and restore a conversation
7. **Computation Eliminated** -- which matmuls are removed entirely

---

### Implementation Plan

#### Phase 1: Synthetic Benchmark

- Generate synthetic KV tensors (random + structured)
- Measure compress/decompress throughput for Standard KV and TurboQuant
- Measure Markov RS window operations
- Baseline memory and latency numbers

#### Phase 2: Real Model Integration

- Wire up Gemma 3-4B forward pass
- Instrument per-layer KV cache usage
- Collect ground truth attention patterns for head classification
- Validate Markov RS bit-perfect property on real sequences

#### Phase 3: Shader Benchmarks

- Metal compute shaders for TurboQuant WHT + quantize
- Metal compute shaders for Markov RS window update
- Metal compute shaders for vindex walk (already proven)
- Compare CPU vs GPU throughput per strategy

#### Phase 4: Multi-Turn Simulation

- 20-turn conversation benchmark
- Measure memory growth per turn for each strategy
- Measure wall clock per turn for each strategy
- Identify crossover points

#### Phase 5: Graph Walk Experiments

- End-to-end vindex walk inference (no forward pass)
- Accuracy validation against full model
- Measure graph construction cost (one-time)
- Measure per-query walk latency

#### Phase 6: Comparative Table

Final results table:

```
                        Standard KV     TurboQuant 4-bit    Markov RS           Hybrid RS+CA             RS Graph Walk
Memory @ 4K tokens      544 MB          ~90-136 MB          10 KB + window      ~20-37 MB                10 KB + graph*
Memory @ 370K tokens    56 GB           ~9-14 GB            55 MB               ~150-300 MB              ~1.5 GB*
Cold storage / conv     978 MB          ~160-240 MB         10.2 KB             10.2 KB + template ID    10.2 KB
Forward pass?           yes (34L)       yes (34L)           yes (window)        partial (~1-2L attn)     NO
FFN matmuls?            34L             34L                 34L                 ZERO (vindex)            ZERO (vindex)
Attn matmuls?           34L             34L                 window              ~1-2L (dynamic only)     ZERO
Wall clock turn 1       3.1s            ~2.5s               4.8s                ~1-2s                    <0.1s
Wall clock turn 20      13.5s           ~11s                ~7s                 ~1-2s (stable)           <0.1s
Getting slower?         yes             yes                 no                  no                       no
```

---

### TurboQuant Implementation Notes

- Use Algorithm 1 (MSE-only), which community has validated beats MSE+QJL after softmax
- Validated targets:
  - TQ4: MSE = 0.009
  - TQ3: MSE = 0.034
- WHT rotation is the key preprocessing step (Walsh-Hadamard Transform)
- Lloyd-Max scalar quantization per-channel after rotation
- Shannon limit bounds: cannot beat ~3 bits for KV without information loss

---

### Markov RS Implementation Notes

- **Bounded window**: only the last W residual stream states are kept (W = 4-8 typical)
- **Cold tier**: token IDs stored at 4 bytes each for tokens outside the window
- **Checkpoints**: periodic full-state snapshots for rewind/branching
- **Two-stroke decoupling**: separate the residual update from the attention computation
  - Stroke 1: compute residual delta from FFN (or walk)
  - Stroke 2: apply attention correction using window

---

### Hybrid RS + Cracked Attention Implementation Notes

- **Head classification**: classify each attention head as static (cacheable) or dynamic
  - Static: cosine similarity >= 0.942 across entities (95.5% of heads)
  - Dynamic: remaining ~4.5% of heads that vary per-entity
- **Cached attention output format**: precomputed attention output per template cluster
  - 8 template clusters (from session 2026-03-29 analysis)
  - Each cluster stores per-layer, per-head output vectors
- **Dynamic-head-only KV cache**: tiny KV cache for just the ~4.5% dynamic heads
  - At 4K tokens: ~4.5% of 544 MB = ~24 MB
  - At 370K tokens: ~4.5% of 56 GB = ~2.5 GB (but windowed, so ~150-300 MB)
- **Per-token inference pipeline**:
  1. Look up template cluster from entity embedding
  2. Load cached static head outputs (zero compute)
  3. Run attention ONLY for dynamic heads (~1-2 equivalent layers of compute)
  4. FFN via vindex walk (zero matmul)
  5. Combine and project to logits
- **Cost breakdown**:
  - FFN: 0 matmuls (vindex walk)
  - Static attention: 0 matmuls (cached)
  - Dynamic attention: ~1-2 layers equivalent
  - Logits projection: 1 matmul
  - Total: ~2-3 matmuls vs 102+ in full forward pass

---

### Graph Walk Implementation Notes

Three tiers of graph walk:

- **Tier A: Cached template** -- entity maps to known template, output is a direct lookup
  - Covers ~62% of queries (from routing table, 44 sub-centroids)
  - Latency: <0.1 ms
- **Tier B: Dynamic walk** -- entity requires walking the vindex graph
  - FFN walk through 34 layers of precomputed feature vectors
  - Latency: ~0.5 ms (proven 41x vs dense)
- **Tier C: Hybrid fallback** -- unknown entity or low-confidence routing
  - Falls back to Hybrid RS+CA (Strategy 4)
  - Latency: ~1-2 ms

---

### Video Narrative

Four frames for visual presentation:

1. **Frame 1: Compression ceiling** -- TurboQuant hits Shannon limit at 3-4 bits. This is the best compression can do. The cache still grows with context.
2. **Frame 2: Multi-turn crossover** -- Markov RS crosses under TurboQuant around turn 3-5. By turn 20, the gap is enormous. Cache does not grow.
3. **Frame 3: Forward pass eliminated** -- Hybrid RS+CA removes 95.5% of attention compute and all FFN compute. Only ~4.5% of attention heads need real computation.
4. **Frame 4: Endgame numbers** -- RS Graph Walk removes the forward pass entirely. <0.1s per turn, no growth, no matmuls.

---

### Open Questions

1. **Dynamic head stability**: Do the ~4.5% dynamic heads change across different conversation topics, or is the set stable per model?
2. **Template cluster granularity**: Are 8 clusters enough for production quality, or do we need 16-32 for edge cases?
3. **Window size sensitivity**: How does Markov RS accuracy degrade as window size drops below 4? Is W=2 viable for memory-constrained devices?
4. **Cross-model generalization**: Does the 95.5% static head ratio hold for other architectures (Llama, Mistral, Phi)?
5. **Long-context degradation**: At 370K+ tokens, does the dynamic head set grow or stay bounded?
6. **Quantization interaction**: Can TurboQuant be applied to the tiny dynamic-head KV cache in Strategy 4 for additional compression?
7. **Graph construction cost**: One-time cost to build the vindex graph -- is it amortized over enough queries to be worth it for small deployments?
8. **Distributed inference**: How do Strategies 3-5 interact with tensor parallelism and pipeline parallelism across multiple devices?
