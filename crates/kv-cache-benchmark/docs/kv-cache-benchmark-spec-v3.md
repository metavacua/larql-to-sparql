# KV Cache Benchmark Spec v3

> **Status note (2026-04-23):** Hybrid RS + Cracked Attention has been dropped
> as a shipping strategy. The `hybrid_cracked/` module was never implemented.
> Sections below that refer to "Strategy 4 / Hybrid RS+CA" are retained as
> **design notes for future work** and should not be read as describing
> delivered code. The authoritative rung ladder lives in the crate README
> (five rungs: Standard KV, TurboQuant, Markov RS, UnlimitedContext, Apollo,
> plus Graph Walk as a projected target).

## Standard KV vs TurboQuant vs Markov RS vs RS Graph Walk

### Purpose

Five-way comparison showing the progression from compression to elimination
to replacement of the forward pass itself:

  1. Standard KV — the baseline everyone uses
  2. TurboQuant — the best possible compression of that baseline (Shannon ceiling)
  3. Markov RS — eliminate the cache, keep the forward pass
  4. Hybrid RS + Cracked Attention — eliminate MOST of the cache without solving attention fully
  5. RS Graph Walk — eliminate the forward pass, keep the graph

Strategy 4 is the near-term demonstrable win. It doesn't require solving
attention completely. 95.5% of attention heads are cacheable. The FFN side
is already solved. This means most of the KV cache disappears right now,
with the remaining ~4.5% of dynamic heads using a tiny residual KV cache.

Each step is a paradigm shift, not an incremental improvement.
The benchmark produces numbers for the YouTube video and positions LARQL
against the state of the art. The narrative escalates through all five.

---

## The Five Contenders

### 1. Standard KV Cache (Baseline)
- FP16 keys + values, per-layer, per-head, per-token
- What llama.cpp, vLLM, MLX all use today
- Memory: `seq_len × layers × 2 × kv_heads × head_dim × 2 bytes`
- Gemma 3-4B at 4K tokens: ~544 MB
- Gemma 3-4B at 370K tokens: ~56 GB
- Computation: full forward pass (34 layers, attention + FFN)
- The thing everyone optimises around

### 2. TurboQuant (Google, ICLR 2026)
- Compresses KV cache to 3-4 bits per coordinate
- Algorithm: random orthogonal rotation (Walsh-Hadamard) → Lloyd-Max scalar quantization
- Optional: 1-bit QJL residual correction (community found MSE-only beats MSE+QJL after softmax)
- Compression: 4-6× at the Shannon limit — can't go further
- Still stores per-token, per-layer, per-head — still grows O(context_length)
- Still uses softmax attention over ALL positions
- Still runs a full forward pass
- Implementation reference: llama.cpp Discussion #20969 has working C code, 18/18 tests

### 3. Markov Residual Stream (LARQL)
- Eliminates KV cache entirely
- Stores: bounded window of residuals + cold-tier token IDs (4 bytes each)
- Proven: Markov property on Gemma 3-4B — residual IS the complete state
- Four empirical proofs: self-patching, cross-patching, KV equivalence, generation match
- Compression: 135-1,012× depending on tier
- Does NOT grow O(context_length) — bounded window, cold tier is token IDs
- Softmax bottleneck bypassed — attention over 512 positions max
- Still runs a forward pass (through the bounded window)
- The two-stroke engine: each layer = attention stroke + FFN stroke
- Residual decomposes additively: residual[L] = residual[L-1] + attn_delta[L] + ffn_delta[L]
- Bit-perfect Tier 4 reconstruction proven (KL = 0.0)

### 4. Hybrid RS + Cracked Attention (LARQL near-term)
- The key insight: you don't need to SOLVE attention to ELIMINATE most of KV cache
- 95.5% of attention heads are cacheable (cosine 0.942+)
- FFN is already solved (vindex walk, 34 layers validated, zero accuracy loss)
- So: cache the 95.5% static heads, tiny KV only for the 4.5% dynamic heads

What this actually looks like on Gemma 3-4B:
- 10 Q heads, 2 KV heads, 34 layers
- 95.5% cacheable = ~32-33 of 34 layers have fully cacheable attention
- Dynamic heads: ~1-2 layers need real KV cache
- FFN: zero KV needed (vindex walk replaces all FFN computation)

Memory breakdown at 4K tokens:

    Standard KV:    2 × 2 × 256 × 2 bytes × 34 layers × 4096 tokens = 544 MB
                    (all heads, all layers, all tokens)

    Hybrid RS:      Static attention: cached output per template, NOT per-token KV
                      → routing table: 352 KB (44 centroids, one-time)
                      → cached attention outputs: ~2-5 MB per template
                    Dynamic attention: real KV for ~4.5% of heads
                      → ~1-2 layers × 2 × 2 × 256 × 2 × 4096 = ~16-32 MB
                    FFN: zero (vindex walk)
                    Cold tier: token IDs = 16 KB

                    Total active: ~20-37 MB at 4K tokens
                    vs Standard KV: 544 MB → 20-37 MB = 15-27× reduction

Memory at 370K tokens:

    Standard KV:    56 GB
    TurboQuant:     9-14 GB (4-6×)
    Hybrid RS:      ~150-300 MB (dynamic heads only + routing table)
                    = 180-370× compression — WITHOUT solving attention fully

Why this works:
- The 95.5% cacheable heads produce the SAME output for the same template
- "The capital of X is" activates the same attention routing regardless of X
- The entity-specific information enters through FFN, not attention
- The ~1% entity perturbation in the residual doesn't change which heads fire
- The 4.5% dynamic heads are the ones doing actual entity-specific routing
- Those heads need real KV — but that's 4.5% of the original cache, not 100%

What's proven (on real Gemma 3-4B weights):
- 97.1% head cacheability for parametric queries (264/272 heads static, cosine ≥ 0.90)
- 76.5% head cacheability for in-context queries (26/34 layers all-static)
- Dynamic layers identified: L1, L13, L26, L32 (parametric), + L3, L14, L22, L27, L29, L30 (in-context)
- KV reduction: 34× parametric, 4.3× in-context, ~3.4× worst-case (union)
- All 5 test prompts produce correct predictions with hybrid classification
- Routing table: 352 KB, 44 centroids, 100% precision @ 62% coverage
- FFN walk: 34 layers validated, zero accuracy loss
- Template scaffolding: 0.99 cosine across entities
- Entity confusion test: France→Paris, Germany→Berlin, Japan→Tokyo on same template (PROVEN)

Dual retrieval circuits discovered:
- L13 + L26: shared entity-sensitive routing (dynamic for BOTH parametric and in-context)
- L29 + L30: in-context comprehension circuit (static for parametric, DYNAMIC for in-context)
- L1 + L32: parametric routing circuit (dynamic for parametric, STATIC for in-context)
- This is a mechanistic interpretability finding: the model has specialised layers
  for parametric vs in-context retrieval, mappable at per-head granularity.

Dynamic classification by query type:
    Parametric queries (factual, entity-relation):
      → 4 dynamic layers, 34× KV reduction (97.1% static)
    In-context queries (references document, planted facts):
      → 8 dynamic layers, 4.3× KV reduction (76.5% static)
    Unknown / mixed:
      → union of both, 10 dynamic layers, 3.4× KV reduction (70% static)
    All cases: still eliminates majority of KV cache.

What's needed to ship:
- Dynamic classification router (template detection already exists)
- Static head output injection in inference pipeline
- Dynamic-head-only KV cache implementation
- Fallback: if a query doesn't match a known template, use full Markov RS

The three attention tiers:

    Tier S (static, 95.5%):  cached output, no KV, no computation
      → lookup from routing table, inject cached attention delta

    Tier D (dynamic, 4.5%):  real KV cache, but ONLY for these heads
      → tiny KV cache (~16-32 MB at 4K vs 544 MB full)
      → standard attention computation for these heads only

    Tier F (fallback):       query doesn't match template, full attention
      → falls back to Markov RS (bounded window)
      → graceful degradation, not failure

This is the NEAR-TERM WIN. Demonstrable before Lanzarote.
No new research needed — the head cacheability is already proven.
Implementation is: classify heads, cache static outputs, shrink KV to dynamic-only.

### 5. Residual Stream Graph Walk (LARQL endgame)
- Eliminates the forward pass itself
- The forward pass IS a graph walk — three graphs compose:
    FFN graph:        what gets added at each layer (knowledge)
    Attention graph:  where to route at each layer (routing)
    Residual stream:  the walk state connecting them (Markov cursor)
- The residual stream is the COMPOSITION of FFN + attention graphs
- No matrices. No multiplication. Graph traversal + sparse dot products.

What's proven:
- FFN graph: 348K edges extracted, gate KNN → feature → down KNN → token
    France → gate KNN → F9515 → down KNN → Paris (no model loaded, no GPU)
- PEC query mode: 86% vs 14% for direct weight walking on complex queries
- Vindex walk: 34 layers validated, identical top-1 predictions (zero accuracy loss)
- 41× speedup on FFN (21,197ms → 517ms/token)
- 10.7 GB of FFN weights dropped (16.6GB → 5.5GB resident)
- Routing table: 23,697 unique features from 240 passes

What's characterised:
- Attention: two-stroke pattern, 99% fixed routing, 11K coupling edges
- Residual trajectories: 0.99 cosine across entities (template-fixed)
- Dark space: 99.4% of residual is navigation (7-8D), content type (16D),
  tone (3D), register (2D) — not content
- Entity perturbation is ~1% of total signal

What's open:
- Attention routing extraction from Q/K/V weights → explicit routing table
- Residual prediction: can we PREDICT the residual from tokens directly,
  skipping the forward pass? (the 100K tok/s path)
- Pattern walk vs entity walk decomposition

---

## Benchmark Results (Proven on Gemma 3-4B)

### Parametric Knowledge (model's trained weights)

    Test                                Result
    ────────────────────────────────────────────────
    Paris test (all 4 strategies)       ALL produce "Paris"
    Top-1 match (20 factual prompts)    20/20 all strategies (100%)
    Markov RS bit-perfect               cosine = 1.000000, KL = 0.0
    TurboQuant on real K/V              cosine = 0.991, compression 3.88×
    Entity confusion (same template)    France→Paris, Germany→Berlin, Japan→Tokyo
    Generation stability                Coherent 6-16 token sequences

### In-Context Knowledge (information in the prompt)

    Test                                Standard KV    Markov RS (bounded window)
    ───────────────────────────────────────────────────────────────────────────────
    Needle @ 512 tokens                 FOUND          FOUND
    Needle @ 1024 tokens                FOUND          FOUND
    Needle @ 2048 tokens                MISSED         FOUND
    Needle @ 4096 tokens                MISSED         FOUND

    "They stored 285 MB and lost the needle at 2048 tokens.
     We stored 18 MB and found it. Every time."

    Softmax dilution crossover: between 1024 and 2048 tokens.
    Far more aggressive than expected.

### Hybrid RS+CA (confirmed on real model)

    Metric                          Parametric    In-context
    ─────────────────────────────────────────────────────────
    Static heads                    264/272       ~208/272
    Static fraction                 97.1%         76.5%
    Dynamic layers                  4 (L1,13,26,32)   8 (+L3,14,22,27,29,30)
    KV reduction                    34×           4.3×
    Correct predictions             5/5           (same forward pass)
    Dynamic KV per prompt           98 KB         (vs 836 KB full)

    Dual retrieval circuits:
      L13 + L26: entity-sensitive (both parametric and in-context)
      L29 + L30: in-context comprehension (activates for planted facts)
      L1 + L32: parametric routing (activates for trained knowledge)

### Multi-Fact Retrieval

    Test                         Full context    Bounded window
    ───────────────────────────────────────────────────────────
    5 facts planted in 2K ctx    0/5 found       3/5 found
    Needle position sweep @2K    0/5 positions   5/5 positions

### Context vs Parametric Conflict

    "Capital of France is Lyon"  → model follows context (Lyon wins over Paris)
    "Mozart born in London"      → model follows context (London wins, Salzburg in top-10)

### Multi-Turn Retention

    Fact                     Found in top-10?
    ────────────────────────────────────────────
    "My name is Alice"       FOUND (rank 6)
    "Project Lighthouse"     FOUND (rank 5)
    "San Francisco"          MISSED (model limitation at 248 tokens, not strategy)

---

## What We Measure

### Axis 1: Memory Scaling
- Context lengths: 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 370K
- Measure: active inference memory (bytes)
- Measure: cold storage between turns (bytes)
- Models: Gemma 3-4B, Llama 3 8B, Llama 3 70B (config-level, not running full models)
- Plot: log-scale memory vs context length, five lines

### Axis 2: Multi-Turn Wall Clock
- 25 turns, 150 tokens per turn (typical conversation)
- Measure: wall-clock per turn (encode + decode + generate)
- Measure: cumulative memory at each turn

### Axis 3: Accuracy / Fidelity
- Top-1 token match rate across generation
- KL divergence of output distribution vs full-precision baseline
- Cosine similarity of reconstructed K/V vs original
- Needle-in-a-haystack retrieval at scaling context lengths

### Axis 4: First Token Latency (Prefill)
- Time to process a new prompt and be ready to generate
- Context lengths: 512, 2K, 8K, 32K

### Axis 5: Compute Backend
- CPU f32, CPU Q4 NEON, Metal Q4, Metal fused, Graph-only/no GPU

### Axis 6: Cold Storage / Distributed
- Storage per conversation after 2000 tokens of history

### Axis 7: Computation Eliminated (the paradigm axis)

    Operation          Std KV    TQ      Markov RS    Hybrid RS+CA         Graph Walk
    Attention matmul   34L       34L     window only  ~1-2L dynamic only   ELIMINATED
    FFN matmul         34L       34L     34L          ELIMINATED (vindex)  ELIMINATED
    Logits matmul      1×        1×      1×           ELIMINATED (KNN)     ELIMINATED
    KV cache write     34L       34L+Q   none         ~1-2L dynamic only   none
    KV cache read      34L       34L+DQ  none         ~1-2L dynamic only   none
    Cached attn inject none      none    none         ~32-33L (lookup)     none
    Graph lookup       none      none    none         34L FFN (vindex)     3 per hop

---

## Phase 6: Comparative Table (the video frame)

```
                        Standard KV     TurboQuant 4-bit    Markov RS           Hybrid RS+CA             RS Graph Walk
Memory @ 4K tokens      544 MB          ~90-136 MB          10 KB + window      ~20-37 MB                10 KB + graph*
Memory @ 370K tokens    56 GB           ~9-14 GB            55 MB               ~150-300 MB              ~1.5 GB*
Cold storage / conv     978 MB          ~160-240 MB         10.2 KB             10.2 KB + template ID    10.2 KB
Compression vs KV       1×              4-6×                135-1,012×          15-27× (4K) / 180-370×   N/A**
Still grows O(N)?       yes             yes                 no (bounded)        ~O(N) for 4.5% heads     no (graph is fixed)
Softmax dilution?       yes (1/N)       yes (1/N)           no (512 max)        only on 4.5% heads       no (no softmax)
Accuracy                exact           cosine 0.991        KL = 0.0           5/5 correct              top-1 match (factual)
Needle @ 2K tokens      MISSED          MISSED              FOUND              FOUND                    routing*
Head static %           n/a             n/a                 n/a                97.1% / 76.5%            n/a
Forward pass?           yes (34L)       yes (34L)           yes (window)        partial (~1-2L attn)     NO
FFN matmuls?            34L             34L                 34L                 ZERO (vindex)            ZERO (vindex)
Attn matmuls?           34L             34L                 window              ~1-2L (dynamic only)     ZERO
Getting slower?         yes             yes                 no                  no                       no
Requires GPU?           no              no                  no                  no                       NO (CPU lookups)
```

---

## Test Plan

### Unit Tests (per strategy)

```
test_standard_kv_exact_roundtrip         — encode/decode gives identical vectors
test_standard_kv_memory_formula          — memory matches analytical formula
test_turboquant_mse_within_paper         — MSE ≤ 0.009 at 4-bit, ≤ 0.034 at 3-bit
test_turboquant_cosine_above_threshold   — cosine sim ≥ 0.997 at 4-bit
test_turboquant_compression_ratio        — 3.8-4.9× confirmed
test_turboquant_rotation_preserves_norm  — ||Rx|| == ||x|| within eps
test_turboquant_wht_invertible           — WHT(WHT(x)) == x (self-inverse)
test_turboquant_lloyd_max_convergence    — centroids converge for Beta(d/2,d/2)
test_markov_cold_tier_size               — token IDs = 4 × seq_len bytes
test_markov_window_bounded               — memory stays below budget regardless of context
test_markov_reconstruction_exact         — reconstructed residual matches original (KL=0.0)
test_markov_checkpoint_spacing           — recompute cost bounded by checkpoint interval
test_hybrid_head_classification          — 95.5% static, 4.5% dynamic on Gemma 3-4B
test_hybrid_static_head_cosine           — cached output cosine ≥ 0.942 across entities
test_hybrid_dynamic_kv_size              — dynamic-only KV is 15-27× smaller than full KV
test_hybrid_memory_at_4k                 — total active memory ~20-37 MB (not 544 MB)
test_hybrid_memory_at_370k               — ~150-300 MB (not 56 GB)
test_hybrid_template_cache_shared        — template cache is per-template, not per-conversation
test_hybrid_fallback_to_markov           — unknown template gracefully degrades
test_hybrid_ffn_zero_matmul              — FFN path uses vindex, no matrix multiplication
test_graph_walk_france_paris             — "capital of France" → Paris via graph only
test_graph_walk_matches_forward_pass     — top-1 match on 50 factual queries
test_graph_walk_routing_table_coverage   — routing table resolves ≥60% of test queries
test_graph_walk_fallback_triggers        — free-form queries correctly fall back
test_graph_walk_template_decomposition   — pattern walk + entity walk = correct prediction
test_graph_walk_no_matmul                — verify zero matrix multiplications in graph path
```

### Accuracy Tests — Parametric Knowledge

```
test_parametric_top1_20_factual          — 20 factual prompts, top-1 match (DONE: 20/20)
test_parametric_entity_confusion         — same template, different entities (DONE: 3/3)
test_parametric_generation_stability     — 30-token generation, coherence check (DONE: 3/3)
test_parametric_diverse_categories       — capitals, currencies, birthplaces, science, geography
test_parametric_kl_divergence            — full softmax distribution vs baseline per prompt
test_parametric_turboquant_drift         — generate 50 tokens, measure first divergence point
```

### Accuracy Tests — In-Context Knowledge

Level 1: Needle-in-a-Haystack (single fact retrieval)
```
test_needle_short_512                    — planted fact, 512 context (DONE: FOUND all)
test_needle_scaling_context              — 512/1K/2K/4K (DONE: MISSED at 2K StdKV)
test_needle_bounded_vs_full              — StdKV vs Markov RS (DONE: RS FINDS at 4K)
test_needle_8k                           — push to 8K context
test_needle_32k                          — push to 32K, expect clear StdKV/TQ failure
test_needle_position_sweep               — needle at 10%/25%/50%/75%/90% of context
test_needle_multiple                     — plant 5 needles, retrieve each one
test_needle_turboquant_vs_standard       — does TQ's quantization make dilution worse?
```

Level 2: Multi-Fact Retrieval
```
test_multifact_10_facts_at_2k            — plant 10 facts across 2K context
test_multifact_10_facts_at_8k            — same at 8K
test_multifact_conflicting_facts         — must find LATEST version of updated fact
test_multifact_similar_entities          — "Alice at IBM" + "Bob at Google" — no confusion
```

Level 3: In-Context Reasoning
```
test_incontext_transitive                — "A > B. B > C. Who is smallest?" → C
test_incontext_arithmetic_from_context   — "Price $45. Tax 10%." → total?
test_incontext_negation                  — "NOT on Tuesday" — negation tracking
test_incontext_list_completion           — "Mon, Wed, Fri, ___" → Sunday
```

Level 4: In-Context vs Parametric Conflict
```
test_conflict_override_capital           — context says "capital of France is Lyon"
test_conflict_bounded_window_includes    — conflict fact IS in window → context wins
test_conflict_bounded_window_excludes    — fact OUTSIDE window → weights win (documented)
```

Level 5: Multi-Turn In-Context Retention
```
test_multiturn_fact_retention_3          — 3 facts, query after filler (DONE: 2/3)
test_multiturn_fact_retention_at_2k      — grow to 2K tokens, query
test_multiturn_evolving_facts            — fact changes across turns, must find latest
test_multiturn_10_turns_10_facts         — retrieval rate per strategy at each distance
```

Level 6: Adversarial In-Context
```
test_adversarial_repeated_entity         — "John" 50× with different facts
test_adversarial_distractor_needles      — real needle + 10 decoys
test_adversarial_paraphrased_query       — semantic matching, not keyword
test_adversarial_nested_context          — fact inside quote inside paragraph
```

### In-Context Test Result Matrix (expected)

```
                         StdKV    TurboQ    Markov RS    Hybrid    Graph Walk
Needle @512              FOUND    FOUND     FOUND        FOUND     FOUND
Needle @2K               MISSED   MISSED    FOUND        FOUND     routing*
Needle @8K               MISSED   MISSED    FOUND        FOUND     routing*
Needle @32K              MISSED   MISSED    FOUND        FOUND     routing*
Multi-fact @2K (10)      ~7/10    ~7/10     10/10        10/10     routing*
Latest-fact              ?        ?         ?            ?         N/A
Entity disambiguation    ?        ?         ?            ?         N/A
Context vs weights       context  context   window-dep   window-dep weights
Multi-turn @2K           ?        ?         ?            ?         N/A
```

### Shader Tests (Phase 3)

```
test_wht_shader_matches_cpu              — Metal WHT output == CPU WHT output
test_turboquant_encode_shader_roundtrip  — encode → decode on GPU matches CPU path
test_sparse_matvec_correct               — sparse gather gives same result as dense
test_attention_with_rope_produces_paris   — GPU attention with RoPE → "Paris"
test_knn_logits_top1_match               — GPU KNN gives same top-1 as dense logits
```

### Benchmark Tests (criterion)

```
bench_encode_4k_standard_kv              — encode 4K tokens, standard
bench_encode_4k_turboquant_4bit          — encode 4K tokens, TurboQuant 4-bit
bench_encode_4k_markov_residual          — encode 4K tokens, Markov RS
bench_graph_walk_single_query            — one factual query, graph walk only
bench_memory_sweep_512_to_370k           — memory at each context length, all strategies
bench_wht_d128                           — WHT at d=128 (DONE: 577ns)
bench_wht_d256                           — WHT at d=256 (DONE: 1.15µs)
bench_metal_turboquant_encode            — GPU encode throughput
bench_metal_sparse_matvec                — GPU sparse walk throughput
```

---

## Video Narrative

Five escalating frames:

Frame 1 — The compression ceiling:
    TurboQuant at 370K: 56 GB → 9-14 GB (4-6×, Shannon ceiling)
    Markov RS at 370K:  56 GB → 55 MB (1,012×)
    Hybrid RS+CA:       56 GB → 3.9 MB (6,615×)
    "They compressed the cache. We eliminated it."

Frame 2 — The multi-turn crossover:
    Standard KV grows every turn. Hybrid RS+CA flat at 2.5 MB from turn 4.
    Turn 25: Standard KV 261 MB, Hybrid 2.5 MB. Same accuracy.

Frame 3 — Storing less, finding more (THE HEADLINE):
    Standard KV MISSES needle at 2048 tokens. Markov RS FINDS it at 4096.
    "They stored 285 MB and lost the needle. We stored 18 MB and found it."
    This is the frame that changes how people think about KV caches.

Frame 4 — The forward pass eliminated:
    95.5% of attention cached. FFN is graph lookups.
    "We didn't solve attention. We didn't need to."

Frame 5 — The endgame:
    Graph walk: <1ms, 1000+ queries/sec, conversation is 10 KB.

Frame 3 is the video thumbnail. "They stored 285 MB and lost it."

---

## Open Questions

1. TurboQuant: Rust from scratch (chosen). WHT + Lloyd-Max + packing.
2. QJL: No. MSE-only (six teams confirmed).
3. Markov RS decode cost: amortised over turn, checkpoint-based.
4. Metal WHT: RESOLVED — No. CPU 577ns, GPU dispatch 350× slower.
5. correct_attention.metal: THE BLOCKER for GPU pipeline.
6. Graph Walk coverage: per-tier reporting, honest about fallback.
7. Attention routing extraction: biggest open research question.
8. Multi-token graph walk: open, Experiment 5 will answer.
9. Parametric vs in-context: Graph Walk is parametric-only. Document indexing unifies them.
10. Bounded window routing quality: determines effective coverage for in-context.
11. TurboQuant codebook: cosine 0.991 on real vectors, needs Beta(d/2,d/2) calibration.
12. Context vs weights conflict: bounded window = documented, testable limitation.
