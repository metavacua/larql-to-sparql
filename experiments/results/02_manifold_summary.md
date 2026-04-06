# Experiment 02: Knowledge Manifold & Attention Replacement

## Summary of Results

### Part 1: Gate Vector Dimensionality
- **Gate vectors are full rank.** 2410/2560 dims for 99% variance. No low-dimensional manifold.
- Gate vectors use all available capacity. Knowledge is distributed, not compressed into a subspace.

### Part 2: Entity Embedding Dimensionality  
- **Entity embeddings are lower-rank** but not dramatically: 298D for 90% (453 entities in 2560D).
- Per-relation **difference vectors** (e.g., embed(Paris) - embed(France)) are much lower: 5-22D for 50%.
- The "9-11D" finding from cross-model alignment lives in the relation difference space, not raw embeddings.

### Part 3: Attention Output Analysis
- The **capital embedding subspace** (10D) captures only 0.5-4.3% of attention output variation.
- Cross-relation cosines are 0.85-0.98: attention output is **relation-independent** at most layers.
- Entity variation in attention output is 3-11D (for 90%), but NOT in the relation subspace.
- Attention output ≈ large shared scaffolding + small entity signal (~15-35%) + tiny relation signal (<5%).

### Part 4: Attention Replacement (Within-Set)
- **15/15 correct** replacing attention at ALL 34 layers with scaffolding + 11D entity projection (K=11).
- Real attention only gets 12/15. Synthetic is cleaner.
- K=7 is the crossover point (13/15, beats real).
- **This was memorization** — the entity projection used per-entity norm scaling from test data.

### Part 5: Generalization Test
- **0/20 on unseen entities** for all approaches:
  - SVD entity subspace from 12 examples: memorizes those 12 entities.
  - OV circuit (O @ V @ embedding): wrong because attention reads residuals, not embeddings.
  - Scaled OV (with per-head attention weights): still wrong for same reason.
  - Learned linear projection (ridge, 81 train): R² negative on test set. 0/21.
  - SVD-truncated regression (rank 5-81): 0/21 at all ranks. Best cosine only 0.46.

### Part 6: Attention Pattern Analysis
- No layer strongly attends to entity token. Max mean attention to "France" is 0.26 (L14).
- Entity token contributes only 5-38% of attention output per layer.
- "The" position contributes up to 52% (L20) — its residual carries accumulated computation.
- Attention patterns are 99% stable across entities (template-fixed). ✓ confirmed.

## Key Findings

### What IS true:
1. **Scaffolding is real.** 0.91 cosine across different templates. Template-fixed component is large and stable.
2. **99% of attention is template-fixed.** Confirmed by direct measurement.
3. **Entity-specific signal is ~15-35%** of attention output at each layer, in ~10 dimensions.
4. **Within known entities, replacement is perfect** (15/15 with K=11, better than real attention).

### What IS NOT true:
1. Gate vectors are NOT low-dimensional. Full rank.
2. Entity signal is NOT a linear function of the raw token embedding.
3. OV circuit on raw embeddings does NOT approximate real attention output.
4. The embedding→entity_signal mapping does NOT generalize linearly.

### The Fundamental Barrier:
Attention reads the **residual stream** at each token position, which is the result of all prior layers of computation. The entity signal in attention depends on:
- The entity's token embedding (10-35%)  
- The accumulated computation at the entity position from prior layers
- The accumulated computation at ALL OTHER positions (60-80% of output)

This makes attention fundamentally **sequential and context-dependent** — each layer's output depends on all prior layers' outputs at all positions. A static mapping from embedding to attention output cannot capture this.

## Files
- `02_svd_spectrum.json/png` — Gate vector SVD (full rank)
- `02_entity_manifold.json/png` — Entity embedding SVD  
- `02_per_relation_manifold.json/png` — Per-relation entity SVD
- `02_relation_difference_manifold.json/png` — Relation difference vectors (5-22D)
- `02_attention_capture.npz` — Raw attention output captures
- `02_attention_relation_subspace.png` — Attention vs relation subspace analysis
- `02_k_sweep_full_replacement.json/png` — K dimension sweep (within-set)
- `02_multi_task_replacement.json/png` — Multi-task replacement results
- `02_generalization.json/png` — Generalization failure
- `02_entity_attention_pattern.png` — Attention pattern analysis
- `02_learned_projection.json/png` — Learned projection failure
- `02_learned_projection_v2.json` — SVD-truncated regression (all ranks fail)
- `02_residual_trajectory.json/png` — Answer trajectory through all layers
- `02_attn_vs_ffn_decomposition.json/png` — Attention vs FFN decomposition

## Documentation

Full writeup: [docs/residual-trace.md](../../docs/residual-trace.md)
