# Backpropagation is INSERT — Experiment Results

**Chris Hay | LARQL Project | April 2026**

---

## Summary

We trained a 20M-parameter Gemma-like transformer from scratch on synthetic data containing three knowledge types — factual relations (synthetic KG), syntactic relations (WordNet), and code (Python/Rust) — and measured where gradient descent writes to the FFN layers during training.

**The headline finding: gradients ARE sparse, targeted INSERTs — but the addressing scheme is feature-level, not layer-level.**

Different knowledge types write to **different FFN features within the same layer**. The layer-band structure seen in large models (Gemma 3-4B) is a consequence of scale, not the fundamental mechanism. At 12 layers, the model partitions its feature space within each layer. At 34+ layers, that feature partitioning stratifies into layer bands.

---

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Architecture | Gemma-like decoder-only (gated FFN, RoPE, GQA, RMSNorm) |
| Parameters | ~20M |
| Layers | 12 |
| Hidden dim | 256 |
| FFN dim | 1024 |
| Tokenizer | Gemma 3 (google/gemma-3-4b-pt), clamped to 32K vocab |
| Training data | 1,710 samples: 750 factual, 900 syntax (WordNet), 60 code |
| Ground truth | Synthetic KG (50 entities, 3 relations), WordNet (synonym, hypernym, antonym, meronym, morphology), Python/Rust AST |
| Training | 40 epochs, AdamW, lr=3e-4, batch=8, MPS (Apple Silicon) |
| Measurements | Gradient anatomy every ~250 steps, feature tracking every 25 steps |

---

## Finding 1: Feature-Level Separation Confirmed

**Feature overlap between syntax and knowledge gradients: 0.00–0.14 across all layers and all training steps.**

When a syntactic example ("A dog is a type of animal") computes a gradient, it modifies specific FFN features. When a factual example ("The capital of Freedonia is Markov") computes a gradient, it modifies **different** features — even within the same layer.

This was measured via contrastive gradient anatomy: for each layer, we compared the top-10 features receiving gradient from syntax vs knowledge inputs. The Jaccard similarity was consistently near zero, meaning the two knowledge types write to almost entirely non-overlapping feature sets.

**This is the INSERT confirmed.** The gradient writes to a specific address — (layer, feature_index) — and different knowledge types have different addresses.

---

## Finding 2: Layer Bands Do NOT Emerge at This Scale

The spec hypothesised three layer bands: syntax (L0-3), knowledge (L4-7), output (L8-11). The data shows:

| Step | Syntax→Syntax Band | Knowledge→Knowledge Band | Δ(Syn-Kn) |
|------|-------------------|-------------------------|------------|
| 0 | 53.7% | 27.3% | +0.264 |
| 200 | 55.0% | 25.6% | +0.294 |
| 1000 | 48.7% | 28.6% | +0.201 |
| 3000 | 40.6% | 28.8% | +0.117 |
| 8560 | 21.4% | 29.9% | -0.085 |

The gradient profile **inverts** during training:
- **Early training:** gradients concentrate in early layers (backprop chain effect from random init)
- **Late training:** gradients shift to the final layer (L11) where the logit bottleneck lives
- **Knowledge band targeting stays flat at ~28-30%** throughout — indistinguishable from chance (4/12 layers = 33%)

**Interpretation:** 12 layers is not enough depth for layer-band specialisation. The model uses feature-level partitioning instead. The layer-band structure in Gemma 3-4B (34 layers) likely emerges when there's enough depth for feature clusters to stratify spatially.

---

## Finding 3: Feature Addresses Stabilise During Training

We tracked which specific FFN features (by index) received gradient writes from each relation type across training. The Jaccard similarity of top-10 features between consecutive checkpoints reveals a three-phase stability trajectory:

### Phase 1: Initial Stability (step 0-50)
All relations show high stability (0.64–0.82 Jaccard). This is misleading — at random init, features are interchangeable and the "top" features are noise.

### Phase 2: Volatile Reshuffling (step 50-250)
All relations drop to volatile (0.06–0.33 Jaccard). Training begins and gradients rapidly reorganise which features serve which purpose. This is the "database schema being written" phase.

### Phase 3: Settling (step 500+)
Features gradually stabilise. The trajectory differs by knowledge type:

**Code stabilises first** (step 2000):
```
python:if     ███████████████████████  0.775  [STABLE]
rust:struct   ██████████████████████   0.766  [STABLE]
rust:impl     █████████████████████    0.713  [STABLE]
rust:fn       █████████████████████    0.704  [STABLE]
python:for    ███████████████████      0.643  [STABLE]
python:call   ██████████████████       0.627  [STABLE]
```

**Syntax and facts settle later** (step 4000):
```
president_of  ███████████████████      0.640  [STABLE]
rust:while    ███████████████████      0.636  [STABLE]
capital_of    ████████████████         0.563  [settling]
plural        ████████████████         0.552  [settling]
synonym       ███████████████          0.515  [settling]
hypernym      ███████████████          0.517  [settling]
```

**Why code first?** Code samples have the most distinctive token patterns (keywords, brackets, indentation). The gradient signal is cleaner — fewer competing relations claim the same features. Factual relations compete more because "The capital of X is Y" and "The president of X is Y" share template structure.

---

## Finding 4: Rare Relations Get Exclusive Features

The most exclusive features (highest fraction of gradient from a single relation) consistently belong to **rare relations**:

| Relation | Samples | Best Exclusivity | Persistent Feature |
|----------|---------|------------------|--------------------|
| antonym | 4 | 0.405 | L1 #251, L4 #52, L10 #48 |
| meronym | 28 | 0.427 | L6 #491, L8 #711 |
| plural | 8 | 0.377 | L1 #977, L4 #344 |
| rust:fn | 5 | 0.369 | L5 #757, L4 #996 |
| python:call | 5 | 0.391 | L7 #525 |

**Why?** Fewer gradient sources = less competition for feature ownership. Antonym (4 training samples) faces almost no competition for its features. Capital_of (250 samples) shares features with president_of and currency_of because all three use the same "The X of Y is Z" template structure.

This is the capacity version of the INSERT: **each training example is one vote for feature ownership, and rare facts win exclusive addresses while common facts share.**

---

## Finding 5: Some Features Are Persistent Across Training

Several features maintained their relation assignment from step 500 through step 4000+:

| Feature | Relation | First Seen | Layers |
|---------|----------|------------|--------|
| L4 #344 | plural | step 500 | Stable through step 4000 |
| L8 #711 | meronym | step 500 | Stable through step 4000 |
| L10 #48 | antonym | step 500 | Stable through step 4000 |
| L4 #996 | rust:fn | step 0 | Stable through step 4000 |
| L9 #69 | antonym | step 1000 | Stable through step 4000 |
| L7 #933 | antonym→plural | step 250 | Changed owner at step 2000 |

Feature #933 at L7 is interesting: it was initially claimed by antonym, then plural took over at step 2000. This is a feature **UPDATE** — the database reassigned the address when a different relation's gradient signal became stronger at that location.

---

## Finding 6: Exclusivity Increases in Later Layers

At step 4000, mean exclusivity shows a gradient from early to late layers:

```
L0-3:   0.114, 0.115, 0.116, 0.115  (lower exclusivity)
L4-7:   0.119, 0.119, 0.122, 0.122  (medium)
L8-11:  0.126, 0.129, 0.131, 0.136  (higher exclusivity)
```

Later layers have more exclusive features. This hints at the band structure: **if the model had 34 layers, this gradient would have room to stratify into distinct bands.** At 12 layers, it's a smooth gradient. At 34 layers, the gradient steepens into bands.

---

## The Revised Thesis

The original hypothesis was: "each gradient step is structurally equivalent to a graph INSERT, writing a key-value pair into the appropriate FFN layer band."

The revised thesis based on experimental evidence:

**Each gradient step IS a sparse INSERT into specific FFN features.** But the addressing scheme is `(layer, feature_index)`, not `layer_band`. Different knowledge types write to different features, confirmed by near-zero overlap. Feature addresses stabilise during training (Jaccard 0.5–0.77 by convergence), with code stabilising first, then facts. The layer-band structure seen in large models is the feature-level partitioning stratified across depth — a consequence of scale, not the fundamental mechanism.

The database analogy holds, but the schema is more nuanced:

| Original Analogy | Revised Understanding |
|-----------------|----------------------|
| Layer band = table | Feature cluster = table (within any layer) |
| Layer = schema | Layer = partition key (features cluster by type within layers, then by layer at scale) |
| INSERT target = layer | INSERT target = (layer, feature_index) |
| Schema is architectural | Feature→relation mapping is learned via gradient competition |
| Bands crystallise sequentially | Feature addresses crystallise sequentially: code first, then facts, then syntax |

---

## What This Means

1. **Training IS writing to a database.** The gradient is sparse (top-5 features per layer), targeted (different for each relation type), and writes to a stable address. The INSERT metaphor is validated at the feature level.

2. **The database schema is learned, not architectural.** Which feature stores which relation is determined by gradient competition during training, not by the architecture. The architecture provides the storage format (gate→down key-value structure). The optimiser fills it.

3. **Rare facts get clean storage; common facts share.** This is exactly how a real database works under capacity pressure — unique keys get dedicated rows, frequent patterns get compressed.

4. **Layer bands are an emergent property of scale.** At 12 layers, features partition within layers. At 34 layers, the within-layer partitions stratify across the depth axis. The bands in Gemma 3-4B are real — they're just the large-scale version of what we see at the feature level.

5. **Code is the first knowledge type to crystallise.** Programming language syntax has the most distinctive token patterns, so its features stabilise first. This matches the observation from Gemma 3-4B that code AST features are among the cleanest in the vindex.

---

## Next Steps

- **Option A: Gradient anatomy on pre-trained Gemma 3-4B** — validate that the feature-level separation holds at scale AND stratifies into layer bands
- **Option B: Scale sweep** — train 12, 18, 24, 34 layer models to find the depth threshold where feature partitioning becomes layer banding
- **Option C: Feature persistence in pre-trained models** — check if the persistent features we found (L4 #344 → plural, L8 #711 → meronym) have analogues in Gemma 3-4B's vindex gate vectors

---

---

## Finding 7: FFN and Attention Are Separable (Freeze-FFN Experiment)

The strongest result. Three training runs on identical architecture and data:

| Run | Final Loss | Time | Trainable Params |
|-----|-----------|------|-----------------|
| **Baseline** (full training) | 0.8209 | 1151s | 19,994,880 |
| **Freeze-FFN** (trained DB, attention only) | **0.8188** | **570s** | 10,557,696 |
| **Progressive** (freeze FFN after epoch 5) | 0.8275 | 646s | 10,557,696 |

### Loss curve comparison:

| Epoch | Baseline | Freeze-FFN | Progressive |
|-------|----------|-----------|-------------|
| 5 | 1.7149 | **1.2579** | 1.7206 |
| 10 | 1.0925 | **0.8502** | 1.0936 |
| 15 | 0.9247 | **0.8189** | 0.9108 |
| 20 | 0.8614 | **0.8065** | 0.8570 |
| 30 | 0.8262 | **0.8010** | 0.8271 |
| 40 | 0.8209 | 0.8188 | 0.8275 |

### What this proves:

1. **The FFN database and attention query engine are separable.** Freeze-FFN matches baseline loss while training only attention (47% fewer trainable params, 2x faster wall clock).

2. **Freeze-FFN BEATS baseline mid-training.** It reaches 0.7956 at epoch 38 — lower than the baseline ever achieves. When the database is correct, the query engine trains better because it doesn't have to compensate for a still-forming FFN.

3. **5 epochs is all the FFN needs.** Progressive freeze (5 epochs full, then freeze FFN) matches baseline at convergence. The volatile phase is sufficient for FFN construction.

4. **1.7x convergence speedup.** Freeze-FFN reaches baseline epoch-10 loss (1.09) by epoch 6.

### Implication:

If you populate the FFN database correctly, you only need to train the query engine. And it trains better. This directly enables Experiment A (graph-to-weights compilation): compile the FFN from structured data, then train attention only.

---

## Files

| File | Contents |
|------|----------|
| `experiment.py` | v1: char-level tokenizer, basic gradient anatomy |
| `experiment_v3.py` | v2: Gemma tokenizer, WordNet data, contrastive tests |
| `experiment_v4_features.py` | v3: feature-level tracking, stability, exclusivity |
| `synth_data.py` | v1 synthetic data (hand-crafted) |
| `synth_data_v2.py` | v2 synthetic data (WordNet + synthetic KG) |
| `model.py` | TinyGemma (20M params, 12 layers) |
| `results/` | v1 results |
| `results_v3/` | v3 results (anatomy, contrastive, loss) |
| `experiment_v5_freeze.py` | Freeze-FFN: baseline vs freeze vs progressive |
| `results_v5_freeze/` | Freeze-FFN results |

All experiments ran on Apple Silicon (MPS) in 6–40 minutes.
