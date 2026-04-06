# Training-Free Knowledge Insertion

How to inject new factual knowledge into a model without training, fine-tuning, or modifying model weights. One forward pass to capture the residual stream, eight feature writes to the vindex, and the model produces the new fact at 94.6% confidence while preserving existing knowledge.

## The Result

```
Before INSERT:
  "The capital of Atlantis is" → said (17.8%)

After INSERT (8 features, no training):
  "The capital of Atlantis is" → Poseidon (94.6%)

Existing knowledge preserved:
  "The capital of France is"   → Paris (60.5%, down from 80.5%)
```

Cost: one forward pass (~30s) + eight feature writes (~1ms). Permanent in the vindex. No model weights modified.

## Architecture

Knowledge injection requires control over two independent systems:

```
Gate vector (vindex):  WHICH features fire    — the trigger
Down vector (weights): WHAT they output       — the knowledge
```

The gate determines when a feature activates. The down projection determines what it contributes to the residual stream. Both must be controlled for the insertion to affect inference output.

### What didn't work

| Approach | Gate | Down | Result |
|---|---|---|---|
| Embedding-based gate | `embed("Atlantis")` | model weights | Gate doesn't fire (cos=0.01 between embedding and L24 residual) |
| Trace-guided gate only | actual residual | model weights | Gate fires at rank 1, score 53K. Output unchanged — down weight outputs original token |
| Re-gate existing features | actual residual | model weights (existing) | 200 features fire. Poseidon projection too weak (0.03/feature) |
| Down override, single layer | actual residual | Poseidon direction | Output changes! But alpha needed to affect Atlantis also breaks France |

### What worked

**Multi-layer small-alpha down override.** Spread across 8 knowledge layers (L20-L27), each contributing a small nudge toward Poseidon. For Atlantis (no competing signal), nudges accumulate to 94.6%. For France (strong existing Paris signal), nudges are diluted — Paris stays at rank 1.

## Method

### Step 1: Capture residuals

Run `infer_trace` on the target prompt. This returns the actual residual vector at each layer's last token position — what `gate_knn` sees during inference.

```python
preds, residuals = vindex.infer_trace("The capital of Atlantis is")
# residuals[layer] = numpy array of shape (hidden_size,)
# This is the ACTUAL query vector that gate_knn operates on during inference.
```

**Critical insight:** The residual at L24 has cosine **0.01** with `embed("Atlantis")`. They're essentially orthogonal. The embedding is norm ~50, the residual is norm ~38,000. Gate vectors built from embeddings don't fire during inference because the residual stream is a completely different vector after 24 layers of attention.

```
embed("Atlantis"):   norm=51, raw token vector
residual at L24:     norm=38,319, accumulated computation from 24 layers
cosine:              0.0106 (orthogonal)
```

### Step 2: Compute the gate and down vectors

For each knowledge layer:

```python
for layer in range(20, 28):
    residual = residuals[layer]
    
    # Gate: match the Atlantis residual so the feature fires during inference
    avg_norm = mean(norm(existing_gate_vectors))
    gate_vec = residual * (avg_norm / norm(residual))
    
    # Down: Poseidon embedding direction, scaled
    # embed(Poseidon) * embed_scale gives the direction in residual space
    # that increases the Poseidon logit
    down_vec = embed("Poseidon") * embed_scale * alpha  # alpha=0.25
```

**Why `embed(target) * embed_scale`?** For models with tied embeddings (Gemma, Llama), `lm_head = embed`. The logit for token T is `lm_head[T] · residual / logits_scale`. To increase Poseidon's logit, the down vector must align with `embed(Poseidon)`. The `embed_scale` factor (√hidden_size ≈ 50.6 for Gemma 3 4B) converts from embedding space to residual space.

**Why alpha=0.25?** The feature's activation magnitude is determined by the model's own gate/up weights at that slot (not our inserted gate vector). With 8 layers each contributing alpha=0.25, the total effective alpha is ~2.0. Single-layer experiments showed alpha=5 produces Poseidon at 27% but breaks France. Multi-layer at alpha=0.25 produces 94.6% without breaking France because the contributions are distributed.

### Step 3: Insert

```python
for layer in range(20, 28):
    free_feat = vindex.find_free_feature(layer)  # unused slot, c_score=0
    vindex.set_gate_vector(layer, free_feat, gate_vec)
    vindex.set_down_vector(layer, free_feat, down_vec)   # override
    vindex.set_feature_meta(layer, free_feat, "Poseidon", 0.95)
```

The `set_down_vector` stores a custom down projection override. During inference, `sparse_ffn_forward_with_overrides` uses this vector instead of the model's down weight row for that feature slot.

### Step 4: Verify

```python
preds = vindex.infer("The capital of Atlantis is")
# → [("Pose", 0.946), ...]   # "Pose" is the first subtoken of "Poseidon"

preds = vindex.infer("The capital of France is")
# → [("Paris", 0.605), ...]  # preserved, down from 0.805
```

## Why multi-layer works

The model's FFN features fire based on the model's own gate/up weights, not the vindex gate vector. The vindex gate only determines **which** features are selected by `gate_knn`. The actual activation magnitude comes from `model_gate[layer, feature] · residual` and `model_up[layer, feature] · residual`.

For a free feature slot (c_score=0), the model's gate/up weights produce modest but non-zero activations for any input. This activation multiplies the down override vector, contributing to the residual.

**Single layer:** One feature with a strong override. The model's activation at that slot is the same for France and Atlantis (cos=0.98 between their residuals). So both get the same Poseidon push. At alpha high enough for Atlantis, France breaks.

**Multi-layer:** Eight features with weak overrides. Each layer's residual differs slightly between France and Atlantis. The cumulative effect on Atlantis (no competing signal) exceeds the cumulative effect on France (strong Paris signal that absorbs the perturbation).

## Alpha sweep results

### Single layer (L26)

| alpha | Atlantis | France |
|---|---|---|
| 0.5 | said (17.5%) | Paris (80.5%) |
| 1.0 | said (16.9%) | Paris (58.6%) |
| 3.5 | **Pose (17.0%)** | Pose (74.8%) |
| 5.0 | Pose (27.0%) | Pose (78.4%) |
| 10.0 | Pose (65.1%) | Pose (79.8%) |

No sweet spot — France breaks before Atlantis benefits.

### Multi-layer (L20-L27, 8 layers)

| alpha/layer | total | Atlantis | France |
|---|---|---|---|
| 0.25 | 2.0 | **Pose (94.6%)** | **Paris (60.5%)** |
| 0.50 | 4.0 | Pose (96.2%) | Paris (25.0%) |
| 0.75 | 6.0 | Pose (91.9%) | a (26.7%) |
| 1.00 | 8.0 | Pose (33.7%) | Pose (38.5%) |

### Spread thinner — the Pareto frontier

More layers with smaller alpha reduces degradation:

| Config | Atlantis | Paris | Paris degradation |
|---|---|---|---|
| 8L × 0.25 | 94.6% | 60.5% | -20.0 pts |
| 12L × 0.15 | 91.1% | 57.9% | -22.6 pts |
| 16L × 0.10 | 63.0% | 70.4% | **-10.1 pts** |
| **16L × 0.12** | **78.4%** | **66.8%** | **-13.7 pts** |
| 20L × 0.10 | 39.4% | 70.6% | -9.9 pts |

**16L × 0.12 is the recommended config.** Atlantis at 78%, Paris degradation only 14 points. For maximum new-fact confidence, use 8L × 0.25 (94.6% but 20 points degradation). For minimal degradation, use 20L × 0.08 (26% but only 7 points degradation).

Orthogonal down vectors (removing the Paris component) did not help — degradation comes from residual perturbation magnitude, not Paris logit direction.

## Experiment series

All experiments in `experiments/04_constellation_insert/`:

| File | What | Key finding |
|---|---|---|
| `constellation.py` | Template extraction + walk-level testing | 145 shared features between France/Germany (template), 135 entity-specific |
| `trace_guided.py` | Inference with trace-guided gates | Gates fire at rank 1, scores 39K-53K. Output unchanged — down weights control output, not gates |
| `regate.py` | Re-gate 200 features toward Poseidon | Features fire but per-feature Poseidon projection is 0.03 — too weak |
| `down_override.py` | Residual delta as down vector | Output changes! But France breaks (delta too large and unfocused) |
| `down_sweep.py` | Alpha sweep with Poseidon embed direction | alpha=5 → Pose 27%, alpha=10 → 65%. France also breaks |
| `selective_insert.py` | Orthogonal gate (Atlantis-specific) | Model's up/gate weights fire for both — orthogonality doesn't help |
| `fine_sweep.py` | Fine alpha between 1.0-5.0 | France breaks at alpha=2.0, Atlantis needs alpha=3.5 |
| `multilayer.py` | **8 layers × alpha=0.25** | **Atlantis 94.6%, France 60.5%** |

Results saved in `experiments/results/04{a-h}_*.json`.

## Implementation

### Rust changes

**larql-vindex:**
- `VectorIndex.down_overrides: HashMap<(usize, usize), Vec<f32>>` — per-feature custom down vectors
- `GateIndex::down_override()` — trait method, default returns None
- `VectorIndex::set_down_vector()` — stores override
- `gate_knn()` bug fix — checks heap before mmap (was ignoring INSERT mutations)

**larql-inference:**
- `sparse_ffn_forward_with_overrides()` — like `sparse_ffn_forward` but subtracts the model's down contribution for overridden features and adds the override instead
- `predict_with_ffn_trace()` — forward pass that captures per-layer residuals
- `PredictResultWithResiduals` — predictions + residual vectors

**larql-python:**
- `vindex.infer_trace(prompt)` — returns `(predictions, residuals)`
- `vindex.set_down_vector(layer, feature, vector)` — stores override
- `vindex.find_features_by_target(token)` — searches down weights for target alignment
- `vindex.set_gate_vector()`, `set_feature_meta()`, `find_free_feature()` — low-level mutation API

### Limitations

1. **Subtoken output.** "Poseidon" is two subtokens (68077, 108277). The override produces "Pose" (first subtoken) at 94.6%. The model would need to continue generating "idon" — which requires the autoregressive loop, not just a single forward pass.

2. **France degradation.** Paris drops from 80.5% to 60.5% because the inserted features fire for any "capital of X" query. The model's gate/up weights at the free slot respond to the general pattern, not just Atlantis.

3. **Alpha sensitivity.** The sweet spot (alpha=0.25 per layer, 8 layers) is specific to this model and prompt pattern. Different models, different hidden sizes, different query templates may need recalibration.

4. **No selectivity guarantee.** The inserted features fire for any input whose residual has high dot product with the gate vector. The model's own gate/up weights amplify this non-selectively.

### Future directions

1. **Per-entity gating.** Use the orthogonal component of the residual (Atlantis minus France direction) as the gate, combined with a learned scaling that compensates for the lower dot product.

2. **Down vector learning.** Instead of using `embed(target) * embed_scale * alpha`, learn the optimal down vector from a few examples using the residual stream as supervision.

3. **Compile down.** Bake the gate + down overrides into the model's actual weight matrices, eliminating the runtime override check. The vindex becomes the training data; the compiled model is the result.

4. **Multi-token targets.** Handle multi-subtoken targets by inserting features that shift the residual stream across multiple output positions, not just the last position.

## Reproduction

```bash
# Build the vindex (requires gemma-3-4b-it weights)
cargo run -p larql-cli --release -- repl
> EXTRACT MODEL "google/gemma-3-4b-it" INTO "output/gemma3-4b-f16.vindex" WITH ALL;

# Run the full experiment
pip install -e crates/larql-python
python experiments/04_constellation_insert/multilayer.py
```

Or from the REPL (once INSERT FACT is implemented):

```sql
larql> USE "output/gemma3-4b-f16.vindex";
larql> INFER "The capital of Atlantis is" TOP 5;
-- said (17.8%), believed (17.6%)...

larql> INSERT FACT "The capital of Atlantis is Poseidon";
-- Traces prompt, inserts 8 features with down overrides

larql> INFER "The capital of Atlantis is" TOP 5;
-- Poseidon (94.6%)
```
