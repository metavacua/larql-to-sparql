# LARQL Confidence Scoring

## Overview

Every edge extracted by `weight-extract` carries a confidence score derived from the raw logit magnitudes of the FFN feature that produced it. Confidence separates `(France, L26-F9298, Paris)` at 0.89 from `(France, L3-F2041, crawl)` at 0.002.

Extraction is always complete — all edges are stored regardless of confidence. Filtering by confidence happens at query time or as a post-processing step.

## How confidence is computed

Each FFN feature `i` at layer `L` has two projections:

**Input side (W_gate):** `embed @ W_gate.T` — projects the embedding matrix through the gate weights. The top score for feature `i` is `c_in`: how specifically this feature responds to one trigger token vs many. High `c_in` = entity-selective.

**Output side (W_down):** `embed @ W_down` — projects the embedding matrix through the down weights. The top score for feature `i` is `c_out`: how strongly this feature pushes toward one answer token. High `c_out` = strong writer.

**Raw product:** `c_in × c_out` — a feature that fires specifically for "France" AND writes strongly toward "Paris" has a high raw product. A feature that fires vaguely AND writes weakly is noise.

**Per-layer normalization:** After all features in a layer are walked:

```
c = (c_in × c_out) / max(c_in × c_out across this layer)
```

This gives confidence in [0, 1] normalized within each layer.

## Why per-layer normalization

Different layers serve different functions in the transformer:

| Layer range | Role | Signal type |
|---|---|---|
| L0–L14 | Dark accumulation | Structural, low factual confidence |
| L14–L25 | Relation differentiation | Mixed, relations emerging |
| L26 | Fact explosion | Highest factual confidence |
| L27–L33 | Refinement | Copy, format, consolidation |

A confidence of 0.8 at L26 means "strong factual edge." A confidence of 0.8 at L3 means "strong structural edge." Both are valid but serve different purposes. Per-layer normalization keeps scores comparable within their function. The `layer` field lets you weight across layers at query time.

## Two scores: confidence vs selectivity

Empirical results from Gemma 3-4B show that **confidence and selectivity measure different things:**

| Score | What it measures | Peaks at | Correlates with |
|---|---|---|---|
| `c` (confidence) | Combined signal: `c_in × c_out / max` | Early/mid layers (L6–L12) | Structural edges — function words, syntax |
| `selectivity` | Input specificity: `c_in / max(c_in)` | Late layers (L25–L33) | Factual edges — proper nouns, entities |

Early layers have features that fire broadly (low c_in) but write strongly to common tokens (high c_out). This gives high confidence but low selectivity — these are structural edges ("the", "is", "a").

Late layers have features that fire specifically for entities (high c_in) but write with moderate strength. This gives lower confidence but high selectivity — these are the factual edges you want.

**For factual knowledge:** filter on `selectivity` + late layers.
**For structural analysis:** filter on `confidence` + early layers.

## Edge schema

```json
{
  "s": "France",
  "r": "L26-F9298",
  "o": "Paris",
  "c": 0.89,
  "src": "parametric",
  "meta": {
    "layer": 26,
    "feature": 9298,
    "c_in": 8.7,
    "c_out": 12.4,
    "selectivity": 0.72
  }
}
```

| Field | Description |
|---|---|
| `c` | Normalized confidence [0, 1] — `(c_in × c_out) / max` per layer |
| `selectivity` | Normalized input selectivity [0, 1] — `c_in / max(c_in)` per layer |
| `c_in` | Raw input selectivity (gate projection magnitude) |
| `c_out` | Raw output strength (down projection magnitude) |
| `layer` | Source transformer layer |
| `feature` | Source FFN feature index |

## Filtering at query time

Extraction stores everything. Filtering happens when you load or query:

```rust
// Factual edges: high selectivity at late layers
let factual: Vec<&Edge> = graph.edges()
    .iter()
    .filter(|e| {
        let meta = e.metadata.as_ref().unwrap();
        let layer = meta["layer"].as_u64().unwrap();
        let sel = meta["selectivity"].as_f64().unwrap();
        layer >= 25 && sel >= 0.15
    })
    .collect();
```

## Layer statistics

The `--stats` flag writes per-layer statistics for validation:

```bash
larql weight-extract google/gemma-3-4b-it \
    -o knowledge.larql.json \
    --stats stats.json
```

Stats file contains per-layer:

| Field | Description |
|---|---|
| `mean_confidence` | Average normalized confidence (c_in × c_out) |
| `max_confidence` | Highest confidence edge |
| `mean_selectivity` | Average normalized selectivity (c_in) |
| `max_selectivity` | Highest selectivity edge |
| `mean_c_in` | Average raw input selectivity |
| `mean_c_out` | Average raw output strength |
| `self_loop_count` | Edges where subject == object (identity reinforcement) |
| `self_loop_pct` | Self-loop percentage |
| `top_subjects` | Top 10 subjects by frequency, with avg confidence |
| `top_objects` | Top 10 objects by frequency, with avg confidence |
| `edges_found` | Total edges extracted from this layer |
| `features_scanned` | Number of FFN features walked |

**Validation targets:**
- Factual layers (L25+) should have the highest `mean_selectivity`
- Early layers should have high `self_loop_pct` (identity reinforcement)
- `top_subjects` at factual layers should include proper nouns
- `top_subjects` at early layers should be dominated by function words

## Expected scale

For Gemma 3-4B-IT (34 layers, 10240 features/layer):

| Metric | Approximate value |
|---|---|
| Total edges | ~8M |
| Edges at c >= 0.1 | ~500K–1M |
| Edges at c >= 0.5 | ~30K–50K |
| JSON file (complete) | ~1.5 GB |
| JSON file (c >= 0.1) | ~200 MB |
| MessagePack (complete) | ~700 MB |
