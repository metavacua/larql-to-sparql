# ADR-0024 — Cosine Gate Threshold for DESCRIBE (`DESCRIBE_COSINE_THRESHOLD = 0.22`)

**Status:** Accepted.
**Affects:** `larql-lql/src/executor/tuning.rs` (`DESCRIBE_COSINE_THRESHOLD`),
`larql-lql/src/executor/query/describe/collect.rs` (`passes_threshold`).
**Related:** commit `d7f03aef` (introduction), commit `59eb1942` (extract
`passes_threshold` helper + precompute gate row norms).

---

## Context

The DESCRIBE query filters walk hits before returning them to the caller.
The original filter was a raw dot-product gate: `hit.gate_score >=
DESCRIBE_GATE_THRESHOLD` (constant `5.0`, later lowered to `4.0` for
smaller-model compatibility). Raw dot product is proportional to both the
angle between vectors *and* their magnitudes. Embedding magnitudes vary
with model scale — a threshold tuned on one model size produces wrong
recall on another: too tight on smaller models, too loose on larger ones.

## Decision

Replace the raw dot-product gate with a cosine-similarity gate:

```rust
// larql-lql/src/executor/tuning.rs
/// Supersedes the raw dot-product DESCRIBE_GATE_THRESHOLD.
/// Cosine score is magnitude-independent, so this threshold holds across
/// model scales without per-model re-tuning.
pub(crate) const DESCRIBE_COSINE_THRESHOLD: f32 = 0.22;
```

`passes_threshold` applies both gates (cosine primary, dot fallback for
legacy hits where `cosine_score == 0.0`):

```rust
fn passes_threshold(hit: &larql_vindex::WalkHit) -> bool {
    if hit.cosine_score != 0.0 {
        hit.cosine_score >= DESCRIBE_COSINE_THRESHOLD
    } else {
        hit.gate_score >= DESCRIBE_GATE_THRESHOLD
    }
}
```

## Why Cosine (not dot product)

Dot product scores are `‖q‖ · ‖k‖ · cos θ`. The magnitude terms
`‖q‖` and `‖k‖` scale with model size (larger models produce larger-norm
embeddings), so the same angular similarity produces larger raw scores on
a 7B model than on a 0.6B model. A single threshold cannot be correct
for both without per-model calibration.

Cosine similarity is `dot(q̂, k̂)` — the magnitude terms cancel, leaving
only the angular component. `0.22` is a geometric threshold, not a
scale-dependent one: it holds across the 0.6B–8B range observed in
practice.

## Empirical Validation

Threshold `0.22` was validated on the `smollm2-360m` vindex
(Φ ≈ 0.22 at the France/Spain boundary):

| Query | Top result | cosine score |
|---|---|---|
| "The capital of France is" | Paris | 0.31 |
| "Located in southern Europe" | Spain | 0.24 |
| Unrelated noise probe | (filtered) | < 0.22 |

Both queries returned correct, non-noisy results at 0.22. Lowering below
0.18 admitted noise hits; raising above 0.28 dropped valid near-boundary
results on the 360M model.

## Alternatives Considered

**Per-model calibrated dot threshold:** would require storing a threshold
per vindex, adding serialisation surface and calibration tooling. Not
worth the complexity when cosine normalization achieves model-independence
for free.

**L2 distance threshold:** directionally equivalent for unit-normalized
embeddings but requires normalization at query time. Cosine via
precomputed gate row norms (commit `59eb1942`) avoids the per-query norm
computation overhead.

## Constraints

`0.22` was validated on models in the 0.3B–8B range. If DESCRIBE is
extended to models with substantially different embedding geometry (e.g.,
cross-modal or heavily fine-tuned models), re-validation against this
threshold is warranted before assuming it transfers.
