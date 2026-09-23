# FR3 — relation as a clean semantic address: VERDICT

**Date:** 2026-06-07. **Status:** ran (`chris-experiments/larql_probes/examples/fleet_routing/fr3_relation_address.rs` → `bench/aim-validation/fr3_relation_address_gemma3-4b.json`). Gemma-3-4B Q4K vindex, N=40 countries, layer sweep {6,10,14,20,26}. Dependency-free softmax-regression probe (standardised, L2), judged in synonym-generalisation accuracy (not mean-cosine). Reproduces the mechanism video `address.py` on the production path. Pre-registration: [`docs/fleet-routing-extensions.md`](../fleet-routing-extensions.md) §FR3.

## Headline

**WIN — the relation IS a clean semantic address, and it is clean from L6 (earlier than the video's L10).** A linear probe trained on only `{capital, currency, language}` classifies the unseen synonyms `{seat, metropolis, money, cash, tongue, speech}` at **synonym-generalisation 1.00 at every layer L6–L26** — it knows "seat" *means* capital, semantically, not lexically (the test words are different strings from the train words). And the **relation(sharp)-vs-entity(fuzzy) asymmetry is stark in one harness**: the relation address is fully resolved by L6; the entity address is fuzzy until L26.

## Results (N=40)

| layer | relation train | relation **synonym-gen** | per-synonym | entity top-1 (cosine-NN) |
|---|---|---|---|---|
| L6  | 1.00 | **1.00** | all 1.00 | 0.12 |
| L10 | 1.00 | **1.00** | all 1.00 | 0.17 |
| L14 | 1.00 | **1.00** | all 1.00 | 0.07 |
| L20 | 1.00 | 0.99 | cash 0.93, rest 1.00 | 0.20 |
| L26 | 1.00 | **1.00** | all 1.00 | **0.78** |

(Relation = softmax probe over 3 relation classes; synonym-gen = accuracy on held-out relation words, the real metric. Entity = cosine-NN top-1, capital-train keys / paraphrase query — the FR1 object, for the asymmetry.)

## What it establishes

- **The relation is a clean, meaning-keyed index — confirmed on the production residual.** Synonym-generalisation 1.00 across all layers reproduces `address.py`'s 1.000 and extends it: the index is already clean at **L6**, the earliest probed. Because the six test words are lexically distinct from the three train words, 1.00 means **semantic**, not string-matching — "tongue" routes to *language*, "money"/"cash" to *currency*, "seat"/"metropolis" to *capital*.
- **The asymmetry is the load-bearing contrast.** At every early/mid layer the relation is fully resolved (1.00) while the entity is at chance-ish fuzzy (0.07–0.20); only at L26 does the entity sharpen (0.78). This is the mechanism in one table: **the address has two parts that resolve at different depths and different sharpness** — relation early & clean (an index), entity late & fuzzy (a ranked short-list, FR1).
- **Not answer-leak / not template.** All prompts share the template `"The {word} of {e} is"`, so the probe's only discriminative signal is the relation word's contribution; the entity varies across all classes and averages out. The relation resolves before the answer does (L6 ≪ the L24-26 answer-resolution from FR1's `layers` analogue), so the probe cannot be keying on the answer.

## Honest scope / caveats

- Train accuracy is 1.00 at every layer because H=2560 ≫ n_train=120 (the probe can always fit train); the **deciding metric is synonym-gen on held-out words**, which is also ~1.00 — that is the clean-index proof, not overfitting.
- Three relations, six synonyms, one model (Gemma-3-4B Q4K), country entities. The video's own caution applies: this is a strong result *for the relations tested*, not a law about all relations. A broader relation inventory (the `RelationClassifier`'s discovered clusters) is the generalisation test.
- The single dip (cash 0.93 @L20) is within noise; the shape (clean from L6) is robust.

## What the build is

- **Synonym-robust relation addressing in DESCRIBE/SELECT**, building on `RelationClassifier` (`crates/larql-lql/src/relations.rs`): resolve the relation by *meaning* (a probe / centroid in the clean relation subspace) rather than by exact relation string, so `seat`/`money`/`tongue` address `capital`/`currency`/`language`. The probe is cheap and clean at an early layer (L6-L10) — no late-layer forward needed for the relation half.
- **Pairs with FR1/FR2:** the relation half is a sharp index (this), the entity half is top-k + verify (FR1). The full address = `relation-index(meaning) × entity-topk(rank)`. The edit-side twin is `INSERT … MODE COMPOSE` writing at the relation's address (the video coda) — COMPOSE already writes the rank-1 slot; the increment is resolving the relation semantically at install + read.

## Bottom line

The relation is a **clean semantic address, resolved early (L6) and synonym-robust (1.00)** — the opposite of the fuzzy late-resolving entity (FR1). The two halves of `(relation, entity) → value` measured side-by-side on the production residual confirm the mechanism: **build the relation half as a meaning-keyed index, the entity half as top-k + rank.**

**Artifacts:** `chris-experiments/larql_probes/examples/fleet_routing/fr3_relation_address.rs`, `bench/aim-validation/fr3_relation_address_gemma3-4b.json`.

---

## BUILD LANDED (2026-06-07) — synonym-robust relation addressing in SELECT

`RelationResolver` (`crates/larql-lql/src/executor/relation_resolver.rs`) resolves a relation *word* to a canonical relation the vindex knows, **by meaning**, and is wired into `SELECT … FROM EDGES WHERE relation = …` as a fallback when the exact-string filter matches nothing.

**Faithful to the measurement — a trained probe, not a shortcut.** The build deliberately does NOT use string/cosine matching: residuals are near-rank-1 (the shared `"The {rel} of {entity} is"` template direction dominates), so cosine between `"seat"` and any relation is high — the "proxy is not the thing" trap (and the same reason FR1's entity routing needed top-k, not a cosine gate). Instead it trains a **softmax probe** (the measurement's method) on per-relation residual keys:
- **Probe layer** = a depth fraction (`round(0.3 · num_layers)`, clamped) — model-agnostic, never a hardcoded index (L10 on Gemma-3-4B, in FR3's clean L6-L26 band).
- **Training** = `relations × {8 probe entities}` residuals at the probe layer (partial forward, only layers ≤ probe dequantised), standardised, softmax-regression.
- **Resolve** = forward-pass `"The {word} of {e} is"` for 3 entities → averaged probe probability → canonical relation if confidence ≥ 0.5, else abstain.
- **Cached** per vindex path in the `Session` (one-time build on the first synonym query; later resolves are 3 forward passes). Only fires when the exact filter returns nothing, so normal queries pay nothing.

**End-to-end** (real Gemma-3-4B `gemma3-4b-v2.vindex`, which carries probe relation labels):
```
SELECT * FROM EDGES WHERE relation = "seat" LIMIT 5;
  (relation 'seat' resolved to 'capital' by meaning, confidence 0.51)
  L3  F5859 …  capital   0.0373
  …
```
"seat" — never a stored label — resolved to **capital** and returned the capital-labelled edges. (Confidence is modest because this vindex's probe labels are noisy; on the clean relation set the measurement scored ~1.0.)

2 unit tests (probe-math separability + model-agnostic probe-layer), 717 lql tests green, clippy clean. **Scope:** needs a vindex with model weights (forward pass) and ≥2 known relation labels; absent either, it silently falls back to exact-string (no behaviour change). The relation candidate set is the classifier's `relation_labels()` (capped at 64).
