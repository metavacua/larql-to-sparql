# FR1 — top-k fuzzy entity router on a real LARQL vindex: VERDICT

**Date:** 2026-06-07. **Status:** ran (`chris-experiments/larql_probes/examples/fleet_routing/fr1_topk_fuzzy_router.rs` → `bench/aim-validation/fr1_topk_router_gemma3-4b.json`). Gemma-3-4B Q4K vindex, production `KnnStore` cosine-NN router, N=150 real countries, layer sweep {20,22,24,26}, judged in predictive units (recall@k / margin / confident-wrong — mean-cosine banned). Reproduces fleet E15 on the production path. Pre-registration: [`docs/fleet-routing-extensions.md`](../fleet-routing-extensions.md) §FR1.

## Headline

**WIN — but the fix is a calibrated/verified gate at the right layer, not merely "expose top-k."** The entity activation key is real and *stronger than E15's MLP* under plain cosine-NN at the resolve layer (L26 paraphrase top-1 **0.89**, top-5 0.95; cross-relation top-5 **1.00**), and it is a **genuine entity key, not answer-leak** (the cross-relation confound holds at L24/L26). But the live inference path — `query_top1` + a **fixed 0.75 cosine gate** (`infer_patched.rs:162-163`) — is **non-discriminative and brittle**: the gate fires on **150/150 queries at every layer** (near-rank-1 residuals all sit above 0.75 against the template-shaped keys, with razor-thin margins ~0.01), so it provides zero right-vs-wrong discrimination and **injects a confident-wrong fact on 11% of queries at the best layer (L26) and 84% at L20.**

## Results (N=150, chance@5 = 0.033)

| layer | PARA top1 | top3 | top5 | top10 | margin mean | gate@0.75 fires | confident-wrong | CROSS top5 |
|---|---|---|---|---|---|---|---|---|
| L20 | 0.16 | 0.31 | 0.38 | 0.47 | 0.0001 | 150/150 | **126 (84%)** | 0.23 |
| L22 | 0.38 | 0.50 | 0.57 | 0.62 | 0.0002 | 150/150 | 93 (62%) | 0.45 |
| L24 | 0.86 | 0.92 | 0.93 | 0.95 | 0.0030 | 150/150 | 21 (14%) | **1.00** |
| **L26** | **0.89** | 0.94 | 0.95 | 0.97 | 0.0106 | 150/150 | **17 (11%)** | **1.00** |

(Confident-wrong = top-1 cosine > 0.75 **and** wrong entity. Since the gate fires on all 150, this is also the absolute fraction of routed queries that inject a wrong fact.)

## What it establishes (against the pre-committed outcomes)

- **The entity key is real, and the layer is load-bearing.** It *builds* with depth — L20 0.16 → L22 0.38 → L24 0.86 → L26 0.89 — exactly the fleet's `route_sweep` shape. L20/L22 are **phrasing-traps** (paraphrase looks middling but cross-relation collapses to 0.23/0.45 → keying on the template, not the entity); the genuine entity key resolves at **L24-26** (cross-relation 1.00).
- **Cross-relation confound PASSED — genuine entity key, not answer-leak.** Train on `capital`, route a `currency` prompt → top-5 **1.00** at L24/L26 (≥ paraphrase). The answer differs across relations, so the key is the entity. The circular failure mode the design was built to catch did not fire.
- **Cosine-NN on the production residual beats E15's trained MLP** (E15 paraphrase top-1 0.75 / top-5 0.86; here 0.89 / 0.95 at L26). **No router training is needed** — the production `KnnStore` cosine path, captured at the resolve layer, is already a strong key. This is good news for the build: the lever is the *gate*, not a learned router.
- **The live 0.75 gate is the defect, confirmed in predictive units.** It fires 150/150 at every layer (absolute cosine is uninformative — near-rank-1) and the margin is razor-thin (mean 0.011 at L26, ≈E11's ~0.002 order). So the override **cannot tell a correct route from a wrong one by cosine or margin**, and commits a confident-wrong fact at the layer-dependent rate above. This is the indictment FR1 set out to test.

## SEE IT

`"Germany's capital city is"` @L26 top-5 → `[Spain, Germany, Italy, Poland, Ukraine]` — Germany at **rank 2**. A ranked short-list a verifier picks from, not a pinpoint (top-1 = Spain would be the confident-wrong inject under the live path).

## What the build must be (revised from the spec by the measurement)

1. **Move/choose the layer.** Routing at L20 is catastrophic (84% confident-wrong), at L26 acceptable (11%). The KNN install/query layer is decisive — the gate must operate where the entity key has resolved (L24-26 on Gemma-3-4B), and the cross-relation gap is the diagnostic for "is this a real entity layer or a phrasing-trap."
2. **Retire the absolute 0.75 gate.** Absolute cosine clears 0.75 on ~everything (near-rank-1); it is not a confidence signal. Replace with **top-k candidate generation + a verifier** (exact-string match of the candidate entities against the query) and **abstain** when the verifier finds no candidate — top-5 contains the answer 95% of the time, so a verifier catches most of the 11% confident-wrong. A margin/relative gate alone won't do it (margins are razor-thin too).
3. **`query_knn` is the right primitive** — already built (`knn_store.rs:132`), just unused by the override. The increment is wiring it + the verifier into `apply_knn_override`, default off = byte-identical (parity spine).

## Honest scope / caveats

- One model (Gemma-3-4B Q4K), one entity class (countries the model knows), capital/currency relations. Aliases (Persia→Iran) and novel facts are FR2/E14, not tested here.
- The keys are the **TRAIN-phrasing** residuals (`"The capital of {e} is"`), matching how `INSERT … MODE KNN` captures. Paraphrase/cross are held-out phrasings.
- Confident-wrong is measured against the *current* gate (0.75); a verifier-gated build is expected to convert most confident-wrong into abstain (FR1 build) or correct-via-fallback (FR2).
- Margins reported are cos1−cos2; `margin_min` ≈ 0 means at least one near-tie at every layer — the near-rank-1 geometry, not noise.

## Bottom line

The fuzzy entity key is **real, strong, and answer-leak-free at L24-26** — and the production cosine-NN path already delivers it (better than a trained MLP). The defect is entirely in the **consumer**: a fixed-0.75 top-1 gate that fires on everything and injects an 11–84% confident-wrong rate. **Build greenlit**, scoped to: route at the resolved layer, replace the absolute gate with **top-k + verify + abstain**, parity-first. FR2 (symbolic-primary, this as the fuzzy fallback) is the natural wrapper.

**Artifacts:** `chris-experiments/larql_probes/examples/fleet_routing/fr1_topk_fuzzy_router.rs`, `bench/aim-validation/fr1_topk_router_gemma3-4b.json`.

---

## BUILD LANDED (2026-06-07) — top-k + verify + abstain, opt-in, parity-first

`apply_knn_override_verified` (`crates/larql-inference/src/forward/infer_patched.rs`) implements the revised fix and is wired into both forward entry points (`infer_patched`, `infer_patched_q4k`) via `route_knn_override`. **Opt-in** behind `LARQL_KNN_VERIFY` — default off = byte-identical to the legacy `query_top1`+0.75 path (the parity spine; all 14 legacy `apply_knn_override` tests unchanged and green).

**What it does** (per the verdict above):
1. **Resolved-layer-first** — iterates whatever layers the store holds **highest-first** (no hardcoded layer; the resolved layer is model-dependent — ~L24-26 on Gemma-3-4B, derived from the store's own install layers).
2. **Top-k + verify** — among the top-`k` candidates (`LARQL_KNN_TOPK`, default 5), overrides only with a fact whose stored `entity` the prompt names → a cross-entity collision (the confident-wrong case) is rejected; a correct entity at rank 2-5 is still found.
3. **Abstain** — no verified candidate → no override (the model answers).

Env knobs: `LARQL_KNN_VERIFY` (enable), `LARQL_KNN_TOPK` (candidates, default 5), `LARQL_KNN_MIN_COS` (floor, default 0.75 = `KNN_COSINE_THRESHOLD`).

**End-to-end validation** (real Gemma-3-4B vindex, via `larql lql` INSERT MODE KNN + INFER, reproducing FR1's measured collision — Germany's paraphrase routes top-1 to Spain):

| query | LEGACY (top-1+0.75) | VERIFIED (FR1 build) |
|---|---|---|
| "Germany's capital city is" | **SpainX** ❌ (cos 0.90, confident-wrong) | **GermanyX** ✓ (verify picks rank-2 Germany) |
| "Poland's capital city is" | PolandX ✓ | PolandX ✓ (no regression) |

5 new unit tests (`verified_*`) cover: entity-in-prompt overrides, entity-absent abstains (the cos=1.0 confident-wrong fix), top-k rescue of a rank-2 correct entity, resolved-layer-first, below-floor abstain. Clippy clean.

**Scope:** the verifier is entity-**in-prompt** (FR1's exact-entity case). Alias resolution where the prompt does *not* name the canonical entity (Persia→Iran) is FR2's two-tier job — the same machinery with the symbolic tier relaxed. An LQL surface (`ROUTE TOPK k VERIFY`) is a follow-up; the env-gated path is the production MVP.
