# FR3b — relation resolution: probe is phrasing-brittle, explicit rewrite wins

**Date:** 2026-06-08. **Status:** ran (`chris-experiments/larql_probes/examples/fleet_routing/fr3_template_ablation.rs`, `chris-experiments/larql_probes/examples/fleet_routing/fr3_explicit_rewrite.rs` → `bench/aim-validation/fr3_{template_ablation,explicit_rewrite}_gemma3-4b.json`). Gemma-3-4B Q4K vindex. Follow-on to [`fr3-relation-address.md`](fr3-relation-address.md) — refines, doesn't overturn, the FR3 WIN.

## Headline

**The FR3 residual probe is synonym-robust but PHRASING-brittle, and an explicit few-shot classification (with a `none` escape) is the fix.** FR3's 1.00 was measured on synonym *words* substituted into one fixed template (`"The {w} of {e} is"`). That generalisation is real — but it does **not** extend to unseen *phrasings* (a different template structure). On a held-out phrasing the probe sits at **chance** at its own probe layer, and diversifying the training templates does **not** rescue it. The model's own answer — asked directly "this word → which relation?" — nails synonyms **and** unseen phrasings (12/12), and a `none` option stops it confident-wronging out-of-domain inputs (distractor false-fires 2/3 → 0/3). This is chris's call ("explicit rewrites unseen phrasings to relevant templates"), measured.

## Results

### Probe vs phrasing (ablation, `fr3_template_ablation.rs`, N=6 entities)
Train BASE `{capital,currency,language}` over the first `k` of the resolver's templates; test SYN `{seat,money,tongue}` on a **held-out** phrasing `"The {r} for {e} would be"` (chance 0.33):

| layer | k=1 | k=2 | k=4 |
|---|---|---|---|
| L6  | 0.39 | 0.33 | **0.83** |
| **L10** (resolver probe layer) | 0.33 | 0.39 | **0.39** |
| L14 | 0.33 | 0.33 | 0.33 |
| L20 | 0.17 | 0.11 | 0.17 |

- At the **resolver's L10**, more templates = **no-op** (0.33→0.39, chance). The multi-template change was **reverted** (4×'d build cost for nothing at the probe layer).
- Signal for an unseen phrasing is **early (L6) and decays with depth** — the opposite of "deeper = more normalised". The model resolves surface form early then consumes the relation representation computing the answer; it does not hold a stable canonical form at depth. (N=6 is coarse — L6's 0.83 is 15/18; the *shape* is the point.)

### Explicit classification (`fr3_explicit_rewrite.rs`, one forward via `predict_kquant`)
Few-shot `word -> relation` over `{capital,currency,language[, none]}`, read top-1:

| candidate set | synonyms | unseen phrasings | distractor false-fires |
|---|---|---|---|
| forced-choice (no escape) | 6/6 | 6/6 | **2/3** (weather→capital, altitude→capital) |
| **+ `none` escape** | **6/6** | **6/6** | **0/3** (banana/weather/altitude → none) |

`head city`/`main city`→capital, `legal tender`/`unit of money`→currency, `spoken language`/`mother tongue`→language — all the phrasings the probe sat at chance on.

## What it establishes

- **Two different generalisation axes.** The probe generalises across synonym *words* (FR3's 1.00, real) but **not** across *phrasings* (chance at L10). "Synonym-robust" ≠ "phrasing-robust"; the original WIN holds for the former only.
- **You can't fix phrasing-brittleness by diversifying the probe's training templates** — measured no-op at the probe layer. The relation signal isn't a phrasing-invariant direction at depth; it's early and transient.
- **Explicit model classification is phrasing-robust** because it uses the LM head + full language understanding, not a thin residual probe — 12/12 including the exact cases the probe missed.
- **Forced-choice is the project's recurring confident-wrong trap** (cf. FR1's 0.75 gate, FR2's fallback). A closed "map X to one of {…}" prompt forces a relation even for `weather`/`altitude`. **The `none` escape is the verify/abstain** — 0/3 once present, 12/12 preserved.

## Honest scope / caveats

- 3 relations, ~12 phrasings, 1 model (Gemma-3-4B Q4K), country entities, N=6 in the ablation (coarse). Strong + consistent, not a law.
- Explicit classify costs a **full forward (lm_head)**; the residual probe is cheaper (partial forward to ~L10, no lm_head). So explicit is the *fallback*, not the default — it earns its cost only when the probe abstains.
- Few-shot prompt + `none` example chosen by hand; not prompt-robustness-swept. A second few-shot frame should reproduce 12/12 + 0/3 before this is load-bearing.

## What the build is (NEXT SESSION — not yet wired)

**Probe-first / explicit-classify-fallback in `resolve_relation_synonym`** (`larql-lql/src/executor/query/select/edges.rs`) — the FR2 two-tier shape, for relation resolution:

1. **Tier 1 — residual probe (unchanged, cheap):** the existing `RelationResolver`. When it clears `MIN_CONFIDENCE`, use it (rides the model's implicit normalisation on canonical synonyms — "implicit happens sometimes").
2. **Tier 2 — explicit classify (on abstain):** few-shot `word -> {relations, none}`, one forward, top-1; accept iff it's a real relation (not `none`).

**Wiring wrinkle (the one real structural choice):** `RelationResolver` only dequantises layers `0..=probe_layer` (≈L10) → it **cannot run lm_head**, so Tier 2 must run via the **Session's already-loaded vindex** (`predict_kquant`/`InferenceWeights`, the same path INFER uses), not the resolver's partial setup. ~30 lines crossing the resolver→session boundary. Add an LQL knob if the explicit pass should be opt-in (it's a full forward per abstain).

Harnesses to lift the prompt/matching from: `chris-experiments/larql_probes/examples/fleet_routing/fr3_explicit_rewrite.rs` (the few-shot frame + `none`-gated accept + prefix-matching over top-k).

## BUILD LANDED (2026-06-09)

**Wired the two-tier resolver into `SELECT … FROM EDGES WHERE relation = …`, opt-in, default off = byte-identical.** When the exact/substring relation match returns nothing, `resolve_relation_synonym` runs Tier 1 (the cached residual probe, unchanged); on probe abstain it falls through (`.or_else`) to **Tier 2 — `resolve_relation_explicit`** (`crates/larql-lql/src/executor/query/select/edges.rs`):

- **Few-shot frame lifted verbatim** from `chris-experiments/larql_probes/examples/fleet_routing/fr3_explicit_rewrite.rs` (`word -> relation` + `music -> none`), one **full forward** via `InferenceWeights::predict_dense` (the INFER path — for a Q4_K vindex this is exactly `predict_kquant`, lm_head included). The resolver's partial `0..=L10` dequant can't run lm_head, so Tier 2 goes through `InferenceWeights`, not the resolver's setup — the one structural wrinkle the pre-registration called out.
- **`none`-gated accept** (`match_relation_top1`, unit-tested): prefix-match top-1 against the candidate relations; `none` / out-of-domain → no match → abstain.
- **Gated by `LARQL_FR3_EXPLICIT`** (full forward + model load per probe-abstain). Default off → SELECT is byte-identical to FR3 (probe-only).

**Refinement forced by the real vindex — frequency-ranked candidates, not alphabetical.** The measurement used a clean 3-relation set; production `gemma3-4b-q4k-v2.vindex` carries **2890** noisy probe labels. Two consequences the clean measurement hid:
1. `relation_labels()` is a `BTreeSet` (**alphabetical**), and both tiers cap at `MAX_RELATIONS=64`. An alphabetical top-64 *keeps* a rare early-alphabet label (`food_animal`) while *dropping* `language` — so "mother tongue" couldn't resolve while "banana" could (exactly backwards). Fixed with `RelationClassifier::relation_labels_ranked(top_n)` — **by feature count, most-common first** — used for Tier 2's prompt enumeration + matching. Now the meaningful relations are always in the candidate set, and the `none` escape strengthens (rare labels fall out).
2. Enumerating all 2890 labels is a ~10K-token prompt; the ranked top-64 keeps it to "one short forward," matching the measured intent.

**E2E on real Gemma-3-4B (`LARQL_FR3_EXPLICIT=1`):**
- `mother tongue` → **`language` by explicit classification (0.97)** — probe abstained, Tier 2 resolved. The headline win, on the exact phrasing the probe sits at chance on.
- `weather` → **abstain** (no resolution) — the `none` escape fires, no confident-wrong.
- Default off: `mother tongue`/`weather` → no resolution (probe-only) — **byte-identical**.

**Honest scope correction.** On the production probe (64-class, its own `"The {r} of {e} is"` template) the Tier-1 probe is **stronger than the 3-class ablation implied** — it resolves `head city`→capital (0.97), `legal tender`→currency (0.80), `altitude`→elevation (0.96) *by meaning* without Tier 2. The ablation's "chance" was specific to a 3-class probe on a held-out *template structure*; many real phrasings still ride Tier 1. Tier 2 is the **safety net for genuine abstains** (e.g. `mother tongue`), and on a rich relation set the `none` escape is necessarily weaker (`banana`→`food_category` — a real, common relation here, a defensible resolution, not the clean-world `none`). Tests: `match_relation_top1` (2) + `relation_labels_ranked` (2), 726 lql lib green, clippy clean.

## Bottom line

FR3's relation address is a clean index **for synonym words within a template**, not a phrasing-invariant one — the probe is the right *cheap tier*, not the whole story. **Explicit model classification with a `none` escape is the robust tier** (12/12 phrasings + synonyms, 0/3 confident-wrong on the clean set). **Built:** the two-tier resolver — probe-first, explicit-`none`-gated fallback over frequency-ranked candidates, opt-in `LARQL_FR3_EXPLICIT`.
