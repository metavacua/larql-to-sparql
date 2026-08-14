# FR2 — two-tier router (symbolic-primary → activation-fuzzy fallback): VERDICT

**Date:** 2026-06-07. **Status:** ran (`chris-experiments/larql_probes/examples/fleet_routing/fr2_two_tier_router.rs` → `bench/aim-validation/fr2_two_tier_router_gemma3-4b.json`). Gemma-3-4B Q4K vindex, store over 115 canonical countries, 10 historical/alternate-name aliases, layers {24,26}. Reproduces fleet E16's alias slice on the production path. Pre-registration: [`docs/fleet-routing-extensions.md`](../fleet-routing-extensions.md) §FR2. Depends on FR1.

## Headline

**WIN — the two-tier router reaches what exact-string routing structurally cannot.** Symbolic exact-match resolves **0/10** aliases (the canonical name is absent from the query — "Persia" ≠ "Iran"); the **activation fallback recovers 10/10 top-1** at both L24 and L26. This is E16's alias result reproduced on LARQL: exact-string is the precise primary, the activation key is the alias/paraphrase fallback.

## Results

| tier | metric | result |
|---|---|---|
| **Symbolic** (`entries_for_entity`, exact string) | aliases resolved | **0/10** (the gap) |
| **Activation fallback** (cosine-NN top-k, L24) | alias top-1 / top-5 | **10/10 / 10/10** |
| **Activation fallback** (cosine-NN top-k, L26) | alias top-1 / top-5 | **10/10 / 10/10** |

Every alias routes to its canonical entity: Persia→Iran, Siam→Thailand, Burma→Myanmar, Ceylon→Sri Lanka, Holland→Netherlands, Britain→United Kingdom, Abyssinia→Ethiopia, Rhodesia→Zimbabwe, Zaire→Congo, Formosa→Taiwan.

## What it establishes

- **The sequencing is the increment, and it pays.** Both pieces exist in LARQL — `entries_for_entity` (exact, `knn_store.rs:172`, used for DESCRIBE) and `query_knn` (fuzzy, FR1) — but inference uses neither in a two-tier order. Sequenced (exact primary → activation fallback), coverage jumps from 0/10 to 10/10 on the alias slice.
- **The fallback earns its keep exactly where exact-match can't reach** (E16's framing): don't run the fuzzy tier on exact names (symbolic is precision-1.0 there); run it only when exact-match misses. The activation key resolves the alias because the model's residual for "The capital of Persia is" sits in the same place as "...of Iran is" — the entity is the same; only the surface string differs.

## Honest caveats (load-bearing — carried from E16/FR1)

- **These are FAMOUS aliases — the EASY end.** 10/10 is not the general fuzzy rate. The general number is FR1's **~0.9 top-5 / 0.7–0.89 top-1**; this slice is small (n=10) and the aliases are high-frequency (the model has strong reps). Do not read "alias 10/10" as the production fuzzy-router rate.
- **gate_wrong = 0 here only because the slice is easy.** On the general case (FR1) the live 0.75 gate is confident-wrong 11% at L26. Mis-routes inject a confident-wrong fact; the **verifier/abstain (FR1) is what bounds that cost** — it is not optional, it just had nothing to catch on this easy slice.
- One model, country entities, capital relation, single-token answers. Multi-token answers are RAG's region (V1).

## What the build is

A two-tier dispatch in the override (and surfaced in LQL): **exact-string (`entries_for_entity`) → FR1 activation top-k at the resolved layer → verify → else abstain.** Parity: with the fallback disabled it reduces to FR1's parity case (default off = byte-identical). This is the assembled E16 router, in Rust, on the production path.

## Bottom line

Exact-string is the precise primary (1.0 on names); the activation key is the alias/paraphrase fallback that recovers what exact-match misses (0/10 → 10/10 on famous aliases, ~0.9 top-5 in general). Sequenced two-tier with the FR1 verifier bounding mis-routes, this is the resolved routing architecture — **build greenlit**, with the general-rate and confident-wrong caveats stated.

**Artifacts:** `chris-experiments/larql_probes/examples/fleet_routing/fr2_two_tier_router.rs`, `bench/aim-validation/fr2_two_tier_router_gemma3-4b.json`.

---

## BUILD LANDED (2026-06-07) — two-tier router, opt-in, parity-first

`apply_knn_override_two_tier` (`crates/larql-inference/src/forward/infer_patched.rs`) extends the FR1 build with a tier-2 alias fallback. Wired into `infer_patched`/`infer_patched_q4k` via `route_knn_override`, **opt-in behind `LARQL_KNN_VERIFY` + `LARQL_KNN_FALLBACK`** (default off = byte-identical; the FR1 verified path and the legacy path are both unchanged).

**The two tiers** (shared helpers `verified_route` / `fallback_route`, resolved-layer-first):
1. **Tier 1 — symbolic-primary (the FR1 verify):** if the prompt names a top-`k` candidate's entity → override (precision-1.0, the confident-wrong fix).
2. **Tier 2 — activation fallback:** if tier 1 abstains (no entity named — the alias/paraphrase case), take the **top-1 activation candidate** above the floor. Recovers aliases exact-string can't, at the honest cost: it is a fuzzy ~0.7-0.9 route with nothing to string-verify against, so a mis-route here injects a wrong fact (the floor is the only guard).

**End-to-end validation** (real Gemma-3-4B, novel facts installed, `IranX` reveals the routed entity):

| query | FR1 verify-only | FR2 two-tier |
|---|---|---|
| "The capital of Iran is" (named) | IranX ✓ | IranX ✓ (no regression) |
| "The capital of Persia is" (alias) | Tehran (tier-1 abstains, model answers) | **IranX** ✓ (tier-2 fallback recovers, cos 0.97) |

4 new unit tests (`two_tier_*`): verify-tier fires when named, fallback recovers alias, both abstain below floor, tier-1 preferred over tier-2. 23 total infer_patched tests green, clippy clean.

**Honest caveat (carried from the measurement):** tier 2 is the fuzzy ~0.7-0.9 route; on a mis-route it injects a confident-wrong fact (the easy famous-alias slice was 10/10, the general case is not). It only fires when symbolic/verify already missed, so it trades coverage for some error — appropriate as an explicit, opt-in fallback, not the default.
