# Fleet Routing Extensions (FR1–FR4)

**Spec + frozen pre-registrations. Status: measurement-experiment-first — no
build lands before its falsification gate runs.** Added 2026-06-07.

Four LARQL explorations seeded by the `chris-experiments/fleet` native-store arc
(E10–E17) and the `videos/the-mechanism` build story. The fleet and LARQL's
KNN/COMPOSE are two independent programs that already converged on the same
architecture (`fleet/SYNTHESIS.md` §9); these items port the fleet's *measured*
findings into LARQL's routing/edit surface — but only after re-measuring each on
a real LARQL vindex, in predictive units, with the want-guards below. This is the
Query/Edit/Interpret track (the moat), not the V1–V4 bandwidth compound.

> Discipline carried from the fleet and from `feedback_metric_matches_operation`:
> **mean-P / mean-cosine are banned** as deciding metrics — judge in recall@k,
> NLL, KL, drift, confident-wrong rate. **Falsification is the cheap research
> gate** (run the probe before building). **Parity is the engineering gate** —
> any build lands behind a flag, default off = byte-identical to today, *before*
> any quality/latency number is quoted (the spine).

---

## The mechanism, in one paragraph (why these four)

A transformer's factual memory is addressed by **(relation, entity) → value**,
and the two halves behave oppositely. The **relation is a clean, sharp, semantic
index** — a probe trained on `{capital, currency, language}` reads unseen
synonyms (seat/money/tongue) at 1.000 (`the-mechanism/address.py`). The **entity
is fuzzy** — its own L26 activation is a usable content-addressable key but only
at **re-ranker grade**: top-1 ≈ 0.7, top-5 ≈ 0.9, with a razor-thin top-1 margin
(E15: `fleet/E15_real_routing/`). The model **does not unpack** a constructed
packed channel (reads at chance — the de-mix wall, `the-mechanism/wall.py`); the
separation happens at *write* time, upstream. And the **operations** over those
facts split too: anything that factors through a linear aggregate (count /
threshold / majority) rides the read for free; joint-bit operations (parity, and
— conjecturally — argmin/distance/optimization) wall (E17:
`fleet/E17_compute_ladder/`). FR1–FR4 are these four facts, made operational.

| # | Discovery | LARQL exploration | Fleet anchor |
|---|---|---|---|
| FR1 | Entity key is top-k fuzzy, not top-1 exact | Top-k candidate router + verifier/abstain | E15, E16 |
| FR2 | Symbolic-primary, activation-fuzzy-fallback | Two-tier router sequenced into inference | E16 |
| FR3 | Relation is a clean semantic index | Synonym-robust relation addressing | mechanism `address.py`, E10 |
| FR4 | Compute splits at linear-aggregate vs joint | Operation-class dispatch boundary | E17 (conjecture) |

Coupling: **FR1 ⊂ FR2** (the top-k fuzzy tier *is* the fallback half of the
two-tier router). FR3 is orthogonal and the cleanest standalone win. FR4 is the
most speculative — gated on FR1–FR3 and on closing E17's own open conjecture.

---

## FR1 — Top-k fuzzy entity router + verifier

**Question.** LARQL's KNN override currently routes on **top-1 cosine above a
fixed 0.75 gate**. The fleet measured that exact path as brittle (E11/E15:
near-rank-1, top-1 ≈ 0.7, margin ~0.002). Does a **top-k candidate list + a cheap
verifier** recover the accuracy the top-1 gate leaves on the table — on a *real
LARQL vindex* — and does the current top-1 gate measurably inject confident-wrong
facts?

**Current LARQL state (source-verified).**
- `KnnStore::query_top1(layer, residual)` and `query_knn(layer, residual, k)`
  both exist: `crates/larql-vindex/src/patch/knn_store.rs:121,132`.
- The inference override only consumes **`query_top1`** + `KNN_COSINE_THRESHOLD =
  0.75`: `crates/larql-inference/src/forward/infer_patched.rs:150-164`. **`query_knn`
  is unused at inference** — the top-k machinery is built and idle.

**The measurement (run first; predictive units).** Install N entity facts into
the KnnStore (`INSERT … MODE KNN`) on Gemma-3-4B; query with **held-out
paraphrases**; report:
- **recall@k for k ∈ {1,3,5,10}** (the E15 reproduction — expect top-1 ≈ 0.7,
  top-5 ≈ 0.9).
- **top-1 margin distribution** (cos₁ − cos₂) — the razor-thin-margin claim.
- **confident-wrong rate** = fraction where cos₁ > 0.75 (fires the current gate)
  but the entity is wrong. This is the indictment of the status-quo path.
- **cross-relation confound** (the E15 firewall): install `capital`, query
  `currency` → does the key still route to the entity? If cross-relation collapses
  while paraphrase succeeds, the "key" is answer-leak, not an entity key.

**Want-guards.** The want is "top-k rescues routing." A WIN requires top-5 ≫
top-1 **and** a non-trivial confident-wrong rate at the 0.75 gate (proving today's
path is brittle) **and** cross-relation holds (≥ chance×large). If top-1 is
already ≥0.9 with wide margins on real vindex, the fuzzy tier is unnecessary —
**report that as FR1 falsified**, keep top-1. mean-cosine is not a deciding metric.

**Pre-committed outcomes.**
- *WIN* → build the top-k override + verifier.
- *Already-exact* (top-1 ≥0.9, wide margin) → no build; the 0.75 gate stands.
- *Answer-leak* (cross-relation collapses) → the activation key is circular; FR1
  reduces to FR2's symbolic tier only.

**Build (parity-gated) — ✅ BUILT 2026-06-07.** `apply_knn_override_verified`
(`crates/larql-inference/src/forward/infer_patched.rs`): top-k candidates
(`LARQL_KNN_TOPK`, default 5) + **entity-in-prompt verifier** + **abstain**,
**resolved-layer-first** (highest stored layer; no hardcoded index — the
resolved layer is model-dependent). Wired into `infer_patched`/`infer_patched_q4k`
via `route_knn_override`, **opt-in behind `LARQL_KNN_VERIFY`** — default off =
byte-identical (14 legacy tests green). End-to-end on real Gemma-3-4B: legacy
"Germany's capital city is" → SpainX (confident-wrong); verified → GermanyX
(fixed), Poland correct in both (no regression). 5 unit tests, clippy clean.
**LQL surface landed:** first-class `INFER … ROUTE VERIFY [FALLBACK] [TOPK n]`
(`KnnRouteMode` threaded through `infer_patched`, default `Legacy` =
byte-identical; env vars set the default when no clause). Detail:
[`docs/diagnoses/fr1-topk-fuzzy-router.md`](diagnoses/fr1-topk-fuzzy-router.md)
§"BUILD LANDED".

**Crates:** larql-vindex (`patch/knn_store.rs`), larql-inference
(`forward/infer_patched.rs`), larql-lql (grammar + executor).

---

## FR2 — Two-tier router: symbolic-primary → activation-fuzzy fallback

**Question.** E16 assembled the resolved architecture: **exact entity-string
lookup as the primary key** (precision 1.0), with the FR1 activation top-k as the
**alias/paraphrase fallback** for queries exact-match can't reach (Persia→Iran,
Siam→Thailand). Sequenced this way, does LARQL recover aliases it currently
misses — at a bounded confident-wrong cost?

**Current LARQL state.** The symbolic tier exists as *data* but is not sequenced
into routing: `KnnStore::entries_for_entity(entity)` is an exact, case-insensitive
entity-string lookup (`knn_store.rs:172`) — used for **DESCRIBE**, not as a
primary router. Inference routes purely by residual cosine (FR1). So the two-tier
*sequencing* is the gap, not the pieces.

**The measurement (run first).** Build a held-out set with three slices (the E16
shape): **exact names**, **aliases** (canonical string absent), **novel** (model
can't know). Report per slice:
- **symbolic-only recall** (expect 1.0 exact, 0.0 alias).
- **activation-fallback recovery on the alias slice** (E16: 10/10 on *famous*
  aliases; the honest general rate is E15's 0.7 top-1 / 0.9 top-5 — do not read
  the easy slice as the general number).
- **confident-wrong rate on the alias slice** (E16's load-bearing caveat:
  mis-routes inject a confident-wrong fact; abstain/verify is the mitigation).

**Want-guards.** A WIN = alias recovery > 0 that symbolic-only cannot reach,
**with** confident-wrong bounded by the FR1 verifier/abstain. Quote the *general*
fuzzy rate, not the famous-alias slice. Symbolic stays primary — "don't route
what the model already knows for free" (V1/E16: base owns the known region).

**Build (parity-gated) — ✅ BUILT 2026-06-07.** `apply_knn_override_two_tier`
(`crates/larql-inference/src/forward/infer_patched.rs`): tier 1 = the FR1
verify (symbolic-primary, entity-in-prompt), tier 2 = top-1 activation fallback
when tier 1 abstains (alias case). Opt-in `LARQL_KNN_VERIFY` + `LARQL_KNN_FALLBACK`,
default off = byte-identical. E2E real Gemma-3-4B: "capital of Persia" →
verify-only abstains (model says Tehran) → two-tier recovers IranX (cos 0.97);
"capital of Iran" → IranX both (no regression). 4 unit tests, clippy clean.
Honest caveat: tier 2 is the fuzzy ~0.7-0.9 route (mis-routes inject a wrong
fact), fires only when verify missed. Detail:
[`docs/diagnoses/fr2-two-tier-router.md`](diagnoses/fr2-two-tier-router.md)
§"BUILD LANDED".

Original plan: An explicit dispatch order in the override:
exact-string (`entries_for_entity`) → FR1 activation top-k → verify → else
abstain. **Parity:** with the fallback disabled, behavior reduces to FR1's parity
case. Pairs with the existing `MODE KNN` / `MODE COMPOSE` install modes.

**Crates:** larql-inference, larql-vindex, larql-lql. (Depends on FR1.)

---

## FR3 — Relation as a clean semantic address

**Question.** The relation half of the address is *clean* where the entity is
fuzzy: a relation probe generalizes to unseen synonyms (seat=capital,
money=currency, tongue=language) at 1.000 (`the-mechanism/address.py`; E10's
native COMPOSE read fires the model's own FFN slot). Can LARQL **address by
relation semantically** — synonym/alias-robust — instead of by exact relation
string, and measure the relation-vs-entity sharpness asymmetry directly on a real
vindex?

**Current LARQL state.** `RelationClassifier` (`crates/larql-lql/src/relations.rs`)
already classifies edges into relation types via discovered clusters + embedding-
direction heuristics, for DESCRIBE. It is a typing/display tool, not yet a
routing/addressing key.

**The measurement (run first).**
- **Relation synonym generalization** — derive/reuse a relation probe, train on a
  few relation words, test held-out synonyms → reproduce ≈1.000 (the clean-index
  claim), scoped honestly to the relations tested (the video is explicit: "not a
  law about all of them").
- **The asymmetry** — on the *same* vindex, relation top-1 accuracy (sharp) vs
  entity top-1 accuracy (fuzzy, FR1). The headline is the contrast: address the
  relation by index, the entity by top-k + rank.

**Want-guards.** Scope to measured relations; report which relations are clean and
which aren't (don't generalize the handful). Don't let a clean relation index
imply a clean entity index — that's the FR1 fuzziness, kept distinct.

**Build — ✅ BUILT 2026-06-07.** `RelationResolver`
(`crates/larql-lql/src/executor/relation_resolver.rs`): a **trained softmax
probe** (not string/cosine — residuals are near-rank-1, the "proxy is not the
thing" trap) on per-relation residual keys at a model-agnostic probe layer
(`round(0.3·num_layers)`). Wired into `SELECT … FROM EDGES WHERE relation = …`
as a fallback when exact-string matches nothing; resolves the word by meaning,
re-runs against the canonical relation, prints a note. Cached per vindex
(one-time build). E2E real Gemma-3-4B: `WHERE relation = "seat"` → resolved to
"capital", returned capital edges. 2 unit tests, 717 lql tests green, clippy
clean. Detail: [`docs/diagnoses/fr3-relation-address.md`](diagnoses/fr3-relation-address.md)
§"BUILD LANDED".

Original plan: Synonym-robust relation addressing in DESCRIBE/SELECT
(route by relation meaning, not exact string), building on `RelationClassifier`.
The video coda's literal reading — `INSERT INTO EDGES (entity, relation, target)
MODE COMPOSE` as "write at the relation's address" — is the edit-side twin;
COMPOSE already writes a rank-1 slot, so the increment is *resolving the relation
semantically* at install + read.

**Crates:** larql-lql (`relations.rs`, executor), larql-vindex (cluster/probe
artifacts).

> Guardrail carried from E10 / the mechanism video: **read hundreds, write a
> dozen.** Native COMPOSE writes rank-cap at ~10–14 new entities/relation
> (orthogonality runs out after decoy-refine). Past that, spill to external
> (FR2's symbolic store), same addressing. A future FR3b could *enforce/warn* the
> per-relation cap and auto-spill — tracked, not specced here.

---

## FR4 — Operation-class dispatch boundary (the compute ladder)

**Question.** E17 measured, on one packed store, that operations which **factor
through a linear aggregate** (COUNT, THRESHOLD, MAJORITY) ride the read for free
(L1–L2), while **joint-bit** operations (PARITY) wall — and that the split is a
property of the *operation in a bounded reader*, **not** the packing. Can this
become a dispatch criterion for LQL: compute linear-aggregate operations in-band
over retrieved facts, route joint-nonlinear ones to external compute?

**The honest caveat (load-bearing — from E17's own review ledger).** E17 ran
count/threshold/majority/parity. The E4 internal/external taxonomy's *external*
class is geometric/optimization/temporal — **none of those ran**; parity is a
**stand-in**, and E17 explicitly demotes "E4 is derived" to **a conjecture**
(`E17_VERDICT.md` ledger #2). So FR4 is research-first by necessity.

**Current LARQL state.** `crates/larql-router/src/dispatch.rs` is **grid/expert**
dispatch (routing experts across machines), *not* operation-class dispatch. No
LQL planner classifies aggregate operations by readability today.

**The measurement (run first — this closes E17's open conjecture).** Run the
*actual* external operations (distance, argmin, a small optimization) on the E17
ladder rig (`fleet/E17_compute_ladder/e17_ladder.py`) and test whether they wall
like parity. In parallel, map LQL aggregate verbs to the boundary: linear-
aggregate (COUNT/SUM/AVG/MAX/threshold) vs joint-nonlinear (JOIN/PATH/ARGMIN/
distance). Judge by the same ladder metric (does it clear the bar within the
reader budget), as a *gap*, not a bar-crossing (E17 ledger #3).

**Want-guards.** Do **not** claim E4 is derived. A criterion only earns "real" if
the genuine external ops wall and the linear-aggregate ones ride — on the real
ops, not parity-as-proxy. If the line is operation-specific rather than a clean
linear/joint rule, say so. Packing must not be credited with a tax E17 measured
absent (the √m′ test came back negative).

**Measurement — ✅ RAN 2026-06-07 (conjecture REFINED).** Added the real external
ops (DIST/geometric, ARGMIN/selection, PARTITION/optimization) to the E17 ladder
rig (`fleet/E17_compute_ladder/e17_ladder.py external`). Result, exactly as
pre-registered: **DIST and ARGMIN ride free at L1** (they factor through
per-feature reads + a linear aggregate/static ranking); **only PARTITION (global
subset-sum feasibility) walls like parity** (NO-CLEAR 0.81 @ j=8). So **parity was
NOT a fair stand-in for "external" as a category** — E4's internal/external split
**mis-files the geometric and selection archetypes** (they belong internal). The
real criterion: **factors-through-reads/aggregates (rides) vs global-joint
resolution (walls)**. Verdict: `fleet/E17_compute_ladder/E17_EXTERNAL_VERDICT.md`.

**Consequence for the LARQL dispatch planner:** keep count/filter/aggregate/
threshold/majority **and** distance/selection/argmin **internal** (decode-and-act);
route **globally-joint optimization** (subset-sum/partition/assignment) + parity
**external**. The conjectured boundary is directionally right but re-cut by
measurement.

**Conditional build (far; gated on the measurement + FR1–FR3).** A query planner
that evaluates linear-aggregate LQL ops in-band over retrieved facts and
dispatches joint-nonlinear ops to external compute (e.g. the larql MCP solver).
This is the principled version of E4's hand-drawn internal/external split.

**Crates:** larql-lql (planner), larql-router (`dispatch.rs`), larql-vindex.

---

## Sequencing

1. **FR3 measurement** — cheapest, cleanest, standalone (relation index already
   has `RelationClassifier`).
2. **FR1 measurement** — the headline; reproduces E15 on real vindex and indicts
   the live top-1 gate. Build gated on the result.
3. **FR2** — wraps FR1 as the fuzzy tier behind the symbolic-primary router.
4. **FR4** — research-first; closes E17's open conjecture before any planner work.

Each measurement emits a comparable JSON record (the V0 aim-validation harness
pattern, `bench/aim-validation/`) and a `docs/diagnoses/fr*-*.md` verdict applying
the pre-committed outcome. Builds follow only on a WIN, parity-first.

## References

- `chris-experiments/fleet/SYNTHESIS.md` §9–10 (trilemma resolved; E16 assembled)
- `chris-experiments/fleet/E15_real_routing/` (top-k fuzzy entity key)
- `chris-experiments/fleet/E16_system/` (two-tier router, assembled + benchmarked)
- `chris-experiments/fleet/E17_compute_ladder/` (the compute ladder + its ledger)
- `chris-experiments/videos/the-mechanism/SCRIPT.md` (the build story; coda links
  the addressing to LQL `INSERT INTO EDGES … MODE COMPOSE`)
- LARQL: `docs/training-free-insert.md`, `crates/larql-lql/docs/spec.md`
</content>
</invoke>
