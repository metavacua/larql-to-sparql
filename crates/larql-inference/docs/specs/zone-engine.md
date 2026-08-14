# ZoneEngine — Specification

**Status:** 📝 Outline v0.1 (2026-05-19). Body parts that don't depend
on T4 are filled in; §3.2 zone-map verdict, §9.4 T4 migration step,
and §11 Q1 stay placeholders until the cell-conditional T4 result
lands.
**Audience:** LARQL contributors.
**Scope:** **Top-level inference engine** that composes the forward
pass as a sequence of zones separated by choke points. Each zone is
served by one of three strategies — PREDICT (low-rank transition
map), WALK (per-layer composition via LayerEngine), or CACHE
(template-fixed residual lookup). ZoneEngine is the abstraction the
empirical zone-to-zone result fits; LayerEngine is what runs *inside*
a WALK zone.

**Companion specs:**
- [`state-policy.md`](../../../larql-kv/docs/state-policy.md) — engine identity taxonomy.
  ZoneEngine has its own State Policy triple computed from
  constituents.
- [`engine-state-vs-execution.md`](./engine-state-vs-execution.md) —
  the orthogonal cut. ZoneEngine is on the engine side; choke-point
  configuration is execution-side.
- [`layer-engine.md`](./layer-engine.md) — **the inner per-layer
  composer.** LayerEngine runs *inside* a WALK zone; LayerEngine v0.4
  scopes itself this way. Cross-reference, do not duplicate.
- [`kv-engine-unification.md`](./kv-engine-unification.md) — the
  `KvEngine` trait ZoneEngine's WALK zones compose over.

---

## 1. The diagnosis

> Why "zones" and not "layers"?

- The seven probes falsified within-layer feature selection as a
  tok/s lever (uniform walk, no cheap approximation under
  correctness K).
- The four transitions (T1, T2, T3, T4) falsify the layer as the
  unit of useful skipping. The useful unit is the *transition
  between choke points*: rank-30 linear map at median cosine 0.99
  replaces 15 layers with one matvec at 83.8% top-1 (T2). The map
  doesn't enter the skipped layers; it bypasses them entirely.
- LayerEngine v0.3 assumed per-layer dispatch as the top-level
  abstraction. The choke-to-choke result doesn't fit that. ZoneEngine
  is the abstraction that fits; LayerEngine v0.4 (this spec's
  companion) scopes itself to inner-WALK composition.
- This **subsumes** LayerEngine: per-layer composition is what runs
  inside a WALK zone, where the walk is irreducible (T3 — falsified
  at 41.2%, exactly as predicted).

---

## 2. ZoneEngine is itself an engine

### 2.1 Canonical state

Union of (a) choke-point residuals at zone boundaries (the inputs to
PREDICT zones and inputs/outputs of CACHE zones) and (b) canonical
state of the LayerEngine running inside each WALK zone.

### 2.2 Derivative state

- Trained transition maps `M_i: r_choke_in → r_choke_out` (PREDICT
  zones).
- Per-template residual cache (CACHE zones).
- Inherited derivative state from WALK zones' LayerEngines.

### 2.3 Correctness contract

Meet across zones under the State Policy lattice. The likely
whole-engine contract for a non-trivial composition is the conjunction
of bounds on PREDICT/CACHE zones with the weakest WALK contract.

**PREDICT zones declare both `top1_preserving(τ)` AND `bounded_KL(ε)`**
— top-1 alone is not enough (cf. §4.3, §11 Q6). The composite
contract for ZoneEngine is the conjunction of all zones' bounds.

WALK zones may be `exact_logits` (dense LayerEngine), `bounded_KL`
(quantised), or `confidence_gated(τ)` (compiled-FFN inside the walk,
once a working CompiledLookup design exists). Whichever is weakest
dominates.

### 2.4 Fallback mode

Per-zone, composed. PREDICT zones may fall back to WALK on
calibration-failure prompts. CACHE zones fall back to WALK on
template miss — this is what `CachedLayerGraph` already does. There
is no ZoneEngine-level fallback orchestrator.

---

## 3. Zone taxonomy

### 3.1 The three zone kinds

CACHE and PREDICT are conceptually related (a CACHE is a degenerate
PREDICT with a lookup-table "map" — see §6.2) but they are operationally
distinct kinds because their failure modes and validation disciplines
are different:

- **CACHE** has a hit/miss axis with per-prompt observability. Validation
  = (hit rate × in-class KL) + (fallback contract on miss).
- **PREDICT** has an approximation-error axis with no clean
  miss boundary; every prompt runs the map. Validation = aggregate
  KL/top-1 over the corpus with a confidence-margin distribution.

Conflating them is what got `CachedLayerGraph` mislabeled as a
`confidence_gated(τ)` engine in LayerEngine v0.2 (see
[`layer-engine.md` §7.1](./layer-engine.md#71-compiled-ffn-is-aspirational-under-v03)
and the empirical falsification in
`crates/larql-kv/examples/contract_classify_cached_ffn.rs`). Keeping
the kinds separate at §3.1 keeps that discipline at the API boundary
instead of by convention.

| Kind    | What it does                                          | Cost                       | Contract        | Failure mode                |
|---------|-------------------------------------------------------|----------------------------|-----------------|-----------------------------|
| PREDICT | Linear (or low-rank) map from choke-in to choke-out   | ~0.1 ms (rank-30 matvec)   | `top1_preserving(τ) ∧ bounded_KL(ε)` | distributional drift; corpus-level |
| WALK    | Run each layer normally (LayerEngine inside)          | full per-layer FFN cost    | inherits from LayerEngine | per-layer; well-defined |
| CACHE   | Return stored residual if input matches template      | ~5 ms (lookup) on hit; falls back to WALK on miss | `bounded_KL(ε≈0)` on hit, inherited on miss | hit/miss; per-prompt observable |

### 3.2 The zone map (Gemma 3 4B, validated)

Choke points measured 4× independently in the April–May 2026 work.
Validated transition results:

```
Zone 0: [L0]       embedding
T1:     L0  → L4    PREDICT  rank-30, 88.8% top-1  ✅
Zone 1: [L4]       commit choke
T2:     L4  → L20   PREDICT  rank-30, 83.8% top-1  ✅
Zone 2: [L20]      highway end / retrieval start
T3:     L20 → L29   WALK     (41.2% predicted, irreducible by design)
Zone 3: [L29]      retrieval end / format start
T4:     L29 → L33   PREDICT  rank-30, 68.8% top-1  ⏳ pending cell-conditional
Zone 4: [L33]      output
```

T4 verdict drives whether the production ZoneEngine for Gemma 3 4B is
PREDICT-PREDICT-WALK-PREDICT (4-of-4 PREDICT, ~3.6× projection with
kernel work) or PREDICT-PREDICT-WALK-WALK (3-of-4, ~2.5× projection).

> **Falsifiability gate** (cross-arch portability, §11 Q5): the
> choke-point boundaries above are validated by the residual-cosine
> divergence measurement in [the choke-point map experiments — TODO
> link]. Before any non-Gemma-3-4B model can claim a zone map, the same
> measurement must reproduce a four-zone structure with boundaries
> within ±10% of the Gemma 3 4B fractions (commit @ 12%, highway-end @
> 59%, format-start @ 85%); a model whose measurement disagrees by
> more than that **refuses to ship under ZoneEngine** at construction.
> The depth-fraction law is an organizing claim until a second
> architecture reproduces it; the gate above forces falsification
> before it becomes load-bearing in the framework. See
> `feedback_organizing_vs_empirical_claims.md`.

### 3.3 Cross-architecture portability

Depth-fraction law suggests the zone-fraction structure (commit at
~12%, retrieval start at ~59%, format start at ~85%) generalises;
empirical zone maps for Llama 3 / Mistral / Gemma 4 E4B are required
before ZoneEngine ships for those architectures. See §3.2's
falsifiability gate.

---

## 4. The PREDICT strategy

### 4.1 What it is

A trained map `M_i: r_choke_in → r_choke_out` for transition T_i.
Currently rank-30 linear (lowest-rank that beats rank-100 by < noise);
the spec admits higher-rank or non-linear maps if calibration justifies.

### 4.2 Training

- Inputs: residuals at choke layers on a representative prompt corpus.
- Method: linear regression, absolute or delta mode, rank-K SVD
  truncation. Calibration corpus: same as W10 contract-calibration
  corpus (Shannon-bps held-out set).
- Selection: lowest rank where the contract bound (§4.3) is met.
  Today: rank-30 for T1/T2/T4.

### 4.3 Contract calibration — conjunctive

A PREDICT zone declares **both** of these bounds, **measured against
a stated calibration corpus**:

- **τ — top-1 preservation**: `Pr_{x ~ corpus}[argmax(M(x)) == argmax(reference(x))] ≥ τ`.
  Measured per-corpus; T2 currently ships at τ = 0.838.
- **ε — KL bound**: `KL(softmax(reference) || softmax(M)) ≤ ε`
  at a stated quantile (default p95). Forces the engine to declare
  *distribution* preservation, not just argmax.

Both must be on the map's shipping spec sheet. Either alone is a trap:
top-1-only ships fine for greedy decode and breaks silently for beam
search / perplexity eval / RLHF / distillation. KL-only allows a
top-1 swap inside ε. The conjunction catches both classes.

Additional descriptive metrics required for the calibration corpus
(per State Policy §6, predictive units):

- NLL delta on held-out corpus.
- Top-5 / top-20 agreement.
- Confidence-margin distribution on disagreements (T2's known
  finding: Bern dropped from 74.3% to 33.9% on disagreement prompts —
  documented as part of the ε reporting, not hidden).
- First-divergence trace.

The map ships only with the conjunctive (τ, ε) bound demonstrated
under measurement on the named corpus.

> **Note on contract taxonomy**: State Policy v1 doesn't have a
> single named slot for the conjunction `top1_preserving(τ) ∧
> bounded_KL(ε)`. ZoneEngine v0.1 uses the pragmatic form (a pair).
> If a second PREDICT-style engine wants the same shape, promote
> the conjunction to a `predicted(τ, ε)` slot in state-policy.md
> §2.3 — until then, the pair is fine. Open question §11 Q7.

### 4.4 When PREDICT is permitted

- Source-choke residual is a valid input under the map's training
  distribution (the canonical-state precondition).
- Single-token prefill or compatible decode mode (multi-token decode
  needs the K/V-skip story below).
- The model architecture matches the training architecture
  (zone-fraction structure depends on the choke-point map).

### 4.5 K/V cache interaction — Standard+PREDICT refused at construction

Skipping layers means K/V at skipped layers is not appended.
State Policy classifies this:

- For `MarkovResidualEngine` running inside adjacent WALK zones:
  K/V is derivative, recomputable from the choke-point residual.
  PREDICT zones compose cleanly with MarkovResidual.
- For `StandardEngine`: K/V is canonical; PREDICT-skipped layers
  create K/V gaps that future attention at downstream WALK zones
  can't span. **ZoneEngine refuses the configuration at
  construction.**

This is the State Policy `execution_requirements` check in action.
It's the same conservative refusal pattern as
[engine-state-vs-execution.md §3.4](./engine-state-vs-execution.md#34-engine-to-executor-refusal-contract).

We considered engineering "PREDICT zones recompute K/V on demand
from the choke residual" to relax the constraint. Two paths:

- **Invert the rank-K map** to reconstruct intermediate residuals —
  infeasible; rank-30 throws away information that K/V projection
  needs.
- **Train a parallel K/V predictor per skipped layer** — possible
  but its own contract, its own calibration, doubles training work,
  and the predictor must be jointly calibrated with the residual
  predictor.

The conservative refusal is the right call for v0.1. Migration
implication: **WALK zones must run MarkovResidualEngine** for
ZoneEngine to compose PREDICT zones around them. W10 already
demonstrated MarkovResidual matches Standard's tok/s ceiling
(markov-rs None = 106.8 vs Standard ~100), so the switch is a
non-regression on the WALK side. See §9.1.5.

---

## 5. The WALK strategy

### 5.1 What it is

The irreducible walk through layers that PREDICT can't cover. Inside
a WALK zone, ZoneEngine delegates to a [LayerEngine](./layer-engine.md)
which selects per-layer `(KvEngine_L, FfnBackend_L, LayerGraph_L)`
triples.

### 5.2 LayerEngine inside ZoneEngine

The full LayerEngine surface from `layer-engine.md` (v0.4) applies
inside a WALK zone. The zone defines the layer range; LayerEngine
defines per-layer composition within that range.

The production T3 WALK at L20–L29 is where the `walk-ffn`,
`compiled-ffn`, `virtual-experts` LayerEngine compositions earn
their keep. Kernel-quality work on those 9 layers is the 1.5–1.7×
lever that drops T3 from 64ms to ~40ms FFN.

### 5.3 Contract pass-through

The WALK zone's contract is whatever its LayerEngine declares. No
modification at the zone level.

---

## 6. The CACHE strategy

### 6.1 What it is

The existing `CachedLayerGraph` path: when the input residual
matches a known template (template-fixed prefix on a fixed prompt
class), return the cached output residual without re-running the
layers.

### 6.2 Relation to PREDICT (conceptual, not operational)

CACHE is PREDICT with a degenerate map: the "map" is a lookup table
on the canonical-state input, and the "output" is a memoised exact
residual rather than a learned approximation.

This is why the existing `predict_honest` production path (L0–12
cached + L13–33 walk + GPU logits) is a *special case* of ZoneEngine
with a single CACHE zone followed by a single WALK zone. Lifting
`predict_honest` under ZoneEngine is the migration's bit-parity
floor (§9.1).

The conceptual subsumption is real but **the kinds stay separate in
§3.1**. Validation disciplines diverge (hit/miss + memoisation vs
corpus-aggregate approximation), failure modes diverge (per-prompt
observable vs distributional), and conflating them is what produced
the v0.2 LayerEngine over-claim. The §6.2 framing is a prose hook,
not an API unification.

### 6.3 Contract

`bounded_KL(ε ≈ 0)` on cache hit (memoised exact value, modulo
numerical drift between cache-build path and live path; see
`crates/larql-inference/src/layer_graph/cached.rs` doc-comment and
the contract_classify_cached_ffn example for measured drift). Falls
back to WALK on miss. Hit rate is calibrated per prompt class.

`CachedLayerGraph` as currently implemented in tree is a **per-prompt
memoization** — it does not generalize across same-template prompts
(empirically falsified, 2026-05-19; cf.
[`layer-engine.md` §7.1](./layer-engine.md#71-compiled-ffn-is-aspirational-under-v03)).
ZoneEngine's CACHE zone inherits that limitation: shipping with
non-trivial cache requires either a per-prompt build (rare in
production) or a real cross-prompt cache design that doesn't yet
exist.

---

## 7. Composition rules

### 7.1 Zone sequence

A ZoneEngine is constructed from an ordered sequence of zones, each
tagged with its strategy and (for WALK) its inner LayerEngine.

```text
ZonePolicy {
    choke_points: Vec<usize>,            // layer indices
    zones:        Vec<ZoneStrategy>,     // length = choke_points.len() - 1
}
```

### 7.2 Contract aggregation

Whole-engine contract = conjunction of per-zone contracts.

- τ aggregates as `min` across zones (worst-case top-1 preservation).
- ε aggregates as `max` across zones (worst-case KL bound).
- A WALK zone whose LayerEngine is `confidence_gated(τ_L)` forces the
  whole ZoneEngine to `confidence_gated(min(τ_L, τ_P))` plus any
  PREDICT-zone ε bound. The conjunction is unambiguous; if it's
  unreadable in the table, the engine's calibration is too noisy to
  ship.

### 7.3 K/V / state composition across zones

State Policy `execution_requirements` aggregation across zone
boundaries. PREDICT and CACHE zones must declare what they need from
the prior zone's output state (typically: residual only, no K/V)
and what they hand to the next zone's input state.

If any PREDICT or CACHE zone appears in the policy, every WALK zone
must use `MarkovResidualEngine` (or another engine with derivative
K/V). The §4.5 refusal fires at construction otherwise.

### 7.4 The composition correctness floor

A ZoneEngine where every zone is a WALK over a single layer, with
`exact_logits` LayerEngines, produces bit-identical logits to the
reference Standard decode path. This is the floor test that catches
ZoneEngine framing bugs independent of any PREDICT/CACHE behaviour.

---

## 8. Configuration

- `ZonePolicy::single_walk(LayerEngine)` — convenience: one WALK zone
  covering all layers. Reproduces LayerEngine standalone (and replaces
  what LayerEngine v0.3 called "top-level LayerEngine").
- `ZonePolicy::predict_honest()` — convenience: the existing
  production composition (CACHE L0–12 + WALK L13–33). Bit-parity
  floor for the migration (§9.1).
- `ZonePolicy::full_choke_to_choke(...)` — the production target:
  PREDICT T1, PREDICT T2, WALK T3 (LayerEngine), PREDICT-or-WALK T4
  (cell-conditional verdict).
- Custom: arbitrary zone sequence supplied by caller, validated at
  construction.

Construction validates:

1. Choke points are monotonic and cover `[0, n_layers]`.
2. Each zone strategy is compatible with adjacent zones' state
   requirements.
3. Each PREDICT zone has a trained map with a calibrated `(τ, ε)`
   pair on a named corpus.
4. Each WALK zone's LayerEngine passes its own construction checks.
5. If any PREDICT/CACHE zone is present, every WALK zone uses
   MarkovResidual (or a derivative-K/V engine); §4.5 refusal otherwise.
6. The aggregated contract is computable; the resulting `(τ, ε)` pair
   is the engine's spec sheet entry.

---

## 9. Migration plan

### 9.1 Step 1 — Lift `predict_honest` under `ZonePolicy::predict_honest()`

Construction-only; the existing code path remains. Validation:
bit-parity between ZoneEngine dispatch and the direct `predict_honest`
call. This is the migration's bit-parity floor — if it fails,
ZoneEngine has framing bugs independent of any PREDICT/CACHE work.

### 9.1.5 Step 1.5 — Switch WALK zones from `StandardEngine` to `MarkovResidualEngine`

**Prerequisite for §9.3.** PREDICT zones refuse Standard at
construction (§4.5). The existing production `predict_honest`'s WALK
runs Standard; the W10 measurement showed MarkovResidual under None
mask matches Standard's tok/s ceiling (markov-rs None = 106.8 vs
Standard ~100 on the same machine, with full parity from
`crates/larql-kv/examples/w10_parity_gate.rs`).

Land the switch as its own step so PREDICT-zone work (§9.3) doesn't
also carry the "swap the engine underneath" risk. Validation:
the bench-harness dispatch parity oracle plus the W10 parity gate
both green on the new default. Bit-parity preserved.

### 9.2 Step 2 — Add LayerEngine inside WALK zones

LayerEngine v0.4 lands as the inner composition for WALK zones. The
existing T3 walk becomes `LayerEngine::uniform(MarkovResidual, WalkFfn,
DenseGraph)`. Bit-parity preserved against the step-1.5 baseline.

### 9.3 Step 3 — Train T1, T2 maps and add PREDICT zones

PREDICT zones land behind a feature gate. Calibration corpus identical
to W10's. Shannon-bps measurement plus the conjunctive `(τ, ε)` bound
gates promotion to default. Per-map artifacts published; see the
forthcoming `transition-map-training.md` for the artifact format.

### 9.4 Step 4 — Cell-conditional T4 or T4 WALK

Pending the cell-conditional T4 result. If T4 promotes, ZonePolicy
is PREDICT-PREDICT-WALK-PREDICT; if not, WALK at T4 with a
LayerEngine that earns the per-layer kernel-quality work. Body
finalised after the verdict.

### 9.5 Step 5 — Production target measurement

End-to-end vs Ollama, with full Shannon-bps backing. The 2–3.6×
projection earns its place by measurement or it doesn't ship. §10
validation discipline.

### 9.6 Step 6 — Server wiring (deferred to ComputeBackend redesign)

Same constraint as KvEngine unification §10.6 and LayerEngine §9.
Server stays on `generate_streaming` until ComputeBackend lands.

---

## 10. Validation strategy

Three levels, mirroring State Policy §6 and LayerEngine §8:

1. **Composition floor.** `ZonePolicy::predict_honest()` must
   bit-match the existing `predict_honest` production path.
2. **Contract-claim validation.** Any PREDICT zone shipping with a
   `(τ, ε)` claim must back both halves with Shannon-bps measurement
   on the held-out corpus. T1, T2, T4 maps each carry their own
   measurement. CACHE zones similarly: hit rate, in-class KL,
   fallback contract on miss.
3. **Tok/s ceiling.** A ZonePolicy is accepted only if (a) contract
   holds and (b) tok/s is non-decreasing vs the WALK-only baseline
   for the equivalent zone span. The PCA-90 inversion warning applies
   in full: cosine on hidden state is not sufficient evidence; KL on
   the output distribution is. Confidence margins reported alongside
   top-1 in the (τ, ε) shipping sheet.

---

## 11. Non-goals and open questions

### Non-goals

- Not a replacement for LayerEngine; **subsumes** it as an inner
  composer for WALK zones (cf. LayerEngine v0.4 scope migration §12).
- Not adaptive zone selection (content-dependent zone boundaries are
  out of scope for v0).
- Not a multi-token decode story for PREDICT zones that uses
  StandardEngine at WALK; §4.5 refuses that composition. PREDICT
  zones around MarkovResidual WALK zones support multi-token decode.
- Not a graph-walk engine; LARQL Mode 5 / context graph remains
  above the forward pass.

### Open questions

1. **T4 cell-conditional verdict.** Drives §3.2 zone map and §9.4
   migration step. Pending experiment.
2. **Contract lattice incomparability.** Same as LayerEngine v0.4 §10
   Q2; the conjunction of `(τ, ε)` is what ZoneEngine uses, but the
   State Policy lattice does not yet have a named slot. Open: is the
   conjunction a new lattice element or two independent contracts the
   engine carries?
3. **Multi-token decode through PREDICT zones with Standard.** §4.5
   currently refuses. A K/V predictor (separate low-rank map per
   skipped layer) is engineerable in principle — its own contract,
   doubled training work. Worth revisiting only if a non-MarkovResidual
   surround-WALK case becomes load-bearing.
4. **T3 refinement.** T3 falsified the linear map (41.2%) but cosine
   was still 0.96 — the residual is in the right neighbourhood, just
   not the right point. A non-linear map or a stage-wise WALK at T3
   may reduce the irreducible cost further. Not for v0; flag for
   follow-up.
5. **Cross-architecture portability of zone maps.** §3.2 falsifiability
   gate makes this concrete: each architecture must reproduce the
   four-zone structure within ±10% of Gemma 3 4B fractions before
   shipping. Llama 3 / Mistral / Gemma 4 E4B measurements pending.
6. **Confidence drops on PREDICT zones.** T2 preserves top-1 at 83.8%
   but compresses the probability distribution (Bern at 33.9% vs
   dense 74.3%). Resolved at the spec level by §4.3's conjunctive
   contract: PREDICT zones declare *both* `τ` and `ε`. The empirical
   question of whether (0.838, ε_T2) is acceptable for the production
   target is a §10 measurement question — the contract doesn't hide
   the trade-off.
7. **Promote `(τ, ε)` to a State Policy lattice slot?** ZoneEngine
   v0.1 uses the pragmatic pair. If a second PREDICT-style engine
   wants the same shape, promote the conjunction to a `predicted(τ,
   ε)` named slot in state-policy.md §2.3. Until then, the pair is
   fine.

---

## 12. Cross-references

- [`state-policy.md`](../../../larql-kv/docs/state-policy.md) — engine identity taxonomy.
- [`engine-state-vs-execution.md`](./engine-state-vs-execution.md) —
  the orthogonal cut; §3.4's refusal contract is the template for
  §4.5's Standard+PREDICT refusal.
- [`layer-engine.md`](./layer-engine.md) v0.4 — the inner per-layer
  composer for WALK zones. Subsumed by this spec at the top level.
- [`kv-engine-unification.md`](./kv-engine-unification.md) §4.4 —
  the W10 mask cascade WALK zones ride.
- [`markov-residual-engine.md`](./markov-residual-engine.md) §14 —
  the engine WALK zones must use when PREDICT/CACHE zones are
  present in the same ZonePolicy.
- [`compiled-ffn.md`](./compiled-ffn.md) — *to be written.* The
  CompiledLookup engine spec; needs a working design before any
  CACHE-like LayerEngine constituent (currently aspirational; cf.
  LayerEngine v0.4 §7.1).
- *Forthcoming:* `transition-map-training.md` — how PREDICT maps are
  trained, calibrated, and shipped. The artifact spec for T1/T2/T4.
- `crates/larql-kv/examples/contract_classify_cached_ffn.rs` —
  reproducible falsification of the v0.2 LayerEngine `compiled-ffn`
  over-claim; constrains §3.1 (CACHE/PREDICT separation) and §6.3
  (CACHE limitations).
- `crates/larql-kv/examples/w10_parity_gate.rs` — bit-identical
  parity proof for `MarkovResidualEngine` under W10 mask cascade;
  unblocks §9.1.5.
