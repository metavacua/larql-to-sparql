# LayerEngine — Specification

**Status:** 📝 Draft v0.4 (2026-05-19). Supersedes v0.3 of the same
day. v0.3 named LayerEngine as the top-level composition seam; v0.4
narrows that scope: **LayerEngine is the per-layer composer that runs
inside a single WALK zone** of [ZoneEngine](./zone-engine.md), which is
the outer choke-to-choke composer. The v0.3 framing was correct that
`PerLayerGraph` is the existing per-layer dispatch seam; what it missed
is that the *useful unit* of skipping work is the transition between
choke points, not the layer. ZoneEngine is the abstraction that fits
that empirical result; LayerEngine remains the right abstraction for
what happens *inside* an irreducible walk (e.g., T3 in the Gemma 3 4B
zone map).

**Audience:** LARQL contributors.
**Scope:** **Inner per-layer dispatch seam** in `larql-inference` that
composes `KvEngine`, `FfnBackend`, and per-layer `LayerGraph` decisions
within a single WALK zone. Owned by ZoneEngine; not itself a top-level
engine.

**Companion specs:**
- [`zone-engine.md`](./zone-engine.md) — **the outer composer.**
  ZoneEngine sequences PREDICT / WALK / CACHE zones between choke
  points; LayerEngine is what serves a WALK zone. Read ZoneEngine
  first for the top-level structure; this spec for the per-layer
  details of WALK.
- [`state-policy.md`](./state-policy.md) — engine identity as
  `(canonical_state, derivative_state, correctness_contract)`. This
  spec inherits that vocabulary.
- [`engine-state-vs-execution.md`](./engine-state-vs-execution.md) —
  the orthogonal cut separating engine identity from execution
  dispatch. LayerEngine sits on the execution side; State Policy
  sits on the engine side. §11 of that spec already documents
  W10's mask cascade as a worked example of this cut; LayerEngine
  rides the same mechanism.
- [`kv-engine-unification.md`](./kv-engine-unification.md) — the
  `KvEngine` trait this spec composes over.

---

## 1. The diagnosis

> **v0.4 reframing**: LayerEngine *is* the right abstraction for
> per-layer composition. The mistake in v0.3 was treating per-layer
> composition as the *top-level* engine — the empirical zone result
> says the useful unit of skipping is choke-to-choke, not layer-by-
> layer. ZoneEngine is the top level. LayerEngine handles the
> irreducible per-layer work inside a WALK zone (T3 in the Gemma 3 4B
> zone map; cf. ZoneEngine §3.2). Within a WALK zone, the diagnosis
> below stands unchanged.

The dispatch seam already exists. In `larql-inference/src/layer_graph/`:

```
LayerGraph              trait — per-layer forward
├── DenseLayerGraph     — matmul attention + pluggable FFN
├── WalkLayerGraph      — dense attention + sparse WalkFfn (CPU walk)
├── PipelinedLayerGraph — CPU attention + Metal Q4 FFN (GPU accel)
├── CachedLayerGraph    — pre-computed residual lookup
└── PerLayerGraph       — per-layer strategy selection
```

`PerLayerGraph` is the LayerEngine. It already chooses a different
`LayerGraph` per layer; `predict_honest` already calls into it for the
production hybrid (cached L0–12 + walk L13–33 + GPU logits). The seam
is **unnamed and unclassified** under State Policy. This spec gives it
a name, classifies each strategy's contract, and writes the validation
discipline a future `LayerPolicy` must pass.

The v0.2 draft framed this as "build a new trait." v0.3 corrects:
**document the existing seam's contract**.

---

## 2. LayerEngine is itself an engine

A `LayerEngine` is **not** a wrapper around its constituents. It is a
new engine with its own State Policy triple, derived from the
constituents but not equal to any of them.

### 2.1 Canonical state

The union of canonical states across all `LayerGraph` strategies
selected at any layer. A LayerEngine that uses `MarkovResidual` at any
layer has *residual stream* in its canonical state; one that uses
`Standard` anywhere has *KV tensors*. The union is computed at
construction.

This is why uniform compositions are not identity reductions:
`LayerEngine::uniform(Standard, Dense, NoDispatch)` is a different
engine from `StandardEngine` standalone. They may produce the same
outputs; they have the same canonical state; but their
`execution_requirements` differ (§6), so they're different slots in
the taxonomy.

### 2.2 Derivative state

The union of all constituent derivatives. A LayerEngine inherits its
constituents' liberties: any cache any sub-engine declares derivative
remains derivative under the composition. **W10's `StateDumpMask`
cascade fires at this level** — see §4.

### 2.3 Correctness contract

The **meet** of constituent contracts under the lattice:

    exact_logits  >  bounded_KL(ε)  >  greedy_equivalent
                  >  confidence_gated(τ)  >  task_level_retrieval

(Read `>` as "stricter than"; total order in v0, see §10 for the
meet-semilattice question.)

The LayerEngine's contract is the **weakest** contract among
constituents active at any layer. Concretely (with v0.3's empirical
update):

- `uniform(Standard, Dense, NoDispatch)` → `exact_logits`
- `uniform(MarkovResidual, Dense, NoDispatch)` → `exact_logits`
  (conditional on arch preconditions)
- `tiered(L0-12 = (Standard, CompiledLookup, _), rest = default)`
  → **currently no defensible contract** (see §7 and §11; the
  underlying `CachedLayerGraph` is per-prompt memoization, not a
  template-class engine, so the meet calculation is moot until the
  constituent has a contract to meet against)

A LayerEngine cannot claim a contract stricter than its weakest
constituent's contract **measured on a stated calibration corpus**.
Shannon-bps measured on the LayerEngine output is the only thing
that licences the contract parameterisation (the ε in `bounded_KL`,
the τ in `confidence_gated`).

### 2.4 Fallback mode

The LayerEngine's fallback is the fallback of its weakest
constituent. If a constituent falls through to dense FFN below
threshold, the LayerEngine inherits that fallback at the affected
layers. Multiple sub-engines compose layer-by-layer — there is no
LayerEngine-level fallback orchestrator.

---

## 3. Composition rules

### 3.1 Per-layer triple

For each layer `L ∈ [0, n_layers)`, a LayerEngine resolves a triple:

    (KvEngine_L, FfnBackend_L, LayerGraph_L)

The third slot is the `LayerGraph` enum — naming the existing
strategies in §1. Layer routing is *configured*, not learned at decode
time (see §10 on adaptive routing).

### 3.2 Dispatch order

Unchanged from the reference forward pass:

    1. Attention (driven by KvEngine_L, possibly via LayerGraph_L)
    2. Residual add
    3. FFN (driven by FfnBackend_L, possibly via LayerGraph_L)
    4. Residual add

LayerEngine does not reorder; it chooses which implementation runs at
each step.

### 3.3 The composition correctness floor

A LayerEngine where every layer uses constituents with `exact_logits`
contracts produces bit-identical logits to the reference Standard
decode path, *provided* every sub-engine's preconditions hold.

This is the test that catches LayerEngine framing bugs:
`uniform(Standard, Dense, NoDispatch)` must produce bit-identical
output to the reference path. Any divergence is a LayerEngine bug,
not a constituent bug.

---

## 4. Work-skipping via the State Policy mask cascade

The naïve framing — "if a sub-engine declares Passthrough, skip
downstream work" — is **wrong** under State Policy. Whether work can
be skipped depends on whether the elided output is derivative or
canonical for downstream sub-engines.

### 4.1 W10's mask cascade is the mechanism

W10 already exposed the right surface in `larql-compute`:

```rust
pub enum StateDumpMask {
    Full,    // capture h_in + k_new + v_new
    HOnly,   // capture h_in only (K/V derivative for engine)
    None,    // capture nothing (both derivative or unused)
}
```

threaded through `KvDispatch::coarse_decode_step_with_state_masked`
and `DecodeBackend::decode_token_with_state_dump_masked`. The mask
is the engine's *intent* — "I treat this slot as derivative; you may
skip the bridge." The backend decides *how* to honor it.

### 4.2 Per-layer mask query

LayerEngine exposes a single query on each constituent's State
Policy:

```rust
fn mask_at(&self, layer: usize) -> StateDumpMask;
```

LayerEngine collects per-layer masks and threads them into the
backend call. The mask cascade is **the** skip rule under W10; this
spec does not invent parallel `permits_*_at(L)` machinery (v0.2
proposed `permits_no_append_at(L)`; v0.3 replaces it with
`mask_at(L)` on the existing trait surface).

### 4.3 The skip discipline

> A LayerEngine may use `StateDumpMask::HOnly` at layer L only if
> `KvEngine_L.mask_at(L) ∈ {HOnly, None}`. It may use `None` only
> if every downstream layer's KvEngine treats the residual at L as
> derivative (i.e. would also return `None`/`HOnly` for h_in).

This holds the State Policy cut at the API boundary instead of by
convention.

### 4.4 Where the tok/s actually comes from

Most measurable tok/s wins from the cascade are **Class A**
(derivative skip — no contract change):

- W10's HOnly mask on `markov_residual:window=512` → +9% tok/s,
  contract preserved.
- W10's None mask on `markov_residual` (windowless) → +21% tok/s,
  contract preserved.
- W10's HOnly mask on `unlimited_context:window=256` → +5% tok/s.

**Class B** (canonical-trajectory edits — contract drops) exists in
theory but **no current LayerEngine constituent supports it**. The
v0.2 draft assumed `CompiledLookup` was a working Class B example;
v0.3's §7 corrects that.

---

## 5. Configuration

A LayerEngine is constructed from a `LayerPolicy`:

    LayerPolicy {
        kv:       Vec<KvEngineKind>,
        ffn:      Vec<FfnBackendKind>,
        graph:    Vec<LayerGraphKind>,
    }

with helper builders:

- `LayerPolicy::uniform(kv, ffn, graph)` — same triple at every layer.
- `LayerPolicy::tiered(ranges: Vec<(Range<usize>, Triple)>)`.

At construction, the LayerEngine:

1. Computes its own State Policy triple from the constituents.
2. Validates `execution_requirements` against the backend (§6).
3. Validates that each sub-engine's preconditions hold for its
   assigned layers.
4. Verifies that any non-trivial contract claim has measured
   calibration data attached.

Construction is fallible. A LayerEngine that fails calibration or
execution-requirement checks does not return a partially-valid object.

---

## 6. Execution requirements

LayerEngine sits on the execution side of the engine-vs-execution cut.
Its constituents' `execution_requirements` drive what backends can
serve it.

### 6.1 Aggregation

The LayerEngine's `execution_requirements` is the union of its
constituents' requirements, plus a LayerEngine-specific requirement:
the backend must support per-layer dispatch (i.e. allow different
strategies at different layers without bouncing the residual through
a unified path). `PerLayerGraph` already satisfies this.

### 6.2 Backend refusal

`LayerShardedBackend` and `RemoteWalkBackend` may decline a
LayerEngine whose requirements they can't serve. Refusal is at
construction. A LayerEngine never silently degrades to a
backend-compatible composition.

Backend choice is informed by `RowLocation` on the W10 handle surface
(`larql_compute::state_handle::RowLocation::{LocalCpu, LocalGpu,
Remote}`). A LayerEngine with `Remote` rows at some layers requires a
grid-capable backend; trying to serve it from a single-node Metal
backend is a construction-time refusal.

---

## 7. Engine inventory under LayerEngine

Existing engines as `LayerPolicy::uniform` configurations (reminder:
these are different engines from their standalone counterparts; see
§2.1):

| Standalone engine        | Uniform LayerPolicy                             | LayerEngine contract           |
|--------------------------|-------------------------------------------------|--------------------------------|
| `StandardEngine`         | `uniform(Standard, Dense, DenseGraph)`          | `exact_logits`                 |
| `NoCacheEngine`          | `uniform(NoCache, Dense, DenseGraph)`           | `exact_logits`                 |
| `MarkovResidualEngine`   | `uniform(MarkovResidual, Dense, DenseGraph)`    | `exact_logits` (cond.)         |
| `UnlimitedContextEngine` | `uniform(Unlimited, Dense, DenseGraph)`         | `exact_logits` (in-window)     |
| `TurboQuantEngine`       | `uniform(TurboQuant, Dense, DenseGraph)`        | `codec_bounded_state`          |
| `Apollo`                 | `uniform(Apollo, Dense, DenseGraph)`            | `task_level_retrieval`         |

Per-layer compositions previously inexpressible:

| Composition          | LayerPolicy                                                              | Contract                           |
|----------------------|--------------------------------------------------------------------------|------------------------------------|
| `walk-ffn`           | `uniform(Standard, WalkFfn, WalkGraph)`                                  | `exact_logits`                     |
| `pipelined`          | `uniform(Standard, BackendFfn, PipelinedGraph)` (CPU attn + Metal FFN)  | `exact_logits`                     |
| `predict-honest`     | `tiered(L0-12 = (Standard, _, Dense), L13-33 = (Standard, WalkFfn, WalkGraph))` | `exact_logits` (current production) |
| `compiled-ffn`       | `tiered(L0-12 = (Standard, CompiledLookup, CachedGraph), rest = default)` | **aspirational — see below**       |
| `full-routed`        | cached early, experts mid, walk-ffn deep, window K/V throughout          | meet of constituents (when constituents have contracts) |

> **v0.4 scope note**: the "uniform LayerPolicy" rows above describe
> what each engine *would do as a top-level engine via LayerEngine*.
> In the v0.4 scoping, that role is `ZoneEngine::single_walk(...)` —
> a degenerate ZonePolicy with one WALK zone covering all layers. The
> table here is preserved for continuity; the operational entry point
> for any of these compositions is the ZoneEngine spec's §8
> `ZonePolicy` builders. `predict_honest` specifically is now
> `ZonePolicy::predict_honest()`, not a LayerPolicy.

### 7.1 `compiled-ffn` is aspirational under v0.3

The v0.2 draft assumed `CompiledLookup` (via `CachedLayerGraph`) was
a working `confidence_gated(τ)` engine. **Empirical measurement
falsifies this** (`crates/larql-kv/examples/contract_classify_cached_ffn.rs`,
2026-05-19):

| Sample test prompt (template = "The capital of France is") | KL_sym | Argmax |
|---|---:|:---:|
| exact build prompt | 0.003 | ✓ |
| "The capital of Germany is" | 2.05 | ✓ but "Paris" wrong city |
| "The capital of Brazil is" | 3.75 | ✗ |
| "She walked to the park" | 8.85 | ✗ |

Aggregate over 17 same-shape prompts: argmax_agreement = 58.8%,
kl_p95 = 8.85, kl_max = 9.38. `CachedLayerGraph` substitutes the same
L0–12 residual regardless of input, so the entity choice is locked in
by L0–12. The contract is `bounded_KL(ε=0)` on the singleton
`{build_prompt}` class and undefined elsewhere.

A viable `CompiledLookup` engine for LayerEngine needs **one of**:

- Per-prompt memoization (cache keyed on input; only useful for
  repeated identical prompts — rare in production).
- A calibrated similarity predicate as a runtime gate (the cosine
  axis fails here — see §11 — so a different gate is needed).
- A learned function-approximator that captures template variance.

Until one of these ships, `compiled-ffn` stays out of any production
LayerPolicy, and §8.4's tok/s-ceiling test does not have a
contract-meeting candidate at the cache slot.

The `full-routed` row above implicitly depends on this; v0.3 leaves
it in the inventory as the architectural intent but flags that it
inherits `compiled-ffn`'s aspirational status.

---

## 8. Validation strategy

Validation runs at four levels, mirroring State Policy §6's
predictive-units discipline:

1. **Composition floor.** `uniform(Standard, Dense, DenseGraph)` must
   be bit-identical to the reference Standard decode path. Catches
   LayerEngine framing bugs.

2. **Sub-engine identity.** Each `uniform(X, Dense, DenseGraph)`
   wrapped existing engine must produce bit-identical output to the
   standalone X under matched preconditions.

3. **Contract-claim validation.** Any LayerEngine claiming
   `bounded_KL(ε)` or `confidence_gated(τ)` must back the claim with
   Shannon-bps measurement on a stated calibration corpus, per State
   Policy §6: KL divergence, NLL delta, top-K agreement at
   K ∈ {1, 5, 20}, first-divergence trace, confidence-margin
   distribution on disagreements. No claim ships without numbers.
   `compiled-ffn`'s falsification (§7.1) is the worked example of
   how this discipline catches over-claims.

4. **Tok/s ceiling.** A composition is only accepted if (a) its
   contract claim holds under (3), and (b) tok/s is non-decreasing
   vs the equivalent non-skipping reference on the same backend. A
   composition that drops a contract step *and* doesn't improve
   tok/s is rejected at PR time. **`predict-honest` currently fails
   (b) on CPU** — see §11.

---

## 9. Migration plan

1. **Name the seam.** Document `PerLayerGraph` as the LayerEngine
   surface; add `LayerPolicy` + `LayerGraphKind` to the public API.
   No behaviour change.

2. **Slot existing engines as `uniform` policies.** Add the
   composition-floor and sub-engine-identity tests (§8.1, §8.2).

3. **Land `mask_at(L)` on each engine's State Policy** so LayerEngine
   can thread W10's mask cascade per layer. (W10 already exposed
   StateDumpMask; this step lifts it to the per-layer query.)

4. **Define `predict-honest` as a named LayerPolicy** with measured
   §8.3/§8.4 numbers. This is the current production composition; the
   spec should describe what's running.

5. **Future: `CompiledLookup` with a real gate.** Blocked on a
   working CompiledLookup design (§7.1). When it arrives, gate
   `compiled-ffn` behind §8.3 validation; do not ship without it.

6. **Future: `Dispatcher` (Virtual Experts), `full-routed`.** Both
   depend on (5); deferred.

---

## 10. Non-goals and open questions

### Non-goals

- **Replacing existing engines.** LayerEngine is a composition seam.
- **Adaptive routing.** §3.1 forbids content-dependent layer dispatch
  in v0. Defer to v1; ship static policies first.
- **Transport layer.** BOUNDARY refs and grid transport are concerns
  of `larql_compute::state_handle::RowLocation::Remote` plus the
  per-engine traits, not LayerEngine.

### Open questions

1. **Gate variable for `CompiledLookup`.** Empirically (§7.1, §11)
   logit cosine does not separate the safe class from the failing
   class — failing prompts spanned cos 0.87–0.97 and the cleanest hit
   was at cos 0.999 (the exact build prompt). What predicate **does**
   identify the cache's domain of validity? Live-vs-cached residual
   cosine at L_N (before substitution) is the obvious candidate;
   needs measurement.

2. **Contract lattice as meet-semilattice.** `bounded_KL(ε)` and
   `confidence_gated(τ)` may not be comparable; v0 ducks this by
   taking the strictly-weaker contract conservatively. v1 may need a
   real lattice.

3. **Where does cold-tier-residual canonical-state classification
   live?** Audit during W10 found `markov_residual` spec says cold
   tier = token IDs but code stores residuals. Different State Policy
   triples. Independent of LayerEngine but worth flagging here — the
   `MarkovResidualEngine` row in §7 inherits whichever classification
   ships.

4. **Quantisation interaction.** Engines validated at FP16; under
   Q4_K / Q6_K the cache hit rates and KL shift. Validation harness
   should run at both precisions per State Policy §6.

5. **Cross-architecture portability.** L0–12 = "template-fixed" is a
   Gemma 3 4B observation. Same fraction-of-layers on Llama 3 /
   Mistral / Gemma 4 E4B needs empirical validation before any
   portable policy claim.

---

## 11. Measurement appendix

### 11.1 Current production composition — `predict_honest`

Measured 2026-05-02 on Gemma 3 4B Q4K (`crates/larql-inference/PERFORMANCE.md`):

```
predict_honest("The capital of France is"):
  Phase 0 (L0-12): CachedLayerGraph    ~5ms  (NOT in production — empty cache)
  Phase 1 (L13-33): CPU attn + WalkFfn  ~195ms
  Phase 2: GPU logits KNN                ~4ms
  Total:                                ~203ms = 4.9 tok/s
```

Every production call site constructs `CachedLayerGraph::from_residuals(Vec::new())`
— an empty cache. The Phase 0 ~5 ms figure is what the cached path
*would* take if populated for the prompt under test; it is not what's
running. The "0.999 cosine" annotation in PERFORMANCE.md is self-cosine
on the build prompt (cache built from X, evaluated on X), not a
generalization claim.

The honest production composition today is `predict-honest =
tiered(L0-12 = (Standard, _, Dense), L13-33 = (Standard, WalkFfn,
WalkGraph))` running entirely on CPU walk except for the lm_head KNN
step. Contract: `exact_logits`. Speed: 4.9 tok/s.

### 11.2 §8.4 tok/s-ceiling test — predict_honest fails on CPU

Current CPU baseline (`larql bench --cpu --engine standard`, same model):
**14.5 tok/s**.

`predict_honest` at 4.9 tok/s is **3× slower** than the dense CPU
baseline. Under §8.4, this composition **does not pass** the
tok/s-ceiling test. The discipline is right; the composition is
incomplete. The savings the cached prefix would deliver are eaten by
the CPU/Metal handoff in Phase 2 (GPU logits KNN over the lm_head
vindex while everything else is CPU). Closing that gap is a separate
work item:

- Land Q4K matmul as a CPU primitive (closes the 55× prefill gap
  flagged in the larql-compute roadmap).
- Run the whole composition on one device (all-CPU walk including
  logits, or all-GPU with PipelinedLayerGraph).

Neither has anything to do with LayerEngine; LayerEngine cannot close
the gap. But the spec discipline catches that the apparent
production-target composition isn't ready.

### 11.3 `compiled-ffn` falsification

Run: `cargo run --release -p larql-kv --example contract_classify_cached_ffn -- \
  --vindex ~/.cache/larql/local/gemma3-4b-q4k-v2.vindex \
  --template "The capital of France is" --cached-until 13`

Result (17 same-template-length prompts, 2026-05-19):

| Metric | Value |
|---|---:|
| argmax_agreement | 58.8 % |
| logit_cos_mean | 0.9414 |
| kl_symmetric mean | 3.01 |
| kl_symmetric p95 | 8.85 |
| kl_symmetric max | 9.38 |
| kl_symmetric on exact build prompt | 0.003 |

`bounded_KL(ε=0)` holds on the singleton `{build_prompt}`. No defensible
contract holds elsewhere. Logit cosine doesn't separate the safe class
from the failing class — the failing prompts had logit cosines from
0.87 to 0.97, while the best-KL prompt had 0.999. Cosine is not the
gate variable for this engine.

This is the §8.3 contract-claim validation discipline doing its job:
the assumption baked into v0.2 ("CompiledLookup is `confidence_gated`,
calibrate τ later") was falsified before it landed in production. v0.3
marks `compiled-ffn` aspirational and lists the open gate-variable
question as §10.1.

---

## 12. Scope migration history

| Version | Date | Scope |
|---|---|---|
| v0.2 | 2026-05-18 | New top-level engine; hypothetical `compiled-ffn` working composition |
| v0.3 | 2026-05-19 | Top-level engine over existing `PerLayerGraph` seam; `compiled-ffn` aspirational |
| **v0.4** | **2026-05-19** | **Inner per-layer composer for WALK zones; top-level role moves to [ZoneEngine](./zone-engine.md)** |

The scope narrowing in v0.4 reflects the empirical finding that the
useful unit of skipping is choke-to-choke, not layer-by-layer. The
per-layer composition mechanism this spec describes is unchanged; only
its scope as "the top-level engine" was wrong.

## 13. Cross-references

- [`state-policy.md`](./state-policy.md) — engine identity taxonomy.
  §3.1 has the W10 worked-example table that LayerEngine §4 builds on.
- [`engine-state-vs-execution.md`](./engine-state-vs-execution.md) —
  the orthogonal cut. §11 (W10's mask cascade as a §2 worked example)
  is the closest pre-W10 cousin of this spec.
- [`kv-engine-unification.md` §4.4](./kv-engine-unification.md) — the
  `StateDumpMask` + `read_kv_row_at` trait surface LayerEngine threads
  per-layer.
- [`markov-residual-engine.md` §14](./markov-residual-engine.md),
  [`markov-residual-codec-engine.md` §14](./markov-residual-codec-engine.md),
  [`windowed-checkpoint-engine.md` §8](./windowed-checkpoint-engine.md) —
  per-engine W10 opt-in tables; the mask cascade these refer to is the
  same mechanism LayerEngine reuses per layer.
- `crates/larql-kv/examples/contract_classify_cached_ffn.rs` — the
  reproducible §11.3 falsification.
- `crates/larql-inference/src/layer_graph/cached.rs` doc-comment — the
  per-prompt-only warning panel on `CachedLayerGraph`.
- `crates/larql-inference/src/layer_graph/dense.rs` — `PerLayerGraph`,
  the existing seam.
