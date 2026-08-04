# `build_router` Design Draft — `ValidatedFfnLayerPolicy` and Construction Handles

**Status:** 📝 Draft v0.1 (2026-05-24); implementation shipped
2026-05-24 — see "Resolutions" below.
**Audience:** larql-inference contributors maintaining the FFN policy
and the `LayerFfnRouter` integration.
**Scope:** Resolves the design decisions blocking the
`FfnLayerPolicy::build_router` implementation slice.
**Consumers:** the future CLI-wiring slice (`larql bench --ffn`,
`larql accuracy --ffn`) and the K=100 cross-trait sweep harness
extension.

## Resolutions (post-draft)

The four open questions in §7 resolved as follows. The implementation
that landed reflects these positions.

- **§2.1 (consume vs borrow on `validate_for`)** — consumes `self`,
  returns `Result<ValidatedFfnLayerPolicy, _>`. Multi-model
  validation via `into_policy()` round-trip if needed.
- **§3 (backend lifetime ownership)** — Option C (sibling
  `BoundFfnRouter`), variant *kept in `larql-inference::ffn_policy`*
  rather than `larql-kv` after the FFN policy module was relocated
  from `larql-kv` to `larql-inference` (§3.3 resolved toward the
  module-home being natural for both code and router type).
- **§3.3 (module home)** — `larql-inference::ffn_policy/`. The full
  ffn_kind monolith was split into six files
  (`mod.rs`/`backend_kind.rs`/`routing.rs`/`policy.rs`/
  `validated.rs`/`router.rs`) and moved out of `larql-kv` because
  `larql-kv` is for KV engines, not FFN policy.
- **§7.4 (`BuildError::RemoteWalkConstruction` typed or `Box<dyn>`)** —
  moot; v0 of `build_router` errors with
  `BuildError::RemoteWalkNotYetWired` (a variant with no inner
  source) rather than attempting remote construction. Typed source
  decision deferred to the slice that actually wires `RemoteWalk`.

Where the draft text below contradicts these resolutions — chiefly
the assumption that the policy code lives in `larql-kv` — read it as
historical context for the decision process, not as the current
shape of the code.

---

## 1. The diagnosis

The Item 2 v0 parser (`crates/larql-kv/src/ffn_kind.rs`) lands the
spec language and validates a policy against a known layer count, but
stops short of turning a policy into a live `&dyn FfnBackend`
dispatcher. Three design calls block the `build_router` slice:

1. Does `validate_for` return the same `&FfnLayerPolicy` it was called
   on (validation as documented convention), or does it return a new
   `ValidatedFfnLayerPolicy` newtype (validation as type-system
   invariant)?
2. What shape do the construction handles take — `(&ModelWeights,
   Option<&VectorIndex>)` mirroring `EngineKind::build(&backend)`, or
   something else?
3. How does the policy → router conversion compose with the existing
   `LayerFfnRouter` — fresh construction, in-place reconfiguration,
   or via a sibling owned-router type?

This document resolves (1) and (2) as the framing position, flags (3)
as needing the `LayerFfnRouter` owner's adjudication, and surfaces a
fourth question — backend lifetime ownership — that fell out of
reading the router code while drafting (3).

The doc is markdown-first by intention. The three calls above are
decisions, not code; landing them in prose before the implementation
PR is the same discipline that produced Item 2 v0's clean shape
(parse and validate separated *because* the design was settled
before the parser was written, not refactored mid-implementation).

---

## 2. Settled design calls

### 2.1 `ValidatedFfnLayerPolicy` as a newtype — **adopt**

**Position:** add the newtype.

```rust
/// A policy that has been validated against a known model's layer
/// count. Distinct from [`FfnLayerPolicy`] at the type level so the
/// type system enforces "validate before build" — the only way to
/// obtain one is via [`FfnLayerPolicy::validate_for`], which performs
/// the layer-coverage and out-of-range checks.
///
/// `pub` (so callers can name the type in signatures and `Result`
/// arms), but the inner field and constructor are *not* `pub`. The
/// non-public constructor is the load-bearing mechanism — without it
/// the newtype is documentation, not enforcement.
pub struct ValidatedFfnLayerPolicy {
    policy: FfnLayerPolicy,
    num_layers: usize,
}

impl FfnLayerPolicy {
    /// Validate against a model's layer count and produce a
    /// validated handle. Replaces today's `validate_for` returning
    /// `Result<(), PolicyValidationError>`.
    pub fn validate_for(
        self,
        num_layers: usize,
    ) -> Result<ValidatedFfnLayerPolicy, PolicyValidationError> {
        // current validate_for body, then wrap on success
        Ok(ValidatedFfnLayerPolicy { policy: self, num_layers })
    }
}
```

**Why newtype over convention.** The convention approach (keep
`validate_for` returning `Result<(), …>`, document that callers must
call it before `build_router`) has one cost line per caller — the
`policy.validate_for(num_layers)?;` line everyone would write
defensively anyway. The newtype costs the same line but converts
"forgot to call validate" from a runtime failure (which today would
mean a panic in `build_router` on a layer index assumption, or worse,
silent wrong behaviour) into a compile error. The trade is **one
type for one bug class**, which is the right exchange for a public
API.

**Why constructor is non-public.** A public constructor would let
callers write `ValidatedFfnLayerPolicy { policy, num_layers: ... }`
directly, bypassing `validate_for` and reproducing the bug class the
newtype was meant to eliminate. Keeping construction non-public is
the only way the type's invariant ("this policy has been
validated") is actually enforced. The cost is that
`ValidatedFfnLayerPolicy` cannot be constructed in downstream tests
without going through `validate_for` — that's the desired property,
not a limitation.

**API impact.** `FfnLayerPolicy::validate_for` changes from
`&self -> Result<(), ...>` to `self -> Result<Validated..., ...>`
(consumes the policy). Callers that wanted to inspect a policy after
validating still can — `ValidatedFfnLayerPolicy::policy(&self)` for
read-only access; we don't expose mutable access because that would
let callers invalidate the policy without re-validating.

### 2.2 Construction handles shape — **mirror `EngineKind::build`**

**Position:** `build_router` lives on `ValidatedFfnLayerPolicy`, takes
`(&ModelWeights, Option<&VectorIndex>)`, returns `Result<…, BuildError>`.

```rust
impl ValidatedFfnLayerPolicy {
    pub fn build_router<'a>(
        &self,
        weights: &'a ModelWeights,
        index: Option<&'a VectorIndex>,
    ) -> Result<BoundFfnRouter<'a>, BuildError> {
        // see §3 for the return-type discussion
    }
}
```

**Why this shape.**

- **Policy validated separately from model handles.** Same separation
  as `EngineKind` — parse a policy from a spec string, validate
  against a layer count, bind to weights at build time. Each step has
  a clean role and a clean error type. Lets a single parsed policy
  bind to multiple model instances (the K=100 sweep harness loads
  several models against the same policy).
- **`Option<&VectorIndex>` for the index handle.** `Dense`, `Null`,
  and `RemoteWalk` don't need a `VectorIndex`; only `Walk { k }`
  does. Making `index` optional lets the most common policies
  (`dense`, `walk:k=N`) be built without forcing the caller to load
  a vindex they don't use. `build_router` errors with
  `VectorIndexRequired { binding_index }` if a `Walk { k }` binding
  is present and `index` is `None`.
- **Lifetime parameter `'a` on the return type.** Carries the
  weights / index borrows correctly so the router can hold
  references into them. The actual ownership shape of the returned
  type is **unresolved** — see §3.

### 2.3 `BuildError` taxonomy

```rust
pub enum BuildError {
    /// A `Walk { k }` binding requires a `VectorIndex`, but `None`
    /// was passed. The `binding_index` is the position in the
    /// policy's binding list; useful for error reporting when the
    /// policy was constructed from a multi-binding spec string.
    VectorIndexRequired { binding_index: usize },
    /// Remote backend construction failed (network, endpoint, wire
    /// protocol). Source is the underlying error from
    /// `larql_inference::ffn::remote::RemoteFfnError`.
    RemoteWalkConstruction {
        endpoint: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Internal invariant violation: the validated policy references
    /// a layer outside `weights.num_layers`. **Should be unreachable
    /// for a `ValidatedFfnLayerPolicy`** — its presence indicates a
    /// bug in `validate_for` or that `weights` was swapped between
    /// validate and build. Surfaces as a typed error rather than a
    /// panic so callers can log + recover.
    LayerOutOfRange {
        layer: usize,
        num_layers: usize,
    },
}
```

The third variant is defensive — it shouldn't fire if the type
system did its job, but a typed return is preferable to a panic if
something upstream gets clever about reusing handles. Mirrors finding
(2) from the trait-extraction ROADMAP entry (InvariantViolation as a
distinct alerting category).

---

## 3. Unresolved design call — backend lifetime ownership

The framing from the original sequencing message — *"`build_router`
constructs a fresh `LayerFfnRouter::PerLayer(Vec<&dyn FfnBackend>)`
from the policy bindings"* — runs into a constraint that requires
either changing `LayerFfnRouter`'s signature or returning a sibling
owned type. The constraint isn't a design choice; it's how the
existing router takes its backends.

### 3.1 Why this is harder than it looked

`LayerFfnRouter` (in `crates/larql-inference/src/ffn/mod.rs:46`)
takes its backends by reference:

```rust
pub struct LayerFfnRouter<'a> {
    backends: Vec<&'a dyn FfnBackend>,
    num_layers: usize,
}

impl<'a> LayerFfnRouter<'a> {
    pub fn per_layer(backends: Vec<&'a dyn FfnBackend>) -> Self { ... }
}
```

Existing callers (`larql-cli/src/commands/extraction/walk_cmd.rs:868`
and `larql-inference/chris-experiments/larql_probes/examples/misc/walk_boundary_sweep.rs:205`) own the
concrete backend instances as local values and pass references:

```rust
let weight_ffn = WeightFfn { weights };
let walk_ffn = WalkFfn::new(weights, &index, top_k);
let mut backends: Vec<&dyn FfnBackend> = vec![&weight_ffn; num_layers];
backends[switch..].fill(&walk_ffn);
let router = LayerFfnRouter::per_layer(backends);
```

The backend instances (`weight_ffn`, `walk_ffn`) outlive the router
because they're owned by the caller's stack frame. The router holds
references into them.

`build_router` cannot follow this pattern directly — if it
constructs `WeightFfn { weights }` internally and then takes a
reference, the `WeightFfn` drops when `build_router` returns and the
reference dangles. **The router needs to either own the backends or
be paired with something that does.**

This is the load-bearing question the original message marked as
unresolved (*"the one I haven't seen the code for yet"*); reading the
router confirmed the unresolvedness rather than dissolving it.

### 3.2 Four implementation options

| Option | Mechanism | Cost | Preserves existing API |
|---|---|---|---|
| **A. Caller-owned pool** | `build_router` returns `Vec<&'a FfnBackendKind>` (the per-layer kind expansion) only. A separate `construct_pool(kinds, weights, index)` helper builds owned backends; the caller assembles `Vec<&dyn FfnBackend>` and constructs the router themselves. | Two-step ritual for callers; "build_router" name overpromises. | Yes |
| **B. Change `LayerFfnRouter` to own** | Modify `LayerFfnRouter<'a>` to hold `Vec<Box<dyn FfnBackend + 'a>>` instead of `Vec<&'a dyn FfnBackend>`. `build_router` then returns the router directly. | Breaking API change; two existing callers must migrate. | No |
| **C. New sibling owned router** | Add `pub struct BoundFfnRouter<'a> { backends: Vec<Box<dyn FfnBackend + 'a>>, num_layers: usize }` in larql-kv (or larql-inference, see §3.3) with the same `get(layer)` API. `build_router` returns this; callers wanting the &-flavored router keep using `LayerFfnRouter`. | One new type; minor API surface bloat. | Yes |
| **D. Self-referential wrapper** | A type that owns the backends and a router whose references point into the owned backends. Requires `ouroboros` or unsafe; ergonomics bad; not idiomatic. | High complexity. | Yes |

**Recommendation: C.** Reasoning:

- **A** pushes the integration cost to every caller. The whole point
  of `build_router` is that the caller wrote
  `walk:k=100,layers=14-27;dense:layers=0-13,28-33` and shouldn't have
  to know that turns into "construct WeightFfn, construct WalkFfn,
  assemble a Vec, pass to LayerFfnRouter::per_layer." If we make them
  do that anyway, the parser slice barely earned its keep.
- **B** is the cleanest semantic outcome but requires migrating
  `walk_cmd.rs` and `walk_boundary_sweep.rs` in the same PR. Doable
  but inflates the PR's review surface from "new feature" to "new
  feature + breaking change + two-file migration." Bad split.
- **D** is right out — self-referential structs in safe Rust are
  ouroboros territory and don't justify the complexity here.
- **C** lets the policy-as-data property of `FfnLayerPolicy` extend
  cleanly: a validated policy + model handles → a self-contained
  bound router that the caller doesn't have to reason about
  lifetimes for. The owned-backends choice is local to the new type
  and doesn't ripple into `LayerFfnRouter`'s existing callers. One
  new type, no breaking change.

If the `LayerFfnRouter` owner wants to migrate to Box-flavored
storage (Option B) as a separate cleanup later, `BoundFfnRouter`
either becomes a thin wrapper or is deprecated in favor of the
unified type. Option C doesn't commit us to the sibling-router
forever; it just gets `build_router` shipping without forcing a
breaking change in the same PR.

### 3.3 Where does `BoundFfnRouter` live?

Two reasonable homes:

- **`larql-kv` (alongside `ffn_kind.rs`).** Keeps the build_router
  slice's surface contained within larql-kv. larql-kv already depends
  on larql-inference for `FfnBackend`, so the trait bounds work.
  Downside: `larql-inference`'s `predict_with_router` and friends
  (`larql-inference/src/forward/predict/ffn.rs:106`) consume
  `&LayerFfnRouter` directly; they'd need overloads or conversions to
  accept `&BoundFfnRouter` too.
- **`larql-inference::ffn` (alongside `LayerFfnRouter`).** Natural
  home — both routers live in the same module, share the `get(layer)`
  interface, can share a `RouterView` trait that
  `predict_with_router` consumes. Downside: the build_router PR now
  touches larql-inference, which crosses a crate boundary and adds a
  reviewer.

**Recommendation: `larql-inference::ffn`, with a `RouterView` trait
abstracting `get(layer)`.** The conversion overhead of keeping
`BoundFfnRouter` in larql-kv is the same blast radius as adding the
trait in larql-inference, and the latter produces the more honest
public surface (both routers are equally first-class). The build_router
slice's PR therefore crosses two crates — flag this in the slice's
roadmap entry so reviewers expect it.

---

## 4. End-to-end usage sketch

What a caller writes, post-slice, to express the K=100 hybrid
sweep on Gemma 3 4B (34 layers):

```rust
use larql_kv::{FfnLayerPolicy, FfnBackendKind};
use larql_inference::{InferenceModel, ffn::RouterView};

let model = InferenceModel::load("...")?;
let weights = model.weights();
let index = model.vindex();           // Option<&VectorIndex>

let policy = FfnLayerPolicy::from_spec(
    "{walk:k=100}@layers=14-27;{dense}@otherwise",
)?;
let validated = policy.validate_for(weights.num_layers)?;
let router = validated.build_router(weights, index)?;

// `router: BoundFfnRouter<'a>`; `&router: &dyn RouterView`.
let logits = predict_with_router(weights, tokenizer, &tokens, top_k, &router);
```

Four lines from spec to dispatcher. The previous shape required the
caller to write the `vec![&weight_ffn; num_layers]` mutation pattern
inline; the slice's value is collapsing that to two lines (the parse
+ the build).

Sweep loop over K values:

```rust
for k in [None, Some(50), Some(100), Some(500), Some(2048)] {
    let spec = match k {
        None => "{walk:k=None}@layers=14-27;{dense}@otherwise".to_string(),
        Some(k) => format!("{{walk:k={k}}}@layers=14-27;{{dense}}@otherwise"),
    };
    let policy = FfnLayerPolicy::from_spec(&spec)?
        .validate_for(weights.num_layers)?;
    let router = policy.build_router(weights, index)?;
    // ... drive accuracy/bench through the router, record K → metrics
}
```

The K-sweep that was a "rewrite the parser and the harness loop"
project becomes a six-line for-loop. That's the leverage the parser
slice was buying.

---

## 5. Non-goals

- **Sparse / LayerSharded / RemoteMoe variants.** Deferred from Item 2
  v0 for the same reason here — each needs a CLI invocation that
  doesn't exist today. Add when a sweep use case forces the design.
- **Per-prompt routing predicates** (`confidence>0.9`, etc.). The
  `RoutingPredicate` enum is shaped to accommodate them but the
  `build_router` slice ships with only `Layers` / `All` / `Otherwise`.
  Confidence-gated routing requires the trait extraction's typed
  `Result` to know which path was taken; gate accordingly.
- **Async / streaming policy reconfiguration.** A built router is
  static for the duration of its lifetime. Dynamic per-step routing
  is a separate feature.
- **Touching `EngineKind` or `KvEngine`.** This slice is the FFN
  axis; the engine axis is independent. The accuracy harness already
  composes them orthogonally (engine × ffn = product).
- **CLI flag wiring.** `larql bench --ffn` and
  `larql accuracy --ffn` come in the *next* slice after `build_router`
  lands. Keeping them separate so the wiring PR can be reviewed
  against a known-working `build_router`, not in lockstep with it.

---

## 6. Sequencing

Restated from the original message, with this draft as step 2:

1. **Item 1 PR #134** in review (accuracy schema fix + ROADMAP entry).
2. **Now:** this design draft on `design/ffn-build-router`. Markdown
   only, reviewable in parallel with #134.
3. Item 1 merges → rebase `feat/ffn-backend-kind` on main → open
   Item 2 v0 PR (`FfnBackendKind` + `FfnLayerPolicy` parser).
4. Item 2 v0 merges → `build_router` *code* slice, against the
   adjudicated design from this doc. New branch off main; consumes
   the merged parser; lands `ValidatedFfnLayerPolicy` newtype,
   `BoundFfnRouter`, `RouterView` trait, `BuildError`. Touches
   `larql-inference` for the trait/router additions.
5. `build_router` merges → CLI wiring slice (`--ffn` flag on bench
   and accuracy, plumbing the JSON's `ffn_backend` column to reflect
   the actual policy).
6. CLI wiring merges → cross-product harness extension (the K=100
   sweep as a single CLI invocation rather than a script).

Each slice has one reviewable concern with a tight review question.
The design decisions for step 4 are settled in this draft before any
code commits to them; reviewers of step 4 are reviewing the
implementation against an adjudicated design, not adjudicating the
design and the implementation in the same pass.

---

## 7. Open questions for this draft's reviewers

1. **§2.1 — should `validate_for` consume `self` or borrow?** The
   sketch consumes (`self -> Result<Validated, _>`); the alternative
   is `&self -> Result<&Validated, _>` returning a borrowed handle
   that's tied to the original policy's lifetime. Consuming is
   cleaner (one ownership, no aliasing question) and matches the
   "build_router needs an owned policy anyway" usage; borrowing
   would help if a single policy needs to be validated against
   multiple models with different layer counts. Position: consume,
   but flagging because the trade-off depends on whether multi-model
   validation is a real use case.
2. **§3 — does the `LayerFfnRouter` owner endorse Option C, or do
   they want B as a coordinated cleanup?** This is the load-bearing
   question. If the owner says "let's do B and migrate the two
   callers in the same PR," that's a different slice shape — bigger
   review but cleaner end state. If "stick with C and don't touch my
   router," the slice ships as drafted.
3. **§3.3 — `larql-kv` vs `larql-inference` home for
   `BoundFfnRouter`.** The recommendation is `larql-inference` but
   that crosses a crate boundary. Flag if the larql-kv owner prefers
   to keep the new type local to larql-kv even at the cost of
   conversion glue.
4. **Should `BuildError::RemoteWalkConstruction` carry the typed
   `RemoteFfnError`, or stay `Box<dyn Error>`?** Typed is more
   informative but pulls `larql_inference::ffn::remote` into the
   public error surface of larql-kv (more coupling). `Box<dyn Error>`
   is more loosely coupled. Position: `Box<dyn Error>` for v0 of
   `build_router`; tighten to typed later if a caller surfaces a
   real need.

---

## 8. Cross-references

- [`ROADMAP.md`](../ROADMAP.md) "P0 — sibling trait extraction for
  non-K/V engines (Apollo, Mode 5)" — the parent design context;
  this draft sits on top of the same state-policy framing.
- [`src/ffn_kind.rs`](../src/ffn_kind.rs) — the Item 2 v0 parser
  this slice extends.
- `larql-inference/src/ffn/mod.rs:46` — existing `LayerFfnRouter`
  shape that §3 is reasoning against.
- `larql-cli/src/commands/extraction/walk_cmd.rs:868` and
  `larql-inference/chris-experiments/larql_probes/examples/misc/walk_boundary_sweep.rs:205` — existing
  call sites that exemplify the "caller-owns-backends" pattern
  `build_router` is collapsing.
- [`docs/state-policy.md`](./state-policy.md) — the
  `(canonical_state, derivative_state, correctness_contract)` triple
  this slice doesn't change but composes with on the engine axis.
