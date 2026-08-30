# SemanticPromotionEngine

**Rust name:** `larql_kv::engines::semantic_promotion::SemanticPromotionEngine`
**CLI name:** `semantic-promotion` (`--engine semantic-promotion:base=standard:window=512,mode=observe`)
**Status:** Phase A implemented. Model-walk layer implemented (observe + enforce). Phase B implemented on the CPU per-layer path, validated on a Gemma-shaped interleaved architecture (B10) — see §6.3 for what it does *not* yet cover. Phases C–H not started.
**Measured basis:** EXP-25 (`chris-experiments/rsl/exp25_late_promotion_matrix.py`), Gemma-3 27B, 65 536-token context, 8 global attention layers of 48.

---

## 1. What it is

A **policy wrapper over an exact decode engine**, not another attention
implementation. It manages long-context state as a set of semantic
authorities rather than an undifferentiated token-age-ordered KV cache:

```text
distant source authority
        ↓  typed compact record
        ↓  late promotion near the active boundary
        ↓  source excluded from current execution
        ↓  local read or computation
        ↓  scoped retirement or cold replay
```

The wrapper owns authority, visibility, the promotion lifecycle and
qualification. The base engine still owns prefill, decode, attention and
FFN dispatch, and the caller's `FfnBackend` is threaded through
untouched on all eighteen `KvEngine` methods. That is what lets the same
engine wrap the dense standard path today and a K3 MoE path later
without a second redesign.

---

## 2. The measurement this is built around

EXP-25 promoted a compact record at 65K distance with the source span
excluded from the global-attention layers, then teacher-forced the
*reference* continuation through each arm and compared distributions
position by position at a 0.05 bit tolerance.

| question | family | canonical: payload | canonical: trajectory | derived: payload | derived: trajectory |
|---|---|---|---|---|---|
| `copy` | read | PASS | PASS | — | — |
| `transform_inc` | compute | PASS | **fail** | PASS | PASS |
| `transform_rev` | compute | PASS | **fail** | **fail** | **fail** |

Controls held on every row: a length-matched pad failed, and a corrupted
record carried its corruption into the answer.

Two facts drive the design.

**Capability is per-operation.** `transform_rev` refuses the very derived
record that carried `transform_inc` — it answered `7431` where `1347` was
required. So `CapabilityKey` keys on the operation signature, and no type
in this module has a per-record capability flag.

**The scored object is part of the claim.** Both canonical compute
failures are a *single position*: the first termination token.

| arm | payload positions (max) | first termination | verdict at 0.05 b |
|---|---|---|---|
| `copy` / canonical | 0.0001 | 0.0441 | trajectory PASS |
| `transform_inc` / canonical | 0.0010 | **0.1472** | payload PASS, trajectory fail |
| `transform_rev` / canonical | 0.0223 | **0.0610** | payload PASS, trajectory fail |
| `transform_inc` / derived | 0.0008 | 0.0003 | trajectory PASS (peak 0.0218 at position 9, post-termination) |

The same record is sound for the answer and unsound for the decision to
stop. Note that invariant I10 — excluding repeated termination probes
from scoring — does **not** rescue these rows: the peak is *inside* the
reachable window, not past it.

### Consequence: gate–claim congruence is a type, not a convention

`ScoredObject` records what was measured. `AnswerBoundary` records what
is being claimed. `RetirementScope::check_covered_by` refuses every
combination where the second exceeds the first:

| certificate scored | `ExactPayloadLength(n)` | `FirstTermination` | `ExternalCommit` |
|---|---|---|---|
| `PayloadOnly { payload_tokens: m }` | `m ≥ n` | **refused** | **refused** |
| `ReachableTrajectory { payload_tokens: m }` | `m ≥ n` | allowed | **refused** |

A payload-scored certificate cannot authorise a scope running through
first termination — which is exactly the licence "the answer was right"
would otherwise have granted. This is standing rule R12 (`docs/dec-funnel.md`)
applied to a runtime decision.

`GeneralState` needs more still: `Option<GeneralStateEvidence>`, a token
whose only field is private and whose issuer is crate-internal. A boolean
would have been one careless `true` away from a permanent, unevidenced
retirement. **Nothing issues one today** — general state means the
replacement holds for operations that have not been measured, and no
result in the programme supports that. The issuer arrives with an offline
general-state certification process, not with a KL threshold.

### The certificates EXP-25 actually supports

| record / operation | evidence | may authorise | must refuse |
|---|---|---|---|
| `copy` / canonical | `ReachableTrajectory(4)` | `ExactPayloadLength(4)`, `FirstTermination` | `ExternalCommit`, `GeneralState` |
| `transform_inc` / canonical | `PayloadOnly(4)` | `ExactPayloadLength(4)` | `FirstTermination` and beyond |
| `transform_rev` / canonical | `PayloadOnly(4)` | `ExactPayloadLength(4)` | `FirstTermination` and beyond |
| `transform_inc` / derived | `ReachableTrajectory(4)` | `ExactPayloadLength(4)`, `FirstTermination` | `ExternalCommit`, `GeneralState` |
| `transform_rev` / derived | *none* | — | everything; payload itself fails |

---

## 3. Architectural position

```text
SemanticPromotionEngine
    authority + record policy, source visibility,
    promotion lifecycle, qualification, phase events
            ↓
base KvEngine  (Standard today; MarkovResidual / K3Local later)
    prefill, append/decode, attention, FFN dispatch
            ↓
larql-compute / Metal / CPU / remote FFN
```

The engine implements no model math.

### Capability admission

Construction fails rather than degrading. `PromotionEngineCapabilities`
carries six claims; `PromotionMode::admit` checks them:

| mode | requires |
|---|---|
| `Observe` | `caller_supplied_ffn_backend` |
| `Shadow` | all five core capabilities |
| `EnforceResidentRollback` | core + `snapshot_restore` as the rollback route |
| `EnforceColdReplay` | core + `source_replay` |

A base engine's own claim is still only `caller_supplied_ffn_backend` —
none of them implement masking. `EnforceResidentRollback` becomes
constructible when the **wrapper** declares what it provides for itself
via `PromotionEngineCapabilities::wrapper_exclusion()`; see §7.2 for the
provenance table. `EngineKind::build` deliberately does not do this, so
`--engine semantic-promotion:mode=enforce` still refuses: a CLI caller
cannot assert on the wrapper's behalf.

---

## 4. Module map

`crates/larql-kv/src/engines/semantic_promotion/`

| file | concept |
|---|---|
| `ids.rs` | stable `u128` identities, domain-tagged, string-encoded on the wire |
| `authority.rs` | `SourceAuthority`, `SourceAuthorityRef`, SHA-256 content digest |
| `graph.rs` | authority graph; `ReplacesFor` edges; per-operation resolution |
| `record.rs` | `CanonicalFact`, `DerivedResult` (discharged by construction) |
| `materialisation.rs` | `RecordMaterialisation` (v1: `TokenSequence` only), `MaterialisationKind` |
| `operation.rs` | `OperationClass`, `OperationSignature`, `AnswerBoundary` |
| `scoring.rs` | `ScoredObject`, `DivergenceClass`, `score_trajectory`, I10 window |
| `qualification.rs` | `CapabilityKey`, certificates, shadow comparisons |
| `scope.rs` | `RetirementScope` + the congruence rule |
| `visibility.rs` | `SemanticVisibility` × `ResidencyTier` legality |
| `replay.rs` | `RollbackPlan`, `ReplayPlan`, `AccessPlan` |
| `promotion.rs` | lifecycle types; `PreparedPromotion`/`QualifiedPromotion` are unforgeable |
| `exclusion.rs` | `LayerAttention`, architecture-derived `AccessPlan`, `ExclusionJournal` |
| `capabilities.rs` | admission gate + capability provenance |
| `mode.rs`, `config.rs` | `PromotionMode`, `SemanticPromotionConfig` |
| `session.rs` | session state machine |
| `engine.rs` | `KvEngine` delegation |
| `control.rs` | `SemanticKvControl` surface, including the enforcing lifecycle |

`crates/larql-kv/src/model_walk/` — the walk layer, a peer of `engines/`:

| file | concept |
|---|---|
| `graph.rs` | `ModelNode`, `ModelEdge`, `ModelGraph` |
| `plan.rs` | `WalkStep`, `ModelWalkPlan`, `ExpertPrefetchHint` |
| `planner.rs` | `WalkPlanner` — binds one qualified path |
| `validate.rs` | `WalkValidator`, `ValidatedWalkPlan` |
| `executor.rs` | `ObserveWalkExecutor`, `WalkTrace`, `PhysicalEffect` |
| `enforce.rs` | `EnforceWalkExecutor` — the same plan, performed |
| `events.rs`, `metrics.rs`, `error.rs` | phase events, counters, failure taxonomy |

---

## 5. Deviations from the source specification

| # | Spec said | Implemented | Why |
|---|---|---|---|
| 1 | `SemanticPromotionEngine<B>` generic over the base | holds `Box<dyn KvEngine>` | `EngineKind::build` already yields a boxed engine, and `BoundaryKvEngine` — this crate's existing wrapper — holds a concrete inner. A generic would need `impl KvEngine for Box<dyn KvEngine>` in `larql-inference` for no gain. |
| 2 | `RecordMaterialisation` has four variants | one variant; the other three reachable only through `MaterialisationPolicy`, which refuses | §7 says v1 implements `TokenSequence` only. Three unconstructible variants would be dead code claiming capability; `MaterialisationKind` still names all four so certificates cannot be confused across encodings. |
| 3 | `DerivedResult.operation: OperationExpr` | reuses `OperationSignature` | "Does this record answer that operation" becomes a structural comparison rather than two types that can drift. |
| 4 | `QualificationMetrics` as listed | plus `m_effective`; `ScoredObject` added | The scored object has to travel with the metrics or the congruence rule cannot be checked. `m_effective` makes the I10 window auditable. |
| 5 | `RetirementScope::AnswerScoped` unconstrained | constrained by `check_covered_by` | See §2. `AnswerBoundary::ExternalCommit` is covered by no finite scoring and is always refused. |
| 6 | Compose with `vindex2::OperationPlan` | dropped; see §5.1 | **`OperationPlan` does not exist** — not on this branch, not on `origin/worktree-vindex2`. There is nothing to compose with. |
| 7 | `AccessPlan` implicit | carries an explicit layer set | EXP-25 excluded the span on the 8 global layers only. Excluding it on a sliding layer whose window *does* cover the span would change what that layer sees, so the layer set is stated rather than assumed. |

### 5.1 The real integration seam

Two plans, kept apart, joined by a layer above both:

```text
semantic authority plan        expert-owner plan
  context-side objects           weight-side stable identities
  AuthorityId / RecordId         BankRef
  transitions + scopes           prefetch hints
                    \           /
                     ModelWalkPlan
                   (walk / planner layer)
```

`BankRef` is real (`larql-vindex/src/format/moe_manifest/bank_ref.rs`)
and is what `ExpertPrefetchHint` must carry — never raw expert indices.
The composed plan belongs in the walk/planner layer; it must not be
retrofitted into `larql-vindex`, and this engine must not grow toward it.

The walk layer is implemented at `crates/larql-kv/src/model_walk/` — a
**peer** of `engines/`, not a part of one. The dependency runs walk →
engine and never back; a source-scanning gate asserts that no file under
`engines/` mentions `model_walk`. `expert_hints` is present on the plan
and stays empty until Phase H.

**The governing rule.** The graph decides which semantic path is valid.
The KV engine materialises that path. VINDEX materialises its
expert-weight requirements.

```text
ModelGraph          what relationships exist
    ↓ WalkPlanner   choose one executable path
ModelWalkPlan       a typed, immutable step list
    ↓ WalkValidator whole-plan legality, once, up front
ValidatedWalkPlan   the only thing an executor accepts
    ↓ executor      physical steps against the engine
WalkTrace           what was visited, proposed and emitted
```

Two properties are worth naming because they were discovered by
building it rather than designed in:

- **The walk binds operands; the engine executes the binding.** A graph
  holds *alternatives* for an operand slot — for `transform_inc` both a
  canonical fact and a derived result are `OperandOf` the operation. The
  `OperationSpec` in the graph is an arity-correct template, and the
  executor rebinds `operands` from the walk's `SelectRecord` steps before
  opening the operation. Without this the engine would have been handed a
  menu and forced to choose, which is exactly the leak W8 forbids.
- **Where evidence stops before the stop decision, recovery is ordered
  before `FinishOperation`.** A payload-bounded walk emits
  `… Generate → Restore/Replay → FinishOperation`; a
  termination-scoped one emits `… Generate → FinishOperation →
  Restore/Replay`. The step *order* encodes that a payload-only
  replacement must hand the source back before the model decides to
  stop. This is structural, not a heuristic.

---

## 6. Test gates

Unit gates live beside each module. Cross-module gates are in `tests/`:

- **G0 — base parity.** Wrapped vs bare `StandardEngine`: prefill and
  three decode steps compared with `assert_eq!` on the returned
  `Array2<f32>`. Bit-exact.
- **G8 — FFN dispatch.** A recording `FfnBackend` confirms the wrapped
  engine issues the identical layer call sequence. Catches a wrapper that
  swallows `prefill_quant` and falls through to `prefill`.
- **Lifecycle.** Planning works in Observe; `prepare`/`commit`/`abort`
  refuse with `UnsupportedCapability` and leave the source attendable;
  phase events carry stable ids; `Unknown` operations can use records but
  cannot retire a source.
- **Qualification matrix.** The EXP-25 traces driven through
  `qualify_promotion`: `copy` qualifies through first termination,
  `transform_inc` qualifies for a payload-bounded scope and is refused a
  termination-bounded one, the same numbers relabelled as trajectory
  evidence are refused as self-inconsistent, `transform_rev`/derived
  cannot be certified at all, and a certificate does not transfer between
  operations.

Not yet covered (needs the phases below): G1 snapshot parity, G2
visibility, G3 causal replacement, G4 scope expiry under real decode, G5
replay, G6 compaction, G7 residency semantics.

### 6.1 Revised Phase E gates — boundary-scoped enforcement

Phase E is *not* "canonical promotion passes all operations through
termination"; the measurement forbids that. It is a test of the
permission system, which is a stronger gate than reproducing answers:

```text
E1  copy / canonical        commit through FirstTermination        SUCCEEDS
E2  +1 / canonical          commit through ExactPayloadLength(4)   SUCCEEDS
                            commit through FirstTermination        REFUSED
E3  reversal / canonical    commit through ExactPayloadLength(4)   SUCCEEDS
                            commit through FirstTermination        REFUSED
E4  +1 / derived            commit through FirstTermination        SUCCEEDS
E5  every certificate       a boundary beyond the scored object is
                            unrepresentable or ScopeNotQualified
```

E1–E4's *refusal* halves already pass at the policy layer (see the
qualification-matrix gates above). What Phase E adds is that they hold
against a live decode with the source actually hidden.

### 6.1a Walk gates W0–W8

The planner reaches the EXP-25 verdicts *from the measured traces* — the
fixtures carry per-position KL and let
`certificate_from_measurement` decide which scored object the numbers
support, so no fixture constant can assert a capability the experiment
does not.

```text
W0  planning is deterministic: same graph + operation → same fingerprint
W1  copy / canonical            walks through FirstTermination
W2  increment / canonical       payload-bounded; never emits termination
W3  reversal / canonical        payload-bounded; never emits termination
W4  increment, both forms       derived wins; canonical becomes fallback
W5  reversal / derived only     NoQualifiedWalk
W6  every promotion walk        carries recovery, ordered per §5.1
W7  observe execution           bit-identical decode vs the base engine
W8  plan/engine separation      one action per planned step, and no file
                                under engines/ references model_walk
```

Plus three negative gates on the validator: a hand-widened scope, a
promotion stripped of its recovery step, and a walk importing a record
that is not an operand all fail validation.

### 6.1b Phase B: the decisive gate

The strongest gate is no longer "compaction matches masking" — that is a
precondition. It is:

> The enforcing executor performs exactly one physical action per
> validated walk step, in the same order as observe mode, introducing no
> semantic decision.

`tests/enforce_gates.rs` runs the same `ValidatedWalkPlan` through
`ObserveWalkExecutor` and `EnforceWalkExecutor` and asserts
`proposed_actions`, `visited_nodes` and `traversed_edges` are equal.
`WalkTrace::physical_effects` is the only field allowed to differ, and it
is empty in observe by construction.

### 6.1c Competing-path costs — C0–C10

`model_walk/cost.rs` + `tests/cost_gates.rs`. Three strategies compete
for one operation: **A** keep the distant source, **B/C** promote a
record (canonical or derived), **D** replay a checkpoint then compute.
Only B/C replace a source, so only they need a certificate; A and D
leave the source authoritative and are feasible on structure alone.

Two rules are structural:

- **Qualification is feasibility, never a penalty.** An unqualified path
  is *absent* from the candidate set. No weight is large enough to make
  a semantically invalid route correct, and any tuning of the other
  weights could eventually outrun one.
- **Cost stays a vector.** `WalkCost` has nine terms and a
  `RankingPolicy` orders candidates without compressing them; the sort
  key is returned as a tuple so a reader can see which term decided.
  `pareto_frontier` keeps incomparable candidates visible.

| gate | result |
|---|---|
| C0 | an uncertifiable record is absent, not penalised |
| C1 | ranking reproducible across all three policies |
| C2 | promotion beats keeping the source once the span is large |
| C3 | reversal never selects the derived record |
| C4 | **residency changes the answer** |
| C5/C10 | greedy first-edge disagrees with the complete-walk oracle |
| C6 | estimate and observation stay separate objects |
| C7 | coverage is never traded for lower cost |
| C8 | recovery and pinned journal are attributed, not free |
| C9 | the selected candidate is directly executable |

Three things the gates found, all of which changed the code rather than
the test:

1. **The estimator was charging read once, not per decode step.** Decode
   is read-bound: every step attends over the whole resident cache. With
   the per-step model, promotion's arithmetic becomes what it physically
   is — a fixed cost up front to make every later step cheaper.
2. **It was using the certificate's payload length as the excluded row
   count.** Those are different quantities: the answer is 4 tokens, the
   retired span is whatever the source span is. `ModelGraph` now carries
   `source_span_tokens` and `record_tokens`, because a planner cannot
   cost a promotion without knowing how much it retires and how compact
   the replacement is.
3. **A per-token append constant was double-counting.** Appending a
   token *is* a decode step, already priced in the step count. At
   read-bound scale the constant swamped the saving promotion exists to
   buy.

C4 is the one that matters: the same query over the same graph selects
`keep_source` when the span is 512 rows out of 100 000, and promotion
when it is 512 out of 1 024. The planner is a physical optimiser, not a
static semantic chooser.

C6 recorded a result worth keeping: on the three terms it models
structurally — promoted tokens, pinned journal bytes, execution steps —
the estimator is **exact** against an enforcing run's `PhysicalEffect`s.
So a future non-zero error on those is signal, not expected drift. The
vectors still differ, because the estimator predicts read bytes and
latency that no `PhysicalEffect` reports; that is why the two stay
separate objects rather than being reconciled.

### 6.2 Phase B gates — required before any enforce mode

```text
B0   no-op parity              an empty exclusion changes no bits
B1   gather-oracle parity      compacted global K/V == index-gather oracle
B2   logical-position          promoted append uses the original absolute
     preservation              position, not the resident row count
B3   global-layer restriction  only layers where is_sliding_window_layer
                               == false are modified
B4   live-window refusal       an exclusion intersecting a live sliding
                               window returns UnsupportedCapability
B5   rollback parity           exclude → restore is bit-exact
B6   continued-decode parity   exclude → append neutral tokens matches the
                               gather-oracle execution
B7   repeated-compaction       logical identities still resolve after an
     safety                    earlier exclusion
B8   overlap refusal           overlapping active exclusions are refused
B9   FFN parity                every append and replay retains the supplied
                               FfnBackend call sequence
B10  row-count heterogeneity   per-layer cache lengths stay valid when
                               global and sliding layers differ
```

Status: **B2, B3, B4, B5, B8, B9, B10 pass.** B0/B1 have dispatch-level
precursors in `tests/position_seam.rs`. B6 and B7 remain uncovered — see
§6.3.

#### Scope: what the CPU per-layer path does and does not prove

Checked 2026-08-05, and it narrows every Phase B claim below.

`larql_inference::kv_dispatch::helpers` — the per-layer decode path these
gates run on — calls `attention_step` with **no per-layer window**, then
`clip_kv(handle, w)` with the *engine-level* window uniformly across every
layer. It never consults `arch.is_sliding_window_layer`; that lives in
`pipeline_layer.rs`, a different path.

So on this path every layer attends over everything resident, and Gemma's
global/sliding interleave is a **policy** distinction the promotion
wrapper applies rather than something attention enforces.

```text
proven      bit-exact semantic exclusion and ragged per-layer storage
            under the CPU per-layer executor

NOT proven  bit-exact exclusion under Gemma's production global/sliding
            interleaved attention semantics
```

B10's mechanics stand — layers can hold different extents and decode
safely — but "Gemma-shaped fixture" must not be read as full Gemma
attention parity. Closing that is the next production seam, and R8d below
is the gate that would do it.

#### B10 — heterogeneous per-layer row counts

The other Phase B gates run on an architecture whose layers are all
global, where "exclude the global layers" happens to mean all of them and
the cache stays rectangular. That is not the measured shape: EXP-25
excluded a span from 8 global layers of 48.

`tests/heterogeneous_gates.rs` runs Gemma-3's real interleave — a 6-layer
fixture with a deliberately narrow 4-token sliding window so a planted
span falls outside it. Layers 0–4 slide, layer 5 is global, and only
layer 5 is excluded. The answer to the question that could have sunk the
approach:

> **Decode survives the ragged cache.** Per-layer attention reads each
> layer's own resident rows; nothing reconstructs position or geometry
> from a shared length. Appends grow every layer and preserve the
> raggedness; the journal restores every layer exactly.

The reporting side needed a real fix. `memory_bytes` already summed
per-layer and stays correct. **`window_tokens` read `handles.first()`** —
and on this architecture layer 0 is *sliding*, so after a global-only
exclusion it would have reported the unexcluded length. It now derives
from the engine's own logical position and declared window, which is
identical in every rectangular case and truthful in the ragged one.

`larql_inference::kv_engine::KvExtent` replaces the scalar as the
description of cache shape, keeping four notions apart: logical position
(one per session), resident rows (per layer), visible rows (wrapper
policy, not represented here), and window size (declared, not measured).
`uniform_resident_rows()` returns `None` under heterogeneity so a caller
needing one length refuses or falls back to the logical position rather
than guessing from layer 0.

`CpuQ4kCacheHandle::cached_len()` keeps its layer-0 assumption, now
documented: it is the *coarse* handle, and `PerLayerKvAccess` is not
offered on the coarse path, so a heterogeneous coarse cache is currently
unreachable.

### 6.4 B1/B6 — the exclusion oracle

`tests/exclusion_oracle.rs`, on the Gemma-shaped interleaved fixture,
compares full-vocabulary logits through the real `StandardEngine`
forward pass:

- **candidate** — excise the span once, then decode over the compacted
  cache;
- **reference** — keep the full cache as the authority and materialise
  the gathered view per step: excise, decode one token, splice back.

Both present attention with the same per-layer visible rows, in the same
order, at the same logical position — by different machinery. The
theorem:

> Compacting once and decoding N times is bit-identical to compacting,
> decoding and restoring N times, at the level of full-vocabulary logits.

| gate | what it fixes |
|---|---|
| O0 | an excise/splice round-trip is a no-op — the harness's null hypothesis |
| O1 | immediate post-exclusion logits bit-identical |
| O2 | parity over 3 neutral continuation tokens, step by step |
| O3 | parity throughout a 5-token record encoding |
| O6 | **control**: a mismatched span *does* diverge |
| O7 | identical `FfnBackend` call sequences in both arms |
| O8 | extents differ by exactly the excluded span, logical position equal |

All bit-exact — no tolerance anywhere. O0 and O6 are the two that stop
the rest being vacuous: without O0 both arms could be equally corrupted
by the round-trip itself; without O6 both could be ignoring the
exclusion mechanism entirely.

**Named limit.** The gather here is *materialised* (rows removed and put
back) rather than *indexed* (rows retained, attention told which to
read). A true indexed gather needs a per-layer visibility parameter on
`attention_step`, which does not exist and should not be added as a
production masking feature merely to serve a test. The remaining gap —
would an indexed gather agree with a materialised one — is covered for a
single step at dispatch level by `position_seam.rs`. Closing it at engine
level is a Phase D question.

### 6.3 What Phase B does not yet cover

Stated plainly, because the gates that pass could otherwise read as
broader than they are.

- **Sparse position + windowing is incorrect on the decode path.** The
  dispatch's `clip_kv` keeps the tail *N physical rows*, so across a
  positional hole it retains rows thousands of positions old inside a
  small window (gate `r7b`). Sparse replay is validated only on the
  unwindowed path.
  `larql_inference::kv_row_positions::KvRowPositions` is the row-age map
  that fixes it. It is now **wired through storage** — the engine owns
  one, every append/excise/splice/checkpoint keeps it congruent, and
  `clip_layer_to_logical_window` selects by logical age — but the decode
  path has **not** been switched onto it. R8 splits accordingly:

  ```text
  R8-storage  (LANDED)
  R8e  excise/splice/checkpoint round-trip preserves row positions

  R8-policy   (LANDED as an operation — see the scope note below)
  R8a  gap > window   no pre-hole row survives on a sliding layer
  R8b  gap < window   only rows still in logical range survive
  R8c  global layer   pre-hole rows stay visible across the same hole
  R8d  decoded-then-excised == offset-replay under real mixed policy
  ```

  `window_policy::apply_window_policy` reads the architecture's own
  `is_sliding_window_layer` classification and clips sliding layers by
  logical age while leaving global layers whole. R8d passes bit-exactly
  in K, V and retained positions at gaps 2/4/6/16.

  **Scope.** These license "the mixed policy, applied to a cache,
  produces the extents the architecture asks for, by whichever physical
  route the cache was reached". They do **not** license "production Gemma
  decode is logical-age correct": `kv_decode_step_via_dispatch` still
  calls `clip_kv` with the engine-level window uniformly on every layer,
  and routing decode through the policy is a separate, behaviour-changing
  step that these gates qualify but do not perform.

- **B1/B6's oracle is materialised, not indexed.** See §6.4 — the
  theorem proved is real and logits-level, but it is not
  "attention-level index-gather against a retained full cache", which
  would need a per-layer visibility parameter in
  `KvDispatch::attention_step` that deliberately does not exist.
- **B7 is not covered.** A second exclusion after an earlier one is
  refused outright (one journal, spec §7.5), so repeated-compaction
  identity resolution is deferred rather than solved. `read_kv_row_at`
  still confuses physical row with absolute position.
- **Coarse and Metal paths refuse**, they do not work.
  `per_layer_kv_mut` returns `None` for `PrefillDispatchMode::Coarse`, so
  preparation fails with `UnsupportedCapability`.
- **`ReplayAuthority` refuses.** The journal is resident, not durable, so
  an enforcing walk that reaches a replay step returns
  `ReplayUnavailable` after aborting cleanly.
- **Nothing is reclaimed.** `source_bytes_reclaimed` stays 0 and
  `net_bytes_saved()` is ≤ 0 — a gate asserts it. Resident rollback buys
  correctness headroom at a memory cost; Phase F is what makes it a
  saving.

---

## 7. Substrate work the next phases need

`Observe` is the ceiling because none of the enforcing hooks exist.

**Phase B — `logical_source_masking`.** `larql_compute::KvDispatch` has no
non-contiguous attention mask: `clip_kv` drops a prefix and
`attention_step_windowed` takes a contiguous recent window.

The EXP-25 reference implementation is an index-gather over the assembled
K/V (`mx.take` on a `setdiff1d` of the span) with **RoPE already applied
at write time and positions never renumbered** — the remaining rows keep
their original positions and a positional hole is left behind. That means
physical row removal is *semantically equivalent* to logical masking for
a contiguous span, so Phase B can be built at the engine layer from
`read_kv_to_host` → drop rows → realloc + re-append, with **no kernel
change**. The exclusion must be restricted to global-attention layers
(`Architecture::is_sliding_window_layer` in `larql-models` provides the
split) and must refuse when the span intersects a live sliding window.

### 7.1 Seam audit: logical position vs resident row count

Surveyed 2026-08-05, because Phase B is only equivalent to masking if
nothing re-derives position from the number of retained rows.

**The separation already holds on the CPU per-layer path.**

- `StandardEngine::abs_position` is engine-owned: set from
  `token_ids.len()` at prefill, `+= 1` per decode, never read off a
  handle.
- `KvDispatch::attention_step(…, abs_position, …)` takes it as an
  explicit parameter and uses it only to RoPE the *new* token; cached K
  rows carry their own baked phase.
- `append_kv`'s `abs_position` is documented as informational on CPU.
- Decode attention is one query against all resident rows — no causal
  mask is built from a row count.
- `cached_len()` has **zero** production consumers on the position path.
  Its only two non-test uses (`standard.rs:142`, `:538`) are
  `memory_bytes()` and `window_tokens()` — reporting, not semantics.

`tests/position_seam.rs` pins this: compacting a span attends identically
to never having appended it; `abs_position` demonstrably changes the
result; and appending after compaction uses the untouched logical
position while `cached_len()` has shrunk.

**Three places it does not hold**, in ascending severity:

1. **`read_kv_row_at` indexes by physical row while documenting absolute
   position** ("cache rows are indexed by absolute stream position
   (prefill row 0 onward)"). After a compaction those diverge *silently*
   — the call returns the wrong row rather than failing.
   `WindowedCheckpointEngine::close_window` depends on it. Phase B needs
   either a logical→physical map on the handle or explicit invalidation.
   This is **B7**.
2. **`CpuQ4kCacheHandle::cached_len()` reads layer 0 and assumes every
   layer agrees** (`.map(|(k, _)| k.shape()[0]).next()`). Compacting
   global layers only makes that a lie, and since Gemma-3's layer 0 is
   sliding it would report the *un*compacted length. The good news for
   **B10**: the cache is `Vec<Option<(Array2, Array2)>>` per layer, so
   heterogeneous lengths are *representable* — only the reporting is
   wrong. A contained fix, not a representation change.
3. **The coarse path has no per-layer granularity at all, and the Metal
   handle is opaque.** `MetalCoarseHandle::cached_len()` returns `0`
   ("backend-side state; not exposed through the handle") and does not
   implement `read_kv_to_host`. Host-side row removal is therefore
   **CPU per-layer only** — and the coarse path is what `prefill_quant`
   prefers when unwindowed. Phase B must *refuse* on coarse rather than
   silently no-op. Reaching EXP-25's regime on Metal needs either the
   per-layer path or a Metal-side exclusion hook; that choice is not made
   here.

### 7.2 Capability provenance — as implemented

`KvEngine` gained one optional accessor:

```rust
fn per_layer_kv_mut(&mut self) -> Option<&mut dyn PerLayerKvAccess> { None }
```

`PerLayerKvAccess` offers two mechanical operations — `excise_kv_rows`
and `splice_kv_rows` — and nothing else. **Offering them is not a masking
claim.** Which rows, on which layers, under what evidence, with what
rollback are all wrapper decisions. `StandardEngine` implements the
accessor and still reports `logical_source_masking: false`; it returns
`None` on the coarse path, where there is no per-layer granularity.

The wrapper then declares what it provides for itself:

```text
caller_supplied_ffn_backend        BaseEngine
logical_source_masking             SemanticWrapper
append_while_masked                SemanticWrapper   (ordinary decode)
snapshot_restore                   SemanticWrapper   (exclusion journal)
stable_token_identity              BaseEngine
source_replay                      — nobody, yet
```

`PromotionEngineCapabilities::wrapper_exclusion()` is that row set.
`EngineKind::build` still constructs with `ffn_only()`, so
`--engine semantic-promotion:mode=enforce` continues to refuse: a CLI
caller cannot assert on the wrapper's behalf.

### 7.3 `AccessPlan` must be constructed, not trusted

The caller must not hand in an arbitrary layer set. Build it from the
architecture — global layers get the exclusion; a sliding layer whose
live window intersects the span is a refusal, and one whose window does
not is a no-op because the source is already absent there. The plan
carries an architecture fingerprint, and execution re-validates the
fingerprint, the source epoch, the logical range, the absence of any
newly-intersecting sliding window, and that no prior compaction
invalidated the logical→physical mapping.

### 7.4 Phase B rollback: an exclusion journal, not a snapshot

The base lacks snapshot/restore, but the wrapper can give *exact*
rollback for its own operation by retaining the removed rows:

```text
drop rows from the active cache
retain the exact removed K/V + prior logical position + epoch
append the promoted record
abort  → splice the rows back exactly
commit → hold the journal until the boundary expires
```

This saves no memory yet — the rows still exist in rollback storage — but
it establishes correct semantic exclusion with exact recovery, which is
the right Phase B milestone. The journal then grades into the later
modes: in RAM → resident rollback; persisted → cold replay; discarded →
only once durable replay authority exists.

### 7.5 Concurrency: one of everything, at first

The first enforcing version allows one active operation, one prepared
promotion, one excluded contiguous span, one rollback journal. Refuse
overlapping promotions, nested exclusions, operation changes during
preparation, multi-source replacement, and multiple independently
compacted spans. Those come after the basic lifecycle is proven.

**Phase C — `snapshot_restore`.** `read_kv_to_host` + `alloc_kv_buffer` +
bulk `append_kv` gives an O(full-KV) round trip, matching what the
harness did. G1 requires bit-exactness, which holds only if `append_kv`
takes post-RoPE rows — the `position_seam` gates establish this for the
CPU path.

**Phase D — shadow qualification.** Needs C plus a forked branch.
`ShadowComparison::into_certificate` and the controls check are already
implemented; what is missing is the machinery to run the two arms.

**Phase F — `source_replay`.** Durable checkpoints. Until then
`ReplayPlan::KeepResidentRollback` is correctly reported as
non-durable, and `EnforceResidentRollback` shows a *negative*
`net_bytes_saved` — it costs memory and buys correctness headroom.

**Phase H — K3.** See §5.1 for the seam. `SemanticPhase` already emits the
join points (`PromotionEncoding` → prefetch, `PayloadGeneration` →
retain, `FirstTermination` → switch).

**`ConsumptionMode` is the stronger prefetch signal.** A phase event says
*where* the session is; the certificate's consumption mode says *what
computation remains*:

```text
OperandForOperation          the operation is still to run
                             → compute-expert envelope

AnswerWithoutReapplication   computation is discharged
                             → readout / termination envelope
```

EXP-25CF showed the *same record* taking both modes at different
distances, so the envelope a promotion implies is not a property of the
record and cannot be read off `DerivedClaimActive` alone. The planner
should therefore hand K3 the qualified capability edge — mode **and**
regime — rather than the phase. That is an argument for `ExpertPrefetchHint`
being derived from the selected candidate's certificate, not from the walk
step alone.

---

## 7.6 Composition and reached-operand binding: unproven

A 4K smoke run of the multi-hop promotion experiment returned a negative
result. Recorded here because it bounds what this engine may claim, and
because it is the trigger condition B7 was waiting on.

- **`missing_middle` passed in both record orders**, with tiny KL. The
  answer was available despite an incomplete promoted path, so a passing
  arm cannot be attributed to the model *composing* the promoted
  records.
- **`operand_only_corrupt` flipped the answer with no resolvable path.**
  The promoted operand behaved as a nearby dominant statement, not as a
  node reached and bound by a walk.
- **Stepwise walking showed no capability beyond closure replay.** Where
  replay and walk agree at every readable hop, sequential promotion has
  not been shown to perform a computation that one-shot retrieval of the
  closure cannot.

### Counterfactual absence: measured (EXP-25CF, 2026-08-05)

The control below was run. `chris-experiments/rsl/exp25cf_counterfactual_promotion.py`,
Gemma-3-12B-it, at 4K and 64K. One checkpoint before the planted span; two
histories replayed over it — source tokens vs an equal-length decoy — then the
same tail, with the span rows excluded in *both* at the query boundary.

- **Payload: 12/12.** Canonical promotion passes in every cell — 3/3
  counterfactual and 3/3 original, at both distances, with the span 49K back
  at 64K. Counterfactual pad-only fails every row; corrupt steers every row.
  So EXP-25's payload result is **not** an artefact of propagated source
  state, and it survives the long-distance regime.
- **Trajectory: structural.** All 12 canonical arms peak on the *first
  termination token*. Only original/`copy` passes at both distances. The
  original history also fails 2 of 3, so the stop-state defect is attributable
  to neither counterfactuality nor distance — it is a property of the promoted
  record.

Net effect on §2: the certificates EXP-25 supports are unchanged, and now rest
on a stronger footing. Independent replacement is licensed to
`ExactPayloadLength`; `FirstTermination` remains earned only where measured.

### Retirement is not counterfactual absence

The caveat applies to every result in this document, including the ones
that passed. Once later positions have attended to a planted span, its
information propagates into their K/V and residual states. Retiring the
original span removes it from *present attention*; it does not remove
those descendants.

```text
retired now  ≠  never visible during prefill
```

Distance attenuates the effect but cannot logically eliminate it. So a
long-context arm tests whether propagated information *remains
behaviourally usable at range* — not whether promotion works when the
information was genuinely absent from the continuation state. The
decisive control is a counterfactual suffix prefill in which no later
cached position ever saw the span, comparing:

```text
3. counterfactually absent span
4. counterfactually absent span + promoted closure
```

The engine already carries the machinery for the cheap form of that:
checkpoint immediately before the planted chain and replay the suffix
under different visibility, rather than rebuilding the context each time
— `ReplayPlan::BoundaryReplay` is exactly that shape, and it is the
strongest argument yet for building Phase F's durable checkpoints.

### R0–R5: the replay infrastructure the counterfactual needs

`semantic_promotion/checkpoint.rs` + `model_walk/tests/replay_gates.rs`.
Minimum durable `BoundaryReplay`: capture full per-layer K/V and the
logical position at a boundary, restore it (which *shortens* the cache —
splice cannot, which is why `replace_layer_kv` was added), and replay a
suffix under a named visibility.

```text
[ prefix ][ chain ][ filler ][ query ]
          ^ checkpoint
```

`SuffixVisibility::ChainReplacedByDecoy` replays decoys over the chain's
positions rather than omitting them. Omission would shift every later
position and change its RoPE phase, making the branches incomparable; a
decoy of the wrong length is refused with that stated as the reason.

| gate | result |
|---|---|
| R0 | a checkpoint records the boundary it claims |
| R1 | **two replays of the same checkpoint under the same visibility are bit-identical**, in logits and K/V |
| R2 | replay reproduces the uninterrupted run, after unrelated decoding |
| R3 | the counterfactual branch is positionally aligned with the reference |
| R4 | **control**: the counterfactual branch *does* diverge at the query |
| R5 | restore shortens the cache back to the boundary |

R1 is the null case the whole experiment rests on — without it, a
difference between branches could be replay noise. R4 is its converse:
if decoying the chain changed nothing, the branch would not be
counterfactual and every downstream comparison would be vacuous.

Not built, deliberately: a closure index, automatic join
materialisation, multi-promotion planning, B7's general row map, learned
traversal. The experiment decides which of those deserves to exist.

### Consequence for the primitive

The evidence currently favours

```text
retrieve a validated closure → bind it explicitly → execute or verify
```

over

```text
promote edge 1 → promote edge 2 → promote edge 3 → assume composition
```

— closer to a **materialised join result** than an interpreted
traversal. That is still useful for acceleration: a graph index that
retrieves a small causally sufficient closure beats making the model
rediscover it through a full forward pass. It is not built here, and
should not be until the counterfactual control settles.

`tests/composition_gates.rs` pins the posture: no planned walk chains
promotions, a fallback is an alternative rather than a second hop, and
every emitted strategy holds at most one exclusion with exactly one
recovery. The engine happened not to support chaining; these gates make
that a decision. A future multi-hop planner must delete a test whose
name states why.

**B7 stays deferred.** Its trigger was "EXP-26 demonstrates genuine
multi-hop promoted traversal". That did not happen, so a general
logical→physical row map would be speculative plumbing for a capability
with no evidence behind it.

---

## 8. What version 1 may claim

> Given an explicitly registered canonical record, source span,
> operation and exact qualification certificate, the engine plans a
> promotion, checks the certificate against the live regime, and refuses
> any retirement scope the certificate's scored object does not cover.
> In `Observe` mode it wraps an exact decode engine bit-identically.

Narrower, and accurate as of the smoke run: **the model supports three
controlled semantic hops in the original context, while promoted-record
composition and reached-operand binding remain unproven.**

It may **not** claim: automatic context compilation, universal record
sufficiency, permanent source deletion, derived-result safety across
operations, multi-hop promoted traversal, reached-operand binding, or
improved K3 expert prediction. Nor may any passing result here be read
as counterfactual — see §7.6.
