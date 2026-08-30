# Runtime model lifecycle — design notes

Status: **all four rungs of §6 are done.** `ModelSet` is
interior-mutable (rung 1), `V3Model` has an in-flight counter (rung
2), `RouterTopology` freezes the 0↔1 invariant (rung 3), and
`POST`/`DELETE /v1/runtime/model` are live on `single_model_router`
(rung 4, `routes/runtime_lifecycle.rs`). No event stream yet — polling
`/v1/runtime` (`routes/runtime.rs`, `runtime_stats.rs`) remains
sufficient, per §5's non-goals.

## 1. What exists today (grounded in code, not assumption)

- ~~`AppState.models: Vec<Arc<LoadedModel>>` and `v3_models:
  Vec<Arc<V3Model>>` are populated once... plain `Vec`s, not
  interior-mutable~~ **Done (rung 1).** Both now live in one
  `ModelSet { models, v3_models }` behind a single
  `AppState.model_set: RwLock<ModelSet>` (`state/model_set.rs`) —
  deliberately one lock, not two independent ones, so a reader can
  never observe the V2 and V3 registries at two different points in
  time. Every resolution method (`model()`, `served()`,
  `is_multi_model()`, `first_model()`, `models_snapshot()`) takes the
  read guard, finds and clones the `Arc` it needs, and releases the
  guard before returning — no inference work and no `.await` ever
  happens while it's held. `ServedModel` dropped its lifetime
  parameter accordingly (it now owns its `Arc`). Bootstrap still
  constructs the set exactly once, no user-visible behavior changed —
  there is still no mutation API. **The remaining seam is exactly what
  §2 says it is: nothing yet writes to this lock after boot.**
- Route dispatch already goes through `AppState::model(id)` /
  `AppState::served(id)` (`state/model_set.rs`) at request time,
  searching the snapshot by id. So a future load/unload **doesn't need
  to touch router or handler code at all** to be picked up — the seam
  really is this narrow.
- **The router topology is a separate, bigger seam.** `bootstrap/mod.rs:340-353`
  picks `single_model_router` or `multi_model_router` **once**, based
  on `state.is_multi_model()` (`models.len() + v3_models.len() > 1`)
  at boot. Those two routers have different route tables (multi-model
  adds `/v1/{model_id}/...` prefixed paths). Consequences:
  - 0 models and 1 model are **both** "single" mode — so an idle
    server picking up its first model, or a single model being
    swapped for a different one, never crosses the single/multi
    boundary. **This is the tractable first slice.**
  - Going from 1 loaded model to 2 **would** flip which router
    variant is correct. Axum's `Router` isn't swappable in place
    without extra machinery. **Loading a second concurrent model into
    an already-bound single-model server is a materially bigger
    change than "swap the bound model" — don't scope them together.**
- `LoadedModel.weights: OnceLock<RwLock<ModelWeights>>` +
  `weights_init: Mutex<()>` solve *lazy first-load* single-flighting
  within one already-registered model. They have no "unset" — a
  `LoadedModel` cannot have its weights freed while keeping the rest
  of the struct alive. **Unload is necessarily "drop the whole
  `Arc<LoadedModel>`/`Arc<V3Model>`", not a finer-grained operation.**
- Because generation handlers already clone the model's `Arc` for the
  duration of a request (e.g. `model_arc = model.clone()` before
  `spawn_blocking` throughout `routes/openai/*`), **removing a model
  from `AppState.models` is memory-safe immediately** — Rust's
  refcounting keeps any in-flight request's own clone alive until that
  request finishes, with no coordination required. The drain pattern
  below exists for *policy* (don't claim "unloaded" — or start loading
  a replacement — while the old one might still be resident), not for
  safety.
- **In-flight accounting is inconsistent across the things that would
  need it.** `LoadedModel.requests_in_flight: AtomicU32` exists but is
  walk-ffn/grid-shard-scoped only (its own doc comment says so; it's
  what GT6 drain reads — see below), and it's a raw `pub` field any
  caller can mutate directly. `RuntimeRecorder.active_requests` is
  OpenAI-generation-scoped but **server-wide**, not per-model.
  ~~`V3Model` has no in-flight counter of any kind today~~ **Done
  (rung 2).** `V3Model::requests_in_flight() -> u32` now exists,
  backed by a *private* counter — the only way to change it is
  `V3GenerationGuard`, entered once as the first statement of
  `generate_v3_request` (the same choke point that already carries V3's
  timing) and dropped on every exit, `Ok` or `Err`. A concurrency test
  (`test_vindex3_serve.rs::v3_generation_in_flight_counter_reflects_genuine_concurrency`)
  proves the counter reads `1` *while* a real generation is mid-flight
  on another thread, not just before/after — a before/after check alone
  would pass even if the guard were scoped wrong. This is stricter than
  `LoadedModel`'s equivalent (private + guard-only vs. a raw mutable
  `pub` field) — a deliberate improvement, not an inconsistency to
  paper over; `LoadedModel`'s stays as-is since GT6 already depends on
  its exact shape.
- **A drain-then-signal pattern already ships** — GT6 in `announce.rs`
  (`drain_requests`, `DRAIN_TIMEOUT`, `DroppingMsg`): on `UnassignMsg`
  from the grid router, stop accepting new shard work, poll
  `requests_in_flight` every 100 ms up to a timeout, then announce
  `Dropping`. **This is the right shape to imitate for local unload**
  (stop resolving new requests → poll a counter with a timeout →
  proceed) but it is not directly reusable code: it's about leaving a
  *distributed grid's* routing table, not about freeing a *local*
  `Arc`. There is no equivalent "stop resolving new local requests"
  primitive today — that's new.
- **No graceful shutdown exists at all.** No SIGTERM/SIGHUP handling,
  no quiescing, anywhere in `bootstrap/`. The server runs until killed.
  A lifecycle endpoint would be the *first* thing in this codebase
  that needs "stop routing new work to X, then wait" outside the grid
  context.
- **Session and N1 KV state are keyed by model-id string, not by
  `Arc` identity.** `ResponseKvCache::take` (`response_kv/mod.rs:202-215`)
  already refuses a resume when `entry.model_id != model_id` — a real,
  existing guard. But it can't detect "same id, reloaded with
  different weights" (a swapped quantization under the same name,
  say) — the guard only compares strings. **Any lifecycle design that
  lets a model id be reloaded (not just removed) must sweep session +
  KV-cache entries for that id as part of the transition**, or a
  resumed KV state can silently pair with the wrong weights.
- `/v1/runtime`'s `memory.resident_bytes` is `getrusage`'s **peak**
  RSS (documented as such in `runtime_stats.rs`), which is monotonic
  for the process lifetime. **Once unload exists, this becomes
  materially misleading**: unloading a model will not move this number
  down, ever, even if the memory really was freed. Not fixing this
  now — flagging it because it's a direct, foreseeable consequence of
  work already merged, not a new speculative gap.

## 2. Scope for the *first* lifecycle cut

Given the router-topology seam above, the first tractable slice is
**single-bound-model lifecycle only** — matches the "tiny Mac app"
target anyway (one local model, not a fleet):

- load into an idle (zero-model) server
- swap the bound model for a different one (implies unload-then-load)
- unload back to idle

**Explicitly out of scope for the first cut** (bigger, separable
seams): holding two independently-loadable models at once on one
server (router topology change), any multi-model `/v1/{id}/...`
lifecycle, and anything about the grid/router protocol.

## 3. State machine

One state machine per **bound-model slot** (today: exactly one slot,
since multi-model dynamic loading is out of scope). `/v1/runtime`'s
`model` field already reports `null` in every state except `ready` and
`generating` — that's unchanged.

```text
                                   ┌────────────────────────────┐
                                   │                            │
                                   ▼                            │
                              ┌─────────┐   load ok        ┌─────────┐
             load requested   │ loading │ ───────────────► │  ready  │
        ┌─────────────────────┤         │                  │         │◄───┐
        │                     └────┬────┘                  └────┬────┘    │
        │                          │ load failed                │         │ generation
        │                          ▼                             generation  completes
   ┌─────────┐               ┌─────────┐                        starts    │
   │  idle   │◄──────────────┤ failed  │                         │        │
   │         │  surfaced,    │(logged, │                         ▼        │
   └────┬────┘  slot freed   │ no slot)│                   ┌────────────┐ │
        │                    └─────────┘                   │ generating │─┘
        │ (already idle —                                  └─────┬──────┘
        │  no-op / 409,                                          │
        │  see §4)                                     unload requested
        │                                                        │
        │                          unload requested (from ready) │
        │                                        ┌───────────────┴──┐
        │                                        ▼                  ▼
        │                                  ┌────────────┐    ┌────────────┐
        └──────────────────────────────────┤ unloading  │◄───┤ unloading  │
                     drain complete,        │ (draining) │    │ (draining, │
                     Arc dropped            └────────────┘    │  cancel    │
                                                                │  requested)│
                                                                └────────────┘
```

Two notes the diagram can't carry on its own:

- **`unloading` while `generating`** is not a separate state — it's
  `unloading` with the in-flight-generation drain still counting down.
  The distinction the user's edge-case list asks about
  ("cancelled generation" vs "let it finish") is a **policy knob on
  the unload call**, not an extra state: drain-to-completion (default,
  matches GT6) vs. best-effort cancel. Today's generation loops have
  no cooperative-cancel hook (the `/v1/infer` timeout path documents
  exactly this: on timeout it drops the `JoinHandle` and lets the
  blocking thread finish in the background regardless — see
  `routes/openai/completions.rs`'s timeout comment). So "cancelled
  generation" on unload would, for now, mean the same thing the
  infer-timeout already means: stop *waiting* for it, not stop it
  running. Actually killing an in-flight blocking generation thread
  is a separate, harder problem this design does not solve.
- **`failed` is not sticky.** A failed load returns the slot straight
  to `idle` (with the error surfaced to the caller) — there is no
  persistent "broken" state to recover from, because nothing was
  committed to `AppState.models` on failure.

## 4. Edge cases, resolved explicitly

| Case | Resolution |
|---|---|
| Load B while A is loaded (single-slot scope) | **Rung 4, as built:** no server-side swap operation — `POST` while a different model is bound is refused outright (409, `LoadDecision::Refuse`), naming the bound model and pointing at `DELETE /v1/runtime/model`. The caller sequences unload-then-load itself; the server never holds, or attempts to hold, both. (Supersedes this row's earlier "server does unload(A)-then-load(B) as one sequenced operation" framing — no such internal sequencing exists.) |
| Unload while generation active | `ready → unloading`: stop resolving *new* requests to this slot immediately (an instant Vec/slot mutation); poll the model's in-flight counter with a timeout (GT6 shape); drop the `Arc` once it hits zero or the timeout elapses. See §1 on V3's missing counter — this is the blocking prerequisite for V3 unload specifically. |
| Load while a load is already active | Single-flighted per slot, same shape as `LoadedModel.weights_init` today but at the *admin* layer, not the weights layer — a second concurrent load call on the same slot is rejected outright, not queued (queuing hides operator mistakes; better to fail loud). |
| Failed load | Slot returns to `idle`; nothing was written to `AppState.models`; error surfaced. No retry state to manage. |
| Failed unload (drain timeout) | **Rung 4, as built: policy (b), not the (a) this row originally recommended.** The model is put back in `ModelSet` exactly where it came from, `lifecycle` reverts to `Ready`, and the call returns a 409 — the drain timeout fails closed rather than force-dropping the `Arc`. Reasoning: this codebase has no cooperative generation cancellation (§3), so force-dropping on timeout would let the server claim "unloaded" while a generation might still be reading the weights through its own clone — an honest-but-surprising state for a caller who just got told the unload succeeded. Fail-closed keeps the invariant simple: the server never reports a mutation as done unless it verified the model's in-flight count actually reached zero. |
| Cancelled generation (mid-unload) | See §3 — "stop waiting," not "stop running," matches the existing infer-timeout precedent. Don't promise more than the codebase can deliver today. |
| VINDEX2 vs VINDEX3 lifecycle | Same state machine, different mechanics: V2 unload drops `Arc<LoadedModel>` (weights may or may not be loaded yet, per `weights: OnceLock`); V3 drops `Arc<V3Model>` (operands are lowered at bind time — `Vindex3Runtime::prepare`, see `vindex3.rs` module docs — so a V3 "load" is heavier up front and a V3 "unload" has nothing lazy left to *not* free). The important asymmetry is the in-flight counter gap noted in §1, not the drop itself. |

## 5. Explicit non-goals for this pass

- No atomic A→B replacement — a client swaps a model by calling
  `DELETE` then `POST`; `POST` while a different model is bound is
  refused outright, naming the unload step. (This supersedes §4's
  "swap request" framing above, which predates rung 4's actual
  implementation — see rung 4's note in §6.)
- No event stream (`runtime.model.loading` etc.) — polling
  `/v1/runtime` is sufficient until a real client (the Mac app)
  demonstrates it isn't.
- No multi-model dynamic loading (router-topology change, §1/§2).
- No memory-accounting fix for the peak-RSS-after-unload confusion
  (§1, last bullet) — noted for whoever picks this up next, not solved
  here.
- No cooperative generation cancellation — "drain or force-drop after
  timeout" is the ceiling of what's honestly deliverable right now.

## 6. What actually has to land first, in order

1. ~~Make `AppState.models` / `v3_models` interior-mutable for a single
   slot~~ **Done** — `ModelSet` behind one `RwLock` (§1).
2. ~~Add the missing V3 in-flight counter at the `generate_v3_request`
   choke point~~ **Done** — `V3Model::requests_in_flight()` (§1).
3. ~~Before the two endpoints: settle §7's `is_multi_model()` /
   router-topology invariant explicitly~~ **Done** — `RouterTopology`
   frozen at boot, `AppState::validate_lifecycle_mutation` enforces
   0↔1 (§7).
4. ~~Only then: the two endpoints, built directly against the state
   machine in §3 and the edge-case table in §4~~ **Done** —
   `POST`/`DELETE /v1/runtime/model` (`routes/runtime_lifecycle.rs`),
   wired on `single_model_router` only. `decide_load`/`decide_unload`
   (`state/lifecycle.rs`) are the pure decision functions behind the
   two handlers; the drain-timeout fail-closed policy is §4's updated
   row above. On every successful unload, `SessionManager::drop_sessions_bound_to`
   and `ResponseKvCache::drop_owned_by_model` purge state tied to the
   unloaded model's id (§1's id-reuse trap) — unconditionally, not
   only on a subsequent reload.

## 7. `is_multi_model()` must not go dynamic by accident — resolved

`AppState::is_multi_model()` (`state/model_set.rs`) reads the *current*
`ModelSet` and returns `models.len() + v3_models.len() > 1`. Today
that's safe *only* because nothing ever changes the set after boot —
the value `is_multi_model()` computes at request time is, in practice,
identical to the value `bootstrap::serve` computed once to choose
`single_model_router` vs. `multi_model_router` (§1's router-topology
seam). Rung 3 breaks that coincidence the moment a mutation endpoint
exists: the router variant stays whatever axum built at boot — static,
un-swappable without extra machinery — while `is_multi_model()` would
start reporting the *current*, potentially different, model count. A
handler that trusts `is_multi_model()` to mean "which router shape is
active" would then be reading a lie.

Two ways to close this before any mutation endpoint lands (either is
acceptable; picking neither is not):

- **Enforce the 0↔1 invariant at the mutation boundary.** A load/unload
  call that would ever make `models.len() + v3_models.len()` exceed 1
  is rejected outright — not deferred to a "multi-model dynamic
  loading" follow-up, refused *at the point of mutation*, with a clear
  error naming the reason. This keeps `is_multi_model()`'s current
  reactive definition truthful, because the thing it worries about
  (crossing into multi-model territory) is structurally impossible.
- **Or separate the two questions the name currently conflates.**
  Freeze a `boot_router_topology: SingleModel | MultiModel` fact
  (computed once, matching what `bootstrap::serve` actually built),
  and give `is_multi_model()` — or a differently-named method — a
  documented answer to "how many models are bound *right now*" that
  makes no claim about routing. Callers that care about routing shape
  read the frozen fact; callers that care about current count read the
  live one; nothing reads one and assumes it means the other.

**Resolution: both, combined.** `RouterTopology::SingleModel |
MultiModel` (`state/lifecycle.rs`) freezes the boot-time fact —
`RouterTopology::for_boot_count` is called once, from the same total
`bootstrap::serve` uses to pick the axum `Router` variant, so the two
can never disagree. `AppState::validate_lifecycle_mutation(proposed_count)`
then enforces the 0↔1 invariant against that frozen fact: a
`MultiModel` boot refuses every mutation outright regardless of
`proposed_count`; a `SingleModel` boot allows one only while
`proposed_count` stays ≤ 1. `is_multi_model()` itself is unchanged —
still a live, reactive read of the current `ModelSet` — but it can no
longer silently drift from what the router actually is, because
`RouterTopology` is the fact anything that cares about *routing*
should read instead, and no mutation can ever make the two disagree
about which router shape is live. Every entry into `routes::runtime_lifecycle`'s
`load_model`/`unload_model` checks `validate_lifecycle_mutation` first,
before touching `lifecycle` or `model_set` at all.
