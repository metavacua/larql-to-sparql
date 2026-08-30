# Roadmap — larql-server / larql-router

For shipped work, see [CHANGELOG.md](CHANGELOG.md) — including the 2026-05-07
state-of-the-server snapshot and perf tables that used to live at the top of
this file, and the full 2026-05-10 code review. Design rationale is in
[THESIS.md](THESIS.md); protocol detail in [docs/server-spec.md](docs/server-spec.md).

## Current state (verified 2026-08-22)

**Tests.** 1132 passing across the lib and the `tests/` integration
suites (`cargo test -p larql-server`).

**Route surface.** 22 modules under `src/routes/`: the
OpenAI-compatible group (`openai/` — chat, completions, embeddings,
models, responses, schema, plus the V3 arms), the LQL group
(`describe`, `explain`, `insert`, `patches`, `relations`, `select`,
`walk`), the grid group (`expert/`, `shard`, `topology`, `walk_ffn/`),
and operational endpoints (`health`, `stats`, `sessions`, `stream`,
`warmup`, `embed`, `infer`).

**Coverage.** Enforced: 90% per file, 65% crate total, and 90% over the
*included* set (I/O-bound wrappers that need a live grid are excluded
and tracked separately). Currently 82.4% total / 93.1% included, with
12 debt baselines. Run `make larql-server-coverage-summary` to
re-measure.

**Performance.** The last full measurement is the 2026-05-07 M3 Max
2-shard grid snapshot in [`CHANGELOG.md`](CHANGELOG.md), which also
holds the remote MoE expert path table. Nothing in this crate has been
re-benched since; treat those numbers as a baseline, not a current
claim.

---

## Open defects

None currently tracked. The last one — P1, unbounded in-memory growth
with dead eviction logic (raised 2026-05-28) — was fixed 2026-08-22;
see [`CHANGELOG.md`](CHANGELOG.md).

---

## Great new functionality (next big-ticket items)

The numbered F0..F23 items below are mostly **incremental polish**
(metrics, shutdown drain, RBAC, OpenAPI, etc.) — necessary but not
load-bearing for new use cases. The items in this section are
**new capabilities** that would unlock production deployment shapes
the server can't currently serve. Ranked by how much they expand
the addressable surface, not by implementation effort.

### N0. OpenAI API compatibility — DONE

**Shipped.** `/v1/models`, `/v1/embeddings`, `/v1/completions`,
`/v1/chat/completions`, `/v1/responses` (+ `GET`/`DELETE
/v1/responses/{id}`), SSE streaming on all generating endpoints, tools
and structured output via constrained decoding (N0.6), on both VINDEX2
and VINDEX3 runtimes — and mirrored on `larql-router` so a grid is a
single OpenAI endpoint (N0-router).

- Wire contracts and semantics: [`docs/server-spec.md`](docs/server-spec.md) §4.6.
- What landed when: [`CHANGELOG.md`](CHANGELOG.md) (2026-05-02, 2026-08-22)
  and [`crates/larql-router/CHANGELOG.md`](../larql-router/CHANGELOG.md).

Supersedes F10 ("OpenAI-compat `/v1/chat/completions`"), which scoped
only the chat endpoint shallowly.

**Remaining polish** — none of it blocks a client, all of it is
observable divergence from OpenAI's wire behaviour:

- **`top_logprobs`** returns picked-token entries only. Real top-K
  alternatives need the sampler to surface the pre-sampling
  distribution — see F18. The V3 arms answer `logprobs: null`
  entirely.
- **Per-token tool-argument streaming.** Constrained decoding runs
  buffered under `stream: true` and emits one fat
  `delta.tool_calls[0]` chunk. Wire-compatible (clients accumulate
  and act on `finish_reason`), but not byte-for-byte OpenAI.
- **Rate-limit headers** (`x-ratelimit-limit-requests`,
  `x-ratelimit-remaining-requests`, …) — the `--rate-limit` machinery
  already has the numbers; nothing surfaces them.
- **`n > 1`** is a 400 on every generating endpoint (one completion
  per prompt).
- **Reasoning output items.** `/v1/responses` has no `type:
  "reasoning"` entry, so thinking-trace models emit their traces as
  ordinary output text.

---

### N1. Stateful chat sessions (KV-cache as a first-class resource)

**Why.** Every production LLM API is session-aware: the client sends
the new turn and the server remembers prior context via KV-cache.
`/v1/infer` is single-shot — every request re-prefills from scratch,
which is ~100 ms of wasted compute per turn at 4 K context and seconds
at 16 K.

**Shipped so far** (detail in [`CHANGELOG.md`](CHANGELOG.md) 2026-08-22,
contracts in [`docs/server-spec.md`](docs/server-spec.md) §4.5 and §4.6):
`previous_response_id` chains on a V3 runtime resume from a resident KV
state instead of re-prefilling; resumption is purely an optimisation
(exact ids-prefix or full-prefill fallback, gated bit-for-bit) and the
cache is model-keyed. `GET`/`DELETE /v1/sessions` observe and evict
session state, with continuations owned by the session that produced
them and deletion proof against a late re-insert from an in-flight
generation.

**Still open:**

- **The payoff is currently invisible.** Measured on a real chain,
  cache on vs off is 41.14 s vs 41.75 s — the V3 serve path's fixed
  per-request cost dwarfs what resumption saves. See **V3-SERVE** below;
  re-measure N1 after that lands, not before.
- **V2-runtime KV residency.** Only the V3 arm keeps state resident;
  a V2 chain still re-prefills.
- **Bound the session *count*.** The map is TTL-bounded but not
  count-bounded, and `X-Session-Id` is client-chosen — the same
  exposure `/v1/patches` has always had. Needs an eviction policy that
  does not silently discard a client's patches.
- **RSS-budget eviction.** Today the KV cache is count-bounded
  (`--v3-kv-cache-entries`); the useful bound is bytes, with LRU under
  memory pressure.
- **Token-level prefix survival on real tokenizers.** The template
  census is string-level; whether a family's real BPE tokenizer
  preserves the id prefix under conversation growth is per-family
  empirical.
- Pairs with **N3 (LoRA hot-load)** — sessions can pin an adapter.

---

### V3-SERVE. Make the server invoke VINDEX3 at the runtime's own speed

**The measurement.** Profiled below HTTP against a real
`granite-4.1-3b` container (`examples/v3_request_phase_profile.rs`); a
warm 5-token / 1-token request costs 7.23 s in-process against 7.28 s
over HTTP, so HTTP is ~50 ms and not the subject:

| phase | time | share |
|---|---|---|
| `prefill_into` | 3.83 s | 53.0% |
| `session_with_kv` | 3.27 s | 45.2% |
| decode | 0.13 s | 1.8% |

Opening the runtime at startup costs **1 ms** — the plan and operand
store are already hoisted. **94% of a warm request is operand
materialisation, not arithmetic**, and the same work with operands
already resident measures **0.84 s against 7.44 s (~8.8×)**.

**V3-SERVE-1. Hoist operand residency to server lifetime — DONE.**
Shipped 2026-08-22: `PreparedOperands` / `PreparedVindex3` lower a
container's operands once at bind time, and both traversals read that
image; preparation takes an `ExecutionSlice`, so residency is
slice-shaped from the start, and resolves through the operand-source
seam, so it composes with compose mutation rather than competing with
it. Confirmed on two real models — gpt-oss-20b 31.93 s → 0.75 s
(**42.7x**), Granite 4.1 3B 7.46 s → 0.576 s (**12.95x**), with
`session_with_kv` collapsing to 0.000 s on both. See
[`CHANGELOG.md`](CHANGELOG.md).

**V3-SERVE-2. Batched prefill that also populates KV — DONE, and it
exposed the next cost.** `PlanBackend::attention` now returns
`AttentionOut { outputs, keys, values }`: the batched realisation always
computed the conditioned rows and discarded them, which is what forced a
caller wanting a populated cache down the per-position path. The two
axes are now independent, and a fresh prefill takes the batched
realisation and populates the provider from the one traversal it
already performs.

Proven before the trait moved: the batched and per-position
realisations agree **bit-for-bit** on K, V and outputs — reference and
production backends, every layer, and all 40 layers of a real Granite
container (`exec/tests/attention_kv_parity.rs`, with a control that
fires on a perturbed input so the zeros are agreement rather than an
inert harness).

Measured on Granite 4.1 3B, battery, prefill_into:

```text
prompt      post-2B    post-2C    gain
     5      0.448 s    0.442 s    1.01x
    64      4.099 s    3.428 s    1.20x
   325     34.587 s   25.024 s    1.38x

prefill rate    post-2B   post-2C
  5 ->  64       16.16     19.76  tok/s
 64 -> 325        8.56     12.09  tok/s
```

No change at n=5 and a gain that grows with n is the signature of
fixing a per-position realisation — a thermal or power drift would have
moved all three points.

**The remaining gap is no longer attention.** Measured in the same
session and power state, `larql vindex3 exec`'s batched path runs
19.50 tok/s over 64 → 256 where the server's prefill runs 11.37 — still
**1.72x**. (An earlier 21.4 tok/s figure for the CLI was taken on AC;
re-measuring it on battery is what makes this comparison honest.) Both
now run the same batched attention, so the difference is what the
server does *around* it: populating the provider — `kv.append` per
position per layer, and whatever `CanonicalKvState` does to store a
row. That is the next thing to measure, and it was invisible while
per-position attention dominated.

Not done, and deliberately: a **resumed** prefill still steps. A
batched pass conditions position `p` as the `p`-th token of the
sequence it is handed, so it cannot express a prefill that starts
part-way through one. Giving `AttentionCall` an absolute-position base
would lift that, and would matter for long-history N1 resumes; it is
its own rung.

**V3-SERVE-3. Backend injection.** The server hardcodes
`ProductionBackend::new()`, so serving cannot use the Metal backend
that `vindex3 exec` already offers. Architecturally wrong regardless,
but *not* the performance lever here: at these sizes metal rung-1
measured 20.4 against production's 21.4 tok/s prefill. Do it after
V3-SERVE-1 and 2, and re-measure rather than assuming.

**V3-SERVE-4. Ship a servable container.** `load_v3_model` needs
`<container>/tokenizer.json`. Evidence that this is not yet a property
of the format: of the two containers on disk on 2026-08-23, the freshly
re-extracted Granite ships one and serves with no overlay, while
gpt-oss-20b does not and still needs a hand-placed tokenizer. main's
`compact` / `compile` paths *preserve* an existing one; `vindex3 encode`
does not write one by construction. So servability is a property of how
a particular artefact was built.

"One container happens to contain a tokenizer" is a different claim from
"every servable VINDEX3 artifact the canonical encoder produces is
self-contained". The gate is the second claim, end to end:

```text
encode a real model
  → the resulting container carries its own tokenizer authority
  → move the artifact away from the source checkpoint
  → larql-server opens it
  → real OpenAI inference succeeds
```

No source-checkpoint fallback, no symlink, no fixture-added tokenizer.
The existing serve tests all write a synthetic tokenizer into their
fixture, which is exactly why this class of defect is invisible to
them.

**V3-SERVE-5. Sharding, decoupling, grid.** Not started, and now
failing closed rather than silently: a V3 binding refuses `--layers` /
`--experts` / `--ffn-only` / `--embed-only` / `--no-infer` / `--units`.
`/v1/walk-ffn`, `/v1/expert/*`, `/v1/infer`, `/v1/embed` and
`/v1/shard` resolve V2 only, and a V3 server does not join the grid
(it warns; the router answers 503 `no OpenAI-capable server`). Order
this after the perf rungs — sharding a runtime that pays 6.7 s per
request just buys several copies of the problem.

**Distributed state, when it comes.** Three tiers, and the boundary
matters: process-local and hot (prepared operands, backend-native
weights, live KV, scratch) stays on the worker; distributed control
state (session → owning worker, response id → continuation owner, patch
generations, TTL/leases, routing and admission) is what a Redis-class
service should hold; durable/cold (containers, `.vlp` patches,
persisted conversations) stays on disk or object storage. Redis answers
"where does session `abc` live", never "give me layer 17's gate/up/down
tensors" — shipping weights or KV over a socket per request would be
this rung's defect wearing a distributed hat. Worth building when the
router does real multi-worker inference and `X-Session-Id` /
`previous_response_id` need affinity, so a continuation returns to the
worker that physically holds its KV and degrades to a cold prefill when
that worker is gone. Not before.

**N1 — three-point ledger.** Measured at each state rather than only at
the end, because each rung changes what the previous measurement meant:

| state | cache off | cache on | saving |
|---|---|---|---|
| pre-2B (gemma-2-2b) | 41.75 s | 41.14 s | 1.5% — inside noise |
| post-2B (gpt-oss-20b) | 45.02 s | 18.77 s | **2.40x** |
| post-2B (granite-4.1-3b) | 15.94 s | 7.69 s | **2.07x** |
| post-2C (granite-4.1-3b) | 20.75 s | 8.63 s | **2.40x** |

N1 was never broken; it was masked by ~7 s of per-request model
materialisation. The post-2C row was expected to *shrink* N1's absolute
saving — faster prefill means less to skip — and on the ratio it did
not (2.07x → 2.40x). Read it cautiously: the cache-off arm's first turn
was an outlier (6.63 s against ~2 s in every other run of it), which
inflates that arm's total, and the runs are minutes apart on battery.
What is solid is the shape, which is what the row was taken for: with
the cache, turn time stays near new-turn cost; without it, it grows
with history. N1 survives 2C as a structural win rather than a
workaround for slow prefill.

A longest-common-prefix resume is worth reconsidering after 2C, on the
post-2C numbers, not before.

---

### V3-DECOUPLE. Execution slicing as VINDEX3 semantics

**Why this is not optional.** V2's decoupled surfaces — `/v1/walk-ffn`,
`/v1/expert/*`, `/v1/embed`, `/v1/shard` — are one of larql's
distinguishing properties, and they are the whole basis of the expert
grid. A V3 that can only do monolithic `generate()`, however fast, has
*lost* that. These surfaces currently 404 on a V3 server (see
V3-SERVE-5), and the fix is not to reimplement each of them against V3.

**The shape.** Every granularity should be a projection of the same
`ComponentOpPlan`, not a separately engineered API:

```text
ComponentOpPlan
      ↓  select
ExecutionSlice
      ├── Full
      ├── LayerRange(10..20)
      ├── Attention(layer)
      ├── Ffn(layer)
      ├── Experts(layer, [3, 7, 19])
      ├── Embedding
      └── Head
      ↓  validate closure
prepared execution state       ← only the slice's operands
      ↓
execute
```

`ExecutionSlice` and slice-shaped preparation landed with V3-SERVE-1
(`Full` and `LayerRange` today, refusing what the plan cannot satisfy).
What remains is the rest of the vocabulary and the execution entry
points that consume them.

**Attention as a unit** must mean the plan identifies and runs the
attention subgraph — norm → Q/K/V → rope → attention → output
projection → residual — with explicit inputs, outputs and KV side
effects. Not "run the layer and discard the FFN result". That is what
buys attention-only nodes, separate attention/FFN residency, KV
research, and alternate attention implementations — and V3-SERVE-2's
batched KV-filling prefill is naturally a property of that unit rather
than something buried inside a full forward.

**FFN as a unit** likewise: norm → router → expert selection → gate/up
→ activation → down → reduction → residual, with a further level for
MoE (route only / select / execute selected experts / combine). That
level is where an expert server becomes a first-class VINDEX3 citizen
rather than a bespoke endpoint.

**Sharding is then a consequence, not an implementation.** A shard is an
executable submodel — select slice, validate closure, prepare that
slice's operands, serve it — so `--layers 0-9` stops being a filter in
a loop somewhere in `larql-server`:

```text
node A   layers 0–9, attention + FFN
node B   layers 10–19, attention only
node C   layer 10–19 FFN, experts 0–31
node D   layer 10–19 FFN, experts 32–63
```

**Then the server routes become thin mappings** onto capabilities the
plan already describes:

```text
/v1/infer      → full or specified slice      /v1/embed   → embedding slice
/v1/walk-ffn   → FFN slice                    /v1/logits  → head slice
/v1/expert/…   → expert slice                 /v1/shard   → advertised prepared slice
OpenAI APIs    → full-model slice
```

**Ordering.** After V3-SERVE-2 — the attention/FFN unit boundaries and
the batched-KV work are the same seam, and doing them in the wrong
order means cutting that seam twice. This is also the point at which
distributed coordination becomes worth having: a router/Redis control
plane knows *where* each execution capability lives, while VINDEX3
defines *what* each node can execute. Explicitly not before a single
worker is fast — see the note under V3-SERVE.

---

### N2. Asynchronous batch inference job queue

**Why**: Real-time chat is one model; **bulk inference** (RAG document
processing, embedding pre-compute, reranker scoring, evaluation
harnesses) is another. They have very different SLOs. A batch job
submitter doesn't care about per-token latency; it cares about
throughput, cost, and being able to run while the cluster is otherwise
idle. Today users have to wrap `/v1/infer` in their own retry/queue
glue.

**Proposal**:
- `POST /v1/jobs` → submit `{prompts: [...], model_id, params}` →
  returns `{job_id}`.
- `GET /v1/jobs/{id}` → status + partial results.
- `POST /v1/jobs/{id}/cancel`.
- Optional `webhook_url` in the submit body for completion callback.
- Worker pool: independent rayon thread pool, capped concurrency,
  prioritises real-time `/v1/infer` traffic (job worker yields when a
  real-time request arrives).
- Persistence: jobs survive restarts (write-ahead log to disk).

**Pairs with**: F12 (batched infer in same request), F22 (persistent
state). Together those two are the building blocks; this item is the
asynchronous wrapper.

**Implementation surface**: ~800 LOC. New `routes/jobs.rs`, new
`worker::Pool`, persistence to a `jobs/` directory. The hardest piece
is the priority scheduler — getting it wrong means batch starves
real-time or vice versa.

### N3. LoRA / adapter hot-loading per session

**Why**: Multi-tenant production. Today every tenant either gets the
same base model or has to spin up a separate process. Real production
serving (Anthropic, OpenAI, Together, Replicate) supports per-request
adapter swap. Adapters are 10-100 MB vs the 16 GB base model —
hot-loading hundreds of them is feasible if we have the surface.

**Proposal**:
- `POST /v1/adapters/load` → `{adapter_id, source: "hf://..."|"file://..."|"http://...",
  model_id}` → loads into RAM.
- `GET /v1/adapters` → list loaded adapters with size + last-used.
- `DELETE /v1/adapters/{id}` → evict.
- Inference / sessions take an optional `adapter_id` field — applies
  the LoRA delta to gate/up/down/q/k/v/o matmuls per layer per call.
- Eviction: LRU + total-RSS budget, configurable.

**Pairs with**: N1 (sessions pin adapters). Independent enough to ship
first if N1 is too heavy.

**Implementation surface**: ~500 LOC. The LoRA forward-pass plumbing
already exists at the inference-crate level (per
`larql-inference/ROADMAP.md` § F4 LoRA loading). The server piece is
the lifecycle + RSS management.

### N4. Multimodal API surface (vision tower, mixed image+text infer)

**Why**: Gemma 3/4 ships vision variants; Llama 3.2 too. The vindex
extractor already handles vision tower weights (per
`larql-inference/ROADMAP.md → vision`). We're missing the API
surface — there's no way to send an image to the server today.

**Proposal**:
- `POST /v1/embed/image` → multipart upload → vision tower forward →
  returns `{embedding: [...], hidden_size}`.
- `POST /v1/infer` accepts `images: [base64, ...]` field; server
  routes through the vision tower then concatenates with text tokens
  for the language decoder.
- `POST /v1/sessions/{id}/append` accepts images for multimodal chat.

**Implementation surface**: ~400 LOC server-side once the inference
crate's vision forward path is exposed (currently tracked separately).
Big use-case unlock: docVQA, ChartQA, image classification, image
embedding service.

### N5. Federated knowledge graph over multiple vindexes

**Why**: The DESCRIBE/WALK/SELECT trio makes a vindex a queryable
knowledge graph. Multi-model serving (`--dir`) puts multiple
graphs side-by-side — but each is queried independently. There's no
way to ask "describe France using Gemma's knowledge AND Llama's
knowledge AND my custom vindex". This is a unique capability the
larql architecture enables that nothing else (vLLM, TGI, OpenAI) can
do, and it's invisible.

**Proposal**:
- `GET /v1/federated/describe?entity=X&models=gemma,llama,custom` →
  merges edges across vindexes, sourcing each edge with its origin
  model.
- `POST /v1/federated/select` with cross-model joins ("entities
  Gemma calls capitals AND Llama calls capitals").
- New LQL syntax: `DESCRIBE "France" USING gemma, llama;` already
  hinted in the REPL doc (`USE REMOTE`); the server-side surface is
  the missing half.
- Surfacing model disagreement is a research-grade capability:
  "Gemma says Paris is the capital of France with score 1436;
  Llama says Lyon with score 320. Confidence-weighted merge?"

**Implementation surface**: ~600 LOC. New `routes/federated.rs`,
extends multi-model serving to do cross-model fan-out + merge.

### N6. Live blue-green vindex deployment

**Why**: Production model rollouts. Today swapping a vindex requires
restart (modulo F8 hot-swap, which is admin-only and atomic). True
blue-green wants: load v2 alongside v1, route X% of traffic, observe
metric drift, ramp or rollback.

**Proposal**:
- `POST /v1/admin/deploy` → load `v2.vindex` alongside the active
  `v1.vindex`, returns `{green_id}`.
- `POST /v1/admin/traffic` → set weighted routing
  (`{"v1": 0.9, "v2": 0.1}`).
- `GET /v1/stats.deployment` → per-vindex per-endpoint p50/p99/error
  rate side-by-side. Pairs with F3 metrics.
- `POST /v1/admin/promote/{id}` → atomically swap routing to 100%
  green; old vindex becomes stale-evictable.

**Pairs with**: F8 (admin endpoints), F3 (metrics for traffic
comparison). N6 is the **product** built on top of those primitives.

**Implementation surface**: ~700 LOC. New `routes/admin/deploy.rs`,
extends `AppState` to hold multiple model versions, weighted routing
logic in the request entry points.

---

## P0: Active

### F-LOCAL-MOE. Local Metal MoE optimisations (CPU staging + batched dispatch)

**Status**: Not started.

**Driver**: same 2026-05-02 bottleneck analysis. On the local Metal
MoE path, **67% of wall is CPU work**, only 33% is GPU active (51 ms
wall = 17 ms GPU + 33 ms CPU + sync). The GPU is barely loaded — the
CPU-side per-layer router + memcpy of 8 expert Q4_K byte slices into
staging buffers + commit/wait sync is dominating.

For the "run large models on consumer hardware" axis, every ms here
matters — the user runs LARQL on a single M3 Max, the grid isn't
available.

**Two levers, both CPU-path-safe**:

1. **Zero-copy expert byte aliasing**: today
   `gpu_moe_dispatch_with_scratch` memcpys ~300 KB per expert × 8 ×
   30 layers = ~72 MB of Q4_K bytes per token into pre-allocated
   staging buffers. The infra already exists —
   `MetalBackend::cached_buffer_for_bytes` does
   `new_buffer_with_bytes_no_copy` for the shard server's pre-staged
   path. Wiring it for the local path eliminates the per-layer
   memcpy entirely; experts alias the model's mmap directly.
   **Estimated win: 5–10 ms/tok.**

2. **Batched expert GPU dispatch**: today each MoE layer issues 24
   GPU dispatches (8 × `q4k_ffn_gate_up` + 8 × `geglu` + 8 ×
   `q4k_matvec` for down). Batching these into ~3 dispatches/layer
   using per-expert offsets into the already-staged buffers reduces
   dispatch overhead from ~720 calls/token to ~90.
   **Estimated win: 3–5 ms/tok.**

Combined: **8–15 ms/tok off the local path → 23–28 tok/s** on Gemma 4
26B-A4B Metal MoE (from 19.4 tok/s today).

**Acceptance**: `LARQL_GPU_TIMING=1` shows `cpu` shrunk by ~10 ms/tok;
`larql bench gemma4-26b-a4b-q4k-v2` shows ≥23 tok/s warm-state on
M3 Max with output unchanged.

### F-FLY. Remote multi-shard deployment on fly.io

**Status**: Not started — next session.

**Goal**: validate the HTTP CPU-path optimisations from the 2026-05-01 session
on a real network (LAN-class RTT ≥ 100 µs), not just M3 Max loopback. Most
of what we shipped is designed to win on real links but is invisible on
loopback (TCP_NODELAY, f16 wire). This is the apples-to-apples test that
tells us whether the in-room engineering translates to a deployable grid.

**Setup target (~2 hosts, then 4-8 if Phase 1 looks good)**:

- 1× client host (Mac dev box or fly.io VM): runs `larql run --moe-shards`
  with attention + dense FFN compute. Holds the 2 GB attention/router/dense
  weight set.
- N× shard hosts (fly.io VMs, ~16 GB RAM each): each runs
  `larql-server --experts START-END --grpc-port 9081 --uds-path ...`
  on a slice of the expert table. 26B-A4B has 128 experts × 30 layers;
  e.g., 4 shards × 32 experts × 30 layers ≈ 4 GB Q4_K + 2 GB working set
  per shard.
- Network: same fly.io region (intra-DC ~0.5 ms RTT) for Phase 1; a second
  region (cross-region ~30-100 ms RTT) for Phase 2 to stress the streaming
  overlap.

**What we expect to learn from this**:

1. Whether the **f16 wire** opt-in actually wins on real links (estimate:
   +3-5% on 1 Gbps, more on slower). On loopback it was within noise; we
   need real RTT to see the wire-bytes saving translate.
2. Whether **gRPC SPLIT default** (now on by default for gRPC) holds its
   ~12% steady-state win when the network leg is bigger than the dense
   FFN GPU leg (instead of comparable). The overlap math says the win
   grows when RTT > dense_FFN_time.
3. End-to-end tok/s ceiling on a real grid — we currently know loopback
   is ~19.7 tok/s; a multi-host grid should be slower per-token but
   throughput-scalable (more shards per host = more concurrent expert work).
4. Whether **predispatch (`batch` dispatch mode)** actually breaks
   generation on every multi-host setup or just on M3 Max loopback. We
   saw garbage output on loopback; might be a different story with real
   network timing.

**Prerequisites already in place** (from this session):

- gRPC streaming default-on for gRPC shards (~12% loopback gain,
  expected to grow on RTT-heavier links)
- TCP_NODELAY on accepted connections (defensive against tail-packet
  stalls on real LAN)
- f16 wire as opt-in (`LARQL_MOE_WIRE_F16=1`)
- Unix domain sockets (`--uds-path`, `unix:///path` URL) for same-host
  shard collocation
- `LARQL_HTTP_TIMING=1` per-call instrumentation (encode / send_total /
  recv_body / decode breakdown)
- `LARQL_MOE_TIMING=1` per-token MoE summary (route / collect / server
  compute / network estimate)
- 9.6× CPU MoE speedup on the shard side (bench: 30-layer sweep
  221 → 22.9 ms; production: 2.3 → ~19.7 tok/s end-to-end on M3 Max
  loopback)

**fly.io specifics worth pinning down before deploy**:

- VM size for shards: 26B-A4B vindex is ~16 GB on disk; needs ~10 GB
  RSS at warmup. `performance-cpu-2x` (~7 GB RAM) won't fit a full
  shard; need `performance-cpu-4x` (~14 GB) at minimum, or shard the
  vindex finer.
- Vindex distribution: cheapest is to ship the full 16 GB to each shard
  and let `--experts START-END` cap working set; alternative is per-shard
  vindex slicing (`larql slice` exists but needs a per-shard variant).
- Persistent volume vs in-memory: with `--warmup-walk-ffn` the boot
  cost is ~6-7 s; if VMs reboot per deploy, that adds up. Consider
  fly.io persistent volumes for the vindex.
- Health check: `/v1/health` is already there.
- Authentication: the existing `--api-key` flag works but a multi-tenant
  fly.io setup probably wants per-shard token rotation (out of scope for
  Phase 1).

### F1. Router-side expert-shard fan-out
**Files**: `crates/larql-router/src/main.rs`, `crates/larql-router/src/grid/`,
`crates/larql-router-protocol/proto/*.proto`.
The grid router fans out `walk-ffn` by layer ranges only. For MoE, the
remote-expert client (`RemoteMoeBackend` in `larql-inference`) carries the
expert→shard map itself; nothing on the router side. Means clients can't just
point at the router for MoE. Add `POST /v1/expert/{layer}/{id}` and
`POST /v1/expert/batch` to the router, with shard discovery via the existing
gRPC announce stream. Pairs with **F11** (topology endpoint).

### F2. Streaming HTTP infer (SSE)
**Files**: `crates/larql-server/src/routes/infer.rs` (new sibling
`infer_stream.rs`).
`/v1/infer` is single-shot — full output buffered, no incremental tokens. WS
has it (`WS_CMD_INFER`) but most chat UIs talk SSE. Add
`POST /v1/infer/stream` with `text/event-stream`. Same generation loop, yield
each token. Mid-generation cancellation on client disconnect (see **F16**).

### F3. `/metrics` (Prometheus) — server side only
`larql-router` ships `/metrics` (ADR-0017); the **server** does not.
`src/metrics.rs` exists but only holds the per-layer latency tracker
that feeds `/v1/stats`.

**Files**: `crates/larql-server/src/bootstrap/`, `crates/larql-server/src/metrics.rs`.
No latency histograms, no per-endpoint counters, no rate-limit drops, no
shard-call durations today. Wire `metrics` + `metrics-exporter-prometheus` (or
hand-rolled). Histograms for: `walk-ffn` per `layer_count`, `forward_moe` per
`top_k`, queue wait, auth failures, rate-limit drops, shard-call latency.

### F4. Graceful shutdown with in-flight drain
**Files**: `crates/larql-server/src/main.rs`.
SIGTERM today probably cuts long-running walks. Standard axum + tokio shutdown
signal: stop accepting, drain N seconds (configurable), hard-kill. Important
for grid rolling restarts.

### F5. Readiness vs liveness split
**Files**: `crates/larql-server/src/routes/health.rs`, `routes/mod.rs`.
`/v1/health` returns `{status, uptime, requests_served}`. Add `GET /v1/ready`
returning 503 until weights are loaded (under `--warmup-walk-ffn` or first
lazy load); include `model_id`, `mode`, `version`, `git_sha`, `format`
(per-layer vs legacy) in the readiness payload. Standard k8s liveness/readiness
split.

---

## P1: Active

### Q1.10 Reduce `routes/stream.rs::handle_stream_infer` (327 LOC) — deferred

The remaining open code-quality item from the 2026-05-01 audit. The other
nine (Q1.1–Q1.9) shipped — see "Completed → 2026-05-01 (continued) — Q1
code-quality cleanup". Q1.10 is deferred until N0.1 (OpenAI Chat
Completions SSE) forces a similar streaming state-machine shape; the
two should share infrastructure. Effort estimate: ~3 hours when picked up.

---

### F6. Replica round-robin + retry on shard failure
**Files**: `crates/larql-router/src/grid/`.
Router picks first owning shard; no load-balancing across replicas, no retry
on 5xx. `--shards "0-15=A,0-15=B"` doesn't fan evenly today.

### F7. KV-cache prefix sharing for chat — partly done
Delivered for chained `/v1/responses` on V3 runtimes under **N1**
(exact ids-prefix resumption, keyed by response id and owned by the
session). Still open here: the same residency for **V2** runtimes, and
for the native `/v1/infer` path, where every call is still a fresh
prefill.

**Files**: `crates/larql-inference/src/layer_graph/generate/*`,
`crates/larql-server/src/routes/infer.rs`.

### F8. Vindex hot-swap admin endpoints — done (single-model topology)
Shipped as `POST`/`DELETE /v1/runtime/model` rather than the
`/v1/admin/vindex/...` shape originally sketched here — no separate
`/reload` verb; a client swaps by calling `DELETE` then `POST`. Scoped
to single-model-topology servers only (`RouterTopology`, 0↔1
invariant) — a boot-time multi-model server still requires a process
restart to change its bound set, and dynamic multi-model loading
remains unimplemented. Gated by the existing single API key like every
other route, not a separate admin key — real per-key admin scoping is
still **F14**, unimplemented.
**Files**: `crates/larql-server/src/routes/runtime_lifecycle.rs`,
`crates/larql-server/src/state/lifecycle.rs` (state machine),
`crates/larql-server/src/state/model_set.rs` (mutable model registry).
See `docs/runtime-lifecycle-design.md` for the full design record.

### F9. Binary wire format for `expert/batch`
**Files**: `crates/larql-server/src/routes/expert/`,
`crates/larql-inference/src/ffn/moe_remote/`.
A K=8 batch on Gemma 4 26B-A4B is ~90 KB JSON per call. The
`application/x-larql-ffn` binary format already exists for `walk-ffn`; mirror
it for `expert/batch`. Expected 3–5× wire reduction.

### F10. OpenAI-compat `/v1/chat/completions` — superseded by N0, DONE
Kept for cross-references only. This item scoped the chat endpoint
shallowly; the delivered surface is **N0** above.

### F11. Expert topology endpoint
**Files**: new `crates/larql-server/src/routes/topology.rs`.
`GET /v1/expert/topology` returns `{model_id, layers, num_experts, owned: [start,end]}`.
Lets clients build the shard map dynamically instead of having it baked in.
Pairs with **F1** (router fan-out).

### F12. Batched infer
**Files**: `crates/larql-server/src/routes/infer.rs`.
`/v1/infer` takes one prompt today. RAG workloads send N prompts; one batched
call across them amortises router/dispatch overhead. Either accept
`prompts: [...]` or new `/v1/infer/batch`.

### G3a. Cross-host RTT measurement *(forward-looking)*
**Status**: open. Requires two physical machines on the same LAN.
The same-host validation establishes correctness; cross-host
measures the additional TCP overhead per fan-out.

## P2: Forward-looking

### G-SCALE. Run T-class models on grid (Kimi K2.6, DeepSeek V4 scale)

**Driver**: LARQL's strategic axis is "run large models on consumer
hardware OR split across grids." T-class MoE models (Kimi K2 ≈ 1T total
params, top-K ≈ 8; DeepSeek V3 ≈ 671B, top-K=2; future K2.6 / V4 likely
similar shape) can't fit on any single consumer machine — the grid
deployment shape is **the only way** to run them locally.

**What changes vs Gemma 4 26B A4B (today's reference)**:

| Dimension | Gemma 4 26B-A4B | Kimi K2 (~1T) | DeepSeek V3 (~671B) |
|---|---|---|---|
| Total params | 26B | ~1T | 671B |
| Layers | 30 | ~60 | 61 |
| Experts/layer | 128 | ~384 | 256 |
| Top-K active | 8 | 8 | 8 |
| Active params/token | ~5B | ~37B | ~37B |
| Q4_K vindex size (estimate) | 16 GB | ~600 GB | ~400 GB |

**Implications for the grid primitives**:

1. **Memory-conscious shard layout**. A T-class model's expert table is
   100× our current. With 16 GB consumer-class RAM per shard, K2 needs
   ~40 shards just to fit. Per-shard memory targeting matters: each
   shard owns a tight `(layer, expert_id)` set of mmap pages and never
   loads the rest. The `--units PATH` JSON manifest already supports
   per-(layer, expert) ownership; **G5 below** (per-shard expert routing
   in router-protocol) lights it up at the router layer.
2. **Parallel shard collect is non-negotiable**. With 40+ shards,
   sequential collect would compound to seconds/token. **F-COLLECT**
   above is the prerequisite.
3. **Streaming expert byte transfer**. T-class expert weights per layer
   may not fit in RAM even on a fat shard if it owns many experts. The
   shard's mmap+page-fault behaviour does the right thing today (only
   active expert pages are paged in), but **G4 mmap residency control**
   below becomes operationally important — long-running shards need
   `madvise(DONTNEED)` after a layer to reclaim RSS.
4. **Router-side fan-out batching**. With 40+ shards and 30+ layers,
   per-layer round-trips dominate. Multi-layer `forward_moe_predispatch`
   (already exists) becomes the default rather than an opt-in; the
   pass-1 approximation cost is negligible compared to 40-shard ×
   30-layer sequential RTT.

**Status**: Forward-looking. **F-COLLECT** + **G5** + **G4** are the
direct prerequisites; once those land we should attempt a multi-shard
deployment of one T-class model end-to-end as a capability check, even
if perf is exploratory rather than production-tuned.

### G4. mmap residency control endpoint
**Impact**: For long-running shards under memory pressure, expose
`POST /v1/mmap/advise {layers, advice: "willneed"|"dontneed"}` so
operators can trim RSS or pre-warm specific layer ranges without
restarting.

### G5. Per-shard expert routing
**Impact**: For DeepSeek-V3+/Kimi K-class models (1k+ experts), shard
by expert ID within a layer rather than by layer range. Needs an
`ExpertRoute` message type in `larql-router-protocol` and
GridState dispatch updates. Mentioned in larql-vindex P2. Subsumed by
**F1** (router-side expert fan-out) at the router layer; G5 covers the
router-protocol changes specifically.

### G6. Live router-shard topology change
**Impact**: Today shards are static (`--shards` flag at router boot).
For ops convenience, expose `POST /v1/router/shards` (admin-gated)
to add/remove a shard without restarting the router. Pair with
`--grid-port` health checks.

### F13. OpenTelemetry tracing exporter
**Files**: `crates/larql-server/src/main.rs`.
Per-request spans across HTTP→shard fan-out. `tracing_subscriber::fmt` is the
only output today. Wire `tracing-opentelemetry` + OTLP exporter, configurable
via `--otel-endpoint`. Pairs with **F3** (metrics).

### F14. Per-key quotas + audit log
**Files**: `crates/larql-server/src/auth.rs`, `crates/larql-server/src/main.rs`.
Single API key today; no per-key quotas, no rotation, no scoped tokens. Add
`--api-keys keys.toml` (name + role + per-key rate). Structured audit on
patches + admin ops to a configurable sink (file / stdout / OTel).

### F15. RBAC (read-only vs admin keys)
**Files**: `crates/larql-server/src/auth.rs`, all mutating routes.
Today any key can patch the loaded model. Add `role` per key
(read / infer / patch / admin). Mutating endpoints (`patches/apply`,
`insert`, future `admin/*`) require the matching role.

### F16. Mid-generation cancellation on HTTP infer
**Files**: `crates/larql-server/src/routes/infer.rs`.
Client disconnect on `/v1/infer` waits for the full max_tokens. Wire
`tokio::select!` against an axum `OnUpgrade`-style cancellation token (or just
poll the connection on each decode step) to abort early.

### F17. Structured-output / grammar-constrained generation — partly done
Shipped on the OpenAI surface as N0.6: `response_format` and `tools`
mask the LM head per token through a schema-typed JSON FSM, on both V2
and V3 runtimes (`docs/server-spec.md` §4.6).

Still open: the same hook on the **native** `/v1/infer` endpoint
(`{format: "json", schema: ...}` / `{grammar: "gbnf:..."}`), and a real
GBNF parser — the FSM today compiles JSON Schema directly, so a
caller-supplied grammar has no front end.

### F18. Log-prob / perplexity endpoint
**Files**: new `crates/larql-server/src/routes/logprobs.rs`.
`POST /v1/logprobs {prompt, top_k}` — return per-token log-probabilities.
Needed for ranking, classification, and eval workflows.

### F19. OpenAPI schema route — DONE
Shipped: `src/openapi.rs` (utoipa) serves `GET /v1/openapi.json` plus a
`/swagger-ui`. Every route added since is registered there, and
`tests/test_openapi_coverage.rs` gates it.

### F20. Compression negotiation
**Files**: `crates/larql-server/src/main.rs`.
No `Content-Encoding: gzip|zstd` advertised; relies on a reverse proxy. Wire
`tower-http::compression`. Particularly useful for `walk-ffn` JSON responses
on slow links.

### F21. `/v1/stats` per-layer mmap residency
**Files**: `crates/larql-server/src/routes/stats.rs`.
Existing `q4k_ffn` block exposes cache slots/bytes; extend with per-layer
hot/cold (resident vs paged-out) so operators can see what `--release-mmap-after-request`
actually buys them.

### F22. Persistent patches
**Files**: `crates/larql-server/src/session/`,
`crates/larql-server/src/routes/patches.rs`.
Patches are session-scoped today; no on-disk overlay. Add a durable
`POST /v1/patches/save` + auto-apply on boot. Pairs with **F8** (hot-swap)
so a patched model survives restart.

### F23. Python HTTP client SDK
**Files**: new `crates/larql-python/src/http_client.rs` (or new crate).
`larql-python` is walk-only against a local vindex; no HTTP client. Add a
`pip install larql` package speaking the server's HTTP API (sync + async),
mirroring the OpenAI Python SDK shape. Pairs with **F10** (OpenAI compat) so
the SDK is a thin wrapper over the OpenAI client.

---
