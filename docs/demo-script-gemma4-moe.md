# LARQL Demo: Gemma 4 in 8 GB — Then the Experts Weren't Even on My Laptop

> Runtime: ~11 min. Two acts. Act 1 runs Gemma 4 31B (dense) on a laptop in
> ~8 GB. Act 2 runs Gemma 4 26B A4B (MoE) with the experts distributed
> across other machines.
>
> **Numbers in this draft are scaled approximations.** Measure on the real
> releases before filming and replace every `~` figure. Do not let
> aspirational numbers into the final cut.

---

## Cold Open (30 s)

Gemma 4 31B at f16 is about 62 GB of weights. That's not a laptop model.
That's a workstation with a 48 GB card, or two. The rule everyone accepts:
big model, big hardware.

Here's the thing nobody talks about. In a 31B dense transformer, **~90% of
those weights are FFN** — `up`, `gate`, `down`. Attention, norms, and
embeddings together are about 7 GB. And on any given token, the FFN
activates maybe ten of twenty thousand features per layer. The other
99.95% of FFN neurons contribute nothing to this token. They don't have to
be in RAM.

That's Act 1. 31 billion parameters on a laptop in ~8 GB.

Then it gets interesting. Google already shipped a model that admits this.
Gemma 4 26B A4B is a Mixture of Experts — 26 billion total, **3.8 billion
active** per token. Google's own framing: 85% of the model is dormant every
forward pass. That's the product pitch. So the question writes itself —
does the sleeping 85% have to be on my laptop?

Act 2: the experts live somewhere else.

---

## Act 1 — Gemma 4 31B in 8 GB (dense, local walk-only)

### 1.1 The setup

```bash
# One-time: decompile the model into a vindex.
# Q4_K on the forward-pass weights (attention + interleaved FFN);
# f16 on the side-channel tensors the walk reads directly.
larql extract google/gemma-4-31b-it \
  -o gemma4-31b.vindex \
  --quant q4k
```

**On-disk layout** (narrate while `du -sh` runs):

```
gemma4-31b.vindex/
├── attn_weights_q4k.bin    ~4.8 GB   ← stays in RAM (dequantised per layer)
├── embeddings.bin          ~2.6 GB   ← stays in RAM (f16)
├── norms.bin                ~5 MB    ← stays in RAM
├── gate_vectors.bin         ~13 GB   ← mmap (demand-paged, f16 walk keys)
├── interleaved_q4k.bin      ~11 GB   ← mmap (demand-paged, per-layer FFN)
├── down_meta.bin            ~108 MB  ← token labels per feature
└── index.json
```

Total on disk: ~32 GB (Q4_K). Stock f16 weights on HuggingFace would be
~62 GB — the decompile is already half the win. Total **resident** during
a forward pass: about 8 GB. The rest is cold pages in a memory-mapped
file.

### 1.2 Load it

```bash
$ larql run gemma4-31b.vindex
larql chat — gemma4-31b.vindex (Ctrl-D to exit)
  Loaded: gemma-4-31b-it (60 layers dense, Q4_K)
  Mode:   walk (FFN weights mmap'd, not resident)
  RSS:    ~8 GB
>
```

Pause on that number. **Eight gigabytes.** 31 billion parameter model.
Chrome-with-a-few-tabs RSS.

### 1.3 Actually use it

```
> The capital of France is
  Paris       0.994
  **          0.005
  ’           0.000
  ...         0.000
  known       0.000

  latency: ~1.2 s  (Metal target; measured at ~20 s CPU today)
  FFN:     walk (gate KNN k=10, interleaved_q4k per-layer)
  RSS:     ~8.1 GB   (attn Q4_K resident + gate/ffn mmap pages faulted in)
```

Real forward pass. Attention on CPU, FFN served from mmap. Gate KNN picks
~10 features per layer out of tens of thousands; those down-vectors read
straight from the mapped file into BLAS — zero copy.

```
-- Paraphrases work — not a lookup table:
> France's capital is                     → Paris  99.85%
> The French capital is called            → Paris  95.78%
> La capitale de la France, c'est         → Paris  92.12%
```

### 1.4 Why this works (30 s — animation)

> **On-screen visual:** a 20,480-cell grid representing one FFN layer. A
> residual vector comes in at the left. The gate fires — ~10 of 20,480
> cells light up. The other 20,470 stay dark. Cut to the next layer: a
> different ten light up. Loop across a handful of layers so the viewer
> sees the sparsity pattern shift per token.

A transformer layer is `attn(x) + ffn(x)`.

**Attention is dense.** Every token attends to every other token through
every head. Q, K, V, O all participate every time. Attention weights have
to sit in RAM, hot.

**FFN is sparse at runtime.** The gate picks a tiny fraction of features
per token. Ten out of twenty thousand. Less than 0.05%. The other 99.95%
contribute nothing and can stay on disk as cold mmap pages.

The trick is storing FFN in a layout where "give me feature 4,732's gate
+ up + down block" is a single contiguous read, not three strided gathers
across three separate matrices. That's what `interleaved_q4k.bin` is —
per-layer, per-feature blocks laid out so the walk touches one page per
active feature. The kernel faults in exactly those pages, nothing else.

### 1.5 Why hasn't anyone done this? (20 s aside)

Two reasons.

**One.** Safetensors and every standard checkpoint format stores FFN
hidden-major — great for dense matmul, terrible for "give me feature X."
You have to re-lay-out on disk before any of this works. `larql extract`
is literally that step. The decompile is the unlock.

**Two.** Treating weights as a queryable structure rather than an opaque
blob isn't how the field thinks. Tensor in, tensor out, don't look inside.
LARQL's pitch is that the structure was there all along — it's just been
hidden behind `torch.load`.

Now here's the thing. In Act 1 I had to do that decompile myself to expose
the structure. What if Google already did it for us?

---

## Act 1.5 (pre-roll / proof shot) — FFN already separable over HTTP

Before Act 2 claims "the experts live somewhere else," a 30-second beat
that proves the claim on the dense model we just ran.

Same vindex, two machines:

```bash
# On the server — the FFN service (disables /v1/infer, advertises
# mode: ffn-service in /v1/stats, skips gate warmup).
$ larql serve gemma4-31b.vindex --port 8080 --ffn-only
  Loaded: gemma-4-31b-it (60 layers dense, Q4_K)
  Mode:   ffn-service (--ffn-only)
  Warmup: skipped (lazy gate decode on first request)
  Endpoints:
    POST /v1/walk-ffn     ← the one we care about
    GET  /v1/stats
  Listening on 0.0.0.0:8080
  RSS at startup: ~5.6 GB
```

```bash
# On the laptop — attention local, FFN over HTTP.
$ larql run gemma4-31b.vindex --ffn http://server.local:8080 \
    "The capital of France is"
  Paris       0.994
  **          0.005
  ...
  Forward pass: ~21 s  (CPU; FFN → http://server.local:8080)
```

Same answer as Act 1's all-local run, same top-1, full forward pass with
every layer's FFN hop going across the network. Confirm the path label
says `walk (q4k + ffn remote)` — if it says `walk (q4k)` the client
silently fell back to local FFN. If this
works on the dense 31B, it's going to work on the MoE — and the MoE
already admits which experts it needs, so the wire protocol gets tighter.

---

## Act 2 — Gemma 4 26B A4B: The Experts Live Elsewhere

Gemma 4 26B A4B is a Mixture of Experts. 26 billion total parameters. **3.8
billion active per token.** A router in each layer scores experts and
picks the top-K; only those experts run.

That's not my interpretation. That's Google's product page. The
architecture is *pre-factored* into the decomposition Act 1 extracted by
hand — each expert is an independent FFN block, named, enumerable,
hot-swappable by construction.

Everything Act 1 motivated — "the FFN is separable, 90% of it isn't doing
anything on any given token" — Gemma 4 26B A4B ships as the design.

### 2.1 The topology — experts distributed by ID

```
┌─────────────────────────────┐
│  Laptop (client)            │
│                             │
│  attn_weights.bin   3.5 GB  │
│  embeddings.bin     1.8 GB  │
│  norms.bin            6 MB  │
│  router_weights.bin  20 MB  │   ← the router stays on the client
│                             │
│  Resident: ~5.5 GB          │
│  Runs: attention + router   │
└──────────────┬──────────────┘
               │ HTTPS (K parallel calls/layer, K = top-K)
       ┌───────┼───────┬─────────────┐
       ▼       ▼       ▼             ▼
   ┌───────┐ ┌───────┐ ┌───────┐   ┌───────┐
   │ Expert│ │ Expert│ │ Expert│ … │ Expert│
   │ server│ │ server│ │ server│   │ server│
   │  #1   │ │  #2   │ │  #3   │   │  #N   │
   └───────┘ └───────┘ └───────┘   └───────┘
```

Router lives on the client — it's tiny (`~20 MB` for the whole model).
Client scores experts locally, sends "I need expert 42 at layer 18" to
whichever server holds that expert, gets back a residual delta. Across
all active experts per layer, that's K parallel calls — not 62 serial
round trips.

### 2.2 Carve the attention+router slice for the client

```bash
# On the server side — slice the client-side pieces from the full vindex.
larql slice gemma4-26b-a4b.vindex \
  -o gemma4-26b-a4b.client.vindex \
  --parts attn,embed,norms,router,index,tokenizer

scp -r server:gemma4-26b-a4b.client.vindex ./
```

```
  attn_weights.bin        3.5 GB  ✓
  embeddings.bin          1.8 GB  ✓
  norms.bin                 6 MB  ✓
  router_weights.bin       20 MB  ✓   ← new on the client list
  index.json                8 KB  ✓
  tokenizer.json           ~2 MB  ✓
  (skipped gate_vectors.bin + per-expert weights)
  Total: ~5.4 GB
```

A 26-billion-parameter MoE, and the client-side footprint is **5.4 GB**.
That's smaller than the Gemma 3 4B dense model.

### 2.3 Stand up the expert server(s)

Simplest case — one server holds every expert:

```bash
larql serve gemma4-26b-a4b.vindex \
  --port 8080 --host 0.0.0.0 --cache-ttl 300
```

```
  Loaded: gemma-4-26b-a4b (MoE, N experts × L layers, active=3.8B/26B)
  Endpoints:
    POST /v1/expert/{layer}/{expert_id}   ← new
    POST /v1/expert/batch                 ← new (K calls in one request)
    GET  /v1/stats
  Listening on 0.0.0.0:8080
```

To demo the fan-out, we can also shard experts across multiple processes:

```bash
# Terminal A: experts 0-31
larql serve gemma4-26b-a4b.vindex --port 8081 --experts 0-31

# Terminal B: experts 32-63
larql serve gemma4-26b-a4b.vindex --port 8082 --experts 32-63

# … and so on
```

### 2.4 Wire the client

```sql
larql> USE "gemma4-26b-a4b.client.vindex" WALK ONLY
  ...> WITH EXPERTS REMOTE {
  ...>   "0-31":  "https://a.example.com:8081",
  ...>   "32-63": "https://b.example.com:8082"
  ...> };

Connected: gemma-4-26b-a4b  (MoE, 64 experts × 48 layers, top-K=8)
  Attention:  local   (5.4 GB resident)
  Router:     local   (20 MB)
  Experts:    remote  (sharded across 2 endpoints)
  Health:     OK      (RTT 38 ms / 41 ms)
  RSS:        ~5.5 GB
```

### 2.5 Generate

```sql
larql> INFER "The capital of France is" TOP 5;
  Paris       0.889
  the         0.041
  a           0.019
  located     0.010
  one         0.007

  latency:       ~2.4 s
     attention + router:  ~0.5 s   (48 layers, local)
     expert compute:      ~0.4 s   (avg 8 experts/layer, server-side)
     network RTT:         ~1.5 s   (48 layers × ~8 parallel calls, RTT-bound)
  experts invoked:  ~384 (48 layers × top-8)
  bytes sent:       ~480 KB
  bytes recv:       ~480 KB
  RSS (client):     ~5.5 GB
```

Stop on the byte count. **Under a megabyte round-trip for a 26B MoE
forward pass.** Less data than a phone photo.

Stop on the latency breakdown. Same pattern as Act 1's dense-remote
variant — most of the wall clock is RTT, not compute. On a LAN this
collapses to sub-second. Public internet is the floor, not the ceiling.

Same answer as the local run. Paris at 0.889 vs the dense 31B's 0.891 —
different models, four-thousandths apart, same right answer.

### 2.6 Prove the experts really are remote

```sql
-- Kill the server holding experts 32-63.
larql> INFER "The capital of France is" TOP 5;
Error: expert 47 at layer 12 unreachable:
  connection refused (https://b.example.com:8082)

-- Route those experts to the other server on the fly.
larql> RESHARD EXPERTS { "0-63": "https://a.example.com:8081" };
  Reshard OK. All 64 experts now on a.example.com.

larql> INFER "The capital of France is" TOP 5;
  Paris       0.889
  ...
```

No experts, no knowledge. Attention + router alone produces nothing
coherent. The knowledge is the experts. The experts are on those other
machines. We're just borrowing them — and we can redistribute them live.

---

## Closing (60 s) — the teaser earns it

Three things fall out of this.

**One.** The hardware wall on frontier models is mostly a packaging
problem. Dense or MoE, most weights are sparse-at-runtime. If you stop
insisting they live in RAM, a 31B dense model fits in 7 GB and a 26B MoE
fits in 5.

**Two.** The FFN — or the expert bank — is a service. Under a megabyte on
the wire for a full forward pass. One expert server can fan out to many
clients. Or you can shard experts across many servers. Or — the
interesting one — you can move experts between machines *while the model
is running*.

**Three.** Experts have matching dims by construction within an
architecture. Expert 42 at layer 18 from the base model has the same input
and output shape as expert 42 at layer 18 from a model fine-tuned on
medical papers. Swapping one for the other is a file copy on the server.

> Tease for Video 3: replace a handful of experts in Gemma 4 26B A4B with
> medically-tuned variants. Watch the same prompts route the same way
> topologically, but produce different answers where the swapped experts
> fire. That's surgical fine-tuning — not retraining, not LoRA, not
> merging. Just rerouting.

---

## Shot list / B-roll cues

| Time | Shot | Voice |
|------|------|-------|
| 0:00 | Title card: "Gemma 4 in 7 GB — Then the Experts Weren't Even Here" | Cold open |
| 0:30 | Google Gemma 4 26B A4B product-page screenshot, highlight "3.8B active" | Hook |
| 1:00 | `du -sh` + `htop` split screen | Act 1 footprint |
| 1:45 | Gate-activation animation (10 of 20,480 cells lit per layer) | 1.4 |
| 2:30 | Side-note title card: "why nobody's done this" | 1.5 aside |
| 3:00 | Transition: "Google already did it" → MoE diagram | Act 2 bridge |
| 3:30 | Topology animation — router on laptop, fan-out to expert servers | 2.1 |
| 5:00 | Three terminals: client + two expert shards | 2.3–2.4 |
| 6:30 | `tcpdump` with parallel request counter | "K calls, not K sequential" |
| 7:00 | Latency bar chart: attn+router / experts / network | Breakdown |
| 7:30 | Kill the second shard, show the error, reshard live | Proof shot |
| 8:30 | Title card: "the experts are a service" | Closing |
| 9:30 | Teaser: expert-42 file swap, medical prompt shift | Video 3 hook |

---

## Open issues before filming

### Phase 0 — dense-remote baseline — **SHIPPED**

- [x] Extended `POST /v1/walk-ffn` with `full_output: true` +
  `seq_len: N` for batched residuals. Server runs the architecture-correct
  `WalkFfn` forward and returns `[seq_len × hidden]` row-major.
  gRPC mirror landed too — proto gained `seq_len` + `output` fields.
  (`crates/larql-server/src/routes/walk_ffn.rs`, `grpc.rs`, `proto/vindex.proto`)
- [x] `RemoteWalkBackend` in `larql-inference` implements `FfnBackend`,
  POSTs residuals, reshapes the reply. Plugs into `predict_with_ffn`
  unchanged. (`crates/larql-inference/src/ffn/remote.rs`)
- [x] `larql run <model> --ffn URL` wired end-to-end in the CLI. The
  walk command also gained a `--ffn-remote URL` flag for the power-user
  path (`larql dev walk --ffn-remote …`).
- [x] `larql serve --ffn-only` — server declares itself an FFN-service
  endpoint, disables `/v1/infer`, advertises `mode: ffn-service` in
  `/v1/stats`.
- [x] Localhost-to-localhost parity probe at
  `crates/larql-inference/examples/remote_walk_parity.rs`. Target
  tolerance: `max_abs ≤ 1e-5` (f32-through-JSON precision floor).
  Run manually against a real vindex; not in CI.
- [x] **Memory-footprint follow-up on `--ffn-only`**: `--ffn-only` now
  skips the f16→f32 gate warmup at startup (largest eager cost). Measured
  on Gemma 4 31B Q4_K: server RSS 55 GB → 5.6 GB at startup. Grows to
  ~23 GB after a forward pass as layers lazy-decode. Further wins (f16
  gemv without decode, LRU of decoded layers) are follow-ups.
- [x] **Q4_K walk-ffn server path**: `state.rs::get_or_load_weights`
  routes Q4_K vindexes through `load_model_weights_q4k`; the walk-ffn
  `full_output` handler calls the new `q4k_ffn_forward_layer` helper to
  dequantise gate/up/down per layer on demand. Working-set stays ~3 GB
  per request instead of the ~120 GB that eager dequant would cost.
- [x] **Q4_K client `--ffn` path**: `run_predict_q4k_remote` in
  `walk_cmd.rs` + `predict_q4k_with_ffn` in `q4k_forward.rs`. Client
  dequantises attention per layer, delegates FFN to `RemoteWalkBackend`.
  Path label `walk (q4k + ffn remote)` confirms it's not falling back.

### Phase 1 — MoE inference path (blocks Act 2)

- [ ] **MoE-aware forward pass.** `larql-inference` has zero mentions of
  `expert`/`MoE` today — the forward pass runs dense FFN only. Add a MoE
  layer path that (a) calls the router, (b) picks top-K experts, (c)
  dispatches to a per-expert FFN backend, (d) sums weighted outputs.
- [ ] **Gemma 4 MoE architecture hooks.** `crates/larql-models/src/architectures/gemma4.rs`
  is dense-only. Copy the Mixtral pattern (`is_moe`, `num_experts`,
  `num_experts_per_token`, `moe_router_key`, `expert_ffn_{gate,up,down}_key`)
  to support the 26B A4B variant.
- [ ] `RouterIndex` (already exists at
  `crates/larql-vindex/src/index/router.rs`) wired into the client-side
  forward pass so the router runs locally.

### Phase 2 — remote expert protocol (Act 2 wire format)

- [ ] `POST /v1/expert/{layer}/{expert_id}` — input residual, output
  residual delta (hidden-size). One expert per call.
- [ ] `POST /v1/expert/batch` — list of `{layer, expert_id, residual}`,
  returns list of deltas. Lets the client collapse a layer's K experts
  into one HTTP round trip per server.
- [ ] `--experts 0-31` flag on `larql serve` — only load and serve the
  listed expert IDs. The same binary, the same vindex, just filters
  which experts get mmap'd on startup.
- [ ] `RemoteExpertBackend` in `larql-inference` — the MoE-path analog
  of `RemoteWalkBackend`. Handles the sharding map (expert ID range →
  URL), parallel dispatch per layer, error handling per expert.

### Phase 3 — ergonomics

- [ ] `USE "..." WALK ONLY WITH EXPERTS REMOTE { "range": "url", ... };`
  grammar. Extend `crates/larql-lql/src/parser/lifecycle.rs` + executor.
- [ ] `RESHARD EXPERTS { ... };` statement for live redistribution. Small
  change — reconfigures the expert backend's URL map, no model reload.
- [ ] `larql walk --experts-remote '0-31=URL1,32-63=URL2'` CLI flag.

### Phase 4 — data prep

- [ ] `larql slice <vindex> -o <out> --parts attn,embed,norms,router,index,tokenizer`
  — pure file I/O. The `router` part is new relative to the dense demo;
  verify it copies `router_weights.bin` + relevant index.json fields.

### Phase 5 — deferred (film first, then wire in)

- [ ] GPU attention on the client side (`run_attention_block_gpu` exists
  in `crates/larql-inference/src/attention/gpu.rs` but isn't the default
  path). Metal/CUDA for the laptop attention, experts remote over HTTP —
  the endgame for Apple Silicon.

### Pre-film checklist

- [ ] Confirm Gemma 4 26B A4B config: expert count per layer, top-K, exact
  active-param figure, GQA ratio. Adjust script numbers accordingly —
  don't ship with "~" figures once the model card is in hand.
- [ ] Measure real numbers on Gemma 4 31B (dense) for Act 1. Replace every
  `~` RSS/latency/byte figure with the measured value.
- [ ] Decide the repo-public date. `cargo install larql-cli && larql
  serve` should be live the week the video drops, so "you can do this
  too" lands with a working command.
- [ ] Reliability pass: `RemoteWalkBackend` + `RemoteExpertBackend` under
  real network conditions (timeouts, retries, mid-layer failure, partial
  shard outage). A hung HTTP call during recording kills the take.
- [ ] Pick expert IDs for the Video 3 teaser swap — one that fires on
  medical prompts, one that doesn't — so the teaser shot lands with a
  concrete "expert 42 at layer 18" not a generic "some expert."
