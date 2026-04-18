# LARQL Roadmap

Top-level plan of record. Per-crate specifics live in
`crates/<crate>/ROADMAP.md`; this file tracks user-visible features,
the demo narrative, and cross-crate work.

## Current state

- **490 tests passing** across 14 suites, 0 build warnings.
- **Primary CLI verbs** in place: `run`, `chat`, `pull`, `list`, `show`,
  `rm`, `link`, `serve`. Legacy research commands under `larql dev
  <subcmd>` with argv trampoline for backwards-compat.
- **Dual cache** (HuggingFace hub + `~/.cache/larql/local/`) with
  shorthand resolution (`larql run gemma3-4b-it-vindex …`).
- **Remote FFN path (Phase 0 — dense):** `POST /v1/walk-ffn`
  `full_output: true` returns hidden-size output vectors per layer;
  `RemoteWalkBackend` in `larql-inference` drops into `predict_with_ffn`
  unchanged; `larql run --ffn URL` + `larql serve --ffn-only` wire it
  end-to-end. gRPC mirror also landed.
- **Vindex size reductions:** `--compact` (drops
  `up_weights.bin`/`down_weights.bin`), `--drop-gate-vectors` (rebuilds
  gate from `interleaved_q4k.bin` at load), `--quant q4k` implies f16
  on side-channel tensors. Combined: a new 31B q4k extract is **~22 GB
  vs 52 GB before** (~60% smaller).

---

## P0 — Act 2 of the demo: "The experts live elsewhere"

### Phase 1 — MoE inference path (blocks Act 2)

The whole Act 2 story is MoE-distributed. The primitives don't exist
in `larql-inference` yet.

- [ ] **MoE-aware forward pass.** `larql-inference` has zero mentions
  of `expert`/`MoE` today. Need a layer path that calls the router,
  picks top-K experts, dispatches to a per-expert FFN backend, sums
  weighted outputs. Fits on top of the existing `FfnBackend` trait.
- [ ] **Gemma 4 MoE architecture hooks** in
  `crates/larql-models/src/architectures/gemma4.rs` — copy the Mixtral
  pattern (`is_moe`, `num_experts`, `num_experts_per_token`,
  `moe_router_key`, `expert_ffn_{gate,up,down}_key`).
- [ ] Wire `RouterIndex` (already exists at
  `crates/larql-vindex/src/index/router.rs`) into the client-side
  forward pass so the router runs locally.

### Phase 2 — Remote expert protocol (Act 2 wire format)

- [ ] `POST /v1/expert/{layer}/{expert_id}` — input residual, output
  residual delta (hidden-size).
- [ ] `POST /v1/expert/batch` — list of `{layer, expert_id, residual}`,
  returns list of deltas. Collapses a layer's K experts into one HTTP
  round trip per server.
- [ ] `--experts 0-31` flag on `larql serve` — load + serve a subset
  of expert IDs so experts can be sharded across machines.
- [ ] `RemoteExpertBackend` in `larql-inference` — MoE-path analog of
  `RemoteWalkBackend`. Handles the sharding map (expert ID range →
  URL), parallel per-layer dispatch, per-expert error handling.

### Phase 3 — LQL / CLI ergonomics

- [ ] `USE "..." WALK ONLY WITH EXPERTS REMOTE { "range": "url", ... };`
  grammar. Extend `crates/larql-lql/src/parser/lifecycle.rs` + executor.
- [ ] `RESHARD EXPERTS { ... };` statement for live redistribution
  (for the "kill one shard, rewire on the fly" proof shot).
- [ ] `larql run --experts '0-31=URL1,32-63=URL2'` CLI flag (MoE
  counterpart to `--ffn`).

### Phase 4 — Data prep

- [ ] `larql slice <vindex> --parts attn,embed,norms,router,index,tokenizer`
  (new subcommand) — carve an attention-only / router-only vindex out
  of a full one without re-extracting from the source model.

### Phase 5 — Deferred until film

- [ ] GPU attention on the client side. `run_attention_block_gpu`
  already exists in `crates/larql-inference/src/attention/gpu.rs` but
  isn't the default path in `forward/layer.rs`. Wire Metal/CUDA into
  the walk-only forward pass so client-side attention runs on GPU
  while FFN/experts go remote.

---

## P1 — Autoregressive generation quality

### CPU KV cache for autoregressive generation — **SHIPPED**

Two-phase autoregressive decoder in `larql-inference/src/forward/kv_generate.rs`:

- **Prefill** uses `run_attention_with_kv` to capture post-RoPE K and
  post-V-norm V per layer into a `KvCache`.
- **Decode** step in `crates/larql-inference/src/attention/decode.rs`:
  `run_attention_block_decode_step` takes the new token's hidden +
  the layer's existing cache, computes Q/K/V for just that row with
  `apply_rope_partial_at(position=cached_len)`, concatenates the new
  K/V onto the cache, runs `gqa_attention_decode_step` (O(cached_len)
  per head), returns updated cache.

Backend-agnostic via `FfnBackend` — works with `WalkFfn` (local) and
`RemoteWalkBackend` (FFN over HTTP). Measured on Gemma 3 4B f32:

- **Local, no cache (before):** ~1.2 s per decode step, O(N²) growing
- **Local, KV-cached (now):** ~0.6 s/token steady
- **Remote FFN, KV-cached (now):** ~0.5-0.6 s/token steady — same
  protocol as the no-cache version, just many fewer tokens re-shipped

Limitations:
- Skips Gemma 4 E2B per-layer embeddings (PLE) and layer-scalar
  application in the decode loop. Fine for Gemma 3. For full
  Gemma 4 correctness wire `apply_per_layer_embedding` + `apply_layer_scalar`
  into `generate_cached`'s decode layer.
- Q4K CPU path still uses its own no-cache loop (`run_q4k_generate_cpu`).
  Q4K + Metal shader `generate()` remains the fast Q4K path.

### KV cache strategy selector — **SHIPPED (partial)**

`larql run --kv-cache <strategy>` selects how past-token state is kept:

- `standard` *(default)* — full FP32 K/V, unbounded. Shipped.
- `markov-bounded` — sliding window (StreamingLLM-style). Shipped.
  Pass `--context-window N` for the window size. Older tokens drop
  off; memory stays O(window) regardless of generation length.
- `none` — re-run full forward per decode step. O(N²). Shipped as
  correctness fallback.

Not yet wired into the live decode path (all in `crates/kv-cache-benchmark/`):

- `markov-full` — active residual window + cold-tier reconstruction
  via checkpoint layers. Compressed storage via residuals not K/V.
  See `crates/kv-cache-benchmark/src/markov_residual/`. Needs a
  reconstruction primitive that rehydrates K/V for cold-tier
  positions from `token_ids + checkpoint_residual`.
- `turboquant` — per-tensor Q4/Q8 compression of cached K/V. See
  `crates/kv-cache-benchmark/src/turboquant/`. Needs per-step
  quantize/dequantize around the cache append.
- `graph-walk` — experimental, unclear production viability.

### Shader attention + remote FFN

### Shader attention + remote FFN (Act 2 endgame)

Q4K + Metal + remote FFN — the ultimate Act 2 configuration. The
shader pipeline (`full_pipeline_q4` / `decode_token`) currently
dispatches attention AND FFN as fused GPU kernels reading from the
Q4K mmap. For remote FFN we'd need to decompose per-layer into:
attention-only GPU kernel → copy residual to host → HTTP round trip
→ copy FFN output back to GPU → next layer's attention. Per-layer
host+network hop kills throughput unless we batch across layers or
use async pipelining.

Worth doing for the Act 2 demo but non-trivial. See
`larql-inference/src/layer_graph/{generate,pipeline_layer,prefill}.rs`
— the fused paths need splitting at the attention/FFN seam.

## P1 — Loose ends in shipped features

### `--compact` loader reconstruction — WalkFfn-only today

`larql extract --compact` drops `up_weights.bin` + `down_weights.bin`
from the extract. `WalkFfn` (the production inference path) works fine
— it reads feature-major `{up,down}_features.bin` directly. The dense
ground-truth path (`WeightFfn`, used by `larql dev walk --compare` for
validation) panics with a clear message.

**Why deferred.** The naive fix is to reconstitute
`Array2<f32>` tensors in `ModelWeights.tensors` at load time. For
`down_proj` this requires a transpose (feature-major `[intermediate,
hidden]` → safetensors `[hidden, intermediate]`) which means an owned
copy — **~27 GB of extra heap on 31B**, not viable.

**Proper fix.** Refactor `WeightFfn::forward` (or `ModelWeights`) to
accept feature-major views and pass the transpose flag through to BLAS
gemm. Cross-cutting change: `crates/larql-inference/src/ffn/weight.rs`,
`crates/larql-inference/src/model.rs`, and the `dot_proj` helpers. ~1
focused session.

**Impact.** Unblocks `--compact --compare` for validation workflows.
Does not affect `larql run` or the demo.

### MoE compact mode — refused today

`larql extract --compact` on an MoE architecture refuses with:
> *"ffn_compact not yet supported for MoE architectures — per-expert
> feature-major files don't exist yet"*

**Why deferred.** Two blockers:

1. **Router lives in `up_weights.bin`.** The MoE write path stuffs
   per-expert up weights *and* the router matrix together into
   `up_weights.bin`. Skipping that file loses the router, so the model
   can't dispatch to experts at all. Fix: split the router into its
   own file (`router_weights.bin` already exists as the intended home
   — see `crates/larql-vindex/src/index/router.rs`).
2. **No per-expert feature-major files.** `up_features.bin` /
   `down_features.bin` are single-matrix-per-layer. MoE-compact would
   need per-expert equivalents (~N× the file count or a new layout),
   plus a tool that produces them. No consumer exists yet.

**When to do it.** Pairs naturally with Phase 1 (MoE inference path)
and Phase 2 (per-expert server endpoint). Building those requires a
per-expert-addressable storage layout anyway; compact-MoE falls out of
it.

### `larql dev walk --compact` compatibility

`larql dev walk --compare` against a `--compact` vindex panics (see
above). The panic message points at `WalkFfn` but doesn't explain
`--compare` is the specific operation that's blocked. Improve the
error or disable the `--compare` flag at arg-parse time when the
target vindex is compact.

### Cross-vindex dedup (tokenizer, down_meta)

Tokenizer (~32 MB) and `down_meta.bin` (~30 MB) are identical across
different-precision extracts of the same base model. With ~7 linked
vindexes in the local cache that's ~200 MB of duplicate data. Low
priority — worth doing as a content-addressed store if the cache
grows, otherwise skip.

---

## P2 — Demo production

### Pre-film checklist for the Gemma 4 MoE video

- [ ] Confirm Gemma 4 26B A4B config once the model card is public:
  expert count per layer, top-K, exact active-param figure, GQA ratio.
  Every `~` figure in `docs/demo-script-gemma4-moe.md` needs a real
  number before recording.
- [ ] Measure real footprint + latency on `google/gemma-4-31b-it` for
  Act 1. Replace every `~` in the Act 1 section.
- [ ] Reliability pass on `RemoteWalkBackend` (timeouts, retries,
  mid-layer failure, partial shard outage). A hung HTTP call during
  recording kills the take.
- [ ] `RemoteExpertBackend` (doesn't exist yet — see Phase 2) same
  pass.
- [ ] Decide the repo-public date. `cargo install larql-cli && larql
  serve` should be live the week the video drops so "you can do this
  too" lands with a working command.
- [ ] Pick expert IDs for the Video 3 teaser swap — one that fires on
  medical prompts, one that doesn't — so the "replace expert 42 at
  layer 18" shot lands concretely.

### Memory-footprint `--ffn-only` on the server

`larql serve --ffn-only` today is an operating-mode declaration — it
disables `/v1/infer`, advertises `mode: ffn-service` in `/v1/stats`,
but still loads full `ModelWeights` into RAM. A real FFN-service
doesn't need attention weights resident.

Add `load_model_weights_ffn_only` to `larql-vindex` that skips
attention tensors on the server side. Payoff: serve an MoE without
the attention weights taking a third of RAM.

---

## Done (ship log)

### CLI redesign (primary / dev split)
- New verbs: `run`, `chat`, `pull`, `list`, `show`, `rm`, `link`.
- Research commands moved under `larql dev <subcmd>`; legacy names
  transparently trampolined.
- Dual cache (HuggingFace hub + `~/.cache/larql/local/`) with
  shorthand resolution and source disambiguation.
- `larql serve --ffn-only` flag propagated through CLI → server →
  `/v1/stats`.

### Phase 0 — dense remote FFN baseline
- `POST /v1/walk-ffn` extended with `full_output: true` +
  `seq_len: N`. Server runs the architecture-correct `WalkFfn`,
  returns `[seq_len × hidden]` row-major.
- gRPC mirror (`WalkFfnRequest` / `WalkFfnLayerResult` proto fields).
- `RemoteWalkBackend` in `larql-inference` implements `FfnBackend`,
  slots into `predict_with_ffn` unchanged.
- `larql run --ffn URL` + `larql dev walk --ffn-remote URL` CLI flags.
- `examples/remote_walk_parity.rs` localhost parity probe.

### Vindex size reductions
- `--quant q4k` defaults gate_vectors + embeddings to f16 (previously
  f32 — silent ~32% bloat on every q4k extract).
- `--compact` skips `up_weights.bin` + `down_weights.bin` (saves 3.4
  GB on 4B f16 / ~14 GB proportionally on 31B non-Q4K).
- `--drop-gate-vectors` skips `gate_vectors.bin` on Q4K extracts;
  loader reconstructs from `interleaved_q4k.bin` at load time. 2.3 s
  on 4B / ~12 s on 31B cost, saves 1.7 GB / 13.9 GB respectively.
  Measured via `crates/larql-vindex/examples/bench_gate_dequant.rs`.

### Decoupled-inference memory asymmetry (real, pre-load filtered)
- `LoadWeightsOptions { skip_attn, skip_ffn, skip_lm_head, skip_embed }`
  filters weight manifest entries before mmap+decode — peak RSS
  reflects only what the caller wanted (no allocator-pooling lie).
- Server `--ffn-only`: skips attn + ffn + lm_head + embed at load.
  Walk-ffn endpoint uses `walk_ffn_full_mmap` which reads
  feature-major mmap, not heap tensors.
- Client `--ffn URL`: skips FFN tensors at load. Attention + embed +
  norms + lm_head only on heap.
- Measured on Gemma 3 4B f32 (`gemma3-4b-v2.vindex`):
  - Server RSS: 12.8 GB idle → **12.8 GB through inference** (never grew)
  - Client load: 22.5 s → **7.9 s** (2.8× faster)
  - Forward pass: 3.83 s → **0.83 s** (4.6× faster — no FFN tensor
    touches on the client)
  - Paris @ 80.66% — bit-identical to local unlimited-K walk
- Drop-post-load helpers (`ModelWeights::drop_{attn,ffn,lm_head,embed}_weights`)
  still exist but Rust's system allocator pools freed memory —
  post-load drops reduce heap accounting but not process RSS.
  Superseded by the pre-load filter for the demo path.
- `larql serve` now resolves cache shorthands (`larql serve gemma4-31b-q4k`
  works, not just full paths) via the same `cache::resolve_model`
  logic `larql run` uses.
- `larql run` / `larql dev walk` default `--top-k` to `usize::MAX`
  (unlimited). The old `top-k=10` default silently produced garbage
  on stale/low-K vindexes; removing the cap matches the server's
  `WalkFfn::new_unlimited` behavior.

### Extract tiers + default flip
- New `ExtractLevel::Attention` tier sits between `Browse` and
  `Inference`: includes attention + norms but not FFN. This is the
  first-class way to carve a client-side vindex for the Act 2 demo
  (`larql extract <model> --level attention`). No more ad-hoc slicing.
- Strict `Browse < Attention < Inference < All` ordering + helper
  methods (`writes_attn()` / `writes_ffn()` / `writes_lm_head()`)
  drive what each tier writes. Writers now actually honor the
  boundaries — previously only Browse was meaningfully different from
  non-Browse.
- **Default flip.** `larql extract` now defaults to `--level inference`
  + f16. The common case (`larql extract <model> -o x.vindex`) produces
  an inference-ready vindex out of the box, no flags needed. `--f32`
  opts out of f16 for the rare case someone wants it.

### Gemma 4 config plumbing
- Fixed three missing `final_logit_softcapping` initializers
  (pre-existing compile break on the `architecture-b` branch).
- Dropped an unused `mut` on a closure binding in
  `format/weights/write.rs`.

### Test coverage
- **490 tests across 14 suites**, zero warnings.
- New: cache resolution (19), argv trampoline (8),
  `RemoteWalkBackend` wire format + config + error shape (10), server
  validation + stats mode advertisement (7), local-cache scan
  end-to-end.

---

## Non-goals

- **Not a general model-serving framework.** LARQL's pitch is "the
  model is the database"; inference is a vehicle for the interpretable
  vindex, not the product. We optimize for composability, editability,
  and the demo narrative — not raw throughput against vLLM/TensorRT.
- **Not a training system.** `COMPILE` writes into weights; that's
  patch-level edits, not gradient descent. Stays out of scope.
- **Not HF-compatible on the output side.** We extract *from* HF
  models but the vindex format is our own. A vindex is not meant to be
  loadable by `transformers.AutoModel`.
