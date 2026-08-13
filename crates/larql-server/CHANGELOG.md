# Changelog — larql-server

All notable changes to `larql-server` are documented here.

The format follows the conventions of [Keep a Changelog](https://keepachangelog.com/),
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

## [2026-05-28] — Hardening findings from the whole-codebase review

From the whole-codebase review ([`docs/audits/codebase-review-2026-05-28.md`](../../docs/audits/codebase-review-2026-05-28.md)):

- **P1 — unbounded in-memory growth with dead eviction logic.** The session map (`src/session.rs:184`) and rate-limit buckets (`src/ratelimit.rs:83`) never evict. Memory/DoS class — wire up the eviction that already exists but isn't called.

Recorded, not fixed. Still open — see [`ROADMAP.md`](ROADMAP.md) §"Open
defects".

## [2026-05-07] — State of the server at this date (migrated "Current state")

Migrated from `ROADMAP.md` on 2026-08-04, where it had stood as the *current*
state for nearly three months. Read as a snapshot with a date on it, not as a
description of the server today.

### 2026-05-07 — Wire format evolution + WebSocket streaming + Criterion benchmarks

GT1–GT4 and GT9 shipped. See `G-TRANSPORT` and `G-BENCH` sections below for full details.

**GT1 (f16 wire default)**: `application/x-larql-ffn-f16` content-type; Accept header
negotiation; `encode_binary_output_f16` in server; `decode_binary_single/batch_f16`
in client. Client sends `Accept: i8, f16, f32`; server honours f16 by default
(`LARQL_F16_WIRE_DISABLE` opt-out). 50% bandwidth reduction on all grid paths.

**GT2 (i8 residuals)**: `application/x-larql-ffn-i8`; per-position symmetric
quantisation (scale = max(|x|)/127, zero_point = 0); `encode_binary_output_i8` +
`decode_binary_single/batch_i8`. Opt-in via `LARQL_I8_WIRE=1`. 75% bandwidth reduction.

**GT3 (per-layer latency)**: `LayerLatency { layer, avg_ms, p99_ms }` added to
`HeartbeatMsg.layer_stats` and `ServerInfo.layer_stats` in grid.proto. Server collects
EMA + p99 ring-buffer per layer in `metrics::LayerLatencyTracker`; heartbeat sends
snapshot every 10s. Router stores per-layer latency in `ServerEntry.layer_latencies`
and prefers lowest `avg_ms` for the requested layer when routing replicas.

**GT4 (WebSocket generate)**: `WS /v1/stream` now supports `{"type":"generate",
"prompt":"...","max_tokens":N}` command. Streams `{"type":"token","text":"...","index":N}`
frames per token; emits `{"type":"done","tokens":N,"latency_ms":M}`. Client can send
`{"type":"cancel"}` to abort. Uses same `generate_streaming` engine as SSE chat completions.
SSE streaming on `POST /v1/chat/completions` (N0.1 slice 3) confirmed already wired.

**GT9 (Criterion benchmarks)**: `larql-inference/benches/wire_codec.rs` (encode/decode
throughput at h2560/h4096/h5120, seq1/32/256). `larql-router/benches/routing.rs`
(route/route_all/update_heartbeat/rebuild at 1/10/100 servers). `larql-router` now
has a `lib.rs` exposing `grid` for tests/benches. Makefile targets: `bench-wire`,
`bench-routing`, `bench-grid`, `bench-all`.

### 2026-05-01 — HTTP CPU-path optimisation session

End-to-end Gemma 4 26B-A4B grid jumped from ~17.7 → ~19.7 tok/s on
M3 Max with one local gRPC shard. New per-call wire format,
streaming-overlap default-on, UDS transport, TCP_NODELAY, f16 wire
opt-in. See `Completed` section below for the full per-change list.

### Inherited state (2026-04-26)

- Code quality pass complete: modularity refactor + magic string cleanup + test restructure (see Completed below).
- Follow-up review fixes complete: rate limiting no longer trusts
  `X-Forwarded-For` by default, route/path strings are centralized,
  server loader options are grouped, embed errors use the standard JSON
  error envelope, and server-local clippy allows were reduced.
- Test coverage: **74.2% line / 81.2% function** at the 2026-04-26
  baseline (478 tests). 2026-05-01 (post Q1 cleanup): **131 lib tests +
  37 integration files (~580 tests total), all green**.
- 2026-05-10 re-measurement (post REV1–REV5 fixes, full `--tests`
  run): **65.68% line / 72.18% function** across ~660 tests.
  Coverage drifted ~8 pts down vs the 2026-04-26 claim — partly
  because the upstream `larql-inference` / `larql-compute` API
  refactor temporarily broke `tests/test_expert_endpoint.rs` (now
  fixed) and the four `routes/expert/*` modules sit at 0% line
  coverage because they need a live grid harness to exercise. Per-
  file 90% floor + debt baselines now codified in
  `crates/larql-server/coverage-policy.json`; run
  `make larql-server-coverage-summary` to re-measure (added in this
  session — see CHANGELOG.md).
- Q1 code-quality cleanup (2026-05-01) shipped 9 of 10 items: 1044-LOC
  `routes/expert.rs` split into 7 focused files; 656-LOC `main.rs` reduced
  to 26 LOC with `bootstrap::serve(cli)` as the orchestration point; new
  `env_flags.rs` (single source of truth for `LARQL_*` knobs) and `wire.rs`
  (shared content-type detection); body-size / JSON-content-type / Cli
  default literals all lifted to typed consts. Q1.10 (stream.rs WebSocket
  state machine) deferred until N0.1 SSE infrastructure lands. See
  Completed → "2026-05-01 (continued) — Q1 code-quality cleanup".
- Server-local clippy was clean at the 2026-04-26 baseline with
  `cargo clippy -p larql-server --tests --no-deps -- -D warnings`,
  re-verified clean post-Q1 on 2026-05-01.
  The dependency-checking form still stops in `larql-vindex`; that is
  tracked outside this server-only pass.
- Examples and synthetic benchmarks checked on 2026-04-26 and re-verified
  2026-05-01 (post Q1 cleanup, re-validated): `server_demo`, `embed_demo`,
  `server_bench --release`, `bench_expert_server` (live MoE bench against
  `gemma4-26b-a4b-q4k.vindex`), `bench_embed_server` (live f16 mmap embed
  against `gemma3-4b-q4k-streaming.vindex`) all pass. Numbers within
  noise of pre-Q1 baselines — see Live perf snapshot below.
- Grid route-table checks are now covered by `cargo test -p larql-router`
  (20 tests, including 7 grid-state tests) plus server announce-envelope tests.
- 2-shard local grid validated end-to-end on Gemma 4 26B-A4B (30 layers,
  inclusive layer ranges 0-14 + 15-29).
- W2 feature-major down retrofittable in-place via
  `larql convert add-feature-major-down --input <vindex>` (1.12 s for
  30 layers, 152 MB output).
- Live W2 surface on `GET /v1/stats.q4k_ffn`:
  `{cache_slots, cache_bytes, feature_major_down}`.
- `--warmup-hnsw` flag eager-builds HNSW across owned layers at boot
  (~325 ms for 15-layer shards on Gemma 26B).
- Grid memory profile (per-shard, single-machine): **9.1 GB RSS**,
  6.7 GB MALLOC_LARGE (gate f32 cache), `down_features_q4k.bin`
  resident at 0 K (capability, not yet exercised on dense path).

## [2026-05-07] — Perf snapshot: M3 Max, 2-shard grid, 26B-A4B

Point-in-time measurements, migrated from `ROADMAP.md` on 2026-08-04. The
"Remote MoE expert path" table that `README.md` links to lives here.

### Dense walk-ffn / gate-KNN path

| Operation | Cold | Warm |
|---|---|---|
| `walk-ffn` 1 layer (router) | 12.8 ms | **0.2–0.3 ms** |
| `walk-ffn` 6 layers fanout | — | **1.3 ms** |
| `walk-ffn` 12 layers fanout | 64 ms | 2.6 ms |
| `walk-ffn` 24 layers fanout | 75 ms | 5.0 ms |
| `walk-ffn` 30 layers (full) | 30 ms | **5.9 ms** |
| `walk` (gate KNN, 30L) | — | 8.4 ms |
| 8-way concurrent × 15L fan-out | 112 ms wall | ~1070 layer-evals/sec |

P99 under 8-way contention: 24 ms.

### Remote MoE expert path (Gemma 4 26B-A4B, single in-process shard, layer 15, top-K=8)

`bench_expert_server` against per-layer Q4_K vindex
(`output/gemma4-26b-a4b-q4k.vindex`). Hidden=2816, 128 experts,
moe_intermediate=704, 30 MoE layers.

**bench numbers (2026-05-01, re-validated post Q1 cleanup; same hardware,
same vindex, same kernel path — confirms the refactor is bit-exact):**

| Operation | Result | (vs 2026-05-01 pre-Q1) |
|---|---|---|
| Vindex load | 5.4 s, +6.0 GB RSS | 5.2 s, +6.0 GB RSS |
| Lazy `get_or_load_weights()` | 1.36 s, +2.85 GB RSS | 1.3 s, +2.8 GB |
| Per-expert bytes (one bench layer, all 128) | 285 MB gate_up + 156 MB down (Q4_K) | unchanged |
| `forward_moe` warm (router + layer-batch HTTP + combine) | **0.78 ms** mean / 0.78 p50 / 0.88 p99 | 0.80 / 0.79 / 1.09 |
| `cpu_moe_forward` floor (no HTTP, same weights) | **0.34 ms** mean / 0.35 p50 / 0.43 p99 | 0.37 / 0.37 / 0.49 |
| 30-layer sweep (1 decode-step's worth of MoE blocks) | **23.24 ms** (0.77 ms/layer) | 24.8 ms (0.83 ms/layer) |
| Steady RSS | **10.5 GB** | 10.5 GB |

The 2-3% delta between pre- and post-cleanup runs is hardware noise (M3
Max thermal state varies 1-3% across runs) — the refactor moved code
across files but did not change any kernel.

**End-to-end Gemma 4 26B-A4B grid generation (`larql run --moe-shards`,
M3 Max, single local shard, 100-token poem, 3-run avg)**:

| Mode | tok/s |
|---|---|
| HTTP unary (`http://...` shard) | **17.8** |
| gRPC unary (`grpc://...` + `LARQL_MOE_NO_SPLIT=1`) | 17.7 |
| **gRPC + SPLIT overlap (default for gRPC)** | **19.7** |
| UDS HTTP/1.1 (`unix:///path` shard) | 18.2 |
| UDS + f16 wire (`LARQL_MOE_WIRE_F16=1`) | 20.5 (warm); within noise vs UDS f32 |

**Per-call HTTP overhead (loopback, post TCP_NODELAY)**:

| Stage | TCP HTTP | UDS HTTP | gRPC streaming |
|---|---|---|---|
| Server compute (run_experts_cpu_batch) | ~400 µs | ~400 µs | ~400 µs |
| spawn_blocking transition | ~25 µs | ~25 µs | ~25 µs |
| Transport RTT + axum dispatch | ~100 µs | ~50 µs | ~30 µs (multiplexed) |
| Encode + decode | ~5 µs | ~5 µs | ~5 µs (binary protobuf) |
| **Total per-call** | **~660 µs** | **~510 µs** | **~460 µs** |

For comparison, the historical baseline before any of this session's work
was 4.86 ms `forward_moe` warm and 16.6 GB steady RSS on the BF16
monolith (per-expert refactor + Q4_K migration cut that to 1.91 ms / 9.7
GB at 2026-04-26). The 2026-05-01 session took 1.91 ms → 0.78 ms
(another 2.4×) on the same per-call measurement, 56 ms → 23.24 ms
(2.4×) on the 30-layer sweep, and end-to-end ~17.7 → ~19.7 tok/s
(+12%) on the production grid. Cumulative session-on-session win is
**8.6× from the 2.3 tok/s pre-Q4K baseline** (see
`larql-inference/ROADMAP.md → M-CPU-1..6`).

### Embed-service path (Gemma 3 4B, ADR-0008 f16 mmap)

`bench_embed_server` against `gemma3-4b-q4k-streaming.vindex` (262144 ×
2560 vocab × hidden, ~1.34 GB f16 embeddings.bin):

| Operation | Result |
|---|---|
| mmap open (cold, no faults) | 0 ms, RSS 280 MB |
| L1 cache fill (5000 hottest tokens) | 25.2 ms, RSS 426 MB |
| f16 embed 1 token — L1 hit | **4.3 ns/op** (232 M ops/s) |
| f16 embed 1 token — mmap decode (L1 miss) | 3.22 µs/op (310 K ops/s) |
| f16 embed 32 tokens (prefill, mmap decode) | 59.07 µs/op |
| f16 embed 128 tokens (prefill, mmap decode) | 239.18 µs/op |
| f16 embed 512 tokens (prefill, mmap decode) | 1.10 ms/op |
| Logits projection (262208 × 2560, full vocab, CPU) | 335.6 ms (Metal: ~0.67 ms) |

Memory comparison (`--embed-only`, ADR-0008):

| Layout | RSS |
|---|---|
| f32 heap eager decode | ~2.9 GB |
| **f16 mmap + L1 cache (5000 tokens)** | **~1.6 GB** (48% reduction) |

## [2026-05-17] — Q4K synthetic vindex fixture; completions un-excluded

Built the Q4K fixture I'd named as the gate for chat / completions /
stream coverage in the prior session-close note. Generation paths
now exercise real Q4K storage without panicking on `attn Q4K slices
missing for layer N`.

### Added

- **`tests/common/synthetic_q4k_vindex.rs`** — `build()` produces a
  full Q4K vindex on disk by (a) writing a tiny Llama-shaped
  safetensors model to a tempdir, (b) running it through
  `larql_vindex::build_vindex_streaming` with
  `QuantFormat::Q4K`. Mirrors the gold-standard pattern from
  `larql-vindex/tests/test_vindex_to_q4k.rs::q4k_end_to_end_from_synthetic_safetensors`.
  Dims: hidden=8, intermediate=4, num_layers=2, vocab=16 — small
  enough that each tensor pads to exactly one 256-element Q4_K
  super-block.
- **`tests/common/mod.rs::model_with_q4k_weights()`** — returns
  `(Arc<LoadedModel>, SyntheticQ4kVindex)`. Mirrors production
  `bootstrap.rs:238-256`: calls `VectorIndex::load_attn_q4k` +
  `VectorIndex::load_interleaved_q4k` explicitly after the base
  `load_vindex` so the Q4K data is actually attached to the index
  (without those calls, `insert_q4k_layer_tensors` still panics
  even though the on-disk files exist). The fixture's tokenizer is
  overridden to a 12-entry WordLevel — the streaming pipeline
  defaults to an empty BPE, which would encode every prompt to 0
  tokens and short-circuit the generation loop.
- **`tests/test_synthetic_q4k_smoke.rs`** — 3 tests: file-layout
  inventory, `get_or_load_weights` succeeds, `insert_q4k_layer_tensors`
  returns Ok. The third was the actual gate from the prior session.
- **`tests/test_openai_chat_coverage.rs`** — 11 tests covering the
  chat endpoint against the Q4K fixture: basic non-streaming and
  streaming, system message → template rendering, empty messages
  400, n>1 400, invalid JSON 400, sampling params, stop strings,
  `response_format: json_object`, tools, and multi-model dispatch.
  (Not yet measured at time of writing — handed off to a parallel
  session.)
- **`safetensors = "0.7"`** added as a larql-server dev-dependency
  (mirrors the version pinned by larql-vindex).

### Changed

- **`tests/test_openai_completions_coverage.rs`** swapped to use
  `model_with_q4k_weights` instead of `model_with_real_weights`.
  Drains the streaming SSE body fully (was capped at 64 KiB) and
  asserts on the `[DONE]` terminator.
- **`coverage-policy.json`** — `routes/openai/completions.rs`
  removed from `exclude_globs` and added at debt baseline 86%
  (was 40% pre-session, lifted to 86.85% via the Q4K backing +
  streaming-drain + stop-string tests). The remaining ~13% is the
  per-token streaming callback body — the synthetic Q4K generator
  returns 0 tokens with max_tokens=2 so the callback is never
  invoked.

### Known follow-up

- **Per-token streaming callback path stays uncovered.** The Q4K
  fixture's generator produces 0 tokens given the synthetic weights
  (the diagonal-ramp weights produce all-`-inf` logits after rope +
  Q4K dequant). Hitting the per-token callback body would need
  either a longer `max_tokens` budget or tuning the weights so the
  logits aren't degenerate. Tracked as a finer-grained follow-up,
  not blocking.

## [2026-05-17] — Coverage push session close: 75% total, 37 files at default 90%

Closing pass after the chat/completions/stream fixture wall was
diagnosed. Realised the wall isn't NaN-prone weights — it's that
`generate_with_sampling` panics with `attn Q4K slices missing for
layer 0` when called on a non-Q4K vindex (confirmed in
`vindex/kquant_forward/cached.rs:106`). The generation paths require
a Q4K-quantised synthetic vindex, not just stable weights. So picked
off the remaining files that don't depend on that fixture instead:

### Un-excluded this round

- **`routes/openai/schema/mask.rs` (0% → 93.44%)** — 6 in-test cases
  cover lazy surface-table init, the cache-hit replay path, the
  cache-miss-falls-through path, prompt-prefill replay, EOS masking
  while FSM incomplete, and the out-of-table token graceful fallback.
- **`openapi.rs` (0% → 100%)** — 3 tests against `swagger_router()`
  and `ApiDoc::openapi()`: serves `/v1/openapi.json` (validates
  `openapi: "3.x"`), serves the Swagger UI index, and the `ApiDoc`
  struct emits a non-empty `paths` object.

### Debt-baseline ratchets

Three files reached or cleared the 90% default and were lifted out
of the `per_file_line_min_percent` debt list entirely:

- `env_flags.rs` 59.6% → **100%**
- `shard_loader.rs` 88.0% → **91.72%**
- `state.rs` 85.8% → **91.90%**

One file ratcheted upward within the debt list:

- `routes/warmup.rs` 80.8 → **84.0** (deeper paths still need fixture)

### Q4K-fixture confirmation

A diagnostic smoke test (deleted after diagnosis) panicked with
`attn Q4K slices missing for layer 0` from
`vindex/kquant_forward/cached.rs:106` when calling
`generate_with_sampling` on the synthetic f32 vindex. That confirms
chat / completions / stream / walk_ffn/q8k all need the same
fixture: a Q4K-quantised synthetic vindex with `attn_q4k.bin` +
`interleaved_q4k.bin` plus the required dimensions for the K-quant
kernels. Not built this session.

### Session totals

| Metric | Pre-session | End of session |
|---|---|---|
| Total | 69.82% | **75.05%** (+5.23 pp) |
| Included files | 91.87% | **92.62%** |
| Files at 90% default | 26 | **37** (+11) |
| Debt baselines | 10 | **8** (3 freed, 1 ratcheted up) |
| Tests | 739 | **813** (+74) |

Files un-excluded across the session: `routes/explain.rs`,
`routes/infer.rs`, `routes/openai/schema/mask.rs`, `openapi.rs`.
Plus `walk_ffn.rs` split into a 7-file module folder (5 of 7 land at
≥93% out of the gate; 2 — core.rs and q8k.rs — stay excluded for
documented MoE + Q4K fixture reasons).

## [2026-05-17] — infer.rs un-excluded; fixture wall hit on chat/completions/stream

Continued coverage push. `routes/infer.rs` joins `routes/explain.rs`
in the included set (50% → 97% with 9 new integration tests +
2 new in-file unit tests for `format_knn_override`). Started
coverage tests for `routes/openai/completions.rs` and
`routes/expert/warmup.rs` — both plateau without further fixture
work:

- **`openai/completions.rs` (40% → 56%)**: synthetic vindex's
  diagonal-ish f32 weights NaN partway through
  `generate_with_sampling`, so the per-prompt completion-loop body
  + the streaming spawn_blocking body stay uncovered. Tests
  exercise the handler validation (n>1, empty prompt, echo+stream
  rejection, batched+stream rejection, sampling params, stop
  strings, logprobs, SSE content-type), which lifts the entry
  branches. Stays excluded until stable synthetic generation weights
  land. 12 integration tests pass.
- **`expert/warmup.rs` (0% → 9%)**: only the env-gated and
  `is_hybrid_moe() == false` early-return branches are reachable
  with the current dense-llama synthetic. The deeper unit/expert
  filter resolution + HNSW build path needs a synthetic hybrid-MoE
  arch (router weights + N experts). 2 integration tests pass.

### Realistic-ceiling note

Pushing the remaining excluded files (`openai/chat.rs`,
`openai/completions.rs`, `stream.rs`, all `expert/*.rs`,
`walk_ffn/core.rs`, `walk_ffn/q8k.rs`) above 90% requires three
substantial fixture extensions:

1. **Stable generation weights** — synthetic vindex weights tuned
   so `predict_with_ffn` / `generate_with_sampling` produce
   finite (non-NaN) output for multiple tokens. Unlocks chat,
   completions, stream, infer's compare-mode deeper paths.
2. **Synthetic hybrid-MoE vindex** — router_proj + N expert weights
   + arch override returning `is_hybrid_moe() == true`. Unlocks
   all `expert/*.rs` files + `walk_ffn/core.rs`'s MoE branch.
3. **Synthetic Q4K-quantised vindex** — different storage format
   (interleaved_q4k.bin etc). Unlocks `walk_ffn/q8k.rs`.

Each is ~1-2 days of fixture engineering. The honest path to total
≥ 90% is to land those fixtures, not to keep adding ineffective
tests against the current dense-llama synthetic.

### Headline numbers (this session)

| Metric | Pre-session | End of session |
|---|---|---|
| Total | 69.82% | **74.52%** (+4.70 pp) |
| Included files | 91.87% | **92.61%** |
| Files at 90% default | 26 | **32** (+6) |
| Files excluded | 13 | 12 (replaced walk_ffn.rs monolith with core.rs + q8k.rs; net −1 after explain + infer un-exclusion) |
| Test count | 739 | **804** (+65) |

## [2026-05-17] — walk_ffn split into a module folder + coverage progress

Continuing the coverage push: split the 1434-line `routes/walk_ffn.rs`
monolith into a 7-file `routes/walk_ffn/` module folder, then drove
coverage on the new sub-files. Net: total 71.18% → **73.76%**;
included 91.98% → **92.42%**; 27 → **31** files at default 90%.

### Changed

- **`routes/walk_ffn.rs` → `routes/walk_ffn/`**: split into seven
  files matching the natural section headers in the original:
  - `types.rs` (100%) — `WalkFfnRequest`, `FfnEntry`, `FfnOutput`,
    `RifGuard`, `BINARY_CT`, `BATCH_MARKER`.
  - `binary.rs` (98.56%) — codec (`decode_binary_request`,
    `encode_binary_output` / `_f16` / `_i8`, `encode_json_full_output`)
    + the existing in-file unit tests + new tests for the f16 / i8
    encoder branches that the previous monolith left uncovered.
  - `validate.rs` (93.55%) — pure-function validators
    (`collect_scan_layers`, `validate_residual`, `validate_owned`)
    with 7 new in-file unit tests.
  - `dispatch.rs` (98.39%) — JSON entry (`run_walk_ffn` +
    `run_full_output` / `run_features_only`).
  - `handler.rs` (86.90% — debt baseline) — axum entrypoint
    `handle_walk_ffn`. The 13% uncovered are error paths inside
    `tokio::task::spawn_blocking` closures (no-model-loaded,
    read-body-error, JSON-serialize-error) that are hard to drive
    from integration tests without a special harness.
  - `core.rs` (24.89% — still excluded) — `run_full_output_core`.
    The MoE branches (~80 lines) need a remote-MoE backend the
    synthetic fixture doesn't provide.
  - `q8k.rs` (34.18% — still excluded) — `handle_walk_ffn_q8k`. Needs
    a Q4K-quantised vindex; same fixture gap.
  - `mod.rs` — re-exports preserving `crate::routes::walk_ffn::X`
    paths so callers (`routes/mod.rs`, `openapi.rs`) didn't have
    to change. utoipa's generated `__path_*` types are re-exported
    so the `#[derive(OpenApi)]` macro keeps finding them at the
    pre-split path.

### Coverage policy

- `routes/walk_ffn.rs` removed from `exclude_globs`; replaced with
  `routes/walk_ffn/core.rs` + `routes/walk_ffn/q8k.rs` (the two
  submodules that need MoE / Q4K fixtures to cover).
- `routes/walk_ffn/handler.rs` added at 86% debt baseline.
- Gate: total 73.76% lines, included 92.42%, 42 files checked,
  31 at default 90%, 11 debt baselines. **Up from 36 files / 26 at
  default / 10 baselines pre-split.**

### Tests

- 787 server tests pass (was 748 at start of session, +39 from
  walk_ffn split's in-file unit tests + coverage tests + new
  validate tests). No regressions across `test_grid_*`,
  `test_http_*`, `test_synthetic_vindex`, `test_walk_ffn_coverage`.

## [2026-05-17] — Synthetic-vindex test fixture + coverage push begins

Phase 1 of the larql-server coverage push (target: total ≥ 90%). Built
a reusable test fixture that constructs a complete f32 vindex on disk
from synthetic deterministic weights, then drove the pilot file
`routes/explain.rs` from 44.86% → 93.46% (clears the 90% per-file
floor). Total server coverage 69.82% → 71.18%; included-files
coverage 91.87% → 91.98%.

### Added

- **`tests/common/synthetic_vindex.rs`** — `build()` produces a tempdir
  with `index.json` (`has_model_weights: true`), `weight_manifest.json`,
  `gate_vectors.bin`, `attn_weights.bin`, `up_weights.bin`,
  `down_weights.bin`, `norms.bin`, `lm_head.bin`, `embeddings.bin`,
  `down_meta.bin`, and `tokenizer.json` — exactly what
  `larql_vindex::load_model_weights_with_opts` consumes. Synthetic
  weights match `larql-vindex/tests/test_vindex.rs::make_synthetic_model`:
  2 layers × hidden=8 × intermediate=4, vocab=16. Build time ~10ms.
- **`tests/common/mod.rs::model_with_real_weights(id)`** — returns
  `(Arc<LoadedModel>, SyntheticVindex)`. `LoadedModel.path` points at
  the fixture so `get_or_load_weights()` (called by every
  `full_output=true` route handler) succeeds. Sibling
  `_and_labels(id, labels)` variant seeds `probe_labels` for tests of
  the `relations_only` branches.
- **`tests/test_synthetic_vindex.rs`** — 9 tests:
  fixture smoke (2) + explain-handler coverage (7 — basic, attention,
  relations_only, relations + labels, band filter, multi-model,
  multi-model 404, invalid JSON). Runs in ~30ms.

### Changed

- **`coverage-policy.json`** — `routes/explain.rs` removed from
  `exclude_globs`; per-file 90% default now applies. 27 files (was 26)
  clear the default; 10 debt baselines unchanged. Total floor
  unchanged at 65%, included floor unchanged at 90%.

### Playbook for next sessions (Tasks #94 — remaining 7 excluded files)

Pattern that worked on explain.rs:

1. Pick the next excluded file from `coverage-policy.json::exclude_globs`.
   Recommended order by ROI: `routes/walk_ffn.rs` (874 missed lines, biggest
   single gain), `routes/openai/chat.rs` (547), `routes/openai/completions.rs`
   (303), `routes/stream.rs` (421), `routes/infer.rs` (195),
   `routes/expert/*` (5 files at 0-57%), `grpc.rs` (302).
2. Write one smoke test using `common::model_with_real_weights` that POSTs
   to the file's main route handler. Measure coverage delta.
3. Add 4-6 more tests targeting uncovered branches surfaced by
   `cargo llvm-cov report --package larql-server --json`. The vocab in
   `synthetic_vindex.rs` is small — most uncovered ranges are
   reachable by adjusting query params (`band`, `relations_only`,
   `with_attention`, `top_k`, full_output flags) or by giving the
   fixture a payload it can run through.
4. When the file clears 90%, remove it from `exclude_globs` and
   confirm `make larql-server-coverage` passes.

Caveats observed during the pilot:

- The fixture's tokenizer needs at least a small WordLevel vocab.
  An empty BPE encodes every prompt to 0 tokens; every per-token
  branch in the route handler then stays uncovered. The shipped
  fixture uses 12 WordLevel entries; adjust as needed.
- The fixture's intermediate / hidden sizes are tiny on purpose
  (build time matters). If a route needs larger shapes to exercise a
  specific branch (e.g. multi-head attention paths), bump
  `ModelDims` in `make_weights()`.
- `LoadedModel` is `!Clone`; pass `probe_labels` at construction
  via `model_with_real_weights_and_labels(id, labels)` rather than
  mutating after `Arc::new`.

## [2026-05-16] — Mode B / QUIC ROADMAP backfill + GT5 end-to-end test

ROADMAP-drift sweep: three G-MODEB / G-TRANSPORT items previously
listed as "Not started" were actually shipped between 2026-05-13 and
2026-05-15 (on the router side) and earlier on the server side. The
server ROADMAP was updated to reflect reality and the missing
end-to-end test was added.

### Fixed

- **GT5 (Mode B gap-fill) — server-side ROADMAP corrected + new
  end-to-end test.** `announce.rs::run_available_loop` had been wired
  end-to-end (`AvailableMsg` → handle `AssignMsg` →
  `shard_loader::download_and_load_shard` → `ReadyMsg` /
  `RefuseMsg` → loop until `AckMsg`) since 2026-05-13, but no
  integration test drove the *production* loop —
  `mode_b_full_vertical_handoff` inlined the protocol in the test
  body. New test
  `mode_b_try_once_available_drives_full_handshake` spawns the real
  loop via the newly-public `announce::try_once_available` entry
  point against an in-process router fixture and asserts Available →
  Assign → download → Ready → Ack lands in <3s.
- **Misleading Mode A AssignMsg log.** `announce.rs:413` used to log
  `"Received AssignMsg but Mode B not implemented — ignoring"` when
  a Mode A (already-serving) stream received an unexpected AssignMsg.
  Mode B *is* implemented, in `run_available_loop`; the stub message
  was misleading. Now logs a descriptive warning calling out that
  the router shouldn't target Mode A streams with AssignMsg.
- **Three stale ROADMAP entries marked shipped.** GT5, GT6 (dynamic
  rebalancing / drain-then-reassign — ADR-0011 §Phase B2), and GT7
  (QUIC transport — ADR-0010) all moved from `Not started` →
  `✅ Shipped` with code pointers and test references.
- **Three integration tests un-bit-rotted.**
  `tests/test_grid_mode_b.rs`, `tests/test_grid_replication.rs`,
  `tests/test_grid_drain_reassign.rs` had been broken since
  ADR-0018 (MoE expert routing) widened `try_assign_gap` to take
  `expert_start` / `expert_end` and moved `GridServiceImpl` to
  `larql_router::grid::service`. Patched all three (new
  signatures + import paths + `parking_lot::RwLock` for `GridState`
  to mirror the router's 2026-05-16 lock primitive swap). 9 tests in
  3 files all pass.
- **`parking_lot` added as a server dev-dependency.** Mirrors the
  router's `GridState` lock primitive so test fixtures can construct
  an `Arc<parking_lot::RwLock<GridState>>` directly.

### Known follow-up

- **GT5 hash-verification mismatch (P1).** `vindex_identity_hash`
  emits a 16-hex model-identity tag, but `shard_loader` expects a
  SHA-256 content hash on `AssignMsg.shard_hash`. Today deployments
  must pass an empty/placeholder hash so the verification is
  skipped. Real content hashing wants a new optional
  `shard_content_sha256` field on `AnnounceMsg` distinct from
  `vindex_hash`. See `ROADMAP.md` G-MODEB §GT5 "Known follow-up".

## [2026-05-10] — REV: code review

Migrated from `ROADMAP.md` on 2026-08-04. Kept in full: the lens it applies —
per `THESIS.md`, legibility is a primary feature, so a vLLM/SGLang/TGI engineer
should be able to read this and copy ideas out — is the reusable part, and the
findings record diagnoses that exist nowhere else.

Lens: per `THESIS.md`, legibility is a primary feature — a vLLM/SGLang/TGI
engineer should be able to read this and copy ideas out. Findings below are
prioritised by what would damage that primary goal if a stranger landed in
the file today. Severity tags: **P0** = correctness/security ship-blocker,
**P1** = structural/legibility fixes that affect "reads like a citation",
**P2** = defensive polish.

### REV1. Panic on NaN in gRPC sort *(P0)*

**Status**: ✅ **Shipped 2026-05-10.**

Both sort sites in `src/grpc.rs` (the `describe` handler at line ~311 and the
`select` handler at line ~432) used `partial_cmp(...).unwrap()`, which
panics on NaN. Replaced with a shared `cmp_score_desc(a, b)` helper that
returns `Ordering::Equal` on NaN, removing the panic path. New
`#[cfg(test)] mod tests` in `grpc.rs` covers: descending order, NaN→Equal,
sort-with-NaN no-panic + length-preservation, sort-with-infinities
ordering, and all-finite descending sort. Five new unit tests, all green;
`cargo test -p larql-server --test test_grpc` (28 integration tests) still
green.

**Files touched**: `src/grpc.rs` (one new helper, two call sites, one new
test module).

**Note on the strict-weak-ordering trade-off**: NaN-as-Equal violates strict
weak ordering, so `sort_by` cannot guarantee that finite values around
NaN are globally descending — only that the call doesn't panic and that
finite/NaN counts are preserved. Acceptable for a defensive change against
upstream data corruption; downstream `truncate(limit)` still picks
finite values when present, which is the property production cared about.

### REV2. Non-constant-time API key comparison *(P0 security)*

**Status**: ✅ **Shipped 2026-05-10.**

`auth.rs` now hashes both the provided token and the configured key with
SHA-256 and compares the 32-byte digests via `subtle::ConstantTimeEq`.
The hash step removes the length-leak; the constant-time compare removes
the bytewise short-circuit leak. Module-level doc comment names the
threat model so the next reader doesn't accidentally regress to `==`.

**Files touched**:
- `Cargo.toml`: added `subtle = "2.6"` as a direct dep (already in the
  lockfile via rustls — declaring it directly makes the threat-model
  rationale visible at the crate boundary).
- `src/auth.rs`: rewritten with module docs, a private
  `tokens_match(provided, expected) -> bool` helper, and a
  `#[cfg(test)] mod tests` covering equal/unequal/empty/length-different/
  single-byte-difference cases (6 unit tests).

**Verification**:
- 6 new auth unit tests pass.
- All 6 existing `http_auth_*` integration tests in `test_http_core.rs`
  still pass (no behavioural regression).
- `cargo clippy -p larql-server --lib --no-deps -- -D warnings` clean.
- `cargo fmt -p larql-server -- --check` clean.

### REV3. `blocking_read` on tokio RwLock inside async path *(P0)*

**Status**: ✅ **Shipped 2026-05-10.**

`apply_patch` is now structured fast-path / slow-path. The fast path
(session already exists) takes one write guard, mutates, returns. The
slow path drops the sessions write guard, awaits `model.patched.read()`
to get the base, then re-acquires the sessions write guard and uses
`entry().or_insert_with(...)` to absorb the race where another task
inserted the same `session_id` between our drop and re-acquire. No
`blocking_read`/`blocking_write` is reachable from an `async fn` in
`session.rs` anymore (the remaining `sessions_blocking_write` helper is
explicitly intended for `spawn_blocking` callers, documented as such).

A module-level lock-discipline doc-block on `apply_patch` names the
hazard so the next reader doesn't reintroduce it.

**Files touched**:
- `src/session.rs`: `apply_patch` rewritten with fast/slow split + doc
  block.
- `tests/test_http_session.rs`: existing tests' workaround
  (`sm.get_or_create(...)` pre-create dance) removed since the slow path
  is now safe; two new regression tests added:
  - `apply_patch_slow_path_makes_progress_with_held_patched_reader` —
    spawns a task holding `model.patched.read()`, asserts the slow-path
    apply_patch finishes within 5s.
  - `concurrent_apply_patch_same_session_finishes` — 16-way
    `apply_patch("contended", …)` from spawned tasks, asserts all
    finish within 5s and the final patch list has 16 entries.

**Verification**:
- 7/7 session integration tests pass (5 existing, post-cleanup, plus 2
  new).
- 232/232 lib tests pass.
- `cargo clippy -p larql-server --lib --no-deps -- -D warnings` clean.
- `cargo fmt -p larql-server -- --check` clean.

**Note**: `get_or_create` (line ~56) also nests `sessions.write().await`
→ `model.patched.read().await`, but uses async-aware `read().await`
(not `blocking_read`), so it doesn't block a worker. The lock-ordering
hazard (deadlock if anywhere acquires `patched.write` then
`sessions.*`) is not reachable today — the only `patched.write()` call
sites in `routes/patches.rs` don't touch `sessions`. Documented in the
new doc-block on `apply_patch` so future contributors keep the order
consistent.

### REV4. OpenAI error envelope diverges from spec *(P0 — breaks OpenAI SDKs)*

**Status**: ✅ **Edits applied 2026-05-10; verification pending stable workspace build.**

Two-envelope split, per the original recommendation:

- **LARQL paradigm endpoints** keep the flat `{"error": "msg"}` shape via
  the existing `crate::error::ServerError` (unchanged).
- **OpenAI-compat endpoints** (`/v1/embeddings`, `/v1/completions`,
  `/v1/chat/completions`) now use a new `OpenAIError` type that renders
  the canonical nested envelope:
  ```json
  {"error": {"message": "...", "type": "invalid_request_error",
             "param": null, "code": null}}
  ```
  with the OpenAI-canonical `type` strings (`invalid_request_error`,
  `not_found_error`, `service_unavailable_error`, `server_error`).
  `param` and `code` are always emitted (possibly `null`) since some SDKs
  hard-key on those fields.

**Files touched**:
- New `src/routes/openai/error.rs` — `OpenAIError`, `OpenAIErrorBody`,
  `OpenAIErrorPayload`, constructor helpers
  (`invalid_request`/`not_found`/`service_unavailable`/`server_error`),
  `From<ServerError>` impl, `IntoResponse`, 7 unit tests covering all
  status classes, the nested-envelope shape, the From mapping, and the
  always-present `param`/`code` keys.
- `src/routes/openai/mod.rs` — registers the module + re-exports
  `OpenAIError`.
- `src/routes/openai/chat.rs` — entry-point return type
  `Result<Response, ServerError>` → `Result<Response, OpenAIError>`.
  All 8 direct-`return Err(ServerError::X(...))` sites in the entry
  handler converted to `OpenAIError::invalid_request(...)` or
  `service_unavailable(...)`. Internal helpers
  (`run_chat_generation`, `resolve_tools`,
  `schema_for_response_format`) keep `ServerError`; their results
  propagate via `?` through `From<ServerError> for OpenAIError`.
- `src/routes/openai/completions.rs` — same pattern: 5 entry-point
  error sites converted; internal helpers unchanged.
- `src/routes/openai/embeddings.rs` — same pattern: 3 entry-point sites
  converted.
- `src/openapi.rs` — registers `OpenAIErrorBody`/`OpenAIErrorPayload`
  in the `components(schemas(...))` block; the three OpenAI handlers'
  utoipa annotations now reference `OpenAIErrorBody` for 400/500
  responses (LARQL endpoints keep `ErrorBody`).
- `docs/server-spec.md` §8.3.1 rewritten to document the two-envelope
  split with the canonical `type` table.
- `tests/test_http_embed.rs` — added 6 integration tests
  (`http_openai_embeddings_400_uses_nested_envelope`,
  `_empty_uses_nested_envelope`,
  `http_openai_completions_400_uses_nested_envelope`,
  `_503_uses_nested_envelope`,
  `http_openai_chat_completions_400_uses_nested_envelope`,
  `_503_uses_nested_envelope`) using a shared
  `assert_openai_error_envelope` helper that asserts the response body
  has the nested shape with the expected `type` and the always-present
  `param`/`code` keys.

**Verification**:
- `cargo test -p larql-server --lib openai::error` — 6/6 passed.
- `cargo test -p larql-server --test test_http_embed _uses_nested_envelope`
  — 6/6 passed (covers the three handlers × 400/503 cases).
- `cargo test -p larql-server --test test_http_embed http_openai`
  — 55/55 passed (no regression in the existing OpenAI surface).
- `cargo test -p larql-server --lib` — 247/247 passed.
- `cargo clippy -p larql-server --lib --no-deps -- -D warnings` clean.
- `cargo fmt -p larql-server -- --check` clean.

### REV5. Tool-call JSON parsing via `find('{')` / `rfind('}')` returns 500 instead of 400 *(P0 legibility)*

**Status**: ✅ **Shipped 2026-05-10.**

`build_tool_call_message` rewritten as a straight-line
`serde_json::from_str(text.trim())`. The `find('{')` / `rfind('}')`
heuristic is gone — it was supposed to absorb model-output drift
(trailing junk, multiple JSON objects, markdown wrappers) but in
practice silently picked the wrong slice and surfaced the failure as
500 Internal at the call site. The new parser fails fast with a clean
`invalid JSON: …` diagnostic, distinguishes "not an object" from
"missing field", and the call site flips from `ServerError::Internal`
→ `OpenAIError::invalid_request` so the client now sees a
**400 invalid_request_error** with a concrete message (and the raw
output included for debugging) instead of a 500 server_error.

The schema FSM in `routes/openai/schema/` is for *constraining*
generation, not validating it post-hoc. Re-parsing with the FSM would
duplicate the Schema-shaped state machine that already runs during
generation. The straight-line `serde_json` parse is the right level
of validation for the post-emit path.

**Files touched**:
- `src/routes/openai/chat.rs::build_tool_call_message` — rewritten;
  added `json_value_kind` helper for clean error messages.
- `src/routes/openai/chat.rs::handle_chat_completions` (line ~377) —
  parse-failure error switched from `ServerError::Internal` to
  `OpenAIError::invalid_request`.
- `src/routes/openai/chat.rs` — added `#[derive(Debug)]` to
  `ChatChoiceMessage`, `ToolCall`, `ToolCallFunction` to support
  `Result::unwrap_err()` in tests.
- `src/routes/openai/chat.rs` `#[cfg(test)] mod tests` — 9 new unit
  tests:
  - happy path
  - tolerates surrounding whitespace
  - handles nested braces in arguments (locks the property the old
    code was trying to preserve)
  - rejects trailing junk with a clean `invalid JSON:` prefix
    (pre-REV5 this was the 500 trap)
  - rejects empty input
  - rejects non-object top-level (with kind reported)
  - rejects missing `name`
  - rejects missing `arguments`
  - rejects invalid JSON

**Verification**:
- `cargo test -p larql-server --lib build_tool_call` — 9/9 passed.
- `cargo test -p larql-server --lib` — 247/247 passed.
- `cargo clippy -p larql-server --lib --no-deps -- -D warnings` clean.
- `cargo fmt -p larql-server -- --check` clean.

**Note on the streaming path** (chat.rs ~line 586): mid-stream
parse-failures still emit an SSE `data: {"error": ...}` chunk via
`error_chunk(...)` and let the stream terminate gracefully. That path
already returned the OpenAI nested error shape in chunks, so no
behaviour change there — the fix here is only on the buffered
non-streaming path.

---

### REV6. Split `bootstrap.rs` (1279 lines) *(P1 legibility)*

**Files**: `src/bootstrap.rs` → new `src/bootstrap/{parsing,loader,warmup}.rs`.

Natural seams:
- `bootstrap/parsing.rs` (~lines 38–141): `parse_ram_bytes`, `parse_layer_range`, `UnitManifest` and friends.
- `bootstrap/loader.rs` (~lines 143–370): `load_single_vindex`, `discover_vindexes`, vindex selection logic.
- `bootstrap/warmup.rs`: the three warmup blocks (walk-ffn, hnsw-units, metal-experts).
- Remaining `bootstrap.rs`: `Cli`, `serve()` orchestration, listener setup, gRPC + announce spawn — should land at ≤ ~500 lines.

**Acceptance**: `serve()` reads top-to-bottom in one screen. Per-file
coverage stays ≥ 90% (project floor). No public API change.

### REV7. Split `routes/walk_ffn.rs` (1347 lines) *(P1 legibility)*

**Files**: `src/routes/walk_ffn.rs` → new `src/routes/walk_ffn/binary.rs`.

The binary codec (`decode_binary_request`, `encode_binary_output*`,
~lines 1–170 + helpers, ~240 LOC total) is orthogonal to the compute
core and HTTP handler logic. Move it out. Move the in-file `#[cfg(test)]`
block (~230 lines) into a sibling test module. Core compute + handlers
stay together (~600 LOC).

**Acceptance**: `walk_ffn.rs` ≤ ~700 LOC; binary codec testable in
isolation; existing integration tests unchanged.

### REV8. Split / de-duplicate `routes/openai/chat.rs` (1214 lines) *(P1 legibility)*

**Files**: `src/routes/openai/chat.rs`.

Streaming and non-streaming handlers duplicate setup (tokenization, model
lock, schema/sampling config). Extract a `ChatPrep` (or similar) struct so
each handler body is short and the difference between the two paths is
obvious at a glance. Tool-call/tools rendering and tool-call parsing
(~lines 822–998) belong in their own submodule.

**Acceptance**: `chat.rs` ≤ ~700 LOC; streaming and non-streaming entry
points fit on one screen each.

### REV9. gRPC error mapping is uniformly `Status::internal` *(P1)*

**Files**: `src/grpc.rs:99,115,134,157,175,194` (and similar `.map_err`
sites in `grpc_expert.rs`).

Every blocking-task failure coerces to `Status::internal`. Tokenization,
validation, and real internal failures collapse to one code; clients can't
distinguish recoverable from non-recoverable. Map at least
`ServerError::BadRequest`/`NotFound` → `invalid_argument`/`not_found`.

**Acceptance**: a single `From<ServerError> for tonic::Status` impl that
preserves status class. Test asserting a malformed `DescribeRequest`
returns `Code::InvalidArgument`, not `Code::Internal`.

### REV10. Single-vs-multi-model handler duplication *(P1 legibility)*

**Files**: `src/routes/describe.rs:279-314` (and the same shape in
`walk.rs`, `select.rs`, `infer.rs`, `relations.rs`, `patches.rs`).

Single-model and `/v1/{model_id}/...` handlers are copy/paste with one
line difference (`model_or_err(None)` vs `model_or_err(Some(&model_id))`).
A porter has to mentally diff every pair. Consolidate behind one handler
that takes `Option<&str>`, or behind a small macro.

**Acceptance**: each route has one handler body; the router still wires
both paths; tests cover both.

### REV11. Streaming client-disconnect leak (chat / completions) *(P1)*

**Files**: `src/routes/openai/chat.rs:517-520`,
`src/routes/openai/completions.rs:243`.

When the SSE channel closes, the on-token callback flips `early_stop` and
returns, but the `spawn_blocking` generation task continues to natural EOS
— burns a CPU-bound worker on a gone client. The expert layer-batch route
already models the right pattern (semaphore permit held across spawn,
released on cancellation); port it to the OpenAI streaming handlers.

**Acceptance**: a test that opens an SSE stream, drops the receiver, and
asserts the spawn_blocking handle finishes within a small bounded window
of the disconnect rather than running to completion.

---

### REV12. Float validation gaps (NaN/Inf) *(P2)*

**Files**: `src/routes/describe.rs:46` (`min_score`),
`src/routes/walk_ffn.rs` (residual deserialisation),
`src/routes/openai/util.rs:113-145` (`temperature`/`top_p` silent clamp).

Incoming `f32`s are not checked for finitude; OpenAI sampler params are
silently clamped rather than rejected. Reject NaN/Inf with 400; reject
out-of-range with a typed message instead of clamping silently.

### REV13. `max_tokens` upper bound unenforced *(P2)*

**Files**: `src/routes/openai/{chat,completions}.rs` request structs.

Raw `usize` is accepted; allocates token buffers from arbitrary client
input. Add a per-model cap (config-derivable from `LoadedModel`) and
return 400 above it.

### REV14. Cache key truncates float to u32 *(P2)*

**Files**: `src/cache.rs:34`.

`format!("{:x}", min_score as u32)` collides `5.2` and `5.7`. Either
encode the full f32 (e.g. `min_score.to_bits()`) or document the
intentional bucketing.

### REV15. Tests use bare `.unwrap()` on RPC results *(P2)*

**Files**: `tests/test_grpc.rs` (17 instances at lines 34, 43, 51, 68,
92, …).

When the RPC itself errors, `.unwrap()` panics on the wrong line and
hides which assertion was being checked. Replace with
`.expect("describe should succeed")` (or similar).

### REV16. `#[allow(dead_code)]` on library APIs *(P2)*

**Files**: `src/session.rs:55` (`get_or_create`), `:166` (`session_count`);
`src/ratelimit.rs:82` (`evict_stale`); `src/ffn_l2_cache.rs:92` (`stats`);
`src/error.rs:24` (`InferenceUnavailable`).

Either delete or add a one-line doc comment saying they're public for
out-of-tree consumers; right now a reader can't tell intent.

---

### REV-COVERAGE. Test coverage gaps vs 90%/file project floor *(P1, partly addressed)*

**Status**: 🟡 **Tooling + policy shipped 2026-05-10; per-file gap-fill ongoing.**

Done in this session:
- `make larql-server-{test,fmt-check,lint,coverage,coverage-summary,coverage-html,coverage-policy,ci}`
  added to the workspace Makefile (mirror the `larql-compute` /
  `larql-vindex` patterns).
- `crates/larql-server/coverage-policy.json` created. Default per-
  file floor is **90.0%**; 28 debt baselines snapshotted from the
  2026-05-10 measurement; `total_line_min_percent: 65.6` matches the
  current floor. New / split files automatically inherit the 90%
  default.
- Real coverage measured: 65.68% line / 72.18% function (was
  claiming 74.2% / 81.2% at 2026-04-26 — drift since then).
- Mainline files added in this session land at 100% (`routes/openai/error.rs`)
  or close to it (`auth.rs` 98.0%, `session.rs` 96.1%).

Still open (per-file gap-fill, listed roughly by impact):
- `routes/openai/completions.rs` 40.3% → 90% (REV8 split helps)
- `routes/walk_ffn.rs` 49.0% → 90% (REV7 split helps)
- `routes/openai/chat.rs` 53.4% → 90% (REV8 split helps)
- `routes/stream.rs` 53.3% → 90% (Q1.10 split helps)
- `routes/explain.rs` 44.8% → 90%
- `routes/infer.rs` 50.2% → 90%
- `routes/expert/{batch_legacy,multi_layer_batch,single,warmup}.rs`
  all 0% — these need a live-grid harness and are best treated as
  one ticket once the grid test fixture exists.
- `routes/openai/schema/mask.rs` 0% — orthogonal to the splits;
  needs unit tests on the FSM-mask behaviour.
- Behavioural-test gaps still open from the original review:
  malformed-JSON rejection (axum `JsonRejection` → 400),
  body-size-limit 413s, ETag 304 paths beyond the one
  `test_http_describe.rs` happy path, gRPC stream backpressure,
  gRPC client cancel mid-stream, OpenAI SSE `[DONE]` framing,
  OpenAI streaming error chunk shape.

**Acceptance**: each file ≥ 90% line coverage (project floor) — track
via `coverage-policy.json` ratchet; each behavioural gap above has at
least one direct test.

### REV-SPEC. Spec / OpenAPI drift *(P1)*

**Files**: `src/openapi.rs:113-118` vs `proto/vindex.proto:194-196`
(`LoadedCapabilities` differs between HTTP and gRPC); `proto/vindex.proto`
populates both `predictions` and `walk_predictions`/`dense_predictions`
in `InferResponse` regardless of mode (HTTP differentiates). Both are
intentional but undocumented and untested across both transports.
Separately, `docs/server-spec.md` does not mention 422 anywhere and
handlers do not emit it — pick a stance (probably 400 only) and state it.

**Acceptance**: a cross-transport contract test that exercises the same
operation over HTTP and gRPC and asserts equivalent semantics. A
paragraph in `docs/server-spec.md` documenting the shape differences
that remain.

---

### Strengths to preserve (do not regress)

- `ServerError` enum + `IntoResponse` discipline — no generic `Internal`
  catch-all in handler paths.
- Centralised `env_flags` with cached reads and README cross-reference —
  the right pattern for a reference implementation.
- Rate limiter degrades open on poisoned mutex; `X-Forwarded-For` only
  honoured under explicit flag.
- Expert `layer_batch` semaphore-as-backpressure — a clean illustration
  of the right way to bound rayon under HTTP load. Treat as a teaching
  artefact and consider citing in the README.
- Sampling clamping + stop-string handling in `routes/openai/util.rs` is
  well-tested and idiomatic.
- Binary wire format with `Accept`-header negotiation in `walk_ffn` is
  elegant; the f16/i8/f32 negotiation is the kind of concrete pattern the
  THESIS expects to diffuse into other stacks.

## [2026-05-10] — Code-review P0 sweep + coverage scaffolding

Five P0 fixes from the in-tree code review (REV1–REV5 in `ROADMAP.md`)
plus the missing larql-server Makefile coverage targets and a per-file
90% coverage policy.

### Fixed

- **REV1 — gRPC sort panics on NaN scores.** `grpc_describe` and
  `grpc_select` used `partial_cmp(...).unwrap()`, which panics on NaN.
  Replaced both call sites with a shared `cmp_score_desc(a, b)` helper
  that maps NaN → `Ordering::Equal`. A corrupted vindex or a future
  patched-scoring path that produces NaN no longer takes a gRPC worker
  down. Five new unit tests in `grpc.rs` lock the property.
- **REV2 — Non-constant-time API key comparison.** `auth.rs` used
  `==` on `&str`, which short-circuits and leaks bytewise progress
  through request timing. Tokens are now SHA-256-hashed and the digests
  compared via `subtle::ConstantTimeEq`. Module-level doc block names
  the threat model. `subtle` (already in the lockfile via rustls)
  added as a direct dep. Six new unit tests in `auth.rs`; six existing
  `http_auth_*` integration tests still pass with no behavioural
  change.
- **REV3 — `blocking_read` on tokio RwLock inside async path.**
  `SessionManager::apply_patch` previously called
  `model.patched.blocking_read()` while holding `sessions.write().await`
  on a worker thread, which on a multi-thread runtime stalls the
  worker (and risks deadlock against any task acquiring those locks
  in the opposite order). Restructured into fast-path / slow-path:
  the slow path drops the sessions write guard, awaits
  `model.patched.read()`, then re-acquires and uses
  `entry().or_insert_with(...)` to absorb the race where another task
  inserted the same `session_id`. No `blocking_read`/`blocking_write`
  on tokio locks is reachable from an `async fn` in `session.rs`
  anymore. Two new regression tests assert (a) forward progress when
  another task holds a `patched.read()` and (b) 16-way concurrent
  `apply_patch` on the same `session_id` finishes within a bounded
  deadline.
- **REV4 — OpenAI error envelope diverged from spec.** Non-streaming
  responses on `/v1/embeddings`, `/v1/completions`, and
  `/v1/chat/completions` returned `{"error": "msg"}` (flat); the OpenAI
  Python and JS SDKs expect
  `{"error": {"message", "type", "param", "code"}}` (nested) and broke
  on field access against the flat shape. Streaming SSE error chunks
  already used the nested form, so non-stream and stream errors were
  inconsistent. Introduced a new `OpenAIError` type with constructor
  helpers (`invalid_request`, `not_found`, `service_unavailable`,
  `server_error`) and an `IntoResponse` that renders the canonical
  nested envelope with `param`/`code` always present (possibly null).
  `From<ServerError>` lets internal helpers keep `ServerError` and
  propagate via `?`. The three OpenAI handler entry-point return
  types flipped to `Result<_, OpenAIError>` and 16 direct
  `return Err(ServerError::X(...))` sites converted to the matching
  `OpenAIError::Y(...)` constructor. LARQL paradigm endpoints keep the
  flat envelope. Six integration tests assert the nested shape on
  400/503 paths across the three handlers; seven unit tests cover the
  type itself.
- **REV5 — tool-call JSON parser surfaced 500 instead of 400 on
  malformed nested-brace output.** `build_tool_call_message` used
  `find('{')` + `rfind('}')` to extract JSON from constrained-decoder
  output, which silently picked the wrong slice on trailing junk /
  multiple objects / markdown wrappers and surfaced the parse failure
  as `ServerError::Internal` (500). Rewrote as a straight-line
  `serde_json::from_str(text.trim())` with structured diagnostics
  (`invalid JSON: …`, `tool output must be a JSON object`, missing-
  field reports), and flipped the call-site error class from
  `Internal` to `OpenAIError::invalid_request` so the client now sees
  **400 invalid_request_error** with a concrete message. Nine new
  unit tests cover happy path, surrounding whitespace, nested-brace
  arguments, trailing junk, empty/whitespace, non-object top-level,
  missing `name`/`arguments`, and invalid JSON.

### Added

- **Two-envelope error documentation.** `docs/server-spec.md §8.3.1`
  rewritten with the LARQL-flat / OpenAI-nested split and a canonical
  `type` table. README `Error Codes` section updated to match.
- **Makefile coverage targets** for larql-server, mirroring the
  larql-compute / larql-vindex pattern:
  `larql-server-test`, `larql-server-fmt-check`, `larql-server-lint`,
  `larql-server-coverage`, `larql-server-coverage-summary`,
  `larql-server-coverage-html`, `larql-server-coverage-policy`,
  `larql-server-ci`. Threshold variables: `LARQL_SERVER_COVERAGE_MIN`
  (default 65 — current baseline), `LARQL_SERVER_COVERAGE_POLICY`,
  `LARQL_SERVER_COVERAGE_REPORT`.
- **`coverage-policy.json`** with default 90% line floor, 28 per-file
  debt baselines snapshotted from the 2026-05-10 measurement, and the
  total floor at the measured 65.6% baseline. Policy semantics
  ratchet upward only — new / split files automatically inherit the
  90% default.

### Internal

- Cleared 5 pre-existing clippy errors in lib (`bootstrap.rs:230`
  boolean simplification, `metrics.rs:64` missing `Default` for
  `LayerLatencyTracker`, `walk_ffn.rs` doc indentation + needless
  lifetimes + redundant closure). `cargo clippy -p larql-server --lib
  --no-deps -- -D warnings` now clean.
- Updated `tests/test_expert_endpoint.rs` import: `cpu_moe_forward`
  and `MoeLayerWeights` moved from `larql_inference` to
  `larql_compute` in the upstream refactor; the test had a stale
  import that blocked `--tests` builds. Pure plumbing — matches the
  cargo error hint.
- Added `#[derive(Debug)]` to `ChatChoiceMessage`, `ToolCall`,
  `ToolCallFunction` to support `Result::unwrap_err()` in the new
  `build_tool_call_message` tests.

### Coverage snapshot (2026-05-10)

- **TOTAL**: 65.68% line / 72.18% function / 64.90% region.
- **At-or-above 90% default**: `routes/openai/error.rs` (100%),
  `routes/openai/util.rs` (99.6%), `routes/openai/embeddings.rs`
  (93.2%), `session.rs` (96.1%), `state.rs` (85.8% — debt baseline),
  `auth.rs` (98.0%), `wire.rs` (96.9%), `etag.rs` (100%), and 16
  others.
- **Largest debt items** (all carry baselines, must ratchet up):
  `routes/expert/{batch_legacy,multi_layer_batch,single,warmup}.rs`
  at 0% (need a live grid harness),
  `routes/openai/schema/mask.rs` at 0%, `bootstrap.rs` at 29.7%,
  `routes/openai/completions.rs` at 40.3%, `routes/walk_ffn.rs` at
  49.0%, `routes/openai/chat.rs` at 53.4%.

## Completed milestones (migrated section)

Migrated from `ROADMAP.md` on 2026-08-04. Predates the dated-entry convention
above and carries its own internal dates; kept because it is the only record of
the 2026-04/05 shipping sequence. `THESIS.md` links here for the
cos-similarity / tok/s / RSS / latency numbers.

### 2026-05-02 — F0 closed + N0 slices 1 + 2 (OpenAI compat: models + embeddings + completions + chat completions)

**F0 closed.** `larql run output/gemma4-26b-a4b-q4k.vindex "The capital
of France is" --max-tokens 5` (no `--moe-shards`, no `--metal`) returns
**"Paris."** Local in-process CPU MoE on the per-layer Q4_K hybrid-MoE
vindex now produces the correct answer; the M-CPU kernel work shared
the code path with the 2026-04-30 server-side fix, so the local route
inherited correctness for free. Marked closed under P0 Active.

**N0 slice 1 + slice 2** — four OpenAI-compatible endpoints landed
end-to-end on `larql-server`, live-validated against
`output/gemma3-4b-q4k-streaming.vindex`:

| Endpoint | Slice | Notes |
|---|---|---|
| `GET /v1/models` | 1 | OpenAI `{object: "list", data: [{id, object: "model", created, owned_by: "larql", ...}]}`. Larql-specific extras (`path`, `features`, `loaded`) preserved. |
| `POST /v1/embeddings` | 1 | All four `input` variants (`string`, `string[]`, `int[]`, `int[][]`). Mean-pooled static-embedding lookup. `encoding_format: "base64"` returns 400 (follow-up). |
| `POST /v1/completions` | 1 | Non-streaming; un-KV-cached generation loop. `stream=true` and `n>1` return 400. |
| `POST /v1/chat/completions` | 2 | Multi-turn chat with chat-template auto-detection (Gemma / Llama / ChatML / Mistral / Plain) from `arch.family()`. Same generation path as `/v1/completions`. `tools` / `tool_choice` / `response_format: json_*` / `stream=true` / `n>1` return 400 with clear messages. |

Implementation surface: ~1600 LOC across three new files
(`src/routes/openai_embeddings.rs`, `src/routes/openai_completions.rs`,
`src/routes/openai_chat.rs`) + reshape of `src/routes/models.rs` + 4
routes wired into both single-model and multi-model routers + 23 unit
tests + 19 integration tests + new live `examples/openai_demo.rs`
walkthrough that boots the server in-process via
`tower::ServiceExt::oneshot` and exercises every endpoint.

Live smoke (`gemma3-4b-q4k-streaming.vindex`, port 18081):
- `/v1/models` → OpenAI shape with `gemma-3-4b-it`, `created`, `owned_by`, larql extras.
- `/v1/embeddings input="France"` → 2560-dim pooled vector + correct usage block.
- `/v1/completions max_tokens=5` → wire-correct response (`cmpl-...`,
  `text_completion`, `usage`).
- `/v1/chat/completions max_tokens=8` with system + user → wire-correct
  response (`chatcmpl-...`, `chat.completion`, `choices[0].message.{role:
  "assistant", content}`, `usage`). Output content quality on the
  un-KV-cached path is poor (degenerate greedy on un-trained
  base-decode-without-template); wire is what's verified here.

**Tests** — full sweep:
- `cargo test -p larql-server --lib`: 154 lib tests
- 14 integration files: 392 integration tests
- Total: ~546 tests, 0 failures
- `cargo clippy -p larql-server --tests --no-deps -- -D warnings`: clean
- `cargo fmt -p larql-server -- --check`: clean

**Open follow-ups** (per-item in N0 sub-headers above):
- **Slice 3 (N0.1 SSE)** — `text/event-stream` for both
  `/v1/completions` and `/v1/chat/completions`. Bundles with Q1.10
  (stream.rs reduction) since both touch the same streaming
  state-machine shape.
- **Slice 4 (N0.6)** — constrained decoding for `tools` / `tool_choice`
  / `response_format: json_schema` via JSON schema → GBNF mask.
- **Slice 5 (N0.3)** — `/v1/responses` Responses API, pairs with N1
  stateful sessions.
- **N0.2-fast (shipped 2026-05-02)** — KV-cached generation path now
  live for both `/v1/completions` and `/v1/chat/completions`.
  `LoadedModel.weights` migrated from `OnceLock<ModelWeights>` to
  `OnceLock<RwLock<ModelWeights>>`; OpenAI handlers acquire a write
  guard via `lock_weights_for_gen()` and call
  `larql_inference::layer_graph::generate{,_streaming}` which auto-
  dispatches f16 vindexes to the fused KV-cached path and Q4_K +
  CPU vindexes to the per-step `predict_q4k` fallback. Output on
  Gemma 3 4B: "The capital of France is" → " Paris.\n\nParis is"
  (was " is is is is" pre-fix). Multi-turn chat template rendering
  moved into `larql_inference::prompt::ChatTemplate::render_messages`,
  shrinking the openai handlers further. `bootstrap.rs` now mirrors
  `larql_inference::open_inference_vindex` by loading
  `attn_weights_q4k.bin` + `interleaved_q4k.bin` for inference-capable
  vindexes (without these the Q4_K decode panics).
- **base64 encoding** for `/v1/embeddings` — small follow-up.
- **N0-router** — OpenAI surface on `larql-router` (grid front);
  tracked under "Router-side OpenAI surface" in P1.

### 2026-05-01 (continued) — Q1 code-quality cleanup (9 of 10 items)

The Q1 audit catalogue from earlier the same day, executed in a follow-on
session. All public APIs preserved; existing test surface unchanged.
Q1.10 (stream.rs WebSocket state machine) deferred until N0.1 (OpenAI
Chat Completions SSE) forces a similar shape.

| Item | Outcome |
|---|---|
| **Q1.1** Split `routes/expert.rs` (1044 LOC, 6 concerns) | New `routes/expert/{mod,single,batch_legacy,layer_batch,cpu,metal,warmup}.rs` directory. mod.rs (90 LOC) re-exports the historical public surface (`run_expert`, `run_experts_cpu_batch`, `run_experts_metal_batch`, `warmup_*`, `handle_*`); each sibling file is ~100-225 LOC with one clear concern. `metal.rs` is `#[cfg(all(feature = "metal-experts", target_os = "macos"))]`-gated so non-Metal builds compile clean. |
| **Q1.2** Centralise env-var flags into `src/env_flags.rs` | New module with one `pub const` per `LARQL_*` name + cached presence accessors backed by `std::sync::OnceLock` (process-wide, not TLS — env vars don't change at runtime). Replaced 12 raw `std::env::var(...)` call sites in `routes/expert/*` and `grpc_expert.rs`; removed two ad-hoc `thread_local! { static HTTP_TIMING ... }` blocks. README env-var table now references the same names that show up in `env_flags::*`. |
| **Q1.3 + Q1.9** Shared `wire::has_content_type` | New `src/wire.rs` with `has_content_type(headers, expected) -> bool` (uses `contains` so parameterised types like `application/json; charset=utf-8` match). Replaced 4 inline header-detection patterns in `routes/walk_ffn.rs`, `routes/embed.rs` (×2), `routes/expert/batch_legacy.rs`. 4 unit tests cover exact-match, parameterised, mismatch, and missing-header cases. |
| **Q1.4** Body-size limit constants | `REQUEST_BODY_LIMIT_BYTES = 64 MB` and `REQUEST_BODY_LIMIT_LARGE_BYTES = 256 MB` in `src/http.rs`. Replaced 3 bare literals; `EXPERT_BATCH_BODY_LIMIT` in `routes/mod.rs` now references the same const. |
| **Q1.5** `JSON_CONTENT_TYPE` const | Added to `src/http.rs` next to `BINARY_FFN_CONTENT_TYPE`. Replaced 3 bare `"application/json"` literals across walk_ffn / embed / expert. |
| **Q1.6** Typed `DEFAULT_*` consts | `DEFAULT_PORT`, `DEFAULT_HOST`, `DEFAULT_HNSW_EF_SEARCH`, `DEFAULT_MAX_CONCURRENT`, `DEFAULT_DESCRIBE_CACHE_TTL_SECS`, `DEFAULT_LOG_LEVEL`, `DEFAULT_SESSION_TTL_SECS`, etc. Moved into `bootstrap.rs` (alongside the new `Cli` struct from Q1.8); `clap` now uses `default_value_t = ...`. `SessionManager::new` references the same `DEFAULT_SESSION_TTL_SECS` instead of re-encoding `3600`. |
| **Q1.7** `announce.rs` reconnect/heartbeat consts | `RECONNECT_INITIAL_BACKOFF` / `RECONNECT_MAX_BACKOFF` / `HEARTBEAT_INTERVAL` lifted to module consts; the previous `Duration::from_secs(1) / 60 / 10` magic numbers are gone. |
| **Q1.8** Reduce `main.rs::main` (656 LOC → 26 LOC) | Moved `Cli` struct + `pub async fn serve(cli: Cli)` into `bootstrap.rs`. `main.rs` is now: parse Cli, install tracing, call `bootstrap::serve(cli).await`. Boot orchestration (vindex loading, warmups, listener+TLS+UDS, gRPC, grid announce) is callable from anywhere that wants to drive the server without going through `clap::Parser::parse_from`. |
| **Q1.10** stream.rs reduction | **Deferred** — see P1: Active. Bundling with N0.1 SSE infrastructure when that lands. |
| Tests | 126 → **131 lib tests** (4 new for `wire::has_content_type`, 1 for `env_flags::names_are_larql_prefixed_and_unique`); 37 integration tests unchanged; ~580 tests across lib + integration, 0 failures. |
| Clippy | `cargo clippy -p larql-server --tests --no-deps -- -D warnings` clean. |
| `cargo fmt -p larql-server -- --check` | Clean. |

LOC delta (per-file):

| File | Before | After |
|---|---|---|
| `main.rs` | 656 | **26** |
| `bootstrap.rs` | 464 | 1073 (Cli + serve moved in) |
| `routes/expert.rs` | 1044 | (deleted) |
| `routes/expert/mod.rs` | — | 90 |
| `routes/expert/single.rs` | — | 155 |
| `routes/expert/batch_legacy.rs` | — | 105 |
| `routes/expert/layer_batch.rs` | — | 226 |
| `routes/expert/cpu.rs` | — | 195 |
| `routes/expert/metal.rs` | — | 204 |
| `routes/expert/warmup.rs` | — | 140 |
| `env_flags.rs` (new) | — | 122 |
| `wire.rs` (new) | — | 64 |

The bulk of the `bootstrap.rs` size growth is the Cli struct (~200 LOC of
clap doc-comments + `#[arg]` attributes) and the `serve` function body
that used to live in `main`. The orchestration is unchanged; only its
location moved.

### 2026-05-01 — HTTP CPU-path optimisations + UDS transport + layer-batch wire

End-to-end ~17.7 → ~19.7 tok/s on Gemma 4 26B-A4B (M3 Max, single local
gRPC shard, 100-token poem). Per-call HTTP overhead dropped from ~660 µs
to ~460 µs on gRPC streaming, ~510 µs on UDS, ~660 µs on TCP HTTP (now
with TCP_NODELAY). All optimisations preserve bit-exact semantics
(verified by output equivalence on the same prompts).

| Item | Outcome |
|---|---|
| **`POST /v1/experts/layer-batch`** new endpoint | One residual + K (expert_id, weight) pairs → one router-weighted-sum response. Replaces the K-residual-copies legacy `/v1/expert/batch` for the common-case `forward_moe`. Saves ~2.6 MB/token of redundant wire data + K-1 redundant `pre_experts_norm` + Q8_K quants on the server. |
| **`POST /v1/experts/layer-batch-f16`** new endpoint | f16 variant — halves wire bytes (5.5 KB request + response). Opt-in via `LARQL_MOE_WIRE_F16=1` for LAN deployments. f16 conversion CPU cost (~9 µs/call) cancels the wire saving on loopback; expected +3-5% gain on 1 Gbps Ethernet. |
| **Unix Domain Socket transport** (`--uds-path`, `unix://` URL) | Hand-rolled HTTP/1.1 over `UnixStream` (no new dep). Saves ~150 µs/call on loopback (~3% end-to-end). Persistent stream behind a `Mutex`, lazy reconnect on disconnect. Same wire format as TCP HTTP, so f16 + layer-batch semantics carry through unchanged. |
| **TCP_NODELAY on accepted connections** | `axum::serve::ListenerExt::tap_io` hook calls `set_nodelay(true)` per accept. Defensive against tail-packet stalls (40-200 ms on Linux/BSD delayed ACK) on real LAN; within noise on loopback. |
| **gRPC SPLIT default-on for gRPC shards** | Streaming fire/collect overlap now default for `grpc://` shards. Reliably ~12% steady-state win on M3 Max loopback (re-measured 19.5 vs 17.7 tok/s, alternating-cooled). The historical "20 → 4 tok/s catastrophic regression" warning predates the Metal MoE accuracy fix and the predispatch refactor; under thermal pressure both unary + SPLIT regress similarly, but stable-state SPLIT wins. Set `LARQL_MOE_NO_SPLIT=1` to opt out. |
| Per-call timing instrumentation | `LARQL_HTTP_TIMING=1` (server: decode / spawn_overhead / compute / encode µs; client: encode / send_total / recv_body / decode µs). `LARQL_MOE_TIMING=1` (per-token: per-layer route+fire / collect / server compute estimate / network estimate). Used for the diagnostic round that found `__powisf2` libcall in the f16 decode hot path (now bit-manipulated). |
| Test suite restored | 7+ test files had `LoadedModel { ... }` literals missing the `unit_filter` field added recently — all 9 LoadedModel literal sites in tests/ + tests/common/ patched. Test count went from 119 lib-only (broken integration tests) to **494 total across lib + 14 integration test files, all green**. |
| README + docs updated | `README.md` rewrite: new headline mentioning MoE grid as first-class use case, full env-var reference table, refreshed CLI Options with `--uds-path`/`--units`, rewritten "Remote MoE shard topology" recipe with current numbers, new `/v1/experts/layer-batch[-f16]` API section, accurate Crate Structure (28 source files vs the 16 the doc previously listed). `docs/server-spec.md`: §4.5 Remote MoE Expert Endpoints added, §13.4 dropped "planned" status, §10.2 fly.io references `F-FLY`. |
| `bench_expert_server` re-validated | Refreshed numbers in the Live perf snapshot section above. `cpu_moe_forward` floor 0.10 → 0.37 ms (the 0.10 was a buggy measurement on empty buffers — see prior compute ROADMAP). `forward_moe` warm 1.91 → 0.80 ms. 30-layer sweep 56 → 24.8 ms. RSS unchanged at ~10.5 GB. |

Tried-but-reverted (kept in source for future hardware where the trade
may flip):
- `tokio::task::block_in_place` instead of `spawn_blocking` — server-side
  faster (no transition cost) but tokio kept spawning replacement OS
  workers when every request blocked, regressing sweep ~0.3 ms.
- f16 wire as default — within noise on loopback (CPU conversion cancels
  wire saving); kept as opt-in for LAN.

### 2026-05-01 (continued) — larql-server review pass

Same calendar day, separate session. Audit + fixes across the entire
larql-server crate to land a clean baseline alongside the perf work.

| Item | Outcome |
|---|---|
| Test suite restored | 7+ stale `LoadedModel` test fixtures + 1 stale `PatchOp` example fixture missing recently-added struct fields. All 9 LoadedModel literal sites + 1 PatchOp site patched. **Test count went 119 lib-only → 501 across lib + 14 integration files; all green.** |
| `bench_expert_server` extended | New `--uds` and `--wire f32\|f16` flags. Spawns server bound to both TCP and UDS so the bench can A/B per-call cost. Confirmed UDS gives ~10% loopback win (0.82 → 0.74 ms `forward_moe` warm); f16 is a clear LOSS on loopback (1.05 ms — CPU conversion dominates) but expected to win on LAN. |
| README rewrite | Added env-var reference table, `/v1/experts/layer-batch[-f16]` API section, "Remote MoE shard topology" recipe with current numbers, accurate Crate Structure (28 source files vs the 16 the doc previously listed), "What's coming" section pointing to N0..N6 + F-FLY. ~880 → ~1110 LOC. |
| `docs/server-spec.md` updated | §3 CLI flags get `--uds-path` / `--units` / `--warmup-walk-ffn` / env-var section. New §4.5 Remote MoE Expert Endpoints (full layer-batch + f16 + transport coverage). §13.4 dropped "planned" status. §10.2 fly.io references `F-FLY`. |
| ROADMAP additions | New "Great new functionality" section (N0..N6) at the top — N0 is OpenAI API compatibility (chat completions + completions + responses + embeddings + models), highest-leverage item. F-FLY at top of P0: Active. F0 status updated (server path correct, local in-process TBD). Q1 (code-quality review) added at P1 with 10 sub-items targeting modularity + magic literals. |
| `cargo clippy -p larql-server --tests --no-deps -- -D warnings` | Was failing on 6 errors (manual `is_multiple_of`, `let_unit_value`, dead env-var unpacks, `path_used` unused initial assignment). All fixed. Server-only clippy now clean. |
| `cargo fmt -p larql-server -- --check` | Clean. |
| Coverage | 69.24% line / 75.64% function via `cargo llvm-cov`. Slight regression from 74.2/81.2 baseline attributable to new code added without proportional tests; mitigated by adding `topology.rs` tests (3) + `routes/expert.rs` `layer_batch_wire_tests` mod (4). |
| Code-quality findings catalogued | New Q1 section in ROADMAP with 10 concrete items (Q1.1 split `routes/expert.rs` 1049 LOC, Q1.2 centralise env flags into `src/env_flags.rs`, etc.) — all with file:line references and effort estimates. Total ~7-8 hours for the full sweep. |
| README + ROADMAP doublecheck | Fixed `gemma3-4b.vindex` references (file doesn't exist; replaced with `gemma3-4b-v2.vindex` which does), removed stale `ADR-009` reference (no such file), harmonised the two perf reference tables (Examples vs Recommended setups now reference each other), updated stale "2026-04-26" date stamp. |

### 2026-04-26 — Per-expert byte table refactor + `experts_packed.bin` removal

`MoeLayerWeights.experts_{gate_up,down}` migrated from `&[u8]` (monolith +
`expert_idx * stride` arithmetic in the compute path) to `Vec<&[u8]>`
(per-expert slice table). The CPU MoE consumer (`cpu_moe_forward` and
`run_single_expert{,_with_norm}`) now indexes by expert id directly, with
format dispatch (BF16 vs Q4_K) at the cache layer.

| Item | Outcome |
|---|---|
| `larql-compute` | `cpu/ops/moe/{cache,expert,forward,mod}.rs` and `pipeline.rs::MoeLayerWeights`. `cached_dequant(bytes, format, expected_floats)` dispatches BF16/Q4_K. `expert_byte_slice` deleted. Tests updated. 94/94 pass. |
| `larql-vindex` | `cpu/ops/q4_common.rs::dequantize_q4_k` lifted to module scope so the compute crate can dequant Q4_K without a `larql-models` dependency. |
| `larql-inference` | `build_moe_weights` builds per-expert tables from either `weights.get_layer_entry_bytes(...)` (per-layer Q4_K) or BF16 stride slicing (legacy). `QuantFormat` re-exported. |
| `larql-server` | `routes/expert.rs::run_expert` resolves per-expert bytes through whichever path the vindex provides; honours `expert_filter` ownership. `tests/test_expert_endpoint.rs` updated to slice synthetic monoliths into per-expert tables. 4/4 parity tests pass. |
| 26B-A4B vindex | `weight_manifest.json` stripped of `packed_bf16` rows for experts (60 → 421 entries). `experts_packed.bin` deleted (43 GB freed; vindex 58 → 16 GB). |
| Bench parity | `bench_expert_server` re-runs end-to-end against the per-layer-only vindex. `forward_moe` warm latency unchanged at 1.91 ms (was 1.93 ms when monolith was still on disk). 30-layer sweep at 56 ms (cold-page sweep on BF16 monolith was 866 ms). |

`bench_expert_server` and the parity tests both detect the format
automatically (`weights.has_per_layer_ffn()`); legacy BF16 vindexes still work
unchanged. Future MoE vindexes only emit per-layer files — the q4k extractor
at `format/weights/write_q4k/mod.rs` already does this.

### 2026-04-30 — gRPC grid: end-to-end accuracy

The grid produced semantically wrong text on Gemma 4 26B-A4B-it ("The capital
of France is **not specified in the text**…") despite each shard correctly
running its expert FFN. Root cause was on the **client** side
(`larql-inference::layer_graph::grid`) — chat-template handling, detokeniser,
EOS detection, and special-token suppression — not the shard server. The
server work here was confirming the contract: shards return correct expert
outputs given the right top-K input. Documenting for future grid changes.

| Item | Notes |
|------|-------|
| Server shards verified correct | A 2-shard split (experts 0-63 on `:9081`, 64-127 on `:9082`) running against the unit manifest serves expert outputs that, when combined client-side with the proper detokenisation + EOS + special-token suppression + default system prompt, produce "**Paris**" as the answer |
| Shard contract: per-(layer, expert) ownership via `--units` | The `parse_unit_manifest` path is what the client's `--moe-units-manifest` resolves against; ownership is the strict source of truth and `forward_moe_seq` rejects layers/experts not owned by any shard |
| Decode throughput (loopback, M3 Max) | 2.3 tok/s end-to-end on the 26B-A4B with two shards in the same process — expected to climb meaningfully when shards run on separate hosts (less GPU contention with the client) |

### 2026-04-30 — Metal expert dispatch: 3.7× speedup found, blocked on kernel bug

`LARQL_MOE_TIMING=1` showed the grid bottleneck is **server compute = 95%** of token wall time (network = 2%, route+fire = 3%). Per layer: 8.36ms server / 0.18ms net. Each shard runs its 4 picked experts (gate + GELU + down) on CPU-rayon BLAS — that's where the time goes. Sub-arc:

| Item | Notes |
|------|-------|
| Bottleneck localised | CPU experts = 250ms/token (95%) on the loopback 2-shard setup. Network = 5ms (2%). The grid-side overhead is negligible — accelerating the shard's expert math is the only meaningful lever |
| `--features metal-experts` measured: **3.7× speedup** | Server with Metal expert dispatch: 264ms → 117ms per token, 2.3 tok/s → **9.4 tok/s** (preselected path → 11.2 tok/s). Significant — server compute drops from 250ms → 115ms |
| **Accuracy bug blocks shipping** | Metal expert kernel (`MetalBackend::run_experts_preselected_metal` and `_prestaged_metal`, both routes) produces numerically wrong outputs for Gemma 4 26B-A4B-it MoE shape (cos≈0.7 vs CPU, \|metal\|≈70% of \|cpu\|). End-to-end output: "**Paris**" via CPU vs "answer is in the context" via Metal. Same kernels are correct for dense FFN at inter=2560/10240/21504 — bug is specific to MoE inter=704 dispatch |
| Workaround: default to CPU even on metal-experts builds | `run_experts_metal_batch` now early-returns `None` unless `LARQL_USE_METAL_EXPERTS=1` is set. Shipping correctness over speed; the Metal path stays opt-in for kernel-debug runs |
| Diagnostic: `LARQL_METAL_VS_CPU_DEBUG=1` | Server-side per-call A/B compare in `run_experts_metal_batch` — runs both Metal and CPU on the same input, prints max\|Δ\|, \|metal\|, \|cpu\|, cos. Ready to use when someone digs into the kernel |
| See also | `larql-compute/ROADMAP.md` "Open: Metal MoE expert kernel — accuracy bug at inter=704" for the kernel-side investigation plan |

### 2026-04-26 — examples, synthetic benchmark, grid checks

| Item | Outcome |
|---|---|
| `server_demo` | Runs locally with synthetic data; fixed invalid probe-label JSON comma output and updated rate-limit text for `--trust-forwarded-for`. |
| `embed_demo` | Runs locally with synthetic embed/logits/token responses and binary-wire examples. |
| `server_bench --release` | Synthetic benchmark completed: `gate_knn` top-5 0.022 ms/op, 8-layer `walk` 0.203 ms/op, single-layer `walk-ffn` 0.032 ms/op, batched 8-layer `walk-ffn` 0.321 ms/op, describe simulation 0.298 ms/op, 512-token embed prefill 0.114 ms/op. |
| `bench_embed_server` | Example builds under `cargo check -p larql-server --examples`; execution requires a real vindex path. |
| Grid unit coverage | Added `GridState` tests for inclusive ranges, default single-model routing, least-loaded replica selection, deregistration, batched gap reporting, and status gaps. `cargo test -p larql-router` now runs 20 tests. |
| Docs | Updated server README examples/benchmarks/testing, router README validation, and router spec validation commands. |

### 2026-04-26 — coverage round-6 (embed + walk-ffn reachable gaps)

| Item | Outcome |
|---|---|
| `routes/embed.rs` modularity | Extracted binary embed/logits parse helpers and binary embed response encoder |
| `routes/embed.rs` coverage | **66.7% → 86.5% line**, **70.7% → 86.3% function** |
| `routes/walk_ffn.rs` coverage | **76.7% → 79.5% line**, **77.3% → 82.0% function** |
| Tests | 458 → **478** tests |
| Coverage | **71.9% → 74.2% line**, **78.9% → 81.2% function** |

### 2026-04-26 — modularity + coverage round-5

| Item | Outcome |
|---|---|
| Boot/loading modularity | Moved parse/discovery/vindex-load helpers out of `main.rs` into `bootstrap.rs`; binary now keeps CLI orchestration while library code is directly testable |
| `routes/stream.rs` | Extracted pure `stream_describe_messages`; describe stream behavior can be tested without a WebSocket client |
| `routes/infer.rs` | Extracted mode selection and prediction formatting helpers |
| `routes/explain.rs` | Extracted band mapping, probability/gate/attention rounding, prediction formatting, and lens formatting helpers |
| Clippy | Server-local clippy clean with `--no-deps`; full dependency-checking command is blocked by existing `larql-vindex` warnings |
| Coverage | **69.2% → 71.9% line**, **77.1% → 78.9% function** (458 tests) |

### 2026-04-26 — coverage round-4 (T2 reachable gaps)

| Item | Outcome |
|---|---|
| `embed_store.rs` | 25% → **98% line** with tiny f16 mmap fixtures and L1 cache behavior tests |
| `announce.rs` | 6% → **56% line** by extracting/test-covering announce, heartbeat, dropping, and bearer helpers |
| `main.rs` | 0% → **23% line** with binary unit tests for parse/discovery/serve-alias helpers |
| `routes/stream.rs` | 0% → **28% line** with pure WebSocket message shape builders |
| `routes/infer.rs`, `routes/explain.rs` | Default/request deserialization coverage added; full paths remain weight-gated |
| Coverage | 63.9% → **69.2% line**, 73.4% → **77.1% function** (430 → 458 tests) |

### 2026-04-26 — coverage round-3 (T2 partial) + magic strings round-2

| Item | Outcome |
|---|---|
| `test_grpc.rs` — 28 new gRPC handler tests | Direct method calls on `VindexGrpcService` — no network socket; health, stats, describe, walk, select, relations, walk_ffn, infer, stream_describe |
| `grpc.rs` coverage | 0% → **65%** (169 lines uncovered, all gated on real model weights or gRPC streaming) |
| Magic strings — `"probe"` | `PROBE_RELATION_SOURCE` constant in `band_utils.rs`; used in describe.rs, grpc.rs, stream.rs |
| Magic strings — `"ok"` | `HEALTH_STATUS_OK` constant; used in grpc.rs health handler |
| Magic strings — gRPC modes | `INFER_MODE_WALK/DENSE/COMPARE` applied to grpc.rs (was using bare strings) |
| Magic strings — WebSocket types | `WS_TYPE_ERROR/LAYER/DONE/PREDICTION/INFER_DONE` and `WS_CMD_DESCRIBE/INFER` in stream.rs |
| Coverage | 57.2% → **63.3% line**, 65.3% → **73.2% function** (402 → 430 tests) |

### 2026-04-26 — coverage round-2 (T1)

| Item | Outcome |
|---|---|
| `functional_tokenizer()` in common | WordLevel tokenizer (France→0, …) added to test infra; unblocks describe/walk/walk-ffn body paths |
| `test_http_full_routes.rs` | 39 new HTTP integration tests exercising full describe/walk/walk-ffn code paths |
| `test_unit_band_utils.rs` | 13 pure unit tests for `band_utils.rs` constants + helpers |
| Infer + ratelimit branches | `infer_disabled=false` model builder; ratelimit middleware axum tests |
| Coverage | 49.1% → **58.0% line**, 56.4% → **65.3% function** (345 → 402 tests) |

### 2026-04-26 — code quality round-1

| Item | Outcome |
|---|---|
| Modularity — deduplicate `session_id()` | 3 identical private fn definitions → 1 `pub fn extract_session_id` in `session.rs` |
| Modularity — `get_layer_bands()` / `filter_layers_by_band()` | 5 / 3 duplicated blocks → `src/band_utils.rs` |
| Modularity — `model_or_err()` | 25 repeated `ok_or_else(NotFound)` sites → `AppState::model_or_err()` |
| Modularity — `elapsed_ms()` | 20 repeated latency-rounding expressions → `src/state::elapsed_ms()` |
| Magic strings — band names | `"syntax"/"knowledge"/"output"/"all"` → `BAND_*` constants in `band_utils.rs` |
| Magic strings — infer modes | `"walk"/"dense"/"compare"` → `INFER_MODE_*` constants |
| Magic strings — insert modes | `"constellation"/"embedding"` → `INSERT_MODE_*` constants |
| Magic strings — patch names | `"unnamed"/"inline-patch"` → `PATCH_UNNAMED`/`PATCH_INLINE_NAME` constants |
| Magic strings — HTTP headers | `"x-session-id"` → `HEADER_SESSION_ID`; `"etag"/"cache-control"/"if-none-match"` → axum `header::*` |
| Test restructure | `test_api.rs` (2600 L) + `test_http.rs` (1400 L) → 10 focused files (100–350 L each) + `tests/common/mod.rs` |
| Coverage baseline | 39.7% → **49.1% line**, 41.6% → **56.4% function** (345 tests, 0 failures) |

### 2026-04-26 — perf round-1 (G1+G2+G3)

| Item | Outcome |
|---|---|
| G1 cold-start profile | Two-phase: 1.27 s lazy weight load + 17 ms/layer mmap page-in. Warm steady state 0.2–0.3 ms/layer. |
| G2 `/v1/warmup` + `--warmup-walk-ffn` | First walk-ffn 1247 ms → 12.6 ms (99×). Boot trades ~1.3 s + 3.2 GB pre-allocation. HTTP endpoint also exposed for live re-warm. |
| G3 self-assembling gRPC grid | Live-validated `--grid-port` + `--join`: auto-join, coverage tracking, graceful failure (clean HTTP 400 on uncovered layer), auto-recovery on rejoin. |

### 2026-04-26 — W2 retrofit + grid validation

| Item | Outcome |
|---|---|
| `--warmup-hnsw` flag | Eager-builds HNSW across owned layers at boot via `warmup_hnsw_all_layers()`. Reports correct owned-layer count under `--layers`. |
| Boot log: W2 status | `Down features Q4K: loaded (W2 — per-feature decode skips q4k_ffn_layer cache)` when `down_features_q4k.bin` is present. |
| `/v1/stats.q4k_ffn` field | `{cache_slots, cache_bytes, feature_major_down}` — operators can verify W2 active + cache empty in steady state. |
| `larql convert add-feature-major-down` | New CLI subcommand. Retrofits an existing Q4K vindex without re-quantising the rest. 30 layers / 152 MB / 1.12 s on Gemma 26B. Idempotent. |
| Live grid validation | 2-shard layer-range split (0-14 + 15-29) on real 26B vindex, full fan-out via router, 8-way concurrent stress, 0.2 ms warm per-layer, 5.9 ms full-30-layer fan-out. |

### Pre-2026-04-26 — foundations (already in place)

- HTTP API: `/v1/walk`, `/v1/walk-ffn`, `/v1/stats`, `/v1/health`,
  `/v1/infer`, `/v1/insert`, `/v1/expert/{layer}/{id}`, etc.
- `--layers START-END` shard slicing (mmap pages outside range stay
  paged out, RSS proportional to shard size).
- `--max-q4k-cache-layers` LRU bound on the legacy Q4K dequant cache.
- `--ffn-only` / `--embed-only` mode flags.
- gRPC self-assembling grid (`--grid-port` / `--join` / `--grid-key`).
- Bench rig daemon-aware (`larql-vindex` benches refuse if a server
  shares the host; override with `LARQL_BENCH_ALLOW_DAEMONS=1`).
