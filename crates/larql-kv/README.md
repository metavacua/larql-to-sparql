# larql-kv

**LARQL KV engines separate model continuation state from execution
cache.** Standard engines store K/V as state. Residual-state engines
store the residual stream and derive K/V only when execution needs
it. The choice changes how the engine composes with the dispatch
hot path — and as of 2026-05-21, the three derivative-K/V engines
match `standard`'s fused-kernel speed because they can elide the
GPU→CPU state bridge entirely (W10 mask cascade, default-on).

> *KV cache is an implementation detail. Continuation state is the
> real abstraction.* — [State Policy §2](docs/state-policy.md)

### Engine ladder at a glance

| Engine | Canonical state | K/V role | Contract | Bench (tok/s) |
|---|---|---|---|---:|
| `standard` | K/V tensors | canonical | exact logits | 97.6 |
| `no-cache` | tokens | recomputed | exact logits | (debug) |
| `markov-rs` | residual stream | **derivative** | exact logits under arch contract | **98.0** |
| `markov-rs-codec` | compressed residuals | **derivative** | bounded KL | **98.1** |
| `boundary-per-layer` | per-layer codec residuals | **derivative** | bounded KL per-layer | **98.7** |
| `windowed-checkpoint` | KV (within window) + checkpoints | **derivative** | exact within window | 94.2 |
| `turbo-quant` | quantised K/V | canonical (destructive) | bounded KL | 85.0 |
| `boundary-kv` | K/V + boundary frames | canonical | exact logits | composes `standard` |
| `apollo` *(RetrievalEngine)* | boundary retrieval store | n/a (retrieval) | task-level | orthogonal |

Gemma 3 4B Q4K, Metal, M3 Max, 50 decode tokens, W10 default-on
(2026-05-21). The 13% gap between the four derivative-K/V engines
and `turbo-quant` is the cleanest available evidence that the
canonical-vs-derivative classification is *operationally* load-bearing,
not just descriptive. See [docs/state-policy.md §3.1](docs/state-policy.md)
for the prediction; task #31 is the falsification path.

### Trait location

The `KvEngine` + `RetrievalEngine` traits, the `EngineError` /
`EngineInfo` / `DecodeStageSummary` types, and the
`AnyEngine::{Kv, Retrieval}` dispatch enum all live in
`larql-inference::kv_engine`; this crate re-exports them
(`larql_kv::KvEngine` / `RetrievalEngine` / `AnyEngine` /
`EngineError`) so callers don't need to depend on `larql-inference`
directly. The traits live upstream so `larql-inference`'s decode
dispatch (`generate_with_engine`) can reference them without a
circular dep. See
[`crates/larql-inference/docs/specs/kv-engine-unification.md`](../larql-inference/docs/specs/kv-engine-unification.md)
for the dep-graph rationale.

**Two trait surfaces:**

| Trait | For | Per-step contract |
|---|---|---|
| `KvEngine` | per-token K/V cache engines (`standard`, `no-cache`, `markov-rs`, `markov-rs-codec`, `windowed-checkpoint`, `turbo-quant`, `boundary-kv`, `boundary-per-layer`) | append K/V per layer; dispatches FFN through `&dyn FfnBackend`; state reconstructible to K/V tensors |
| `RetrievalEngine` | retrieval-injection engines (`apollo`, future Mode 5) | no per-token K/V append; no `FfnBackend` dispatch; state is residual delta + token list |

Both return `Result<Array2<f32>, EngineError>` from `prefill` /
`decode_step` (and the `*_quant` variants). `EngineError` carries
five variants — `EmptyPrompt`, `BackendUnavailable`,
`RetrievalMiss { reason }`, `InvariantViolation { what }`,
`BackendFailure { details }` — and the typed error replaces the
historical `Option<T>` that collapsed all five into a silent
`None`. See the 2026-05-24 entry in [`CHANGELOG.md`](CHANGELOG.md).

## Full engine catalog

Nine engines total. `Standard` and `NoCache` wrap today's production
behaviour; the others are research engines that trade accuracy or
memory for different state-policy properties (compressed cold tier,
windowed checkpoints, retrieval injection, etc.).

| Engine | Mechanism | Hot state (Gemma 3 4B) | Metal tok/s (Full) | W10 best | Accuracy | Spec |
|---|---|---:|---:|---:|---|---|
| [`standard`](src/engines/standard.rs) | Production K/V tensor cache; `window=None` unbounded, `Some(N)` sliding | full K/V, backend-managed | **~100** | n/a (reference) | exact — the reference | [standard-engine.md](../larql-inference/docs/specs/standard-engine.md) |
| [`boundary_kv`](src/engines/boundary_kv) | `standard` + `larql-boundary` chunk frames for cross-session resume | same as standard | ~100 | n/a | exact | [boundary-kv-engine.md](../larql-inference/docs/specs/boundary-kv-engine.md) |
| [`no_cache`](src/engines/no_cache.rs) | No K/V; full re-forward per step (O(N²)) | token list only | — | n/a | exact (correctness fallback) | [no-cache-engine.md](../larql-inference/docs/specs/no-cache-engine.md) |
| [`markov_residual`](src/engines/markov_residual) | Residual-stream replacement, K/V derived from stored residuals (W2 cache) | 54.4 MB → **0 MB** | 88.2 | **99.5** (None, +13%) | exact (KL = 0.0) under contract | [markov-residual-engine.md](../larql-inference/docs/specs/markov-residual-engine.md) |
| [`markov_residual_codec`](src/engines/markov_residual_codec) | `markov_residual` + bf16-encoded cold-tier residuals (2× cold saving) | 54.4 MB → **0 MB** | 87.2 | **99.8** (None, +14%) | bounded-KL vs markov_residual | [markov-residual-codec-engine.md](../larql-inference/docs/specs/markov-residual-codec-engine.md) |
| [`boundary_per_layer`](src/engines/boundary_per_layer) | Per-layer codec policy on cold tier; calibration-driven; W1-GPU + W10 wired | 19.6 MB → **0 MB** | 86.9 | **99.3** (None, +14%) | per-layer KL bound | [boundary-per-layer-engine.md](../larql-inference/docs/specs/boundary-per-layer-engine.md) |
| [`windowed_checkpoint`](src/engines/windowed_checkpoint) | Per-window K/V checkpoint + token archive; supports replay | 15.7 MB → **0 MB** | 86.1 | **95.0** (HOnly, +10%) | exact within window | [windowed-checkpoint-engine.md](../larql-inference/docs/specs/windowed-checkpoint-engine.md) |
| [`turbo_quant`](src/engines/turbo_quant) | WHT + Lloyd-Max 3/4-bit K/V codec, in-place compression | 0.7 MB | **37.7** (10-tok) | n/a (canonical K/V) | cos ≈ 0.991 | [turbo-quant-engine.md](../larql-inference/docs/specs/turbo-quant-engine.md) |
| [`apollo`](src/engines/apollo) | Constellation map + boundary-residual injection (retrieval) | scales w/ store | requires store | n/a | task-level | [apollo-engine.md](../larql-inference/docs/specs/apollo-engine.md) |

**Numbers are post W2 (hot K/V cache), W1-GPU (per-layer state-dump
dispatch), W7 (blit-encoder fusion), and W10 (state-bridge mask
cascade).** Three derivative-K/V engines (`markov_residual`,
`markov_residual_codec`, `windowed_checkpoint`) now match or exceed
`standard`'s fused-kernel ~100 tok/s ceiling under their best mask,
with engine-side memory shadows fully eliminated on Metal. See
[`PERFORMANCE.md`](PERFORMANCE.md) for per-token cost decomposition,
the `state_capture` / `state_materialise` / `state_append` timer
cascade, and the bench protocol. [`CHANGELOG.md`](CHANGELOG.md) has
the milestone history.

### W10 — state-bridge mask cascade (default-on since 2026-05-21)

Engines that treat K/V as **derivative** state (see
[`docs/state-policy.md`](docs/state-policy.md)) automatically take a
mask cascade:

- **`HOnly`** — skip the GPU→CPU K/V staging blit + readback on
  Metal. Triggered when the engine drops its `hot_kv` shadow.
- **`None`** — also skip the h_in staging blit + readback. Triggered
  when the engine *additionally* drops its residual store (only safe
  with `window=None` — no cold-tier eviction can fire).

The cascade is bit-identical to `Full` under the engine's
exact_logits contract (proven by `examples/w10_parity_gate.rs`); it
closes ~13% of the gap to `standard`'s fused-kernel ceiling and zeros
out the engine-side memory shadow. Backends without an optimised
masked path fall through to `Full` via the trait's default impl —
correct everywhere, perf-positive only on Metal.

Opt out with `LARQL_W10_DISABLE=1` (debug instrument; useful for
bisecting backend-side masked-kernel regressions). The legacy
`LARQL_W10_HONLY=1` env var is still accepted but is now a no-op.

`turbo_quant` doesn't take the cascade — its codec is destructive,
so K/V can't be derived from residuals. It stays on `Full` mask
regardless of the env var.

### Standard vs MarkovResidual

`Standard` (this crate's wrapper over the production K/V cache) and
`MarkovResidual` (residual-stream replacement) are **different
mechanisms** that happen to produce bit-identical output on supported
architectures. Don't conflate them — the CLI's historical `--kv-cache
markov-bounded` flag maps to `Standard { window_size: Some(N) }`, **not**
`MarkovResidual`. Use the spec's table in §5 when in doubt.

### Windowed engines and the fused path

A window means two promises at once: attend at most `N` positions, and
hold at most that much K/V. Until 2026-08 the only way to keep both was
to leave the backend's fused (coarse) path for the generic per-layer
route — and on Metal every per-layer dispatch method delegates to the
CPU, so a windowed engine ran its whole forward on the host while the
bench row still said `[metal (GPU)]`. On Gemma 3 4B Q4K that cost ~9x,
for a window that should make attention *cheaper*.

Windowed engines now stay on the fused path. The window is requested via
`coarse_prefill_windowed` / `coarse_decode_step_windowed`, which **fail
closed**: a backend that cannot bound both attention and K/V answers
`None` and the engine falls back to per-layer, exactly as before. Both
`CpuBackend` and `MetalBackend` implement them.

Two caveats worth knowing:

- **A prompt longer than the window still takes the per-layer path.** The
  fused prefill has no per-query-position masking, so accepting would
  attend the whole prompt while advertising a bound.
- **Metal holds up to 2x the window** between compactions. Attention is
  still bounded at the window every step (the kernel attends
  `[T - window, T)`); the surplus rows are resident but never read.
  Compacting every step would memmove the window per token.

Measured on Gemma 3 4B Q4K, Metal, 80 decode steps at `window=8`:

| | latency | hot K/V |
|---|---|---|
| unwindowed | 12.06 ms | 23.6 MB |
| `window=8` | **11.61 ms** | **2.4 MB** |

The same config cost 115.44 ms before. A bench row reports which shape
it actually took — `[coarse]` or `[per-layer->host]`.

## Usage

```rust
use larql_kv::{AnyEngine, EngineKind};
use larql_inference::{cpu_engine_backend, ffn::WeightFfn};

// Parse a CLI engine spec.
let kind = EngineKind::from_name("standard:window=512").unwrap();

// Build an engine bound to a compute backend. Returns AnyEngine —
// the dispatch enum that wraps either a KvEngine (per-token K/V
// cache) or a RetrievalEngine (Apollo). Callers reach the prefill /
// decode_step methods through AnyEngine's forwarding methods, which
// pattern-match the variant internally.
let mut engine: AnyEngine = kind.build(cpu_engine_backend());

// FFN router — `WeightFfn` reads weights locally; pass any FfnBackend
// impl (e.g. `RemoteWalkBackend` for remote-FFN dispatch). Retrieval
// engines ignore this parameter; per-token K/V engines use it.
let ffn = WeightFfn { weights: &weights };

// Prefill, then decode autoregressively. Both methods return
// `Result<Array2<f32>, EngineError>`; the bench / accuracy harnesses
// route on the typed error kind (see EngineError::is_recoverable).
let hidden = engine.prefill(&weights, &ffn, &prompt_tokens)?;
let next = engine.decode_step(&weights, &ffn, last_token)?;

// Or use the helper that drives the full prefill + sample + decode loop.
let generated = larql_kv::generation::generate_with_engine(
    &mut engine,
    &weights,
    &tokenizer,
    &ffn,
    &prompt_tokens,
    max_tokens,
    |id, tok| { /* on-token callback */ },
);
```

The engines also expose quantised entry points
(`prefill_quant` / `decode_step_quant`) that route through the
`VectorIndex` for backend-fastest decode (Metal Q4K kernel when
available, CPU f32 fallback otherwise). The methods are
quant-agnostic — they dispatch on whatever format the vindex
carries (`Q4_K`, `Q6_K`, future formats).

## CLI selectors

The CLI parses engine specs as `name` or `name:key=value[,key=value]`:

```text
standard                                  # production K/V cache, unbounded (default)
standard:window=1024                      # sliding-window K/V
no-cache                                  # full re-forward per step (O(N²)), debug only
markov-rs                                 # residual-stream replacement
markov-rs:window=1024
windowed-checkpoint:window=256
turbo-quant:bits=3        # alias: tq3
turbo-quant               # bits=4 default; alias: tq4
apollo:layer=25,coef=8.0,top_k=12
```

Legacy aliases for `standard`: `full`, `fp32`. Legacy aliases for
windowed standard: `markov-bounded`, `bounded`, `sliding`. Legacy aliases
for `no-cache`: `none`, `off`.

All engines are reachable via `larql bench <model> --engine <spec>`.
`larql run` and `larql walk` route through `KvEngine` dispatch by
default — the legacy `--kv-cache standard|markov-bounded|none` flag now
resolves to `Standard { window_size }` / `NoCache` engines transparently
(spec §6.1 mapping table). `larql run --engine` and `LARQL_KV_ENGINE`
shipped 2026-05-16 on run/walk; Apollo is bench-only. Server wiring is
deferred to the `AsyncComputeBackend` rollout (kv-engine-unification
spec §10.6) — without it the server would silently downgrade GPU
decode to CPU.

## Async opt-in (`StandardEngine`)

`StandardEngine` accepts either a synchronous `EngineBackend` (the
default `--kv-cache standard` path) or an `AsyncComputeBackend` for
deferred-dispatch GPU batching:

```rust
use larql_kv::engines::standard::StandardEngine;
use larql_inference::AsyncComputeBackend;
use larql_compute::CpuBackend;

// Sync (default).
let mut sync_engine = StandardEngine::new(None);

// Async opt-in. On CpuBackend (degenerate `Ready*` wrapper) output is
// bit-identical to the sync path; on Metal once Step A4's deferred
// dispatch lands, it becomes one GPU command buffer per decode step.
let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
let mut async_engine = StandardEngine::with_async_backend(None, backend);
```

The other research engines (`MarkovResidual`, `WindowedCheckpoint`,
`TurboQuant`, `NoCache`, `Apollo`) gain the same `with_async_backend`
constructor in subsequent slices. Spec:
[`async-compute-backend.md`](../larql-inference/docs/specs/async-compute-backend.md).

## Crate layout

```
larql-kv/
├── src/
│   ├── lib.rs          — EngineKind dispatch + re-exports of the trait surface
│   ├── accuracy.rs     — cosine, MSE, KL, JS, compare_hidden helpers
│   ├── accuracy_suite/ — parametric/in-context/conflict split-axis evaluation
│   │   ├── prompts.rs    — 101 parametric prompts (KnowledgeSource::Parametric)
│   │   ├── needle.rs     — needle-in-haystack 512→32K (KnowledgeSource::InContext)
│   │   ├── conflict.rs   — in-context-contradicts-parametric (KnowledgeSource::Conflict)
│   │   ├── runner.rs     — KvEngine drivers + Shannon scorer + split table
│   │   └── measurement.rs — KL/JS/softmax/top_k_overlap helpers
│   ├── cache.rs        — legacy `KvCache` shape used by StandardEngine
│   ├── generation.rs   — `generate_with_engine`, `generate_cached_*` parity oracle
│   ├── vindex_compare.rs — A/B comparison of two vindexes on the same model
│   ├── profiler.rs     — per-stage decode timing accumulators
│   └── engines/
│       ├── standard.rs           — production K/V tensor cache (default)
│       ├── no_cache.rs           — full re-forward per step (debug fallback)
│       ├── apollo/               — boundary-residual injection, ~4,000× compression
│       ├── markov_residual/      — residual-stream KV replacement, KL = 0
│       ├── turbo_quant/          — WHT + Lloyd-Max K/V codec (3- or 4-bit)
│       └── windowed_checkpoint/    — windowed re-prefill from checkpoints
├── benches/            — criterion microbenchmarks
├── examples/           — end-to-end demos on synthetic test_utils
├── baselines/          — committed `larql accuracy` regression baselines
└── coverage-policy.json — per-file ≥90% line-coverage policy
```

The `KvEngine` trait itself lives in
[`larql-inference/src/kv_engine.rs`](../larql-inference/src/kv_engine/mod.rs).

## Architecture notes

- **Metal Q4K path.** All four engines route through the Metal
  `decode_token` full pipeline when a Q4K `VectorIndex` and Metal backend
  are available — 93–95 tok/s on Gemma 3 4B, matching the standard
  larql-metal path.
- **CPU fallback.** When Metal is unavailable, engines fall back to a CPU
  path using dequantised attention tensors (lazily inserted into
  `weights.tensors`) and `WalkFfn` for Q4K FFN.
- **Apollo compressed path.** When the store has boundary residuals
  captured at `crystal_layer` (default 30), `forward_from_layer` runs only
  `crystal_layer..num_layers` layers (~4 instead of 34), ~8.5× faster per
  step.

## Relationship to other crates

- **`larql-inference`** — provides the transformer primitives that engines
  compose (`attention::*`, `forward::*`, `ffn::BackendFfn`,
  `vindex::WalkFfn`, `model::ModelWeights`, `residual::*`,
  `layer_graph::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ`).
- **`larql-compute`** — the `ComputeBackend` trait engines dispatch through.
- **`larql-vindex`** — the `VectorIndex` engines query for Q4K weights.
- **`larql-cli bench`** (`larql_cli::commands::primary::bench`) — `--engine
  <spec>` selector dispatches every engine through a uniform criterion-style
  harness; cross-engine **throughput** comparisons live here.
- **`larql-cli accuracy`** (`larql_cli::commands::primary::accuracy_cmd`) —
  drives `accuracy_suite` against any model + engine list, splits results
  by parametric / in-context / conflict, scores with top-1 + Shannon
  bits-per-token. `larql accuracy <model> --quick --engines standard,markov-rs`
  is the fast smoke run; full corpora are 101 + 7 + 20 prompts. JSON export
  via `--output-file`. The historical `kv-cache-benchmark` crate that hosted
  synthetic-strategy comparators was retired in 2026-05-16 — its surviving
  pieces (`accuracy_suite` + `vindex_compare`) live in this crate.

## Design

- [`docs/state-policy.md`](docs/state-policy.md) — engine identity =
  `(canonical_state, derivative_state, correctness_contract)`. The
  vocabulary used to slot new engine proposals + judge whether a
  derivative cache changes engine identity (it doesn't).
- [`engine-state-vs-execution.md`](../larql-inference/docs/specs/engine-state-vs-execution.md)
  — the orthogonal cut: engine ≠ dispatch decisions.

## History

Extracted from `larql-inference::engines` on 2026-05-09. See
[`CHANGELOG.md`](CHANGELOG.md). Forward-looking work in
[`ROADMAP.md`](ROADMAP.md).
