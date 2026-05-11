# larql-kv

Pluggable KV-cache engines for `larql-inference`. Each engine implements the
full prefill + autoregressive decode loop but manages persistent inference
state differently — trading memory, accuracy, and speed.

## Engine ladder

Numbers measured on Gemma 3 4B @ 370K-token corpora, M3 Max, Metal Q4K.
See [`PERFORMANCE.md`](PERFORMANCE.md) for the full audit.

| Engine | Speed (tok/s) | KV memory | Compression | Accuracy |
|---|---|---|---|---|
| [`markov_residual`](src/engines/markov_residual) | ~95 | ~171 MB | ~287× | exact (KL = 0.0) |
| [`unlimited_context`](src/engines/unlimited_context) | ~94 | ~193 MB | ~254× | exact within window |
| [`turbo_quant`](src/engines/turbo_quant) | ~95 | ~12.7 GB | ~4× | cos ≈ 0.991 |
| [`apollo`](src/engines/apollo) | ~8× faster with boundaries | ~11 MB | ~4,414× | task accuracy |

Reference for "compression" is full f16 KV at 49 GB on the same corpus.

## Usage

```rust
use larql_kv::{EngineKind, KvEngine};
use larql_compute::default_backend;

// Parse a CLI engine spec.
let kind = EngineKind::from_name("markov-rs:window=512").unwrap();

// Build an engine bound to a compute backend.
let mut engine: Box<dyn KvEngine> = kind.build(default_backend());

// Prefill, then decode autoregressively.
let hidden = engine.prefill(&weights, &prompt_tokens).unwrap();
for _ in 0..n {
    let hidden = engine.decode_step(&weights, last_token).unwrap();
    let logits = larql_inference::forward::hidden_to_raw_logits(&weights, &hidden);
    // sample, append, loop
}
```

The engines also expose Q4K-quantised entry points
(`prefill_q4k` / `decode_step_q4k`) that route through the Metal
`decode_token` pipeline when a Q4K `VectorIndex` and a Metal backend are
available, falling back to the f32 path otherwise.

## CLI selectors

The CLI parses engine specs as `name` or `name:key=value[,key=value]`:

```text
markov-rs                                 # default
markov-rs:window=1024
unlimited-context:window=256
turbo-quant:bits=3      # alias: tq3
turbo-quant             # bits=4 default; alias: tq4
apollo:layer=25,coef=8.0,top_k=12
```

All four engines are benched via `larql bench <model> --engine <spec>`.

## Crate layout

```
larql-kv/
├── src/
│   ├── lib.rs          — KvEngine trait, EngineKind, EngineInfo, dispatch
│   ├── accuracy.rs     — cosine, MSE, KL, JS, compare_hidden helpers
│   ├── profiler.rs     — per-stage decode timing accumulators
│   └── engines/
│       ├── apollo/             — boundary-residual injection, ~4,000× compression
│       ├── markov_residual/    — residual-stream KV replacement, KL = 0
│       ├── turbo_quant/        — WHT + Lloyd-Max K/V codec (3- or 4-bit)
│       └── unlimited_context/  — windowed re-prefill from checkpoints
├── benches/            — criterion microbenchmarks
├── examples/           — end-to-end demos on synthetic test_utils
└── coverage-policy.json — per-file ≥90% line-coverage policy
```

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
- **`kv-cache-benchmark`** — criterion-driven comparison of all engines plus
  baselines (Standard KV, Graph Walk).

## History

Extracted from `larql-inference::engines` on 2026-05-09. See
[`CHANGELOG.md`](CHANGELOG.md). Forward-looking work in
[`ROADMAP.md`](ROADMAP.md).
