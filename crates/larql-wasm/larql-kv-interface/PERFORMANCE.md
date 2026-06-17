# Performance — larql-kv

Machine: M3 Max, macOS. Numbers carried from the engine-level audits that
preceded the crate extraction (2026-04-23 onward), with the source bench
identified for each row. The extraction itself was a code move — no
performance changes expected, none observed in the cross-check.

> ⚠️ Single-machine benches on M3 Max are subject to thermal-throttle
> artifacts under sustained GPU load (1.5–3× regressions can appear that
> aren't real). When in doubt, cool-machine rerun before bisecting.

## Engine ladder (Gemma 3 4B, Metal Q4K, 370K-token corpus)

| Engine | Decode (tok/s) | KV memory | Compression | Accuracy |
|---|---|---|---|---|
| `markov_residual` | ~95 | ~171 MB | ~287× | KL = 0.0 (exact) |
| `unlimited_context` | ~94 | ~193 MB | ~254× | exact within window |
| `turbo_quant` (4-bit) | ~95 | ~12.7 GB | ~4× | cos ≈ 0.991 |
| `apollo` (boundaries) | ~8× faster | ~11 MB | ~4,414× | task-level |

Reference: full f16 KV is ~49 GB on the same corpus.

## Per-engine notes

### markov_residual

- **Mechanism.** Stores the pre-layer residual stream and re-projects K/V
  at decode time. The pre-layer residual is the complete Markov state, so
  recomputed K/V is bit-identical to a full-KV baseline.
- **Validated 2026-04-23.** KL = 0.0 vs full-KV on Gemma 3 4B over a
  10-prompt corpus. Survives the 077884b bisect of the 81-84 tok/s
  measurement bug (see project memory note —
  `project_metal_decode_81_was_buggy`).
- **Profiler.** Per-stage breakdown lands in `EngineProfiler`:
  embed, recompute_cold, recompute_hot, attention, ffn, total.

### unlimited_context

- **Mechanism.** Sliding window over the active K/V cache plus a
  checkpoint of the pre-window residual. Decode beyond the window
  re-prefills lazily from the checkpoint. Exact within the window.
- **Tunable.** `window=N` controls the hot tier; default 512.

### turbo_quant

- **Mechanism.** Walsh-Hadamard rotation followed by Lloyd-Max codebook
  quantisation. Encodes K/V at 3- or 4-bit per scalar.
- **Decode.** ~95 tok/s decode at 4-bit, cos ≈ 0.991 vs full-precision K/V.
- **Memory.** ~4× compression of the f16 baseline (so still ~12.7 GB at
  Gemma 3 4B / 370K tokens — orders of magnitude above the residual
  engines, useful when window bounds aren't acceptable).

### apollo

- **Mechanism.** Boundary-residual injection. A constellation index over
  pre-captured boundary points lets decode start the forward pass at the
  configured `crystal_layer` (default 30 of 34) instead of layer 0.
- **Speed.** ~8× decode speedup when the prompt hits a captured
  boundary; falls back to full-stack forward when it doesn't. Memory ≈
  11 MB regardless of corpus size — the constellation is small, the win
  is in skipped layer compute.

## Reproducing

The criterion bench in this crate (see `benches/`) covers each engine's
hot path under a synthetic 2-layer model so it runs anywhere without a
vindex on disk. For end-to-end real-model numbers on a downloaded
checkpoint, use:

```sh
cargo run -p larql-cli --release -- bench gemma3:4b --engine markov-rs
cargo run -p larql-cli --release -- bench gemma3:4b --engine unlimited-context:window=256
cargo run -p larql-cli --release -- bench gemma3:4b --engine turbo-quant:bits=4
cargo run -p larql-cli --release -- bench gemma3:4b --engine apollo:layer=30
```

The `kv-cache-benchmark` crate also runs all four under `cargo bench
-p kv-cache-benchmark --bench kv_strategies` against synthetic K/V tensors
(real-model variant gated behind the `real-model` feature).

## See also

- [`ROADMAP.md`](ROADMAP.md) — open performance / capability work.
- [`CHANGELOG.md`](CHANGELOG.md) — extraction history.
- `larql-compute/PERFORMANCE.md` — Metal pipeline numbers; engines ride
  the `decode_token` path so end-to-end gains often live there.
