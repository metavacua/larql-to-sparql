## Why

PR #147 (`vindex-qwen35moe-extraction`) shipped the writer half: GGUF
→ vindex Q4_K conversion for Qwen 3.6 hybrid SSM/DeltaNet + 256-
expert PerExpert MoE now produces a structurally complete vindex
(`deltanet_weights_q4k.bin`, `attn_weights_q4k.bin`,
`layers/layer_{LL}.weights`, DeltaNet small tensors in `norms.bin`).

But nothing reads it yet. `qwen35_forward_step` (the validated
forward in `crates/larql-inference/src/attention/qwen35_forward.rs`)
consumes a `Qwen35Weights` struct built only by
`qwen35_load::load_qwen35_weights` from a `larql-models::load_gguf`
mmap. The server's vindex loader (`VectorIndex::load_attn_q4k` etc)
has no DeltaNet reader and no qwen35-aware `Qwen35Weights` adapter.

This change closes the reader side so a vindex on disk can drive
the qwen35 forward end-to-end.

## What this change ships

### New code

- **`crates/larql-vindex/src/index/storage/deltanet.rs`** (NEW) —
  mmaps `deltanet_weights_q4k.bin`, parses
  `deltanet_weights_q4k_manifest.json` into per-layer (offset,
  length, format) tuples. Mirrors `storage/attn.rs`'s
  `load_attn_q4k` / `attn_q4k_layer_data` pattern. Adds
  `VectorIndex::deltanet_q4k_layer_data(layer)` returning the 5
  (data, format) tuples for `attn_qkv`, `attn_gate`, `ssm_alpha`,
  `ssm_beta`, `ssm_out`.

- **`crates/larql-inference/src/attention/qwen35_load_vindex.rs`**
  (NEW) — `load_qwen35_weights_from_vindex(dir) -> Qwen35Weights`.
  For each linear layer: pulls Q4_K bytes from
  `deltanet_q4k_layer_data`, dequantises into the
  `Qwen35Weights::layers[L]` slot's DeltaNet projections; for each
  full-attn layer: pulls Q/K/V/O from `attn_q4k_layer_data`,
  dequantises into the attention slots; for each layer: pulls 256
  MoE experts from `layers/layer_{LL}.weights`, dequantises into
  the MoE-expert slots. Small tensors (norms, conv1d, ssm_dt/a)
  come from `norms.bin` via the existing `load_vindex_norms`
  reader. The lm_head comes from `lm_head_q4.bin`.

### Wire-up

- **`crates/larql-inference/src/attention/qwen35_load.rs`** — re-
  export `load_qwen35_weights_from_vindex` alongside the existing
  GGUF loader so callers can pick either.

- **`crates/larql-server/...`** — when a vindex's `index.json`
  reports `arch_family == "qwen35moe"` (or `"qwen35"`), the server's
  per-request decode path SHALL dispatch to a new helper
  `qwen35_forward_chat_completion` that wraps
  `qwen35_forward_step` over the chat-template-rendered token
  sequence. The default (non-qwen35) path is unchanged.

### Validation

- Live smoke: with the vindex from PR #147 at
  `/tank/ai/Qwen/Qwen3.6-35B-A3B-vindex`, the server SHALL respond
  to a `/v1/chat/completions` request without panicking and produce
  some deterministic completion.

- **Parity vs llama.cpp** is out of scope here — the vindex-load
  path's dequant noise may diverge from the GGUF-direct path's
  dequant noise. A follow-up parity change locks the floor.

## Capability deltas

Under `vindex-extraction/` — extends the writer-side coverage from
PR #147 with the matching reader contract.

## Out of scope (named follow-ups)

- Forward parity vs llama.cpp on vindex-loaded weights (separate
  change with a fresh oracle harness).

- `gate_vectors.bin` 40 GB → Q4_K — a separate writer is still
  emitting MoE router weights as f32. Independent size win.

- DeepSeek V4 Flash MLA path — still parked.

- Direct byte-slice forward (bypass the dequant into
  `Qwen35Weights`). The PR ships the conservative
  load-and-dequantise path; a follow-up can route bytes through the
  existing `q4k_q8k_matvec_avx2` kernels for the same RAM savings
  the Gemma path got from #146.

## Estimated effort

- Storage reader: ~150 LoC
- Vindex-loader for `Qwen35Weights`: ~300 LoC
- Server dispatch wire-up: ~100 LoC
- Smoke + tests: ~100 LoC

Total ~650 LoC. Single focused PR target.
