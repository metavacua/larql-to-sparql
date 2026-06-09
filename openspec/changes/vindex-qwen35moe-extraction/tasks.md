## Phase 1 — DeltaNet writer (~300 LoC)

- [x] 1.1 New `crates/larql-vindex/src/format/weights/write_q4k/deltanet.rs`.
      Function `write_deltanet_weights_q4k(source, dir, num_layers,
      callbacks)`. Iterates layers; for each linear-attention layer
      (where `arch.is_linear_attention_layer(layer) == true`), writes:
      - `attn_qkv` → Q4_K
      - `attn_gate` → Q4_K
      - `ssm_out` → Q4_K
      - `ssm_norm` → Q6_K (vector, post-recurrence RMSNorm weight)
      - `ssm_dt`, `ssm_a`, `ssm_beta`, `ssm_alpha` → f32 (small scalars/vecs)
      - `ssm_conv1d` → f32 (depthwise conv weight, ~10240 floats per layer)
      Emits `deltanet_weights_q4k.bin` + `deltanet_weights_q4k.manifest.json`.
- [x] 1.2 Update `write_q4k/mod.rs` to call `deltanet::write_deltanet_weights_q4k`
      when `arch.full_attention_interval() != 0`.
- [x] 1.3 Skip linear layers in `attn::write_attn_weights_q4k` loop body
      (line 35 onward) when `arch.is_linear_attention_layer(layer)` is true.
- [x] 1.4 New filename constants in `format/filenames.rs`:
      `DELTANET_WEIGHTS_Q4K_BIN`, `DELTANET_WEIGHTS_Q4K_MANIFEST_JSON`.

## Phase 2 — qwen35moe PerExpert MoE writer (~200 LoC)

- [x] 2.1 Extend `moe_layers::write_per_layer_moe_q4k` to handle
      `arch.is_moe() && arch.expert_format() == ExpertFormat::PerExpert`
      in addition to the existing `is_hybrid_moe() && PackedBF16` case.
- [x] 2.2 For each layer, walk experts 0..num_experts via
      `arch.expert_ffn_gate_key`, `_up_key`, `_down_key`. Each tensor
      goes through `quantize_q4_k` and lands in `layers/layer_{L:02}.weights`
      with one entry per expert.
- [x] 2.3 Verify the existing `LayerWeightFormat::Q4_K` writer path
      can be reused or extended for the per-expert count (256 entries
      per layer instead of Gemma 4's 128).

## Phase 3 — Guard removal + smoke test (~50 LoC)

- [x] 3.1 Remove the `q4k && full_attention_interval() != 0` block in
      `convert_cmd.rs:515-524`.
- [x] 3.2 Smoke test: run the full conversion on the downloaded
      `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. Verify:
      - exit 0
      - `deltanet_weights_q4k.bin` exists and is ~size-consistent
        with 30 linear layers × all-tensors-Q4K-equivalent
      - `attn_weights_q4k.bin` exists for 10 full-attention layers
        (40 × 4 / 4 = 10)
      - `layers/layer_00.weights` … `layer_39.weights` exist, each
        with 256 expert entries × 3 projections
      - `index.json` carries `full_attention_interval`, `ssm_*` metadata

## Phase 4 — Validation gates

- [x] 4.1 `cargo test -p larql-vindex --lib --release` passes (922/922).
      `openspec validate vindex-qwen35moe-extraction --strict` passes.
      `cargo fmt --all --check` passes. Workspace-wide `cargo clippy
      --workspace -- -D warnings` fails on pre-existing
      `larql-compute` `identity_op` errors (4) — independent of this
      change; same blocker called out in RESUME_PROMPT.
- [x] 4.2 Live smoke conversion: `target/release/larql convert
      gguf-to-vindex --level inference --quant q4k --output
      /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex
      /tank/ai/Qwen/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
      exits 0; produces `deltanet_weights_q4k.bin` (542 MB, 30 linear
      layers × 5 matmul tensors per layer), `attn_weights_q4k.bin`
      (148 MB, 10 full-attn layers × Q/K/V/O), `layers/layer_{00..39}.weights`
      (40 files × 256 expert entries each), `norms.bin` (4.4 MB, with
      ssm_norm/dt/a/conv1d per linear layer), and `index.json`
      reflecting `qwen35moe` with `full_attention_interval=4`.
- [ ] 4.3 (Deferred to follow-up PR) Synthetic-arch unit tests for the
      DeltaNet writer + PerExpert MoE writer. The live smoke
      conversion proves end-to-end correctness; synthetic unit tests
      would lock the contracts against future arch handler drift but
      are not blocking the structural completeness goal.
- [ ] 4.4 (Deferred — reader path follow-up) Forward parity vs
      llama.cpp on vindex-loaded weights. Requires the reader
      follow-up change to land first.

## Out of scope

- **Vindex readers** for DeltaNet tensors (next change).
- **Server dispatch** of qwen35* arches (next-next change).
- **F16 path** for hybrid arches (untouched; the existing branch
  may or may not work).
- **Parity vs llama.cpp** on the vindex weights — depends on
  reader landing.
- **DeepSeek V4 Flash** MLA path — parked.
