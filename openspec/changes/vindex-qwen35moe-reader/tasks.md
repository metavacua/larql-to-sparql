## Phase 1 — DeltaNet storage reader (~150 LoC)

- [ ] 1.1 New `crates/larql-vindex/src/index/storage/deltanet.rs`.
      Implement `VectorIndex::load_deltanet_q4k(dir)` mirroring
      `load_attn_q4k`: mmap + manifest parse + storage placement.
- [ ] 1.2 Add `VectorIndex::deltanet_q4k_layer_data(layer) ->
      Option<[(&[u8], &str); N]>` accessor (N = 5: attn_qkv,
      attn_gate, ssm_alpha, ssm_beta, ssm_out).
- [ ] 1.3 Extend `VindexStorage` with the new mmap + per-layer
      manifest slice.
- [ ] 1.4 Unit test on a synthetic 4-layer hybrid manifest.

## Phase 2 — `Qwen35Weights` vindex loader (~300 LoC)

- [ ] 2.1 New `crates/larql-inference/src/attention/qwen35_load_vindex.rs`.
      Function `load_qwen35_weights_from_vindex(dir: &Path) ->
      Result<Qwen35Weights, ...>` that reconstructs every field of
      the struct from vindex files.
- [ ] 2.2 Per linear layer: dequantise Q4_K bytes from
      `deltanet_q4k_layer_data` for attn_qkv / attn_gate /
      ssm_alpha / ssm_beta / ssm_out.
- [ ] 2.3 Per full-attn layer: dequantise from `attn_q4k_layer_data`
      for Q / K / V / O.
- [ ] 2.4 Per layer: dequantise 256 MoE experts from
      `layers/layer_{LL}.weights` (parse the header + offset table
      using `larql_vindex::format::weights::write_layers::parse_layer_weights_header`).
- [ ] 2.5 Per layer: read DeltaNet small tensors (ssm_norm, ssm_dt,
      ssm_a, ssm_conv1d) from `norms.bin` via existing reader.
- [ ] 2.6 Read `lm_head_q4.bin` + dequantise.
- [ ] 2.7 Re-export from `qwen35_load.rs`.

## Phase 3 — Server dispatch (~100 LoC)

- [ ] 3.1 In `crates/larql-server/...` server load path, detect
      `arch_family == "qwen35moe" || "qwen35"` from
      `VindexConfig::model_config`.
- [ ] 3.2 New helper that wraps `qwen35_forward_step` over a
      tokenised prompt for `/v1/chat/completions` decoding.
- [ ] 3.3 Default (non-qwen35) path unchanged.

## Phase 4 — Validation gates

- [ ] 4.1 `cargo test -p larql-vindex --lib --release` green.
- [ ] 4.2 `cargo test -p larql-inference --lib --release` green.
- [ ] 4.3 `openspec validate vindex-qwen35moe-reader --strict`
      passes.
- [ ] 4.4 Live smoke: `target/release/larql serve
      /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex` + an HTTP
      `/v1/chat/completions` request returns a 200 with some
      completion text (no panic, no garbage UTF-8). Coherence
      check (top-1 vs llama.cpp) is a follow-up change.

## Out of scope

- Forward parity vs llama.cpp — separate change.
- `gate_vectors.bin` 40 GB → Q4_K — separate change.
- Direct byte-slice forward (RAM savings) — follow-up after the
  load-and-dequantise path proves end-to-end correctness.
- DeepSeek V4 Flash MLA — separate change family.
