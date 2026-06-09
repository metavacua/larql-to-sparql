## Why

Qwen3.6-35B-A3B Q4_K_M was downloaded to `/tank/ai/Qwen/Qwen3.6-35B-A3B-GGUF/`
(21 GB, 256 experts top-8, full_attention_interval=4). Running
`larql convert gguf-to-vindex --level inference --quant q4k` on it
fails fast with:

```
Error: --quant q4k does not yet support hybrid SSM/DeltaNet
architectures (detected family `qwen35moe` with
full_attention_interval=4). Use `--f16` for now; hybrid-aware Q4_K
writer is a follow-up.
```

— `crates/larql-cli/src/commands/extraction/convert_cmd.rs:515`.

The guard is honest about what's missing: the existing q4k writer
emits only standard transformer Q/K/V/O attention projections and
gates MoE expert writing on `is_hybrid_moe() && ExpertFormat::PackedBF16`
(Gemma 4 26B A4B's path only). qwen35moe needs:

1. **DeltaNet/SSM tensor writers** for linear-attention layers
   (3 of every 4 layers in Qwen 3.6) — `attn_qkv`, `attn_gate`,
   `ssm_conv1d`, `ssm_dt`, `ssm_a`, `ssm_beta`, `ssm_alpha`,
   `ssm_norm`, `ssm_out`. These are silently dropped today.

2. **PerExpert MoE expert layout** for 256-expert top-8 routing.
   The existing `moe_layers.rs` only handles `PackedBF16` (Gemma 4
   26B A4B's hybrid-MoE path). qwen35moe's `ExpertFormat::PerExpert`
   path falls through to nothing.

3. **Guard removal** at `convert_cmd.rs:515` once writers exist.

Reaching the server is **NOT** in scope for this change — see
`9f0cb8a` gap analysis for the full bridge. This change only closes
the writer half. Reader + dispatcher are follow-ups.

## What this change ships

### Code

- **`crates/larql-vindex/src/format/weights/write_q4k/deltanet.rs`**
  (NEW) — per-layer writer for DeltaNet/SSM tensors. Emits
  `deltanet_weights_q4k.bin` + manifest. Q4_K for matmul tensors
  (`attn_qkv`, `attn_gate`, `ssm_out`); Q6_K for `ssm_norm`; f32 for
  the small scalar/vector params (`ssm_dt`, `ssm_a`, `ssm_beta`,
  `ssm_alpha`); int8 packed for `ssm_conv1d` weights (~10240 elements).

- **`crates/larql-vindex/src/format/weights/write_q4k/moe_layers.rs`**
  — extend `write_per_layer_moe_q4k` to handle `ExpertFormat::PerExpert`
  on `qwen35moe` arch. Pulls per-expert gate/up/down via
  `arch.expert_ffn_*_key(layer, expert_id)`, quantises each to Q4_K,
  writes into the same `layers/layer_{L:02}.weights` format used by
  Gemma 4 26B A4B. 256 experts × 3 projections × 40 layers.

- **`crates/larql-vindex/src/format/weights/write_q4k/attn.rs`** —
  loop body skips DeltaNet layers (where `arch.is_linear_attention_layer(layer)
  == true`) so the existing Q/K/V/O writer only fires on full-attention
  layers (1 in 4 for Qwen 3.6).

- **`crates/larql-vindex/src/format/weights/write_q4k/mod.rs`** —
  orchestrator calls the new `deltanet::write_deltanet_weights_q4k`
  for hybrid arches.

- **`crates/larql-cli/src/commands/extraction/convert_cmd.rs`** —
  remove the `q4k && full_attention_interval() != 0` guard at line
  515.

- **`crates/larql-vindex/src/config/model.rs`** — verify
  `full_attention_interval`, `ssm_state_size`, `ssm_inner_size`,
  `ssm_dt_rank`, `ssm_group_count`, `ssm_conv_kernel`,
  `rope_dimension_sections` are all populated into `index.json`
  during the build. (Phase A.2 work — some may already be done.)

### Capability deltas

Under `vindex-extraction/` — new capability. Codifies the rules for
which architectures the GGUF→vindex pipeline supports at each
`--quant` level.

### Validation

- `cargo run --release --bin larql -- convert gguf-to-vindex
   --level inference --quant q4k --output /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex
   /tank/ai/Qwen/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
  succeeds, produces files for: 40 sets of DeltaNet tensors (30
  linear layers' worth) + 10 full-attention layers' Q/K/V/O +
  40 sets of 256 experts in `layers/layer_{L:02}.weights`.

- Structural validation: `larql describe <vindex>` reports 40
  layers (10 full-attn + 30 linear) and 256 experts/layer matching
  the manifest manifest entries' counts.

- **NOT validated in this PR:** forward parity (no reader yet) or
  server dispatch (no routing yet). Both are follow-ups.

## Bench (no perf claims yet)

This PR is structural; nothing runs through the new tensors until
the reader+dispatcher follow-ups land. Expected vindex size for
the converted model: roughly 21 GB → ~20 GB (the f16 Q4_K_M Q->Q
re-quant of `unsloth/Qwen3.6-35B-A3B-UD-Q4_K_M`).

## Out of scope (follow-ups)

- **Vindex readers** for DeltaNet tensors. `larql-inference`'s
  qwen35_forward_step currently consumes `larql-models::load_gguf`
  weights, not vindex. A second change wires the reader path.

- **Server dispatch** routing qwen35* arches through the new
  forward. Third change.

- **F16 / non-Q4_K paths** for hybrid arches. The existing `--f16`
  branch in `convert_cmd.rs` does not have the same guard but isn't
  validated end-to-end either; treated as separate.

- **Parity vs llama.cpp** on the vindex-loaded weights. Gated on
  the reader+dispatcher arc landing first.

- **DeepSeek V4 Flash** (MLA) is parked. Different problem family;
  needs its own change.

## Estimated effort

~600-900 LoC across 4-5 files. Independent of the larger
`inference-qwen35-deltanet` change. Single PR target.
