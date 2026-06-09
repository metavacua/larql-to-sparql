## ADDED Requirements

### Requirement: GGUF to vindex Q4_K conversion SHALL support hybrid SSM/DeltaNet architectures

The vindex conversion pipeline SHALL produce a structurally complete
vindex when invoked with `larql convert gguf-to-vindex --level
inference --quant q4k` on the `qwen35moe` architecture family
(Qwen 3.6 35B-A3B and any future variant carrying
`full_attention_interval != 0` plus per-expert MoE weights).
"Structurally complete" SHALL mean:

- Every tensor surfaced by `Qwen35MoeArch`'s per-layer key
  accessors (DeltaNet `ssm_*` / `attn_qkv` / `attn_gate` for linear
  layers; standard Q/K/V/O for full-attention layers; per-expert
  gate/up/down for FFN) is emitted into one of the per-tensor
  manifests with a matching offset and length.
- No tensor is silently dropped.
- `index.json` records the architecture's `full_attention_interval`,
  `ssm_state_size`, `ssm_inner_size`, `ssm_dt_rank`,
  `ssm_group_count`, `ssm_conv_kernel`, and (when present)
  `rope_dimension_sections`.

The hard-fail guard at `convert_cmd.rs:515` SHALL be removed by the
PR that lands the writer support.

#### Scenario: Qwen 3.6 35B-A3B Q4_K_M converts to a structurally complete vindex

- **GIVEN** `/tank/ai/Qwen/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
  (40 layers, `full_attention_interval=4`, 256 experts top-8) with
  `tokenizer.json` and `config.json` in the same directory
- **WHEN** `larql convert gguf-to-vindex --level inference --quant q4k
  --output /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex <gguf>` runs
- **THEN** the command SHALL exit 0 and the output directory SHALL
  contain `deltanet_weights_q4k.bin` (30 linear layers' worth of
  Q4_K SSM tensors), `attn_weights_q4k.bin` (10 full-attn layers'
  Q/K/V/O), `layers/layer_{00..39}.weights` (each with 256 expert
  entries × 3 projections), `lm_head_q4.bin`, `norms.bin`,
  `embed*.bin`, and an `index.json` reflecting the hybrid layout
<!-- test: unbacked -->

### Requirement: PerExpert MoE layout SHALL be supported by the per-layer Q4_K writer

The vindex MoE per-layer writer SHALL accept architectures with
`is_moe() == true` and `expert_format() == ExpertFormat::PerExpert`
in addition to the existing
`is_hybrid_moe() && expert_format() == ExpertFormat::PackedBF16`
case (Gemma 4 26B A4B's path).

For each layer, the writer SHALL iterate experts `0..num_experts`
and emit per-expert `gate_proj`, `up_proj`, `down_proj` quantised
to Q4_K under `layers/layer_{L:02}.weights` using the existing
`write_layer_weights` machinery.

#### Scenario: PerExpert MoE writes 256 entries per layer

- **GIVEN** a `qwen35moe` arch with `num_experts=256`,
  `expert_format=PerExpert`, `num_layers=40`,
  `moe_intermediate_size=512`, `hidden_size=2048`
- **WHEN** `write_per_layer_moe_q4k` runs
- **THEN** each `layers/layer_{L:02}.weights` SHALL contain
  `256 × 3 = 768` Q4_K entries (gate, up, down per expert),
  with sizes matching `quantize_q4_k(moe_intermediate × hidden)`
  for gate/up and `quantize_q4_k(hidden × moe_intermediate)` for
  down
<!-- test: unbacked -->

### Requirement: Linear-attention layers SHALL be skipped by the standard Q/K/V/O writer

The standard `attn::write_attn_weights_q4k` loop SHALL skip layers
where `arch.is_linear_attention_layer(layer) == true`. Those layers'
weights live in `deltanet_weights_q4k.bin` (written by the new
DeltaNet writer); attempting to enumerate Q/K/V/O for them would
produce `None` and a corrupt manifest.

#### Scenario: 40-layer hybrid arch emits 10 full-attn manifests

- **GIVEN** a 40-layer Qwen 3.6 arch with `full_attention_interval=4`
  (layers 3, 7, 11, …, 39 are full attention)
- **WHEN** `write_attn_weights_q4k` runs
- **THEN** `attn_weights_q4k.bin` manifest SHALL have exactly 10
  layer-worths of entries (40 entries total: Q, K, V, O per
  full-attn layer), not 160
<!-- test: unbacked -->
