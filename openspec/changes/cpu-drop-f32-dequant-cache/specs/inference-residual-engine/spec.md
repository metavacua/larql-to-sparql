## ADDED Requirements

### Requirement: CPU chat-completion decode MUST NOT require `weights.tensors` population

`predict_q4k_hidden_with_cache` SHALL skip the
`insert_q4k_layer_tensors` call when every per-layer dispatch is
guaranteed to take a direct Q4_K × Q8_K path — specifically when ALL
of:

- `arch.is_hybrid_moe() == false`,
- `hidden_size % 256 == 0` (Q8_K activation alignment for the FFN +
  attention input),
- `(num_q * head_dim) % 256 == 0` (Q8_K activation alignment for the
  attention O-projection input),
- no layer reports `arch.kv_shared_source_layer(l).is_some()`.

For Gemma 3 4B Q4_K_M these conditions all hold; the f32 dequant cache
that `insert_q4k_layer_tensors` would otherwise populate (~10 GB of
attn+FFN dequant) is never allocated. RSS drops from ~24.6 GB to
~10.3 GB on this model.

Models where any condition fails (Gemma 3 1B with `hidden=1152`,
Gemma 4 family's hybrid-MoE / cross-layer K/V share, capture-enabled
attention) SHALL still call `insert_q4k_layer_tensors` so their
WeightFfn / multi-row attention fallbacks continue to work.

#### Scenario: Gemma 3 4B chat path skips insert_q4k_layer_tensors

- **GIVEN** a Q4_K vindex for Gemma 3 4B (`hidden=2560`,
  `q_dim=2048`, non-MoE, no cross-layer K/V share)
- **WHEN** `predict_q4k_hidden_with_cache` is invoked for a
  chat-completion (prefill seq>=1 with empty cache, then decode
  with seq=1)
- **THEN** `weights.tensors` SHALL remain empty for every layer;
  the function SHALL produce a coherent forward result via
  `Q4kDirectFfn` + `run_attention_block_prefill_q4k_direct` for
  prefill and `Q4kDirectFfn` + `run_attention_block_decode_step_q4k_direct`
  for decode
<!-- test: unbacked -->

#### Scenario: Non-aligned hidden_size keeps the dequant cache

- **GIVEN** a Q4_K vindex for a model whose `hidden_size` is not a
  multiple of 256 (Gemma 3 1B at `hidden=1152`)
- **WHEN** `predict_q4k_hidden_with_cache` is invoked
- **THEN** `insert_q4k_layer_tensors` SHALL still be called on every
  layer; the fallback `WeightFfn` / `run_attention_block_with_kv_out`
  path SHALL receive the f32 tensors it needs
<!-- test: unbacked -->

### Requirement: Multi-row Q4_K × Q8_K prefill attention SHALL exist

A function `run_attention_block_prefill_q4k_direct(weights, index, h, layer)` SHALL produce the same `(h_post_attn, (k_rope, v_full))` semantics as `run_attention_block_with_kv_out(weights, h, layer, false, None)` for the layers that satisfy the Q8_K alignment guards. It SHALL NOT read from `weights.tensors` — Q/K/V/O matmuls flow through per-row `q4k_q8k_matvec_into` / `q6k_q8k_matvec_into` on the vindex bytes via `index.attn_q4k_layer_data(layer)`.

The output SHALL match the cache-based path within Q8_K activation
noise (≤ 1.5 % relative error per element) on the same input, for any
production seq length the chat completion path emits.

#### Scenario: Prefill direct path produces coherent forward residual

- **GIVEN** Gemma 3 4B Q4_K_M vindex and a 14-token chat-completion
  prompt
- **WHEN** the server prefills via `run_attention_block_prefill_q4k_direct`
- **THEN** the residual stream after each layer SHALL track the
  cache-based path within Q8_K noise; the lm_head over the last row
  SHALL produce the same top-1 token across a 40-token continuation
<!-- test: unbacked -->

### Requirement: lm_head f32 dequant SHALL be skipped when `lm_head_quant` is populated

The vindex loader SHALL skip the f32 dequantisation of
`lm_head_q4.bin` when `hidden_size % 256 == 0` and the
`QuantTensor::from_raw` of those bytes succeeds — in that case
`weights.lm_head_quant` is populated and every production lm_head
caller routes through it (PR #144). The f32 `weights.lm_head` field
gets the tied `embed` clone as a placeholder so the field stays
non-None for any unreached fallback code.

Non-aligned `hidden_size` SHALL keep the f32 dequant so the BLAS
fallback in `project_lm_head_last_row` continues to work.

#### Scenario: Gemma 3 4B vindex load skips lm_head f32 dequant

- **GIVEN** the Gemma 3 4B Q4_K_M vindex (`hidden=2560`,
  `vocab_size=262144`)
- **WHEN** `load_model_weights_q4k` runs
- **THEN** `weights.lm_head_quant` SHALL be `Some`; `weights.lm_head`
  SHALL be the (small) embed-clone placeholder, not the
  2.6 GB f32 dequant array
<!-- test: unbacked -->
