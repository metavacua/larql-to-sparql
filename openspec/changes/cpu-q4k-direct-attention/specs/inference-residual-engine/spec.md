## ADDED Requirements

### Requirement: CPU decode-step attention MUST support direct Q4_K × Q8_K matvec for Q/K/V/O

`run_attention_block_with_kv_out_with_cache` SHALL dispatch the
decode-step attention forward through the direct Q4_K × Q8_K matvec
path (`run_attention_block_decode_step_q4k_direct`) when all of the
following hold:

- A `VectorIndex` reference is provided (`vindex: Some(_)`).
- The residual batch is a single row (decode step, `h.shape()[0] == 1`).
- No `shared_kv` donor is supplied for this layer.
- The KV cache is non-empty for this layer (decode step, not prefill).
- The layer's `hidden % 256 == 0` (Q8_K alignment for `h_norm`
  quantisation).
- The layer's `(num_q * head_dim) % 256 == 0` (Q8_K alignment for the
  `attn_out` quantisation prior to the O projection).

All other branches — `kv_cache=None`, `shared_kv=Some(_)`, empty cache
(prefill), multi-row input, missing vindex, non-aligned `hidden_size`,
non-aligned `num_q * head_dim` — SHALL continue to use
`run_attention_block_with_kv_out` (prefill) or
`run_attention_block_decode_step` (decode over the dequant cache).

#### Scenario: Decode-step on Q8_K-aligned dense layer routes to direct path

- **GIVEN** a Q4_K vindex for a model whose `hidden_size` and
  `num_q * head_dim` are both multiples of 256 (e.g., Gemma 3 4B at
  `hidden=2560`, `q_dim=2048`)
- **WHEN** `run_attention_block_with_kv_out_with_cache` is called with
  `vindex=Some(_)`, a populated `KvCache`, and a single-row residual
- **THEN** the call SHALL dispatch through
  `run_attention_block_decode_step_q4k_direct` and SHALL NOT read
  `weights.tensors[arch.attn_q_key(layer)]` or the K/V/O equivalents
  for that step
<!-- test: unbacked -->

#### Scenario: Non-Q8_K-aligned hidden_size routes to weights.tensors path

- **GIVEN** a Q4_K vindex for a model whose `hidden_size` is not a
  multiple of 256 (e.g., Gemma 3 1B at `hidden=1152`)
- **WHEN** `run_attention_block_with_kv_out_with_cache` is called with
  `vindex=Some(_)`, a populated `KvCache`, and a single-row residual
- **THEN** the call SHALL dispatch through
  `run_attention_block_decode_step` over `weights.tensors`; the direct
  Q4_K × Q8_K path is not engaged
<!-- test: unbacked -->

#### Scenario: Decode-step with vindex=None routes to weights.tensors path

- **GIVEN** a `KvCache` populated by a prior prefill
- **WHEN** `run_attention_block_with_kv_out_with_cache` is called with
  `vindex=None` and a single-row residual
- **THEN** the call SHALL dispatch through
  `run_attention_block_decode_step` over `weights.tensors`. This
  preserves backwards compatibility for non-vindex callers and for
  unit tests that don't construct a vindex.
<!-- test: unbacked -->

### Requirement: Direct Q4_K × Q8_K attention output MUST match weights-tensors output within Q8_K activation noise

`run_attention_block_decode_step_q4k_direct` SHALL produce `h_post_attn` row-wise equivalent to `run_attention_block_decode_step` within the Q8_K activation quantisation envelope (≤ 1.5 % relative error per element) for the same `(h, layer, kv_entry, abs_position)`. The returned `(k_concat, v_concat)` MUST match the weights-tensors path within the same envelope. End-to-end CPU generation under the direct-attention path SHALL produce coherent output on the same Gemma 3 4B Q4_K_M vindex that produced coherent output under PR #139.

#### Scenario: Direct path matches weights-tensors path on real Gemma 3 4B vindex

- **GIVEN** a Q4_K vindex for Gemma 3 4B with the attention dequant
  cache populated for layer L by `insert_q4k_layer_tensors`
- **WHEN** the same `(h, layer, kv_entry, abs_position)` is passed to
  `run_attention_block_decode_step_q4k_direct` and
  `run_attention_block_decode_step`
- **THEN** both outputs SHALL agree element-wise within ≤ 1.5 %
  relative error (Q8_K activation noise envelope) on `h_post_attn`,
  `k_concat[-1]`, and `v_concat[-1]`
<!-- test: unbacked -->
