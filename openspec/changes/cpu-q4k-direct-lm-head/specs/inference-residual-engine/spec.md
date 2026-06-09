## ADDED Requirements

### Requirement: lm_head projection MUST prefer the direct Q4_K × Q8_K matvec path when `lm_head_quant` is populated

The final logit projection (`hidden → vocab`) SHALL dispatch through `QuantTensor::matvec` when `weights.lm_head_quant` is `Some`, and SHALL fall back to the f32 BLAS GEMV against `weights.lm_head` when it is `None`. Both `full_vocab_probs` (chat / sampling path) and `hidden_to_raw_logits` (MoE / constrained path) MUST use a single shared helper so they pick up the direct path together. Models whose `hidden_size` is not a multiple of 256 (Gemma 3 1B at `hidden=1152`) keep using the f32 path — `lm_head_quant` is left as `None` by the vindex loader for those.

#### Scenario: Vindex-loaded Gemma 3 4B uses direct lm_head matvec

- **GIVEN** a Q4_K vindex for Gemma 3 4B (`vocab=262144`, `hidden=2560`)
  loaded via `load_model_weights_q4k`
- **WHEN** the vindex loader trims the on-disk bytes to logical
  `vocab_size` rows and calls `QuantTensor::from_raw`
- **THEN** `weights.lm_head_quant` SHALL be `Some(_)`; AND
  `full_vocab_probs` / `hidden_to_raw_logits` SHALL dispatch the
  vocab × hidden matvec through `QuantTensor::matvec` rather than
  `dot_proj(&_, &weights.lm_head)`
<!-- test: unbacked -->

#### Scenario: Non-Q8_K-aligned hidden_size falls back to f32 BLAS

- **GIVEN** a Q4_K vindex for a model whose `hidden_size` is not a
  multiple of 256 (e.g., Gemma 3 1B at `hidden=1152`)
- **WHEN** the vindex loader processes `lm_head_q4.bin`
- **THEN** `weights.lm_head_quant` SHALL remain `None`; AND the lm_head
  projection SHALL use the f32 BLAS GEMV against `weights.lm_head` —
  output remains coherent end-to-end
<!-- test: unbacked -->

#### Scenario: Vindex loader trims trailing vocab-padding rows

- **GIVEN** a `lm_head_q4.bin` whose byte count corresponds to a
  vocab dimension padded beyond `config.vocab_size` (Gemma 3 4B unsloth:
  262208 stored, 262144 logical)
- **WHEN** the loader constructs `QuantTensor::from_raw`
- **THEN** it SHALL trim the byte buffer to exactly `vocab_size *
  row_bytes` so the QuantTensor's `rows × cols` matches the logical
  vocab — same truncation as the f32 dequant path established in #137
<!-- test: unbacked -->
