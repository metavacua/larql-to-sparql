# Delta — inference-gated-deltanet

## ADDED Requirements

### Requirement: Optional lazy-quantized lm_head matmul

The Qwen3.6 forward path SHALL accept a model weight set in which
`lm_head` is held as a quantized
`larql_models::quant::lazy::QuantTensor` rather than a dequantized
`ArcArray2<f32>`. When the quantized form is present, the final
logits MUST be computed via per-row quantized matvec dispatch
(`q4k_row_dot` / `q5k_row_dot` / `q6k_row_dot`) instead of
`Array2::dot`. The dequantized fallback path remains supported for
backward compatibility and synthetic tests.

#### Scenario: Lazy lm_head produces the same argmax as the dequant path
- **WHEN** `load_qwen35_weights_lazy_lm_head` loads the same GGUF
  used by the dequant path and the same prompt is run through
  `qwen35_forward_step`
- **THEN** the argmax of the resulting logits SHALL be identical to
  the dequant path's argmax for the first 5 decode tokens.
<!-- test: larql_inference::attention::qwen35_load::tests::real_gguf_qwen35_lazy_lm_head_argmax_matches_dequant -->

#### Scenario: Lazy lm_head reduces peak process RSS by at least 4 GiB
- **WHEN** `real_gguf_qwen35_bench` runs Qwen3.6-27B Q4_K_S with
  `LARQL_QWEN35_LAZY_LM_HEAD=1`
- **THEN** the recorded peak `VmRSS` SHALL be at least 4 GiB lower
  than the same bench run without the env var.
<!-- test: larql_inference::attention::qwen35_load::tests::real_gguf_qwen35_lazy_lm_head_rss_smaller -->
