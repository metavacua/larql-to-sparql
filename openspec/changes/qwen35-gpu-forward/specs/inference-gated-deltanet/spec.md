# Delta — inference-gated-deltanet

## ADDED Requirements

### Requirement: Optional GPU-dispatched matvec for Qwen3.6 forward

The Qwen3.6 forward path SHALL accept an optional
`larql_compute::backend::QuantMatVec` implementation (typically a
`CudaBackend`) attached to `Qwen35Weights`. When present, every
`QuantTensor::matvec` call along the forward MUST first attempt the
backend's `quant_matvec(format, weights, x, rows, hidden)` and fall
back to the CPU rayon path only when that returns `None`. The
dequantised f32 path remains supported when no backend is attached
and when `QuantTensor` is `None` for a given weight.

#### Scenario: GPU lm_head matvec produces argmax identical to dequant baseline
- **WHEN** `real_gguf_qwen35_token_diff_vs_llama_cpp` runs with
  `LARQL_QWEN35_GPU=1 LARQL_QWEN35_LAZY_LM_HEAD=1`
- **THEN** the argmax of the resulting logits SHALL match the dequant
  baseline (and llama.cpp's `<think>\n\n</think>\n\nHello` continuation)
  at every step of the first 5 decoded tokens.
<!-- test: larql_inference::attention::qwen35_load::tests::real_gguf_qwen35_gpu_lm_head_argmax_matches_dequant -->
