# Tasks — lazy quantized matmul for Qwen3.6

## Phase 1 — lm_head only (~150 LoC)

- [ ] 1.1 Add `QuantTensor` struct in
      `crates/larql-models/src/quant/lazy.rs`. Fields: `Vec<u8>` raw
      bytes, `tensor_type: u32`, `rows: usize`, `cols: usize`.
      Constructors: `QuantTensor::from_raw(bytes, type, rows, cols)`
      and `QuantTensor::from_f32(array)` (fallback for synthetic
      tests; stores values as f32-as-Q8_0-equivalent or just keeps a
      sentinel marker).
- [ ] 1.2 Add `QuantTensor::matvec(&Array1<f32>) -> Array1<f32>`.
      Dispatch per row: Q4_K → `q4k_row_dot`, Q6_K → `q6k_row_dot`,
      Q5_K → `q5k_row_dot`, f32 → ndarray dot.
- [ ] 1.3 Modify `larql_models::load_gguf` to optionally retain raw
      bytes for `output.weight` (controlled by an arg or a parallel
      method `load_gguf_lazy_lm_head`).
- [ ] 1.4 Add `Qwen35Weights::lm_head_quant: Option<QuantTensor>`
      field. Forward dispatch: if `Some`, call `.matvec`; else fall
      back to `lm_head: ArcArray2<f32>`.
- [ ] 1.5 Env-gated test
      `real_gguf_qwen35_lazy_lm_head_diagnostic`: load GGUF with
      lazy lm_head, run one prefill + 1 decode step, assert argmax
      matches the existing dequant path within Q6_K rounding noise.
- [ ] 1.6 Extend `real_gguf_qwen35_bench` to optionally use lazy
      lm_head via `LARQL_QWEN35_LAZY_LM_HEAD=1`. Record RSS + tok/s
      and append to `bench-baseline.md`.
- [ ] 1.7 PR + bench-table update.

## Phase 2 — FFN tensors (~200 LoC, separate change)

(Scoped after Phase 1 lands. The FFN is ~73 % of model RAM, so this
is where the headline RAM number drops to llama.cpp parity.)

## Phase 3 — x86 AVX2 kernels for the quant path (~300 LoC)

(Separate openspec change. Required to close the speed gap on x86;
today's `q4k_row_dot` scalar path may be slower than BLAS f32 on
this host.)

## Validation

- [ ] V.1 `cargo test -p larql-models --lib quant::lazy` — unit
      tests for `QuantTensor` constructors + matvec dispatch.
- [ ] V.2 `LARQL_QWEN35_GGUF=… cargo test … real_gguf_qwen35_lazy_lm_head_diagnostic`
      — argmax parity vs the dequant path.
- [ ] V.3 Bench delta in `bench-baseline.md` shows RSS drop and
      records new tok/s.
- [ ] V.4 `openspec validate qwen35-lazy-quant-matmul --strict`
      passes.
