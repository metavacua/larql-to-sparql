## Why

DSv4-Flash currently dequantizes every weight to f32 at load time
(`DsV4LayerWeightStorage` holds `Array2/Array3<f32>`). At ~26 GB f32
per layer × 43 layers ≈ 1.1 TB, the full model can't fit in RAM, so
`dsv4_streaming_model_forward_cached` reloads and re-dequantizes each
layer's weights from GGUF **on every token** and drops them between
layers. The 2026-05-25 RTX 4090 benchmark (`dsv4_bench_cpu_vs_cuda`)
measured ~90 s per decode step for only 3 layers — that wall time is
the streaming reload + Q4_K dequant, not compute. The completed GPU
push (PRs #339–#367) routed every matmul through cuBLAS but delivered
0.98× (2 % slower than CPU) precisely because the per-token dequant
dominates and the matmul is noise next to it.

Keeping weights in their compact quantized form resident in RAM —
the same `QuantTensor` machinery qwen35 already uses — shrinks the
working set to ~161 GB (fits a 256 GB host), removes the per-token
reload, and unblocks the project's driving design: FFN computed on
CPU against RAM-resident quantized weights, attention offloaded to
the 4090.

## What Changes

- DSv4 layer/FFN storage gains `Option<QuantTensor>` fields **alongside**
  the existing f32 arrays (dual representation; f32 stays as a
  fallback). No fields removed.
- The GGUF reader stops calling `dequantize()` eagerly for the large
  matmul weights; it hands the raw Q4_K/Q5_K/Q6_K bytes to
  `QuantTensor::from_raw` (built directly from GGUF — **no vindex
  conversion required for v1**; zero-copy mmap is a later optimization).
- The per-layer forward dispatches each matmul: quant present →
  `QuantTensor::matvec` / `matmul` (per-row fused kernel, no full
  dequant); else the existing `dot_proj_gpu(&x, &w, backend)` f32 path.
  MoE expert dispatch uses `QuantTensor::expert_slice` (zero-copy)
  instead of re-dequantizing per expert.
- A non-streaming resident forward loads all layers' quantized weights
  once and keeps them. This replaces the per-token reload that
  dominates decode wall time. **BREAKING** for the streaming-forward
  callers only in that the resident path is a new entry point;
  the streaming functions remain for the model-exceeds-RAM case.
- Hybrid backend placement: attention matmuls route to the GPU
  (`Some(cuda)` — small, dense after dequant), FFN/MoE matvecs run on
  CPU (`None`) against the resident quantized weights — the
  CPU-FFN / GPU-attention split.

## Capabilities

### New Capabilities
- `dsv4-quant-residency`: DSv4-Flash holds its matmul weights as
  resident `QuantTensor`s (quantized bytes in RAM) rather than
  load-time-dequantized f32, runs the forward against them via the
  lazy-quant matvec/matmul path, and loads all layers once instead of
  per-token streaming. Covers the dual-storage representation, the
  quant-aware forward dispatch, and the resident (non-streaming)
  model forward.

### Modified Capabilities
None. The existing `deploy-cpu-gpu-split` capability covers Docker
deployment *topology* (container split), not in-process backend
dispatch — so the DSv4 hybrid placement (FFN-on-CPU / attention-on-GPU
*within one process*) is a requirement of the new
`dsv4-quant-residency` capability, not a change to the deployment spec.

## Impact

- **Code**: `crates/larql-inference/src/attention/dsv4_storage.rs`,
  `dsv4_storage_build.rs`, `dsv4_gguf_reader.rs`,
  `dsv4_full_loader.rs`, `dsv4_per_layer.rs`, `dsv4_attn_block*.rs`,
  `dsv4_ffn_block.rs`, `dsv4_moe_dispatch.rs`,
  `dsv4_compressor_prefill.rs`, `dsv4_indexer.rs`,
  `dsv4_streaming_model_forward.rs` / `dsv4_model_forward.rs`,
  `dsv4_generate.rs`.
- **Dependencies**: `larql-models` `QuantTensor`
  (`quant/lazy.rs`) — reused, not modified. `larql-compute`
  `ComputeBackend` weight cache (PR #368) becomes useful for the
  resident attention weights.
- **Numerics**: quant-vs-f32-dequant introduces small per-element
  differences (per-row Q4_K×Q8_K dot ≠ f32 dequant + BLAS). Parity
  tests must be tolerance-based, not bit-exact.
- **Memory**: ~161 GB resident (Q4_K) vs the prior streaming peak of
  ~26 GB (one layer at a time). Requires a host with enough RAM;
  the streaming path stays available for model-exceeds-RAM.
- **No external API change**; the GGUF on-disk format is unchanged.
