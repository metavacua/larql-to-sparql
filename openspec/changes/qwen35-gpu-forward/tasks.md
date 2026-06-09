# Tasks — Qwen3.6 GPU forward

## E.1 — single-matvec PoC (~150 LoC)

- [x] E.1.1 Add `pub backend: Option<Arc<dyn larql_compute::backend::QuantMatVec
        + Send + Sync>>` to `Qwen35Weights`. Default `None` keeps
      the CPU lazy path. Bench harness constructs
      `CudaBackend::new()?` when `LARQL_QWEN35_GPU=1`.
- [x] E.1.2 Extend `QuantTensor::matvec` (or add
      `matvec_with_backend`) that takes an optional `&dyn
      QuantMatVec`. If `Some`, calls `backend.quant_matvec(format,
      &self.data, x, rows, cols)` and falls back to the existing
      rayon CPU path on `None` return.
- [x] E.1.3 In `qwen35_forward_step`, when computing the final
      `lm_head` matvec, pass `weights.backend.as_deref()` through.
      Default behaviour unchanged.
- [x] E.1.4 Map our `tensor_type` (u32 ggml id) to
      `larql_compute::QuantFormat`. Helper in `quant/lazy.rs`.
- [x] E.1.5 Env-gated test
      `real_gguf_qwen35_gpu_lm_head_diagnostic` — load lazy lm_head,
      construct `CudaBackend`, run prefill + 1 decode step, assert
      argmax matches dequant baseline.
- [x] E.1.6 Extend `real_gguf_qwen35_bench` to use the GPU backend
      when `LARQL_QWEN35_GPU=1`. Print kernel-vs-fallback dispatch
      counts. Record bench delta in `bench-baseline.md`.

## E.2 — FFN on GPU (~50 LoC plumbing)

Once E.1 lands, the rest is just dispatching the FFN tensors the same
way. Already plumbed via Phase 2's lazy lookup.

- [x] E.2.1 `swiglu_ffn_lazy` takes `backend: Option<&dyn QuantMatVec>`,
      passes through to `QuantTensor::matvec_with_backend` for each
      of gate / up / down.
- [x] E.2.2 Bench: actual 0.28 t/s decode; DeltaNet recurrence remains
      the bottleneck. Recorded in
      `openspec/changes/inference-qwen35-deltanet/bench-baseline.md`.

## E.3 — Attn projections on GPU (~50 LoC plumbing)

- [x] E.3.1 Dispatch DeltaNet `attn_qkv`/`attn_gate`/`ssm_out` and
      full-attn `attn_q/k/v/o` matvecs through the same backend.
- [x] E.3.2 Bench: actual 0.33 t/s decode; still recurrence-bound.
      Recorded in
      `openspec/changes/inference-qwen35-deltanet/bench-baseline.md`.

## E.4 — DeltaNet recurrence + Conv1D CUDA kernels (~600 LoC)

- [x] E.4.1 CUDA DeltaNet recurrence kernel — one CUDA block per
      V head, device-cached recurrent state, decay-first algorithm
      matching `ggml_compute_forward_gated_delta_net_one_chunk`.
      First pass uses global state memory; shared-memory tiling/fusion
      remains follow-up because launch/sync overhead dominates.
- [x] E.4.2 CUDA causal Conv1D kernel — depthwise Conv1D with
      device-cached state shift-and-insert for 4-tap × 10240 channels.
- [x] E.4.3 Per-head RMSNorm + L2-norm on GPU (small reductions
      that don't need their own kernel; can fold into conv or
      delta entry).
- [ ] E.4.4 Bench: expect ≥ 10 t/s decode. Actual after E.4.3 is
      still 0.33 t/s decode; target remains open because per-layer
      launch/sync and host↔device round-trips dominate.

## E.5 — Full softmax-attention on GPU (~200 LoC plumbing)

- [ ] E.5.1 Wire existing `cuda/attn.rs` (or `attn_tree.rs`) into
      the full-attn block forward.
- [ ] E.5.2 Bench: expect ≥ 15 t/s decode.

## E.6 — Device-resident weights + KV cache (~400 LoC)

- [ ] E.6.1 Upload all Q4_K/Q6_K weight bytes to VRAM once at load.
      Keep host bytes only when an explicit `--cpu-fallback` flag is
      set.
- [ ] E.6.2 KV cache buffers live in VRAM.
- [ ] E.6.3 CUDA Graphs for the per-token compute path.
- [ ] E.6.4 Bench: expect ≥ 30 t/s decode (within 2× of llama.cpp
      GPU).

## E.6.A — Foundations: fused post-projection chain (~700 LoC)

- [x] E.6.A.1 Loader fix: `ssm_conv1d` uses `as_standard_layout()` so
      `as_slice()` is `Some`, enabling the previously-dormant GPU
      conv1d kernel.
- [x] E.6.A.2 New PTX module `cuda::qwen35_block` with reshape +
      silu mini-kernels.
- [x] E.6.A.3 Trait method `qwen35_deltanet_postproj_step` (default
      `None`); CudaBackend implementation chains all five existing
      deltanet kernels on a single stream with one sync at exit.
- [x] E.6.A.4 Unit tests at Qwen3.6 production shapes (head_v_dim=128,
      n_v_heads=48, n_k_heads=16) for both reshape and recurrence.
- [x] E.6.A.5 Inference-side fast-path in `deltanet_block_step` gated
      by `LARQL_QWEN35_E6A_FUSED=1`; default OFF until multi-token
      parity is sorted.
- [ ] E.6.A.6 Diagnose multi-position parity drift. Single-call
      parity is bit-near-exact (~7e-9 vs CPU at production shape);
      across 9 prompt + 5 decode positions the residual stream
      diverges enough to flip argmax. Suspected cause: fp32 reduction
      order in L2 / rms_norm reductions compounded through the
      recurrent state's per-position update cycle.

## Validation

- [x] V.1 `cargo test -p larql-inference --release --lib
      real_gguf_qwen35_token_diff_vs_llama_cpp` under
      `LARQL_QWEN35_GPU=1` passes (GT rank 0 every step).
- [x] V.2 `openspec validate qwen35-gpu-forward --strict` passes.
- [x] V.3 Each phase's PR includes its bench delta in
      `openspec/changes/inference-qwen35-deltanet/bench-baseline.md`.
