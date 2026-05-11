# larql-metal vs llama.cpp — Metal kernel architecture comparison

**Date**: 2026-05-09
**Methodology**: Metal System Trace via `xctrace record --template "Metal System Trace"` on each engine running matched workloads on Gemma 3 4B Q4_K, plus kernel-symbol extraction from `libggml-metal.dylib` (`strings | grep ^kernel_`). Trace files saved at `/tmp/larql-trace/{larql,ollama}.trace`. The dylib symbol diff turned out more actionable than the trace itself because the default Metal System Trace template doesn't enable Shader Timeline (per-kernel GPU timing); the kernel inventory is what tells you *which* kernels each engine fires, and that's the load-bearing question for closing the gap.

## Three architectural differences that explain the perf gap

### 1. Flash attention — llama.cpp has it, larql doesn't

| | larql | llama.cpp |
|---|---|---|
| Attention pipeline | `qk_norm_rope_fused` + `kv_append_attend_fused` + Wo (3 dispatches/layer; `attn_fused` opt-in tried, regressed) | **`kernel_flash_attn_ext_vec_reduce`** (1 fused dispatch/layer) |
| Decode cost on Gemma 3 4B | attn bucket = 3.3 ms (34% of GPU fwd) | comparable but tighter pipeline |

llama.cpp implements GPU-friendly Flash Attention — Q·K → softmax → ·V processed block-by-block in registers without materialising the full attention matrix. Our `attn_fused` shader exists as an opt-in but the first attempt (2026-05-01) regressed −1.45 ms because the merge collapsed TG count 12→8 (parallelism loss exceeded dispatch saving). The multi-TG-per-head retry (split QKV+attend across 2 TGs/head, keep total ≥12 TGs) is the roadmap entry **D-ATTN-MTG** and is the one untried approach left for the decode-side gap.

**Estimated impact** on closing larql's 1.17× decode gap: the entire attention bucket (3.3 ms) gets compacted to whatever flash-attn's per-token cost is — memory `project_gpu_forward_gap.md` puts the recoverable savings at ~0.85 ms/tok = +5–8 tok/s.

### 2. `simdgroup_matrix` for prefill matmul — llama.cpp uses Apple intrinsics, we don't

`llama-bench` announces it on init:
```
ggml_metal_device_init: simdgroup matrix mul. = true
```

| | larql | llama.cpp |
|---|---|---|
| Prefill matmul | `q4k_matmul` — scalar f32 accumulators in a 2D dispatch grid (`ROWS_PER_TG=4, COLS_PER_TG=4, THREADS_PER_TG=128`) | `kernel_mul_mm_*_*` — Apple `simdgroup_matrix` intrinsics (8×8 register-tile fused-multiply-add) |
| End-to-end on Gemma 3 4B | Tried wiring twice (2026-04-28, 2026-05-09) — both regressed 5–10% on long prompts (L1-thrash on `[seq_len × hidden]` X working set) | Production prefill path; closes the 4–14× prefill gap |

This is what the prefill 12.5× gap is fundamentally about. Our `q4k_matmul` and llama.cpp's `kernel_mul_mm_*` solve the same problem with the same input/output shape, but their kernel uses Apple's `simdgroup_matrix` 8×8 register-tile intrinsics — the same hardware feature that's also load-bearing for their `kernel_flash_attn_ext_*`. A scalar-accumulator matmul can't hold the working set in registers on long prompts, which is why our wiring loses both attempts. Closing this gap means writing a new matmul kernel that uses `simdgroup_matrix` (and only `simdgroup_matrix` — there's no shortcut), or accepting the current ceiling.

The `MTLGPUFamilyMetal3` device-family check that gates these intrinsics is satisfied on M3 Max; this is purely a kernel-implementation gap, not a hardware limit.

**Estimated impact** on closing the prefill gap: 4–14× (closes the gap entirely on long prompts, and the gap *is* the matmul kernel).

### 3. RMS-norm pre-fusion — different fusion direction than ours

| | larql | llama.cpp |
|---|---|---|
| RMS norm kernels | `qk_norm`, `rms_norm` (separate dispatches), `qk_norm_rope_fused` (norm + RoPE), `post_attn_residual_norm_store`, `post_ffn_norm_residual_add` | `kernel_rms_norm_mul_f32` (norm + scalar mul), `kernel_rms_norm_mul_add_f32` (norm + scalar mul + add), `kernel_rope_norm_*`, `kernel_rope_neox_*` |

llama.cpp pre-fuses RMS norm with the immediately-following scalar op (multiply by weight, multiply-and-add for residual). We had a deeper norm-into-matmul fusion (`q4k_q6k_qkv_proj_normed`) — but that one cost more in operand reread than it saved in dispatch (this is what the 2026-05-09 QKV defuse, see ADR-016, fixed). The llama.cpp pattern is **different shape**: norm + outside-the-matmul scalar op, not norm + matmul. Worth exploring as a smaller follow-up — a `rms_norm_mul_f32`-style shader that fuses norm + the per-element weight-multiply but leaves the heavy matmul standalone.

**Estimated impact**: ~0.1 ms decode per layer × 34 = a few hundred microseconds, not a major lever.

## Other inventory differences (architecturally equivalent)

| Op | larql | llama.cpp |
|---|---|---|
| Decode matvec (multi-row) | `q4k_matvec_8sg` (8 simdgroups × 32 threads, 8 rows/TG) | `kernel_mul_mv_ext_*_r1_N` (multi-row variant) |
| MoE expert dispatch | custom `moe_dispatch` | `kernel_mul_mm_id_*`, `kernel_mul_mv_id_*` (built-in) |
| Top-k / argmax | `f32_gemv_topk1`, `f32_gemv_topk` (fused score+topk) | `kernel_argmax_*` (separate) |
| Buffer copies | Metal API calls | `kernel_cpy_*_*` (templated kernel) |
| Softmax | inside attention kernel | `kernel_soft_max_*` (separate; rolled into flash-attn for production path) |
| RoPE | `rope`, `qk_norm_rope_fused` | `kernel_rope_norm`, `kernel_rope_neox`, `kernel_rope_multi`, `kernel_rope_vision` |

## Methodology — how to reproduce

```bash
# 1. Capture larql trace
xcrun xctrace record \
  --template "Metal System Trace" \
  --output /tmp/larql.trace \
  --launch -- ./target/release/larql run \
    ~/.cache/larql/local/gemma3-4b-q4k-v2.vindex \
    "The capital of France is" -n 30 --metal

# 2. Capture ollama trace (attach to runner PID while sending request)
ollama run gemma3:4b "warmup" </dev/null >/dev/null
RUNNER_PID=$(pgrep -f "ollama runner" | head -1)
(sleep 2; ollama run gemma3:4b "The capital of France is. Answer in 30 words." </dev/null) &
xcrun xctrace record \
  --template "Metal System Trace" \
  --output /tmp/ollama.trace \
  --attach "$RUNNER_PID" --time-limit 12s

# 3. Pull llama.cpp kernel inventory
strings /opt/homebrew/Cellar/llama.cpp/*/lib/libggml-metal.0.dylib \
  | grep -E "^kernel_" | sort -u

# 4. Pull larql kernel inventory
ls crates/larql-compute/src/metal/shaders/*.rs \
  | xargs -I{} basename {} .rs | sort
```

For per-kernel GPU timing the default Metal System Trace template is **not enough** — Shader Timeline must be enabled (Edit Active Template in Instruments.app GUI; can't be set via xctrace CLI). Without Shader Timeline the trace contains encoder-create events on the CPU side (`metal-application-encoders-list`) and Metal driver intervals (`metal-driver-intervals`), but `metal-shader-profiler-intervals` returns 0 rows. The dylib symbol approach above works without that GUI step and gives the kernel-name diff directly.

## Roadmap implications

The three gaps map to three roadmap items:

| Gap | ROADMAP entry | Status | Effort | Estimated impact |
|---|---|---|---|---|
| Flash attention | **D-ATTN-MTG** (multi-TG `attn_fused` retry) | Open. First attempt regressed 2026-05-01 (TG-count collapse). Multi-TG-per-head variant untried. | ~2 days kernel work | +0.2–0.4 ms decode, +5–8 tok/s |
| `simdgroup_matrix` matmul | **D-PREFILL-MM2** (new — replaces the closed D-PREFILL-MM track) | Not started. Requires a new `q4k_matmul` shader using Apple intrinsics. | Multi-day kernel research; needs M3 Max-specific tuning. | Closes 4–14× prefill gap |
| RMS-norm pre-fusion | (no entry yet) | Smaller follow-up. Lower priority. | ~half day | ~0.1 ms decode |

The previously-tracked **D-PREFILL-MM** (wire `q4k_matmul` into prefill sites) was closed 2026-05-09 as twice-falsified. This comparison establishes that the *kernel itself* needs to be rewritten — wiring the existing scalar-accumulator kernel will never beat per-position matvec on long prompts because the L1 working-set thrash is structural to the kernel shape, not the wiring. **D-PREFILL-MM2** is the renamed track for the kernel rewrite.

## Why this comparison is more useful than the Metal traces

The original goal was a per-kernel timing comparison from the trace files. That requires Shader Timeline data, which requires the Instruments.app GUI to enable, which means the comparison can't be done programmatically in a session. But the **kernel-name inventory** is what answers the load-bearing question: "what does llama.cpp do that we don't?" The dylib symbols are sufficient for that; the trace would have only added per-kernel timings on top of the same architectural diff.

If you want to drill into specific kernel shapes / dispatches in Instruments.app, the trace files at `/tmp/larql-trace/larql.trace` and `/tmp/larql-trace/ollama.trace` (22 MB / 23 MB) are saved. Open in Instruments, **Edit > Edit Active Template**, enable Shader Timeline, re-record for the full view.

## Related

- `crates/larql-compute/PERFORMANCE.md` — current state, recent-changes table.
- `crates/larql-compute/ROADMAP.md` "P0: Production gap closers" — D-ATTN-MTG and D-PREFILL-MM2.
- `crates/larql-compute/docs/adr/015-isolated-vs-batched-kernel-perf.md` — the iso-vs-batched lesson; `kernel_mul_mm_*` would *change the batched measurement*, which is exactly what the ADR points at.
- `crates/larql-compute/docs/adr/016-defused-rms-norm-qkv.md` — why we defused norm+QKV (different direction from llama.cpp's norm+mul fusion).
- `project_prefill_matmul_falsified.md` (memory) — D-PREFILL-MM falsification record; complements §2 above.
- `project_gpu_forward_gap.md` (memory) — the 1.30× → 1.18× → 1.17× progression and the pre-comparison bucketing.
