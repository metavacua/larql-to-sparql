# ADR-016: Defuse RMS norm + QKV — fusion can be net-negative when the fused kernel rereads operands

**Status**: Accepted
**Date**: 2026-05-09
**Context**: The `q4k_q6k_qkv_proj_normed` kernel (added during the May 2026 dispatch-fusion wave) rolled the input RMS norm into Phase 1 of the QKV matmul, saving 1 dispatch per layer (~0.24 ms/tok across 34 layers). End-to-end A/B 2026-05-09 showed the fusion was a **net regression**: defusing recovers +1.6–1.8 tok/s on Gemma 3 4B.

## Decision

Default for Gemma 3/4 layers (mixed Q4_K Q/K + Q6_K V, RMS norm, no bias) is the defused chain:

1. `encode_q4k_input_norm` — separate `rms_norm` dispatch on the hidden state, writing pre-normalised X to `bufs.norm_out`.
2. `encode_q4k_qkv` — non-fused `q4k_q6k_qkv_proj` reads the pre-normalised X.

The fused `q4k_q6k_qkv_proj_normed` kernel and `encode_normed_q4k_q6k_qkv` dispatcher are retained behind `LARQL_QKV_FUSED=1` as an opt-in fallback for benchmarking and future-hardware experiments.

## Why the fusion lost

The fused kernel's TG geometry is 4 simdgroups × 32 threads = 128 threads, producing 4 output rows per TG (a mix of Q, K, V depending on `tg_id`). Phase 1 (cooperative RMS reduction over H) reads H once across all 128 threads. Phase 2 (matvec) has each of the **4 simdgroups independently re-traverse H + norm_w** with different stride patterns:

- Q4_K Q/K rows: 16 contiguous floats per superblock, indexed by `j*32 + sh*16`.
- Q6_K V rows: 16 non-contiguous floats per superblock at offsets {0..3, 64..67, 128..131, 192..195}.

The two stride patterns can't share a register tile across simdgroups, so the same 2560-element H + norm_w arrays are reread once per simdgroup × 4 simdgroups = **3 redundant device-memory traversals per TG**. Per-kernel batched diag:

| Kernel | bat ms/layer | × 34 | GB/s |
|---|---|---|---|
| `q4k_q6k_qkv_proj_normed` (fused) | 0.135 | 4.59 ms | 199 |
| `q4k_q6k_qkv_proj` (non-fused) + `rms_norm` dispatch | 0.092 + 0.007 ≈ 0.099 | 3.37 ms | 287 |

The 88 GB/s gap (199 → 287) corresponds to ~30% of the H + norm_w traffic getting cut once the inner matvec doesn't have to recompute the per-row normalised input four times.

## End-to-end A/B (2026-05-09)

`larql bench --warmup 8 -n 100`, M3 Max, gemma3-4b-q4k-v2:

| config | tok/s | mean ms/tok | GPU fwd | output |
|---|---|---|---|---|
| baseline 1 (fused, default) | 84.8 | 11.79 | 11.79 | "Paris" ✓ |
| **defused** | **86.5** | **11.57** | **11.52** | "Paris" ✓ |
| baseline 2 (drift check) | 85.0 | 11.77 | 11.86 | "Paris" ✓ |

Drift between baselines: 0.02 ms (well below signal). **Defuse delta: +1.6 tok/s, −0.30 ms/tok GPU fwd.**

Side-by-side post-promotion (warmup 8, n 100, ollama gemma3:4b on same machine):

| | tok/s | ms/tok | gap |
|---|---|---|---|
| larql-metal (defused default) | 88.1 | 11.35 | — |
| ollama gemma3:4b | 103.2 | 9.69 | 1.17× |

## Magnitude undershoot — see ADR-015

Per-kernel diag predicted **−1.22 ms/tok end-to-end** (1.46 ms kernel saving − 0.24 ms dispatch overhead added back). End-to-end measured **−0.22 ms/tok** — direction matched, magnitude was 18% of prediction. ADR-015 § "Magnitude can compress 4×" pins the lesson: bandwidth-headroom wins compress when the surrounding pipeline is already saturating the same bus.

## When dispatch fusion is the wrong call

Two known failure modes for compute-kernel fusion at this point:

1. **TG-count collapse** (ADR-015, `attn_fused`, 2026-05-01): the fused kernel's larger register footprint forces fewer TGs/dispatch than the unfused chain, losing parallelism that exceeds the dispatch saving. Predicted by examining `MTLLibrary.functionInfo.maxThreadsPerThreadgroup` of the fused vs unfused kernels.
2. **Operand reread** (this ADR, `q4k_q6k_qkv_proj_normed`, 2026-05-09): the fused kernel preserves TG count but each TG has multiple independent simdgroups that re-traverse the same operand with different stride patterns, multiplying device-memory traffic. Predicted by examining whether the fused kernel's TG produces multiple output rows that share input data, and whether a register-tile or threadgroup-memory cache is feasible across them.

Both failure modes are detectable from a careful read of the fused kernel before measurement, but the easier signal is the per-kernel batched-diag GB/s comparison vs the unfused alternative.

## Promotion criteria for future fusion candidates

- **Required**: per-kernel batched GB/s ≥ unfused alternative (within 5%).
- **Required**: end-to-end A/B with ≥3-iter drift check shows direction-match with the diag prediction. Magnitude need not match (per ADR-015), but direction must.
- **Sufficient if both above hold**: promote to default with the unfused path retained as a `LARQL_*_FUSED=1` or `LARQL_*_DEFUSED=1` opt-in.

## Consequences

- New default for Gemma 3/4 attention input is 2 dispatches (rms_norm + q4k_q6k_qkv_proj) instead of 1 (q4k_q6k_qkv_proj_normed). +1.6–1.8 tok/s end-to-end, +0.24 ms dispatch overhead, −1.46 ms kernel cost.
- The 5.6 GB f32 lm_head clone on 31B is a separate concern not addressed here (see `project_f16_gemv_wiring_todo.md`).
- The fused kernel and dispatcher are kept reachable as opt-in to support future hardware (where the dispatch-overhead budget could be very different) and for re-validating this decision when the underlying matvec kernel changes.

## Related

- ADR-003 (Fused Q+K+V Projection Kernel) — the base fusion that's *still* in production. This ADR is about the additional norm fusion that landed on top, which we're now reverting.
- ADR-015 (Isolated vs batched kernel perf) — the magnitude-compression lesson.
- `feedback_metal_dispatch_fusion_parallelism.md` — companion lesson on TG-count collapse.
- `crates/larql-compute/PERFORMANCE.md` recent-changes table (2026-05-09 entry).
- `crates/larql-compute/ROADMAP.md` "Track D — QKV defuse PROMOTED 2026-05-09".
