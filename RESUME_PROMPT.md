# larql resume prompt — feed this to a fresh session

## Where we are (state at end of session 2026-05-22 ext.)

**The long-context-prefill arc lands.** Real Qwen3.6-35B-A3B 32K-class
prefill (34485 tokens) now runs in **25.2 minutes at 43.8 ms/tok on
RTX 4090** — the original Phase 4 goal. **43 ms/tok holds flat
across a 62× context range** (557 → 34485 tokens).

### PRs landed this session (12)

| PR | What |
|---|---|
| #246 | Phase 4e A/B switches + Step B routing + skip_ssm_out fix + shmem-by-n_ctx |
| #247 | Same shmem fix on 3 spec-decode prefill variants |
| #248 | 96 KB dynamic shmem opt-in on attention kernels (16K unlock) |
| #249 | 17K f16/iso3 head-to-head |
| #250 | Full 4K → 17K scaling curve |
| #251 | RESUME_PROMPT.md mid-session refresh |
| #252 | FA-v1 tiled-scores decode kernel |
| #253 | FA-v1 tiled-scores prefill kernel |
| #254 | FA-v1 tiled-scores tree-mask prefill kernel (spec-decode) |
| #255 | Tiled dispatch on legacy `fused_prefill_attention_seq_device` |
| #256 | Extended scaling curve to 32K (34485 tok @ 43.8 ms/tok) |
| #257 | iso3 32K methodology note (lazy-load noise dominates slab win) |

### Final scaling curve (Qwen3.6-35B-A3B-vindex-v10, RTX 4090, f16)

| N tok | wall_s | per-tok |
|---|---|---|
| 557 | 24.3 | 43.6 ms |
| 1110 | 45.4 | 40.9 ms |
| 4419 | 183.7 | 41.6 ms |
| 8831 | 369.7 | 41.9 ms |
| 13245 | 557.2 | 42.1 ms |
| 17658 | 752.8 | 42.6 ms |
| 21555 | 920.5 | 42.7 ms |
| 25864 | 1111.5 | 43.0 ms |
| **34485** | **1511.7 (25.2 min)** | **43.8 ms** |

iso3 vs f16: throughput parity within 1% at every scale tested.
End-to-end VRAM peak doesn't show the expected slab-level win
because lazy expert-load variance (~140 MB/expert × 5-10 expert
touch differences) dominates. The slab-level win is real (~900
MiB design delta at max_seq=36000) but needs the preseed
methodology to measure cleanly (`scripts/bench-preseed.md`,
already recorded 480 MiB at max_seq=40000).

### Architecture summary

Decode-step attention + all three prefill-attention variants
(linear, tree-mask, legacy) have **tiled-scores** kernels: K cache
streams through a fixed-size shmem region (TILE_K=1024 elements,
~6 KB total), with per-thread running `(m, l, acc)` state
implementing FlashAttention v1 online softmax. Dispatch threshold:
`max_seq_i > 16384` → tiled, otherwise non-tiled (simpler control
flow stays the fast path for short contexts).

Step B matmul routing (#246) routes Q4_K projections through
`backend.q4k_matmul → gemm_proj_seq` (cuBLAS hgemm on cached f16
dequant). Per the session's bisect, that routing itself contributed
~0% to wall-time — the dominant win was the shmem-occupancy fix
(#246/#247/#248).

### Bench env switches added this session

- `LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL=1` — forces per-token loop
  for clean A/B without diagnostic side-effects.
- `LARQL_QWEN35_NO_BACKEND=1` — when built with `--features cuda`,
  skips the unconditional CudaBackend attach. Lets CPU
  batched-matmul gates (`backend.is_none()`) fire from a cuda-built
  binary.

## What's actually next

Three contained items remain, in rough priority order:

### A. **CPU rayon-parallel attention scan**

The CPU-only bench fit curve was `T(N) ≈ 0.130·N + 5.70e-5·N²` (s).
At 2K+ tokens the O(N²) attention term is already ~half the wall
time. A CPU rayon-parallel attention scan that matches the GPU's
fused_prefill_attention_seq behaviour would unlock more for the
CPU-only path.

Pure CPU code, ~300-500 LoC. Lower production priority than the
GPU arc just completed (CPU-only isn't the deployment mode), but
clean self-contained PR for a session that wants something tidy.

### B. **GPU MoE scatter-gather measurement**

PR #241 (prior session) added CPU scatter-gather for MoE prefill,
gated `backend.is_none()`. With matmul_with_backend routing Q4_K
to GPU, scatter-gather on GPU could try replacing the per-row
`qwen35_moe_ffn_batch` (PR #218) which batches 8 experts per call.
Empirical: which is faster on RTX 4090? Bench-driven session.

### C. **Investigate the iso3 lazy-expert noise**

The Phase 3 slab-level win exists but is invisible at peak VRAM
under the lazy-load methodology. Two possible follow-ups:
1. Modify the bench harness to record VRAM peak *after* the prompt
   completes (when KV slab is loaded but no in-flight expert
   transfers). Smaller noise band.
2. Pre-warm the model by running a dummy 32K prompt then capture
   peak on a second 32K run. Forces all experts to be cached.

These are bench-script changes, not core code. Could land in an
afternoon.

### Smaller follow-ups

- **Phase 4-final dump-env fallback** — qwen35_forward_prefill falls
  back to per-token when `LARQL_QWEN35_DUMP_*` is set. Teach
  batched-prefill helpers to emit dump format too. Not blocking.
- **GPU paired Q4_K matmul** — DeltaNet's per-row form uses
  `qwen35_paired_q4k_matvec`. A paired matmul would batch those.
- **Q5_K / Q6_K / Q8_0 GPU matmul** — Step B routing only fires for
  Q4_K. Other formats stay on CPU. Mostly mechanical: add per-format
  GPU matmul kernels mirroring the Q4_K one.

## Critical context the fresh session needs

- **Project memory**: read
  `~/.claude/projects/-home-ianblenke-github-com-ianblenke-larql/memory/MEMORY.md`
  — especially `project_larql_driving_goal.md` and
  `project_batched_prefill_arc.md` (covers PR #246-#250; will need
  another update to capture #251-#257).

- **Standing rules** (unchanged):
  - **Don't self-merge unattended.** User authorises each merge via
    `merge and continue`. Admin-merge OK when CI's only failures
    are pre-existing main-branch issues (3 known: openspec on main,
    macOS Metal stale trait impls, Ubuntu cuda runner missing nvcc).
  - **No `--no-verify` / hook skipping** without explicit permission.
  - **OpenSpec workflow** — `make traceability` after test line shifts.

- **Hardware**: Threadripper PRO 5965WX (24 cores, 48 threads, AVX2
  no AVX-512), 440 GB RAM, RTX 4090. Models at `/tank/ai/<org>/...`.
  CUDA enabled in the dev build.

- **Branch state**: clean main as of PR #257 merge. Untracked
  diagnostic test files from prior sessions still present
  (`tests/test_gemma3_*.rs`, `tests/test_q6k_roundtrip.rs`,
  `tests/profile_4b_decode.rs`, `tests/test_v_proj_writer_roundtrip.rs`)
  — harmless leftovers; leave them.

## Quick start prompt for the next session

> Read `RESUME_PROMPT.md`. 12 PRs landed last session (#246-#257)
> completing the long-context prefill arc on Qwen3.6-35B-A3B. The
> headline: **real 32K-class prefill (34485 tokens) in 25.2 minutes
> at 43.8 ms/tok** on RTX 4090. **43 ms/tok holds flat across a
> 62× context range** (557 → 34485 tokens) — Phase 4 + Step B +
> shmem fixes + 96 KB shmem opt-in + FA-v1 tiled-scores kernels on
> all four attention paths (decode, prefill, prefill-tree-mask,
> prefill-legacy).
>
> **Three next steps in order of priority:**
>
> 1. **CPU rayon-parallel attention scan.** O(N²) attention is
>    ~half of CPU batched wall time at 2K+ tokens. Pure CPU code,
>    300-500 LoC. CPU-only isn't the deployment mode but contained
>    PR for a tidy session.
>
> 2. **GPU MoE scatter-gather measurement.** Empirical compare
>    against PR #218's `qwen35_moe_ffn_batch`. Bench-driven.
>
> 3. **Iso3 slab-level VRAM bench under controlled conditions.**
>    The lazy-load noise drowns the slab-level win at peak VRAM.
>    Modify bench harness to measure post-prompt or use pre-warming
>    to isolate slab savings.
>
> Per-PR auth via `merge and continue`. Admin-merge OK when CI's
> only failures are pre-existing main-branch issues (openspec on
> main, macOS Metal stale trait impls, Ubuntu cuda nvcc missing).
