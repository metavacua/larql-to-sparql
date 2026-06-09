# Long-context KV-cache bench (Phase 3)

Runs the qwen35 chat completion path against increasingly-long prompts to
measure the RotorQuant Iso3-compressed device KV cache vs the default
f16 device KV cache.

## Usage

Terminal 1 — start the server in the desired KV-format:

```bash
# f16 baseline
LARQL_QWEN35_GPU=1 \
  LARQL_QWEN35_KV_MAX_SEQ=8192 \
  ./target/release/larql-server /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v10 --port 8181

# OR Iso3 compressed
LARQL_QWEN35_GPU=1 \
  LARQL_QWEN35_KV_FORMAT=iso3 \
  LARQL_QWEN35_KV_MAX_SEQ=8192 \
  ./target/release/larql-server /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v10 --port 8181
```

Terminal 2 — run the bench:

```bash
python3 scripts/bench-long-context.py <mode_label> > bench-<mode>.csv
```

`mode_label` is just an annotation for the CSV (e.g. `f16` / `iso3`).
The server's behaviour is controlled by env vars at startup.

## Output columns

```
mode,target_prompt_tokens,actual_prompt_tokens,decode_tokens,wall_s,
decode_per_tok_avg_s,tok_per_s_overall,vram_pre_mib,vram_peak_mib,content
```

- `vram_peak_mib`: peak GPU 0 memory reading during the request, sampled
  every 100 ms via `nvidia-smi`. Includes the model weight caches, the
  KV slabs, and any per-request scratches.
- `wall_s`: end-to-end request time (prefill + decode + network).
- `tok_per_s_overall`: `decode_tokens / wall_s` — dominated by prefill
  at long contexts. The decode-only rate is the steady-state number
  reported in the headline benches; this CSV measures the long-context
  pressure end-to-end.

## Results captured 2026-05-21 (RTX 4090, Qwen3.6-35B-A3B-vindex-v10)

Captured at `LARQL_QWEN35_KV_MAX_SEQ=8192` for both modes.

| Prompt tok | f16 wall_s | iso3 wall_s | f16 VRAM | iso3 VRAM | Δ VRAM |
|---|---|---|---|---|---|
| 145 | 9.6s | 9.7s | 15924 MiB | 15860 MiB | -64 |
| 2212 | 130.7s | 133.6s | 20404 MiB | 20404 MiB | 0 |
| 4419 | 296.1s | 301.2s | 20788 MiB | 20852 MiB | +64 |

Output coherent in both modes ("It appears you have pasted a large
block..."). Throughput delta ≤ 2% (dequant overhead).

## Why the VRAM savings aren't yet visible at 4K context

Theoretical at max_seq=8192, kv_dim=512, 16 full-attn layers:

  f16 KV slabs:   8192 × 512 × 2 bytes × 2(K+V) × 16 layers = 128 MiB
  iso3 codes:     8192 × 200 bytes × 2 × 16                 =  51 MiB
  iso3 scratches: 8192 × 512 × (4+2+2) bytes                =  32 MiB
                                                              ----
  iso3 total:                                                  83 MiB
  Savings:                                                     45 MiB

45 MiB is buried in the ~20 GiB weight-cache noise (lazy-loaded expert
weights vary by 100+ MiB depending on which experts got dispatched
during prefill). To make the win visible the bench needs `max_seq ≥
32K`, where:

| max_seq | f16 | iso3 | Δ |
|---|---|---|---|
| 32K  | 1024 MiB | 330 MiB | 694 MiB |
| 64K  | 2048 MiB | 530 MiB | 1.5 GiB |
| 128K | 4096 MiB | 930 MiB | 3.2 GiB |

That's the "models > VRAM at long context" operating point.

## Why 32K+ bench takes too long for a session

Each prefill token costs ~70 ms on Qwen3.6-35B-A3B. A 32K-token prompt
prefills in ~37 minutes; 128K → ~2.5 hours. The bench script's nominal
`PROMPT_TARGETS` are kept ≤ 4096 so a full sweep fits in ~10 minutes.

Production bench at 32K+ needs one of:
- **Batched prefill** — a custom forward that processes N prompt
  tokens per kernel call instead of N sequential calls. Out of scope
  for this PR; would land in its own arc.
- **Faked-fill cache** — directly populate `slab.codes_*` / `slab.k/v`
  to a target `cached_seq_len` without going through prefill, then
  measure decode VRAM. Useful for VRAM verification without the
  prefill cost. Could be added as a debug-only bench helper.
- **Patience** — let it run overnight. Single 128K prefill at 70 ms/tok
  finishes in ~150 minutes.

For now: the bench script + 4K results document infrastructure
correctness; the architectural projection above documents the design
target. Phase 4 will add either of the above acceleration paths to
make the value-prop bench session-scale.

## Phase 4e on-hardware results captured 2026-05-21 (RTX 4090 + Threadripper PRO 5965WX, Qwen3.6-35B-A3B-vindex-v10)

End-to-end wall-time A/B of the Phase 4 batched-prefill arc
(PRs #230-#245) vs the per-token fallback. `LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL=1`
forces `qwen35_forward_prefill` into its per-token loop without
enabling any diagnostic dumping. `LARQL_QWEN35_NO_BACKEND=1` (added
this session) skips the unconditional CudaBackend attach in the chat
handler so the batched-matmul path's `backend.is_none()` gate fires.

### GPU mode pre-Step-B (LARQL_QWEN35_GPU=1, hybrid GPU attention + projections)

| N prompt tok | batched wall_s |
|---|---|
| 557 | 37.7 |
| 1110 | 104.8 |

Captured before Step B routed Q4_K matmul through the GPU
backend. The batched-matmul path (PRs #239-#244) was gated
`backend.is_none()` in both `qwen35_attention_block_prefill` and
`deltanet_block_prefill`, so a cuda-attached backend fell back to
per-row matvec dispatch.

### GPU mode post-Step-B + shmem fix (current binary, max_seq=20000)

Step B added `matmul_with_backend` routing Q4_K through
`backend.q4k_matmul → gemm_proj_seq` (cached f16 dequant + cuBLAS
hgemm) and dropped the `backend.is_none()` gates in
`qwen35_attention_block_prefill` and `deltanet_block_prefill`.

The "4× regression" originally flagged here turned out to be an
RTX 4090 shmem-occupancy bug in the attention kernels (decode-attn
and prefill-attn): both sized shared memory by `opts.max_seq` (the
slab capacity) instead of `opts.pos + 1` (the actual cached
context). At `max_seq=20000`, that's 80 KB shmem per block — over
the 48 KB sweet-spot for 3 blocks/SM on Ada — dropping to 1 block/SM
for a ~4× wall-time hit independent of actual context length.

The fix tracks `n_ctx` for both shmem allocation and the kernel
`max_seq` arg. Final numbers at `max_seq=20000`:

| N prompt tok | batched | per-token | batched speedup |
|---|---|---|---|
| 557  | **24.3s**  | 45.2s   | 1.86× |
| 1110 | **45.4s**  | 61.2s   | 1.35× |
| 4419 | **183.7s** | 290.0s  | 1.58× |

**Phase 4 + Step B + shmem fix delivers a real 1.35-1.86× GPU
prefill speedup over the per-token loop, sustained as N grows.**
This is the production-deployment number.

The pre-Phase-4 README baseline above (4419 tok at 296.1s =
67 ms/tok at max_seq=8192) was sidestepping the shmem cliff by
using a smaller max_seq. With the fix applied, the same workload
at max_seq=20000 runs in 183.7s — a **1.61× absolute improvement
over the pre-Phase-4 baseline** with 2.4× more KV cache headroom.

Decomposition of the wins:
- Per-token at max_seq=20000: 1416s pre-shmem-fix → 290s post-fix
  (4.88× from shmem alone)
- Batched at max_seq=20000: 1234s pre-Step-B → 1234s post-Step-B
  → 183.7s post-shmem-fix (6.72× from shmem; Step B routing itself
  contributed ~0% incrementally on top, see below)

Step B routing (matmul through GPU) alone delivered ~0% wall-time
improvement — the matmul wasn't the bottleneck. The 1.35-1.58×
batched-over-per-token win comes from the rest of Phase 4
(PR #231 batched attention kernel, PR #236 bulk KV append,
PR #237 RoPE hoisting, PR #238 batched RMSNorm), which all
compound now that the shmem cliff is gone.

### Full scaling curve post #252-#255 (FA-v1 tiled-scores kernels)

After #248 set `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` to
96 KB, the shmem-by-n_ctx kernels could handle up to ~24K per
launch. #252-#255 then added FA-v1 tiled variants of all four
attention launch paths (decode + prefill + tree-mask + legacy),
streaming the K cache in TILE_K=1024 chunks with fixed ~6 KB
shmem — lifting the per-launch cap to whatever the cache slab
holds.

Full f16 batched-prefill curve (max_seq=20000 for ≤17K, max_seq
sized to fit prompt for larger):

| N tok | wall_s | per-tok |
|---|---|---|
| 557 | 24.3s | 43.6 ms |
| 1110 | 45.4s | 40.9 ms |
| 4419 | 183.7s | 41.6 ms |
| 8831 | 369.7s | 41.9 ms |
| 13245 | 557.2s | 42.1 ms |
| 17658 | 752.8s | 42.6 ms |
| 21555 | 920.5s | 42.7 ms |
| 25864 | 1111.5s | 43.0 ms |
| **34485** | **1511.7s (25.2 min)** | **43.8 ms** |

**43 ms/tok holds across a 62× context range** (557 → 34485
tokens) — the tiled kernels scale linearly with zero per-token
overhead vs the non-tiled path at small N. The 34485-token
data point is **real Qwen3.6-35B-A3B 32K-class prefill running
in session-scale time** (25 min) — the original Phase 4 target
finally hit. This is the headline scaling result from the
Phase 4 + Step B + shmem fix + dynamic-shmem-opt-in +
tiled-scores arc landed across #246/#247/#248/#252/#253/#254/#255.

### iso3 vs f16 (Phase 3 value-prop check)

| N prompt tok | f16 wall_s | iso3 wall_s | f16 VRAM | iso3 VRAM | Δ VRAM |
|---|---|---|---|---|---|
| 17658 (max_seq=20000) | 752.8s | 748.9s | 21284 MiB | 21412 MiB | +128 MiB |
| **34485 (max_seq=36000)** | **1511.7s** | **1496.4s** | **22180 MiB** | **22436 MiB** | **+256 MiB** |

**Iso3 does not show measurable VRAM savings at peak**, even at 32K
where the README projected ~700 MiB of slab-level compression.
Throughput is parity across both scales (within 1%). The slab
compression IS happening at the kernel level (iso3 codes
~225 MiB vs f16 ~1152 MiB at max_seq=36000 = ~900 MiB design
delta), but **lazy expert-load variance dominates the measured peak**:
each unique expert touched is ~140 MB Q4_K, and 5-10 experts'
worth of difference between runs (~700-1400 MiB) is larger than
the slab-level compression delta.

Methodology limitation. The slab-level win can be measured with
the preseed methodology (`bench-preseed.md`) which synthetically
populates the cache without running lazy weight loading — earlier
recorded a clean 480 MiB iso3 savings at max_seq=40000 on that
methodology. The lazy-load methodology used here is the wrong
microscope for the Phase 3 thesis at the operating points where
expert touch dominates.

Output coherent in both modes (`"The text you provided..."` at
34K, `"It appears that your message consists of..."` at 17K).

### CPU-only mode (LARQL_QWEN35_NO_BACKEND=1, where Phase 4 actually fires)

Pre-#259 (sequential attention scan):

| N prompt tok | batched wall_s | per-token wall_s | speedup |
|---|---|---|---|
| 281  | 42.5  | 58.1  | 1.37× |
| 557  | 90.0  | 463.6 | — *(per-token hit lazy expert load mid-bench)* |
| 1110 | 208.8 | 417.2 | **2.00×** |
| 2212 | 566.2 | 681.9 | 1.20× |

The 1110-token row is the most reliable A/B (post-cache-warm in both
modes). **Phase 4 batched matmul delivers a 2× wall-time speedup at
1K context on CPU-only mode.** Speedup decreases with N because both
modes share the same attention O(N²) scan and the same per-position
MoE-routing + DeltaNet-conv1d sequential cost — projection-bandwidth
amortisation only addresses the O(N) "everything else" term.

Curve fit on the batched data: `T(N) ≈ 0.130·N + 5.70e-5·N²` (s).
The `5.70e-5·N²` term was attention scan dominating; addressed by #259.

Post-#259 (rayon-parallel attention scan):

| N prompt tok | batched wall_s | Δ vs pre-#259 |
|---|---|---|
| 1110 | **153.8** | **−55s (1.36× faster)** |
| 2212 | **316.6** | **−250s (1.79× faster)** |

Parallelising the O(N²) attention scan across 24 cores delivers a
speedup that **grows with N** — the larger the context, the more
attention work to parallelise relative to the O(N) projection term.
At 2K context the wall-time saving is half the total prior wall time.

**CPU-only is still not the deployment mode** — production runs
hybrid GPU attn + CPU FFN — but the CPU-only path is now more
useful for any debugging / no-GPU benching scenario.

### Why the bandwidth math overpredicted

RESUME_PROMPT's headline was "~40 TB → ~135 GB across the full flow
(~300×)" of projection-bandwidth reduction at 32K. That number is
real for the *weight-read* traffic, but the wall-time speedup is
capped by whatever bottleneck remains after bandwidth is fixed:

- **Attention O(N²)** scan — same code in both modes, dominates at large N
- **MoE per-token top-K routing** — inherently sequential, scales O(N)
- **DeltaNet conv1d state update** — inherently sequential, scales O(N)
- **CPU memory bandwidth saturation** — projections share the same
  DDR4 channels as activations / cache / KV

Phase 4 amortised projection bandwidth from `40 TB` of weight reads
down to `135 GB`, but the *remaining* work (attention, MoE, DeltaNet
recurrence) was the throttle the whole time. 2× wall-time at 1K is
the actual delivered value of PRs #239-#244 on CPU.

### Implications for the roadmap

- **Three landed fixes compound**: PR #246 (shmem-by-n_ctx), PR #247
  (same fix on spec-decode variants), PR #248 (96 KB dynamic shmem
  opt-in). Together they unlock prefill up to ~24K tokens per launch
  at the flat 42 ms/tok rate.
- **Step B's wall-time delta is ~0%** at the sizes tested — the GPU
  matmul wasn't the bottleneck. The routing fix is still correct
  (avoids host-side CPU matmul work for cuda-attached backends) but
  doesn't unblock the next 2× win.
- **32K+ in a single launch needs a tiled-scores rework.** At 32K
  tokens, shmem would need (32K + 256 + 128) × 4 ≈ 130 KB, over the
  100 KB Ada cap. Plumbing scores into global memory and streaming
  through shmem in tiles is the architectural unblock. Substantial
  CUDA work — separate arc.
- **Iso3's value-prop is still pending hardware confirmation at
  32K+.** At 16K the workload is expert-weight bound (VRAM ~21 GB
  is mostly lazy-loaded experts), so iso3 vs f16 looks flat. Need
  32K+ tests to see the predicted ~700 MiB savings, but those need
  the tiled-scores rework first.
- **Attention-scan optimisation is the next CPU bottleneck** — at
  2212 tokens the attention O(N²) term is already ~half of CPU
  batched wall-time. A CPU rayon-parallel attention scan that
  matches `fused_prefill_attention_seq` would unlock more for the
  CPU-only path.

### Bench env switches added this session

- `LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL=1` — forces
  `qwen35_forward_prefill` into its per-token loop. Cleaner A/B
  switch than piggybacking on a diagnostic dump var.
- `LARQL_QWEN35_NO_BACKEND=1` — when built with `--features cuda`,
  skips the unconditional CudaBackend attach in
  `chat.rs::handle_chat_completions`. Lets the CPU batched-matmul
  gates (`backend.is_none()`) fire from a cuda-built binary.

### Step B routing summary

- `matmul_with_backend` added in `quant_dispatch.rs` (mirrors
  `matvec_with_backend`). Q4_K → `backend.q4k_matmul`; other formats
  fall through to `QuantTensor::matmul` (CPU).
- Call sites updated: 4 in `qwen35_attention_block_prefill`
  (Q/K/V/O) and 3 in `deltanet_block_prefill` batched branch
  (attn_qkv / attn_gate / ssm_out). `backend.is_none()` gate
  dropped in DeltaNet.
- MoE scatter-gather left on CPU intentionally — GPU MoE uses
  `qwen35_moe_ffn_batch` which batches 8 experts per call. A
  scatter-gather variant for GPU needs separate measurement.
- Bug fixed during Step B: GPU fused recurrence paths in
  `deltanet_block_step_with_optional_projections` (the
  `LARQL_QWEN35_E6A_FUSED` path and the default `fused_o_normed`
  path) returned `block_out [hidden]` regardless of
  `skip_ssm_out`. Now honour the flag and return `o_flat
  [value_dim]` when the caller is batching the ssm_out matmul.
  Surfaced because pre-Step-B `batched_ssm_out` was gated
  `backend.is_none()` and never fired in GPU mode.

### Reproducing

```bash
# Build (cuda enabled — for the GPU run)
cargo build --release --bin larql-server --features cuda

# CPU batched
LARQL_QWEN35_NO_BACKEND=1 \
  ./target/release/larql-server /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v10 \
  --port 8181

# CPU per-token (separate run, restart server)
LARQL_QWEN35_NO_BACKEND=1 LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL=1 \
  ./target/release/larql-server /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v10 \
  --port 8181

# GPU batched (Step B path — production deployment)
LARQL_QWEN35_GPU=1 LARQL_QWEN35_KV_MAX_SEQ=20000 \
  ./target/release/larql-server /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v10 \
  --port 8181

# GPU per-token (baseline for A/B)
LARQL_QWEN35_GPU=1 LARQL_QWEN35_KV_MAX_SEQ=20000 \
  LARQL_QWEN35_FORCE_PER_TOKEN_PREFILL=1 \
  ./target/release/larql-server /tank/ai/Qwen/Qwen3.6-35B-A3B-vindex-v10 \
  --port 8181

# Drive the bench
LARQL_BENCH_TARGETS=512,1024,4096 LARQL_BENCH_DECODE=4 \
  LARQL_BENCH_HTTP_TIMEOUT=3600 \
  python3 scripts/bench-long-context.py <mode_label>
```
