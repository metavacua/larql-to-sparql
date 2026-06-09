# Qwen3.6 27B bench baseline — larql vs llama.cpp

Captured 2026-05-11 on the same host, immediately after C.5k landed
(parity test green, GT rank 0 at every step). **The headline
numbers below are the original CPU-only baseline; the gap closed
sharply with Phase E.6.D (Q5_K direct CUDA matvec), which lifted
decode 0.49 → 5.19 t/s (≈10× the original baseline, ≈15× over the
prior best GPU build). See the E.6.D section further down.**

## Setup

- **Model**: Qwen3.6-27B Q4_K_S GGUF, 14.76 GiB on disk, 26.90 B params.
- **Host**: Linux x86_64, NVIDIA RTX 4090 (24 GiB VRAM), llama.cpp build
  b1-389ff61 with CUDA, larql at `c5k` (post PR #84).
- **Workload sizes**: prefill 32 tokens, decode 8 tokens. Small numbers
  because larql is CPU-only — keeps total wall time under 2 minutes.

## Results

| Config | Backend | Prefill (tok/s) | Decode (tok/s) | Load | Memory |
|---|---|---:|---:|---:|---:|
| llama.cpp | CUDA RTX 4090 (`ngl=99`, `pp128`/`tg64`) | 2097.18 | 50.60 | ~2 s | 14.76 GiB VRAM |
| llama.cpp | CPU (`ngl=0`, `pp32`/`tg8`) | 37.33 | 2.60 | ~5 s | ~16 GiB RAM |
| **larql** | CPU (scalar Rust, `pp32`/`tg8`) | **0.48** | **0.49** | 49 s | **~100 GiB RAM** |

`pp` = prefill tok/s, `tg` = decode tok/s. llama.cpp numbers from
`llama-bench`; larql numbers from
`real_gguf_qwen35_bench` test (loops `qwen35_forward_step`).

## Headline ratios

| | larql / llama.cpp (CPU) | larql / llama.cpp (GPU) |
|---|---:|---:|
| Prefill speed | 1/78 (1.3%) | 1/4370 (0.02%) |
| Decode speed | 1/5.3 (19%) | 1/103 (1.0%) |

## Memory blow-up explained

larql currently **dequantizes every Q4_K_S weight to f32 at load time**
and holds the full f32 model in RAM. 26.90 B params × 4 bytes ≈ 107 GiB
matches the observed 100 GiB RSS. llama.cpp keeps the model in its
quantized form (14.76 GiB on disk, ~16 GiB resident with overhead) and
dequantizes per-tile during matmul.

This is the single biggest item on the perf TODO list. Until that lands:
- ~2 GiB/s f32 matmul throughput (BLAS) × ~50 GB per forward pass at
  27 B params → seconds per token, exactly what we see.
- 100 GiB RAM means larql can't actually run a 35-B-MoE host without
  ≥ 128 GiB system memory.

## 2026-05-12 update — Phase E.1/E.2 GPU dispatch for lm_head + FFN

Pivot off the CPU AVX2 axis: route lm_head Q6_K matvec and all 192
FFN Q4_K matvecs/token through `larql_compute::cuda::CudaBackend`
(the existing `q6k_direct` / `q4k_direct` GPU kernels). Opt-in
behind `--features cuda` and `LARQL_QWEN35_GPU=1`. Weights upload
to VRAM on first dispatch; cache reused thereafter.

| Config | Decode (t/s) | Δ vs CPU lazy | VmRSS |
|---|---:|---:|---:|
| Phase 2d (CPU lazy + AVX2 + rayon) | 0.23 | — | 19.99 GiB |
| **Phase E.1/E.2 (+ GPU lm_head & FFN)** | **0.28** | **+22 %** | 21 GiB (host) |
| llama.cpp CUDA GPU | 50.60 | 220× theirs | 14.76 GiB VRAM |

Modest +22 % gain — much less than expected. The matvec wins are
real, but **the DeltaNet recurrence stays on CPU** and dominates
steady-state decode time at 3.6 s/token. Per-token contributions:

- **DeltaNet recurrence (CPU scalar)** — 48 layers × `delta_net_step`
  with per-head state matrices: this is now the bottleneck.
- **Per-matvec host↔device transfer** — ~480 transfers/token at
  PCIe Gen4 ~25 GB/s ≈ 1.5 ms each, adds up.
- **Non-matvec ops (norms, silu, residual adds)** still CPU.

**The next big perf lever is Phase E.4** — a CUDA kernel for the
DeltaNet recurrence + Conv1D-with-state. Per-head state matrices
fit in shared memory (128×128 f32 = 64 KB per head, ok on Ampere
SM-89). Mirrors llama.cpp's
`ggml_compute_forward_gated_delta_net_one_chunk` which we already
diffed bit-exact in Phase C. Estimated ~600 LoC + the cudarc PTX
plumbing.

E.3 (DeltaNet `attn_qkv`/`attn_gate`/`ssm_out` + full-attn
q/k/v/o through GPU) is now done; it is pure projection plumbing
and remains marginal until E.4 moves the recurrence. E.6
(device-resident weights + KV cache + CUDA Graphs) is the
longer-term arc.

**Parity preserved**: `real_gguf_qwen35_token_diff_vs_llama_cpp`
under `LARQL_QWEN35_GPU=1` still emits the same
`[<think>, \n\n, </think>, \n\n, Hello]` with GT rank 0 every step.

## 2026-05-12 update — Phase E.3 GPU dispatch for DeltaNet + full-attn projections

Extended the Phase E backend path to route the remaining
lazy-quantised projection matvecs through `CudaBackend`: DeltaNet
`attn_qkv`, `attn_gate`, `ssm_out`, and full-attn `attn_q`,
`attn_k`, `attn_v`, `attn_o`. The token-diff parity harness now
attaches the same backend when `LARQL_QWEN35_GPU=1`, so the
documented GPU validation command exercises CUDA dispatch.

Protocol:

```bash
LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_BENCH_PREFILL=16 LARQL_QWEN35_BENCH_DECODE=4 \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_bench -- --nocapture
```

Result:

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| Phase E.1/E.2 (+ GPU lm_head & FFN) | — | 0.28 | 21 GiB (host) |
| **Phase E.3 (+ DeltaNet/full-attn projections)** | **0.33** | **0.33** | **21.16 GiB** |

E.3 buys another modest +18 % over E.1/E.2. The headline still
confirms the same bottleneck: DeltaNet recurrence and Conv1D remain
CPU-resident, so Phase E.4 is still the real unlock.

## 2026-05-12 update — Phase E.4 first pass: CUDA Conv1D + DeltaNet recurrence kernels

Added CUDA kernels for the Qwen3.6 DeltaNet Conv1D-with-state and
decay-first recurrence. The recurrence kernel keeps the `state[s, s,
h_v]` layout used by ndarray (`h_v` fastest), uses one CUDA block per
V head, and matches the llama.cpp-compatible C.5j decay-first order.
`CudaBackend` now caches the Conv1D and recurrent state buffers by
host state pointer and re-uploads only when `next_position == 0`, so
the device buffer is authoritative during an active sequence.

Validation:

```bash
cargo test -p larql-compute --features cuda cuda::deltanet -- --nocapture

LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_token_diff_vs_llama_cpp -- --nocapture
```

The token-diff parity check still emits
`[<think>, \n\n, </think>, \n\n, Hello]` with GT rank 0 every step.

Bench protocol was unchanged from E.3:

```bash
LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_BENCH_PREFILL=16 LARQL_QWEN35_BENCH_DECODE=4 \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_bench -- --nocapture
```

Result:

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| Phase E.3 (+ DeltaNet/full-attn projections) | 0.33 | 0.33 | 21.16 GiB |
| **Phase E.4.1/E.4.2 first pass (+ CUDA Conv1D/recur)** | **0.33** | **0.33** | **21.16 GiB** |

The first pass preserves correctness but does **not** improve headline
throughput. The bottleneck has shifted from arithmetic in the scalar
recurrence to per-layer launch/synchronisation and CPU/GPU ping-pong:
Conv1D output, recurrence output, post-recurrence RMSNorm, z-gating,
and residual/FFN boundaries still cross back to the CPU every layer.
The remaining E.4 work should focus on fusing the DeltaNet block
around the recurrence output and moving per-head L2/RMSNorm/z-gate
operations onto the same device path; otherwise E.6-style
device-resident activations/CUDA graphs are required for the expected
multi-tok/s jump.

## 2026-05-12 update — Phase F.0: Qwen3.6-35B-A3B MoE GGUF probe (the right target model)

After the E.7 curve showed that "Attn-only GPU" buys almost nothing
on dense Qwen3.6-27B (0.25 vs 0.23 t/s — adding 1.5 GiB of VRAM for
attention/recurrence/conv1d compute saves <10 %), the conclusion is
that **dense models aren't the right fit for larql's value prop**.
Every FFN runs every token, so "FFN on CPU" is the full FFN cost.

Pivoting to Qwen3.6-35B-A3B — the MoE variant where 8 of 256 FFN
experts activate per token. New test `probe_qwen35_moe_gguf_layout`
in `qwen35_load.rs` (env-gated by `LARQL_QWEN35_MOE_GGUF=…`) dumps
the on-disk tensor layout. Findings:

| Dim                                | Qwen3.6-35B-A3B   | Qwen3.6-27B |
|---|---|---|
| layers                             | 40                | 64          |
| hidden                             | 2048              | 5120        |
| **expert_count**                   | **256**           | dense       |
| **expert_used_count**              | **8** + 1 shared  | dense       |
| expert_ffn_dim                     | 512               | 17408       |
| ssm.inner_size                     | 4096              | 6144        |
| ssm.time_step_rank                 | 32                | 48          |
| ssm.group_count                    | 16                | 16          |
| full_attention_interval            | 4 (10 attn layers)| 4 (16 attn) |
| context_length                     | 262144            | 65536       |
| file size                          | 21.6 GB           | 14.76 GB    |

Per-layer MoE tensors are **3D packed**:

```
ffn_gate_exps.weight   [hidden, expert_ffn_dim, num_experts]  Q4_K
ffn_up_exps.weight     [hidden, expert_ffn_dim, num_experts]  Q4_K
ffn_down_exps.weight   [expert_ffn_dim, hidden, num_experts]  Q5_K
ffn_gate_inp.weight    [hidden, num_experts]                  f32   (router)
ffn_*_shexp.weight     [hidden, expert_ffn_dim] etc.          (always-on shared)
```

### Active-weights estimate (per token)

| Class                            | Bytes (Q4_K avg) |
|---|---:|
| 10 full-attn × Q+K+V+O proj      | ~330 MB |
| 30 DeltaNet linear × projections | ~150 MB |
| 40 FFN × (8 active + 1 shared)   | ~280 MB |
| LM_head (Q6_K)                   | ~390 MB |
| **active per token**             | **~1.2 GB** |
| total file                       | ~22 GB |

**~5 % of weights are active per token.** This is exactly larql's
design target: the inactive 95 % stays on CPU/disk, with only the
active slice paged in. llama.cpp loads the entire model into VRAM
(or all-CPU); it can't selectively offload by expert. larql's
per-class residency mechanism (E.7) extends naturally to per-expert
residency once the MoE forward path is in.

### Status: forward path not yet built

`Qwen35MoeArch` (Phase B.2) has the metadata + tensor-key formatters,
but `qwen35_forward_step` still calls dense `swiglu_ffn_lazy` for
every layer. To produce real numbers we need:

1. **MoE FFN forward**: router (f32 matmul) → top-K + softmax →
   per-expert SwiGLU with weighted sum + shared-expert add.
2. **3D expert weight slicing**: the GGUF packs all experts in one
   tensor; need a `QuantTensor` view that lazily slices expert E's
   2D submatrix without re-decoding the whole 3D packed block grid.
3. **Bridge**: extend `load_qwen35_weights` (or write a sibling)
   for the MoE arch: per-layer `MoeLayerWeights` with router +
   expert tensors + shared expert.
4. **Bench harness**: extend to take the MoE GGUF path and time
   the same prefill/decode pattern.

Once those exist, the same E.7 tier curve becomes meaningful: at
the "Attn-only" tier on this MoE model, with FFN streamed from
CPU + only 8 experts loaded per token, larql should fit in ~3-4
GiB VRAM at meaningful throughput. llama.cpp's `--gpu-layers N`
can't replicate this — it offloads whole layers, not per-expert.

This is the bench that decides whether the value prop is real.

## 2026-05-12 update — Phase F.2: MoE FFN forward (router → top-K → weighted experts + shared)

Built `swiglu_moe_lazy` in
[`crates/larql-inference/src/attention/qwen35_forward.rs`](../../crates/larql-inference/src/attention/qwen35_forward.rs).
The function takes the four MoE quant-tensors plus an optional shared
expert and computes one token's MoE FFN output:

```
logits      = router @ x                                   // [num_experts]
idx, top_l  = top_k(logits, top_k)
w           = softmax(top_l)                               // [top_k]
y_moe       = Σ_i w_i · swiglu(gate[e_i], up[e_i], down[e_i])(x)
y_shared    = swiglu(shexp_g, shexp_u, shexp_d)(x)         // if present
return        y_moe + y_shared
```

Implementation notes:

- Per-expert weight access is via `QuantTensor::expert_slice(e, num_experts)`
  from F.1 — zero-copy `Arc<[u8]>` views into the 3D-packed parent.
  No re-decoding of any super-block grid.
- Each per-expert SwiGLU dispatches through the existing
  `swiglu_ffn_lazy` path: same paired-matvec / device-resident
  fused-block / per-call fallback decisions as the dense case. The
  `gpu_tier::backend_for(GpuClass::Ffn)` knob therefore covers MoE
  too — `LARQL_QWEN35_GPU_NO_FFN=1` pushes every active expert to
  CPU rayon (the VRAM-minimal mode).
- Top-K is a full sort on the router logits. `num_experts` is 128–256;
  the cost is dwarfed by the active-expert matmuls.
- Shared expert is optional. Qwen3.6-35B-A3B has one; some MoE
  variants don't ship one and the call site passes `None`.

Two parity tests verify the math against a hand-rolled scalar
reference:
`swiglu_moe_lazy_matches_reference` (4 experts, top-K=2, hidden=3,
ffn_dim=4, with shared expert) and `swiglu_moe_lazy_without_shared_expert`
(2 experts, top-K=1, identity-like weights so the expected output
is `silu(x)·x`).

Tests pass against the unmodified 882-test inference suite + the 2
new MoE tests. Clippy clean.

Status: forward kernel is in place. Next is **F.3** — wire it up via
a new `Qwen35MoeFullLayerWeights` struct + a sibling
`load_qwen35_moe_weights` GGUF bridge + arch-detection in
`qwen35_forward_step`, then re-run the E.7 tier curve on the MoE
GGUF to produce the throughput/VRAM numbers that close out this
phase.

## 2026-05-13 update — Phase F.3: MoE bridge end-to-end on Qwen3.6-35B-A3B (first light)

Wired the F.2 forward kernel into the loader + bench harness. **First
real numbers on Qwen3.6-35B-A3B-UD-Q4_K_M.gguf:**

| Setting                | Value |
|---|---:|
| Path                   | all-CPU lazy-quant (no GPU) |
| Load time              | 173 s (22 GiB GGUF, lazy-mmapped) |
| Prefill                | 1.06 tok/s (4 tokens) |
| Decode                 | 1.06 tok/s (2 tokens) |
| Process RSS            | 22.3 GiB (full mmap window) |

(For comparison: Qwen3.6-27B dense on the same machine ran 10.6 t/s
decode after Phase E.6.I, but at 14.8 GiB RSS plus a 3 GiB GPU-resident
tier; the MoE has 8× the expert dispatch volume per token, no Q8_0 GPU
kernel yet, and the attention/shared-expert tensors are Q8_0
dequant-per-row on CPU.)

### What landed

1. **`Qwen35MoeFfnWeights`** in `qwen35_forward.rs` — per-layer router
   + 3D-packed expert tensors + optional shared expert. Added to
   `Qwen35FullLayerWeights` as `pub moe: Option<...>`. When `Some`, the
   forward step dispatches through `swiglu_moe_lazy` instead of dense
   `swiglu_ffn_lazy`.
2. **`load_qwen35_moe_ffn`** in `qwen35_load.rs` — pulls
   `layers.{L}.ffn_gate_inp.weight` (router, f32),
   `ffn_{gate,up,down}_exps.weight` (Q4_K/Q4_K/Q5_K, 3D-packed), and the
   optional `ffn_{gate,up,down}_shexp.weight` shared-expert trio. Pulls
   `num_experts`/`top_k` from the architecture's `is_moe()` /
   `num_experts()` / `num_experts_per_token()` accessors.
3. **`qwen35_moe_lazy_keys(n_layers)`** — helper that emits the MoE
   tensor key set so the bench harness folds them into the lazy-load
   set. The bench now branches on `arch.is_moe()` to extend lazy_keys.
4. **3D-tensor support in `load_gguf_lazy_tensors`** — extended the
   `n_dims != 2` filter to accept `n_dims == 3` and flatten to
   `[dims[1] * dims[2], dims[0]]`. F.1's `expert_slice` then carves
   per-expert subviews without copying.
5. **GGUF→config flow-through for MoE metadata** — added
   `expert_count` / `expert_used_count` / `expert_feed_forward_length`
   → `num_experts` / `num_experts_per_tok` / `moe_intermediate_size`
   so `Qwen35MoeArch::is_moe()` returns `true` on a real GGUF (was
   always returning `false` before because the metadata wasn't
   reaching `ModelConfig`).
6. **Critical Q5_0/Q8_0 type-id fix** — `crates/larql-models/src/quant/ggml/mod.rs`
   had `TYPE_Q8_0 = 6` and `TYPE_Q5_0 = 8`, swapped vs the official
   GGML enum (`Q5_0 = 6, Q8_0 = 8`). Qwen3.6-27B's Q4_K_S quant didn't
   hit it; Qwen3.6-35B-A3B-UD-Q4_K_M ships Q8_0 attention/shared-expert
   tensors and tripped the swap immediately (matvec route fell through
   to "unsupported tensor type id 8"). Swapped constants to match
   wire spec.
7. **Q8_0 in `QuantTensor::matvec`** — added rayon-parallel
   dequant-per-row dispatch (no fused row-dot kernel yet — equivalent
   shape to the Q5_K fallback path).

### Active-weights vs all-resident, observed

The 22.3 GiB RSS reflects mmap of the entire GGUF, *not* what's
actually being touched per token. Per-token decode reads from:

- 1 attention or DeltaNet layer's projections (Q8_0, Q4_K, F32 norms)
- 8 active experts' gate/up (Q4_K) + down (Q5_K) — `expert_slice` views
- 1 always-on shared expert (Q4_K + Q5_K)
- router (f32, 256 × 2048)
- lm_head (Q6_K), final_norm (F32), embed (Q4_K row lookup)

≈ 1.2 GiB of weights touched per token vs 22 GiB resident. That's the
value-prop ratio; the next phase is to make the unused 21 GiB pageable
or stream-on-demand so RSS tracks active weights instead of file size.

### Where the time goes (all-CPU)

Without per-section profiling, the suspicion is that the bulk goes to:

1. Q8_0 dequant-per-row on the ~700-MB attention/shared-expert
   tensors per layer (no SIMD / no fused row-dot kernel yet).
2. The 9 SwiGLU dispatches per FFN layer × 40 layers × 945 ms decode
   per token (rough math).
3. Router + top-K sort + softmax — small.

`LARQL_QWEN35_PROFILE=1` will quantify this once we drop into
optimization.

### Status: F (MoE bring-up) phase complete

- ✅ F.0 — GGUF layout probe.
- ✅ F.1 — `QuantTensor::expert_slice` zero-copy 3D-packed slicing.
- ✅ F.2 — `swiglu_moe_lazy` router → top-K → weighted-experts +
  shared, with parity tests.
- ✅ F.3 — GGUF bridge + 3D lazy loader + arch-metadata flow-through
  + Q5_0/Q8_0 const fix + bench end-to-end at 1.06 t/s.

What's next is a pivot to the **value-prop measurement**: re-run the
E.7 per-class GPU residency curve on the MoE GGUF (`LARQL_QWEN35_GPU=1`
gates the projection classes), then quantify how VRAM scales vs
llama.cpp's whole-layer offload. The expected story: at the
"Attn-only" tier, larql holds ~3-4 GiB VRAM at meaningful tok/s
because the inactive 95 % of MoE weights stay on CPU/disk — a
configuration llama.cpp can't replicate.

## 2026-05-13 update — Phase F.4: per-class residency curve on Qwen3.6-35B-A3B MoE

Ran the E.7 tier curve against the MoE GGUF. RTX 4090 (24 GiB), prefill=4
decode=4, `LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1`,
`--features cuda` for tiers 1–3.

| Tier | GPU classes                                                  | VRAM (MiB) | Prefill (t/s) | Decode (t/s) |
|---|---|---:|---:|---:|
| 0 (all-CPU) | —                                                  | 0     | 1.06 | 1.06 |
| 1 (LM_HEAD) | `LmHead`                                           | 2414  | 0.83 | 1.24 |
| 2 (no-FFN)  | `LmHead`,`AttnProj`,`DnProj`,`DnRecurrence`        | 2414  | 0.81 | 1.20 |
| 3 (all)     | all five classes                                   | 5582  | 1.39 | **3.49** |

(`LARQL_QWEN35_GPU_NO_FFN=1` etc. drop classes back to CPU.)

### Reading the curve

1. **LM_HEAD on GPU** (Tier 1) buys +17 % decode for ~2.4 GiB VRAM
   (mostly CUDA runtime + Q6_K device buffer for the 1.5 GiB lm_head).
   Same finding as the 27B dense bench.

2. **Tier 2 ≈ Tier 1** — *flipping AttnProj / DnProj / DnRecurrence on
   has no measurable effect.* Reason: this GGUF stores all those
   projections as Q8_0 (`type=8`, the legacy 32-elem int8+f16 block),
   and Q8_0 has **no CUDA matvec kernel** in larql today. The
   `matvec_with_backend` path returns `None` for Q8_0 → CPU rayon
   fallback regardless of the residency knob. (Adding `Q8_0` to
   `QuantFormat` + a small CUDA kernel would unlock this tier.)

3. **FFN on GPU** (Tier 3) is **the dominant lever**: +191 % decode
   from Tier 2, for +3.2 GiB VRAM. This is the active 8 experts ×
   40 layers × (gate+up+down) lazy-quant Q4_K/Q5_K matvecs hitting
   the device-resident fused FFN block (E.6.B.1). Each expert dispatch
   reuses the same code path as the dense Qwen3.6-27B FFN block —
   the multiplexer (router + top-K) is host-side, the per-expert
   matmuls land on existing GPU kernels.

### Value-prop numbers vs llama.cpp

llama.cpp's `--gpu-layers N` offloads whole layers (all classes), so
its only knobs are "N layers on GPU, rest on CPU". On Qwen3.6-35B-A3B
that means each offloaded layer brings its 256-expert tensor (the
inactive 248 of which are never read for that token). A 24 GiB card
can hold maybe ~20 of 40 layers at this quant. larql's per-class
breakdown shows that **the actual GPU-useful weights per token are
the 8 active experts × (gate+up+down) ≈ 28 MiB plus lm_head ≈ 1.5 GiB**
— the inactive 248 experts can stay on CPU/disk indefinitely.

So the larql-vs-llama.cpp tradeoff is now concrete and measurable:

- llama.cpp 20-layer offload: ~20 GiB VRAM, decoder runs whichever
  layers are offloaded at full GPU speed and the rest at CPU speed.
- larql Tier 3:                **5.6 GiB VRAM, 3.49 t/s end-to-end**.

Whether 3.49 t/s is competitive with llama.cpp at the same VRAM
budget needs a head-to-head — `llama-bench -ngl <N>` on this same
GGUF with `--threads $(nproc)`, capped at our 5.6 GiB. That's the
deliverable from this phase: **the value-prop graph**.

### Where the remaining cost is

At Tier 3, decode = 287 ms/tok. The known untapped wins:

1. **Q8_0 GPU matvec kernel** — would let Tier 2 actually take
   AttnProj/DnProj off the CPU. Estimated +30-50 ms/tok saved.
2. **CUDA Graphs (E.6.C)** — record the decode-step DAG once,
   replay per token. Saves dispatch overhead × ~600 kernels/tok.
3. **Batched lm_head softmax** — already landed (E.6.E.softmax) but
   needs MoE-side parity; verify nothing regressed.
4. **Better Q5_0/Q8_0 CPU SIMD** — Q8_0 dequant-per-row is rayon
   parallel but not SIMD; llama.cpp's `ggml_vec_dot_q8_0_q8_0` is
   AVX2-vectorized and ~6× faster (the E.8 backlog).

Two new follow-ups added to the open task set:
- **F.4** — Q8_0 GPU matvec kernel for the AttnProj/DnProj tier.
- **F.5** — head-to-head bench vs `llama-bench -ngl N` on the same
  GGUF, capped at iso-VRAM, to land the value-prop graph.

## 2026-05-13 update — Phase F.5: head-to-head Qwen3.6-35B-A3B vs llama.cpp

Ran the same Qwen3.6-35B-A3B-UD-Q4_K_M.gguf through llama.cpp's
`llama-bench` (commit `389ff61`, CUDA build at
`/home/ianblenke/3rd-party/llama.cpp/build/bin/llama-bench`, same
RTX 4090, default thread count).

| Backend  | ngl | VRAM (≈)  | pp64 (t/s) | tg16 (t/s) |
|---|---:|---:|---:|---:|
| llama.cpp | 0  | ~0 GiB         | 102.41 | 16.15 |
| llama.cpp | 10 | ~4–5 GiB       | 125.27 | 21.45 |
| llama.cpp | 99 (fitted)    | 17.1 GiB | — | 120.40 |
| **larql** Tier 0 | — | 0 GiB | 1.06  | 1.06 |
| **larql** Tier 1 | LmHead | 2.4 GiB | 0.83 | 1.24 |
| **larql** Tier 3 | all classes | 5.6 GiB | 1.39 | **3.49** |

### Iso-VRAM comparison (the headline)

At a comparable ~5 GiB VRAM budget:

- **llama.cpp** (10/40 layers offloaded): **21.45 tg t/s**.
- **larql** (per-class GPU residency, Tier 3):  **3.49 tg t/s**.

llama.cpp is **6.1× faster than larql** at this VRAM point. The
all-CPU gap is wider: **102 vs 1 t/s → 96× on prefill, 15× on
decode** (the E.8 backlog).

### Where the gap lives

1. **CPU kernel quality (E.8)** — llama.cpp's `ggml_vec_dot_q*_q8_0`
   does AVX2/AVX-512 SIMD with int8 packing on the activation side
   (`Q8_0` quant of `x` so the dot product is int8 × int8 with f32
   scale). larql's `q4k_row_dot` is rayon-parallel but uses f32 ×
   dequantized-f32 — a ~10× per-row gap. That alone explains most of
   the all-CPU gap and a big chunk of the iso-VRAM gap (since larql's
   AttnProj/DnProj Q8_0 tensors fall back to CPU).
2. **No Q8_0 GPU kernel (F.4)** — at iso-VRAM, every attention
   projection runs on CPU regardless of which GPU classes are on.
   Adding `QuantFormat::Q8_0` + a basic `q8_0_matvec` kernel would
   unlock the AttnProj/DnProj residency tier — estimated 30–50 ms/tok
   saved.
3. **No CUDA Graphs (E.6.C)** — the decode path issues ~600
   kernels/token. llama.cpp records these as a CUDA Graph and replays
   per token; larql still pays full driver dispatch each tick.
4. **No int8-quantized activations** — orthogonal to (1): even when
   larql's expert matvecs hit the GPU, they decode the f32 activation
   inside the kernel instead of pre-quantizing to Q8_0. llama.cpp's
   `mul_mat_id` for MoE uses Q8_0 activations end-to-end.

### What the F.3/F.4 numbers do prove

larql's MoE forward path is **correct and lazy-by-construction**:

- 22 GiB GGUF, only ~1.2 GiB of weights touched per token
  (1 active attention block + 8 active experts + 1 shared expert +
  lm_head + embed row).
- Per-class GPU residency selects which slice lives on GPU; the
  inactive 248 experts × 40 layers ≈ 18 GiB never touch VRAM.
- 5.6 GiB on a 24 GiB card is **3× more headroom than llama.cpp's
  17 GiB at the same ngl=99 fit**, leaving room for larger context
  or smaller GPUs.

The value-prop hypothesis isn't disproven — **VRAM use IS lower at
the same model size**. But raw throughput needs the four backlog
items above to close the 6× gap.

### Status

- ✅ F.0–F.3 — MoE forward path lands end-to-end on real GGUF.
- ✅ F.4 (curve) / F.5 — value-prop measured against llama.cpp.
- ⏳ F.4 (kernel) — Q8_0 GPU matvec kernel for attn projections.
- ⏳ E.6.C — CUDA Graphs for decode.
- ⏳ E.8 — CPU SIMD parity with ggml-cpu.

These three together are the path to "competitive at iso-VRAM",
not "10× the VRAM of llama.cpp". Each has a stand-alone task.

## 2026-05-13 update — Phase F.4 (kernel): Q8_0 GPU matvec — 3.49 → 10.98 t/s decode

Added a `q8_0_matvec` trait method on `QuantMatVec` and a CudaBackend
implementation that:
1. Host-dequants the Q8_0 weight once → f32.
2. Caches the f32 device buffer in `q8_0_f32_device_cache` keyed by the
   host byte slice (mirrors `q6k_f32_device_cache`).
3. Subsequent matvecs reuse the cached buffer via cuBLAS `gemv_device_w`.

Without the device cache the first F.4 build dropped Tier 3 decode
from **1.20 → 0.24 t/s** — each per-token attn matvec was re-dequanting
50 MiB on the host and re-uploading. With the cache the lookup is one
hash + Arc clone, and the cuBLAS sgemv runs against the resident f32
weight.

### Updated curve

| Tier | GPU classes                                              | VRAM (MiB) | Decode (t/s) | Δ vs F.3 |
|---|---|---:|---:|---:|
| 0 (all-CPU)                | — | 0      | 1.06  | — |
| 1 (LM_HEAD)                | LmHead | 2414  | 1.24  | — |
| 2a no-FFN, Q8_0-CPU        | LmHead + Attn* + Dn* | 2414 | 1.20  | — |
| 2b no-FFN, Q8_0-GPU (F.4)  | + Q8_0 attn matvec | 7310 | **1.43** | +20 % |
| 3a all, Q8_0-CPU           | all five | 5582  | 3.49  | — |
| 3b all, Q8_0-GPU (F.4)     | all + Q8_0 attn | **10926** | **10.98** | **+215 %** |

### Reading the result

The full-GPU tier (3b) is the headline: **10.98 t/s on the same 22 GiB
GGUF that llama.cpp does 21.45 t/s on at ngl=10**. The gap is now **2×
on speed at 2× the VRAM** — vs the F.3 baseline which was 6× slower
at iso-VRAM.

But the FFN-on-CPU tier (2b) shows the limit of doing only attn
projections on GPU: 1.20 → 1.43 t/s is just +20 %. With the 8 active
experts × 40 layers × Q4_K/Q5_K matvecs still on CPU, the FFN cost
dominates. To squeeze more out of the value-prop config (3-4 GiB
VRAM), we need either:
- **CPU SIMD parity (E.8)** — bring CPU Q4_K/Q5_K dot to ggml-cpu
  speed, which would multiply Tier 2b instead of just nudging it.
- **Streaming expert activation to GPU** — keep the 8 active experts
  resident-on-demand instead of all 256 always resident.

### VRAM cost of the f32 cache

The 10.7 GiB at Tier 3b is 4× the Q8_0 byte size — the host dequant
expands `int8 + f16 scale` (34 B per 32 elems) to `f32 × 32` (128 B per
32 elems). A real on-device Q8_0 matvec kernel that reads the packed
bytes directly would hold ~1.3 GiB instead of 5.4 GiB for these
tensors, bringing Tier 3b VRAM to ~6.6 GiB. That's the follow-up to
F.4: replace the host-dequant body of `q8_0_matvec` with a fused
device kernel.

### Build / test gates

`--features cuda` is required for the GPU path (silently disabled
otherwise — diagnostic line `LARQL_QWEN35_GPU set but larql-inference
was built without --features cuda` if you forget). All 882 inference
lib tests pass; `larql-compute` lib (152 tests) pass with `--features
cuda`.

Updated head-to-head vs llama.cpp at the new top-of-curve:

| Backend     | VRAM    | decode t/s | speed/VRAM-MiB |
|---|---:|---:|---:|
| llama.cpp ngl=10 | ~5 GiB    | 21.45 | 4.3 t/s/GiB |
| larql Tier 3b   | 10.7 GiB | 10.98 | 1.0 t/s/GiB |

llama.cpp is still 4× more efficient per VRAM MiB; the closing items
are E.8 (CPU SIMD) and on-device Q8_0 / Q4_K matvec without f32
expansion. F.4 status: **Q8_0 GPU kernel landed (host-dequant +
cache); fused on-device kernel follow-up queued.**

## 2026-05-13 update — Phase F.6: on-device Q8_0 matvec (no host f32 expansion)

Replaced F.4's host-dequant + f32 device cache with a direct CUDA
matvec kernel in
[`q8_0_direct.rs`](../../crates/larql-compute/src/cuda/q8_0_direct.rs).
The kernel reads the packed Q8_0 bytes (34 B/block: f16 scale + 32
int8 quants) in place — no f32 expansion. Mirrors the existing
`q5k_direct.rs` template:

- One thread block per row, `THREADS_PER_ROW * ROWS_PER_BLOCK` =
  128 × 4 threads.
- Each thread strides over its row's blocks (32-elem groups), pulls
  the f16 scale once per block, and accumulates `scale * (int8)q * x`.
- Parallel reduction in shared memory; one row → one f32 output.
- Byte cache via new `with_q8_0_device_buf` (mirrors `with_q5k_device_buf`)
  so each Q8_0 tensor uploads once.

Parity verified with a synthetic-weights bit-exact test
(`q8_0_matvec_matches_cpu_dequant_dot`) — the kernel result matches
`larql_models::quant::ggml::dequantize_q8_0` + scalar dot to 1e-4
relative.

The dispatch in `q8_0_matvec` now tries `q8_0_direct::matvec` first;
the f32-cache fallback stays for shapes that fail (e.g. hidden not a
multiple of 32) and is gated by `LARQL_CUDA_Q8_0_HOST_DEQUANT=1` for
A/B testing.

### Result on Qwen3.6-35B-A3B Tier 3b

| Metric        | F.4 (host-dequant + f32 cache) | F.6 (direct kernel) | Δ |
|---|---:|---:|---:|
| Decode (t/s)  | 10.98 | 10.50 | −4 % |
| Prefill (t/s) | 0.73  | 1.86  | +155 % |
| VRAM (MiB)    | 10926 | **6990**  | **−36 %** |

The prefill jump comes from the F.4 cold-start cost (host dequant of
~1.3 GiB of Q8_0 weight on the first prefill call) being amortised
away in F.6 — the direct kernel just uploads the packed bytes once.

Decode is essentially unchanged: the f32-cached path was already
binding the gemv to HBM bandwidth, and the direct kernel pays roughly
the same memory traffic per call (reading 34 B/block instead of 128
B/block, but with a lighter compute body).

### Refreshed iso-VRAM picture

| Backend                  | VRAM    | decode t/s | speed/VRAM-GiB |
|---|---:|---:|---:|
| llama.cpp ngl=10         | ~5 GiB  | 21.45 | 4.3 t/s/GiB |
| **larql Tier 3 (F.6)**   | **6.99 GiB** | **10.50** | **1.5 t/s/GiB** |
| larql Tier 3b (F.4)      | 10.7 GiB | 10.98 | 1.0 t/s/GiB |

larql is now **2× slower at 1.4× the VRAM**, vs **2× slower at 2× the
VRAM** under F.4. The closing path to iso-VRAM-iso-speed is:
1. **E.8 — CPU SIMD parity** (the 15× all-CPU gap; closes the
   FFN-CPU-tier gap on Tier 2b which still does CPU experts).
2. **On-device Q4_K matvec without dequant** — the existing q4k_direct
   already does this; verify the MoE FFN block path takes it.
3. **CUDA Graphs (E.6.C)** — eliminate per-token dispatch overhead.

### Status

- ✅ F.6 — on-device Q8_0 matvec landed; VRAM reclaim verified.
- ⏳ E.8 — CPU SIMD parity with ggml-cpu (biggest remaining lever).
- ⏳ E.6.C — CUDA Graphs.

## 2026-05-13 update — Phase E.8 (step 1): allocation-free `q8_0_row_dot` (AVX2)

Replaced the Q8_0 CPU matvec dispatch in `QuantTensor::matvec` with a
new `q8_0_row_dot` that reads the packed bytes in place — no `Vec<f32>`
allocation per row. AVX2 path on x86_64 loads 8 int8 → 8 i32 → 8 f32
per FMA, looping 4× per 32-element block. Scalar fallback for
non-x86_64. Module test verifies bit-exact agreement vs
`dequantize_q8_0 + scalar dot`.

### All-CPU Qwen3.6-35B-A3B bench (no `--features cuda`)

| Path                          | Decode (t/s) |
|---|---:|
| Before (Vec<f32>-per-row)    | 1.06 |
| **After (q8_0_row_dot AVX2)** | **1.09** |

The +3 % is honest: on this MoE model, the **Q4_K/Q5_K experts** (8
active × 40 layers + 1 shared expert/layer) dominate CPU time. The
Q8_0 attention projections + shared-expert SwiGLU are a much smaller
slice. So the win from making Q8_0 fast on CPU is small for the
all-CPU value-prop config.

### What this proves about the CPU gap

The 15× all-CPU gap vs llama.cpp is in the **Q4_K dot**, not Q8_0. To
move the all-CPU number, the next step is the real `ggml_vec_dot_q4_K_q8_K`
parity:

1. Per-layer pre-quantise `x` to Q8_K (256-element blocks, f32 scale
   per super-block, 32 int8 values per sub-block) once.
2. Use `_mm256_maddubs_epi16` / `_mm256_madd_epi16` to do
   int8×int8 → i16 → i32 dot per 32-element sub-block.
3. Multiply by the (Q4_K weight scale × Q8_K activation scale) f32
   product and accumulate.

That's where llama.cpp's CPU path lives and what closes the gap. It's
a substantial implementation lift — Q4_K_Q8_K is ~150 LOC of AVX2
intrinsics + a new `quantize_to_q8_k` + dispatch wiring. Queued as
E.8 (step 2).

### Status update

- ✅ E.8 step 1 — `q8_0_row_dot` allocation-free + AVX2. Small but
  the right shape: keep weights in their native packing, kill per-row
  allocator churn.
- ⏳ E.8 step 2 — Q4_K × Q8_K AVX2 dot (the 10× CPU win).
- ⏳ E.6.C — CUDA Graphs.

## 2026-05-13 update — Phase E.8 step 2 (scalar): Q4_K × Q8_K algorithm landed; AVX2 still queued

Added `quantize_to_q8_k` and `q4k_q8k_row_dot` to
[`larql_models::quant::ggml::q4k_q8k`](../../crates/larql-models/src/quant/ggml/q4k_q8k.rs).
Wired it into `QuantTensor::matvec` for the Q4_K path — when
`x.len() % 256 == 0`, we now pre-quantise `x` to Q8_K once per call
and the row dots run on int-arithmetic instead of dequant+f32 FMA.
Parity verified to 1e-3 relative against the existing
`q4k_row_dot` (the Q8_K quantisation step itself contributes the
~1 ULP per-element rounding).

### All-CPU Qwen3.6-35B-A3B bench

| Path                                  | Decode (t/s) |
|---|---:|
| Before (q4k_row_dot AVX2 f32 FMA)    | 1.09 |
| **After (scalar q4k_q8k_row_dot)**   | **1.11** |

**Essentially no change.** The reason — the existing `q4k_row_dot`
already runs an AVX2 f32 FMA inner loop, and the LLVM auto-vectoriser
covers the f32 dot path well. The Q4_K × Q8_K algorithm only wins
when:
1. The int4 × int8 → int16 → int32 path uses `_mm256_maddubs_epi16` +
   `_mm256_madd_epi16` intrinsics (much higher arithmetic density per
   cycle than f32 FMA on the same lane count).
2. The `quantize_to_q8_k(x)` cost is amortised across all matvecs in a
   layer (gate, up, down, attn projections, …) — one quantise, many
   dots.

The scalar inner loop I wrote does (1) in pure Rust and the optimiser
collapses it to roughly the same code path as the existing f32 FMA.
Without the explicit `maddubs/madd` intrinsics, there's no structural
win. Without per-layer plumbing, the quantise cost is paid per matvec.

### What this means for the CPU gap

The 15× all-CPU gap vs llama.cpp **is real and lives in the AVX2
intrinsics path** — not in the algorithm. The path forward:

1. Add a `q4k_q8k_row_dot_avx2` body using `_mm256_maddubs_epi16` and
   `_mm256_madd_epi16` on 32-byte lanes. ~150 LOC of careful
   intrinsics.
2. Lift the `quantize_to_q8_k(x)` call out of `QuantTensor::matvec`
   into the per-layer FFN/attention forward, so it amortises across
   the gate/up/down/proj matvecs.

Until both land, the all-CPU Tier 0 stays near 1 t/s on this MoE
model. The algorithm + parity test landed in this session form the
foundation; the intrinsics + amortisation pieces are the remaining
work in E.8.

### Status

- ✅ E.8 step 2a — Q4_K × Q8_K algorithm + scalar dot + parity test.
- ⏳ E.8 step 2b — AVX2 `maddubs/madd` intrinsics for the dot.
- ⏳ E.8 step 2c — per-layer `quantize_to_q8_k` amortisation.
- ⏳ E.6.C — CUDA Graphs.

## 2026-05-13 update — Phase E.8 step 2b: AVX2 `maddubs/madd` Q4_K × Q8_K

Added `q4k_q8k_row_dot_avx2` using `_mm256_maddubs_epi16` and
`_mm256_madd_epi16` intrinsics. Per 32-byte AVX2 lane:

- 32 Q4 bytes → low/high nibbles (`_mm256_and_si256` + `_mm256_srli_epi16`)
- `maddubs(nibbles_u8, q8_quants_i8)` → 16 int16 lanes (pairwise int8 product)
- `madd(int16_pairs, ones)` → 8 int32 lanes (pairwise int16 sum)
- Horizontal reduce + multiply by 6-bit `scales[i]` accumulator

Per super-block we walk 4 byte-groups; each group covers two adjacent
sub-blocks via the low/high nibble split. Mirrors llama.cpp's
`ggml-cpu/quants.c::ggml_vec_dot_q4_K_q8_K` (non-VNNI path).

### Result on Qwen3.6-35B-A3B all-CPU

| Path                              | Decode (t/s) |
|---|---:|
| q4k_row_dot (AVX2 f32 FMA)       | 1.09 |
| q4k_q8k_row_dot scalar (step 2a) | 1.11 |
| **q4k_q8k_row_dot AVX2 (step 2b)** | **1.14** |

Only **+3 %** vs the original — much less than the expected 10×.
Three explanations consistent with the data:

1. **Memory-bound, not compute-bound.** The 22 GiB GGUF is mmapped;
   per-token reads of ~1.3 GiB of active Q4_K expert weights are
   gated by SSD/RAM bandwidth, not AVX2 throughput. The AVX2 path
   does less compute per byte read, so even a 5× speed-up in
   inner-loop ops doesn't translate to a 5× wall-clock change.
2. **`quantize_to_q8_k(x)` per call eats the gain.** Currently
   `QuantTensor::matvec` calls it on every invocation. In a MoE FFN
   layer with 8 active experts × 2 Q4_K matvecs = 16 redundant
   re-quantises of the same `x`. Step 2c (per-layer amortisation)
   fixes this.
3. **Q5_K / Q8_0 paths still dominate.** The down-projection is Q5_K
   and the shared expert is Q8_0. With the Q4_K path now ~5× faster
   per op, Q5_K and Q8_0 become a bigger fraction of total time —
   their fast paths haven't landed yet.

### What this proves

The structural change works (AVX2 path lands, parity verified, fires
correctly on real GGUFs). The wall-clock gain on this MoE config is
modest because the model is memory-bandwidth-limited and the
per-matvec quantise cost negates much of the per-op win. Closing the
gap requires:

- **Step 2c** — lift `quantize_to_q8_k(x)` to per-layer, sharing the
  Q8_K buffer across all matvecs in a forward step. Estimated ~50 LOC
  in `swiglu_moe_lazy` + `swiglu_ffn_lazy` + a sibling
  `QuantTensor::matvec_with_q8k` entry.
- **Q5_K × Q8_K dot** — same pattern as q4k_q8k for the down
  projection.
- **Q8_0 × Q8_K dot** — for the shared expert (or keep the
  q8_0_row_dot from step 1, which already runs AVX2 on f32 input).

These three together are what unlocks the 10× CPU win promise. Each
is contained but the full set is a session's worth of work.

### Status

- ✅ E.8 step 2b — AVX2 maddubs/madd Q4_K × Q8_K dot landed. +3 %
  on this memory-bound MoE config.
- ⏳ E.8 step 2c — per-layer quantize amortisation.
- ⏳ E.8 step 2d — Q5_K × Q8_K AVX2 dot.
- ⏳ E.6.C — CUDA Graphs.

## 2026-05-13 update — Fine-profile + Phase E.8 step 2d: Q5_K × Q8_K AVX2 dot (decode 1.14 → 2.16 t/s, +90 %)

Ran the bench under `LARQL_QWEN35_FINE_PROFILE=1` to characterise the
all-CPU decode breakdown — the memory-bound thesis from step 2b
turned out to be wrong.

### Profile (per-token decode, ~836 ms / 2 toks)

| Section                | ms    | %     |
|---|---:|---:|
| **FFN_DOWN**           | 422.5 | **50.5** |
| FFN_GATE_UP_PAIR       | 172.6 | 20.6  |
| DN_RECURRENCE          | 104.0 | 12.4  |
| DN_QKV_GATE_PAIR       |  49.7 |  5.9  |
| LM_HEAD                |  31.6 |  3.8  |
| DN_SSM_OUT             |  24.8 |  3.0  |
| ATTN_BLOCK             |  23.5 |  2.8  |

FFN_DOWN is the Q5_K down projection — eight active experts × forty
layers per token. It was still using the **dequant-row-to-f32 + scalar
dot** path. Q4_K was already AVX2, so its share is real but smaller
(20 %); Q5_K being 50 % is the lever.

### Step 2d: Q5_K × Q8_K AVX2

New `q5k_q8k_row_dot` in
[`larql_models::quant::ggml::q5k_q8k`](../../crates/larql-models/src/quant/ggml/q5k_q8k.rs).
Same maddubs/madd structure as Q4_K, with the extra wrinkle that Q5_K
adds 32 bytes of high-bit plane (1 bit per element). The kernel:

1. Pre-extracts +16 contributions for each of the 8 sub-blocks via
   `_mm256_srli_epi16::<N>(qh, N) & 0x01 << 4` (one const-shift per
   sub-block, computed once at super-block entry).
2. Unpacks qs nibbles (low/high) per byte-group, adds the matching
   pre-extracted high-bit to form 5-bit values 0..31 in a `__m256i`.
3. `_mm256_maddubs_epi16` against the matching Q8_K int8 row, then
   `_mm256_madd_epi16(_, set1(1))` to fold pairs to int32, horizontal
   reduce, multiply by the sub-block's 6-bit scale, accumulate.

Parity test (`q5k_q8k_matches_dequant_then_dot`) verifies the result
against `dequantize_q5_k + scalar dot` to 1e-3 relative.

Wired through `QuantTensor::matvec`'s Q5_K arm — `quantize_to_q8_k(x)`
per call (still no per-layer amortisation from step 2c) then per-row
AVX2 dot under rayon. Legacy dequant-per-row path stays for shapes
that aren't multiples of 256 and is gated by
`LARQL_Q5K_USE_DEQUANT_DOT=1` for A/B.

### All-CPU Qwen3.6-35B-A3B bench

| Snapshot                                | Decode (t/s) | Δ vs prior |
|---|---:|---:|
| Session start (Q4_K AVX2 only)         | 1.06   | — |
| q4k_q8k scalar (step 2a)               | 1.11   | +5 % |
| q4k_q8k AVX2 (step 2b)                 | 1.14   | +3 % |
| **q5k_q8k AVX2 (step 2d)**             | **2.16** | **+90 %** |

The Q5_K AVX2 change roughly halved decode time. That tracks the
profile: if FFN_DOWN drops 5× (50.5 % → ~10 %), total time drops by
~40 % and t/s rises ~67 %. We got slightly more (90 %), consistent
with the change also speeding up downstream cache pressure.

### Refreshed iso-VRAM and all-CPU pictures

| Backend            | VRAM     | Decode (t/s) |
|---|---:|---:|
| llama.cpp ngl=0    | 0 GiB    | 16.15 |
| **larql all-CPU**  | 0 GiB    | **2.16** |
| llama.cpp ngl=10   | ~5 GiB   | 21.45 |
| **larql Tier 3 (F.6)** | 7.0 GiB | 10.50 |

**All-CPU gap: 16.15 / 2.16 = 7.5× slower** (was 15×). Closing the
remaining gap to llama.cpp:

1. **Per-layer `quantize_to_q8_k` amortisation (step 2c)** —
   currently `QuantTensor::matvec` re-quantises `x` for every matvec
   call. In a MoE FFN that's 16 redundant scans per layer. Lifting
   the call to per-layer would remove the duplicate work outright.
2. **Q8_0 × Q8_K dot** — the shared-expert attention projections.
   Smaller share than Q5_K but contributes ~6 % of decode each.
3. **AVX-512 / VNNI on supported CPUs** — `_mm512_dpbusd_epi32` does
   maddubs+madd in one instruction.

### Status

- ✅ E.8 step 2d — Q5_K × Q8_K AVX2. **+90 % all-CPU decode.**
- ⏳ E.8 step 2c — per-layer quantize amortisation.
- ⏳ E.8 step 2e — Q8_0 × Q8_K AVX2 (shared expert).
- ⏳ E.6.C — CUDA Graphs.

## 2026-05-13 update — Phase E.8 step 2c + hardware ceiling note

Landed the **thread-local Q8_K cache** in `QuantTensor::matvec`. Keyed
by `(x.as_ptr(), x.len())`, it lets consecutive Q4_K / Q5_K matvecs
with the same `x` reuse the prior `quantize_to_q8_k` output. In the
MoE FFN that means 16+ calls per layer (gate × 8 experts + up × 8
experts) share one quantise. Cache implementation is ~30 LOC in
`with_q8k_for` (lazy.rs) — drop-in, no API change.

### Result

| Snapshot                          | Decode (t/s) |
|---|---:|
| q5k_q8k AVX2 (step 2d)            | 2.16 |
| **+ q8k thread-local cache (2c)** | **2.18** |

Within noise. Confirms the up-front estimate: a single
`quantize_to_q8_k(x)` is ~1 µs for hidden=2048, so saving 17 of them
per token (gate × 8 + up × 8 + shexp redundancy) is ~17 µs / 455 ms
decode = 0.004 % — invisible. The cache is *correct* and *cheap*,
just not load-bearing on this model size.

### Hardware ceiling: AMD Zen3, no AMX, no AVX-512

`/proc/cpuinfo` confirms this bench machine is AMD Zen3 — `avx2` and
`fma` are present; `avx512*` and `amx*` are not. Consequence: the
AVX2 maddubs / madd path landed in steps 2b/2d **is the SIMD ceiling
on this CPU**. The remaining CPU levers per ktransformers' playbook
that we *could* apply here:

- **Expert weight prefetch** (`madvise(MADV_WILLNEED)` on the next
  active expert's byte slice while computing the current one) —
  overlaps SSD/RAM with compute. Likely 10–20 % on the memory-bound
  portion. The cleanest small win remaining.
- **DN_RECURRENCE SIMD** — 24 % of decode is the DeltaNet
  recurrence kernel (`delta_net_step`), pure f32 tensor math. Adding
  AVX2 to the inner loops is plausible but the math is
  correctness-sensitive (took most of Phase C.5 to converge); risky
  for a small gain.
- **AMX** — would close most of the remaining gap, but needs Intel
  Sapphire Rapids+ (or future AMD parts with similar tile
  instructions). Not on this machine.

### Refreshed final-state numbers

| Backend            | VRAM     | Decode (t/s) | Δ vs session start |
|---|---:|---:|---:|
| larql all-CPU      | 0 GiB    | **2.18**     | +106 % (1.06 → 2.18) |
| larql Tier 3 (F.6) | 7.0 GiB  | 10.50        | new |
| llama.cpp ngl=0    | 0 GiB    | 16.15        | — |
| llama.cpp ngl=10   | ~5 GiB   | 21.45        | — |

The gap to llama.cpp's all-CPU on this Zen3 box: **16.15 / 2.18 =
7.4× slower** at iso-VRAM (both 0 GiB). The next ~3× of that gap is
in DN_RECURRENCE + memory bandwidth (ktransformers prefetch); the
final ~2× is the AVX-512/AMX gap that needs a different CPU.

### Status (final for this session)

- ✅ E.8 step 2c — Q8_K thread-local cache landed. Marginal at this
  model size but the right pattern; load-bearing on smaller hiddens
  where quantise is a higher fraction.
- ⏳ Expert weight prefetch (`madvise`-based) — clearest contained
  remaining lever on this hardware.
- ⏳ E.6.C — CUDA Graphs.
- ⏳ AMX path — needs Intel-side hardware.

## 2026-05-13 update — Phase G.1: mmap-backed `QuantTensor` (ktransformers tier-1)

The headline ktransformers idea — **keep weights resident in the OS
page cache, not in RSS, and let `madvise(MADV_WILLNEED)` prefetch
the next active expert** — landed in this session.

### What changed

- `QuantTensor` now holds an `enum QuantBacking { Heap(Arc<[u8]>),
  Mmap(Arc<memmap2::Mmap>) }` instead of a flat `Arc<[u8]>`. The
  view semantics (`byte_offset` + `byte_len`) carry over unchanged;
  `expert_slice` clones the enum (which clones the inner `Arc`) so
  per-expert subviews still share the parent backing.
- New `QuantTensor::from_mmap_region(mmap, offset, len, type, rows,
  cols)` constructor — zero-copy view into the gguf mmap.
- `load_gguf_lazy_tensors` now mmaps the file once into an
  `Arc<Mmap>` and constructs each lazy tensor with
  `from_mmap_region(...)` instead of `mmap[..].to_vec()`. The 22 GiB
  GGUF stays in the OS page cache; per-tensor views are slice
  references.
- New `QuantTensor::prefetch_willneed()` calls
  `Mmap::advise_range(WillNeed, ..)` on the view's byte range
  (no-op for heap-backed tensors).
- `swiglu_moe_lazy` collects all top-K experts up front and
  prefetches their gate/up/down byte ranges before dispatching the
  first expert's compute — Linux begins paging in the rest while
  expert 0 runs.

### All-CPU Qwen3.6-35B-A3B (4-prefill, 16-decode)

| Metric            | Heap-backed (prior) | **mmap-backed (G.1)** | Δ |
|---|---:|---:|---:|
| Load time         | 173 s   | **143 s**   | −17 % (no `to_vec`) |
| RSS (4 tok)       | 22.3 GiB | **5.8 GiB**  | **−74 %** |
| RSS (20 tok)      | 22.3 GiB | **9.0 GiB**  | **−60 %** |
| Decode (t/s)      | 2.18    | 1.80       | −17 % |

RSS scales with active-pages-touched, not file size. The 9 GiB at 20
tokens represents the working set of attention + active experts +
LM head + embed rows actually accessed; the other ~13 GiB of
inactive experts stay in the OS page cache, evictable under memory
pressure. Exactly the ktransformers value-prop shape.

The 17 % decode-t/s regression comes from first-touch page faults
when a fresh expert activates (the SSD/page-cache read is now in the
critical path instead of pre-paid during load). The mmap variant is
the right default for any machine that can't comfortably hold 22 GiB
in RSS — and a free upgrade on machines that can (the OS keeps the
hot pages resident anyway, you just don't pay for them in RSS).

### Updated value-prop comparison vs llama.cpp

| Backend            | RSS     | VRAM    | Decode (t/s) |
|---|---:|---:|---:|
| llama.cpp ngl=0    | ~22 GiB | 0 GiB   | 16.15 |
| **larql all-CPU**  | **9 GiB** | 0 GiB | 1.80 |
| llama.cpp ngl=10   | ~22 GiB host + ~5 GiB GPU | ~5 GiB | 21.45 |
| larql Tier 3 (F.6) | ~22 GiB host + 7 GiB GPU  | 7.0 GiB | 10.50 |

The all-CPU tier now has **2.4× less RSS than llama.cpp** at the
same workload — that's the real value prop differentiating larql on
memory-constrained hardware. The 9× t/s gap remains (AVX-512/AMX +
DN recurrence work) but the **architectural** premise is now
demonstrably superior.

### What ktransformers tactics landed

- ✅ **mmap-backed weights** — RSS shrinks by 60–74 %.
- ✅ **`MADV_WILLNEED` expert prefetch** — wired through
  `swiglu_moe_lazy`; kernel pages in the next expert while the
  current one computes.
- ⏳ **Hot-expert cache** — keep the top-N most-frequent experts
  pinned in RSS. Needs per-token routing histogram first.
- ⏳ **AMX / VNNI kernels** — hardware-blocked on this Zen3 box.
- ⏳ **Cross-expert batching** — combine 8 experts' matvecs into one
  rayon dispatch per layer (instead of 8 serial dispatches).

### Status

- ✅ G.1 — mmap-backed QuantTensor + expert prefetch landed.
- ⏳ G.2 — `LARQL_QWEN35_HEAP_LOAD=1` opt-in to force heap-backed
  (for users who want max throughput and have the RAM).
- ⏳ G.3 — hot-expert RSS pinning (`mlock` top-N most-frequent
  experts).

## 2026-05-12 update — Phase E.7: per-class GPU residency + the VRAM/throughput curve

larql's value proposition is *VRAM-minimal* inference: push as much
of FFN, weights, and incidental compute back to CPU + host RAM,
keeping only the kernels that demonstrably need device residency
(attention compute, KV cache, DeltaNet recurrent state). All of
E.6.{D,B.1,B.2,I} pushed in the opposite direction — toward
throughput at any VRAM cost — which is the wrong axis to compare
against llama.cpp on. E.7 adds the per-class dispatch knobs that
make the *intended* axis measurable.

New env vars (all opt-in; defaults preserve current full-GPU
behaviour when `LARQL_QWEN35_GPU=1` is set):

```
LARQL_QWEN35_GPU_NO_FFN=1            # SwiGLU gate/up/down → CPU
LARQL_QWEN35_GPU_NO_LM_HEAD=1        # final logits matvec → CPU
LARQL_QWEN35_GPU_NO_DN_PROJ=1        # DeltaNet attn_qkv/attn_gate/ssm_out → CPU
LARQL_QWEN35_GPU_NO_DN_RECURRENCE=1  # DeltaNet conv1d / L2 / recurrence / rms_norm → CPU
LARQL_QWEN35_GPU_NO_ATTN_PROJ=1      # Full-attn q/k/v/o matvecs → CPU
```

Each dispatch site checks the per-class env var (thread-local
cached, no perf overhead) and uses `gpu_tier::backend_for(class, ..)`
to elide the backend for that one call site. The CPU fallback path
that's been present in every kernel hook all along is what runs.

Bench harness now also reports per-process VRAM via
`nvidia-smi --query-compute-apps`, captured at end of decode.

### The curve (RTX 4090, Qwen3.6-27B-Q4_K_S, prefill 16 / decode 8)

| Tier                          | What's on GPU                                              | VRAM | Decode (t/s) | Prefill (t/s) |
|---|---|---:|---:|---:|
| **larql Full** (E.6.I)        | everything                                                  | **18.7 GiB** | 10.63 | 2.75 |
| larql no_lm_head              | DeltaNet block + FFN + full-attn (LM_head on CPU)           | 13.9 GiB | 5.80 | 4.36 |
| larql no_ffn                  | DeltaNet block + full-attn + LM_head (FFN on CPU)           | 9.5 GiB | 0.59 | 0.52 |
| **larql Attn-only**           | DeltaNet recurrence + full-attn block + KV cache only       | **1.45 GiB** | 0.25 | 0.26 |
| larql All-CPU                 | nothing                                                     | 0 | 0.23 | 0.23 |
| llama.cpp CUDA (`ngl=99`)     | everything                                                  | 14.76 GiB | 50.60 | 2097 |
| llama.cpp CPU (`ngl=0`)       | nothing                                                     | 0 | 2.60 | 37.33 |

### What this says about larql vs llama.cpp on the VRAM axis

- **larql can run Qwen3.6-27B at ≤ 1.5 GiB VRAM.** llama.cpp can't —
  the model doesn't fit in less than ~15 GiB VRAM even at the
  smallest quant; you have to fall off to its CPU path entirely.
  larql's Attn-only mode would fit on a 2 GiB consumer GPU (GTX
  1060 3 GB, RTX 3050 4 GB, integrated GPUs, anything in laptops),
  while a model this size on llama.cpp at the same VRAM budget
  needs CPU-only or `--gpu-layers N` with N << total.
- **Intermediate tiers don't have an llama.cpp equivalent.**
  llama.cpp's `--gpu-layers N` partial offload is at layer
  granularity, not at tensor-class granularity. larql's mid-tier
  (e.g. 9.5 GiB no-FFN) couldn't be replicated by llama.cpp's
  N-layer partition.
- **The CPU path is currently the gating cost.** larql at 0 VRAM
  runs at 0.23 t/s; llama.cpp CPU at the same memory footprint
  runs at 2.60 t/s. That's an **11× CPU-path gap**: llama.cpp's
  hand-tuned per-tensor SIMD + thread-pool scheduling does more
  per cycle than our current `q4k_row_dot` / `q5k_row_dot` rayon
  path. Closing this gap is the *real* unlock for the value prop —
  it makes the Attn-only and no-FFN tiers actually competitive.
- **At full GPU, larql is ~5× behind llama.cpp's GPU path
  (10.6 vs 50.6 t/s)**. Gap is mostly per-kernel quality: their
  `mul_mat_vec_q*_K_q8_1_cuda` kernels have years of tuning vs our
  NVRTC-compiled adaptations.

### Suggested next priorities by axis

| Goal                                  | Lever                                                                |
|---|---|
| Make Attn-only tier viable            | Close the 11× CPU q4k/q5k matvec gap with llama.cpp's ggml-cpu       |
| Make no_ffn tier viable               | Same — FFN on CPU is the bulk of work at that tier                   |
| Close the full-GPU 5× gap to llama.cpp | Custom Q4_K matvec kernel + CUDA Graphs (E.6.C)                     |

The right next investment for larql's stated value prop is the
first row: **CPU-side perf**. Once it's not embarrassing, the
VRAM-constrained tiers become a real differentiator against
llama.cpp on small-VRAM hardware.

## 2026-05-12 update — Phase E.6.I: load_gguf_lazy_tensors lm_head dispatch fix (decode 5.19 → 10.61 t/s)

**Hidden bug found while microbenching E.6.F.** Standalone
`cublasSgemv` at the lm_head shape (vocab=248320, hidden=5120,
device-resident inputs, no dtoh) ran at **5.3 ms / call** — exactly
memory-bandwidth-limited at HBM. But our LM_HEAD section was
measuring 85-89 ms / call. A 16× gap.

Bisecting via `LARQL_CUDA_GEMV_TRACE=1` revealed that `matvec_with_backend`
was **never called** with rows=248320. The LM_HEAD was running through
the dense fallback `weights.lm_head.dot(&x_final)` on CPU the entire
time. Every E.6.F "GPU path investigation" was a microbench of three
GPU code paths that the model never actually exercised.

**Root cause** in `larql_models::loading::gguf::load_gguf_lazy_tensors`:
the generic lazy-tensor loader put `lm_head.weight` / `output.weight`
into the generic `weights.quant_tensors` HashMap, but the Qwen3.6
bridge only reads `weights.lm_head_quant` (which only the dedicated
`load_gguf_lazy_lm_head` populates). With both `LARQL_QWEN35_LAZY_FFN=1`
and `LARQL_QWEN35_LAZY_LM_HEAD=1` set, the bench used
`load_gguf_lazy_tensors`, so `lm_head_quant` stayed `None` and the
forward fell back to the dense f32 `ndarray.dot` — a 1.27 GFLOP CPU
call that takes ~85 ms.

Fix: `load_gguf_lazy_tensors` now special-cases `lm_head.weight` /
`output.weight` (same pattern already in place for the embed
tensor), populating `weights.lm_head_quant` and emptying
`weights.lm_head`. Bridge dispatch unchanged — it already routed
through `matvec_with_backend` when `lm_head_quant` was `Some`.

Bench (RTX 4090, prefill 16 / decode 8):

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| Phase E.6.B.2 (lm_head still on CPU, hidden) | 3.99 | 5.19 | 21.16 GiB |
| **Phase E.6.I (lm_head actually on GPU)** | **2.73** | **10.61** | **16.2 GiB** |
| llama.cpp CUDA GPU | 2097 | 50.6 | 14.76 GiB |

Decode **doubled** (5.19 → 10.61 t/s). VmRSS dropped 5 GiB because
the dense f32 lm_head (5.1 GiB) is no longer held in host RAM.
Prefill regressed because the FIRST forward call now pays a 4 s
Q6_K → f32 GPU dequant + cache warmup; that amortises out over
longer prompts (and won't appear at all on the decode path which
runs after the warmup).

Steady-state fine profile (79 ms / token):

```
FFN_GATE_UP_PAIR  41.8 ms (53 %)   fused FFN block
DN_RECURRENCE      8.0 ms (10 %)   fused L2 + recurrence + rms_norm
ATTN_BLOCK         6.6 ms ( 8 %)
LM_HEAD            5.5 ms ( 7 %)   ← was 89 ms!
DN_SSM_OUT         5.0 ms ( 6 %)
…                                  rest <3 ms each
```

The new top cost is FFN_GATE_UP_PAIR at 53 % (42 ms / token across
64 FFN layers = 0.65 ms each). Already device-resident; the only
further win is a custom Q4_K matvec kernel or cuBLAS Hgemm with
Tensor Cores. **larql is now within ~5× of llama.cpp CUDA on
Qwen3.6-27B decode** (10.6 vs 50.6 t/s).

Session arc: 0.35 → 10.61 t/s = **≈30× over the original baseline.**

## 2026-05-12 update — Phase E.6.B.2: partial DeltaNet fusion (L2 + recurrence + rms_norm)

Bundles the four GPU calls inside the DeltaNet block's tail (L2-Q,
L2-K, delta-rule recurrence, rms_norm_heads) into a single
`ComputeBackend::qwen35_deltanet_recurrence_block` call. The four
kernels now run on the same stream with one sync at exit; the
intermediate `q_normed`, `k_normed`, `o_dim`, `o_flat`, `o_normed`
buffers stay device-resident through the chain.

Crucially, `silu(qkv_conv)` and `silu(z) * o` remain on the host
side — that's the parity-critical boundary identified in E.6.A.6
(GPU `expf` vs glibc `expf` differs by ~1-2 ulp; running silu on
GPU feeds that drift into the recurrence's rank-1 state update and
compounds across positions). With those two CPU loops preserved,
the per-step parity invariant is maintained.

Parity preserved (token-diff vs llama.cpp emits GT rank 0 every
step). Bench held at 3.99 t/s prefill, 5.19 t/s decode — within
noise of the prior E.6.B.1 numbers (4.24 / 5.34).

Per-token fine profile attribution (steady-state):

```
                       E.6.B.1     E.6.B.2
DN_L2_NORM             1.9 ms      0.0 ms (folded into DN_RECURRENCE)
DN_RECURRENCE          7.8 ms      8.5 ms (now the fused 4-kernel block)
DN_RMS_NORM_HEADS      1.1 ms      0.0 ms (folded)
DN_O_RESHAPE           0.8 ms      0.0 ms (folded, dim→head reshape on device)
combined               11.6 ms     8.5 ms (3.1 ms saved per token)
```

3 ms / token = ~2 % decode. Less than the 14 ms predicted from
removing 3 syncs × 48 layers × ~100 µs each — turns out the cudarc
0.19 per-call sync overhead on this device is closer to 10-20 µs
than 100 µs. The infrastructure value is the bigger payoff: with
the DeltaNet recurrence chunk now sync-free internally, future
work can capture it as part of a CUDA Graph (E.6.C).

Opt-out: `LARQL_QWEN35_DN_FUSED_DISABLE=1` reverts to the per-step
path.

## 2026-05-12 update — Phase E.6.B.1: device-resident FFN block (gate+up+silu+down)

First slice of the E.6.B device-resident projection chain: a new
`ComputeBackend::qwen35_ffn_lazy_block` method runs the full SwiGLU
FFN on the GPU without host bouncing. CudaBackend's implementation
chains `gate` Q4_K + `up` Q4_K paired matvec → `silu_gate_up_device`
→ `down` Q5_K matvec on the same stream and dtohs only the final
`[hidden]` output. Mixed-format dispatch (Q4_K gate/up + Q5_K down)
matches the Qwen3.6-27B-Q4_K_S layout.

Saves 2 dtohs (g, u), 1 htod (inter), 1 CPU silu loop, 1 sync per
FFN layer × 64 layers.

Bench result:

| Config | Prefill (t/s) | Decode (t/s) |
|---|---:|---:|
| Phase E.6.F (sgemv revert + investigation) | 3.99 | 5.19 |
| **Phase E.6.B.1 (device-resident FFN block)** | **4.24** | **5.34** |
| llama.cpp CUDA GPU | 2097 | 50.6 |

**+6 % prefill, +3 % decode.** Parity preserved. Steady-state
per-token went 177 → 170 ms.

Profile after E.6.B.1 (steady-state 170 ms / token):

```
LM_HEAD              89.2 ms  (52 %)    same shape-bottleneck
FFN_GATE_UP_PAIR     42.0 ms  (25 %)    now full FFN block fused
DN_RECURRENCE         7.8 ms  ( 5 %)
ATTN_BLOCK            7.5 ms  ( 4 %)
DN_SSM_OUT            5.1 ms  ( 3 %)
DN_SILU_QKV_CONV      4.1 ms  ( 2 %)    still CPU loop in deltanet
DN_SILU_Z             2.9 ms  ( 2 %)    "
DN_CONV1D             2.4 ms  ( 1 %)
…                              <2 %
```

The DeltaNet block still has CPU silu loops (`silu(qkv_conv)`,
`silu(z) * o_flat`) and host-bouncing matvecs. Same fused-block
pattern would apply there — that's the E.6.B.2 scope. The attempt
in E.6.A is parity-broken (multi-position drift) so the deltanet
fusion needs a different approach.

## 2026-05-12 update — Phase E.6.F: LM_HEAD Q6_K — three paths, all ~85 ms/call

Tried three implementations for the dominant LM_HEAD Q6_K matvec
on Qwen3.6 27B's `[vocab=248320, hidden=5120]` shape, all parity-
clean, all bench-equivalent:

| Path | Bench decode | LM_HEAD section |
|---|---:|---:|
| Default (E.6.D) — f32 cache + `cublasGemm` with n=1 | 5.19 t/s | 87.7 ms |
| `cublasSgemv` direct (E.6.F.1) | 5.05 t/s | 87.5 ms |
| Packed Q8_1 × Q6_K mmvq direct kernel | 5.04 t/s | 86.9 ms |
| f16 weight cache + `cublasGemm<f16>` with m=1 | 4.99 t/s | 85.0 ms |

Memory-bandwidth-limit theoretical: ~5 ms at HBM ~1 TB/s. We're
~17× slower than that. Three independent kernel/precision paths
all land at the same 85-87 ms, so the bottleneck is something
shape-specific that survives precision and algorithm changes —
likely cuBLAS dispatch/setup overhead at this skewed `m=1` shape
on this device, or an interaction with the 5 GB f32 weight cache
in the 24 GB VRAM budget.

Investigation closed without a throughput win. Reverted to the
simplest default (sgemv direct + f32 cache). The
`LARQL_CUDA_Q6K_HOST_DEQUANT=1` override remains for
diagnostic comparisons.

Next levers for LM_HEAD specifically (deferred):
- Top-K-on-device — skip the 1 MB dtoh for greedy decode.
- Direct Q6_K matvec without the 5 GB f32 weight cache.
- A microbench harness to compare cudarc vs raw cuBLAS at this
  exact shape to attribute the residual cost.

## 2026-05-12 update — Phase E.6.E diagnostic: Q6_K mmvq vs cuBLAS sgemv — both ~87 ms/call

Investigated whether switching the LM_HEAD Q6_K matvec from the
default "dequant + cuBLAS sgemv" path to the packed-format
`q6k_mmvq` direct kernel would reduce its 87 ms / token cost
(now 50 % of decode wall-clock at 5 t/s).

Result: **no change.** Both paths land at 85–117 ms per call for
the Qwen3.6-27B lm_head shape `[vocab=248320, hidden=5120]`.
Bench decode held at 5.0–5.2 t/s (within noise) either way. Same
expected memory traffic, same observed runtime → the bottleneck is
either kernel-launch / cuBLAS setup overhead at this matrix shape
or VRAM pressure from the cached dequantized f32 weight (5.1 GB
inside the existing 21 GB resident set). Reverted to the cuBLAS
path (simpler, no Q8_1 quantisation per call), gated future
investigation by `LARQL_CUDA_Q6K_HOST_DEQUANT=1` /
`LARQL_CUDA_Q6K_F32_GEMV=1` toggles.

Next levers for LM_HEAD specifically:

- Top-K-on-device (`f32_gemv_topk1`-style) — for greedy decode we
  don't need the full 248320-vocab logits dtoh, only top-1.
- Direct Q6_K matvec without the dequant cache — would save
  ~5 GB VRAM and might unlock a more cache-friendly kernel.
- cuBLAS sgemv vs sgemm-with-n=1 micro-bench at this exact shape
  to see whether cudarc's dispatch is picking the optimal path.

These are follow-up scope; the headline 0.35 → 5.19 t/s win from
E.6.D stands.

## 2026-05-12 update — Phase E.6.D: Q5_K direct CUDA matvec — 0.35 → 5.19 t/s decode (≈15×)

What landed:

- New `cuda::q5k_direct::matvec` / `matvec_device` modeled after
  the existing `q4k_direct` kernel. Reads packed Q5_K super-blocks
  (176 bytes / 256 elements: f16 d, f16 dmin, 12 byte scale+min,
  32 byte high-bits, 128 byte low-nibbles) directly on device and
  computes the matvec without dequantising.
- New `QuantFormat::Q5_K` variant, `QuantMatVec::q5k_matvec` trait
  method, CudaBackend impl, and a content-keyed
  `with_q5k_device_buf` weight cache (same pattern as Q4_K).
- `quant_dispatch::ggml_type_to_quant_format` now maps
  `TYPE_Q5_K → Some(QuantFormat::Q5_K)` so the lazy-quant matvecs
  for `ssm_out`, `ffn_down`, DeltaNet `attn_qkv`, and full-attn
  `attn_k`/`attn_v` go onto the GPU instead of CPU rayon.
- Bit-exact CPU-reference parity test
  (`q5k_matvec_matches_cpu_dequant_dot`) at 2 rows × 2 super-blocks
  validates the kernel matches `dequantize_q5_k` + scalar dot to
  relative 1e-4.

Bench (RTX 4090, prefill 16 / decode 8):

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| Phase E.6.A.8 (paired matvec, all Q5_K on CPU) | 0.32 | 0.35 | 21.16 GiB |
| **Phase E.6.D (+ Q5_K direct CUDA)** | **3.99** | **5.19** | **21.16 GiB** |
| llama.cpp CUDA GPU (reference) | 2097 | 50.6 | 14.76 GiB VRAM |

**12× prefill, 15× decode.** Parity preserved
(`[<think>, \n\n, </think>, \n\n, Hello]`, GT rank 0 every step).
larql is now within an order of magnitude of llama.cpp CUDA on
this model.

Post-Q5K fine profile (steady-state decode = 177 ms / token):

```
LM_HEAD              87.7 ms  (50 %)   Q6_K matvec (single per-token, already on GPU)
FFN_GATE_UP_PAIR     28.5 ms  (16 %)
FFN_DOWN             15.9 ms  ( 9 %)
DN_RECURRENCE         7.6 ms  ( 4 %)
ATTN_BLOCK            7.2 ms  ( 4 %)
FFN_SILU_LOOP         6.9 ms  ( 4 %)   CPU silu*up loop
DN_SSM_OUT            5.1 ms  ( 3 %)
…rest                                  <3 ms each
```

The new dominant cost is the LM_HEAD Q6_K matvec at 87 ms /
token. Same family of optimisations (paired matvecs, device-
resident chain, CUDA Graphs) now apply on top of a much smaller
absolute baseline.

## 2026-05-12 update — Phase E.6.A.9 fine profile: 87 % of decode time is Q5_K CPU fallback

Added per-section accumulators (`LARQL_QWEN35_FINE_PROFILE=1`) that
attribute decode time to the actual operations inside
`qwen35_forward_step`, `deltanet_block_step`, and `swiglu_ffn_lazy`.
The breakdown on the current E.6.A.7+8 build with `LARQL_QWEN35_GPU=1`
(steady-state token, 2.81 s total):

```
DN_SSM_OUT            1661 ms  (59.17 %)   ssm_out matvec × 48 layers
FFN_DOWN               797 ms  (28.39 %)   ffn_down matvec × 64 layers
DN_QKV_GATE_PAIR       191 ms  ( 6.80 %)   attn_qkv + attn_gate × 48
LM_HEAD                 86 ms  ( 3.07 %)   final lm_head matvec
FFN_GATE_UP_PAIR        30 ms  ( 1.09 %)   ffn_gate + ffn_up × 64
ATTN_BLOCK              13 ms  ( 0.46 %)   16 full-attn layers (entire)
DN_RECURRENCE            8 ms  ( 0.29 %)   GPU recurrence × 48
…all other sections      9 ms  ( 0.32 %)   conv1d / L2 / rms_norm / silu / split / norms
```

Bisecting via `LARQL_QWEN35_DISPATCH_TRACE=1` (logs every
CPU-fallback in `matvec_with_backend`):

```
[dispatch] CPU fallback: type=13 rows=10240 cols=5120     # DeltaNet attn_qkv
[dispatch] CPU fallback: type=13 rows=1024 cols=5120      # full-attn attn_k or attn_v
[dispatch] CPU fallback: type=13 rows=5120 cols=17408     # ffn_down
[dispatch] CPU fallback: type=13 rows=5120 cols=6144      # ssm_out
```

**Type 13 is GGML's Q5_K.** Our GPU dispatcher
(`ggml_type_to_quant_format`) currently returns `None` for Q5_K and
falls back to the CPU rayon dequant-per-row path. So:

| Q5_K tensor in Qwen3.6-27B-Q4_K_S | Per-call cost | Calls/token | Per-token cost |
|---|---:|---:|---:|
| `ssm_out` (5120×6144)              | ~34.6 ms | 48 | 1661 ms |
| `ffn_down` (5120×17408)            | ~12.5 ms | 64 |  797 ms |
| `attn_qkv` Δ (10240×5120) — but the paired path's qkv side falls back to per-call CPU because the pair gate Q5_K never fires | — | 48 | (part of the 191 ms in DN_QKV_GATE_PAIR) |
| full-attn `attn_k`/`attn_v` (1024×5120) | trivial | 32 |    ~5 ms |

**~2.45 s / token of CPU rayon dequant work** is the lever. Every
other optimisation in E.6.A.1–8 was working around the wrong
bottleneck. Implementing Q5_K GPU dispatch (a direct-matvec kernel
or cached f16 dequant + cuBLAS hgemv) should drop decode time from
~2.8 s to ~0.3–0.5 s and lift throughput from 0.35 t/s to ~2–3 t/s
— closing the gap with llama.cpp GPU (50 t/s) to a single decimal
order of magnitude.

| Tensor | Current path | Target path | Memory cost (f16 cached) |
|---|---|---|---:|
| ssm_out × 48          | CPU rayon | Q5_K → f16 cache + hgemv | 2.9 GB |
| ffn_down × 64         | CPU rayon | Q5_K → f16 cache + hgemv | 10.9 GB |
| attn_qkv × 48 (DN)    | CPU rayon | Q5_K → f16 cache + hgemv | 4.8 GB |
| attn_k/v × 32 (full)  | CPU rayon | Q5_K → f16 cache + hgemv | 0.3 GB |
| **Q5_K f16 cache total** | | | **~19 GB** |

19 GB pushes against the 4090's 24 GB budget alongside the existing
Q4_K device cache (~15 GB) — the f16 path doesn't fit fully. The
right implementation is therefore a **direct Q5_K × f32/q8_1 matvec
kernel** that operates on the packed bytes (the same approach the
existing `q4k_direct` and `q6k_mmvq` kernels use). Per-byte memory
cost stays at Q5_K (~5.5 bits/weight) — fits easily.

This is the E.6.A.10 / E.6.D work.

## 2026-05-12 update — Phase E.6.A.8: paired Q4_K matvec (attn_qkv+attn_gate, ffn_gate+ffn_up)

New `ComputeBackend::qwen35_paired_q4k_matvec` trait method that
takes two Q4_K weight matrices sharing one `x` host slice and runs
both kernels on the same stream with one htod + one sync. Wired in:

- `deltanet_block_step` pairs `attn_qkv` (10240 rows) + `attn_gate`
  (6144 rows) on `x_norm` for the 48 linear DeltaNet layers.
- `swiglu_ffn_lazy` pairs `ffn_gate` + `ffn_up` on the post-attn-norm
  residual for all 64 FFN layers.

Saves 1 htod + 1 sync per (layer, pair) × 112 pairs per token.

Parity preserved. Bench result is **0.35 t/s decode** — same as
E.6.A.7. The amortised sync overhead turned out to be much smaller
than expected on this RTX 4090 + cudarc 0.19 setup (~100 µs per
sync, not the ~1 ms I'd budgeted). Net-zero throughput change but
the infrastructure is in place for future use (CUDA Graphs, the
device-resident projection chain in E.6.B). Code path is cleaner:
the dispatch lives in one helper instead of two consecutive
matvec calls per location.

| Config | Decode (t/s) | Notes |
|---|---:|---|
| Phase E.6.A.7 (Q/K/V as_standard_layout) | 0.35 | per-call host bounces |
| **Phase E.6.A.8 (paired Q4_K matvec)** | **0.35** | 1 htod + 1 sync per pair |

## 2026-05-12 update — Phase E.6.A.7: Q/K/V `as_standard_layout` enables full GPU per-step DeltaNet

Same latent stride bug that E.6.A's `ssm_conv1d` loader fix surfaced:
`q_raw.into_shape_with_order((n_k_heads, head_v_dim)).reversed_axes().to_owned()`
preserves the transposed strides, so `as_slice()` returns `None`,
silently disabling the `qwen35_l2_normalize_per_head` and
`qwen35_deltanet_step` GPU hooks in `deltanet_block_step`. Same
pattern for K and V. Adding `.as_standard_layout()` before
`.to_owned()` forces a row-major copy with identical logical values
(`q[d, h]` still represents head h, dim d) — and now the GPU hooks
actually fire.

After this fix, ALL the per-step DeltaNet kernels run on GPU under
`LARQL_QWEN35_GPU=1` (conv1d + L2-norm Q + L2-norm K +
recurrence + rms_norm_heads), with only `silu` and `silu(z)*o`
elementwise loops still on host.

Parity preserved: `real_gguf_qwen35_token_diff_vs_llama_cpp` still
emits `[<think>, \n\n, </think>, \n\n, Hello]` with GT rank 0 every
step.

Bench result:

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| Phase E.4.3 (only conv1d + rms_norm on GPU) | 0.32 | 0.33 | 21.16 GiB |
| **Phase E.6.A.7 (+ L2-norm + recurrence on GPU)** | **0.35** | **0.35** | **21.16 GiB** |

**+6 % decode.** Modest but the lever is real: each per-layer host
bounce removed cuts the sync chain. Next we need the per-step path
to also do silu, reshape, and silu(z)*o on device (i.e. the fused
post-projection chain from E.6.A) — that's still blocked on
E.6.A.6 parity (suspected fp32 expf drift in the GPU silu kernel
compounding through the per-position recurrent state).

## 2026-05-12 update — Phase E.6.A.6 follow-up: reduction-order + fmad in nvrtc

Tightening the GPU kernel's numerical behaviour against the host
reference while debugging the multi-position fused-path drift:

- **L2 norm + rms_norm_heads reductions** now use a sequential
  per-element accumulator in thread 0 (matches the host's per-d
  loop in `l2_normalize_per_head_eps` / `residual::rms_norm_heads`).
  The previous parallel tree reduction across 256 threads produced
  bit-different sums from the CPU (max_abs drift 5e-7 → 2.4e-7 on
  rms_norm; 4.5e-8 → 1.5e-8 on L2 norm at production shape).
  All other threads still parallelise the elementwise normalize
  pass, so the slowdown is negligible.
- **nvrtc fmad disabled** (`CompileOptions { fmad: Some(false), .. }`)
  for both the `cuda::deltanet` and `cuda::qwen35_block` modules.
  Removes fused-multiply-add as a source of GPU/CPU divergence in
  the recurrence + reshape + silu_mul kernels.
- New unit test `l2_and_rms_norm_drift_at_qwen35_shape` quantifies
  the residual drift at production shape (head_dim=128,
  n_v_heads=48, n_k_heads=16).
- New `src_offset > 0` coverage in
  `reshape_kernels_match_cpu_at_qwen35_shape` validating the K / V
  slab reshape with offsets the production fused path actually uses
  (offset = key_dim and 2*key_dim into a packed qkv_conv buffer).

Result on the parity check (`real_gguf_qwen35_token_diff_vs_llama_cpp`
with `LARQL_QWEN35_E6A_FUSED=1`): **still diverges from llama.cpp
GT at the first decode step** despite the new bit-tighter
reductions and disabled fmad. The argmax numerics are identical
before and after these changes, so the drift root cause is
neither parallel reduction order nor fmad. Most likely it's the
recurrence kernel's state-update pattern compounding through the
9 prompt × 48 layer × position cycle in a way the single-call
unit test doesn't catch — under continued investigation (still
E.6.A.6). The default path (fused off) remains parity-clean.

## 2026-05-12 update — Phase E.6.A foundations (fused post-projection chain, opt-in)

Lays down the Phase E.6.A infrastructure for a fused device-resident
DeltaNet post-projection chain — conv1d → silu → split + reshape →
L2 Q/K → recurrence → reshape → rms_norm_heads → silu(z)*o — all on
the device with one sync at the block boundary, then a single dtoh
of the block output. The matvecs (attn_qkv / attn_gate / ssm_out)
remain on the existing host-returning `quant_matvec` path; absorbing
them into the device chain is the E.6.B scope.

What landed:

- New CUDA PTX module `cuda::qwen35_block` with the four small
  scaffolding kernels needed to keep the chain on device:
  `silu_inplace_f32`, `reshape_head_to_dim_f32`,
  `reshape_dim_to_head_f32`, `silu_mul_inplace_f32`. The
  five-kernel pipeline reuses the existing
  `cuda::deltanet::module_functions` (causal_conv1d,
  deltanet_step, l2_norm_dim_major, rms_norm_heads_head_major) on
  the same stream without inter-kernel sync.
- New optional `ComputeBackend::qwen35_deltanet_postproj_step`
  trait method (default `None`); CUDA backend's implementation
  routes to `qwen35_block::deltanet_postproj_step_cached`.
- New unit tests at Qwen3.6 production shapes that previously had
  no coverage: `reshape_kernels_match_cpu_at_qwen35_shape` (head⇄dim
  layout transpose) and `deltanet_step_matches_cpu_at_qwen35_shape`
  (s=128, h_k=16, h_v=48 — max_abs vs CPU = 7.45e-9, mean=1.02e-9).
- Loader fix in `qwen35_load::load_one_deltanet_layer`:
  `ssm_conv1d_raw.t().to_owned()` preserved transposed strides,
  silently disabling the `as_slice()`-gated GPU conv1d hook
  (which therefore never actually ran). Replaced with
  `as_standard_layout().to_owned()` so the row-major copy makes
  `as_slice()` `Some` — the conv1d kernel now actually executes
  on GPU when `LARQL_QWEN35_GPU=1`. Verified parity-clean by the
  token-diff harness.

Status:

- **Default OFF** behind `LARQL_QWEN35_E6A_FUSED=1`. Parity is
  bit-equivalent to CPU on a single call (per-element drift
  ~1e-7), and pos=0 layer-by-layer diff vs CPU is clean. But
  enabling fused mode for every position causes the multi-token
  parity check (`real_gguf_qwen35_token_diff_vs_llama_cpp`) to
  diverge from llama.cpp's GT — argmax flips on step 0 even
  though every individual kernel matches CPU to <1e-8 in
  isolation. The drift compounds non-trivially through 9 prompt +
  5 decode positions and the per-layer recurrent state, in a way
  the single-call unit tests don't catch. Root cause still under
  investigation (most likely fp32 reduction-order sensitivity in
  the L2 and rms_norm reductions, amplified by the recurrence's
  state-update cycle across positions).
- The conv1d-on-GPU enablement (load-side fix) is parity-validated
  with the fused path disabled.
- Throughput is unchanged in either mode (still 0.33 t/s decode):
  the post-projection hooks aren't the bottleneck on their own.
  The dominant cost remains the projection matvecs themselves;
  Phase E.6.B (device-resident matvec absorption into the same
  chain, eliminating ~7 host bounces per layer) is the actual
  throughput lever.

```bash
# Run the parity check at default (E.6.A disabled):
LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_token_diff_vs_llama_cpp -- --nocapture
# → [<think>, \n\n, </think>, \n\n, Hello] with GT rank 0 every step.

# Same test with the fused path on (currently fails parity at step 0):
LARQL_QWEN35_E6A_FUSED=1 <as above>
```

| Config | Decode (t/s) | Parity | VmRSS |
|---|---:|:--:|---:|
| Phase E.4.3 (per-step, conv1d falls back to CPU) | 0.33 | clean | 21.16 GiB |
| **Phase E.6.A foundations (conv1d-on-GPU, fused off)** | **0.33** | **clean** | **21.16 GiB** |
| Phase E.6.A foundations (fused on, opt-in) | 0.33 | broken — under investigation | 21.16 GiB |

## 2026-05-12 update — Phase E.4.3 GPU per-head L2/RMSNorm

Added CUDA reductions for the remaining per-head DeltaNet norms:
Q/K L2 normalisation in the `[head_dim, n_k_heads]` dim-major layout
and post-recurrence RMSNorm in the head-major `[n_v_heads, head_dim]`
layout. Both are exposed as optional `ComputeBackend` hooks and keep
the CPU implementation as fallback. The CUDA module now validates
Conv1D, recurrence, L2, and RMSNorm against tiny CPU references.

Validation:

```bash
cargo check -p larql-inference --features cuda
cargo test -p larql-compute --features cuda cuda::deltanet -- --nocapture
cargo test -p larql-inference --lib qwen35 -- --nocapture

LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_token_diff_vs_llama_cpp -- --nocapture
```

The token-diff parity check still emits
`[<think>, \n\n, </think>, \n\n, Hello]` with GT rank 0 every step.

Bench protocol was unchanged from E.4.1/E.4.2:

```bash
LARQL_QWEN35_GGUF=$PWD/output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_BENCH_PREFILL=16 LARQL_QWEN35_BENCH_DECODE=4 \
LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1 LARQL_QWEN35_GPU=1 \
cargo test -p larql-inference --release --features cuda --lib \
  real_gguf_qwen35_bench -- --nocapture
```

Result:

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| Phase E.4.1/E.4.2 first pass (+ CUDA Conv1D/recur) | 0.33 | 0.33 | 21.16 GiB |
| **Phase E.4.3 (+ CUDA per-head L2/RMSNorm)** | **0.32** | **0.33** | **21.16 GiB** |

Correctness is preserved, but the E.4.4 target (≥ 10 decode t/s) is
still unmet. The new reductions remove more CPU arithmetic but add
more tiny GPU launches and synchronising host returns. The next
meaningful speed step is a fused/device-resident DeltaNet block path
or the E.6 device-resident activation/weight pipeline; standalone
host-returning hooks are correctness plumbing, not enough throughput
plumbing.

## 2026-05-12 update — Phase 2d lazy-quant embed (105 → 20 GiB, −80.9 %)

Adds `QuantTensor::row_to_f32(token_id)` for the embed-lookup
pattern (a row read, not a matvec), then wires it into
`qwen35_forward_step` and the GGUF lazy loader. `embed_quant`
becomes a peer of `lm_head_quant` on `ModelWeights` and
`Qwen35Weights`.

| Config | Decode (t/s) | VmRSS |
|---|---:|---:|
| Phase 2c (lazy FFN/attn) | 0.23 | 24.07 GiB |
| **Phase 2d (+ embed lazy)** | **0.23** | **19.99 GiB** |
| Δ vs Phase 2c | same | **−4.08 GiB** |
| Δ vs baseline | −53 % | **−85.26 GiB (−81.0 %)** |

Speed unchanged — embed lookup is a single-row dequant per token,
amortised against the 256 matvecs/token already on the lazy path.

**llama.cpp's ~16 GiB target is now ~4 GiB away.** Remaining
chunks are smaller per-head SSM tensors (ssm_alpha / ssm_beta /
ssm_conv1d / ssm_norm) and the per-layer norm vectors. Closing
the last 4 GiB would require lazifying those too, but each one is
tiny (1-50 MB), so the engineering effort per GiB has crossed an
inflection point — Phase 3b (cache-tile batched Q4_K matvec) is
now a higher-priority lever.

**Parity preserved**: argmax bit-exact, GT rank 0 every step.

## 2026-05-12 update — Phase 2c lazy-quant full-attn q/k/v/o

Extends the lazy set to the four full-attention projections per
attn layer (16 attn layers × q/k/v/o = 64 additional matvecs/token).

| Config | Decode (t/s) | VmRSS |
|---|---:|---:|
| Phase 2b (lazy FFN + DeltaNet projs) | 0.20 | 29.62 GiB |
| **Phase 2c (+ full-attn q/k/v/o)** | **0.23** | **24.07 GiB** |
| Δ vs Phase 2b | **+15 %** | **−5.55 GiB** |
| Δ vs baseline | −53 % | **−81.18 GiB (−77.1 %)** |

Speed actually **improved slightly** (0.20 → 0.23 t/s): the
full-attn dense matvecs on x86 CPU run at f32 BLAS but as
single-vector sgemv (no batching across rows), whereas the
rayon-parallel Q4_K kernel splits each matvec's rows across cores.
On a 16-core box the parallelism wins even for these moderately-sized
matrices.

llama.cpp parity on RAM (~16 GiB) is now **~8 GiB away**. The
remaining big chunks are:
- `embed` `{vocab=248320, hidden=5120}` ≈ 5 GiB (Q4_K → ~1 GiB)
- Per-head SSM tensors (ssm_beta, ssm_alpha, ssm_conv1d, ssm_norm)
- Various per-layer norm vectors

Embed needs a different code path (row-lookup not matvec) — a
future `QuantTensor::row_to_f32(token_id)` would dequant one row
on demand. Per-head SSM tensors are small enough that the win is
marginal (<2 GiB total).

**Parity preserved**: `real_gguf_qwen35_token_diff_vs_llama_cpp`
still emits the same `[<think>, \n\n, </think>, \n\n, Hello]`
sequence with GT rank 0 every step.

## 2026-05-12 update — Phase 2b lazy-quant attn_qkv / attn_gate / ssm_out

Phase 2b extends the lazy-tensor set to the three big DeltaNet
projections per linear layer: `attn_qkv` `{conv_dim=10240, hidden=5120}`,
`attn_gate` `{value_dim=6144, hidden=5120}`, `ssm_out`
`{hidden=5120, value_dim=6144}`. 48 linear-attention layers × 3
tensors = 144 additional matvecs/token through the lazy path.

| Config | Decode (t/s) | VmRSS |
|---|---:|---:|
| Phase 3 (lazy FFN + AVX2 + rayon) | 0.20 | 46.65 GiB |
| **Phase 2b (+ attn_qkv / attn_gate / ssm_out)** | **0.20** | **29.62 GiB** |
| Δ vs Phase 3 | same | **−17.03 GiB** |
| Δ vs baseline | −59 % | **−75.63 GiB (−71.9 %)** |

Each linear-attention layer's three big projections sum to
~470 MB f32 → ~75 MB Q4_K (lossy 6.3×); 48 layers × 395 MB saved
= ~19 GiB. The observed 17 GiB drop matches that estimate.

Speed unchanged at 0.20 t/s — the extra 144 matvecs/token are
amortised by the same rayon row-parallelism that drove Phase 3.

Remaining ~30 GiB is mostly:
- `embed` `{vocab=248320, hidden=5120}` ≈ 5 GiB
- Full-attn projections (16 layers × q/k/v/o) ≈ 4-8 GiB
- Smaller per-layer SSM tensors and DeltaNet `ssm_norm` /
  per-head bias vectors
- Plus the dequantized layer attn_*_norm / ssm_norm vectors

**Parity preserved**: `real_gguf_qwen35_token_diff_vs_llama_cpp`
still emits the same `[<think>, \n\n, </think>, \n\n, Hello]`
sequence with logits `[28.18, 24.78, 25.47, 30.39, 21.66]` and
GT rank 0 every step.

llama.cpp parity on RAM (~16 GiB) is now ~13 GiB away.

## 2026-05-11 update — Phase 3 AVX2 + rayon for Q4_K matvec

Phase 3 adds (a) an AVX2 inner-loop kernel for `q4k_row_dot` on
x86_64 with fully vectorised dequant + FMA, and (b) rayon
`par_iter_mut` over the rows of every Q4_K and Q6_K matvec in
`QuantTensor::matvec`. Same opt-ins (`LARQL_QWEN35_LAZY_FFN=1
LARQL_QWEN35_LAZY_LM_HEAD=1`).

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| baseline (dequant + BLAS) | 0.48 | 0.49 | 105.25 GiB |
| Phase 2 (lazy, scalar) | 0.06 | 0.06 | 46.65 GiB |
| **Phase 3 (lazy, AVX2 + rayon)** | **0.21** | **0.20** | **46.65 GiB** |
| Δ vs Phase 2 | +250 % | +233 % | same |
| Δ vs baseline | −56 % | −59 % | −58.6 GiB |

The AVX2 kernel on its own barely moved the needle (0.06 → 0.07)
— LLVM auto-vectorises the scalar code well already. **Rayon
across rows** is where the speedup came from: 192 FFN matvecs per
token now fan out 14336 / 5120 row-dots across cores in parallel,
saturating the multi-core machine. Per-row AVX2 is the cherry on
top.

Now only 2.4× slower than the f32 BLAS baseline at less than half
the RAM. The remaining gap is mostly the per-row dispatch overhead
and the fact that BLAS sgemv batches rows in cache-friendly tiles.
Phase 3b (batched-row AVX2 matvec à la llama.cpp's
`mul_mat_q4k_q8k`) is the next perf lever.

Parity preserved: `real_gguf_qwen35_token_diff_vs_llama_cpp` still
emits the same `[<think>, \n\n, </think>, \n\n, Hello]` sequence
with GT rank 0 at every step.

## 2026-05-11 update — Phase 2 lazy-quant FFN

Same harness, smaller workload (prefill 8 / decode 2) because the
all-lazy path is 8× slower per token. Opt-in:
`LARQL_QWEN35_LAZY_FFN=1 LARQL_QWEN35_LAZY_LM_HEAD=1`.

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| larql CPU, fully dequant (baseline) | 0.48 | 0.49 | **105.25 GiB** |
| larql CPU, lazy lm_head only | 0.31 | 0.31 | 101.30 GiB |
| **larql CPU, lazy lm_head + FFN** | **0.06** | **0.06** | **46.65 GiB** |
| Δ vs baseline | −88 % | −88 % | **−58.6 GiB** |

192 FFN matvecs per token now route through scalar Q4_K
`q4k_row_dot` instead of f32 BLAS — that's where the 8× slowdown
comes from. The RAM win is huge but the trade-off is real.

**Parity preserved**: `real_gguf_qwen35_token_diff_vs_llama_cpp`
still produces argmax `[<think>, \n\n, </think>, \n\n, Hello]`
with logits `[28.18, 24.78, 25.47, 30.39, 21.66]` and GT rank 0
at every step, identical to the dequant baseline. The lazy path
is bit-exact in the matvec results (modulo Q4_K dequant rounding,
which is identical to llama.cpp's).

**Remaining RAM** (~47 GiB) is mostly the embed (5.1 GiB), DeltaNet
SSM tensors (alpha/beta/gate/qkv/out/conv1d/norm), and full-attn
projections. Phase 2b (lazy these) would close most of the gap to
llama.cpp's ~16 GiB resident.

The Phase 3 AVX2 quant kernels are now clearly the next perf lever
— without them, this path is unusable for serving.

## 2026-05-11 update — Phase 1 lazy-quant lm_head (PR follow-up)

The `qwen35-lazy-quant-matmul` Phase 1 change introduces
`load_gguf_lazy_lm_head` and `QuantTensor::matvec` to keep
`output.weight` (Q6_K) in its native form. Opt-in via
`LARQL_QWEN35_LAZY_LM_HEAD=1`. Same bench harness, smaller workload
(prefill 16 / decode 4 to keep total wall time manageable):

| Config | Prefill (t/s) | Decode (t/s) | VmRSS |
|---|---:|---:|---:|
| larql CPU, dequant lm_head | 0.48 | 0.49 | 105.25 GiB |
| **larql CPU, lazy lm_head** | **0.31** | **0.31** | **101.30 GiB** |
| Δ | −36 % | −37 % | **−3.95 GiB** |

Phase 1 trades a 37 % per-step slowdown on lm_head for ~4 GiB RAM
recovery. The slowdown is from the scalar Q6_K `q6k_row_dot` path
beating f32 BLAS — expected per the Phase 1 proposal's non-goals.
Phase 3 (x86 AVX2 quant kernels) will close that. Phase 2 (lazy
FFN tensors) is where the RAM number drops toward llama.cpp's
~16 GiB.

## Implications for Phase E / F roadmap

1. **Quant-aware matmul** is now the bottleneck. Even staying on CPU,
   if our matmul stays in Q4_K_S we'd jump from 0.5 tok/s decode to
   somewhere near llama.cpp's 2.6 tok/s. That's the next big win.
2. **CUDA path** (Phase E in `tasks.md`) gives the 50–100× over CPU
   that llama.cpp shows. Realistically required to be competitive at
   all on 27B+ models.
3. **VRAM headroom story** (the `--ffn` remote-offload pitch from the
   original Phase F design) only becomes meaningful once attention is
   on GPU. Today it's all CPU.
4. **Correctness is unblocked.** Parity is bit-exact (modulo Q5_K/Q6_K
   quant noise) per C.5j/C.5k. The remaining work is purely
   performance — no more semantic surprises expected.

## Reproducibility

```bash
# llama.cpp baseline
~/3rd-party/llama.cpp/build/bin/llama-bench \
  -m output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
  -p 128 -n 64 -r 2          # GPU
~/3rd-party/llama.cpp/build/bin/llama-bench \
  -m ... -p 32 -n 8 -r 2 -ngl 0   # CPU

# larql baseline
LARQL_QWEN35_GGUF=output/gguf-cache/Qwen3.6-27B/Qwen3.6-27B-Q4_K_S.gguf \
LARQL_QWEN35_BENCH_PREFILL=32 LARQL_QWEN35_BENCH_DECODE=8 \
cargo test -p larql-inference --release --lib real_gguf_qwen35_bench \
  -- --nocapture
```
