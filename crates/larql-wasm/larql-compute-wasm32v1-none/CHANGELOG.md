# Changelog — larql-compute

Reverse-chronological ship log. Entries migrated from ROADMAP.md on
2026-05-09; pre-2026-05-09 entries preserve the date and voice they were
originally written in.

## 2026-05-09

### QKV defuse promoted (Track D, +1.6 tok/s, magnitude undershoot)

The fused `q4k_q6k_qkv_proj_normed` kernel had been the production default for Gemma 3/4 layers since the QKV-stage fusion landed earlier in the year. It saved ~0.24 ms/tok of dispatch overhead but ran at 199 GB/s vs the non-fused `q4k_q6k_qkv_proj`'s 287 GB/s — the per-TG H + norm_w reread (3× redundant device traffic across the 4 simdgroups) was costing ~1.46 ms/tok in the per-kernel batched diag.

Diag prediction: defusing recovers 1.46 ms − 0.24 ms = **−1.22 ms/tok** end-to-end.

End-to-end A/B (warmup 8, n=100, drift 0.02 ms):

| config | tok/s | mean ms | GPU fwd | output |
|---|---|---|---|---|
| baseline (fused, default) | 84.8 / 85.0 | 11.79 / 11.77 | 11.79 / 11.86 | "Paris" ✓ |
| defused (`LARQL_QKV_DEFUSED=1` at the time) | **86.5** | **11.57** | **11.52** | "Paris" ✓ |

**Measured Δ: +1.6 tok/s, −0.30 ms/tok GPU fwd** — direction matched the diag prediction but **magnitude was 18% of predicted**.

**Promoted to default** 2026-05-09; env var renamed to `LARQL_QKV_FUSED=1` (opts back in to fused). Fused shader + `encode_normed_q4k_q6k_qkv` retained as the opt-in fallback. The +1.6 tok/s landing matches the +1.6-tok/s gap that existed at the time of investigation between us and the next-best decode milestone.

**Why magnitude undershot.** The non-fused kernel's 287 GB/s peak is a single-kernel-isolated number. In the production decode pipeline, the candidate shares the LPDDR5X bus with attention, FFN gate+up, FFN down, lm_head — all also bandwidth-bound. The candidate doesn't get to *claim* its kernel-isolated headroom because the surrounding pipeline already saturates the same bus. Predicted batched-diag deltas should be discounted ~3-5× for bandwidth-headroom candidates; cycle-headroom or dispatch-count candidates don't show this discount as severely. Pinned as the first magnitude-compression case study in ADR-015.

**Cross-arch validation 2026-05-09 (post-thermal-cooldown)**: A/B on Gemma 4 26B A4B (60 layers, hidden=5376, MoE) confirms defuse is also net-positive on the larger MoE arch:

| config | tok/s | GPU fwd | mean ms |
|---|---|---|---|
| defused (current default, run 1) | 14.3 | 62.90 ms | 69.92 |
| Fused (`LARQL_QKV_FUSED=1`) | 13.9 | 65.91 ms | 71.82 |
| defused (drift check, run 2) | 13.9 | 64.53 ms | 71.94 |

**Δ defused vs fused: +0.4 tok/s, −2.19 ms GPU fwd** (median across both defused runs vs fused). Drift between defused runs is 1.6 ms; signal is 2.19 ms — at the noise floor but directionally consistent with the Gemma 3 4B finding (+1.6 tok/s defused). **No arch-specific dispatch path needed** — the same defused default works for both 4B and 26B. (Smaller win on 26B reflects the larger absolute GPU-fwd budget; the fixed dispatch-overhead saving is a smaller fraction of the total.) Same A/B also confirmed there's no QKV-defuse-driven regression on 26B; the earlier "2.4× regression vs 19.4 tok/s baseline" was a thermal artifact from sustained load (`feedback_thermal_perf_artifacts.md` memory entry).

### NR2 candidate removed (re-bench fell, kernel deleted)

Track C — `q4k_ffn_gate_up_nr2` candidate ROUND-TRIPPED 2026-05-02, REMOVED 2026-05-09.

NR2 (2 rows / simdgroup variant of `q4k_ffn_gate_up`) was a strong isolated profiler candidate after the 8sg landing — `diag_profile_kernels` showed:

| | iso ms | iso GB/s | **batched ms** | **batched GB/s** |
|---|---|---|---|---|
| 8sg (default) | 0.591 | 51.4 | 0.106 | **278.9** |
| NR2 (candidate) | 0.401 | 76.8 | 0.110 | **267.0** |

End-to-end A/B (warmup 8, decode 30, quiet GPU, three runs, 2026-05-02):

| config | tok/s | GPU fwd | lm_head | output |
|---|---|---|---|---|
| baseline (8sg) | **75.9** | **11.19 ms** | 2.99 ms | "Paris" ✓ |
| NR2 (`LARQL_GATE_UP_NR2=1`) | 72.9 | 11.80 ms (**+0.62 ms**) | 2.96 ms | "Paris" ✓ |

NR2 wins isolated by 1.47× and **loses batched by 4%**. End-to-end tracks the batched number, not the isolated one — the 1.47× iso win was dispatch-overhead amortisation that disappears once n_layers calls share one cmd buffer. **Initial outcome (2026-05-02): not promoted; kept as opt-in.**

**Re-benched 2026-05-09** alongside the QKV-gap investigation:

| config | tok/s | GPU fwd |
|---|---|---|
| 8sg baseline (run 1) | 86.3 | 11.71 ms |
| NR2 (`LARQL_GATE_UP_NR2=1`) | 83.4 | 12.05 ms (**+0.34 ms**) |
| 8sg baseline (run 2, drift check) | 86.2 | 11.64 ms |

The 0.07 ms baseline drift between runs 1 and 2 confirms the −0.34 ms NR2 regression is real signal, not thermal noise (unlike the f16_acc story where a +23% measurement turned out to be thermal artifact). Direction matched the batched-diag prediction; magnitude was ~3× the predicted delta but the ordering held — exactly the contract ADR-015 makes.

**Outcome (2026-05-09): kernel and `LARQL_GATE_UP_NR2` env var removed.** The candidate had two independent benches against it; leaving it as opt-in dead code was inviting re-litigation. Shader (`shaders/q4k_ffn_gate_up_nr2.rs`), pipeline field, env-var constant, dispatch-branch in `encode_ffn`, diag-profile entry, and shader-bench inventory entry all deleted. Historical record preserved here, in `PERFORMANCE.md`, and in ADR-015.

**Same A/B run also confirmed** that the v5 stride-32 lm_head is the *fast* path, not just the correct one: `LARQL_LM_HEAD_STRIDE32=0` regressed lm_head 2.99 → 4.08 ms (+1.09 ms, 75.9 → 69.5 tok/s). The "+0.7 ms cost" framing in PERFORMANCE.md is relative to the pre-fix broken-output kernel, not the current fallback. No tradeoff to chase.

### D-RMS-FUSE Phase 1 implemented + falsified

RMS-norm pre-fusion with surrounding scalar mul/add (mirrors llama.cpp's `kernel_rms_norm_mul_f32` / `kernel_rms_norm_mul_add_f32`). Predicted ~0.1-0.2 ms decode on non-Gemma. **Measured null end-to-end** (within drift).

Most of `_mul_f32` / `_mul_add_f32` was already shipped: our `rms_norm` shader already does norm + scalar mul, and `residual_norm_store` / `post_attn_residual_norm_store` / `post_ffn_norm_residual_add` already fuse norm + residual add at all the load-bearing boundaries except the post-FFN→next-layer-input step on non-Gemma archs. Phase 1 implemented that last gap via `LARQL_FUSED_PRELAYER_NORM=1` opt-in (Llama 2 7B / Mistral 7B). Parity preserved (bit-identical output across 3 archs). End-to-end A/B: Llama 2 7B `+0.0 tok/s within drift`, Mistral 7B `−0.3 tok/s within drift`. The dispatch-overhead saving from skipping rms_norm in layer N+1 is offset by the heavier `residual_norm_store` (does the RMS reduction inline) replacing the lighter `residual_add` in layer N. Per ADR-015 magnitude-compression at the extreme.

Touched: `metal/decode/encode_post_ffn.rs` (`PreLayerNormFusion` struct + branch), `metal/decode/encode_qkv.rs` (`input_already_normed` skip flag), `metal/decode/mod.rs` (call sites + `prelayer_norm_active` plumbing). Kept opt-in per ADR-017 retention rules; revival scenario: different K dimensions or future hardware where the rms_norm dispatch overhead vs `residual_norm_store` extra reduction balance shifts.

### D-PREFILL-MM closed (twice-falsified)

Wire `q4k_matmul` into FFN gate/up/down + QKV (prefill only). Was estimated 3–4× prefill speedup. **FALSIFIED twice end-to-end.**

Wiring tried at O-proj (2026-04-28: −10% on long prompts), at FFN gate+up (2026-04-28: 2933 → 3268 ms on 340 tokens), and re-validated FFN gate+up under post-defuse state (2026-05-09: 5–7% regression at 10/50/150-token prompts). Diagnosis: the `q4k_matmul` kernel is bandwidth-bound; the [seq_len × hidden] X working set thrashes GPU L1 on long prompts and the dequant amortisation gain is paid back in DRAM↔L1 traffic. See **D-PREFILL-MM2** (in ROADMAP) for the kernel-rewrite track. The current `q4k_matmul` shader + `MetalBackend::q4k_matmul` method + parity tests stay shipped for re-validation on future hardware; production prefill stays per-position matvec.

### D-GEMMA4-E2B diagnosed (Per-Layer Embeddings missing in Metal)

Cross-arch bench captured `gemma4-e2b-q4k` at 4205 ms/tok GPU fwd vs Gemma 3 4B at 140 ms/tok. **Root cause**: E2B has Per-Layer Embeddings (PLE, `hidden_size_per_layer_input: 256`) which **the Metal pipeline does not implement**. `larql-inference/src/layer_graph/generate/gpu.rs:372-374` explicitly checks `weights.arch.has_per_layer_embeddings()` and routes to `generate_via_cpu_q4k` when true — the entire decode path falls back to CPU. The `LARQL_GPU_TIMING=1` line never fires for E2B because Metal's `decode_token_with_moe_split_fn` is never called. Documented in `gpu.rs` comment: "Without this routing the model produces multilingual gibberish." Diagnosis confirmed by reading the dispatch code and checking E2B's vindex `index.json` (`per_layer_embed_dim: 256`). Llama 2 / Mistral / Gemma 4 31B don't have PLE so route through Metal normally.

Closes the bug as "expected behaviour given PLE-not-in-Metal limitation". The actual perf fix is **D-METAL-PLE** (still open in ROADMAP) which implements PLE on Metal.

### D-CROSS-PARITY Phase 1 — cross-arch decode bench shipped

`scripts/bench-cross-arch.sh` + `bench/baselines/cross-arch/` + `make bench-cross-arch[ ARGS=--save-baseline\|--compare]`. Captured first under-load run; cool-machine baseline pending. Operationalises ADR-017 model-agnosticity check; surfaces thermal artifacts (every-arch-regresses-simultaneously signature).

**Phase 2 (per-shader)** still open in ROADMAP — current sweep is end-to-end-only; per-shader parity tests (e.g. dispatching `q4k_ffn_gate_up_8sg` against synthetic input on multiple K dimensions to surface kernel-specific arch sensitivities) would catch deeper regressions earlier.

Touched: `scripts/bench-cross-arch.sh`, `bench/baselines/cross-arch/README.md`. Phase 2: parametrised tests in `crates/larql-compute/tests/`.

## 2026-05-04

### Metal GPU dense FFN server — `run_dense_ffn_q4k`

`MetalBackend::run_dense_ffn_q4k` in `crates/larql-compute/src/metal/moe_dispatch.rs`
provides a Metal GPU forward pass for one dense FFN layer given pre-loaded Q4K weight
buffers. Mirrors the structure of `run_experts_prestaged_metal` but takes separate
gate, up, and down buffers (not combined gate+up as in the MoE path).

Used by `larql-server::routes::walk_ffn::handle_walk_ffn_q8k` (under
`--features metal-experts`) to serve the dense remote-FFN path on GPU.

**Per-layer dispatch geometry** (`q4k_matvec_pipeline.rows_per_tg` / `threads_per_tg`,
not hardcoded) — same fix as the 2026-04-28 dispatch geometry correction.

Bench (Gemma 4 31B Q4K, M3 Max, single-machine localhost):

| Metric | Before | Metal server | Δ |
|---|---|---|---|
| Streaming (60 × sequential HTTP, CPU NEON) | 0.1 tok/s | 0.6 tok/s | 6× |
| Batch (1 × parallel HTTP, CPU NEON) | — | 1.6 tok/s | — |
| Batch (1 × parallel HTTP, Metal GPU) | 1.6 tok/s | **6.5 tok/s** | **4×** |

Bottleneck at 6.5 tok/s: attention at 92ms/token (60%). Two-pass batch structure
(capture pass + apply pass) doubles the local Metal attention cost. FFN at 60ms
is at the 400 GB/s GPU bandwidth ceiling for 11.7 GB/token of Q4K weight reads.

**Build separation required**: `--features metal-experts` must NOT be used for
`larql-cli` (causes 10.7 vs 18.9 tok/s regression on Gemma 4 26B-A4B due to Metal
pipeline init overhead in the standard decode path). Only the server binary uses that flag.

### D-FFN-PROFILE — split `encode_ffn` profiler boundary

Split `encode_ffn` profiler boundary (gate_up vs activation+down). Diagnostic, not perf.

`LARQL_PROFILE_SPLIT=1` + `--profile` bench now shows three separate GPU buckets per step. Measured on Gemma 3 4B (10-token steady state): **attn=3.3ms (34%), gate+up=3.5ms (36%), act+down=2.8ms (29%)** — all three roughly equal thirds. Gate+up is the largest single kernel. See `metal/decode/encode_ffn.rs` (split helpers) + `profile.rs` (GateUp/Down stages) + `bench_cmd.rs` (sub-rows).

### NaN on Gemma 4 31B global-attention layers — fixed

`kv_append_attend_fused` used a fixed `tg_scores[1024]` threadgroup array. Global layers (window_size=0) grow unboundedly — once the KV cache exceeds 1024 positions, `tg_scores[t - t_start]` overflowed, corrupting scores → `exp()` produced Inf → softmax NaN. Fix: guard `use_fused_kv_aa` with `attn_span <= SHORT_ATTENTION_SPAN`; global layers fall through to `encode_kv_attend` which auto-selects `kv_attention_long` (4096-entry array) past 1024 tokens. Also fixed: `v_norm_batched` read/write race when `x` and `out` aliased the same buffer (threadgroup barrier missing between reduction and write-back phases; cos≈0.997 drift on L0).

## 2026-05-02

### GPU expert dispatch — Phase 2: pre-allocated staging buffers (DONE; baseline corrected)

`MoeScratch::new` (in `metal/moe_dispatch.rs`) pre-allocates all expert staging buffers once per model shape and caches by `(top_k, hidden, intermediate_size)` on the backend. Per-layer `gpu_moe_dispatch_with_scratch` only memcpys expert bytes into existing buffer contents — no `bufs.output(...)` calls in the hot path.

**Measured 2026-05-02 (post Phase 2 + dispatch-geometry fix)**:
- 26B A4B Metal: **19.4 tok/s** (was 5.1 pre-2026-05-02 — bug-locked under
  the dispatch-geometry mismatch in the same `moe_dispatch.rs` sites; the
  "Phase 1 shipped 5.1 tok/s" baseline was attributing the bug-locked
  number to Phase 1, which was wrong).
- GPU-only ceiling (`LARQL_SKIP_MOE=1`): **56.8 tok/s**.
- Remaining headroom (19.4 → 56.8): genuine expert dispatch work
  (240/token = 8 experts × 30 layers × 1 fused gate+up + 1 GEGLU + 1 down)
  + 30 commit/wait syncs. Real shader/dispatch work, not allocation.

The pre-2026-05-02 "Phase 2 expected ~4× gain" estimate happened to
match the actual 5.1 → 19.4 perf jump — not because Phase 2 was the
load-bearing fix, but because the dispatch-geometry mismatch was masking
the same ~4× of real perf as 240 broken expert dispatches. With both
fixes in, the new ceiling estimate for 26B A4B is ~25–30 tok/s if the
expert-dispatch fusion levers in `larql-server/ROADMAP.md§F-LOCAL-MOE`
land.

**Scope (single landing):**

1. **Pre-allocate persistent staging buffers** in `decode_token_q4k_moe`
   (`metal/moe_dispatch.rs`). Sizes are constants of `(top_k, inter_padded,
   hidden, row_bytes, down_row_bytes)` — known once per decode, not per layer.
   Buffers to pre-allocate (all `StorageModeShared` so CPU writes via
   `buffer.contents()`):
   - `gate_buf`: `top_k × inter × row_bytes`
   - `up_buf`: `top_k × inter × row_bytes`
   - `down_bufs`: `top_k` × `[hidden × down_row_bytes]` (per-expert; experts
     come from different mmap pages, so K independent staging buffers — not
     a single concatenated one).
   - `g_out`, `u_out`: `top_k × inter × 4`
   - `act_buf`: `top_k × inter_padded × 4`, zero-initialised once
   - `expert_outs`: `top_k × hidden × 4`

   `gpu_moe_dispatch` becomes `gpu_moe_dispatch_with_scratch(scratch, ...)`;
   the per-call body just memcpys expert bytes into the existing buffer
   contents and dispatches. No `self.bufs.output(...)` calls inside the
   per-layer hot path.

2. **Fix activation-stride bug** (P0 correctness blocker #3 in this file).
   Today: `act_buf` allocated at `valid_count × inter_padded × 4`, but the
   geglu kernel writes linearly at stride `inter`. For
   `moe_intermediate_size` not a multiple of 256 (e.g. Gemma 4 26B's 2112 →
   inter_padded 2304), expert `e>0` reads stale/garbage floats. Fix:
   dispatch `geglu_gelu_tanh` per expert with `g_out`/`u_out` linear offset
   `e × inter × 4` and `act_buf` strided offset `e × inter_padded × 4`. K
   extra dispatches per layer (top_k=8 → 8 small dispatches) but each is
   ~5µs — negligible vs the ~120ms allocation overhead this PR removes.
   Alternative: stride-aware kernel — defer if perf demands it post-bench.

3. **Borrow expert slices instead of `to_vec()`** (host-copy churn). Today
   `larql-inference::layer_graph::generate::gpu` allocates two
   `Vec<u8>` per expert per layer (~2.2 MB heap-copy × 30 layers × 8 experts
   per token). Change `get_expert: impl Fn(layer, expert) -> Option<(Vec<u8>,
   Vec<u8>)>` to return `Option<(&[u8], &[u8])>`. Lifetime-bound to the
   weights mmap — borrow lasts only across the `gpu_moe_dispatch` call.
   Updates `decode_token_q4k_moe` signature + the inference-side caller.

4. **Add parity test** `gpu_moe_dispatch` Q4_K experts with
   - aligned `inter` (e.g. 768),
   - misaligned `inter` requiring padding (e.g. 704),
   - `valid_count < top_k` (some experts return None),
   against CPU MoE reference.

**Acceptance criteria**:
- `cargo test -p larql-compute --features metal` green (existing + new parity).
- `larql bench gemma4-26b-a4b` ≥ 15 tok/s (3× from baseline 5.1).
- No regression on `larql bench gemma3-4b-q4k-v2` (dense path untouched).

**Out of scope for this PR**: dense kernel optimisation, fused
QKV V-path correctness blocker (#2), the expert-bytes-→-Metal-buffer copy
itself (already a single memcpy via `contents()` ptr; can't shrink further
without DMA-side weights, which is a larger refactor).

**Root cause of remaining gap.** `gpu_moe_dispatch` calls `self.bufs.output()` ~10 times per
MoE layer to allocate gate, up, per-expert-down, activation, and output Metal buffers.
With 30 MoE layers × ~10 allocations = 300 Metal buffer allocations per decode token,
each allocation of a 1–9 MB `StorageModeShared` buffer costs ~0.4ms on M3 Max.
**Total: ~120ms/token in allocation overhead** (measured: 194ms total − ~40ms compute − ~30ms syncs).

There is also avoidable host-copy churn before those Metal allocations:
`larql-inference::layer_graph::generate::gpu` calls
`weights.get_layer_entry_bytes(...)?` and immediately `to_vec()`s both
expert slices before `gpu_moe_dispatch` copies them into Metal staging.
For Gemma 4 26B A4B, this is 30 layers × top_k=8 × roughly 2.2MB of
heap copies per decode token. Phase 2 should change the API to pass
borrowed mmap slices (`&[u8]`) through the closure and copy exactly once
into reusable Metal buffers.

**Fix.** Pre-allocate all staging buffers once before the layer loop in
`decode_token_q4k_moe` (in `metal/moe_dispatch.rs`), identical to the pattern that
eliminated 550→20 allocations in `decode_token_with_moe_fn`:

```
Pre-allocated once:
  gate_buf:     [top_k × inter × row_bytes]  (gate Q4K staging)
  up_buf:       [top_k × inter × row_bytes]  (up Q4K staging)
  down_bufs:    [top_k] × [hidden × down_row_bytes]  (per-expert down Q4K staging)
  act_buf:      [top_k × inter × 4]  (f32 activations after GELU)
  expert_outs:  [top_k × hidden × 4]  (f32 expert outputs)
```

Sizes are constant per model (determined by `moe.intermediate_size`, `moe.top_k`,
`hidden`). The pre-allocated buffers are reused for all 30 layers via write-in-place
to `buffer.contents()` pointers.

**Effort**: ~1 session. No new shaders needed — just restructure the buffer lifecycle
in `decode_token_q4k_moe`.

## 2026-05-01

### NEON Q4_K matvec — 8.6× CPU MoE sweep speedup

New module `crates/larql-compute/src/cpu/ops/q4k_q8k_dot.rs`
implements Q4_K weight × Q8_K activation matvec mirroring llama.cpp's
`ggml_vec_dot_q4_K_q8_K`. NEON inner kernel uses `SDOT` via inline asm
(stable; `vdotq_s32` is still gated behind unstable `stdarch_neon_dotprod`
on Rust 1.91, rust-lang/rust#117224). Wired as default for Q4_K weights
in `cpu/ops/moe/expert.rs::{run_single_expert,run_single_expert_q4k_q8k_into}`,
`cpu/ops/moe/forward.rs::cpu_moe_forward`, and
`larql-server/src/routes/expert.rs::run_experts_cpu_batch`.
`LARQL_DISABLE_Q4K_DIRECT=1` falls back to BLAS-on-cached-f32.

7 new tests: Q8_K quantiser round-trip, scalar Q4_K×Q8_K vs cached-f32 path
within Q8 noise, multi-block matvec, **NEON vs scalar bit-exact**
(`to_bits()` equality), edge cases.

Bench (Gemma 4 26B-A4B, M3 Max, single-shard loopback):

| Metric | Baseline | + NEON Q4_K | Δ |
|---|---|---|---|
| `cpu_moe_forward` warm floor | 3.52 ms | **0.39 ms** | **9.0×** |
| 30-layer sweep | 221 ms | **25.6 ms** | **8.6×** |
| Steady RSS | 11.4 GB | 10.5 GB | -8% (f32 cache mostly inert) |

Projects to ~25-30 tok/s on the gRPC grid (vs prior 2.3 tok/s baseline).
See `larql-inference/ROADMAP.md` M-CPU-4 for full attribution and follow-ups.

## 2026-04-30

### Fused Q4_K geglu+down disabled by default — `LARQL_FUSED_DOWN=1` opt-in

The `q4k_geglu_silu_down` / `q4k_geglu_gelu_tanh_down` shaders pass their unit tests but produce all-NaN at the prefill output for production-shape weights (Gemma 3 4B q4k-downq4k → 2560/2560 NaN; Gemma 4 31B q4k → empty output). Separated path (existing GEGLU dispatch + `q4k_matvec`) is correct for the same shapes. Default flipped in `metal::stages::ffn::encode_gated`; perf parity to be re-tested if/when the fused kernel is fixed.

### Metal MoE expert kernel — accuracy bug at inter=704 (workaround landed)

cos≈0.7 vs CPU reference for Gemma 4 26B-A4B-it MoE; same shaders are correct for dense FFN. Workaround: server defaults to CPU expert dispatch (`LARQL_USE_METAL_EXPERTS=1` to opt back in). Once fixed: ~3-4× grid speedup (3.5 tok/s → ~10 tok/s) since server compute is 95% of token wall time. Bug is still open and tracked in ROADMAP.md "Open: Metal MoE expert kernel — accuracy bug at inter=704".

## 2026-04-28

### Session status snapshot

**Decode**: 78.7 tok/s baseline (corrected from 81-84 buggy number). Gap to ollama 1.30× — distributed across pipeline, not concentrated in any single kernel with obvious headroom.

**Prefill**: 196 ms (18 tok) → 2933 ms (340 tok). 4-14× gap to ollama. Has the headroom; needs `q4k_matmul` wired into more sites.

**Shipped this session**:
- ✓ `q4k_matmul` Metal kernel (1.79× kernel-isolated for prefill); wired at O proj, parity tested
- ✓ `q4k_ffn_gate_up_f16acc` shader, opt-in via `LARQL_F16_ACC=1`
- ✓ Profiler harness fix (`measure_single_cmdbuf_batched`)
- ✓ Encoder coalescing in 3 dispatch sites
- ✓ Magic-number/string audit + extraction (Q4_K constants, manifest kind enum)
- ✓ MoE combine helper unification (CPU vs Metal — fixed 26B-A4B garbage output)
- ✓ lm_head Q4_K vs Q4_0 dispatch fix (was producing gibberish on gemma3-4b-q4k-v2)
- ✓ `larql parity --component layer` end-to-end Metal-vs-CPU diff (proved MoE fix)

**Negative results documented (don't re-try)**:
- ✗ N_DST > 1 (multi-row per simdgroup): register pressure regresses on M3 Max
- ✗ float4 vectorisation in Q4_K kernels: addressing overhead negates gain
- ✗ sumy precompute: neutral (compiler already hoisting)
- ✗ f16 accumulators end-to-end: kernel 1.79× but **end-to-end at parity** on quiet GPU. Initial +23% was thermal-throttle artifact. ALU savings absorbed by surrounding bandwidth-bound kernels.
- ✗ Wiring `q4k_matmul` into prefill (O proj + FFN gate+up, attempted 2026-04-28): kernel-isolated 1.79-3.8× did NOT translate end-to-end. Short-prompt prefill within noise; **long-prompt prefill regressed ~10%**. Root cause: the matmul's `[seq_len × hidden]` X working set thrashes GPU L1 on long prompts, defeating the cache locality the matvec loop had. Reverted. The matmul kernel remains shipped with parity tests but is not worth wiring into the production prefill path on this hardware.

**Pattern across the negative results (3 attempts in a row, then a positive)**: kernel-isolated speedups don't *automatically* translate end-to-end. The 8sg variant did — kernel-isolated 1.37× → end-to-end +2.1% throughput on quiet GPU. The difference: 8sg is a pure dispatch geometry change with same per-thread compute, so the GPU schedules more concurrent simdgroups for free; the failed attempts (f16 acc, matmul) changed the per-thread/per-call work in ways that interacted poorly with the surrounding pipeline. Per-kernel optimisations should still be measured end-to-end on a quiet GPU before wiring.

**GPU-time instrumentation finding**: Added `MTLCommandBuffer.gpuStartTime/gpuEndTime` to production decode (`metal/decode/gpu_timing.rs`, env-gated `LARQL_GPU_TIMING=1`). On gemma3-4b-q4k-v2:

```
wall ≈ 10.9 ms  |  gpu ≈ 10.4 ms  |  cpu ≈ 0.5 ms (4-5%)
```

**The 2.5 ms gap to ollama is GPU compute time, not CPU dispatch overhead.** Dispatch fusion saves at most ~5% (entire CPU overhead is 0.5 ms). The "374 vs 272 dispatches" framing was overweighted; the real gap is per-kernel GPU efficiency.

**This invalidates the "no per-kernel headroom" claim** but NOT for the cache-pressure reason I initially guessed. Added cold-cache profiling (`metal/diag/kernel_profile.rs` rotates through 8 distinct weight buffer pairs, ~170-240 MB total — far exceeds L2). Cold-cache result: **identical to warm-cache**:

| kernel | warm GB/s | cold GB/s |
|---|---|---|
| q6k_matvec (down) | 317 | 316 |
| q4k_ffn_gate_up | 274 | 276 |

So cache pressure is NOT the gap. Our kernels really do sustain 274/317 GB/s in production conditions.

**Reframed**: M3 Max LPDDR5X peak is **~400 GB/s** (system-wide, ~320 GB/s practical for GPU). Our kernels at 274 = 68% of peak (gate+up) and 317 = 79% of peak (down). Ollama's hand-tuned llama.cpp kernels likely sit at 85%+ of peak — that's where the 2.5 ms decode gap lives. The headroom is real but in **kernel geometry/occupancy choices**, not cache handling.

Concrete next investigation: try different threadgroup configurations (more simdgroups per TG without per-thread register pressure, larger ROWS_PER_TG with corresponding adjustments) to push toward 85% of peak. The auto-memory's "N_DST > 1 regresses" finding rules out per-simdgroup multi-row, but doesn't rule out per-TG multi-simdgroup at fixed nr0=1.

**Open priorities (best-leverage first, as of 2026-04-28)**:
1. **Wire `q4k_matmul` into FFN gate/up/down for prefill** — ~3× prefill speedup expected (kernel proven at 1.79× isolated, multiple sites compound). Days of careful integration.
2. **Wire `q4k_matmul` into QKV** — fused Q+K+V matmul kernel needed, OR per-projection matmul fallback. Week-scale work.
3. **Fix profiler for remaining kernels** (q4k_matvec for Wo, etc.) — accurate per-kernel numbers. Hour-scale.
4. **Decode is at-or-near M3 Max ceiling for this pipeline architecture** — closing the last 25% to ollama would require fundamental fusion / scheduling changes, not per-kernel optimisation.

### Track A — profiler harness fixed

`metal/diag/kernel_profile.rs` now uses `measure_single_cmdbuf_batched` for q6k_matvec and q4k_ffn_gate_up. Old `measure_batched` is kept (with a "DON'T USE for kernel throughput" doc note) for callers who genuinely want per-call cmd-buffer overhead. **Follow-up**: same fix for q4k_matvec (Wo) and any future kernels added.

### Track B — `q4k_ffn_gate_up_f16acc` SHIPPED (opt-in, no end-to-end win on this hardware)

`metal/shaders/q4k_ffn_gate_up_f16acc.rs` — variant with f16 inner accumulators (per-superblock dot product). Outer accumulator and `sumy` stay f32. Safe because Q4_K nibbles are 0..15 (exact in f16) and RMS-normed X has |x| < ~5, so the 16-FMA partial sum stays well under f16 max (65504).

**Measured 2026-04-28**:

| measurement | f32 (default) | f16 acc | delta |
|---|---|---|---|
| Kernel isolated (N=10240, K=2560) | 0.607 ms | 0.340 ms | **1.79× kernel speedup** |
| End-to-end decode, **thermally loaded** GPU | 16.40 ms/tok | 13.34 ms/tok | +23% (apparent) |
| End-to-end decode, **quiet** GPU | 12.95 ms/tok | 13.06 ms/tok | **at parity (~1% slower)** |
| Numerical drift (max abs in kernel output) | — | 0.155 (≈1.5% relative) | — |
| Output text on 10-prompt corpus | bit-identical to f16 | bit-identical to f32 | full parity ✓ |

**The end-to-end perf win does not reproduce on a quiet GPU.** Initial 5-iter measurement showed +23% throughput, but that was on a thermally-loaded system where the f32 kernel was throttling. On a quiet system both paths run at the same wall-clock — f16 freed ALU cycles get absorbed into pipeline stalls or thermal headroom the surrounding kernels reclaim. The 1.79× kernel speedup is real in isolation; it doesn't translate to end-to-end decode improvement because the kernel was already bandwidth-bound (274 GB/s, 74% of LPDDR5X peak), not ALU-bound.

**Numerical parity is solid**: 10-prompt greedy-decode sweep (knowledge / code / math / creative / translation, 32 tokens each) — all outputs bit-identical between f32 and f16 paths. The 1.5% per-call drift never crosses a top-1 token boundary in the validated corpus.

**Status: kept as `LARQL_F16_ACC=1` opt-in.** Default stays f32. Useful as future-proofing if (a) hardware changes the ALU/bandwidth balance, (b) a future kernel re-fuses the path so ALU becomes the bottleneck, or (c) a sustained-load workload benefits from less thermal pressure. Not promoted to default because there's no measurable steady-state win to justify the precision risk on unvalidated workloads.

**Lesson for future kernel work**: the kernel-isolated profiler can be misleading. A 1.79× isolated speedup ≠ 1.79× end-to-end if the kernel was bandwidth-bound or part of a longer pipeline where other resources serialise the GPU. Always validate end-to-end on a quiet system before adopting.

## 2026-04-27

### Q4_K `sumy` precompute — no measurable gain

Separated the X-sum used in the min-correction term from the FMA dot-product
loop in `q4k_matvec` and `q4k_ffn_gate_up`. Previously both shared one loop
(`dot_acc` and `sum_acc` accumulated together); now a dedicated `sumy` pass
runs first, leaving the dot loop as a pure FMA chain the compiler can
schedule without interleaved additions. Applied to both the standalone matvec
and the fused gate+up shader.

**Measured 2026-04-27 on the all-Q4_K extract (`gemma3-4b-q4k-downq4k`),
3 runs each, identical bench setup:**

| Shader form | Run 1 | Run 2 | Run 3 | GPU fwd |
|---|---|---|---|---|
| With `sumy` precompute (split loops) | 71.7 | 72.3 | 72.1 | 12.67–12.74 ms |
| Without (combined `dot_acc` / `sum_acc`) | 72.4 | 71.6 | 72.9 | 12.62–12.77 ms |

Difference is within run-to-run variance — the Apple Silicon shader compiler
schedules the combined loop just as well as the split form. Kept the split
version anyway since it's cleaner code for future readers; no perf regression
either direction. Worth flagging that this micro-optimisation didn't pan out
so future "split the FMA chain from the sum" attempts know the answer.

### `QuantFormat` classifier helpers shipped (#7 partial)

Classifier helpers shipped: `QuantFormat::is_q4k_family()` /
`is_q4kf()` / `is_legacy_q8()` on `pipeline.rs`. The most-duplicated
predicate (`format == Q4_K || == Q4_KF || == Q6_K`, repeated verbatim
in `decode/mod.rs` ×2 and `decode_hybrid.rs` ×1) collapses to a single
method call. Adding a future Q4_K-style format updates one classifier,
not 3+ OR-chains. Pinned by `quant_format_classifiers` test.

**Full `FormatRoute` enum DEFERRED.** The roadmap intent
(`F32Input { fused_down: Option<&KernelHandle> }` / `Q8Input { norm_q8,
qkv_q8 }` / etc., with the `match QuantFormat::*` confined to one
constructor in `metal/stages/quant_matvec.rs`) is a 49-file refactor —
every dispatch site that currently matches on `QuantFormat` would need
to switch to consuming a `FormatRoute`. Doing it concurrently with the
in-flight MoE struct refactor risks heavy merge conflicts. Defer until
MoE settles AND there's a concrete near-term need (e.g. an FP4 / FP8
format being added). The classifier helpers above absorb the immediate
duplication cost in the meantime.

### `FullPipelineLayer` `Default` impl shipped (#9 partial)

Test ergonomics fix shipped: `FullPipelineLayer` and `QuantWeight` now
implement `Default`, so test code uses
`FullPipelineLayer { wq, ..Default::default() }` instead of spelling out 30
fields. The pre-existing `minimal_layer` helper collapsed from 30 lines to
10. New `default_layer_accepts_local_borrows_via_spread` test pins the
pattern for future tests (verifies `..Default::default()` reborrows the
`'static` defaults at the caller's stack-local lifetime — typical Rust
HRTB territory but worth a test since it's a non-obvious property).

**Full sub-struct split DEFERRED.** The roadmap intent
(`LayerWeights` / `LayerNorms` / `LayerArchParams` / optional `MoeBlock`)
is a 30+ caller-file refactor. Doing it concurrently with the in-flight
MoE struct refactor (ongoing in this branch) risks merge conflicts on
`pipeline.rs`. Pick this back up once MoE work settles. The `Default`
impl removes the immediate test pain — that was the user-visible cost
of #9.

### Encoder coalescing shipped (prefill #1)

Hoisted `cmd.new_compute_command_encoder()` out of the per-position loops in
`dispatch.rs::399` (O proj) and `stages.rs::97`, `:174` (input_norm + QKV).
One encoder per stage instead of `seq_len` of them. **Measured: saves ~5% on
long prompts, within noise on short prompts.** The 5 µs × seq_len savings
is real but dwarfed by per-dispatch kernel compute time. No regression on
decode (seq=1 path runs the loop once, identical semantics). 135 Metal
tests still pass.

### Q4_K matmul (gemm) kernel shipped (kernel + parity tests; partial wiring)

`crates/larql-compute/src/metal/shaders/q4k_matmul.rs` — amortises Q4_K dequant across `COLS_PER_TG=4` positions per super-block. Same `ROWS_PER_TG=4` simdgroup geometry as `q4k_matvec`, plus a per-thread `acc[4]` accumulator array (16 bytes register footprint, fits comfortably). 5 parity tests in `tests/test_kernel_q4k_matmul.rs` assert bit-equivalence with stacked matvec calls across basic / seq_len=1 / ragged-tail / production shapes. Perf spot-check (`tests/test_kernel_q4k_matmul_perf.rs`, gated on `LARQL_PERF_SPOT_CHECK=1`) on N=2560, K=8192, M=18: **3.82× speedup** (4.99 ms stacked matvec → 1.31 ms matmul). At full closure that's ~196 ms → ~51 ms prefill on Gemma 3 4B (ollama parity).

**Wiring status — partial 2026-04-27**: Wired into the O projection site (`dispatch.rs::5. O projection`). Added `q4k_matmul: Option<&KernelHandle>` to `quant_matvec::Pipelines`; threaded through `dispatch_full_pipeline` signature and all callers. Branches on `seq_len > 1 && format == Q4_K && pipeline.is_some()` and falls back to per-position matvec otherwise. Decode (seq=1) keeps the matvec path, decode tests (135 lib) all pass.

**Measured impact of partial wiring**: WITHIN NOISE. Short prompt 196 → 203 ms; long prompt 2933 → 3006 ms; decode 13.78 → 13.45 ms/tok. O projection is only ~1/7 of the per-position Q4_K work in prefill — the 3.8× kernel speedup applied to one site saves ~2 ms on an 18-tok prompt, below the ±5% prefill noise floor. The kernel works, but a single call site doesn't show in the headline number.

(Full FFN/QKV wiring closed under D-PREFILL-MM 2026-05-09 — see top of changelog.)

## 2026-04-26

### q6k_matvec ROWS_PER_TG shader/dispatch mismatch — FIXED

**Root cause of the "regression" to 68-70 tok/s:** the shader constant
`Q6K_ROWS_PER_TG` and the Rust dispatch constant `ROWS_PER_TG` were mismatched:

- **Shader:** `Q6K_ROWS_PER_TG = 2` → `row_idx = tg_id * 2 + sg_id` (sg_id 0..3 = 4 rows per TG)
- **Rust dispatch (HEAD):** `ROWS_PER_TG = 4` → dispatched ceil(N/4) = 640 TGs

With this mismatch, maximum covered row = 639 × 2 + 3 = **1281 of 2560**. Rows 1282–2559 received **zeros** — a silent correctness bug in the FFN down projection for dense models. Model output was degraded but simple prompts (e.g. "Paris") survived because the residual stream carried enough signal.

The stash that fixed the dispatch to `ROWS_PER_TG = 2` made the output correct but dispatched 1280 TGs — 2× more work than necessary (each row computed by two adjacent simdgroups due to the overlap in the formula).

**Fix:** set both constants to `4`: shader `Q6K_ROWS_PER_TG = 4` and Rust `ROWS_PER_TG = 4`. Each TG covers 4 non-overlapping rows (sg_id 0..3), dispatches 640 TGs, correct output, optimal throughput.

**Result:** 78.7 tok/s, GPU fwd 10.8ms — **correct and faster than the broken HEAD**.

(A subsequent restatement was logged as "q6k_matvec ROWS_PER_TG — correctness fix (2026-04-26)" — both refer to the same fix.)

### Mixed Q4_K/Q6_K QKV fused V path — resolved

The named test `q4k_q6k_qkv_proj_normed_matches_separate_norm_and_proj`
passes against `q6k_matvec` at the original 512-hidden test geometry
AND at production hidden=2560 (10 super-blocks/row). Added
`q4k_q6k_qkv_proj_normed_matches_at_production_hidden` regression
test pinning the larger shape so any future drift is caught at
production K, not via a model-output bug report.

### f32_gemv_topk1 GPU argmax (infrastructure)

New `MatMul::f32_gemv_topk1` trait method: runs gemv + GPU argmax in one command
buffer, reads back only 8KB (1024 partial results) instead of 1MB (262K scores).
Saves ~0.33ms for top_k=1 callers. Implemented on MetalBackend. Main decode loop
uses the KNN lm_head path (top_k=5 → KNN fires first), so this doesn't yet
benefit the bench. Useful for non-KNN models and future greedy-decode APIs.

### lm_head GPU partial top-K wired (saves ~0.44 ms)

`top_k_sorted` in `larql-vindex/index/storage/lm_head.rs` now runs an argmax fast path for `k=1` and a size-K min-heap for `k>1` instead of allocating a 2MB `Vec<(u32, f32)>` and `select_nth_unstable` over 262K elements (~0.25ms saved). New `f32_topk_partial` MSL shader emits `K_TOPK = 8` (val, idx) pairs per TG via repeated simd_max + index-mask; backend methods `f16_gemv_topk` / `q4_matvec_topk` route the bench's `top_k = 5` lm_head call through GPU partial top-K + 64KB readback + size-K CPU heap, avoiding the 1MB scores readback and the linear scan over 262K floats (~0.2ms additional). Greedy-decode `f16_gemv_topk1` / `q4_matvec_topk1` are also wired (no production caller yet — bench / generate both use top_k=5).

### Diagnostic infrastructure consolidation

Diagnostics were previously scattered across three locations:
- `src/metal/decode/diag.rs` — NaN detection, residual dumps, per-layer bisect
- `src/metal/decode/profile.rs` — stage-level `ProfileTimings`
- `examples/diag_decode_pipeline.rs` — decode pipeline stage bisect entry point

Now consolidated under `src/metal/diag/`:
- `diag/mod.rs` — public API, re-exports `ProfileTimings`, documents all tools
- `diag/kernel_profile.rs` — `KernelResult` + `profile_all()` for per-kernel
  bandwidth measurement (isolated vs batched, GB/s, bottleneck classification)
- Examples renamed to `diag_*` prefix for clarity

**Key diagnostic commands:**
```bash
# Per-kernel bandwidth profiler (results go to PERFORMANCE.md)
cargo run --release --features metal -p larql-compute --example diag_profile_kernels

# Decode pipeline stage bisect (bisect CPU/Metal divergence)
LARQL_METAL_DUMP_LAYERS=/tmp/dump \
  cargo run --release --features metal -p larql-compute --example diag_decode_pipeline

# NaN/divergence bisect at specific layer (env-gated, zero binary overhead)
LARQL_DECODE_DIAG_LAYER=12 larql infer <vindex> "prompt"
```

### Batched MoE prefill — SHIPPED

Replaced the O(seq_len × num_layers) token-by-token decode loop with a
batched approach: `dispatch_full_pipeline` now accepts an optional
`moe_fn: Option<&mut dyn FnMut(usize, &[f32], &mut [f32])>` callback.
When the callback is present and a layer has MoE, the function commits
the GPU command buffer after that layer's dense FFN, calls the closure
(which runs CPU experts for all seq_len positions and applies outer norm
+ layer_scalar), then restarts the command buffer for the next layer.

**Measured on Gemma 4 26B A4B (5-token prompt, 15 warmup / 30 tokens, M3 Max):**

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Prefill | 1889ms | 1297ms | **−31%** |
| Decode GPU fwd | 334ms/tok | 246ms/tok | **−26%** |
| Decode tok/s | 2.9 | **3.9** | **+35%** |

**Why:** 5-token prefill now uses 26 GPU commits (one per layer) vs 130
(5 positions × 26 layers). Batching all positions per layer also improves
weight cache utilisation. GPU layer_scalar skipped for MoE layers in the
dispatch; the callback applies it correctly after combining dense + MoE.
`kv_copy::populate_kv_one_layer` added for per-layer KV cache population.

## 2026-04-25

### #1 — Q6_K fused activation+down (closed — wrong fix, correct diagnosis)

**Status:** Benchmarked. Not viable. Routing reverted.
Root cause of original regression identified and documented.

**What was tried:** Added threadgroup-memory caching of `gate`/`up`
per super-block so all 4 simdgroups in a TG share one device load
(128 threads × 2 values each). All 5 parity tests pass. But
`larql bench gemma3-4b-q4k-v2` showed 61–62 tok/s — identical to
the unfused-TG-cache attempt and identical to the regression without
TG caching. TG caching had zero effect.

**Root cause (corrected):** bandwidth was never the bottleneck.
gate/up = 80 KB total per dispatch — well within M3 Max GPU L2 cache.
All 640 TGs share the same gate/up data → L2 cache-hits from TG 2
onward. The real regression is GELU-tanh recomputation:

- Separated path: `geglu_gelu_tanh` kernel runs 10,240 threads,
  each computing one `tanh(gate[i])`. Total: 10,240 `tanh` calls.
- Fused path: inner loop computes `tanh(gate[i])` for every output
  row independently. At N=2560 output rows: 2,560 × 10,240 =
  **26.2 M `tanh` calls** — 2560× more than separated.

`tanh` is a transcendental function; GPU ALU cost dominates. The
saved dispatch + buffer round-trip (~0.2 ms) doesn't offset the
extra 2560× `tanh` work at production shape.

**Q4_K fusion wins for a different reason:** the all-Q4_K model
uses SiLU (`x/(1+exp(-x))`), not GELU-tanh. SiLU is cheaper than
`tanh`, so the recomputation overhead is smaller relative to the
heavier Q4_K dequant per cell.

**Remaining Q6_K opportunity:** optimise `q6k_matvec` throughput
directly — currently 79 GE/s vs Q4_K 105 GE/s.
Alternatively: precompute `act[]` via a fast batch activation and
pass a float input to a future `q6k_matvec_f32in` kernel (avoids
the per-row `tanh` recomputation entirely while still fusing
dispatch). ~50 LOC new shader.

### #2 — Single encoder per token (was already implemented)

The decode loop already uses ONE encoder for ALL 34 layers
(non-MoE path). The ROADMAP item was mislabelled — the actual overhead
is per-`dispatch_thread_groups` call (~5-8µs each), not per-encoder.
Current dispatch count: ~14 dispatches/layer × 34 = 476 dispatches/tok
= ~2.4-3.8ms of dispatch overhead. Reducing requires kernel fusion.

### #5 — `q6k_matvec` full rewrite

**Total gain: ~3ms/tok / ~20% / +10 tok/s** (62→72 tok/s), in two phases:

**Phase A — 4-element batching** (+7 tok/s, 62→69):
Scalar inner loop used `(i & 3u) << 1u` — a runtime shift the GPU can't hoist.
Restructured to 4-element groups with compile-time hi2 shifts (0,2,4,6), 16
preloaded scales, and ROWS_PER_TG=8. All tests pass.

**Phase B — inter-superblock interleaving + X preload + deferred scale** (+3 tok/s, 69→72):
Adapted the llama.cpp `kernel_mul_mv_q6_K_f32_impl` strategy to LARQL's linear
Q6_K layout (GGUF's transposed layout can't be ported directly — different format):
- `ix = lane & 1` → adjacent lanes process alternate superblocks, letting DRAM
  serve two memory banks in parallel.
- `xl[16]` preloaded before weight reads → X fetches overlap weight byte loads.
- Deferred scale: `acc += d*sc * (unscaled_sum_4_elems)` — 4× fewer scale mults.
- ROWS_PER_TG dropped from 8→4 (128 threads/TG) → halved register pressure,
  2× more concurrent TGs, better latency hiding on LPDDR5X.
Effective Q6_K bandwidth: ~322 GB/s (up from ~294 GB/s).

### #7 — Dispatch fusion: consolidate per-layer ops (7a/7b/7c/7e shipped, 7d not viable)

**Estimated gain: ~1.0ms/tok** (saves ~200 dispatches at ~5µs each).

Current per-layer dispatch count (~14 for Gemma 3 4B):
1. `rms_norm` (input norm)
2. `q4k_q6k_qkv_proj` (QKV projection)
3. `qk_norm` — Q heads
4. `qk_norm` — K heads
5. `rope_at_pos_batched` — Q heads
6. `rope_at_pos_batched` — K heads
7. `kv_append`
8. `kv_attend`
9. `o_proj` (O projection)
10. `residual_norm` (post-attention residual + FFN norm)
11. `q4k_ffn_gate_up` (fused gate+up)
12. `geglu_gelu_tanh` (activation)
13. `q6k_matvec` (FFN down)
14. `residual_add` (post-FFN)

Three fusions with clear wins (each saves 34 dispatches = ~0.17ms):

**7a — Fused QK-norm Q+K** ✅ done 2026-04-25 (+0.17ms recovered):
New `qk_norm_qk` shader dispatches total_heads = q_heads + kv_heads in one
call; TG index selects Q buffer + q_weight vs K buffer + k_weight.

**7b — Fused RoPE Q+K** ✅ done 2026-04-25 (+0.17ms recovered):
New `rope_at_pos_batched_qk` shader: grid `(rope_pairs, q_heads+kv_heads, 1)`;
thread `h < num_q` selects Q buffer, else K buffer.

**7c — Fused input norm + QKV projection** ✅ done 2026-04-25:
New `q4k_q6k_qkv_proj_normed` kernel: all 128 threads cooperatively reduce
`||h||²` in Phase 1 (barrier), then each simdgroup runs its matvec with inline
`h[i] * rms * (offset + norm_w[i])`. Fires when format is Q4_K Q/K + Q6_K V,
standard RMS norm, no bias (Gemma 3 4B production).

**7e — Fused residual_norm + residual_add** ✅ done 2026-04-25:
New `residual_norm_store` kernel writes both `ffn_norm_out` (normed FFN input)
and `h_post_attn` (raw sum for post-FFN add) in one pass. Replaces the
`residual_norm + residual_add` two-dispatch pair in the Q4_K hot path.

**7d — Fused GEGLU + down** (~0.17ms):
Dispatches 12+13 can be merged for Q4_K down (already done). For Q6_K down,
fusion was attempted but regressed due to GELU-tanh recomputation cost
(see #1 closed). Not viable unless activation is precomputed separately.

### Magic-string kernel names eliminated (#6)

Added `ShaderKernel` trait + `get_shader_pipeline::<T>()` to
`kernel/traits.rs`; 31 magic strings eliminated. Each shader now
exports a compile-time `NAME` constant — renaming a shader causes a
compile error rather than a silent runtime panic.

### `Pipelines` struct asymmetry fixed (#8)

All fields in `metal/stages/quant_matvec.rs::Pipelines` now use
`&KernelHandle`; geometry drift is now a compile error rather than
a silent dispatch mismatch. ~100 LOC mechanical migration across
callsites.

### LM head heap (#4 partial)

`backend_lm_head_topk`: partial-sort 262k → top-k → **min-heap done**: avoids 2MB Vec allocation, saves ~0.1ms (observed lm_head 2.38 → 2.27ms). Remaining overhead after heap: ~0.35ms; GPU dispatch+commit+wait ~200µs (reducible with async readback) and 1MB buffer readback ~150µs (async pipelining needed). The GPU kernel itself (1.55ms) is the irreducible floor.

## 2026-04-24

### Gemma 4 26B A4B — full end-to-end correctness

Four coordinated fixes were needed (earlier "working" claim was only
approximate — the Metal output was degenerate token repetition on a
cold vindex). All verified against HF bf16 via layer-by-layer
residual-cosine diff in `metal/decode/diag.rs::ResidualDump` +
`/tmp/hf_residuals.py` + `/tmp/diff_residuals.py`.

1. **Row-padded Q4_K / Q6_K storage** for matrices whose inner dim
   isn't a multiple of 256. 26B A4B's `intermediate_size=2112` gives
   8.25 super-blocks per row; old extraction stored contiguously and
   the matvec shader read wrong bytes for every `down_proj` row past
   row 0. `pad_rows_to_256` in `larql-vindex/format/weights/write.rs`
   per-row-pads to the next 256-multiple; runtime dispatches
   `down_proj` with `K = inter_padded` (see `metal/decode/mod.rs`).
   `act_buf` allocated to `inter_padded * 4` bytes and zero-inited so
   the trailing columns contribute nothing. Aligned models
   (`inter_padded == inter`) are unchanged.
2. **Parameter-free router RMSNorm** — HF `Gemma4TextRouter.norm` has
   `with_scale=False`; no weight tensor exists on disk. Trait method
   `moe_router_norm_parameter_free()` + `rms_norm_no_weight` branch
   in `cpu/ops/moe/forward.rs`. Also added `router.scale *
   hidden_size^-0.5` multiplier (HF's `scalar_root_size`).
3. **Outer `post_feedforward_layernorm.weight`** (un-suffixed) added
   to extraction + wired through `FullPipelineLayer.moe_outer_post_norm`.
   Distinct from the `_1` dense-branch norm that was previously being
   double-applied.
4. **`layer_scalar` applied to the whole layer output** after residual
   add (`new_h *= layer_scalar`) — matches HF's `hidden_states *=
   self.layer_scalar` at the end of `Gemma4TextDecoderLayer.forward`.
   Prior code folded it into the outer-norm scale (14× magnitude
   error, collapsed the model to degenerate output).

Artifacts for future regression checks:
- `crates/larql-cli/examples/patch_down_proj.rs` — surgical vindex
  patcher (re-quantises `down_proj` rows with per-row padding).
  Avoids re-extracting 42 GB when the extraction side is fixed.
- `crates/larql-compute/src/metal/decode/diag.rs::ResidualDump` —
  env-gated (`LARQL_DUMP_RESIDUALS=<path>`) binary dump of every
  layer's `layer_in` / `h_post_attn` / `layer_out` for HF-ref diff.
- `crates/larql-inference/tests/test_arch_golden.rs` — architecture
  regression suite with one `#[test]` per `(arch × backend)`,
  skip-if-missing for vindexes. Caught the broken output immediately
  and flagged which architecture-specific change broke it.

## 2026-04-20

### CPU MoE decode interleave — DONE

GPU dense FFN + CPU MoE per layer. See `metal/decode/moe_combine.rs`
for the outer combine math (HF Gemma 4 has three post-FFN norms per
MoE layer: `_1` on dense, `_2` on MoE, and un-suffixed outer on the
sum — only the un-suffixed one gets `layer_scalar` applied to the
whole layer output after residual add).

## 2026-04-09

### Exceeded Ollama — 8.5ms / 117 tok/s = 0.83x Ollama (17% faster)

8.5ms / 117 tok/s vs Ollama 10.3ms / 98 tok/s. Key changes:
- Cooperative SIMD norm reduction (O(N²)→O(N)) — saved ~10ms alone
- Q4_KF (GGUF) FFN through llama.cpp-exact q4kf_proj kernel
- Fused gate+up kernels (q4k_ffn_gate_up + q4kf_ffn_gate_up)
- Q4_K matvec rewrite: uint4, 8 rows/TG, multi-row (nr0=2)
- Pre-allocated scratch buffers (550 allocs → 20)
- Batched RoPE + V-norm, SIMD KV attention
- Single cmd buffer + single global encoder

Previous: 29.2ms / 34 tok/s (2.84x Ollama).

### Dispatch merging — complete (negligible impact — Apple Silicon dispatch overhead is ~0ms)

### Optimize KV cache attend kernel — complete

simd_max/simd_sum reductions, float4 Q·K dot products, 1024-entry scores.

### Fix O(N²) norm kernels — complete

Cooperative SIMD reduction in all norms. Saved ~10ms (the single biggest win).

### Q4_KF fused gate+up

`q4kf_ffn_gate_up` — llama.cpp inner loop, shared input.

### Pre-allocated scratch buffers

550 allocs → 20, saved ~2ms.

### Single global encoder

One encoder for all 34 layers (no per-layer create/end).

### Cooperative SIMD norms

O(N²)→O(N) in rms_norm/residual_norm — saved ~10ms.

## 2026-04-08

### Single cmd buffer decode

All 34 layers in one cmd, single encoder per layer.

### Batched RoPE/V-norm

`rope_at_pos_batched`, `v_norm_batched` — 16 dispatches → 3.

### Q4_K FFN format routing

Q4_K weights use `q4k_matvec`, skip Q8 quantize.

### Fused gate+up kernel

`q4k_ffn_gate_up` — single dispatch, shared input.

### Q4_K matvec rewrite

uint4 loads, 8 rows/TG, sub-block striping, nr0=2.

### Q4_KF FFN routing

llama.cpp-exact `q4kf_proj` for FFN gate/up/down.

### SIMD KV attention

simd_max/simd_sum, float4 dot, 3 barriers (was 6).

### Ollama parity

2.84x → ~1.25x at 34 layers, no caching.

## 2026-04-07

### Q4_K FFN pipeline (end-to-end) — DONE

Vindex loader (`load_interleaved_q4k`), inference wiring (`predict_honest` prefers Q4_K FFN), and format tag propagation through `FullPipelineLayer` all wired. When `interleaved_q4k.bin` exists, Q4_K format flows through to compute shader dispatch.

### Sub-block lane assignment

83% utilization.

### llama.cpp kernel architecture port

Register-based input.

### Component profiling

Found real bottleneck.

### Zero warnings

Clean build.

### ADR documentation

8 decisions recorded.

### Partial RoPE (rotary_dim)

`rope_apply` + `fused_attention`, ADR-010.

### Gemma 4 architecture support

Per-layer head_dim, KV heads, K=V, layer_scalar.

### Shader documentation

`docs/shaders.md` — all 44 kernels.

### Quantization format docs

`docs/quantization-formats.md`.

### Decode pipeline docs

`docs/decode-pipeline.md`.

### Example reorganization

25 examples: `demo_`, `compare_`, `profile_`, `best_`, `test_`.

### PERFORMANCE.md refresh

All numbers from fresh benchmark runs.

### ROADMAP.md

P0/P1/P2 targets documented.

### Per-layer architecture params (ADR-011)

18 fields on `FullPipelineLayer`: `eps`, `attn_scale`, `head_dim`, `num_q/kv_heads`, `rope_base`, `rotary_dim`, `sliding_window`, `v_norm`, `layer_scalar`, `norm_type`, `ffn_type`, `activation`, biases.

### pipeline.rs extraction

`FullPipelineLayer` + types moved from `lib.rs` to `pipeline.rs`.

### 7 new shader kernels

`silu`, `gelu_tanh`, `layer_norm` (2), `v_norm`, `scale_vector`, `rope_at_pos partial`.

### Model-agnostic compute

No hardcoded model assumptions — all behavior parameterized per-layer.

## 2026-04-06

### Fused attention (RoPE+GQA+softcap)

Correct output.

### Q8 fused QKV

2.2x vs separate.

### Full pipeline (attn+FFN, 1 cmd)

18.5ms/21L.

### Safe buffer reads

13 unsafe sites → 1.

### CPU Q4_K/Q6_K reference

Cross-backend tests.

### Cross-backend tests (11 tests)

Metal vs CPU verified.

### Q4_K fused QKV

1.78x vs Q8.

### Dual-path decode (Q4_K/Q8 auto)

59 tok/s.

### GPU prefill pipeline

`seq>1` on GPU.

### `skip_rope` flag

Prefill KV cache.

## 2026-04-05

### Q4_0 v1-v5 kernels

v4 at 61 GB/s.

### Multi-layer FFN batch

8.4ms/21L.

## 2026-04-03

### ComputeBackend trait

Foundation.
