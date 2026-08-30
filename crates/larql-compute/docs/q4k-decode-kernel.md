# Q4K decode kernel — close the 1.5× gap to llama.cpp

**Status:** Specced 2026-05-16. Not started. CPU-only.
**Driver:** StandardEngine through the dispatch trait now hits 27.6 tok/s on Gemma 3 4B Q4K on M3 Max (8 threads). llama.cpp hits 41.4 tok/s on the same model + hardware. The remaining 1.50× gap is **entirely in the Q4K × Q8K matvec inner loop** — same algorithm, same NEON `vdotq_s32` instructions, but llama.cpp uses hand-written aarch64 inline assembly with explicit instruction interleaving while ours uses Rust intrinsics lowered by LLVM.

## Diagnosis (already recorded)

See `bench/baselines/cpu/COMPARISON.md` (2026-05-15) and `bench/baselines/cpu/DIAGNOSIS-2026-05-16-thread-scaling.md` (2026-05-16).

Per-core thread-scaling table from the diagnosis:

| Threads | larql | llama.cpp | Per-core ratio |
|---:|---:|---:|---:|
| 1 | 5.7 | 9.88 | **1.73×** |
| 4 | 18.4 | 31.86 | 1.73× |
| 8 | 24.6 | 42.13 | 1.71× |

The ratio is **constant across thread counts** — this is a per-core kernel-quality issue, not a scaling problem. Effective Q4K weight bandwidth:

- llama.cpp: ~95 GB/s (≈80% of M3 Max LPDDR5 peak)
- larql today: ~63 GB/s

We use ~66% of llama.cpp's effective bandwidth on the same hardware.

## Root cause

Both engines:
- Use the same Q4K × Q8K dot-product algorithm.
- Use ARM NEON SDOT (`vdotq_s32`) as the primitive multiply-accumulate.
- Read 4-bit weights, dequantise on-the-fly, accumulate into i32 lanes.

The difference is in the inner loop's instruction stream:

| Layer | llama.cpp (`ggml_vec_dot_q4_K_q8_K`) | larql (`q4k_q8k_matvec_neon`) |
|---|---|---|
| Inner kernel | Hand-written inline aarch64 asm | Rust `core::arch::aarch64::vdotq_s32` intrinsics, lowered by LLVM |
| Block interleaving | Two adjacent super-blocks interleaved in the asm to keep both load units busy | One super-block at a time (parity-tested helpers compose) |
| Prefetch | Explicit `prfm pldl1strm` hints ahead of the matvec stream | LLVM emits prefetch heuristically |
| Block layout | Pre-formatted lo/hi nibble pairs ready for `tbl` / `ushr` | GGUF on-disk Q4K layout; unpack inside the matvec each block |

LLVM's instruction scheduling from intrinsic IR is good but not optimal on this kind of byte-unpacking-heavy hot loop. On a wide-issue core like M3 Max where both load ports + 4 SDOT pipes can fire per cycle, **tight scheduling matters**. The intrinsics produce ~33 cycles/super-block on this workload; hand-asm closes it to ~21 cycles/super-block.

## Proposed work

### Phase 1 — Hand-asm Q4K × Q8K matvec (highest leverage)

Replace `crates/larql-compute/src/cpu/ops/q4k_q8k_dot/`'s NEON intrinsic path with hand-written `global_asm!` (or `asm!` per-function) aarch64 implementation modeled after llama.cpp's `ggml_vec_dot_q4_K_q8_K`. Two-super-block interleaved, explicit instruction scheduling, prefetch hints on the weight stream.

**Acceptance:**
1. Bit-identical output to today's intrinsic path on the full Gemma 3 4B Q4K test corpus.
2. Single-thread tok/s ≥ 9.5 (closing >95% of the per-core gap to llama.cpp's 9.88).
3. 8-thread tok/s ≥ 38 (closing >90% of the wall-clock gap to llama.cpp's 42).
4. No regression on any other workload that uses `quant_matvec`.
5. Cleanly gated `#[cfg(target_arch = "aarch64")]` with the intrinsics path retained as fallback for other architectures.

**Estimated effort:** 1–2 weeks. Mostly:
- Day 1–2: port llama.cpp's kernel structure into a Rust `global_asm!` block with the same register allocation.
- Day 3–5: parity testing against the intrinsic path on synthetic + real-model corpora. Iterate on scheduling.
- Day 6+: thermal-aware benchmarking, p99 latency check, ensure no scaling regression past 8 threads.

### Phase 2 — Pre-formatted Q4K block layout (secondary)

llama.cpp pairs lo/hi nibbles in the block layout so SIMD doesn't need to shuffle per block. We unpack inside the matvec because we share the on-disk GGUF layout.

Two options:
- **(a)** Change the on-disk vindex Q4K layout to pre-paired. Breaks compatibility with existing `.vindex` files; needs a migration path.
- **(b)** Repack to a paired layout on vindex load (one-time cost per process, kept in RAM/mmap). Adds a few MB of RAM and ~100ms load time but keeps disk compatible.

(b) is the pragmatic choice. Estimated gain: 1.1–1.2× on top of Phase 1.

**Estimated effort:** 3–5 days, mostly the loader + the kernel's tighter scheduling that becomes available without the shuffle.

### Phase 3 — Q6K kernel (Gemma 3 `down` projection)

Gemma 3's `down` projection is Q6K, not Q4K. We have a NEON Q6K matvec but it hasn't received the same SDOT treatment. Same shape as Phase 1 but smaller scope (only one matvec per layer per step).

**Estimated effort:** 2–3 days. Gain: ~1.05× — Q6K is a small fraction of total decode time.

### Phase 4 — Rayon launch overhead (sub-percent, last)

Today the per-token decode does **198 separate `par_iter_mut` launches** (34 layers × 6 Q4K projections). Each launch has rayon's join overhead (~5–10 µs). At 198 launches per token × 7.5 µs = ~1.5 ms of pure rayon overhead per token. That's about 4% of the total.

**Fix:** batch each layer's projections into a single rayon sweep (one `scope` per layer or per attention/FFN block).

**Estimated effort:** 2–3 days. Gain: ~1.04×.

## Non-goals

- **Hand-asm Q4 (non-K) matvec.** Q4_0 is legacy; production models use Q4K. Don't spend effort there until a model needs it.
- **Anything Metal-side.** This is a CPU-track item. Metal Q4K is already fast through `decode_token`.
- **Algorithmic changes.** Q4K dequant → Q8 input → SDOT accumulate is the right algorithm. We're optimizing the constant factor.
- **Switching to a different quant format.** FP4 (per memory `project_exp26_fp4_quantisation_q1.md`) is a separate track; orthogonal to this kernel work.

## Acceptance criteria (final)

Phases 1–4 land in order. After all four:

1. **Single-core ≥ 9.5 tok/s** on Gemma 3 4B Q4K via `larql bench --cpu --engine standard -t 1`.
2. **8-thread ≥ 39 tok/s** (closing 96% of the 41.4 tok/s gap).
3. Effective Q4K weight bandwidth ≥ 90 GB/s.
4. p99 step-time within 1.2× of mean (no scheduling cliffs).
5. Bit-parity tests pass at every phase.

## Architectural notes

This work is **orthogonal to the dispatch-trait redesign**. It lands inside `crates/larql-compute/src/cpu/ops/q4k_q8k_dot/` (and friends). The dispatch trait's `coarse_prefill` / `coarse_decode_step` already route through these kernels; no engine code changes when the kernel improves. Same `larql_compute::QuantMatVec::q4k_matvec` entry point.

Phase 2's pre-formatted layout is the only change that potentially touches the vindex layer. If we go with option (b) — repack on load — even that is contained to the loader and doesn't change disk formats or the trait surface.

## What this is NOT

This spec is **not** about getting research engines (MarkovResidual, WindowedCheckpoint, TurboQuant, Apollo) to production speed. Those have a different problem: their bespoke `prefill_quant` overrides bypass the dispatch trait's `coarse_*` intents and use slower CPU code paths. That's covered by `kv-dispatch-quantization.md` Phase 2 (engine migration) — separate work item.

After Phases 1–4 here, `StandardEngine` reaches parity with llama.cpp on CPU Q4K decode. Other engines need their own migration to benefit from the same kernels.

## 2026-06-02 — roofline measurement (falsification-first, before writing asm)

Two of this repo's own docs disagreed on what bounds this kernel:
`DIAGNOSIS-2026-05-16-thread-scaling.md` concluded the gap is
"memory-system-level, not compute or scheduling level"; this spec said
compute/scheduling (33→21 cycles/super-block). Those imply **different
fixes** (a DRAM-bound kernel can't be sped up by better instruction
scheduling). Settled it with a single-thread roofline microbench before
committing to hand-asm — avoids the build-then-measure trap.

**Tool:** `crates/larql-compute/benches/q4k_q8k_matvec.rs`
(`cargo bench -p larql-compute --bench q4k_q8k_matvec`). Times the
production `q4k_q8k_matvec_neon` single-threaded at real Gemma 3 4B
shapes (hidden=2560, intermediate=10240), reports GB/s on the Q4_K
weight stream. This is now the C12 iteration loop — measure every kernel
change here (fast, isolated) before any end-to-end tok/s run.

**Result (M3 Max, single thread):**

| shape | bytes | neon | scalar |
|---|---:|---:|---:|
| `ffn_gate_up` 10240×2560 | ~14 MB | **17.7 GiB/s** | — |
| `ffn_down_shape` 2560×10240 | ~14 MB | 17.8 GiB/s | — |
| `attn_proj` 2560×2560 | ~3.7 MB | 17.7 GiB/s | 9.3 GiB/s |

**Verdict: compute/issue-bound, NOT DRAM-bandwidth-bound.** Two
independent reads:
1. **Same data, scalar 9.3 vs neon 17.7 GiB/s (1.9×).** If the kernel
   were bandwidth-bound, scalar and NEON would tie (both wait on the same
   bytes). They don't — the SIMD does more work per cycle, so the limiter
   is compute throughput, not the memory system.
2. **GB/s is size-invariant** (14 MB vs 3.7 MB → identical rate). A
   bandwidth-bound kernel would slow per-byte once it spills cache.
   (Caveat: M3 Max's ~16 MB P-cluster L2 means even 14 MB partly fits, so
   reading (1) as the decisive evidence; (2) is corroborating.)

**This overturns the DIAGNOSIS doc's "memory-system-level" conclusion and
confirms this spec's framing.** 17.7 GiB/s ↔ 144 B / (144/17.7e9) = 8.1
ns/super-block ↔ ~33 cycles at 4.05 GHz — exactly the spec's "intrinsics
produce ~33 cycles/super-block". The 2-row variant's "BW-bound, kept for
future hardware" note (`cpu/ops/q4k_q8k_dot/`) was based on the same
wrong model and should be re-evaluated under compute-bound.

**Instruction mix per super-block** (`otool -tv` on the release rlib):
16 `sdot.4s` + 8 `and.16b` + 8 `ushr.16b` (nibble unpack) + 8 `ldrb`
(scalar scale/min byte unpack) + ~6 vector loads + ~18 scalar/vector
adds + scale muls ≈ ~54 instructions. At 4-wide issue that's ~13 cycles
if perfectly packed; we measure ~33 → ~20 cycles lost to latency
stalls / port conflicts. **The fat to cut is the non-SDOT overhead:**
the nibble unpack (16 ops) and the scalar scale handling (`ldrb` →
scalar→vector). llama.cpp vectorises both and interleaves two
super-blocks to hide latency.

**Target:** ~28 GiB/s (~21 cycles/super-block) ≈ llama.cpp's per-core
rate ≈ the documented 1.73× close.

**Refined Phase 1 plan** (each step measured in the microbench, parity
gated by `q8k_matvec_neon_matches_scalar_bit_exact`, bit-exact):
1. Single `asm!` block for the whole super-block dot so the schedule is
   ours, not LLVM's — two-super-block interleaved to fill the ~20
   stalled cycles with the next block's loads + unpack.
2. Vectorise the 6-bit scale/min unpack (kill the 8 `ldrb` + scalar
   broadcast); apply scales via a vector path instead of 8
   `vmulq_n_s32`.
3. Prefetch the next super-block's 144 weight bytes (the 2026-05-16
   per-row prefetch test regressed because it was the wrong granularity
   under the wrong (BW) model; re-test at per-super-block lookahead under
   the compute-bound model — latency-hiding, not bandwidth).
The f32 epilogue (`acc += d_w*d_y*sum1 − dmin_w*d_y*sum2`) and `sum2`
stay in Rust → parity reduces to "does the asm produce the same integer
`sum1`", which the bit-exact test pins directly.

### Landed 2026-06-02 — single-super-block asm, 4 accumulators, opt-in

`q4k_q8k_matvec_asm` / `q4k_sb_sum1_asm` in `cpu/ops/q4k_q8k_dot/`: the whole
super-block dot in one `asm!` block, the 8 6-bit scales arriving as two
i32x4 vectors with `mul (by element)` instead of 8 scalar `ldrb` +
broadcast, and **4 independent group accumulators** (each written once,
tree-summed) so no group carries a cross-group RAW chain. Gated by
`LARQL_Q4K_ASM` in `q4k_q8k_matvec_into` (`OnceLock`-cached env read,
never in the hot loop).

> **Corrected 2026-08-23 — this said "opt-in … default off"; it is
> default ON.** `options.rs` builds the flag with `on()`, which is "not
> an opt-out value", so an unset environment gets the asm kernel and
> `LARQL_Q4K_ASM=0` is the way to turn it off. The inversion matters
> because a reader benchmarking "with and without asm" by *setting* the
> variable measures the same arm twice.

- **Parity:** `q8k_matvec_asm_matches_scalar_bit_exact` — bit-exact
  (`.to_bits()`) across 4 shapes incl. odd rows + production widths.
  Holds for both the single-accumulator (v1) and 4-accumulator forms.
- **Isolated (microbench, single thread, quiet machine, criterion
  median):** neon 17.5–17.7 → asm **18.3–18.4 GiB/s, +3.7–4.9%** across
  all three shapes (ffn_gate_up +3.7, ffn_down +4.9, attn +3.8). The
  asm/neon ratio swings ~±1.5% run-to-run, so read it as **~+4%**.
- **End-to-end (`larql bench --cpu`, 8-thread, quiet machine):**
  **~+1–2%** (5 alternating pairs: neon median 23.2 → asm 23.5). The +4%
  isolated dilutes because the opt-in only reaches the `matvec_into`
  callers (attention Q/K/V/O + `down`); the fused `gate_up` path — much of
  the FFN — still runs the intrinsic kernel. Asm-ifying `gate_up` (below)
  is what lifts the e2e ceiling. Also ±10% run-to-run noisy.

**Finding — latency-hiding has low headroom.** A 4-accumulator variant
(breaking the per-super-block accumulator RAW chain) showed **no reliable
gain** over the single-accumulator form — the difference is smaller than
the ±1.5% ratio noise. Likely because the row loop inlines
`q4k_sb_sum1_asm`, so the OoO core already overlaps super-block N+1's
compute with super-block N's accumulator chain — the cross-super-block
window hides the intra-block serial dependency. **Corollary: the
two-super-block interleave is probably also low-value** for the same
reason; the floor is SDOT/unpack throughput, so the real lever is
instruction-count reduction (below), not rescheduling.

**Reframed remaining work to reach ~28 GiB/s (the 1.73× close):**
1. **Instruction-count reduction (the real lever, not rescheduling).**
   ~67 SIMD ops/super-block at IPC ~2 → cut the op count: vectorise the
   nibble→scale path the way llama.cpp does (fewer `and`/`ushr`/`mul`),
   not more accumulators. Needs perf-counter (Instruments) attribution of
   which port saturates before guessing.
2. **Asm-ify the fused `q4k_q8k_gate_up_into`** path — production FFN
   gate/up uses it directly, bypassing `matvec_into`, so today's opt-in
   only covers the separate-matvec callers (attention Q/K/V/O, `down`,
   walk-FFN). Until gate/up is covered the e2e ceiling is capped.
3. **Validate on a quiet machine.** The ±10% 8-thread noise + transient
   external load made fine deltas unresolvable this session; the
   single-thread microbench on an idle box is the gate.
