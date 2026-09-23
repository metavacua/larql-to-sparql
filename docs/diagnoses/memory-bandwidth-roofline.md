# Memory-bandwidth roofline — what is attainable, what we achieve

**Date:** 2026-07-28/29 · **Box:** M3 Max (12 P + 4 E, 128 GB), AC power
**Instruments:** `crates/larql-compute/examples/membw_probe.rs` (new),
`crates/larql-compute/benches/q4k_q8k_matvec.rs::bench_mt_production` (new),
`crates/larql-compute-metal/examples/diag_profile_kernels.rs` (existing)

## Why

Single-sequence decode is a matvec — ~2 flops per weight byte — so it is
bandwidth-bound by construction, and the whole speed programme has been an
argument about bytes. But the two numbers that argument was being conducted with
are not the same kind of number:

- **"larql extracts ~47 GB/s effective vs llama.cpp ~70"** (C10/C12, 2026-06-12)
  — an *achieved* figure, recorded **before** the spin-barrier pool landed +28%.
- **"the 400 GB/s GPU bandwidth ceiling"** (`larql-compute/ROADMAP.md`,
  `CHANGELOG.md`) — the **SoC spec sheet**.

Achieved-vs-spec is not a roofline, and neither number describes the shipping
stack. The missing quantity is what the hardware can *actually* deliver, per
processor. Without it, "is there CPU bandwidth headroom left?" has no answer, and
the prior speed work (Q3 FFN, sparse FFN, delta-walk, disk-resident MoE — all
falsified or marginal) has no denominator to be judged against.

## Attainable

| Processor | Attainable read | Method |
|---|---:|---|
| CPU cluster | **127 GB/s** | `membw_probe`, 8-accumulator NEON read, 256 MiB/thread |
| GPU | **367 GB/s** | `diag_profile_kernels`, `f32_gemv` streaming a 2.68 GB matrix |
| SoC (spec) | 400 GB/s | Apple datasheet |

**The CPU cluster reaches 31% of the SoC's advertised bandwidth**, and saturates
at *two threads* (1 thread 76 GB/s → 2 threads 122 → flat to 16). The GPU reaches
92% of spec. **The GPU has 2.9× the attainable read bandwidth of the CPU on the
same chip.**

This is the load-bearing structural fact, and it is not something a better CPU
kernel can move: it is how many fabric ports the P-cluster has.

### The probe validates itself

C12 established that a naive streaming loop measures **issue rate, not
bandwidth** (scalar 9.3 vs NEON 17.7 GiB/s on identical data, size-invariant —
if it were bandwidth-bound they would tie). So `membw_probe` runs every arm at
both a DRAM-resident and a cache-resident size and reports the ratio. A loop that
cannot outrun its own DRAM number out of L2 is measuring itself.

- `read`: cache 1441 GB/s vs DRAM 127 GB/s = **11.4× — passes**, decisively. The
  127 GB/s is a real memory limit.
- `copy`: 1.05×. Reported as a **floor**, not a ceiling — and correctly so, not
  as a probe defect: shrinking the working set does not make a copy
  cache-resident, because the stores still have to reach memory, so this control
  cannot isolate issue rate for that arm. `read` carries the roofline.

The first version of the probe failed its own self-check for a third reason
worth recording: the cache arm was 256× smaller than the DRAM arm, so a single
pass finished in ~10 µs — *below the cost of spawning the thread that ran it* —
and reported cache as **slower** than DRAM. Holding streamed bytes constant
across arms (`STREAMED_BYTES_PER_THREAD`) fixed it. A self-check that fires is
worth more than a number that looks plausible.

## Achieved

### The 47 GB/s was real, and it was measuring a path we no longer run

`bench_mt_shapes` hand-rolls `par_chunks_mut(32)` over **rayon**. That was the
production geometry when the 47 GB/s was recorded. It is not the production
geometry now: `q4k_q8k_matvec_parallel` has routed through the spin-barrier pool
since 2026-06-13, default-on. The new `bench_mt_production` arm calls the actual
production entry point at the same shapes.

Same binary, same entry point, `LARQL_SPIN_POOL` flipped (GiB/s):

| shape | production (spin pool) | rayon (`=0`) | effect |
|---|---:|---:|---:|
| kv_proj 2048×2816 | 92.0 | 43.1 | **+113%** |
| q_proj 4096×2816 | 93.8 | 64.9 | +45% |
| o_proj 2816×4096 | 108.1 | 64.9 | +66% |
| dense_gu 2112×2816 | 59.7 | 40.2 | +49% |
| **big_65536×2816** | **58.5** | **109.6** | **−47%** |

`kv_proj` at 43.1 GiB/s ≈ 46 GB/s is the origin of the standing "47 GB/s"
figure. On the shipping path that shape runs at **98.7 GB/s**.

**So the achieved effective bandwidth on per-layer shapes is 64–116 GB/s =
50–91% of the 127 GB/s attainable ceiling, not 37%.** The CPU decode path is much
closer to done than the record implied. Combined with llama.cpp at ~70 GB/s
effective (≈55% of attainable), the remaining CPU headroom is at most ~1.6× in
theory and considerably less in practice — no quantized matvec with dequant work
reaches STREAM peak.

### And a live 3.5× regression on a default-on path

The last row is not noise, and criterion's default sampling **flattered** it.
Re-measured with `--measurement-time 20`, the pool holds a consistent 33 GiB/s
at that shape — the short-run 58.5 was optimistic. Sweeping participant count,
same code, same shape (GiB/s):

| participants | run 1 | run 2 |
|---|---:|---:|
| 16 (12 P + 4 E) | 33.0 | 33.6 |
| rayon, 16 | 94.9 | 105.0 |
| **12** | **124.0** | — |
| **8** | **126.7** | — |

At 8–12 participants the pool reaches ~136 GB/s — it *saturates* the 127 GB/s
attainable ceiling (the excess is SLC reuse on the 104 MB slab). At 16 it
collapses **3.8×**.

**Mechanism: efficiency-core stragglers under static partitioning.**
`run_chunks` partitions statically, so an E-core is handed a P-core's share and
the completion barrier waits on the slowest participant. Rayon work-steals
instead, which is why the pathology is specific to this pool. Straggler cost
scales with work per dispatch — so small per-layer matvecs still won by 45–113%
and only the lm_head-class shape fell over. That shape-dependence is what hid it.

Prefetch striding was the first hypothesis and is **not** the dominant mechanism:
`run_chunks` did assign chunks round-robin (`c += n_participants`), making each
owner walk the whole slab at an `n × chunk_bytes` stride, but switching to
contiguous blocks did not close the gap — participant count did. The block change
is kept for the prefetch property and pinned by
`each_participant_runs_a_contiguous_block` (the existing exactly-once test passes
under striding too and could not catch a revert), but it is not the fix.

#### Why no benchmark could see it

`configure_rayon_threads` is called from **exactly one place** —
`crates/larql-cli/src/commands/primary/bench/run.rs:28` — and picks 8 on Apple
silicon. It is the workspace's only `ThreadPoolBuilder` outside a scoped one in
hnsw. `run_cmd.rs` configures nothing, and `spin_pool::global()` sized itself
from `rayon::current_num_threads()`.

⇒ **`larql bench` ran at 8 participants; `larql run` and `larql serve` inherited
rayon's default of `available_parallelism()` = 16.** The benchmark harness was
structurally incapable of observing a bug that every non-bench path hit, and all
the historical spin-pool numbers (+28%, ~35 tok/s, "9% ahead of llama.cpp") were
taken on the one path that avoids it.

End-to-end, `gemma4-26b-a4b-q4k`, `--cpu -n 50 --warmup 5`, story prompt:

| threads | tok/s | median |
|---|---|---:|
| 8 | 35.2, 38.0, 37.3 | **37.3** |
| 16 | 13.2, 10.6, 10.6 | **10.6** |

**3.5×**, matching the 3.8× at the kernel. The predecessor doc
`bench/baselines/cpu/DIAGNOSIS-2026-05-16-thread-scaling.md` recorded ~15% for
>8 threads in the rayon era; the spin pool turned a mild effect into a severe one.

#### Fix

`spin_pool::global()` now caps participants at the performance-core count
(`hw.nperflevels` / `hw.perflevel0.logicalcpu` on macOS; unchanged on
homogeneous CPUs), and an explicitly configured smaller thread count still wins.
Capping in the library rather than the CLI fixes it for every consumer —
larql-server, embedders, tests — not just the one binary that set a thread count.
Bandwidth-bound decode gives up nothing: attainable read saturates at **two**
threads, so the E-cores were contributing ~nothing even before they became
stragglers. Pinned by `global_pool_excludes_efficiency_cores`.

**Verified**, same 26B / prompt / `-n 50`, at `--threads 16` (the unconfigured
case), quiet machine on AC:

| | tok/s | median |
|---|---|---:|
| before | 13.2, 10.6, 10.6 | **10.6** |
| after | 34.2, 35.0, 35.9, 34.7 | **34.9** |

**3.3× recovered.** The remaining gap to the 8-thread reference (37.3) is the
12-vs-8 participant difference already visible at the kernel (124.0 vs
126.7 GiB/s), not a residual defect. Suites green: 775 larql-compute (+1) /
1309 larql-inference / 766 larql-kv.

## KV cache priced — and the bytes hypothesis refuted

The last unpriced bandwidth surface. `CpuKvHandle` holds `k_buf`/`v_buf` as
`Vec<f32>` where llama.cpp's default KV is f16, and C10 recorded larql decaying
with context (27.9 → 24.8 tok/s) while llama.cpp's `tg512` stays flat —
attributed at the time to GQA compute, never tested against the competing
explanation that it is simply twice the bytes.

Real geometry (`config.json` + `attn_weights_q4k_manifest.json`, not assumed):
30 layers, `num_key_value_heads` 8 × `head_dim` 256 ⇒ **kv_dim 2048**,
`sliding_window` **1024**, and an explicit `layer_types` array giving
**5 global / 25 sliding** layers. So 16 KB per layer per context-token at f32.

Expressed against the **measured** per-token budget (35 tok/s × 127 GB/s
attainable = 3.63 GB/token). Using measured time rather than a summed weight
estimate deliberately sidesteps an unresolved question about whether the tied
`lm_head` is read at f16 or q4, which would move a weight-based denominator by
about a gigabyte:

| ctx | KV MB/step | % of token budget | KV ms | f16 would save |
|---:|---:|---:|---:|---:|
| 512 | 252 | 6.9% | 1.98 | 0.99 ms |
| 2048 | 587 | 16.2% | 4.62 | 2.31 ms |
| 8192 | 1091 | 30.0% | 8.59 | 4.29 ms |
| 16384 | 1762 | 48.5% | 13.87 | 6.94 ms |

**The sliding window is doing the work.** Past ctx 1024 only 5 of 30 layers keep
growing, so the slope collapses six-fold:

- ctx < 1024: 480 KB per context-token → **3.87 µs/ctx-token**
- ctx > 1024: 80 KB per context-token → **0.65 µs/ctx-token**

**Verdict: don't build f16 KV as a long-context fix.** Two reasons pointing the
same way. It is too small where it matters — 3–8% of a token at chat-typical
context. And it cannot be the cause of the decay it was proposed to explain: the
recorded decay implies roughly 12 µs per context-token against a byte slope of
0.65 beyond the window, and since bytes are a hard *lower* bound on time, at most
about a third of the sub-window decay and a small fraction of the beyond-window
decay is KV traffic. **The original attribution to GQA compute was substantially
right**; the bytes hypothesis is largely falsified, which is the useful outcome —
dead for a couple of hours of arithmetic rather than a KV-quantisation build.

**Where it does retain value is DEC-2 capacity, not decode speed.** f32 costs
480 KB per context-token *per client* — roughly 1 GB of cache per client at
ctx 2048. That is what caps clients-per-box independently of throughput, so
halving it roughly doubles client density. f16 KV should be priced on the
shared-tier ledger, not this one.

## What this means for the programme

1. **CPU bandwidth work is essentially finished.** At 50–91% of a measured
   127 GB/s ceiling, the remaining headroom is small, and the ceiling itself is
   31% of the chip's. Further CPU kernel effort buys back fractions of a
   resource the processor cannot hold more of.
2. **The bandwidth-bound half belongs on the GPU.** 367 vs 127 GB/s attainable
   is a 2.9× structural advantage, and the Metal production kernels already run
   at 273–314 GB/s (75–86% of the GPU ceiling). This is a much stronger argument
   for the G-ladder than "CUDA parity" was.
3. **`q4k_qkv_proj` at 130.8 GB/s is the one GPU outlier** — 36% of the GPU
   ceiling, against 273–314 for its neighbours. That is the first place to look
   on the Metal side.
4. **Correct the ROADMAP claim.** "FFN at 60ms is at the 400 GB/s GPU bandwidth
   ceiling" compares against the spec sheet; the attainable figure is ~367 GB/s
   and the FFN kernels are at 273–314, i.e. near but not at it.
5. **None of this moves the arithmetic-intensity axis**, which is where the only
   measured un-harvested headroom still lives (DEC-0: 13.9% unique experts at
   B64 ⇒ ~7.2× grouped-scheduler headroom). Bandwidth *ceilings* are a
   denominator; batching changes the numerator.

## Open

- **E-cores are now excluded, not exploited.** The cap is the right fix for a
  bandwidth-bound path, but a compute-bound section would still want those 4
  cores. Recovering them needs either core-class-weighted static blocks or
  work-stealing — and work-stealing is what the current barrier-soundness
  argument trades away (a dynamic resettable cursor is exactly the bug that was
  caught during the pool's original build). Not worth it until something
  compute-bound is on this path.
- **The CLI's non-bench paths still leave rayon itself at 16.** Harmless now
  that the pool is capped (rayon work-steals), but `larql run` and `larql serve`
  configuring no thread count at all is the condition that let this ship.
- **KV bytes are priced and closed** (see above) — but the *compute* side of the
  long-context decay is now the live question, since it inherits most of the
  effect the bytes hypothesis failed to explain. `TurboQuantEngine` exists in
  larql-kv and is not on the production handle; it stays unbuilt on this ledger
  and moves to DEC-2 as a client-density item.
- **The registry is the system of record** for this measurement: experiment
  `c0-bandwidth-roofline` in programme `dec`, run
  `RUN-20260729-221048-00532`.
- **A quiet-machine re-run of `membw_probe`** would be worth having. The numbers
  above were taken at load 1.5–1.9 with <3% spreads, but a later re-run during a
  concurrent `llvm-cov` job (load average 98) produced physically impossible
  figures — a reminder that this probe has no contention guard beyond the spread
  column, and that the spread column is the thing to read before the GB/s one.

## Reproducing

```bash
# CPU attainable ceiling (AC power, quiet machine — check `uptime` first)
cargo run --release -p larql-compute --example membw_probe -- --json membw.json

# Achieved, production path vs the rayon path the 47 GB/s came from.
# Flags INLINE — `env $FLAGS` does not word-split under zsh and silently drops
# all but the first (project_spin_barrier_pool).
cargo bench -p larql-compute --bench q4k_q8k_matvec -- 'q8k_mt_(shapes|production)'
LARQL_SPIN_POOL=0 cargo bench -p larql-compute --bench q4k_q8k_matvec -- q8k_mt_production

# GPU attainable + per-kernel achieved
cargo run --release -p larql-compute-metal --example diag_profile_kernels
```
