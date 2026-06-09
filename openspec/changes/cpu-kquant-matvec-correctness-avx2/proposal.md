## Why

larql's driving value proposition is running models + max context that
**exceed available GPU VRAM** by keeping FFN (and MoE expert) weights in
system RAM and computing them on CPU. GPU is reserved for what genuinely
needs it — attention, KV-cache, and similar bandwidth-bound paths where
the weight footprint is small.

Mid-session 2026-05-13 the team paused to ask whether the CPU FFN matvec
hot path was actually correct and fast enough to deliver on that promise
for Qwen 3.6's mix of Q4_K / Q6_K weights. The answer was **no on both
counts**: a layout bug had been silently corrupting all Q6_K matvec
output since inception, and the production CPU Q6_K matvec was scalar
while Q4_K had been AVX2 for months.

This proposal captures the audit + fix arc that landed in PRs #102,
#103, #104, #105, #106, #107, #108. No additional code changes ship
under this proposal — it is a retroactive contract for what was already
shipped, written to make those guarantees re-checkable.

## What This Change Ships

**Capability deltas** (under `compute-backend-traits/`):
- CPU Q6_K matvec correctness: reads the canonical llama.cpp Q6_K wire
  format on both the f32-input trait path (`CpuBackend::q6k_matvec`) and
  the Q8_K-input AVX2 path (`q6k_q8k_matvec_into`).
- AVX2 dispatch on x86_64 + `avx2` feature for the Q8_K-input Q6_K
  matvec, bit-exact vs scalar.
- Canonical-dequant oracle coverage for both Q4_K and Q6_K production
  matvec entry points.
- `walk_ffn_q8k` Q6_K vs Q4_K format dispatch (the latent bug fix that
  unblocked Q6_K FFN_DOWN through the `/v1/walk-ffn-q8k` endpoint).

**No code changes** under this proposal id — code already merged via the
PR sequence below.

## Audit and fix sequence (2026-05-13 to 2026-05-14)

| PR | Kind | Summary |
|----|------|---------|
| [#102](https://github.com/ianblenke/larql/pull/102) | fix | Q6_K layout fix — `q6k_q8k_matvec_scalar` and `q6k_matvec::dispatch` read the canonical llama.cpp interleaved-stride wire format (replaces the buggy sequential nibble layout). |
| [#103](https://github.com/ianblenke/larql/pull/103) | fix | `q4k_ffn_forward_layer_q8k` dispatches on `ffn[2].1` (`"Q4_K"` / `"Q6_K"`) so default-extracted vindexes (Q6_K FFN_DOWN under `--down-q4k=false`) stop mis-parsing Q6_K bytes through a Q4_K parser. |
| [#104](https://github.com/ianblenke/larql/pull/104) | perf | AVX2 `q6k_q8k_matvec_avx2`, bit-exact vs scalar — sign-trick `maddubs` / `madd` per llama.cpp Q6_K layout. |
| [#105](https://github.com/ianblenke/larql/pull/105) | test | Canonical-dequant oracle for Q4_K × Q8_K matvec (parallel to the one #102 added for Q6_K). |
| [#106](https://github.com/ianblenke/larql/pull/106) | test | Cross-path parity: `q6k_matvec::dispatch` (f32 scalar) vs `q6k_q8k_matvec_into` (Q8K AVX2) on identical Q6_K weights within Q8_K activation noise. |
| [#107](https://github.com/ianblenke/larql/pull/107) | test | `quantize_q6_k` round-trip via canonical `dequantize_q6_k` (defends the vindex extraction input). |
| [#108](https://github.com/ianblenke/larql/pull/108) | bench | Head-to-head AVX2 Q8K-input vs scalar f32-input Q6_K matvec, three reference shapes. |
| [#110](https://github.com/ianblenke/larql/pull/110) | perf | `CpuBackend::q4k_matvec` / `q6k_matvec` (trait dispatch) routed through the AVX2 Q8K kernel — lm-head KNN, speculative draft head, attention V CPU fallback now hit the AVX2 fast path. |
| [#112](https://github.com/ianblenke/larql/pull/112) | perf | `q4k_q8k_gate_up_into` x86_64 fallback dispatches to `q4k_q8k_matvec_into` (AVX2) instead of the scalar reference — walk-ffn-q8k FFN gate+up step ~3.8× faster at prefill_10240. |
| [#113](https://github.com/ianblenke/larql/pull/113) | bench | Head-to-head Q4_K gate+up: fused vs two-matvec, validates #112 empirically. |
| [#114](https://github.com/ianblenke/larql/pull/114) | perf | Q4_KF trait — streaming dequant+dot, eliminates 2.7 GB allocation, 1.75 s → 503 ms at lm_head shape. |
| [#115](https://github.com/ianblenke/larql/pull/115) | perf | Q4_KF AVX2 (`q4kf_q8k_matvec_avx2`) — completes the K-quant AVX2 family. Q4_KF lm_head 503 ms → 74.6 ms (~6.7×); cumulative 1.75 s → 74.6 ms (~23×). |

## Bench results (RTX 4090 host, x86_64 + AVX2)

`cargo bench -p larql-compute --bench quant_matvec -- q6k_q8k_vs_q6k_f32 --quick`

| Shape (rows × hidden) | AVX2 Q8K-input | Scalar f32-input | Speedup |
|------------------------|---------------:|-----------------:|--------:|
| `decode_2560` (2,560 × 2,560)          | **376 µs**   (17.4 Gelem/s) | 7.51 ms (871 Melem/s) | **~20×** |
| `prefill_10240` (10,240 × 2,560)       | **1.56 ms**  (16.8 Gelem/s) | 28.8 ms (910 Melem/s) | **~18×** |
| `lm_head_262144` (262,144 × 2,560)     | **40.6 ms**  (16.5 Gelem/s) | 738 ms (909 Melem/s)  | **~18×** |

**Driving-goal payoff**: the `lm_head_262144` row is the VRAM-constrained
user scenario. When lm_head is forced to CPU (it can't fit in VRAM
alongside attention + KV-cache for large-context configs), Q6_K matvec
goes from 738 ms → 40.6 ms per token. The difference between unusable
and viable for the "models + max context > VRAM" value prop.

## Post-#110 update — trait dispatch numbers

After #110 routed `CpuBackend::q4k_matvec` and `q6k_matvec` (the
trait-dispatched f32-input entry points) internally through the AVX2
Q8K kernel, the existing `quant_matvec_q4_k` / `quant_matvec_q6_k`
Criterion groups (which call `backend.quant_matvec(format, ...)`)
hit the same fast-path numbers as the direct `q*k_q8k_matvec_into`
calls:

`cargo bench -p larql-compute --bench quant_matvec -- "quant_matvec_q4_k|quant_matvec_q6_k" --quick`

| Shape           | Q4_K trait | Q6_K trait | Q4_K throughput | Q6_K throughput |
|-----------------|-----------:|-----------:|----------------:|----------------:|
| decode_2560     |   **360 µs** |   **406 µs** |   18.2 Gelem/s |   16.1 Gelem/s  |
| prefill_10240   |  **1.43 ms** |  **1.56 ms** |   18.3 Gelem/s |   16.7 Gelem/s  |
| lm_head_262144  |  **37.6 ms** |  **42.4 ms** |   17.9 Gelem/s |   15.8 Gelem/s  |

Q4_K is ~10 % faster than Q6_K at the same shape — consistent with
Q6_K's 50 % larger super-block (210 B vs 144 B per 256 elements).

**lm_head trait dispatch went from 738 ms → 37.6 ms** for Q4_K-stored
vindexes (e.g. all Qwen 3.6 weight files extracted via the default
`write_q4k/lm_head.rs:26` path). Speculative decode draft head, lm-head
KNN scoring under VRAM pressure, and attention V CPU fallback all
inherit this win automatically because they reach the trait.

### Q4_KF closure — #114 + #115

Same bench run surfaced Q4_KF at 1.75 s at the lm_head shape (full f32
dequant + scalar dot, no AVX2). Closed in two PRs:

| Phase           | decode_2560 | prefill_10240 | lm_head_262144 |
|-----------------|------------:|--------------:|---------------:|
| Pre-#114 baseline |   6.58 ms |  73.1 ms      | **1.75 s**     |
| Post-#114 streaming dequant |  4.9 ms |  19.6 ms |  503 ms      |
| Post-#115 AVX2  |   **720 µs** |  **2.90 ms** |  **74.6 ms**   |
| Cumulative speedup | ~9× | ~25× | **~23×** |

Q4_KF throughput now ~9 Gelem/s (vs Q4_K AVX2's ~18 Gelem/s) — Q4_KF
uses f32 scale arithmetic per sub-block rather than Q4_K's deferred i32
sum1/sum2. Closing the remaining gap would require an inverted
Q4_KF→Q4_K layout conversion or a hybrid kernel; not pursued because
Q4_KF is not on the vindex production path (see `LayerWeightFormat`
enum in `larql-vindex/src/format/weights/write_layers.rs:28-37`).

## AVX2 K-quant family — complete

After PR #115, every K-quant CPU matvec entry point on x86_64 has an
AVX2 fast path through a Q8_K-input kernel:

| Format | Direct entry | Trait entry | Throughput at lm_head_262144 |
|--------|---|---|---:|
| Q4_K   | `q4k_q8k_matvec_into` (#110-era) | `CpuBackend::q4k_matvec` (#110) | **17.9 Gelem/s** |
| Q4_KF  | `q4kf_q8k_matvec_into` (#115)   | `CpuBackend::q4kf_matvec` (#115) | 9.0 Gelem/s |
| Q6_K   | `q6k_q8k_matvec_into` (#104)    | `CpuBackend::q6k_matvec` (#110) | 15.8 Gelem/s |

Fused gate+up (`q4k_q8k_gate_up_into`, walk-ffn-q8k hot path) also
reaches AVX2 on x86_64 after #112.

## Pre-existing bugs found, fixed, and now regression-protected

1. **Q6_K wire-format layout mismatch** (production correctness):
   `quantize_q6_k` wrote the llama.cpp interleaved-stride layout; the two
   matvec readers (`q6k_q8k_matvec_scalar` and `q6k_matvec::dispatch`)
   read a sequential nibble layout. Internally consistent — and the
   previous parity test that used both readers was tautological — so
   nothing flagged it. Every vindex-extracted Q6_K matvec produced
   garbled output. Smooth-input synthetic test showed 7.2 % rel error;
   real weight distributions would be much worse.

2. **`walk_ffn_q8k` Q6_K vs Q4_K format dispatch missing** (production
   correctness): the `/v1/walk-ffn-q8k` endpoint called
   `q4k_q8k_matvec_into` (a Q4_K parser, 144 B/block) on whatever bytes
   the vindex stored for FFN_DOWN. Default-extracted vindexes store
   FFN_DOWN as Q6_K (210 B/block) — see
   `format/weights/write_q4k/ffn.rs:74`'s `is_down && !opts.down_q4k`
   gate. Result: silently garbled FFN deltas through this endpoint.

3. **NEON Q6_K matvec dormant on legacy layout** (latent correctness):
   the aarch64 `q6k_q8k_matvec_neon` reads the same legacy sequential
   layout. #102 disabled its dispatch (kept as `#[allow(dead_code)]`
   reference); re-vectorisation against the canonical layout is tracked
   under [task #123](../../tasks.md).

## Architectural retrospective

The Q6_K work was on-mission against the driving goal. Earlier in the
same session arc, work had drifted toward GPU-FFN (Phase F.4/F.6 Q8_0
device-resident matvec, Phase E.6.B device-resident FFN block, CUDA
Graphs over FFN decode). Those PRs preserved on branches but **not the
production direction** — they push weights toward the GPU, defeating
the value prop. See [project memory:
larql_driving_goal](../../../../memory/project_larql_driving_goal.md).

For future perf work in this area: the first question on every proposal
is "does this make CPU FFN faster, or does it push weights toward the
GPU?" Only the former is in scope.

## Capabilities

- `compute-backend-traits` — ADDED requirements for CPU K-quant matvec
  correctness + AVX2 dispatch contract.

## Impact

- **Affected files (already merged)**:
  `crates/larql-compute/src/cpu/ops/q4k_q8k_dot.rs`,
  `crates/larql-compute/src/cpu/ops/q6k_matvec.rs`,
  `crates/larql-compute/src/cpu/ops/q4_common.rs`,
  `crates/larql-compute/benches/quant_matvec.rs`,
  `crates/larql-inference/src/vindex/q4k_forward/walk_ffn.rs`.
- **Affected systems**: CPU FFN matvec, vindex Q6_K storage, walk-ffn-q8k
  endpoint, attention V projection (CPU fallback), lm-head KNN (CPU
  fallback).
- **Out of scope**: NEON Q6_K re-vectorisation (#123), Metal Q6_K shader
  audit (#124), Q5_K production handling (vindex doesn't store Q5_K).
