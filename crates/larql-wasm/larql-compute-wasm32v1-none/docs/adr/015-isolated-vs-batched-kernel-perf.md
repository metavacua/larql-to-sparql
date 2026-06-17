# ADR-015: Isolated kernel speedup ≠ end-to-end win when batched throughput is already saturated

**Status**: Accepted (recurring pattern, four direction-mismatch instances + one magnitude-compression instance)
**Date**: 2026-05-02 (initial; updated with NR2 then `q4k_matvec` lm_head; NR2 kernel removed 2026-05-09 after second-confirming bench; QKV defuse added 2026-05-09 as the first magnitude-compression case)
**Context**: A pattern that has now reproduced across three independent kernel
optimisation attempts on Gemma 3 4B decode. Future kernel work needs to budget
benchmark cost against this prior — the isolated `diag_profile_kernels` number
is necessary but not sufficient evidence to promote a new shader.

## The pattern

A candidate kernel shows a meaningful speedup in the **isolated** profiler
measurement (one commit+wait per call, includes ~20 µs GPU spin-up) but
either matches or *regresses* the **batched** measurement (n_layers
dispatches in one cmd buffer, single commit+wait — matches the real decode
pipeline).

End-to-end decode benchmarks then track the batched number, not the isolated
one. The isolated win was real — it just was not load-bearing under the
production workload.

## Four confirmed instances

| Kernel | Isolated speedup | Batched delta | End-to-end | Outcome |
|---|---|---|---|---|
| `q4k_ffn_gate_up_f16acc` (2026-04-28) | 1.79× (0.607 → 0.340 ms) | within noise | parity on quiet GPU | opt-in only (`LARQL_F16_ACC=1`) |
| `attn_fused` (2026-05-01) | merged 2 kernels into 1 | TGs collapse 12 → 8 | **−1.45 ms regression** | opt-in only (`LARQL_FUSED_ATTN=1`) |
| `q4k_ffn_gate_up_nr2` (2026-05-02) | 1.47× (0.591 → 0.401 ms iso) | 279 → 267 GB/s (−4%) | **−0.62 ms regression on GPU fwd** (re-bench 2026-05-09: 11.71 → 12.05 ms, **−0.34 ms / −2.9 tok/s**, direction matched batched diag) | **kernel + `LARQL_GATE_UP_NR2` env-var removed 2026-05-09**; the second bench confirmed the batched-diag prediction so the candidate was retired rather than left dangling as opt-in |
| **`q4k_matvec` lm_head** (broken-fast → fixed) | n/a — different category | 1.47 ms (broken) vs stride-32's 2.95 ms | initially +10 tok/s but FAILED smoke ("Capital" / truncated). **Root cause: dispatch geometry mismatch, not kernel-level drift. Fixed 2026-05-02 — kernel was correct all along.** | now production default; fixed `pipeline.rows_per_tg` / `threads_per_tg` lookup. Net **+8 tok/s end-to-end**. |

The mechanisms differ but the symptom is identical at the perf level — a
candidate that looks like a strict win at one measurement granularity and
loses (or breaks) at the actual production granularity.

- **f16 acc**: the kernel was already at 274 GB/s = 74% of LPDDR5X peak.
  Freed ALU cycles got absorbed by surrounding kernels' bandwidth contention
  rather than translating to wall-clock reduction.
- **attn_fused**: dispatch fusion saved ~30 µs of cmd-buffer overhead but the
  fused kernel's larger register footprint forced 8 TGs/dispatch instead of
  the unfused path's 12. Parallelism loss dwarfed the dispatch saving.
- **NR2**: the isolated measurement caught dispatch-overhead amortisation
  that disappears once n_layers calls share one cmd buffer. The batched
  geometry is the production geometry, and NR2 is *worse* there.
- **`q4k_matvec` lm_head** (initial diagnosis WRONG, corrected 2026-05-02):
  the symptom was a fast-but-broken kernel — identical Q4_K bandwidth as
  `q4k_matvec_stride32` (327 MB/token) but at 1.47 ms vs 2.95 ms, with
  argmax drift on the canonical "Paris" smoke ("Capital" / "is: **"
  truncated). Initial conclusion: 32-lane simdgroup reduction tree drift.
  **Real root cause: dispatch geometry mismatch.** `MetalBackend::q4k_matvec`
  hardcoded the 4sg shader's `THREADS_PER_TG=128` while dispatching the
  8sg `q4k_matvec_pipeline` (production default since 2026-04-28).
  Simdgroups 4..7 of each 8sg TG never executed → half the rows in each
  8-row TG were left unwritten → 50% of lm_head output corrupt → argmax
  flipped on close-call tokens. **Same family as the 2026-04-26 `077884b`
  "81–84 tok/s on broken Q4_K dispatch"** (pre-fix `q4k_matvec` routed
  through `q4_matvec` with mismatched threadgroup geometry, 75% of
  output rows unwritten). Once dispatch was corrected to use
  `pipeline.rows_per_tg` / `pipeline.threads_per_tg`, parity test
  `q4k_matvec_matches_cpu` flipped from 182.89 max diff to passing, and
  the kernel's 1.85 ms/tok lm_head landed +8 tok/s end-to-end.
  **Reclassified: not a broken kernel; a dispatch-geometry-mismatch
  family, distinct from the iso-vs-batched pattern but worth pinning here
  because the diagnostic surface is similar — a "broken-fast" number that
  invites suspicion of the kernel before the dispatcher.**

### Lesson — diagnostic order for "fast but wrong" results

When a candidate kernel produces correct output on some inputs and wrong
output on others (especially close-call top-1 flips), the order to check is:

1. **Dispatch geometry first.** Does the dispatch site use the bound
   pipeline's `rows_per_tg` / `threads_per_tg`, or hardcoded shader-module
   constants? If hardcoded constants and the pipeline binds to a different
   variant, you have an under-dispatch — half the simdgroups don't run,
   half the output rows unwritten. **Two confirmed instances** (077884b
   and the 2026-05-02 `q4k_matvec` lm_head) — both fast-and-wrong with
   correct-looking partial output that masks the bug on simple prompts
   and surfaces on close-call tokens.
2. **Shader correctness next.** Run the kernel with a known-good dispatch
   (or vary the dispatch geometry to match). If parity still fails,
   suspect the shader.
3. **Reduction tree last.** FP rounding from a parallel reduction can
   drift on the order of 1e-3 — enough to flip top-1 only when scores are
   already razor-thin. If the diff is larger than 1e-2, it is almost
   certainly NOT the reduction tree.

## Diagnostic test before promoting any new kernel

Run all three measurements before deciding:

1. **Isolated** (`diag_shader_bench` `iso_ms` column): cheap, fastest signal.
   A regression here is enough to drop the candidate. A win here is
   necessary but not sufficient.
2. **Batched** (`diag_shader_bench` `bat_ms` / `GB/s` columns): the
   production geometry. **This predicts the *direction* of end-to-end
   reliably. It does NOT predict the *magnitude*.** If batched regresses
   or is within noise, the candidate is not a win.
3. **End-to-end bench A/B** (`larql bench --warmup 8 -n 30 --profile`):
   final confirmation, with correctness smoke (`larql run "The capital of
   France is" -n 8 --metal` should still emit Paris).

Steps 1 and 2 take ~30 s total. Step 3 takes another minute. Skipping step 2
and going straight from isolated → end-to-end has burned three sessions; do
not skip it.

### Magnitude can compress 4×: QKV defuse case (2026-05-09)

The QKV defuse change (revert the fused `q4k_q6k_qkv_proj_normed` kernel
back to a separate `rms_norm` + non-fused `q4k_q6k_qkv_proj` pair —
canonical record at [ADR-016](016-defused-rms-norm-qkv.md)) is the first
case where the batched diag's *direction* matched but its *magnitude*
did not.

| measurement | predicted | observed |
|---|---|---|
| batched diag (kernel-only) | 4.59 ms (fused) → 3.13 ms (non-fused) = −1.46 ms |  |
| + rms_norm dispatch added back | +0.24 ms (1 dispatch × 34 layers × 7 µs) |  |
| **predicted end-to-end Δ** | **−1.22 ms/tok** |  |
| **measured end-to-end Δ** |  | **−0.22 ms/tok** (warmup 8, n=100, drift 0.02 ms) |

The non-fused kernel's 287 GB/s peak is a *single-kernel-in-isolation*
number. In the production decode pipeline, it shares the LPDDR5X bus with
attention, FFN gate+up, FFN down, lm_head — all of which are also bandwidth-
bound. The candidate doesn't get to claim its kernel-isolated headroom
when the surrounding pipeline is already saturating the same bus.

**When to expect magnitude compression:** the candidate's win comes from
*bandwidth headroom* rather than *cycles* (compute-bound kernels) or
*dispatches saved* (cross-kernel structural changes). Bandwidth headroom
that exists in isolation may not translate to wall-clock reduction when
the pipeline is already bandwidth-saturated as a whole.

**Implications for promotion decisions:**

- A predicted batched-diag delta is a **lower bound on conviction** ("yes
  this should help"), not an upper bound on celebration ("we'll get the
  full ms back"). Discount the predicted magnitude by ~3-5× when
  budgeting against ollama parity.
- The diag is still load-bearing — without it we wouldn't know the
  candidate is worth running end-to-end. But the *gap-closing budget*
  needs to be sized off the end-to-end measurement, not the diag.
- Direction-mismatch (NR2, f16_acc, attn_fused) remains a strict killer.
  Direction-match + magnitude-undershoot is still a promotion signal,
  just a smaller one than the diag suggests.

### Mechanised flow (2026-05-02)

`diag_shader_bench --profile gemma3` with `--json` and `--compare` automates
steps 1 and 2 against a saved baseline. The full save-then-compare command
pair lives in `crates/larql-compute/PERFORMANCE.md` under "How to A/B a
shader candidate" — that is the canonical promotion gate. Use
`--threshold N` to set the percent regression considered a real loss
(default 5%).

## When the pattern does NOT apply

The 8sg geometry rollout (2026-04-28) showed when isolated wins *do* carry
end-to-end: `q4k_matvec_8sg` at 55% LPDDR5X utilisation gave +5.2% end-to-end;
gate+up at 68% gave +2.1%; q6k_matvec at 84% gave 0% (regressed). The
predictor is **bandwidth headroom under the batched measurement**: kernels
below ~75% of LPDDR5X peak have room to convert isolated wins into batched
wins. Kernels above ~80% don't.

## Decision

1. ADR pinned. New shader work follows the three-step diagnostic above.
2. The lesson lives in three places so it's findable from each entry point:
   this ADR (canonical), `PERFORMANCE.md` current-state data, and
   `PERFORMANCE.md` recent-changes table per instance.

## Consequences

- New candidates that look hot in `diag_profile_kernels` isolated column do
  not justify a session of end-to-end measurement on their own.
- Kernels that pass the batched test (e.g. fused QK norm + RoPE; the
  May 2026 fusion wave that landed −1.5 ms cumulatively) are the
  evidence-based bar for promotion.
- Decode at ~88 tok/s on this hardware (post 2026-05-09 QKV defuse) is
  close to the parallelism / bandwidth ceiling. Closing the remaining
  ~1.17× to ollama needs work that *changes the batched measurement*,
  not work that just makes a single kernel faster in isolation. Was
  ~1.30× when this ADR was first written; the dispatch-geometry fix
  (2026-05-02, +7.7 tok/s) and QKV defuse (2026-05-09, +1.6-1.8 tok/s)
  were both batched-measurement-changing wins, exactly the class of
  work this ADR points at.

## Related

- ADR-008 (Q4_K kernel optimization findings) — predecessor pattern at the
  matvec level.
- ADR-016 (Defused RMS norm + QKV) — the canonical record of the
  QKV-defuse decision; this ADR holds the *general lesson* (magnitude
  compression), ADR-016 holds the specific decision.
- `crates/larql-compute/PERFORMANCE.md` recent-changes table.
- `crates/larql-compute/ROADMAP.md` "P0: Production gap closers" — multi-TG
  `attn_fused` retry is the next target that explicitly works *with* this
  pattern (preserve TG count while fusing).
