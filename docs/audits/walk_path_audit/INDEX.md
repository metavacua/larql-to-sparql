# walk_path_audit — baseline index

Per-path equivalence audit for `WalkFfn` dispatch paths. Each entry below
records a measurement of one (model, vindex variant) pair against the
`WeightFfn` dense matmul reference, with the assertion bounds locked in
from that measurement.

## Methodology

For each `WalkFfn` path a forced-dispatch measurement is taken via a
`MaskedGateIndex` wrapper that hides the `has_*` flags above the target
path in the routing ladder. Three prompts (anchor + factual + code) are
run end-to-end through `predict_with_ffn`, with a per-layer `DualFfn`
capturing the diff between the path's output and the reference at every
(layer, position).

Assertion metrics are **cos** and **relative L2** (`L2 / ‖primary‖`),
both magnitude-invariant. Absolute L2 and max-element drift are kept as
diagnostic columns to surface residual-magnitude outliers (e.g. the
L11/code/1 ` fibonacci` spike on Gemma 3 4B) without driving the
verdict. Each path additionally gates on top-1 token match across all
three prompts and an end-to-end Paris-prompt probability delta.

Per-path bucketing uses `GateIndex::primary_storage_bucket()` —
encapsulates the `has_*`-flag → bucket mapping so audits don't scatter
flag-checks across their bucketing logic. Path bounds are then per-bucket
(see `BOUND_EXACT` / `BOUND_QUANTIZED` / `BOUND_FP4` constants in the
source). The `sparse` path's bucket is vindex-dependent (it walks
whatever data the unified `ffn_row_*` dispatch picks); paths with fixed
precision (`interleaved`, `interleaved_q4k`, etc.) have hardcoded
buckets.

Bound floors use a measure-then-tighten rule: cosine floor at one
decimal less precise than the measured worst (loose enough to survive
an Accelerate FMA reordering); rel_L2 ceiling at measured worst × 4.

Source: `crates/larql-inference/examples/walk_path_audit.rs`.

### On the cos ↔ rel_L2 relationship

For two vectors of similar magnitude, `rel_L2 ≈ √(2·(1−cos))`, so the
two assertion metrics carry the same information up to a monotonic
transform. The implication for bucketing:

- **Exact bucket** (cos ≥ 0.99999): expected rel_L2 ≈ 4.5e-3. The
  current 1e-2 ceiling has 2× headroom over the relationship's lower
  bound — both metrics are useful as independent gates.
- **Quantized bucket** (cos ≥ 0.99): expected rel_L2 ≈ 0.14. The 5e-1
  ceiling reflects measured-worst × 4 honestly; cos is the meaningful
  primary assertion for this bucket. rel_L2 is informational, not a
  tight independent gate.
- **FP4 bucket** (cos ≥ 0.98): expected rel_L2 ≈ 0.20. Same logic as
  quantized — cos primary, rel_L2 informational. Bound TBD pending FP4
  baseline.

If a future cos floor change is contemplated for any bucket, recompute
the corresponding rel_L2 ceiling from the relationship; don't tighten
one in isolation.

## Baselines

| date | model | vindex | bucket | paths tested | min cos | max rel L2 | Paris ΔP | n_obs | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-01 | google/gemma-3-4b-it | gemma3-4b-f16 | Exact | sparse, full_mmap, exact | 0.999997 | 1.881e-3 | 1.43e-4 | 1,326 | 3/3 PASS |
| 2026-05-01 | google/gemma-3-4b-it | gemma3-4b-q4k-v2 | Quantized | sparse, interleaved_q4k | 0.992737 | 1.205e-1 | 2.58e-2 | 1,326 | 2/2 PASS |

### 2026-05-01 — Gemma 3 4B f16 (Exact baseline)

The f32 paths agree at cos = 0.999997 across 1,326 observations, three
independent code paths land on identical assertion values, dispatch
trace verified 102/102 layers per path. Worst rel_L2 observed at
L32/paris/0 (BOS position of the Paris prompt). Top-1 token matches on
all three prompts × three paths; Paris probability holds to within
1.4e-4 of dense.

Bounds locked: `cos ≥ 0.99999, rel_L2 ≤ 1e-2, paris_ΔP ≤ 5e-3`. The
rel_L2 ceiling is intentionally loose pending Q4K and FP4 baseline
measurements — see inline comment at `BOUND_EXACT` for the sequencing
rule. Target post-matrix tightening: ~7.5e-3 (= measured × 4).

Artifacts: `walk_path_audit_gemma3_4b_f16_baseline.{md,json}`.

### 2026-05-01 — Gemma 3 4B Q4K v2 (Quantized baseline)

Both quantized paths preserve top-1 across all three prompts. Sparse
(walks Q4K via `q4k_ffn_row_dot` on this vindex) and
`interleaved_q4k:dequant` agree to within Q4K dequant noise of dense:
cos = 0.996306 / 0.992737, rel_L2 = 9.562e-2 / 1.205e-1, Paris ΔP =
4.171e-3 / 2.576e-2. Worst observations at L14/paris/1 (sparse) and
L10/code/1 (interleaved_q4k) — both early-layer code-prompt positions
where residual magnitudes are largest.

The wide gap between the two paths' rel_L2 measurements (9.6% vs 12%)
sits inside the cos↔rel_L2 envelope above; both reflect the same
underlying directional drift to within block-quantization noise.

Bounds locked: `cos ≥ 0.99, rel_L2 ≤ 5e-1, paris_ΔP ≤ 5e-2`. The
quantized rel_L2 ceiling is loose by design (cos is the meaningful
primary assertion); the Paris ΔP budget matches `walk_correctness.rs`'s
Q4K-down threshold (0.035) with margin for prompts more sensitive to
softmax redistribution than Paris.

Artifacts: `walk_path_audit_gemma3_4b_q4k_baseline.{md,json}`.

## Sequenced follow-ups

Each is its own measure-bound-commit cycle, separate PR:

1. ~~`gemma3-4b-q4k-v2.vindex` → measure `interleaved_q4k:dequant`~~ —
   landed 2026-05-01.
2. `gemma3-4b-fp4a.vindex` → measure `fp4_storage:sparse`, set FP4
   bound at measured × 4. Apply same cos↔rel_L2 sanity check before
   committing.
3. Single cross-bucket bound-tightening commit once all three
   measurements are in (will tighten the f16 exact rel_L2 from the
   intentionally-loose 1e-2 to ~7.5e-3 = f16 measured × 4).
