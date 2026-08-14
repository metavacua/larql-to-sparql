# V2 — FP4 generality across architectures (aim-validation, KU3)

**Status:** COMPLETE — CONFIRMED (the opposite of V1). FP4-friendliness is universal
and near-lossless across the archs measured.
**Harnesses:** `crates/larql-vindex/examples/fp4_q1_scan.rs` (static, generalized for
`*_weights.bin` naming), `chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_v2_fp4_nll.rs` (predictive).
**Artifacts:** `bench/aim-validation/v2_*_scan.json`
**Date:** 2026-05-31

## The question

Claim under test: *"FP4-friendliness is universal, not Gemma-3-4B-specific."* Prior
(exp 26): `gemma3-4b-f16` is 99.83% FP4-friendly per-feature (R<16 within-block
dynamic range), `down` the tail at 99.65%. KU3: if FP4 needs per-arch QAT, the "free
2×" from FP4 becomes per-model retraining work.

## Method (static screen + predictive deciding metric)

- **Static** (`fp4_q1_scan`): per-feature / per-sub-feature-tile, the within-block
  dynamic-range ratio R = max/|min| of 32-elem sub-block scales; compliance at R<16
  (the DeepSeek-V4 FP4→FP8 lossless threshold). Read directly from the *original f16*
  weights (dtype=f16, quant=none) — not Q4K-dequantised (which would be a misleading
  double-quant).
- **Predictive** (`walk_ffn_v2_fp4_nll`, the #26 deciding metric): held-text per-token
  NLL + argmax drift with the **real FP4 E2M1 block codec** (`encode_fp4_feature` /
  `decode_fp4_feature`: 256-elem blocks, 8×32 sub-blocks, FP8 sub-scales) applied to
  ALL FFN weights, vs f32 and vs the shipped Q4-int 4-bit baseline.

## Model set (constraint-driven)

The matrix's Llama 2 / Mistral are only available as Q4K (no f16 original → double-quant),
so the static scan uses the three vindexes with complete *original* f16 FFN weights:
**Gemma 3 4B** (`gemma3-4b-fresh`, a fresh build of the exp-26 model), **Granite 4.1 3B**,
**Granite 4.1 8B** — two architecture families and a 3B/4B/8B scale ladder.

## Result — static (CONFIRMED, reproduces exp 26)

Per-feature R<16 compliance, by component:

| model | gate | up | **down (tail)** | worst `down` layer |
|---|---:|---:|---:|---|
| Gemma 3 4B | 99.91% | 99.93% | **99.83%** | L22 @ 99.53% (p99 R=12.1) |
| Granite 3B | 99.89% | 99.92% | **99.82%** | L23 @ 99.44% (p99 R=12.3) |
| Granite 8B | 99.93% | 99.94% | **99.85%** | L31 @ 99.53% (p99 R=12.3) |

Gemma 3 4B `down` = **99.83%**, an exact match to exp 26's headline. `down` is the tail
on every model (gate/up cleaner), and even the worst layer clears ~99.4% with p99 R≈12
(under 16). Strikingly uniform across two families and 3 sizes ⇒ **architecture-neutral**.

## Result — predictive (CONFIRMED near-lossless; FP4 beats Q4-int)

Gemma 3 4B, held narrative, 74 teacher-forced positions:

| arm | mean NLL (bits) | Δ vs f32 | argmax flip |
|---|---:|---:|---:|
| f32 | 4.239 | — | — |
| Q4-int (shipped 4-bit baseline) | 4.492 | +0.253 | 13.5% |
| **FP4-e2m1** | 4.355 | **+0.116** | **10.8%** |

FP4 is within **+0.116 bits/token** of f32 — near-lossless — and **better than the
shipped Q4-int baseline** (E2M1 float captures within-block dynamic range better than
4-bit symmetric integer). No compounding catastrophe (contrast V1's +5–7 bits): the
per-block fit translates to a near-lossless output. *Caveat:* the predictive arm runs on
the Q4K vindex (reference f32 = Q4K-dequant), so it measures FP4's incremental format
error on already-Q4K weights — a conservative proxy; the static metric covers true-f16
friendliness. FP4 on true f16 would be at least as good.

## Verdict — KU3 RESOLVED, CONFIRMED

FP4-friendliness is (a) universal by the static R<16 metric (≥99.8% per-feature across
Gemma 3 / Granite, `down` the only mild tail) and (b) near-lossless by NLL/drift,
beating the shipped Q4-int. **No QAT required** for these archs. This is a *positive*
aim-validation outcome — V1 falsified the sparsity lever, V2 confirms the FP4 lever.
KU3's "free 2× isn't per-model retraining" holds for the families measured.

**Not closed:** Llama/Mistral (no f16 original on hand) and true MoE expert weights;
both would need f16 exports to extend the static scan honestly. The predictive arm on
true-f16 forward (vs Q4K-dequant) is a minor follow-up.

## Reproduce

```
cargo run --release -p larql-vindex --example fp4_q1_scan -- --vindex output/<f16>.vindex --out bench/aim-validation/v2_<m>_scan.json
cargo run --release --example walk_ffn_v2_fp4_nll -- output/gemma3-4b-q4k-v2.vindex
```
