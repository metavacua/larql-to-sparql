# FP4 Storage — Precision Policy Decision

**Status:** Decision doc, pre-implementation.
**Scope:** How to handle the `down` outlier tail when building the FP4
storage path in `larql-compute`. Decides the disk format, not the walk
kernel; the walk-kernel implementation follows.
**Target delivery:** A policy choice that unblocks step 2 of the shipping
plan without committing to a format the cross-model data can't yet
support.

---

## 1. What the data tells us

From Q1 (reference Gemma 3 4B, full gate + up + down):

| Projection | per-feature block @ R=16 | sub-feature tile (512 elems) @ R=16 |
| ---------- | ------------------------ | ----------------------------------- |
| gate       | 99.91%                   | 99.99%                              |
| up         | 99.93%                   | 99.99%                              |
| **down**   | **99.65%**               | **99.90%**                          |

Cross-model (gate projection only, 4 models spanning 330M–50B):

- Gate is ≥ 99.91% compliant at R=16 everywhere and 100% compliant on the
  smallest model at R=4.
- No non-Gemma 4B-scale unquantised `down` is available locally. Whether
  the 4B down tail is Gemma-3-4B-specific or a general scale/family
  property is **unknown** and cannot be cheaply determined without either
  extending the scanner to Q4_K or extracting a new model.

Design implication: build the storage format to be **correct** whether
the gap-to-unknown data turns out favourable or unfavourable. Don't
assume Gemma 3 4B down is the worst case; don't assume it is
representative.

## 2. The three options

All three options are MXFP4-style: FP4 values (E2M1) in 32-element
sub-blocks, one FP8 (E4M3) scale per sub-block, one FP8 block scale per
feature-level block. They differ only in what is stored as FP4 vs higher
precision.

All three options use **256-element FP8 blocks** (see §3 for the
measurement-backed derivation of this block size). Each FP4 block stores:

- 256 FP4 values = 128 bytes
- 8 FP8 sub-block scales (one per 32-element sub-block) = 8 bytes
- 1 FP8 block scale = 1 byte
- **Total: 137 B per 256 elements, 0.535 B/element**

Baseline for compression ratios is **F16** — the dtype Gemma 4 31B's
vindex already uses and the realistic production default. The 4B vindex's
f32 on-disk format is an extract-time artefact, not the delivered-to-users
format.

### Gate precision: source-dtype today, FP4 deferred

The three options below were originally drafted with `gate: FP4` —
symmetric with up. Q2 implementation surfaced a constraint not
anticipated in v1: **gate KNN requires a dense f32/f16 matrix** for
its batch matmul (`gate_scores_batch` / `gate_walk`), and no FP4-aware
gate-KNN path exists in the walk kernel today. Storing gate in FP4
produces a vindex where the KNN path either can't run (no f32 gate
file) or uses a redundant f32 copy on disk (FP4 gate file is dead
weight). Neither is desirable.

**What the implementation ships today, in all three options:** gate
stays at the source vindex's dtype (typically f32 or f16). Only up
and down carry the policy-specified FP4/FP8/F16 precision. The tables
below describe this "as-implemented" version. True `gate=FP4`
requires an FP4-aware gate KNN path (FP4 bytes → top-K feature
indices without a dense dequant), which is tracked as a follow-up to
exp 26 and is not on the default shipping path for the initial FP4
vindex rollout.

**Storage consequence.** Keeping gate at source dtype costs ~1.22 GB
per projection on a 4B F16 vindex vs hypothetical FP4 gate (0.44 GB
FP4 vs 1.66 GB F16). Each option's 4B numbers in the tables below
reflect the as-implemented gate-at-source reality; the bracketed
`[theoretical]` columns show what the original FP4-gate variant
would land if the KNN work eventually closes the gap.

### Option A — Uniform FP4 (gate=source, up=FP4, down=FP4)

- **As implemented** (gate kept at source dtype):
  - Per 4B feature (2560 elems): 5,120 B (f16 gate) + 1,370 B (FP4 up) + 1,370 B (FP4 down) = **7,860 B**, vs 15,360 B F16 baseline = **1.95×**.
  - Measured on the 4B fixture: gate stays hard-linked from source (3.32 GB f32 on the f16 fixture), up+down FP4 total 0.93 GB. Full FFN 4.25 GB vs 9.96 GB source f32.
- **[Theoretical, if FP4 gate ships]** Per 4B feature: 3 × 1,370 B = 4,110 B, vs 15,360 B F16 = **3.74×**. Blocked on FP4-aware gate KNN.
- **Numerical cost:** 0.05% of 4B down blocks violate R=16 at the 256-element block size. Surfaces as logit drift on prompts activating the 4–5 heaviest down features per layer (see `results/q1_gemma3_4b.json`). Q2 measured cos 0.9952, KL p95 0.316 on 51 prompts — notably worse than Option B's tail.
- **Correctness contract:** decision-level (see §7). Passes loose, one or two prompts off tight at 4B.
- **Risk profile:** if larger-scale down has a heavier tail, the deployed contract tightens on production prompts. No mitigation short of re-quantising.

### Option B — Mixed precision, FP8 down (gate=source, up=FP4, down=FP8)

Up stored in FP4; down in FP8 (E4M3, one FP8 block scale per
256-element block, no per-sub-block scales because E4M3's dynamic
range absorbs the distribution directly).

- **As implemented** (gate kept at source dtype):
  - Per 4B feature: 5,120 B (f16 gate) + 1,370 B (FP4 up) + 2,570 B (FP8 down) = **9,060 B**, vs 15,360 B F16 = **1.70×**.
  - Measured on the 4B fixture: gate stays at source (3.32 GB f32 on the f16 fixture), up 0.44 GB FP4, down 0.85 GB FP8. Full FFN 4.60 GB vs 9.96 GB source f32, **2.17× on the as-shipped vindex**.
- **[Theoretical, if FP4 gate ships]** Per 4B feature: 1,370 + 1,370 + 2,570 = **5,310 B, 2.89×**. The originally-advertised "Option B = 65% savings" number.
- **Delta from Option A (as-implemented):** +1,200 B per feature on down. On 4B FFN ~420 MB; on 31B ~3.3 GB. The split between A and B is independent of the gate-FP4-vs-source question: both options keep gate the same today.
- **Numerical cost:** FP8 E4M3 has ~3-bit mantissa precision across a ±448 range. Does not require any max/min-scale-ratio assumption; absorbs the observed down tail without tension. Q2 measured cos 0.9979, KL p95 0.089 on 51 prompts — **3.5× tighter tail** than Option A.
- **Correctness contract:** decision-level against F16. Passes loose contract cleanly at 4B; meets 3 of 4 tight thresholds (KL mean + argmax are the remaining gaps). See §7.
- **Risk profile:** flat w.r.t. the cross-model down gap. FP8 E4M3 tolerates the observed 4B down tail and any plausible larger-scale tail.

### Option C — Mixed precision, F16 down (gate=source, up=FP4, down=F16)

Up stored in FP4; down bit-identical to the source f16.

- **As implemented:**
  - Per 4B feature: 5,120 B (f16 gate) + 1,370 B (FP4 up) + 5,120 B (F16 down) = **11,610 B, 1.32×** vs F16 baseline.
- **[Theoretical, if FP4 gate ships]** 1,370 + 1,370 + 5,120 = **7,860 B, 1.95×**.
- **Numerical cost:** zero on down (bit-identical). Same as Option B for gate/up.
- **Correctness contract:** strictly tighter than Option B on the down contribution.
- **Risk profile:** none numerically. Costs ~40% of the storage win vs B (as-implemented deltas are similar).

## 3. Block-size as a second lever

Block size is decoupled from A/B/C and applies regardless. The scanner
was extended with a `--tile-sub-blocks` flag and re-run at multiple block
sizes on Gemma 3 4B. The data:

| block_elements | 4B down @R=16 | 4B down max | 31B gate @R=16 | Divides 31B (5376)? | Compression vs F16 |
| -------------- | ------------- | ----------- | -------------- | ------------------- | ------------------ |
| 128            | 99.97%        | 138         | —              | ✓ (42)              | 3.70×              |
| **256**        | **99.95%**    | **161**     | **99.9996%**   | ✓ (21)              | **3.74×**          |
| 512            | 99.90%        | 161         | —              | **✗ (10.5)**        | 3.75×              |
| 1024           | 99.82%        | 194         | —              | ✗ (5.25)            | 3.76×              |
| 2560 (full)    | 99.65%        | 194         | N/A            | ✗                   | 3.76×              |

**Decision: 256-element blocks.** Two reasons:

1. **Universality.** Gemma 4 31B has hidden=5376, which is not divisible
   by 512 or 1024. 256 is the largest block size that divides every model
   scanned so far (4B=2560, 31B=5376, E2B=1536, v10c=512). A format that
   doesn't work on 31B is a non-starter.
2. **Tighter compliance at essentially no storage cost.** 256-element
   blocks push 4B down compliance from 99.90% (at 512) to 99.95% (at
   256) — 2× fewer violating blocks — at a 0.01 percentage-point
   storage regression (3.75× → 3.74×, ~5 bytes per 2,560-element feature).

128-element blocks give a further small compliance gain (down @R=16:
99.95% → 99.97%) at a 1% storage penalty (3.74× → 3.70×). Not worth the
extra overhead and format complexity; 256 is the sweet spot on the
Pareto curve.

The earlier draft's "512-element tile" recommendation was DeepSeek
precedent, not measurement. The measurement-grounded choice is 256.

## 4. Storage comparison, with 256-element blocks

Values are F16-baseline ratios (F16 is the production dtype on Gemma 4
31B's vindex). 4B reference; larger models proportional.

| Option           | bytes/2560 elem feature × 3 projections | compression | down safety on 4B | cross-model down risk |
| ---------------- | ---------------------------------------:| -----------:| ----------------- | --------------------- |
| Baseline F16     | 15,360                                  | 1.00×       | N/A (exact)       | N/A                   |
| A: uniform FP4   | 4,110                                   | **3.74×**   | 99.95% @ R=16     | unknown (could bite)  |
| **B: FP8 down**  | 5,310                                   | **2.89×**   | flat (E4M3 absorbs) | flat                |
| C: F16 down      | 7,860                                   | **1.95×**   | bit-identical     | flat                  |

Absolute storage on full 4B FFN vindex (3 projections × 34 layers ×
10,240 features × 2,560 elements):

| Option       | 4B FFN storage | saved vs F16 | delta vs A |
| ------------ | --------------:| ------------:| ----------:|
| F16 baseline | 5.36 GB        | —            | —          |
| A            | 1.43 GB        | 3.93 GB      | —          |
| B            | 1.85 GB        | 3.51 GB      | +420 MB    |
| C            | 2.74 GB        | 2.62 GB      | +1.31 GB   |

Absolute storage on full 31B FFN vindex (3 × 60 × 21,504 × 5,376):

| Option       | 31B FFN storage | saved vs F16 | delta vs A |
| ------------ | ---------------:| ------------:| ----------:|
| F16 baseline | 41.6 GB         | —            | —          |
| A            | 11.1 GB         | 30.5 GB      | —          |
| B            | 14.4 GB         | 27.2 GB      | +3.3 GB    |
| C            | 21.2 GB         | 20.4 GB      | +10.1 GB   |

Option B costs ~8% of the FFN vindex on 31B relative to Option A. Real,
not a rounding error; the "barely worse than A" framing from the earlier
draft was based on incorrect arithmetic and does not hold.

## 5. The decision

**Recommended default: Option B (FP8 down).** Confirmed by Q2
measurement on Gemma 3 4B, 51 prompts: Option B produces a 3.5×
tighter KL tail than Option A (p95 0.089 vs 0.316) at an ~8% FFN
storage delta. See `results/REPORT_Q2.md` for the ablation.

### Pre-committed triggers for a default change

The following 31B measurement outcomes would reopen the default:

- **All metrics tighten with scale** → tight contract becomes
  shippable; update §7 thresholds to reflect the measured floor and
  promote the stricter gate. Option B remains default.
- **Metrics stay flat** (cos ≥ 0.99 mean, KL p95 ≤ 0.30 at 31B) →
  4B contract is the production bar. Option B remains default.
- **Metrics loosen** (cos < 0.99 mean **or** KL p95 > 0.30 at 31B) →
  format needs adjustment. Options:
    (a) drop block_elements from 256 to 128 — measured to tighten
        compliance at 0.04 pp storage cost;
    (b) mixed-block-size per layer, with worst-offending layers using
        128-element blocks while the rest stay at 256;
    (c) promote Option C (F16 down) if the failure is concentrated
        on down.
  Choice driven by which component is the primary diverger, not
  declared a priori.

These are the concrete triggers, not "may revert" hand-waves. If 31B
comes back inside the cos/KL p95 gates, we ship. If it comes back
outside, we know what lever to pull.

Rationale for B as default:

1. **The storage cost of B over A is real but small** (~420 MB on 4B,
   ~3.3 GB on 31B; about 8% of A's FFN storage allocation). The "not
   worse than A" claim in the earlier draft was wrong — §4 has the
   corrected math. Option B still delivers ~65% FFN-storage savings
   against F16; A delivers ~73%.
2. **Numerically B is substantially safer on down.** FP8 E4M3 absorbs
   the observed 4B down distribution without per-sub-block-scale-ratio
   tension. The 0.05% violation rate (at the 256-element block size)
   disappears.
3. **B is robust to the cross-model down gap.** If 31B down turns out
   worse than 4B, Option A's contract tightens; Option B's does not.
   The unknown-cost of the cross-model down data becomes irrelevant for
   B, not merely "small" as under A.
4. **B preserves a cleaner correctness story.** With FP8 down, gate/up
   take the storage win in FP4 and the distributional property does the
   work; down stays in a precision that requires no distributional
   assumption. Q2 will measure end-to-end logit divergence; the format
   should be constructed so that result is interpretable independently
   of down-tail distributional luck.

**Configurability (not the default, but a knob):**

The vindex format carries per-projection precision tags. Legal values:
`{FP4, FP8, F16, F32}`. The extractor defaults to `{gate: FP4, up: FP4,
down: FP8}`. Users who want the uniform FP4 path can set `down: FP4`
explicitly; users who want paranoid correctness can set `down: F16`. The
walk kernel dispatches on the tag. No code path is removed; the default
is the safe one.

**Non-recommendation: Option A by default.** The asymmetry in 4B is
observed, the cross-model down data is unavailable, and the FP8 skip-cost
for down is negligible. Defaulting to A saves a rounding-error's worth of
storage at the cost of committing to a correctness story that depends on
a distributional assumption we cannot currently verify at scale. Not
worth it.

**Non-recommendation: Option C by default.** 40% worse storage than B to
buy precision that FP8 already provides. Only preferable if FP8 down
turns out (per Q2) to introduce noticeable logit drift in end-to-end
testing, which is not the current expectation.

## 6. What this implies for the extraction pipeline

1. The vindex format adds a manifest entry per projection: `{precision:
   "fp4"|"fp8"|"f16"|"f32", block_elements: 512, sub_block_elements: 32}`.
2. The extractor runs the Q1 scan as a gate. Before committing a new
   format, log per-projection compliance. If any projection falls below
   a configurable floor (default: 99% at R=16 per-feature block), the
   extractor refuses to write FP4 for that projection and downgrades it
   to FP8. The default policy (gate/up FP4, down FP8) is the floor,
   applied uniformly; the scan acts as a safety net for future models.
3. The extractor emits an `fp4_compliance.json` sidecar with the Q1
   scan output for the produced vindex. Users can inspect this to decide
   whether to override the default.
4. Q1's scanner `crates/larql-vindex/examples/fp4_q1_scan.rs` gets
   promoted from experiment binary to a library entry in
   `larql-vindex::quant` or equivalent, called from the extractor.

## 7. What this implies for the correctness contract

- `MarkovResidualEngine` retains its bit-exact contract against
  Standard KV. Unchanged.
- `FP4MarkovResidualEngine` (new) has a two-tier decision-level
  contract against the F16 `MarkovResidualEngine`. The split
  separates **format fidelity** (what quantisation did to the
  distribution) from **user-visible behaviour** (argmax). Those are
  different questions: logit cosine and KL measure the format;
  argmax measures a downstream property dominated by the model's
  own calibration. Mixing them in one contract conflates them.

  | Metric                  | Loose (exploratory)  | Tight (production) |
  | ----------------------- | -------------------- | ------------------ |
  | **Logit cos mean**      | **≥ 0.99**           | **≥ 0.998**        |
  | **Symmetric KL p95**    | **≤ 0.30**           | **≤ 0.10**         |
  | Top-5 Jaccard mean      | ≥ 0.70               | ≥ 0.85             |
  | Symmetric KL mean       | ≤ 0.10               | ≤ 0.02             |
  | Argmax agreement        | report only          | ≥ 95%              |

  Bold rows are the format-fidelity gates. **Argmax is tracked but not
  gated at the loose level** — it surfaces user-visible token flips but
  doesn't reliably measure quantisation quality, because argmax-ties
  get reshuffled by small numerical perturbations regardless of
  whether the perturbation represents a real loss of fidelity. At the
  tight level both format-fidelity and user-visible behaviour are
  gated.

  **This argmax-as-report-only split is measurement-derived, not
  ideological.** The Q2 ablation's failure-mode analysis (3 shared
  misses between Options A and B, all argmax-ties at logit cos ≥
  0.994) is what justified separating "is the format good?" from
  "does the model give consistent answers?" Without that data,
  gating on argmax at the loose level would have been the obvious
  default.

- Thresholds calibrated against Q2 measurements on Gemma 3 4B (51
  prompts). Option B passes the loose contract cleanly and meets 3 of
  4 tight thresholds; KL mean and argmax are the remaining distance
  to tight. See `results/REPORT_Q2.md` §"Revised decision-level
  contract thresholds" for the full data.

- **Scale behaviour is an open empirical question.** Whether Option B
  hits "tight" at 31B / 70B is untested and could go either way:
  independent quantisation noise would average down with more
  parameters, but correlated noise (same training distribution,
  outlier features, numerical conditioning) would concentrate rather
  than disperse. Not predicted by any mechanism we can verify pre-hoc.
  Measured when the 31B FP4 vindex exists.

## 8. Non-goals of this spec

- **Walk kernel implementation details.** This spec picks a storage
  format. The walk kernel reads it; how it reads it is a separate
  implementation spec.
- **Dequant hardware path.** M3 Max has no FP4/FP8 hardware; the walk
  kernel dequantises in software. Whether the dequant is fused into the
  saxpy inner loop, precomputed per layer, or lazy-cached is an
  optimisation decision that follows functionality.
- **Other quantisation schemes.** Q4_K, Q6_K, BF16 variants remain in
  the vindex format as-is. FP4 is a new opt-in mode next to them, not a
  replacement.
- **Cross-format interoperability.** An FP4 vindex does not need to be
  readable by the F16 walk path, and vice versa. Keep the read paths
  separate; the vindex manifest tag determines dispatch.
- **L0 token-indexed fast-path (exp 27).** The Gemma 3 4B L0 hash-routing
  result enables a storage approach that is independent of FP4 block
  quantisation — it compresses the *index*, FP4 compresses the *values*.
  The two do not compose cleanly in their simplest forms and are better
  as separate opt-ins. This spec treats L0 features as uniform with
  every other layer.

## 9. Open questions this spec does not answer

1. **What is the measured logit KL of Option B on the real-model test
   suite?** Q2 answers this. If the answer is < 0.001 across the suite,
   Option B is unambiguously correct. If it is > 0.01 for a subset of
   prompts, the sub-feature tile block size (§3) may need to drop
   further.
2. **Does the 31B down tail confirm Option B's robustness claim?**
   Requires the Q4_K scanner extension or a larger unquantised down
   extract. *Not blocking* — Option B's robustness is precisely the
   reason this question can stay open. A confirms-on-favourable / bites-
   on-unfavourable is exactly the risk profile B is chosen to sidestep.
   The cross-model scan is useful *context* for the writeup, not input to
   the build.
3. **Should block_elements become layer-configurable?** If later
   measurement shows L33 down has a pathological tail on some models,
   the extractor could fall back to 256-element tiles on specific
   (layer, projection) pairs. Not worth building until there is evidence.

## 10. Minimal next action if B is accepted

1. Fix `block_elements = 256`, `sub_block_elements = 32`,
   `sub_block_scale_dtype = FP8`, `block_scale_dtype = FP8`.
2. Add the precision manifest to the vindex format.
3. Build the FP4 writer, the FP8 writer, and the dequant reader in
   `larql-compute::quantisation`. Library API first, walk-kernel hookup
   second.
4. Extend the extractor to produce `{gate: FP4, up: FP4, down: FP8}`
   output with the Q1 scan gate and the `fp4_compliance.json` sidecar.
5. Wire the walk kernel's per-projection dispatch to read the manifest
   tag.
6. Run Q2 — the existing real-model suite against the new path. Report.

## 11. Artefacts this spec depends on

- `results.md` — top-level Q1 consolidated writeup.
- `results/q1_gemma3_4b.json` — the 99.65% down number and the worst-
  offenders list that motivate Option B.
- `results/REPORT_CROSS_MODEL.md` — the "gate generalises, down gap
  unknown" claim that motivates defaulting defensively.
