# SENSITIVITY-1B′ — absolute activation-weighted consequence (pre-registered)

Written before the disjoint calibration set is captured or any score
computed. One shot; no revision afterwards.

## Why this rung rather than 1C

1B-a did not fail for lack of information. It failed because dividing by
`‖XW‖²` reintroduced 1A's bias — both rewarded operands for being small.
Escalating to curvature now would skip the cheaper hypothesis the failure
itself motivated: that activation weighting is sufficient once the
normalisation stops cancelling it.

## What 1B′ is, and what it is not

1B′ is a **within-model ranking proxy**. It is not a calibrated predictor
of absolute Q-BANK benefit, and this rung does not ask it to become one.

That distinction is load-bearing. An unnormalised activation consequence
can legitimately carry very different absolute scales between two models,
so a bar phrased as "Glimmer's predicted return must be under 10% of
Granite's" would test cross-model calibration — a capability 1B′ was never
designed to have — and would fail a proxy that ranks perfectly within each
model. Ranking and calibration are two instruments:

```text
ranking predictor          (this rung)
        +
benefit / fidelity calibration   (not yet built)
        ↓
"worth spending bytes?"    (the STOP decision, later)
```

Only the first is on trial here. See "The STOP decision" at the end for
what would have to be added before `REPRESENT` could decline to spend
bytes on its own.

## Primary score — pre-registered, one form only

For tensor `W` with per-feature input second moments `d_j = E[x_j²]` from
the **disjoint** calibration set, and `ΔW = W − Q(W)`:

```text
consequence(W)  = Σ_j  d_j · ‖ΔW[:,j]‖²

score(candidate) = Σ_{W ∈ candidate} consequence(W)
                   ─────────────────────────────────
                        extra physical MiB protected
```

- No division by `‖XW‖²`.
- No division by any model-level total, or any other alternative
  normalisation.
- No coefficient fitting.

Division by extra MiB is the return metric, not a magnitude normalisation:
it is the axis the Q-BANK verdicts are already expressed on. The
prohibition is on dividing by anything that scales with the operand's own
size, which is what sank both 1A and 1B-a.

No other variant is computed. There is nothing to choose between
afterwards.

## Candidate pool — frozen at eight

The capture taps three sites per layer (attention input, FFN input, FFN
output). There is **no attention-output site**, so `o_proj` has no moments
and cannot be scored.

Every region containing `o_proj` is therefore excluded:

```text
POOL (8)        late5-ffn, late10-ffn, late15-ffn, late10-ffn-v,
                ffn-protected, v-protected, k-protected, down-protected

EXCLUDED (5)    attn-protected, o-protected, late10-ffn-o,
                early10, late10          — all contain o_proj
```

Excluded rather than scored-without-`o_proj`, because the alternative is
silently biased: the region's consequence would omit `o_proj` while its
`extra MiB` denominator still charges `o_proj`'s bytes, deflating exactly
the five regions that contain it.

This is the **same pool 1B-a used**, which is the point. 1B′ then differs
from 1B-a in exactly one variable — the normalisation — so a pass isolates
the normalisation as the cause. Adding an `o_proj` tap would change
coverage and normalisation together, and a pass would no longer say which
one mattered.

Every bar condition remains evaluable: `late5/10/15-ffn` and `v/k/down` are
all `o_proj`-free.

## Controls, before any score

All three must pass before the Granite score is computed. A failure here is
a harness failure, not a result.

**Region congruence.** The screen's regions (`score_sensitivity.py`
`CANDIDATES`) and the Q-BANK arms' protections (`vindex3 represent
--protect`) are defined separately and hand-synced; nothing has ever
checked they agree. For each of the eight, the tensor set the screen sums
over must match the tensor set the corresponding Q-BANK arm protected,
cross-checked against that arm's banked `payload_bytes` delta. If they
disagree, the screen and the truth are ranking *different objects* and
neither a pass nor a fail would mean what it says.

**Reconstruction control.** `down_proj`'s input is reconstructed offline as
`act(gate(x)) ⊙ up(x)` rather than tapped. The executor's FFN output is
tapped, the screen recomputes it, and the two are compared. A
mathematically equivalent reconstruction can still be numerically
different, and it would silently corrupt exactly one of the three frozen
negatives.

**Observed-vs-unobserved parity.** Tapping must not perturb execution: an
observed step is bit-identical to an unobserved one.

## Calibration — disjoint, frozen before capture

`calibration-disjoint.json`: 12 prompts written for this test and **absent
from Q-BANK-1**. Disjointness is *verified* by exact-content check against
`prompts.json`, not asserted in a `note` field and trusted.

Frozen into the file before capture:

- the prompt ids,
- a SHA-256 digest over the normalised prompt text,
- a SHA-256 digest over the **token ids actually fed to the model**,
- the model + container digest the moments were captured from,
- the tokeniser identity.

Both digests are required because they cover different objects. The CLI
consumes JSONL of `{"id", "ids": [u32]}`, while the calibration file holds
`{"id", "category", "text"}` — a tokenisation step sits between them and is
currently untracked. A digest over text alone would not detect a changed
tokeniser, a changed BOS convention, or a truncation, any of which would
change `d_j` while leaving the provenance record looking intact.

**The 1B-a capture is discarded, not reused.** This is a live hazard, not a
formality: `granite-4.1-3b-sensitivity-1b.json` already carries a `num`
field holding `Σ_j d_j ‖ΔW[:,j]‖²` — numerically the 1B′ numerator, but
computed from **bank-derived** activations. Scoring 1B′ from that file
would reproduce the observation that motivated this rung, on the very data
that motivated it, and would make the ladder decorative. 1B′ reads only
moments captured from `calibration-disjoint.json`; the scorer must refuse
the 1B-a artefact as an input rather than rely on nobody pointing it there.

Granite is scored **exactly once** against these moments.

## Containers — frozen, and verified unconfounded

Moments are captured from BF16 canonical, and the banked verdicts must come
from the same object:

```text
Granite  ~/chris-models/granite-4.1-3b.vindex3            BF16 canonical
         model               c0650403e44e78ec0262dab1c90914c65b196c4e
         BF16 payload_sha256 374562b3ff81c81fb72f1fcd4c842912ef75e167f8c0e9568997965a15f2612c
         360 tensors, 6,291,865,600 bytes
         head.output_multiplier 0.1

Glimmer  ~/chris-models/muse-glimmer-30b.vindex3          BF16 canonical
         head.output_multiplier 0.196116…
```

Pinned by **digest**, not by path — the Granite container was renamed on
2026-08-24 (see its `PROVENANCE.md`) and the digest is what survives that.

`granite-4.1-3b-deploy.vindex3` is the derived R0 image the sweep's
`R0-recheck` arm was measured on; its `source_representation_digest`
resolves to the digest above, so the reference and the R0 arm are the same
weights under the same head scale.

**Head-scale confound: checked, absent.** A pre-fix Granite encode carrying
`output_multiplier: 10.0` — the `logits_scaling` defect, a factor of 100 —
existed when these verdicts were banked. It was checked rather than
assumed, because a reference on `0.1` compared against an R0 arm on `10.0`
would have made every banked Granite verdict a measurement of the head
scale:

| container | `head.output_multiplier` |
|---|---|
| `granite-4.1-3b.vindex3` (canonical) | **0.1** — correct |
| `granite-4.1-3b-deploy.vindex3` (R0 arm) | **0.1** — correct |
| the pre-fix encode *(since deleted)* | **10.0** — the defect |

Both Granite arms are on `0.1` and Glimmer's containers all agree at
`0.196116…`, so neither model's verdicts are confounded.

The pre-fix container was verified end to end against the canonical one —
same 4 tokens, identical argmax `89`, logits `+1451.1790` against
`+14.5118`, a ratio of exactly `100.0000` — and then **deleted on
2026-08-24**, once the defect was pinned in-repo and did not need 6.3 GiB
of disk to reproduce:

```text
detect::tests::declared_scalars::granite_spellings_resolve_through_the_canonical_names
    logits_scaling 10.0  ->  logit_scale() == Some(0.1)          [on main]

inventory::tests::resolved::granite_resolves_its_divisor_head_scale_to_a_multiplier
    config in  ->  ResolvedExecution.output_multiplier == 0.1    [this branch]
```

Both falsification-verified: reverting `logit_scale()` to pass the divisor
through fails the second with `must carry 1/10, got 10`.

The head multiplier sits downstream of every tapped site, so it cannot
affect `d_j` — but the container is pinned by digest anyway rather than
relying on that argument.

## The shape statistic

Both models' bars use one dimensionless, within-model quantity — the
collapse in marginal return between the first and second five protected
FFN layers:

```text
        marginal(late5)      [p99(R0) − p99(late5)]   /  ΔMiB(R0→late5)
ρ   =   ───────────────  =   ────────────────────────────────────────────
        marginal(late10)     [p99(late5) − p99(late10)] / ΔMiB(late5→late10)
```

`ρ > 1` means the first five layers buy more per byte than the next five —
a peak exists. `ρ < 1` means no peak. **The boundary is 1, which is
structural rather than fitted**: it is the definition of "is there a peak",
not a number chosen to fit either model.

ρ is a ratio of ratios, so it is invariant to any constant rescaling of the
score. That is why it can be compared across two models without asserting
anything about their absolute scales — it is a *shape* comparison, which is
the only cross-model claim this rung makes.

Banked Q-BANK truth, for reference — not a target to reproduce numerically,
only a sign:

| model | marginal(late5) | marginal(late10) | ρ | shape |
|---|---|---|---|---|
| Granite | +7.9305 p99/GiB | +0.0470 | **168.85** | sharp peak |
| Glimmer | +0.0016 p99/GiB | +0.0078 | **0.208** | no peak; return *rises* |

## The bar — all conditions, no rank correlation rescue

Spearman may be reported. **It cannot rescue a failed condition.**

**Granite — must find the known structure**

1. `late5-ffn` ranks above all three negatives on `score`.
2. `v-protected`, `k-protected`, `down-protected` all fall in the bottom
   half of the pool — with eight candidates, all three at rank 5 or worse.
   (1B-a placed them 1, 2 and 7 of 8.)
3. The knee survives: `score(late5-ffn) > score(late10-ffn)` and
   `> score(late15-ffn)`, and `ρ_1B′(Granite) > 1`.

**Glimmer — shape negative**

4. 1B′ must **not manufacture** Granite's peak on a model that has none:
   `ρ_1B′(Glimmer) < 1`.
5. 1B′ must not claim strong late-FFN concentration on Glimmer:
   `late5-ffn` is not its top-ranked candidate by `score`.

Condition 4 is the one that matters. A predictor that learns "late FFN
matters" from Granite and reproduces that shape on Glimmer has learned an
architecture prior, not a model-specific sensitivity signal — however good
its Granite ranking looks. Glimmer's truth does not merely lack Granite's
peak; its marginal return *increases* across that range, so the two models
sit on opposite sides of the boundary.

Condition 5 is deliberately weak. Glimmer's surface is flat, so its
*ordering* is close to noise, and requiring a proxy to reproduce a specific
flat ordering would be requiring it to reproduce noise. Only the claim of
concentration is tested, not the ranking.

**What is NOT required at this rung**

- That 1B′ predict Glimmer's absolute return is small.
- That 1B′ conclude "R0 is good enough, emit R0".
- Any cross-model comparison of raw score magnitudes.

Failing any of those is not a 1B′ failure, because 1B′ does not estimate
them.

## If 1B′ passes

It may propose **one** unseen Granite precision map, which Q-BANK then
judges once. The screen proposes; Q-BANK promotes. Nothing else changes.

That step is the real transition. Every sensitivity method so far has been
judged retrospectively against known candidates; a map that was never in
the frozen fifteen, whose Q-BANK outcome lands where 1B′ predicted, is the
first evidence of a genuine **precision-map search primitive**.

## If 1B′ fails

1C is then earned, and the conclusion is stronger than "try something
fancier": absolute local consequence is insufficient, so downstream
curvature or context is required.

## The STOP decision — explicitly out of scope here

Glimmer's flat surface means an automatic compiler should eventually be
able to conclude:

```text
R0 already on the Pareto knee
additional protection not worth its bytes
emit R0
```

1B′ cannot support that conclusion and is not being asked to. Doing so
needs a second instrument — a calibration from predicted consequence to
expected Q-BANK fidelity — which is a separate rung with its own
pre-registration and its own falsifier. Recording it here so that a good
ranking proxy is not rejected for failing to predict something it was
never built to estimate.

## Sequence

1. Pre-register (this document), including the eight-candidate pool.
2. Freeze disjoint calibration prompts + both provenance digests; verify
   disjointness against `prompts.json` by content.
3. Capture Granite activation moments from the disjoint set.
4. Run the three controls: region congruence, reconstruction, observed-vs-
   unobserved parity.
5. Score Granite once against the frozen verdicts.
6. If Granite passes, capture the same moments on Glimmer.
7. Check conditions 4 and 5.
8. Only after both pass, let 1B′ propose one unseen Granite map.
9. Q-BANK that map once.

Steps 5 and 9 each happen exactly once. There is no second scoring pass at
either.

---

# Result: 1B′ FAILS on Granite — condition 2, and only condition 2

Scored once, on the disjoint bank, against the frozen verdicts.

Controls first, all three pass:

```
region congruence        all 8 regions select exactly their banked arm's tensors, 0 B delta
provenance               calibration df0e3644…, container c0650403…, every payload_sha256
reconstruction control   layer 0, rel 1.762e-06 vs 1e-05 tolerance   PASS
ffn activation           Silu, read from the plan the executor runs
pool completeness        240 tensors, 40 o_proj absent (no site)
```

```
rank candidate             +MiB     1B′ score  Q-BANK p99/MiB
   1 late5-ffn            431.2       262.764        0.007745
   2 down-protected      1150.0       142.545       -0.000155  (negative)
   3 late10-ffn           862.5       133.086        0.003895
   4 late10-ffn-v         934.4       124.163        0.003720
   5 late15-ffn          1293.7        89.672        0.002607
   6 ffn-protected       3450.0        50.270        0.001147
   7 k-protected           71.9        27.784       -0.000343  (negative)
   8 v-protected           71.9        17.079        0.000272  (negative)

  1. late5-ffn above all three negatives : PASS   (rank 1)
  2. v/k/down all in the bottom half     : FAIL   (down-protected rank 2 of 8)
  3. knee survives, rho > 1              : PASS   (rho_1B' 77.08, truth 168.85)

  Spearman +0.595   (1A was -0.313, 1B-a was -0.524)

=> FAIL
```

## The normalisation hypothesis was right, and not sufficient

1B-a failed because dividing by `‖XW‖²` rewarded operands for being small,
and it put `v` and `k` **first and second of eight**. Removing that
division moves them to **seventh and eighth** — the bottom, as predicted,
and the correlation flips sign from −0.524 to +0.595. The diagnosis was
correct: absolute consequence is the right form, and it recovers the knee
(ρ 77 against a truth of 169, both far above the boundary).

It still fails, on one operand.

## What actually fails: `down_proj`

`down_proj` carries by far the largest local consequence of any projection —
mean 4,098 against 106 for `gate_proj`, 131 for `up_proj`, 198 for
`q_proj`. Q-BANK says protecting it is worth **less than nothing**: p95 got
*worse*, at a cost of 1,150 MiB.

So this is not the screen being noisy. It is the screen being confidently
wrong about one operand, for what looks like a structural reason:

> Local activation-weighted consequence measures where quantisation error
> **is large**. It does not measure whether the model's output is
> **sensitive** to error there.

`down_proj` writes into the residual stream, where its error is added to a
large existing signal; `gate_proj` and `up_proj` errors pass through the
nonlinearity that forms the intermediate. A purely local score cannot tell
those apart, because the difference is in what happens downstream — which
is precisely what a curvature-aware score models.

That is a sharper argument for 1C than "try something fancier", and it was
earned rather than assumed.

## The bank-derived observation was optimistic, as suspected

1B-a's note recorded that the unnormalised variant, computed on
**bank-derived** activations, ranked `down-protected` fourth of eight with
"one misplacement". On the disjoint bank it ranks **second**. The variant
looked better on the data that suggested it, which is exactly why the
pre-registration refused to promote it there and required this run.

Had it been promoted on that evidence, the ladder would have recorded a
pass it had not earned.

## Not done

Glimmer moments were **not** captured. The sequence gates step 6 on a
Granite pass, and tuning this rung to reach Glimmer would defeat the point.
Conditions 4 and 5 remain untested.

The per-tensor consequence records are banked at
`granite-4.1-3b-consequence-1b-prime.json` (240 tensors, with weight,
moment and calibration digests). 1C can compare weight-only, activation
consequence and curvature on exactly this tensor population without
recapturing anything.

## Artifacts

```text
banked      granite-4.1-3b-consequence-1b-prime.json   127 KB, 240 tensors
            calibration-disjoint.json                  prompts + both digests
            calibration-disjoint.tokens.jsonl          the frozen token bank

not banked  granite-4.1-3b-moments-1b-prime.json       114 MB
```

The capture is not committed. It is ~114 MB, dominated by the strided
`ffn_samples` the `down_proj` reconstruction consumes, and it is
reproducible from the pinned container plus the frozen token bank. Every
consequence record carries its digest, so a later rung can prove which
capture produced a number without the bytes:

```text
moment_artifact_digest    5a6064b1b57ea22fd0437848b841a0667be33afb3891cbe0117a74ce3bf76182
calibration_token_digest  df0e3644aba068c4687baefe178399d1f0a62cb7509377211499927a55da96c3
```

Regenerate with:

```sh
larql vindex3 sensitivity ~/chris-models/granite-4.1-3b.vindex3 \
  --output /tmp/unused-1a.json \
  --calibration bench/prompts/quality-bank-1/calibration-disjoint.tokens.jsonl \
  --moments bench/prompts/quality-bank-1/granite-4.1-3b-moments-1b-prime.json
```
