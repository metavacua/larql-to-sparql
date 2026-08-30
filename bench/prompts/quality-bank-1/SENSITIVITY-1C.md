# SENSITIVITY-1C — downstream sensitivity (pre-registered)

Written before any curvature, Jacobian or sensitivity quantity is computed.
One shot; no revision afterwards.

## Question

> Can a downstream-sensitivity signal distinguish the same operand being
> **important in late layers** from that operand being **poor value when
> protected globally**?

Deliberately narrower than "does curvature improve the correlation". A
better correlation is not the finding this rung exists to produce.

## Why 1C is earned

Three rungs, three increasingly specific failures:

```text
1A   weight geometry                 no semantic signal at all
1B-a normalised activation           the normalisation destroys the signal
1B′  absolute activation             finds late-FFN, confuses LARGE error
                                     with IMPORTANT error
```

1B′ recovered the knee (ρ 77 against a truth of 169) and pushed `v`/`k` to
7th and 8th of eight. It failed on one operand: `down-protected` ranked
**2nd of 8**, while Q-BANK says protecting it is worth less than nothing —
p95 got *worse*, for 1,150 MiB.

`down_proj` carries the largest local consequence of any projection (mean
4,098 against 106 for `gate_proj`). So the gap 1C must close is exactly:

> Local consequence measures where quantisation error **is large**. It does
> not measure whether the output is **sensitive** to error there.

## Primary discriminator — `down_proj` late vs early

```text
        mean( score(down_proj, layers 35-39) )
D   =   ──────────────────────────────────────
        mean( score(down_proj, layers 0-9)   )
```

Dimensionless, within-model, invariant to any constant rescaling of the
score — so it survives whatever units 1C's implementation ends up in.

**`D > 1` is a floor, not the test**, and the banked 1B′ records show why in
two separate ways.

First, **1B′ already has D = 4.1213** on `down_proj` and still failed.
Reproducing a depth slope is not the missing capability; 1B′ has one. What
1B′ lacks is that the slope never changes its *conclusion* about the
candidate.

Second — and this is the sharper warning — `D` measured across projections
does not track importance at all:

```text
1B′ depth ratio, from the banked 240 records

  v_proj     D = 9.9119     <-- LARGEST, and a frozen negative
  up_proj    D = 5.6289
  gate_proj  D = 5.2974
  down_proj  D = 4.1213     <-- the operand this rung is about
  q_proj     D = 2.0602
  k_proj     D = 1.1595     <-- smallest, also a frozen negative
```

`v_proj` has the steepest depth concentration of any projection, and Q-BANK
says protecting it moves KL from 0.278 to 0.264 — near-zero benefit. So a
score that rewarded high `D` would **promote a frozen negative**, and the
two negatives sit at opposite ends of the range.

`D` is therefore admissible only as a *within-`down_proj`* sanity floor. It
is not an importance signal, it must not be optimised toward, and no
condition below rewards a larger `D` for its own sake. This was checked
before any 1C quantity existed, precisely so it could not be discovered
afterwards as a convenient explanation.

```text
1B′ already knows      late down > early down          (D = 4.12)
1B′ still concludes    all down protection is valuable  (rank 2 of 8)

1C must learn          late down matters in context,
                       early/mid down mostly does not,
                       therefore GLOBAL down protection is poor value
```

**No fitted magnitude threshold.** A larger `D` requirement cannot be
honestly derived from Q-BANK either: the truth has arms for `late5-ffn`
(which bundles gate, up *and* down at layers 35–39) and for
`down-protected` (all 40 layers), but none isolating late-`down` alone. So
there is no truth ratio to calibrate against, and inventing one from 1C's
own output is the exact retrospective tuning this ladder exists to prevent.

The magnitude requirement is therefore expressed where it is actually
observable — in the candidate ranking, as the joint condition below. A `D`
large enough to flip `down-protected` into the bottom half *while leaving
`late5-ffn` first* is precisely a `D` that is large enough, and it needs no
threshold of its own.

## The decisive joint condition

```text
demote GLOBAL down protection      without demoting LATE5-FFN
```

These are not independent, which is the whole point: `late5-ffn` **contains**
`down_proj` at layers 35–39. A score that simply suppresses `down_proj`
everywhere satisfies the negative and destroys the positive at the same
time, and a bar reading them separately would score that as partial
progress when it is really one undifferentiated move.

Passing both simultaneously requires a depth-conditional judgement about a
single operand. Nothing cheaper will do it.

## Score semantics — fixed here; implementation is NOT

1B′ measured a local quantity:

```text
local consequence      ≈  || X ΔW ||²
```

1C measures the same perturbation weighted by what it does downstream:

```text
downstream consequence ≈  δhᵀ G δh
```

where `δh` is the perturbation quantising the operand injects at its output
site, and `G` expresses how strongly perturbations in that direction move
the model's output distribution.

**What is pre-registered is the semantics, not the estimator:**

> The score must weight the local quantisation perturbation by how strongly
> perturbations in that direction affect downstream model output.

`G` may end up an empirical Fisher, a Gauss–Newton / Jacobian-vector
product, a Hessian approximation, or a projected-logit sensitivity — chosen
for what the executor can measure *faithfully*, once that is known. Naming
"Hessian" now would lock the rung to a word rather than to the property,
and the cheapest faithful implementation may not be the one with the
familiar name.

Fixed regardless of estimator:

- The perturbation is the **actual** quantisation delta, not a proxy.
- No normalisation by the operand's own magnitude — the defect that sank
  both 1A and 1B-a.
- Aggregation stays `Σ tensor score / extra physical MiB`, unchanged from
  1B′, so a difference between the rungs is attributable to the score.
- No coefficient fitting.

Whatever estimator is chosen is written into this document **before** it is
run, with its cost and its approximation stated.

## Frozen — unchanged from 1B′

Pool of eight, `o_proj` excluded (no attention-output site; absent, never
zero or estimated):

```text
late5-ffn, late10-ffn, late15-ffn, late10-ffn-v,
ffn-protected, v-protected, k-protected, down-protected
```

Calibration and containers identical, so 1B′ and 1C differ in the score and
nothing else:

```text
calibration token digest  df0e3644aba068c4687baefe178399d1f0a62cb7509377211499927a55da96c3
Granite BF16 payload      374562b3ff81c81fb72f1fcd4c842912ef75e167f8c0e9568997965a15f2612c
model                     c0650403e44e78ec0262dab1c90914c65b196c4e
head.output_multiplier    0.1
```

The 240 banked 1B′ records carry these digests, so weight-only, activation
and downstream signals compare on the identical tensor population without
recapturing anything.

## Controls — before any score

The four from 1B′ carry over and must pass: **region congruence**,
**provenance refusal**, the **`down_proj` reconstruction gate** (1e-5
relative; 1B′ measured 1.762e-6), and **pool completeness**.

1C adds one, because it measures a directional quantity for the first time:

**Directional control.** A sensitivity metric is only meaningful if it
discriminates *direction*, not just magnitude. Scoring a control
perturbation of the same size in an uninformative direction must yield a
materially lower score than the true quantisation delta. If it does not,
`G` is behaving as a magnitude reweighting and 1C is 1B′ in different units.

**The control must be drawn in the metric's own geometry, not isotropically.**
A naive L2-matched random direction is a known trap here: in high dimension
it concentrates almost entirely in near-invariant directions, so it scores
low for a reason that has nothing to do with the metric being informative,
and the control passes while testing nothing. Draw it within the subspace
`G` actually weights — matched in the tangent space, not in raw L2 — and
state the construction before running it.

## The bar — all conditions, no aggregate rescue

**Primary**

1. `D > 1` on `down_proj`. A floor only — 1B′ already clears it at 4.1213,
   and `v_proj`'s D = 9.9119 shows a large `D` is not evidence of
   importance. Nothing below rewards a larger `D`.
2. **Joint:** `down-protected` falls in the bottom half of the pool **and**
   `late5-ffn` remains rank 1.

**Frozen negatives**

3. `v-protected`, `k-protected`, `down-protected` all rank 5th or worse
   of eight.

**Shape**

4. The knee survives: `score(late5-ffn) > score(late10-ffn) >`
   `score(late15-ffn)`, and `ρ_1C > 1`.

   Condition 4 guards against the cheapest wrong way to pass condition 2:
   flattening the depth structure until nothing stands out. A rung that
   fixed the negative by destroying the very concentration it set out to
   explain has not learned anything.

**Reported, never decisive**

5. Spearman against Q-BANK `p99/MiB`. Recorded for comparison with 1A
   (−0.313), 1B-a (−0.524) and 1B′ (+0.595). **It cannot rescue a failed
   condition**, and a higher correlation with the negatives still ranked
   highly is a failure, exactly as it was for 1A and 1B′.

## PASS

All of 1–4. Then, and only then, 1C may propose **one** unseen Granite
precision map, which Q-BANK judges once. The screen proposes; Q-BANK
promotes.

## FAIL

Any of 1–4. The result is recorded as a failure and the ladder stops for
re-derivation rather than continuing to a fourth estimator. Three rungs
have now failed in three specific ways; a fourth failure would say the
local-signal programme itself needs rethinking, not that a fifth formula is
owed.

Specifically: if 1C passes conditions 1, 3 and 4 but fails the joint
condition 2, the finding is that downstream weighting is **still**
insufficient to make a depth-conditional judgement about a single operand —
which is a sharper and more useful negative than any of the three so far.

## What this rung is the gate on

The representation machinery is done. VINDEX3 can express a precision
program, persist it, execute it with zero runtime manufacture, prove stored
and transient are identical, and validate a candidate against Q-BANK. What
it cannot do is **predict** which program is worth compiling without paying
Q-BANK for every candidate.

```text
canonical model
   ↓ cheap sensitivity model     <-- 1A, 1B-a, 1B′ all failed here
   ↓ candidate precision maps
   ↓ physical byte / runtime costing
   ↓ small Pareto shortlist
   ↓ Q-BANK validation
   ↓ deployment image
```

Everything below that first arrow exists. The arrow itself does not.

And it cannot be replaced by a rule. Granite's damage concentrates sharply
in the last five layers' FFN; Glimmer's surface is flat — the same
protection moves its p99 by 0.0046 for ~2.9 GiB, and its marginal return
*rises* rather than collapsing. So a compiler must not encode "late FFN is
special". It has to **discover whether a special region exists at all**,
per model. That is why Glimmer's role in this ladder is a shape negative
rather than a second confirmation.

If 1C passes, the manual mixed-precision workflow becomes an automatic
representation compiler. If it fails, the shortlist stays hand-derived and
the failure says something specific about what local signals cannot see.

## What 1C does NOT claim

- **Not an optimizer.** No budget solver, no search, no map proposal until
  the frozen bar passes.
- **Not a STOP predictor.** 1C remains a within-model ranking proxy. Saying
  "R0 is good enough, emit R0" needs a separate benefit/fidelity
  calibration with its own falsifier — see 1B′'s note on the two
  instruments.
- **Not general across models.** Glimmer is **not** captured until Granite
  passes. Its shape negative (ρ_Glimmer < 1) stays untested; running it
  early would turn a frozen sequence into a moving target.
- **Not a claim about curvature in general.** Whatever estimator is chosen
  is tested on one model, one format, one bank.

## Sequence

1. Pre-register (this document).
2. Establish what the executor can measure faithfully, and write the chosen
   estimator — with its cost and approximation — into this document.
3. Define and write down the directional control's construction.
4. Compute per-tensor downstream consequence on the frozen calibration set.
5. Run all five controls.
6. Score once, against the bar above.
7. If Granite passes, capture Glimmer and check ρ_1C(Glimmer) < 1.
8. Only after both, propose one unseen map. Q-BANK judges it once.

Steps 4, 6 and 8 happen exactly once each.

---

# Result: suffix replay FAILS feasibility, before any score exists

Step 2 of the sequence — "establish what the executor can measure
faithfully" — closed the intended primitive. **No sensitivity value was
computed.** This is a feasibility failure, not a scientific one, and the
bar above is untouched and unrun.

## What was checked

The candidate primitive was: capture the residual at layer `L`, re-enter
the suffix twice — once clean, once perturbed — and measure the downstream
effect. Three gates.

**Gate 1 — resume fidelity: PASSES.** `ResumePoint` + `execute_plan_streaming`
is public, documented bit-identical, and gated
(`resume_from_a_mid_run_plane_is_bit_identical`: "may not differ in a
single bit, or every parity claim made over a resumed dump would need an
asterisk"). No new executor surface would have been needed.

**Gate 2 — batched-direction equivalence: FAILS BY CONSTRUCTION.**

```rust
pub struct ResumePoint { pub hidden: Vec<Vec<f32>> }   // "one row per position"
pub struct AttentionCall<'a> { pub inputs: &'a [Vec<f32>], .. }
// "attention reads other positions' K/V but never writes them"
```

The executor's row axis **is** the causal position axis. There is no
batch-of-sequences dimension anywhere. Packing six perturbation directions
as six rows would let them attend to one another — they would not be six
independent suffix executions, and the batched result would not equal the
unbatched one. This is architectural, not a missing optimisation.

**Gate 3 — economic: fails, and not narrowly.**

Probing every layer costs **20.5x a full traversal** per prompt per arm,
because early layers carry long suffixes and overlapping suffixes are
re-executed.

```text
6 arms, all 40 layers                       80.8 Q-BANK candidate-equivalents
6 arms, 10 layers                           20.2
1 arm,  all 40 layers  (degenerate floor)   13.5
```

Even the floor — one direction, reference banked, which cannot answer the
six-direction question — is 4x over the ceiling.

Banking the reference statistic during the calibration traversal *does*
work (`LayerTrace` carries `post_attention` / `ffn_input` / `post_layer`
per position; a full run returns logits). It removes one arm of seven and
does not change the verdict.

## Why the estimator was not adjusted to fit

"Probe ten layers and interpolate the rest" would have brought the cost to
~20 candidate-equivalents. It was rejected, for a reason worth recording:
**it assumes depth behaves smoothly between probes, and Granite's own knee
is the counterexample.** The damage concentrates in five layers; marginal
return collapses 169-fold from the first five to the next five. A screen
that reconstructs unprobed depth by interpolation would import exactly the
smoothness assumption the compiler exists to avoid — and it would import it
into the one model already known to violate it.

Changing the estimator after the intended primitive failed is also how a
pre-registration rots. The rung closes instead.

## Economic note — recorded separately, and it does NOT change this verdict

The `<= 3 Q-BANK candidate-equivalents` ceiling was derived against the
wrong question. It answers:

> Is the screen cheaper than testing the eight Granite candidates directly?

At eight candidates **no screen can pay off**, whatever primitive it uses:
Q-BANK on all eight costs eight candidate-equivalents and returns ground
truth rather than an approximation. The screen's value is that scoring the
tensor field once makes any map a summation — the R2 and K3 case, where the
map space is combinatorial.

So a fixed 3x ceiling is not an appropriate general production criterion.
A future rung should derive its ceiling from the search space it unlocks.

**This note is banked as a separate fact and is deliberately not applied
retroactively.** Suffix replay fails gate 2 architecturally, and its
asymptotic shape — re-executing overlapping suffixes per layer per
direction — is wrong for K3 regardless of any ceiling.

## What the next rung must be

1B′ established the requirement:

```text
local perturbation  x  downstream sensitivity to that perturbation
```

Suffix replay obtains the second factor by re-running the suffix per
direction, which is `O(positions x layers x directions x mean suffix)`.
The scalable shape is the reverse:

```text
one reference traversal
      -> one downstream sensitivity field over layer boundaries
      -> score every real quantisation direction locally, by summation
```

roughly `O(positions x traversal)` plus cheap per-direction scoring, and —
critically — **independent of the number of candidate precision maps**.
That is the property SENSITIVITY wanted from the start.

**Known blocker, established here:** VINDEX3 has no gradient, adjoint,
Jacobian, VJP or JVP machinery, and the workspace carries no autodiff
dependency. It is a pure forward engine. An adjoint field is therefore new
construction — every op (attention, FFN, norms, RoPE) needs a transpose —
not a wiring exercise. That cost must be scoped before a rung is
pre-registered around it.

`down_proj` remains the falsifier. Any downstream field must say, at once:

```text
global down-protected   poor return
late5-ffn               high return
k/v                     poor return
```

It cannot reach that by rescaling local magnitude or by reproducing a depth
slope — 1B′ already has both (D = 4.1213 on down_proj), and `v_proj`'s
D = 9.9119 shows the steepest depth concentration in the model belongs to a
frozen negative.
