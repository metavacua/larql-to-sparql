# SENSITIVITY-1 — can a cheap local signal predict the expensive verdict?

Q-BANK is the promotion gate and is too expensive to be the search. Screening
every projection × depth candidate through 1,622 teacher-forced positions
does not scale to R2's combinatorics, let alone K3's.

So the question is narrow and falsifiable:

> Can a cheap **local** score rank precision-map candidates the way
> Q-BANK's **global** verdict ranks them?

## The bar, fixed before scoring

Both halves, or it is not a predictor:

1. **Identifies late-FFN as the highest-return region.**
2. **Rejects `v_proj`, `k_proj` and `down_proj` as low-value.**

An aggregate correlation passing while (2) fails is a *failure*. It would
mean the proxy has learned "protecting more bytes helps", which is true,
useless, and exactly what the frozen negatives exist to catch:

| candidate | +MiB | Q-BANK verdict |
|---|---|---|
| `v_proj` | 72 | KL 0.278 → 0.264. Near-zero benefit, and the canonical ecosystem move. |
| `k_proj` | 72 | Flip difference +4, bootstrap 95% CI [−8, +16]. Indistinguishable from zero. |
| `down_proj` | 1,150 | p95 got **worse** (1.525 → 1.592) for over a gigabyte. |

A ranking driven by byte cost alone would place `down_proj` third of
fifteen. Q-BANK places it near the bottom.

## Validation set

Fifteen Granite candidates with frozen end-to-end outcomes, in
`granite-4.1-3b-sweep.json` — R0, four single-projection probes, two
class probes, two depth probes, five role×depth intersections, and a
determinism control. The knee (`late5-ffn`, +7.93 p99/GiB, with
`late10-ffn` at +0.05) is the shape the screen has to reproduce.

## Method

The screen scores a *region* by what quantising it does locally, with no
forward pass over the bank:

```text
for each candidate region R:
    e(R) = sum over tensors t in R of
             ||W_t - dequant(quant(W_t))||^2 / ||W_t||^2   (relative error)
           weighted by t's share of the region's bytes
```

This is the oQ-style normalised local error, computed from weights alone.
It is deliberately *not* a forward-pass metric: if a local signal suffices,
screening costs one pass over the weights rather than one per candidate.

## Escalation ladder, frozen before any score is computed

The proxy below is weight-only. It measures how far the weights move, not
how strongly the model *uses* the directions they move in — two tensors can
carry identical relative weight error while one sits in an
activation-sensitive region and the other barely matters. So it may well
fail, and the response to failure is fixed in advance rather than chosen
after seeing which formula would have passed.

| rung | score | cost |
|---|---|---|
| **1A** | `‖W − Q(W)‖² / ‖W‖²`, weight-only | one pass over the weights |
| **1B** | `E[‖XW − XQ(W)‖² / ‖XW‖²]`, activation-weighted over a small calibration sample | one forward pass per layer, still no bank |
| **1C** | curvature / Hessian-aware, the oQ/GPTQ-style machinery | calibration plus second-order statistics |

**Do not massage 1A until it passes.** A clean 1A failure is itself a
result: it would say that weight geometry alone does not predict semantic
quantisation sensitivity, which is worth knowing and is the argument for
1B. Tuning a weight-only score until it happens to reproduce fifteen known
verdicts would produce a number that fits this validation set and predicts
nothing.

Each rung is scored against the same frozen fifteen, against the same bar.

## Order

1. Glimmer coarse surface — is its low R0 damage flat or concentrated?
2. Compute local scores over Granite's fifteen candidate regions.
3. Rank by the cheap score alone.
4. Compare against the frozen Q-BANK ranking, and check both halves of the bar.
5. Only if it passes, let it propose a map nobody has tested.
6. Q-BANK remains the promotion gate. The screen proposes; it never promotes.

No optimizer and no budget solver until step 4 passes. A search primitive
that cannot predict the verdict is not a search primitive.


---

# Result: 1A FAILS, and the signal is flat

Scored on Granite in 28 s against 1:51 per Q-BANK candidate — the cost
ratio is right. The prediction is not.

```
rankings (best first)
  1A rel/MiB       k-protected > v-protected > attn-protected > o-protected > ...
  Q-BANK p99/MiB   late5-ffn > late10-ffn > late10-ffn-v > late10 > ...

Spearman vs Q-BANK p99/MiB   -0.313

THE BAR
  1. identifies late-FFN highest-return : FAIL  (top = k-protected)
  2. rejects v/k/down as low-value      : FAIL  (v and k rank 1st and 2nd of 13)
  => FAIL
```

It ranks the two frozen negatives **first and second** — the exact
candidates Q-BANK found buy nothing — and puts the measured knee fifth or
lower. The correlation is *negative*, so the score is not merely
uninformative; following it is worse than ignoring it.

## Why: relative weight error is constant

```
projection     n  rel_error mean      params
o_proj        40        0.009039   6,553,600
up_proj       40        0.008999  20,971,520
gate_proj     40        0.008997  20,971,520
q_proj        40        0.008974   6,553,600
down_proj     40        0.008948  20,971,520
k_proj        40        0.008942   1,310,720
v_proj        40        0.008931   1,310,720
```

Every projection quantises to within 1.2% of the same relative error.
That is what NVFP4 *is*: a fixed relative grid, so it introduces
approximately fixed relative error regardless of what the weights mean.

The numerator therefore barely varies, and a per-byte score degenerates
into ranking by inverse byte cost — smallest tensor wins. `k_proj` and
`v_proj` are 1.3 M parameters against `gate_proj`'s 21 M, so they top the
ranking for costing least, not for mattering most.

## What this establishes

**Weight geometry alone does not predict semantic quantisation
sensitivity** — and specifically, for a fixed-relative-precision format it
carries almost no discriminating signal at all. The differences Q-BANK
measures between late FFN and everything else are invisible to a metric
computed from weights in isolation.

This is the argument for 1B, and it was stated before the score existed:
what varies between these tensors is not how far the weights move but how
much the model's activations amplify that movement. Escalate; do not tune.
