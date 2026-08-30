# SENSITIVITY-1B — activation-weighted local error (pre-registered)

Written before any activation is captured or any score computed. 1A failed
because relative weight error is flat across projections (0.00893–0.00904),
so a per-byte ranking degenerated into "protect the smallest tensor". The
missing factor is not how far the weights move but how much the model's
activations amplify that movement.

## Primary score — pre-registered

For tensor `W` with input activation `X`:

```text
E_local(W) = || XW - X·Q(W) ||²  /  || XW ||²
```

**This is the primary. The decision is recorded here so it cannot be
chosen after seeing which normalisation fits Granite.**

A secondary variant is computed from the same captured activations and
reported alongside, but it is *not* eligible to be promoted to primary
after the fact:

```text
E_residual(W) = || XW - X·Q(W) ||²  /  || residual ||²
```

The difference matters: the primary asks "how wrong is this operand's own
output", the secondary asks "how wrong is it relative to the stream it
joins". A tensor whose output is small compared to the residual can be
badly wrong in its own terms and barely move the model.

## Method

1. **Calibration subset**: a fixed 12 prompts drawn from Q-BANK-1, two per
   category (code, prose, arithmetic, structured, factual, longform), named
   in `calibration.json`. Deliberately not the whole bank — a screen that
   needed the whole bank would not be a screen.
2. **Capture once, from BF16.** Per tensor input site, accumulate the
   per-feature second moment `d_j = E[x_j²]` over the calibration
   positions. That is a vector, not a matrix, and there are three distinct
   sites per layer (attention input, attention output, FFN intermediate).
3. **Reuse those exact moments for every candidate.** Nothing is
   re-captured per candidate; that is what keeps the screen cheap.
4. **Score each tensor** using the diagonal approximation
   `||XΔW||² ≈ Σ_j d_j · ||ΔW_{j,:}||²`, with `ΔW = W − Q(W)`, normalised
   as above.
5. **Aggregate a candidate region** as the sum over the tensors it
   protects, and a per-MiB variant, exactly as 1A did.

## Calibration provenance — 1B-a then 1B-b

The 12 calibration prompts are drawn from Q-BANK-1 itself. That is
acceptable for the *first* falsification and not for a claim of
generality, so the distinction is banked before the result rather than
after:

| rung | calibration | claim it can support |
|---|---|---|
| **1B-a** | 12 frozen Q-BANK prompts | can activation weighting recover the known signal *at all* |
| **1B-b** | independent prompts, disjoint from the bank | the proxy generalises beyond the distribution that defined the verdicts |

1B-a is legitimate because the formula was frozen before any activation
existed, no coefficients are fitted, and the verdicts were frozen earlier
still. But a proxy could in principle score well because its activations
were sampled from the same prompt distribution that produced the ranking
it is being judged against. **A 1B-a pass does not license proposing maps.
Only a 1B-b pass does.**

## The bar — unchanged

Both halves, judged on the primary score:

1. identifies late-FFN as highest-return;
2. rejects `v_proj`, `k_proj`, `down_proj` as low-value.

No relaxation if Spearman looks encouraging. An aggregate correlation with
the negatives still ranked highly is a failure, as it was for 1A.

## Reconstruction control — pinned, not assumed

`down_proj`'s input is reconstructed offline as `act(gate(x)) ⊙ up(x)`
rather than tapped. A mathematically equivalent reconstruction can still
be numerically different — wrong activation, wrong operand order, a
scaling the executor applies and the screen does not — and it would
silently corrupt exactly one of the three frozen negatives.

So it is checked, not asserted: the executor's FFN output is tapped, the
screen recomputes `reconstructed_input · down_proj`, and the two are
compared. If they disagree the reconstruction is wrong regardless of how
plausible the algebra looks.

## If 1B fails

Escalate to 1C (curvature / Hessian-aware), do not add heuristics. A local
output-error score that still ranks V/K/down highly would say local
consequence is insufficient and second-order structure is required — which
is a result, and the earned justification for 1C's cost.

## Note for K3

1A's failure already carries a scaling consequence: precision cannot be
assigned intelligently from weight statistics alone, so any 2.8T-scale
compiler needs representative activation traffic through the model. 1B is
the cheapest form of that, and its per-feature moments are a vector per
site rather than anything that scales with expert count.


---

# Result: 1B-a FAILS on the primary score

Controls first, both pass:

```
reconstruction control   max |reconstructed - executor| 3.18e-06
                         relative to output magnitude   1.07e-07   PASS
observed vs unobserved   an_observed_step_is_bit_identical...      PASS
```

So `silu(gate) ⊙ up` is the executor's semantics and the captured
activations are the ones execution sees. The failure is the score's, not
the harness's.

```
  candidate          +MiB    1B e/MiB   Q-BANK p99/MiB
  late5-ffn           431    0.000293          0.00774
  late10-ffn          862    0.000302          0.00390
  ffn-protected      3450    0.000306          0.00115
  v-protected          72    0.005990          0.00027
  down-protected     1150    0.000297         -0.00016
  k-protected          72    0.004119         -0.00034

  1B primary   v-protected > k-protected > late10-ffn-v > ...
  Q-BANK       late5-ffn > late10-ffn > late10-ffn-v > ...

  Spearman -0.524   (1A was -0.313)

  1. late-FFN highest-return : FAIL  (top = v-protected)
  2. v/k/down low-value      : FAIL  (ranks 1, 2, 7 of 8)
  => 1B-a FAIL
```

## Why: the normalisation reintroduces 1A's bias

`v_proj` and `k_proj` top the ranking again, and for the same underlying
reason. Dividing by `‖XW‖²` measures error *relative to the operand's own
output*, so an operand whose output is small scores high — exactly as 1A's
division by `‖W‖²` rewarded operands whose weights were small.

Activation weighting did supply the missing factor. **The normalisation
then removed it again.**

## An observation that is NOT a result

The secondary unnormalised variant, `Σ_j d_j ‖ΔW_{:,j}‖²` per MiB,
computed from the same capture, ranks:

```
late5-ffn > late10-ffn > late10-ffn-v > down-protected > late15-ffn
          > ffn-protected > k-protected > v-protected
```

against a truth order of `late5-ffn > late10-ffn > late10-ffn-v >
late15-ffn > ffn-protected > v > down > k`. Top three exactly right, both
of `v`/`k` at the bottom, one misplacement (`down-protected`).

**This is not being promoted.** The pre-registration says the secondary is
not eligible to become primary after the fact, and this is precisely the
situation it was written for: the variant looks good *on the data that
suggested it*, which is not evidence. Promoting it here would make the
whole ladder decorative.

If the unnormalised form is to be tested, it must be pre-registered as
1B'-primary and judged on **1B-b's disjoint calibration set**, scored
once. That is its first and only shot.

## What 1B-a establishes

Activation weighting alone does not rescue the screen, and the specific
lesson is about *normalisation*, not about activations: both failures came
from dividing by a quantity that scales with the operand's own size.
A local score must express **absolute consequence**, not consequence
relative to the operand.
