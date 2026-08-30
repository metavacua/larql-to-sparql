# SENSITIVITY-1D — feasibility note (NOT a pre-registered rung)

Paper analysis only. No estimator is pre-registered here, no score is
computed, and nothing is built. The question is whether an amortised
downstream-sensitivity field is *theoretically capable* of separating the
`down_proj` counterexample, and whether it can be obtained at a cost
independent of the number of scored directions.

Written after 1C's suffix-replay primitive failed feasibility (batching
impossible in the current execution topology; `O(directions)` asymptotics).

## Q2 first: for KL truth, the leading term is necessarily second-order

Take the perturbed output `q = softmax(z + δz)` against the BF16 reference
`p = softmax(z)`.

```text
KL(p ‖ q)  =  ½ δzᵀ A δz  +  O(‖δz‖³),        A = diag(p) − p pᵀ
```

`A` is the softmax Fisher. At `q = p` both the value and the **first
derivative vanish** — a divergence is stationary at its reference. So:

> A first-order score against KL is identically zero. It cannot rank
> anything.

This settles the ordering of Q1 and Q2. First-order is only meaningful
against a functional that is *not* stationary at the reference — a specific
token's NLL, or a margin — not against the divergence Q-BANK actually
reports.

## Q1: first-order is cheap but measures something else

For `F = −log p(y)` at the teacher-forced token, `∇_z F = p − onehot(y)`,
which is non-zero at the reference. One reverse pass per position gives
`g_l = J_lᵀ ∇_z F` at every boundary, and `s₁ = g_l · δh` is defined.

Two problems, both naming problems rather than fatal ones:

- **Sign cancellation.** Quantisation error is near-zero-mean across
  positions, so `E[g·δ]` can approach zero for a genuinely damaging
  operand. Taking `E[|g·δ|]` avoids that but no longer estimates the change
  in expected `F` — it estimates *directional susceptibility*. That may be
  the right screening quantity, but it must be called that.
- **Wrong target.** It measures displacement of one token's likelihood, not
  distributional degradation. Q-BANK's headline is KL and p99.

Verdict: worth reporting as a cheap secondary (`r = 1`), never the primary.

## Q5 — the real gate: YES, the field amortises over directions

This is the result that matters. Write the per-boundary Gauss–Newton form:

```text
score(δh)  =  δhᵀ G_l δh ,        G_l = J_lᵀ A J_l ,      J_l = ∂z/∂h_l
```

Materialising `G_l` is hopeless (2560² per layer per position) and one
Jacobian-vector product per direction rebuilds 1C's `O(directions)`
disaster under a new name. Neither is necessary.

`A` factors exactly. With `D = diag(p)` and `√p` the elementwise root:

```text
A  =  D − p pᵀ  =  Mᵀ M ,      M = (I − √p √pᵀ) D^½
```

So drawing `ξ ~ N(0, I_V)` and setting

```text
u  =  Mᵀ ξ  =  √p ⊙ ξ  −  p (√p · ξ)          [V-dimensional, cheap]
```

gives `E[u uᵀ] = A`. Now **one reverse pass seeded with `u` produces
`w_l = J_lᵀ u` at every layer boundary simultaneously** — that is simply
what reverse mode does. And for any direction `δ`:

```text
E[(w_l · δ)²]  =  δᵀ J_lᵀ E[u uᵀ] J_l δ  =  δᵀ G_l δ
```

Unbiased. So `r` reverse passes give an estimate of `δᵀ G_l δ` for **every
direction at every boundary**, and scoring a direction is a dot product.

```text
forward once
reverse r times            <-- r is a variance knob, not a direction count
  -> adjoint vectors w_l at every boundary
  -> contract all 240 banked δ directions by dot product
```

**Cost is O(1) in the number of scored directions.** 240 tensors or
240,000 cost the same. That is the property SENSITIVITY has wanted since
1A, and the property suffix replay could not have.

## Q4: asymptotic cost

Taking a reverse pass at ~2× a forward, over the frozen 458-position
calibration set (one full calibration forward = 0.282 Q-BANK
candidate-equivalents):

```text
   r    traversal-equiv    ×QBANK cand    rel. std per tensor
   1                3.0           0.85                   141%
   4                9.0           2.54                    71%
   8               17.0           4.80                    50%
  16               33.0           9.32                    35%
  32               65.0          18.35                    25%

  suffix replay, 6 arms, 40 layers          80.8   and O(directions)
```

Relative standard deviation is `√(2/r)` per tensor — `(w·δ)²` is a scaled
`χ²₁`. Candidates aggregate 15–120 tensors, so the ranking error falls
further, and ranking tolerates more noise than estimation. `r` is chosen
against a cost ceiling and then frozen; it must never be raised because a
result failed.

## Q3: reverse primitives required

`w_l = J_lᵀ u` needs the transpose of everything between boundary `l` and
the logits:

```text
output head        Wᵀ, and RMSNorm backward
per layer          residual (identity)
                   FFN:       downᵀ, silu′ ⊙, gateᵀ / upᵀ
                   attention: oᵀ, softmax Jacobian over the KV cache,
                              RoPE transpose, q/k/vᵀ
```

Attention is the hard one, and there is a correctness trap: a perturbation
at position `i`, layer `l` changes that position's K/V for every layer
`≥ l`, which changes **later positions'** attention. The adjoint must run
over the whole sequence, not per position. A per-position backward would
silently drop the cross-position path and measure a different quantity.

**Established blocker:** VINDEX3 has no gradient, adjoint, Jacobian, VJP or
JVP machinery, and the workspace carries no autodiff dependency. This is
new construction. The return on it is better than replay's — one backward
implementation, reused `r` times per position, yielding every boundary at
once — but it is not wiring.

## Can it actually fix `down_proj`? The theory permits it; it does not promise it

1B′ scores `‖δh‖²`-like magnitude. The Gauss–Newton form scores
`δhᵀ G_l δh`, which is small when a large perturbation lies in directions
the output distribution barely responds to. So the mechanism needed —

```text
large Euclidean perturbation  ×  low downstream curvature  =  low importance
```

— is exactly what this form can express, and it can express it
**depth-conditionally**, since `G_l` differs per boundary. That clears the
capability bar the joint condition sets: demote global `down_proj`
protection while `late5-ffn` stays first.

Whether Granite's `down_proj` perturbations *actually* lie in low-curvature
subspaces is empirical. The theory says the estimator is not disqualified;
it does not say the answer.

## The honest caveat: the perturbation may not be small

The second-order expansion is a **local** model. Granite's R0 induces
`KL mean 0.278, p99 4.622`. A p99 of 4.6 nats is not a small perturbation,
and a quadratic Taylor term is a poor description of it.

This is a real threat to validity, not a footnote. Two mitigations are
available and neither should be chosen after seeing a result:

- Judge the screen on **ranking**, not magnitude — which the frozen bar
  already does.
- Report the estimator's agreement in the regime where it is defensible
  (the KL median, 0.061) separately from the tail it is not built for.

If the eventual rung passes its bar, this caveat bounds the claim: it would
be evidence for a screen that ranks, not for a calibrated predictor of
Q-BANK KL.

## Verdict

```text
Q1  first-order          defined for NLL, zero for KL, cancellation-prone
                         -> cheap secondary only
Q2  KL leading term      NECESSARILY second order
Q3  reverse primitives   whole-sequence adjoint; attention is the work
Q4  cost                 (1 + 2r) traversals TOTAL over the calibration set
Q5  amortisation         YES — O(1) in directions, via the Fisher sketch
```

The scaling disaster is avoided: `r` is a variance parameter, not a
direction count. The blocker is construction cost, not asymptotics.

**Not decided here:** whether to build it. That needs a scoped estimate of
the attention adjoint against the ladder's remaining appetite, and it
should be weighed against the fact that four rungs have now failed. Nothing
in this note is pre-registered, and no rung is opened by it.
