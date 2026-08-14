# V1 — Hash routing across all layers (aim-validation, KU4)

**Status:** COMPLETE on three dense archs (Gemma 3 4B, Llama 2 7B, Mistral 7B) — unanimous falsification. MoE (26B) deferred (see scope note).
**Harness:** `chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_v1_hash_routing.rs`
**Artifacts:** `bench/aim-validation/v1_<model>.json`
**Date:** 2026-05-31

## The question

Exp 27 measured cheap/hash FFN routing on **Gemma 3 4B, layer 0 only**: top-2048
(~20% of d_ffn) → next-token KL ≈ 0.030. The medium-term ~80% confidence in
`ROADMAP.md` ("Gemma 4 26B-A4B ≥ 10 tok/s on 64 GB, no GPU") rests on a **5× FFN
bandwidth reduction** that *assumes that one-layer result compounds across all
layers and survives at the end-to-end output*. V1 resolves **KU4** by testing it
directly. The prior from the WalkFfn speed thread (#17–#28: the FFN is dense,
faithful K ≈ 4096, gate-KNN ranking dominates) was that the per-layer threshold
balloons at depth and the 5× claim shrinks. V1 is the measurement, not a kernel.

## Method (judged only in predictive units)

Three stages, all on next-token KL (bits) / held-text NLL (bits/token) / argmax
drift — never cosine:

- **Step 0 — parity anchor (spine).** Full-K single-layer walk == dense
  (KL ≈ 0); gate-KNN top-2048 @ L0 reproduces the exp-27 regime.
- **Phase A — per-layer oracle threshold.** For each layer L, with all other
  layers dense, the minimum k (gate-score top-k = the accuracy ceiling for any
  size-k route) such that the **lm_head** next-token KL ≤ 0.05. Isolates L's
  contribution to output divergence. This is a *screening proxy*, not the gate.
- **Phase B — compounding (the claim gate).** All per-layer thresholds applied
  *simultaneously*; held-text NLL distribution + argmax drift + perplexity. This
  is where the #26 lesson bites: single-layer KL ≤ 0.05 can hide compounded cost.
- **Phase C — cheap-route realizability.** At each layer's oracle threshold k,
  does a *cheap* route that doesn't pay the full gate projection (strided lower
  bound; ‖down_row‖ static-importance) hit the same KL? The gap is the price of
  cheap routing; the 5× claim needs cheap routing to clear KL ≤ 0.05 at small k.

Bandwidth is accounted honestly in "FFN weight rows touched / token": dense reads
all gate+up+down; the **cheap** route reads 3·k rows; the **gate-oracle** route
reads `feats + 2·k` (it pays the full gate projection to rank). Phase B uses
oracle selection, so its realised saving is the oracle line — the 5× best case
is only reachable on the cheap line (which Phase C tests).

## Gemma 3 4B — result (decisive)

**Step 0 (spine).** Full-K @ L17 KL = 0.00012 (walk path faithful); exp-27 @ L0
k=2048 KL = 0.011, agree 100% — same regime as exp 27's 0.030, and *lower*
because gate-oracle selection beats a token-ID hash route (the expected ordering).

**Phase A — per-layer threshold is small but non-monotonic.** Mean threshold
fraction **0.1222** (12.2% of features/layer to keep per-layer KL ≤ 0.05). The
profile is *not* "falls off by L3": most layers route at 1.6%–25%, but L3, L5, L33
need 50%. So the per-layer FFN output *is* information-sparse under oracle
selection, even at depth.

**Phase B — per-layer KL ≤ 0.05 does NOT compound (the headline).** Applying all
34 per-layer oracle thresholds at once:

| metric | dense | compounded | Δ |
|---|---:|---:|---:|
| NLL (bits/token, mean) | 4.239 | 9.595 | **+5.356** |
| perplexity | 18.9 | 773.3 | **+3995%** |
| argmax drift | — | **78.4%** | first-divergence pos 1 |

Each layer individually clears KL ≤ 0.05 with *perfect* (gate-oracle) routing, yet
compounding them destroys the model — 40× worse perplexity, 78% of tokens flip,
divergence from token 1. The single-layer screen massively understates the
compounded cost. This is the #26 phenomenon at the inter-layer scale, and it is
consistent with the prior validated #19/#23 result (uniform K=512 flips ~40% of
tokens end-to-end). **The "independent wins compound multiplicatively" assumption
fails — even with an oracle router.**

**Phase C — cheap routing can't realise even the oracle sparsity.** At the oracle
thresholds, the content-blind strided route is hopeless (KL 0.07–2.5), and the
informed ‖down_row‖ static route clears KL ≤ 0.05 at only **16% of the 31
small-threshold layers** (e.g. L0 KL 25.7, L7 5.8, L6 1.57). Where it doesn't, the
per-layer sparsity needs the gate projection — i.e. the bandwidth you were trying
to save.

**Bandwidth, honestly.** Cheap-route best case 0.122× of dense → 8.18× *if*
realisable; but Phase C shows it largely isn't. The actually-deployable
gate-oracle config touches 0.415× → only **2.41×** (gate projection still paid),
and *that* config is the one that compounds catastrophically (Phase B).

### Verdict (Gemma 3 4B)

The 5× compounding hash-routing bandwidth win is **falsified**, for two
independent reasons that converge:

1. **Compounding fails even with oracle routing** (Phase B): per-layer KL ≤ 0.05,
   applied together, gives +5.36 bits/token and 78% drift.
2. **Cheap routing can't realise even the oracle sparsity** (Phase C): the
   ‖down‖ route clears the per-layer KL at only 16% of layers; the rest need the
   gate projection.

This strengthens and is consistent with the WalkFfn speed thread (#17–#28): the
FFN is dense. KU4's medium-term bandwidth assumption shrinks accordingly.

## Cross-arch — unanimous

All three dense architectures falsify the claim, and the effect *sharpens* with
how aggressive the per-layer screen is:

| model | layers | Phase A mean frac | Phase B Δ-NLL | perplexity | drift | first-div | cheap BW (if realisable) | deployable BW (oracle) | Phase C realisable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 3 4B | 34 | 0.1222 | +5.36 | +3995% | 78.4% | pos 1 | 8.2× | 2.41× | 16% |
| Llama 2 7B | 32 | 0.0269 | +7.69 | +20480% | 95.2% | pos 0 | 37.2× | 2.85× | 62% |
| Mistral 7B | 32 | 0.0605 | +7.44 | +17209% | 90.4% | pos 0 | 16.5× | 2.68× | 31% |

Two architecture-neutral conclusions:

1. **The per-layer KL ≤ 0.05 screen is anti-correlated with the truth.** The
   *lower* it lets you push per layer (Llama 2 at 2.7% "looks like" a 37×
   bandwidth win), the *worse* the compounded collapse (+7.69 bits/token, 95%
   drift, divergence from token 0). The screen rewards exactly the aggressive
   sparsity that compounds catastrophically — single-layer KL is the wrong proxy
   for a repeated-across-depth operation (the `feedback_metric_matches_operation`
   discipline).

2. **Deployable bandwidth is ~2.4–2.9×, not 5–37×, and is catastrophic anyway.**
   The cheap "best case" (8–37×) is mostly unrealisable (Phase C realisable at
   16–62%), and even where it is realisable it's moot: the gate-oracle config
   that the cheap route would have to match *already* destroys the model in
   Phase B. The mechanism — residual-stream error accumulating across depth — is
   architecture-neutral, exactly as predicted.

## KU4 resolution + roadmap impact

KU4 ("hash-routing compounding across all layers") is **resolved for dense
architectures: falsified.** The 5× *within-FFN* bandwidth multiplier does not
survive compounding, and the cheap routing needed to realise it largely doesn't
exist. This strengthens, and is consistent with, the WalkFfn speed thread
(#17–#28): the FFN is dense. The medium-term ~80% confidence's driver narrows —
it no longer rests on stacking the FFN-hash-routing lever, but on MoE expert
active-param sparsity (a different mechanism V1 did not touch). The 80% number
itself is flagged for review, not unilaterally changed.

**What V1 does NOT close:** the MoE-within-expert hash-routing question (below),
and V2/V3/V4. V4's "independent wins compound multiplicatively" central claim now
has a second concrete counter-example (after D-RMS-FUSE / ADR-015): per-layer FFN
sparsity is destructively, not multiplicatively, compounding.

## Reproduce

```
cargo run --release --example walk_ffn_v1_hash_routing -- <VINDEX> [--json=PATH] [--quick] [--smoke]
```
`--quick` = parity anchor + 3-layer Phase-A spine check (fast). `--smoke` = skip
the Phase-A sweep (uniform k=feats/8) to exercise Phase B/C + JSON cheaply.
Default = full A+B+C+JSON. Run on the three dense matrix vindexes; artifacts land
at `bench/aim-validation/v1_<model>.json`.

## Scope note — MoE (Gemma 4 26B-A4B) is a different object

The 26B-A4B `interleaved_q4k` stores a single dense MLP per layer
(`layers.N.mlp.{gate,up,down}_proj`), not per-expert tensors — the real experts
live in the remote/shards path. Running V1 through the dense `WalkFfn` /
`predict_with_ffn` harness on the 26B would measure a dense-collapsed FFN, not the
expert routing — the wrong object. The MoE-FFN hash-routing question (sparse
routing *within* experts) is arguably the more important one for the 80% tier, but
it needs expert-aware tooling and is tracked as a V1 follow-up, not forced here.
