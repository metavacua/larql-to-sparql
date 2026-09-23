# V1 (MoE-within-expert) — does feature routing work inside a single expert's FFN?

**Status:** RESOLVED 2026-06-05 — **FALSIFIED** (closes the OPEN half of KU4).
**Aim-validation:** resolves the OPEN half of KU4 (V1 was falsified on dense; the
MoE-within-expert variant was left open because the dense harness measures the
wrong object on the 26B-A4B).
**Harness:** `chris-experiments/larql_probes/examples/walk_ffn/walk_ffn_v1_moe_within_expert.rs`
**Kernel hook:** `crates/larql-compute/src/cpu/ops/moe/within_expert.rs`
**Model:** Gemma 4 26B-A4B (`output/gemma4-26b-a4b-q4k.vindex`), 30 layers (all
MoE), 128 experts, top_k=8, **expert inter = 704**.

## Question

V1 (`docs/diagnoses/v1-hash-routing.md`) tested hash routing *within a dense FFN*
and falsified it on Gemma 3 4B / Llama 2 7B / Mistral 7B: per-layer KL ≤ 0.05
thresholds do **not** compound (+5.4 to +7.7 bits/tok, 78–95 % drift). On the
26B-A4B, each per-layer FFN block is **128 stacked experts**, so that harness
treats the interleaved expert block as one dense 4096-feature FFN and measures
the wrong object. This probe asks the architecturally-correct question:

> Within a single routed expert's gated FFN (704 features), how few
> post-activation features can we keep before the expert's output — and the
> model's held-text NLL — degrades, and does it compound across the
> 8×30 = 240 expert invocations per token?

The expert feature space (~704) is ~6× smaller than dense d_ffn, and the
load-balanced top_k router already concentrates work — so the prior that
"FFN is dense" might not transfer.

## Method (mirrors V1 exactly; judged only in predictive units)

The prune is applied **inside the production expert kernel**
(`run_single_expert_q4k_q8k_into`) via a global, opt-in within-expert routing
schedule — OFF by default → byte-exact parity (a single relaxed atomic load).
Errors therefore propagate through the real forward pass; no reimplemented
numerics (`feedback_engineering_vs_research_posture`: parity is the spine).
The oracle selector keeps the top-`k` features by `|act|` (the post-activation
magnitude entering `down`) — the accuracy ceiling, analogous to V1's
gate-oracle. The cheap selector keeps `~k` features on a fixed stride
(content-blind).

- **Step 0 — parity anchor.** All-dense schedule must equal dense (KL≈0);
  one expert layer pruned hard must bite (KL>0).
- **Phase A — per-expert-layer oracle threshold.** Min keep-frac for
  next-token KL ≤ 0.05, one layer pruned at a time (a SCREEN).
- **Phase B — compounding (claim gate).** All expert layers pruned at their
  thresholds simultaneously → held-text NLL + argmax drift (the #26 lesson:
  single-step KL once *inverted* the decision).
- **Phase C — cheap-route realizability.** Content-blind strided route vs the
  ActMagnitude oracle at the thresholds.

## Results

### Step 0 — parity anchor (GREEN)

| check | result |
|---|---|
| all-dense schedule (instrument off-by-frac) | **KL = 0.00000 bits** (faithful) |
| L15 @ frac=0.125 (88/704 feats) | KL = 0.107 bits, top-1 agree 100 % (knob bites) |

### Phase A — per-expert-layer oracle threshold (min keep-frac for KL ≤ 0.05)

A sharp **depth split** (not a uniform fraction):

| layer band | threshold | reading |
|---|---|---|
| **L0–L13** (14 layers) | **frac = 1.0** (all 704; even 1/2 exceeds KL 0.05) | early experts are **fully dense** in their own feature space |
| **L14–L29** (16 layers) | frac 0.016–0.25 (11–176 feats) | late experts tolerate aggressive single-layer pruning |

Mean threshold fraction **0.52**. This mirrors the dense WalkFfn "scissors"
(`project_walkffn_speed_accuracy_scissors`): sparsity survives only in a thin
late band, and *per-layer* tolerance is a screen, not a deployable saving.

Bandwidth at these per-layer thresholds (per active expert, gate+up not free):
- **oracle (deployable, Phase B cfg): 0.84× of dense → only 1.19× reduction** —
  the ActMagnitude oracle must run the full gate+up to know `|act|`, and half
  the layers save nothing.
- cheap content-blind "best case": 0.52× → 1.91× — *but* unrealizable (Phase C).

### Phase B — compounding (the claim gate; held passage 43 tok)

| | mean NLL (bits/tok) | p90 | max | perplexity |
|---|---|---|---|---|
| dense | 12.443 | 24.094 | 32.725 | 5569.4 |
| comp (all layers @ threshold) | 12.293 | 23.088 | 32.944 | 5019.2 |

**Δmean NLL = −0.15 bits (comp lower); ppl −9.88 %. argmax drift = 50.0 %;
first-divergence pos 2.**

This is the **#26 trap, textbook** (`feedback_metric_matches_operation`): the
mean NLL is *noise-dominated and even points the wrong way* (comp looks
"better"), while the deciding signal — **half the next-token argmaxes flip** —
shows the generation fully diverges. An expectation lies when the use is a
repeated extremum; the mean here is dominated by a few near-random high-entropy
tokens (max 32.7 bits) and is not the ship metric. Verdict reads off drift.

### Phase C — cheap-route realizability (Strided vs ActMagnitude oracle)

Content-blind strided route clears KL ≤ 0.05 at **6 % (1 of 16)** of the
small-threshold layers; strided-KL runs 0.04–1.59 vs oracle 0.01–0.05 (often
10–50× worse). The per-layer sparsity the oracle finds **needs the full gate+up
projection to locate** — so the 1.91× "best case" is not reachable; only the
1.19× oracle line is, and that is the one Phase B shows is catastrophic.

## Verdict

**FALSIFIED — same outcome as dense V1, now for the architecturally-correct
object.** Within-expert feature routing on the 26B-A4B does **not** yield an
exploitable bandwidth multiplier:

1. **Experts are dense in their own feature space.** Early experts (L0–13) need
   *all* 704 features even at a single layer; only a late band (L14–29) is
   sparse-tolerant — and only as a per-layer screen.
2. **It doesn't compound.** At the per-layer thresholds applied together, 50 %
   of next-token argmaxes flip (mean NLL is the #26 lie — don't ship on it).
3. **Deployable bandwidth is ~1.19×, not 5×** (oracle pays gate+up; half the
   layers save nothing), and even the 1.91× "best case" is unrealizable (cheap
   route clears 6 % of layers).

The expert is already a compact, specialized FFN — there is no second sparsity
axis to exploit *inside* it. Combined with the dense V1 result
(`v1-hash-routing.md`), the FFN/expert feature-sparsity bandwidth multiplier is
**dead on this architecture, dense and MoE alike.** KU4 fully resolved.

Caveat: one model (Gemma 4 26B-A4B), one held passage, ActMagnitude oracle (a
‖down‖-weighted importance oracle is a tighter ceiling but would only *raise*
the threshold, not lower it). The instrument is parity-anchored, so a follow-up
on a shared/fine-grained-expert MoE (smaller per-expert inter) is cheap if the
question is ever re-opened.

## Artifacts

- `bench/aim-validation/v1moe_gemma4-26b-a4b-q4k.json`
