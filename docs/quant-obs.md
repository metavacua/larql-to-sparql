# Quant-Obs v0.1 — an observer-metric ladder for quantisation error allocation

**Programme:** OBS-0 → OBS-4, plus two standalone checks OBS-5 (K3 spectrum, laptop-only) and OBS-6 (WalkFfn re-score, free reuse of banked data).
**Scope:** model-agnostic by construction. Falsifiable this week on Gemma 3 4B with instruments already shipped (C6 drift). K3 is the confirmation case, not the discovery case, and its slice (OBS-5, and eventually the [`k3-funnel.md`](k3-funnel.md) ladder) runs last, if ever.
**Status:** v0.1 — draft, mirrored into the experiments registry as programme `quant-obs`, all experiments `planned`.
**Date:** 2026-07-29

---

## 1. Thesis

Every quantisation error a codec makes gets summed with other errors before anything observes the result, and the observer is a linear map with a wildly unequal spectrum. In K3, that observer has a closed form: `RMSNorm → W↑`. In Gemma (and everywhere else), the observer is the residual stream plus the rest of the stack — which is exactly the Jacobian-transport object already being fit under programme `j-space` (experiment registry, not in-tree) (`js-1-basis-rotation-dark-space`, `js-2-sensitivity-in-jacobian-basis`). Same object, different question: j-space asks whether that matrix explains dark-space structure; this ladder asks whether it predicts *quantisation drift* well enough to allocate bits offline.

**K3 is the confirmation case, not the discovery case.** Nothing below requires K3 weights, a GPU, or a download until OBS-4 passes.

## 2. The ladder

| ID | Question | Instrument | Kill criterion | Depends on |
|---|---|---|---|---|
| **OBS-0/1** (merged) | Does a candidate static metric M predict model-space drift — tested the only way that's actually valid in ~2,000+ dimensions? | M(u) = J_post(u)ᵀ·J_post(u), J_post(u) = the analytic Jacobian of Gemma 3's `post_feedforward_layernorm` (the RMSNorm **immediately downstream** of `down_proj`, structurally matching K3's `routed_expert_norm` — see §2c) at u = `down_proj`'s real raw output, averaged over a **fresh** `larql dec-bench capture` of Gemma 3 4B using DEC-0's prompt list (`bench/dec0/prompts.txt`) — the 26B-A4B pool DEC-0 already captured doesn't apply here (§2a). Perturbations to `down_proj`'s weight are constructed as rank-1 ΔW = v ⊗ r — v drawn from M's eigenbasis (top-k / bottom-k / graded mixtures, output-side), r a **fixed** random input-side direction shared across every arm, so the design isolates the output-side transport question and doesn't smuggle in input-side (GPTQ-Hessian-style) importance weighting, which OBS-2's arm B already covers separately — plus a **random-direction arm** (v uniform on the sphere, matched norm, the original design) kept deliberately as a control. Score every arm with `larql shannon score`/`verify` bits/char (in-process, not `dec-bench drift` — that needs a live FFN server and is built for wire-codec fidelity, not scoring N weight variants) | The random arm is expected to collapse to ~2% relative spread around tr(M)/d by concentration of measure — that collapse is the diagnostic, not a null result to worry about (see correction below). Kill: constructed arms don't order top-k > mixtures > bottom-k the way M predicts → *this* candidate M is falsified (a different M can still be tried before the whole family is dead) | — |
| **OBS-2** | Same bytes, same kernel, better allocation? | Reallocate precision inside the existing Q4K block structure using M. Identical file size, identical NEON/Metal kernel, no runtime change. Arms: **A** round-to-nearest, **B** GPTQ/Hessian-style importance (non-negotiable — the real incumbent), **C** observer-metric (M) allocation. Score: C6 drift at fixed bytes | C must beat **B**, not just A — beating only round-to-nearest restates known practice | OBS-0/1 passes |
| **OBS-3** | Does error cancellation need MoE routing at all? | Jointly quantise a dense FFN down-projection's columns so errors anti-correlate over the calibration feature distribution. Metric: ρ_cancel = cross-term / diagonal-term ratio in the summed error covariance. Control: random column pairing at identical per-column budget (separates "optimiser has more freedom" from "real structure exists") | ρ_cancel > −0.1 → fixed codes can't produce useful negative covariance across a varying input distribution — mechanism dead everywhere, including K3 | OBS-0/1 passes; independent of OBS-2, runs in parallel with it |
| **OBS-4** | Does routing structure buy anything *beyond* the dense result? | GPT-OSS-20B (4-of-32, native MXFP4, already local): apply OBS-3's scheme with real routing vs shuffled routing at identical sparsity | real ≈ shuffled → routing structure contributes nothing, dense form is the whole story | OBS-3 passes |

**Caution (applies to OBS-2 and OBS-3):** both are encoder-only by construction — the property that makes them worth having — but the win shows up as *drift at fixed bytes*, not tok/s. Keep these out of DEC scoreboards; a fidelity claim contaminating a capacity claim is a house anti-pattern (see `dec-funnel.md`'s C-numbered claims, which are all capacity/throughput).

### Correction, 2026-07-29 (caught before any run — OBS-0 and OBS-1 merged)

The original OBS-0 design — ~200 *randomly sampled* equal-Frobenius-norm perturbations, scored for raw drift spread — was caught as broken before it ran, by concentration of measure. In d≈3,000+ dimensions, a quadratic form δᵀMδ over uniformly random matched-norm directions δ concentrates tightly around tr(M)/d — relative fluctuation ~O(1/√d), ~2% at this dimensionality — *regardless of how anisotropic the true spectrum M actually is*. OBS-5 measured 55–77× condition numbers on K3's much smaller 3,584-dim analog; random sampling in a Gemma-sized space would have produced a spread comfortably under the 2× kill bar and reported "no anisotropy" when 55×+ (or more) was actually available. That's a **false kill** — exactly the failure mode this ladder's kill-criteria exist to prevent, caused by the test design itself rather than by the underlying claim being wrong.

The fix folds OBS-1 in early: perturbations must be *constructed* from a candidate M's eigenbasis from the start, not drawn at random and checked against a metric as a separate second stage — you cannot validly test "does anisotropy exist" independently of "here's a candidate direction where it might be," because random sampling alone cannot find the extremes in high dimensions. The kill criterion changes to match: it's about whether measured drift *orders* the way M predicts (top-k > mixtures > bottom-k), not raw spread across random draws. The random arm is kept, deliberately, as the control that proves why the original design would have failed — its collapse is the diagnostic, not noise to explain away.

**Registry consequence:** `obs-0-frobenius-anisotropy` and `obs-1-static-metric-predicts-drift` are both marked `superseded`; the combined test is `obs-01-eigenbasis-drift-ordering`. OBS-2/OBS-3's dependency is retargeted to it (see §2 above).

### Scoping decision, 2026-07-29 (M's definition, and the calibration-data gap)

Two more gaps found during build recon, resolved before any code was written:

**M's scope.** The full "Jacobian transport" object (composing through *all* remaining layers to the actual output, matching what programme `j-space` fits) isn't buildable yet — the repo's one hand-rolled reverse-mode backward pass (`crates/larql-inference/src/forward/target_delta.rs`, built for MEMIT) only supports installing at the last layer; `attention_backward_last_pos` is an `unimplemented!()` stub, so mid-layer chaining doesn't exist. Decided: first run uses **M(x) = J_gate(x)ᵀ·G·J_gate(x)** — G = `down_wᵀ·down_w` (closed form, free), J_gate(x) the analytic Jacobian of the gated FFN nonlinearity immediately upstream of `down_proj`, at a real pre-norm activation x, composing the weight Gram with one real layer of local transport rather than the full residual-stream chain. This is a genuine step past OBS-5's weight-only ceiling, not the full observer — flagged explicitly so a future full-transport build isn't confused with this one, and so `obs01-data-weighted-flatness` (§3, OBS-5 correction) is understood as calibrated on this partial composition, not the true end-to-end quantity.

**§2b — a self-caught correction, same day, before any code was written.** The first draft of this section wrote M's upstream operator as an RMSNorm Jacobian (`J_RMS`), borrowing K3's structure directly: K3's LatentMoE has an explicit `routed_expert_norm` sitting between `routed_expert_down_proj` and `routed_expert_up_proj` (see §3 OBS-5), so "RMSNorm → W" is structurally correct *there*. Gemma 3's dense FFN doesn't have that shape — `down_proj`'s input is `gelu_tanh(gate_proj(x)) ⊙ up_proj(x)` (confirmed: `pre_feedforward_layernorm` sits *before* `gate_proj`/`up_proj`, `Activation::GeluTanh` per `crates/larql-models/src/architectures/gemma3.rs:122`; `post_feedforward_layernorm` sits *after* `down_proj`, before the residual add). The operator immediately upstream of `down_proj` is the gated nonlinearity, not a norm. Fixed to `J_gate` before any implementation existed to get it wrong in — but §2c below caught that this was still the wrong *direction*.

**§2c — a second self-caught correction, same day, still before any code.** `J_gate` characterises how sensitive `down_proj`'s *input* is to something upstream — the wrong question for "does perturbing `down_proj`'s own weights cause drift." Perturbing a weight matters through what happens to its *output*: for K3, `routed_expert_norm` sits immediately **after** `routed_expert_down_proj` (its perturbed weight), and OBS-5's `diag(γ)·W↑ᵀW↑·diag(γ)` characterises exactly that — downstream propagation of an error injected at `down_proj`'s output, using the next operation as a cheap ceiling proxy for "everything downstream." Gemma 3's structural match to that position is `post_feedforward_layernorm`, not `pre_feedforward_layernorm`. Confirmed via `crates/larql-models/src/architectures/gemma3.rs:113` (`norm_weight_offset() -> 1.0`, i.e. Gemma 3 RMSNorm is `y = (1+γ)⊙x/rms(x)`) and `:129` (`has_post_norms() -> true`) — `rmsnorm_backward_pos` (`target_delta.rs:168`) already implements this exact formula and is FD-tested; the effective weight to pass it is `1+γ`, not `γ` alone. Fixed to `J_post(u)`, `u` = `down_proj`'s raw output, dimension `hidden_size=2560` (not `intermediate_size=10240` — another reason this is the right object: it composes with the perturbation's *output* dimension, matching how `down_proj`'s weight rows actually act).

Because M now lives in `down_proj`'s **output** space (2560-dim) rather than characterising the weight matrix directly, perturbation arms are constructed as rank-1 `ΔW = v ⊗ r`: `v` (an M eigenvector) sets the *output* direction the perturbation pushes toward, `r` (one fixed random *input*-side direction, shared identically across every arm) holds the input side constant so the experiment isolates the output-transport question cleanly, without accidentally reproducing GPTQ's separate input-side (`E[hhᵀ]`) importance weighting — that comparison is what OBS-2's arm B already owns.

### §2d — a third correction, this one from a real measurement, not pre-run review

First real computation ran at layer 17: `M = J_post(u)ᵀJ_post(u)`, 1024 real calibration positions, condition number 1.6×10⁷, participation ratio 20.6%. Caught on review, before any arm was built:

**M is near-diagonal, and the real-activation averaging was almost a no-op.** `J_post`'s dependence on `u` is entirely through the scalar `‖u‖` (the RMSNorm formula), and `‖u‖` only ranges 910–1136 across all 1024 positions (1.25×). Arithmetic check: `(γ_max·√d/‖u‖)² = (365 × 0.0506)² ≈ 341`, measured `λ_max = 333` — a 2.3% gap, exactly the size of the rank-1 correction the RMSNorm Jacobian applies on top of the diagonal term. So to within ~2%, **M ≈ diag(γ_eff²)·d/‖u‖²**, i.e. this measures almost nothing beyond the `post_feedforward_layernorm` weight vector itself, squared. Averaging over real activations was the methodologically correct move, but empirically inert here because the one thing `u` could have varied on (its norm) barely does.

**This is not the data-weighted term the thesis (§1) was after.** GPTQ-style anisotropy (`E[xxᵀ]`, the 10³–10⁵ benchmark) is an **input-side** second moment over the layer's actual input activations — for `down_proj`, that's `E[hhᵀ]` over its real gated-FFN input `h` (10240-dim), a completely different, not-yet-built matrix. `M` as computed is output-side and structurally close to reading the norm weights off the checkpoint directly; the comparison to GPTQ's Hessian range was an apples-to-a-different-fruit comparison, not apples-to-oranges-but-comparable.

**Condition number is an outlier statistic here, not the headline.** 1.6×10⁷ is driven by two or three channels with `γ_eff ≈ 0.04–0.17` — division by near-zero. The trace-mass curve is the number that actually governs bit allocation: 90% of sensitivity needs 68.6% of directions — a **real but modest** budget (roughly 31% of directions carry only 10% of the mass and are candidates to cheapen), not the dramatic story the condition number implies on its own.

**The originally-planned arm design would have produced a false positive.** Bottom-k eigenvectors of *this* M are, by construction, the near-zero-`γ_eff` channels — any perturbation feeding them gets attenuated to ~nothing by `post_feedforward_layernorm` regardless of what the perturbation is. That would show "spectacular separation" between top-k and bottom-k arms while testing nothing but "does the norm gain vector, which is public information already visible without computing any Jacobian, predict its own effect" — circular, not a result. Fix, not yet built: draw arms from **trace-mass quantiles** of the cumulative eigenvalue curve instead of raw top-k/bottom-k, so each arm carries a known, comparable share of sensitivity.

**A genuine smaller finding, worth keeping separate from the OBS-0/1 ladder:** channels with `γ_eff` under ~1% of the max are candidates to literally **drop** from `down_proj` (not just cheapen) — a real structural result, but a different and much smaller claim than the ladder's. Report the sub-1%-γ channel count alongside the trace-mass curve, not folded into it.

**Caveat before claiming headroom:** per-output-channel scaling is exactly the kind of structure Q4K's existing block quantisation already partially captures (per-block scales). What's actually left as *new* exploitable budget, on top of what the current kernel already does, is unmeasured — check that before treating any of this as free bytes.

**Net status:** the instrument (real-activation averaging, verified RMSNorm Jacobian, validated γ vector — cross-checked independently against `google/gemma-3-4b-it`'s raw safetensors via `safetensors`/`torch`, exact match, not a parsing artefact) is sound. It was pointed at the wrong half of the problem. The input-side `E[hhᵀ]` covariance is the next thing to build, not yet started. Arms are **not built** on the current M — paused here per plan.

### §2e — a fourth correction: the two matrices weren't alternatives, and the resume plan inverts

The §2d "build `E[hhᵀ]` next" plan was itself half wrong, caught before any further code:

**`E[hhᵀ]` and M compose, they don't substitute.** For a weight perturbation `ΔW` on `down_proj`, the output error is `ΔW·h`, and distortion is `D = tr(ΔWᵀ·M·ΔW·E[hhᵀ])` — a Kronecker-structured objective needing *both* matrices. **GPTQ already is this objective with M = identity.** So `E[hhᵀ]` isn't the missing half of the thesis — it's what arm B (OBS-2's non-negotiable GPTQ incumbent) already computes. The entire novel content of "observer-aware" quantisation lives in M, not in building the input-side term.

**This sharpens rather than relieves the problem.** M was measured near-diagonal (§2d). If that survives further scrutiny, observer-aware quantisation reduces to *per-output-row reweighting of the existing GPTQ objective* — scale row `i`'s error contribution by `γ_i`. That's a one-line patch to an existing implementation, not a programme. Worth doing, worth a short note — but the shape of the win needs to be known before building more instruments around it.

**`post_feedforward_layernorm` alone can only ever measure diagonal.** RMSNorm with a learned per-channel gain is diagonal *by construction* — finding §2d's single-hop M near-diagonal isn't evidence about the true observer, it's a consistency check on the derivation (it would have been a red flag if it had come out *non*-diagonal). The actual object — where any real non-diagonal structure would have to live — is the **composed downstream transport** through the remaining 17 layers plus unembedding: precisely the J-lens estimator already being fit under programme `j-space`. That's the real next build, not `E[hhᵀ]`.

**OBS-3 likely just rediscovers GPTQ.** GPTQ's mechanism — greedy sequential quantisation with error compensation pushed into not-yet-quantised columns — *is* "design the codes so errors cancel in the sum," for a single dense matrix. OBS-3 as specified is likely to rediscover it, not find something new. Genuinely untouched cancellation lives across **separate weight matrices summed under data-dependent selection** — co-routed experts, where no existing quantiser jointly optimises expert 17 against expert 443 because they're different tensors and their co-occurrence is a routing statistic. Novelty concentrates in **OBS-4**. OBS-3's real job changes: it becomes the *control* that measures how much cancellation is already free from existing single-matrix methods, not a discovery in its own right.

**Resume plan inverted: run arm B (plain GPTQ) first, before building any more metrics.** Score plain GPTQ on the target layer with the C6 instrument. This bounds everything downstream — if GPTQ's drift at fixed bytes is already near the f32 floor, headroom for any M-weighting is small and that ceiling is known before another session goes into the numerator. If there's a real gap, there's a concrete target with a number attached. This also inverts the ladder usefully: measuring what's *left* rather than what's theoretically available.

**Process note, carried forward:** three real bugs were caught before any conclusion was drawn (concentration of measure, wrong Jacobian direction, near-diagonal-by-construction) — the right outcome for a session, but corrections have been getting cheaper to find than measurements are to make. Running GPTQ as a reference point first is the fix: it grounds the next step in an actual number instead of another layer of theory.

**Calibration data.** OBS-0/1 as originally specified assumed it could reuse "DEC-0 arm-M's existing 64-prompt pool" — but that pool (`bench/dec0/residuals-gemma4-26b-a4b-q4k/`) is captured against **Gemma 4 26B-A4B**, not Gemma 3 4B; no capture exists for Gemma 3 4B anywhere in the repo. Decided: run `larql dec-bench capture` fresh against a Gemma 3 4B vindex, using the same prompt list (`bench/dec0/prompts.txt`) for continuity with DEC-0's calibration set in spirit, not by literal reuse.

## 3. Standalone checks — not gated on the ladder

### OBS-5 — K3 spectrum, laptop-only

Pull one layer's `W↑` and its paired RMSNorm γ from HF as plain tensors. Form `diag(γ)·W↑ᵀW↑·diag(γ)` (3584²) and eigendecompose locally. No inference, no GPU, no 1.5 TB download.

- **Kill:** a flat spectrum means the K3-specific version of this whole programme is dead, known before any of the [`k3-funnel.md`](k3-funnel.md) R1–R3 ladder spends a weekend.
- **No dependency.** Free, can run before OBS-0.

#### OBS-5 run — 2026-07-29

Ran against `moonshotai/Kimi-K3` (layers 3, 45, 90 of 93; `block_sparse_moe.routed_expert_up_proj.weight` / `routed_expert_norm.weight`, both plain BF16, no MXFP4 dequant needed), via HTTP range-GET (~49 MiB/layer, no full-checkpoint download). Condition number 55–77× at all three layers (not flat, kill bar doesn't trigger); participation ratio 61% of n at layer 3 → 76% at layer 45 → 93% at layer 90. Full write-up (v2, with the corrections below) and per-layer numbers on `obs-5-k3-spectrum-laptop-check` in the registry.

**Scope annotation, 2026-07-31 (from `dec8-0-k3-miss-budget`):** `routed_expert_up_proj` is the **layer-level** LatentMoE latent→hidden projection — `[7168, 3584]` BF16, 51.38 MB, **shared by all 896 experts and resident**. It is *not* the routed expert bank, and it is not the per-expert GLU up branch (`block_sparse_moe.experts.{i}.w3`, `[3072,1792]` MXFP4, 5.85 MB each). So OBS-5's anisotropy numbers describe the shared resident projection only — fine for what OBS-5 claimed (it never claimed otherwise), but **do not cite them as evidence about the expert bank's spectrum**. The expert-bank question was answered separately and negatively by `scripts/moe_expert_spectrum.py`.

**Correction, 2026-07-29 — the first write-up overclaimed on three points:**

1. **This measured a ceiling, not the operator that matters.** OBS-5 measured `G = diag(γ)·W↑ᵀW↑·diag(γ)` — the weight-space Gram alone. What actually governs quantisation distortion is G composed with the *distribution of activations feeding it* — schematically `E_u[J_RMS(u)ᵀ G J_RMS(u)]`. Anisotropy in quantisation typically comes from that data term, not the weight spectrum: GPTQ's Hessian is `E[xxᵀ]`, and those routinely run 10³–10⁵. A weight-only Gram at 55–77× is unremarkable in either direction once the data term multiplies in — it neither promises nor forecloses much, because it can't be measured without running the model, which OBS-5 deliberately didn't do (laptop-only, no inference). 55–77× is the honest ceiling of what a laptop check can tell you, not an estimate of the real quantity.
2. **The depth reading likely means the opposite of what the first write-up said.** At layer 90 (of 93), `W↑` *is* nearly the entire remaining observer — three layers of downstream transport left — so a near-flat G there means a genuinely near-flat observer. At layer 3, `W↑` is a small piece of an observer that includes 90 more layers of downstream transport, which is exactly where anisotropy tends to accumulate. So layer 3's 55× is a **lower bound** on something unmeasured and possibly much larger, while layer 90's 77× is close to **tight**. The three layers are not commensurable measurements, and "exploitable structure early, flat late" was the wrong read of that asymmetry.
3. **The "flat" calibration is scale-specific.** Condition-number/participation-ratio numbers here are calibrated on a *weight-operator* spectrum only. OBS-0/1's data-weighted metric on Gemma will differ by orders of magnitude (per point 1) — reusing this threshold there would silently mis-gate. Two named thresholds, not one: `obs5-weight-gram-flatness` (condition number in the tens + participation ratio >90% ⇒ practically flat, calibrated on K3's weight Gram only) and a separate `obs01-data-weighted-flatness` threshold, not yet defined, to be set once OBS-0/1 actually runs.

**Consequence:** OBS-0/1 proceeds unaffected — the kill criterion still didn't trigger. But OBS-5 doesn't locate *where* the exploitable structure is; it only bounds what a weight-only, no-inference check can see, and even that bound points opposite to the original write-up (the late-layer number is closer to real, the early-layer number is a floor on something larger, not evidence of more structure).

### OBS-6 — WalkFfn re-score (free reuse of banked data)

Existing WalkFfn results (see `project_walkffn_speed_accuracy_scissors` memory) were scored in cosine similarity: sparse top-256 walk fidelity 0.475 (filed as a fail), Q4K full walk 0.912, `--down-q4k` ~1.5% softmax drift, the Q8K wire arms. Re-score the same stored perturbations under OBS-1's `δᵀMδ` metric instead of cosine.

- **If it flips a verdict:** a result filed as a negative could be a pass — its error was concentrated in low-gain directions of M that cosine can't see. Feeds `dec-funnel.md`'s DEC-1B transport-compilation framing (per-layer min-bytes-under-KL) directly — same optimisation, network cost function bolted on. *(DEC-1B is not yet a registered experiment; this is a forward pointer, not a claim it exists.)*
- **Kill:** if the OBS-1-metric score doesn't diverge from the cosine verdict, the original WalkFfn negative result stands.
- **Dependency:** needs OBS-0/1's M-estimation code/artifact to exist, not a passing verdict — can run as soon as M is computable for the relevant layer.

## 4. Ordering

OBS-0/1 (merged) gates everything else in the numbered ladder — it is now the combined test of whether anisotropy exists *and* whether a candidate M predicts it, since a random-sampling stage can't validly separate those two questions in this many dimensions (§2 correction). OBS-2 and OBS-3 are independent of each other and run in parallel once OBS-0/1 passes. OBS-4 only if OBS-3 passes. OBS-5 has no dependency and is the cheapest thing in the document — run it whenever. OBS-6 needs only OBS-0/1's M-estimation machinery, not a passing verdict, and reuses data that already exists — also cheap, run early. The K3 slice proper (`k3-funnel.md`'s R1–R3) runs last, if ever, and only once OBS-4 has passed.

**Superseded by §2e for the immediate next action.** The paragraph above is the steady-state ladder ordering once OBS-0/1 is actually built; §2e's finding changes what to do *right now*, on resume: run OBS-2's arm B (plain GPTQ, no observer-metric) on the target layer with C6 first, before finishing M or building any arm-construction code. That measures the ceiling this whole thesis needs to beat, and OBS-3 has been redefined from a discovery step to the control that measures how much single-matrix error-cancellation GPTQ already gets for free — see §2e.

## 5. Relationship to other specs

| Document / programme | Owns | Relationship |
|---|---|---|
| [`dec-funnel.md`](dec-funnel.md) (programme `dec`) | transport: wire, batch curve, router, fleet | OBS-2/OBS-3 results are fidelity-at-fixed-bytes findings that could feed a future DEC-1B (transport compilation); explicitly kept out of DEC's capacity scoreboards (§2 caution) |
| [`k3-funnel.md`](k3-funnel.md) (programme `k3`) | the K3 model-side adapter ladder (R1–R3) | OBS-5 is the cheap pre-check on K3's specific observer matrix; OBS-4 passing is the gate before any of K3's own quantisation-allocation work is worth building |
| programme `j-space` (`js-1-basis-rotation-dark-space`, `js-2-sensitivity-in-jacobian-basis`) | Jacobian-lens / dark-space structure | OBS-0/1 reuses the same Jᵀ J transport object js-1/js-2 already fit, applied to a quantisation-drift prediction task instead of a dark-space-structure question — same instrument, different falsifier |
| this document | the observer-metric quantisation-allocation ladder itself | — |

## 6. Registry conventions

Programme **`quant-obs`**, created 2026-07-29. Originally seven planned experiments mirroring §2 and §3; revised same day after the OBS-0 design correction (§2):

| Slug | Covers | Status |
|---|---|---|
| ~~`obs-0-frobenius-anisotropy`~~ | superseded — random-sampling design, concentration-of-measure false-kill risk | `superseded`, see `obs-01-eigenbasis-drift-ordering` |
| ~~`obs-1-static-metric-predicts-drift`~~ | superseded — folded into the merged test | `superseded`, see `obs-01-eigenbasis-drift-ordering` |
| `obs-01-eigenbasis-drift-ordering` | §2 OBS-0/1 (merged) | `planned` |
| `obs-2-fixed-byte-allocation` | §2 OBS-2 | `planned`, dependency retargeted to `obs-01-eigenbasis-drift-ordering` |
| `obs-3-dense-error-cancellation` | §2 OBS-3 | `planned`, dependency retargeted to `obs-01-eigenbasis-drift-ordering` |
| `obs-4-routing-structure-value` | §2 OBS-4 | `planned` |
| `obs-5-k3-spectrum-laptop-check` | §3 OBS-5 | `completed`, writeup v2 carries the 2026-07-29 correction |
| `obs-6-walkffn-rescore` | §3 OBS-6 | `planned`, dependency retargeted to `obs-01-eigenbasis-drift-ordering`'s M-estimation code |
