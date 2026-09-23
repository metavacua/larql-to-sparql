# LARQL SPEC — Arithmetic Virtual Expert (AVE)

**Status:** draft v0.1 (2026-06-11). **Scope:** larql-rs runtime component.
**Evidence base:** arithmetic_mechanism arc A0–A10 + A9b, all numbers cite frozen
pre-registered runs on Gemma-3-4b-it. **Claim discipline:** every parameter below is
tagged MEASURED / DERIVED / ASSUMED / OPEN.

---

## 1. Design principle

The model is an I/O system, not a calculator (A0–A6). It supplies: tokenization-level
digit decomposition, a causally verified number format (per-digit mod-10 wheels, A2c/A2e),
an involuntary engagement signal (A7), perfect operand extraction (A8), a magnitude prior
(A4c/A5), and a fluent readout (A9/A9b). It structurally cannot supply the serial
algorithm (bounded-depth, A4e/A5). Therefore:

> **Fired ⇒ dispatch, always.** No length threshold. A8 measured native never winning
> once surface form is uncontrolled (template-fragility 0.58–0.67 at sizes where one
> template scored 0.93); A10 measured dispatch ≥ native in every cell at equal-or-known
> token cost. The model's own arithmetic output is consumed only as a verification prior.
>
> Scale note (A14, 12B): the capability WALLS are per-model — at 12B the carry wall moved
> ~7–8 → ~13–16 digits and the magnitude wall ~24–28 → 28–32, while exact-long-random
> stayed ~0 at both scales. This does NOT touch this policy: "fired ⇒ dispatch, always" is
> justified by A8's template-fragility (surface-form), which was never length-based. Any
> future length-aware variant must re-derive its thresholds on the host model.

This component is also **instance #1 of the VirtualExpert trait** (§8) — the gate /
extract / compute / drive / verify decomposition is intended to be reused by future
experts (dates, units, sorting) pending the exhaust-generality result.

## 2. Placement in the workspace

```
larql-inference/
  src/experts/
    mod.rs            // VirtualExpert trait + ExpertController
    arith/
      mod.rs          // AVE: wiring + state machine
      gate.rs         // tier-0 symbolic scan + tier-1 L8 probe
      extract.rs      // symbolic parser + rewrite fallback
      alu.rs          // exact compute (BigInt)
      drive.rs        // forced-decode schedule (+ injection hook)
      verify.rs       // magnitude-prior check
      probe_weights/  // ridge probe artifacts (versioned, per-model)
```

Forward-pass hooks required (larql-compute / larql-models):
- `residual_tap(layer, position) -> &[f32]` — read-only capture at L8, last prompt token.
- `logit_override(step) -> Option<TokenId>` — sampler-level forcing (default drive path).
- `residual_inject(layer, position, vec, lambda)` — optional; reserved (§5.2).
- `terminate_at(step)` — controller-owned generation stop.

All four are trivial given larql-rs owns the pass; none exist over a token API.
**Stack-relative note (A9b):** for digit payloads the *outcome* is replicable via
constrained decoding anywhere; what forward-pass ownership uniquely buys is the gate tap,
non-token-aligned payloads (Lazarus convergence), and conditioning-without-committing.

## 3. Gate

Two tiers, evaluated during the prompt forward pass (which runs anyway — the tap is a
free read, not a 0.24-forward surcharge; the 0.24 framing only applies if an early-exit
dispatch skips the remaining layers, which is an optional optimization, OPEN).

**Tier 0 — symbolic (explicit math notation).** Scanner over the prompt surface for
digit chains joined by math notation. Cost ~0. MEASURED: fire 1.0, extraction
downstream 1.0 (A10 bare cells). **Scope rule (larql-rs v0.1, adversarial-prose
measured):** tier-0 fires on *notation*, never on inferred intent — strong glyphs
(`+`, `*`, `×`, `−`) fire anywhere; ambiguous prose operators (spaced `-`, standalone
`x` — ranges, scores, shifts, dimensions, spaced dates) fire only with an explicit
trailing `=`. Everything else is the designed fallthrough: deciding whether "9 - 5"
is arithmetic is an engagement question and belongs to the model (tier-1 exhaust, or
an FR3-style explicit classify), not to surface heuristics. Adversarial prose corpus:
0 false fires (`chris-experiments/larql_probes/examples/misc/scanner_adversarial.rs`).

**Tier 1 — engagement probe (disguised math). DEMOTED (A11).** Ridge probe on the L8
residual at the last prompt token, reading arithmetic-engagement exhaust (math vs
numbers-present, template-held-out 0.91–0.99, A7b).
- MEASURED specificity: 1.00 (0/18 + 0/48 false fires across A8/A10, incl. long-number
  no-op controls). A fire is always trustworthy.
- MEASURED sensitivity: uneven — 1.0 on sub/mul/multi phrasings, 0.17–0.58 on novel add
  phrasings (A8/A10).
- **A11 demotion:** the probe is parked as an *audit instrument*, not a gate component;
  the gate-hardening workstream is deleted. v0.1 gates on tier-0 only. Disguised-math
  coverage waits on the exhaust-generality instrument science (§8 OPEN) — not on probe
  retraining. The artifact format is retained for audit use
  (`probe_weights/README.md`); weights remain per-checkpoint artifacts if/when refit.

**Policy:** Tier0 fire ⇒ dispatch. No fire ⇒ native path untouched (zero overhead
beyond the tap). **No fire on disguised math is the designed fallthrough, not a
coverage gap:** the §7 decomposition `fleet = fire + (1−fire)·native` makes native the
floor — a silent gate costs exactly nothing relative to not having the expert, and the
dispatch architecture loses nothing when the probe never fires.

## 4. Extract

**Explicit path:** symbolic parse of the operand digit spans and operator(s) from the
token stream. Exact by construction, zero tokens.

**Disguised path:** 2-shot rewrite prompt → parse the emitted expression.
MEASURED: extraction 1.00 in every cell — 16-digit operands, mul, 2-op chains, with an
*untuned* prompt (A8); held in-pipeline (A10: extract = 1.00 of fired, all kinds).
Cost ~2× tokens of native on the rewritten segment. The regex reads the model's
expression, never its sum (rigging-proofed by design, A8).
- OPEN: structured-output extraction (JSON-constrained decode) should beat 2-shot on
  token cost; the 2-shot number is the measured floor.

**Failure handling:** unparseable rewrite ⇒ fall to native, flag `extract_miss`.
MEASURED rate at floor prompt: 0.

## 5. Compute + Drive (return path)

**ALU:** Rust-native exact integer arithmetic (`i128` fast path, BigInt beyond).
Latency ~0 relative to a decode step. Ops in scope v0.1: +, −, ×, integer chains of the
A8 shapes. Division, decimals, negatives: OPEN (extraction for them unmeasured).

### 5.1 Default drive: forced decode
Controller forces the answer token sequence at the sampler, then **terminates at
schedule end**.
- DERIVED from A9b: logit bias β=10 ≅ L30 injection ≅ constrained decoding,
  behaviorally, on greedy. Forcing is the cheapest equivalent and larql-rs owns the
  sampler.
- Schedule-end termination is MANDATORY: it eliminates the one observed delivery defect
  — post-schedule digit continuation, ~4% per-item (129/135 ≈ 0.96 delivery without it,
  A10 correction; mode caught in a logged diagnostic: full correct answer + one extra
  digit). With termination, delivery = 1.0 **by construction**; "the model terminates on
  its own" is demoted from claimed property (~0.96) to unneeded one.
- Forced tokens enter the KV cache normally; MEASURED: the model stays coherent
  conditioned on supplied digits (A9 clean termination, A10 word-continuation cells).

### 5.2 Reserved drive: residual injection
`λ·‖h‖·û(digit)` per decode step. MEASURED: drives 1.00 at any site ≥L16 during
emission, λ clean to 0.25, graded threshold ≈0.1 (A9b); the defended band is defended
only while *computing* (prompt step), not while emitting — the phase map.
Kept as the general mechanism because it is the same operation as Lazarus fact injection
(shared splice infrastructure) and supports conditioning-without-committing
(bias-without-force), which the sampler path cannot express. Not used for digits in v0.1.
- OPEN: emission-commandability lower bound (<L16 untested); per-site λ floor
  (floor swept at L30 only — a weak per-step fight at L16 masked by λ≥0.5 is not excluded).

### 5.3 Token accounting
Explicit path: fleet tokens == native tokens, MEASURED at every length (A10: 26.2 vs
26.2 at 24 digits — the forced answer rides the tokens the model was emitting anyway).
Disguised path: ~2× on the rewrite segment (A8/A10); structured extraction expected to
reduce this (OPEN).

## 6. Verify

The native estimator is retained as a **magnitude prior**, nothing more:
- MEASURED envelope: near-misses magnitude-correct to ~±25–35% through ~24 digits;
  **collapses ≥~28 digits** (wrong digit count, wrong lead, rel-err ~0.6, A5).
- Use: after extraction, compare ALU result's magnitude (digit count, leading digit)
  against the model's native answer *if one was produced*, or skip. Mismatch ⇒ flag
  `extract_suspect`, re-extract once, surface on second failure.
- HARD RULE: the prior is void past 24-digit operands (A5 magnitude wall, **4B**; at 12B
  the wall measured 28–32, A14 — the void threshold is PER-MODEL, re-derive on host).
  Never gate a correctness decision on it; it is a tripwire for extraction bugs only.
- v0.1 status: not exercised in any assembly run (A10 pre-registered it out) — wire it
  but treat its thresholds as ASSUMED until an assembly increment measures the
  false-flag rate.

## 7. State machine

```
IDLE → (prompt pass; tap L8)
  ├─ no fire ──────────────→ NATIVE (untouched)
  └─ fire (T0|T1) → EXTRACT
        ├─ symbolic ok ────→ COMPUTE → DRIVE(forced, schedule) → TERMINATE → VERIFY? → IDLE
        ├─ rewrite ok ─────→ COMPUTE → DRIVE → TERMINATE → VERIFY? → IDLE
        └─ extract miss ───→ NATIVE + flag
```

Telemetry (mandatory — the A10 lesson): per-item logs of fire tier, extracted
expression, ALU result, full emitted string, termination cause. *Per-item logging would
have made the word_16 correction a grep instead of a rerun.* Every counter in the table
below is a mutual-consistency check (fire × extract floors fleet accuracy); the
controller should assert the decomposition `fleet ≈ fire + (1−fire)·native` per batch
and alarm on violation — table arithmetic is a control surface.

## 8. VirtualExpert trait (forward-compatible)

```rust
pub trait VirtualExpert {
    fn gate(&self, tap: &ResidualTap, tokens: &[TokenId]) -> Fire;   // exhaust, not intent
    fn extract(&self, ctx: &GenCtx) -> Result<Payload, ExtractMiss>;
    fn compute(&self, p: &Payload) -> Answer;                        // external, exact
    fn drive(&self, a: &Answer) -> DriveSchedule;                    // forced-decode default
    fn verify(&self, a: &Answer, native: Option<&str>) -> Verdict;   // prior, not judge
}
```

Design constraints baked in from the arc: the gate reads **exhaust, not intent** (A7:
no abstract op object exists to read; the engagement signal is involuntary, cannot be
prompted away, and needs no MoE router); the expert is **invisible to the model** (no
weights touched, no model routing used); compute is **never** the model's.
- OPEN (the fleet's gating science question): exhaust generality — whether dates/units/
  sorting emit separable engagement signatures or one shared "bounded computation
  straining" signal. Determines whether `gate()` is per-expert or shared infrastructure.

## 9. Measured-parameter table

| parameter | value | status | source |
|---|---|---|---|
| probe layer / site | L8, last prompt token | MEASURED | A7b |
| probe arch | ridge, λ ∝ mean feature norm | MEASURED | A7b |
| gate specificity | 1.00 (0/66 false fires) | MEASURED | A8+A10 |
| gate sensitivity (current weights) | 0.17–1.0 by phrasing | MEASURED; probe DEMOTED to audit instrument | A8/A10 + A11 |
| extraction (2-shot floor) | 1.00 all cells | MEASURED | A8/A10 |
| drive equivalence (bias≅inject≅force) | β=10 / λ≥0.25 / forced | MEASURED (greedy) | A9b |
| λ floor / threshold (L30) | clean 0.25 / graded ≈0.1 | MEASURED | A9b |
| emission-commandable sites | ≥L16 (lower bound open) | MEASURED | A9b |
| delivery w/o termination | 129/135 ≈ 0.96 (one mode: +1 digit) | MEASURED | A10+corr |
| delivery w/ schedule termination | 1.0 by construction | DERIVED | A10 corr |
| explicit-path token overhead | 0 | MEASURED | A10 |
| disguised-path token overhead | ~2× (rewrite floor) | MEASURED | A8/A10 |
| estimator prior envelope | ±25–35% to 24 digits; void ≥28 | MEASURED | A4c/A5 |
| end-to-end demo | 24-digit add 0.92 vs native 0.00, equal tokens | MEASURED | A10 |
| forced decode, Metal Q4_K | 6/6 exact, schedule-end 6/6, ~20 tok/s e2e | MEASURED | larql-rs 2026-06-12 |

## 10. Out of scope / risks

1. **Model-version coupling.** Probe weights, L8/L16/L30 sites, and the phase map are
   Gemma-3-4b measurements. The *relative-depth* framing (L8 ≈ 24%, emission-commandable
   ≥ ~47%) is the porting hypothesis (depth-fraction routing law), ASSUMED until the
   12B/other-family run. Ship probes as per-checkpoint artifacts; treat sites as
   fractions with a calibration pass per model.
2. **Sampling.** All drive equivalences are greedy-measured. Under temperature, forced
   decode is unaffected by construction; injection/bias equivalence is OPEN
   distributionally.
3. **Op coverage.** Return path measured for addition; extraction measured for +,−,×,
   2-op chains. Division, decimals, negatives, mixed text-number answers: OPEN.
4. **Gate hardening — workstream DELETED (A11).** Pre-A11 this read as "the single
   component standing between current and ~1.0 disguised accuracy." Post-A11 the probe
   is an audit instrument and disguised coverage is parked behind the exhaust-generality
   instrument science. The explicit path — the measured-1.0 path — is the product
   surface; native is the designed fallthrough for everything else.
5. **Quantization.** Arc measurements bf16/MLX. Forced decode under Q4_K on the Metal
   pipeline: **MEASURED** (2026-06-12, larql-rs assembly run): AT-1 6/6 exact with
   schedule-end termination through the backend-routed constrained path, 345 ms–1.26 s
   per item incl. prefill (24-digit add = 25 forced tokens at ~20 tok/s end-to-end) —
   `bench/aim-validation/ave_demo_gemma3-4b.json`, `ave_demo --metal`. The
   sampler-level argument held: the mask applies to CPU-resident logits, so the drive
   is backend- and quantization-independent by construction *and now by measurement*.
   Probe and injection paths still need a re-calibration run if revived.

## 11. Acceptance tests (assembly increments)

- AT-1: A10 suite rerun in larql-rs, forced-decode drive + termination ⇒ explicit ≥0.99,
  zero post-schedule continuations, tokens == native.
- AT-2: distractor set ×10 size ⇒ false fires = 0.
- AT-3: hardened gate ⇒ disguised single-step fire ≥0.9 (the A8 bar), specificity intact.
- AT-4: verify leg ⇒ injected extraction faults (swapped operands) caught at the
  measured prior envelope; false-flag rate <2%.
- AT-5: per-item telemetry replays the word_16-class consistency check automatically.
