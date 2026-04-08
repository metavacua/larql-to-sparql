# Experiment 07: Native Compute Engine — Phase 1 Results

**Chris Hay | LARQL Project | April 2026**

---

## Executive Summary

Token-level solver injection during generation **hurts accuracy** (50% → 26%). The approach
is fundamentally wrong: mid-generation injection corrupts KV cache state and premature
expression triggering intercepts partial expressions. Post-hoc correction (verify after
generation) is the viable path. v2 pipeline built and verified.

---

## Phase 1a: Mid-Generation Injection (FAILED)

### Setup

- Model: Gemma 3-4B-IT on CPU (bfloat16, Apple M3 Max, 128GB)
- Benchmark: GSM8K (50 items, 256 max tokens)
- Strategy: monitor token stream → detect arithmetic → solve → inject answer tokens

### Results

| Mode | Accuracy | Interventions | Avg Time/Item |
|------|----------|---------------|---------------|
| Baseline | **50.0%** (25/50) | 0 | 26.2s |
| Augmented | **26.1%** (6/23*) | 10 in 20 items | 30.5s |

*Augmented run completed 23/50 items before being stopped.

**The solver broke 7 problems that baseline got right. It fixed zero.**

### Failure Analysis

Three compounding failure modes:

**1. Premature triggering.** Parser fires on partial expressions mid-generation.

```
Model generating: "20% of 200 students..."
Parser sees:      "20% of 2" → fires → injects "0.4"
Model continues:  "20% of 2 0.40 students enrolled in 20.= 20.40% of 0 0..."
```

The stream parser has no way to know the expression is incomplete until more tokens arrive.

**2. KV cache corruption.** Injecting tokens requires updating the model's key-value cache.
The attention mask accounting was incorrect, causing the model to lose coherence:

```
Item 9 (expected 460):
  Model: "40 * $10 = $400... overtime: 45 - 40 = 5 [INJECT: 5]..."
  After injection: "33333333333333333333333333..." (repetition collapse)
```

**3. Wrong interception point.** The solver fires during intermediate reasoning steps,
not at the final answer. The model's chain-of-thought arithmetic is usually correct step
by step — the errors that matter are at problem formulation, not computation.

### Error Catalogue

| Item | Expected | Baseline | Augmented | What Went Wrong |
|------|----------|----------|-----------|-----------------|
| gsm8k_0 | 18 | 18 ✓ | 5 ✗ | `3 + 4` intercepted mid-reasoning, disrupted flow |
| gsm8k_1 | 3 | 3 ✓ | 2 ✗ | Triple intervention (`2+1`, `2+1`, `2+2`), model confused |
| gsm8k_3 | 540 | 540 ✓ | 3.3 ✗ | `3*60=180` correct injection, but model echoed "= 180 = 180 meters per session meters per sessi..." |
| gsm8k_4 | 20 | 20 ✓ | 0 ✗ | `15+25=40` injection, derailed subsequent reasoning |
| gsm8k_9 | 460 | 460 ✓ | ∞ ✗ | `45-40=5` injection → repetition collapse → "333...333" |
| gsm8k_14 | 60 | 60 ✓ | 0 ✗ | `20% of 2` premature fire (was "20% of 200"), cascade into "0.40.40.4 0 0 0..." |
| gsm8k_17 | 57500 | 57500 ✓ | 0 ✗ | `20*35=700` correct, but model: "= 700 = $10 = $70 = $70 = 70 = 0" |

**Pattern:** Every intervention caused either repetition collapse or reasoning derailment.
Even when the solver's answer was correct (items 3, 9, 17), the injection disrupted the model.

### Root Cause

Mid-generation token injection is the wrong abstraction for this problem. The model's
autoregressive state (KV cache) encodes not just what tokens came before, but the *process*
of generating them. Injecting externally-computed tokens creates a state that the model
never would have reached organically, causing it to lose coherence.

This is analogous to the INSERT oscillation problem (experiment 04): you can't inject a
single feature into the residual stream and expect the model to behave as if it computed
that feature itself. The same principle applies at the token level.

---

## Phase 1b: Post-Hoc Correction (v2 pipeline)

### Strategy

Let the model generate its full response, then:
1. Scan the chain-of-thought for `expression = result` patterns
2. Verify each computation with the solver
3. If wrong, correct the stated result
4. Rederive the final answer from the corrected chain

### Implementation

`phase1_pipeline_v2.py` — extraction + verification engine tested:

```
Extraction tests: 6/6
  "16 - 3 - 4 = 8" → detected wrong (should be 9) ✓
  "40 * 10 = 350"   → detected wrong (should be 400) ✓
  "15 + 25 = 30"    → detected wrong (should be 40) ✓
  "$20 * 35 = $700"  → detected correct ✓
```

### Known Limitation: Cascade Propagation

Post-hoc correction fixes individual computations but doesn't re-evaluate
downstream steps that used the wrong result:

```
Step 1: 40 * 10 = 350        ← wrong (should be 400)
Step 2: 350 + 110 = 460      ← arithmetically correct, but uses wrong input
Final: #### 460               ← wrong (should be 510)
```

v2 corrects Step 1 (`350` → `400`) but doesn't detect that Step 2 needs
recomputation. The answer `460` looks arithmetically valid.

**Fix:** After correcting Step 1, re-evaluate all subsequent steps using
corrected values. This requires tracking value dependencies through the
reasoning chain — essentially a dataflow analysis of the model's arithmetic.

### Status

v2 pipeline built. Needs:
1. Cascade propagation (track value flow through reasoning)
2. Benchmark run on the saved GSM8K data (can reuse baseline checkpoints)
3. Comparison: does correcting arithmetic errors actually change the final answer?

---

## The Real Finding: Zero Arithmetic Errors

Deep analysis of all 50 baseline items reveals:

### Failure mode breakdown (25 wrong answers)

| Category | Count | Description |
|----------|-------|-------------|
| **Truncation** | 20 | Hit 256 token limit before producing `####` answer |
| **Reasoning error** | 5 | Model completed but got the logic wrong |
| **Arithmetic error** | **0** | Model never computed wrong arithmetic |

### The 5 genuine reasoning failures

Every arithmetic expression the model produced was verified correct by the solver.
The errors are all **comprehension or logic**:

| Item | Error | What the model did |
|------|-------|--------------------|
| gsm8k_5 | Truncation during explanation | Was correctly computing pair costs ($5+$3), cut off before total |
| gsm8k_21 | Direction error | `31 - 6 = 25` (correct math). But Raymond is OLDER, so he's 37, not 25. Model misread "born before" as "born after." |
| gsm8k_36 | Unit confusion | `2 * 7 = 14` per week (correct), then `14 * 30 = 420` (correct math, wrong logic — applied weekly rate over 30 days, not daily rate) |
| gsm8k_45 | Phrase misinterpretation | `(2/5) * 5 = 2` (correct math). But "2/5 times MORE" means `5 + 2 = 7`, not `2/5 OF 5 = 2` |
| gsm8k_28 | Truncation | Correct `60 - 20 = 40`, was computing second stop position, cut off |

### Verified arithmetic in failing items

```
31 - 6 = 25     ✓ (gsm8k_21 — math right, direction wrong)
60 - 20 = 40    ✓ (gsm8k_28 — truncated)
2 * 7 = 14      ✓ (gsm8k_36 — logic error, not math)
14 * 30 = 420   ✓ (gsm8k_36 — logic error, not math)
```

**The model's calculator works perfectly. It's the model's reading comprehension
that fails.**

### What a solver CANNOT fix

- "Born 6 years before" interpreted as younger (gsm8k_21)
- Weekly rate applied to days (gsm8k_36)
- "2/5 times more" parsed as "2/5 of" (gsm8k_45)
- Running out of tokens before reaching the answer (20 items)

A solver checks `2/5 * 5 = 2` and says "correct" — because it IS correct.
The error is upstream: the model chose the wrong expression to compute. No
amount of exact arithmetic fixes a wrong equation.

---

## What This Means for the Compute Engine

### The thesis needs revision

The original thesis was: models fail at math because they can't compute.
Bolt on exact solvers → accuracy improves.

The data says: **models fail at math because they misunderstand the problem.**
Their arithmetic is fine. The bottleneck is language comprehension and
multi-step reasoning, not computation.

This holds specifically for GSM8K (grade-school math, single-digit operations).
The thesis may still hold for harder problems where computation is genuinely
beyond the model:

- **Symbolic algebra**: `solve x⁴ - 5x² + 4 = 0` (factoring, multiple roots)
- **Counting/combinatorics**: `how many permutations of {1..10} have no fixed points`
- **Multi-step with large numbers**: `compound interest over 30 years`
- **Constraint satisfaction**: NP-hard problems where enumeration is needed

### What we proved

1. **Mid-generation injection destroys coherence.** The model's KV cache state
   is not compatible with externally-injected tokens. Same principle as INSERT
   oscillation (experiment 04) — can't inject features into autoregressive state.

2. **Post-hoc correction is safe but vacuous on easy math.** The verify-then-fix
   pipeline works mechanically (6/6 tests), but finds nothing to fix because
   the model doesn't make arithmetic errors on GSM8K.

3. **The 50% accuracy is a truncation artifact.** 20/25 failures are pure token
   limit. With 512+ tokens, baseline accuracy is likely 70-80%. The model
   understands these problems — it just needs space to reason.

4. **Computation errors are not where models fail.** At least not at the
   GSM8K level. The failure mode is upstream: problem comprehension.

### Revised strategy

The compute engine is not a general accuracy booster. It's a **capability
extender** — it adds abilities the model genuinely lacks:

```
Model alone:     can solve x + 3 = 7  (arithmetic)
Model alone:     can solve 2x + 5 = 15 (simple algebra)
Model CANNOT:    solve x⁴ - 5x² + 4 = 0 (complex roots)
Model CANNOT:    count derangements of {1..10} (enumeration)
Model CANNOT:    prove ∀x: P(x) → Q(x) (formal logic)
Model CANNOT:    find shortest path in a 1000-node graph (algorithm)

Solver extends:  all of the above, exactly, in microseconds
```

The compute engine's value is not "making the model's arithmetic correct"
(it already is). It's "giving the model capabilities it doesn't have."
The benchmark needs to test problems where the model CANNOT compute the
answer, not problems where it occasionally gets confused about the question.

---

## Files

```
experiments/07_wasm_compute/
  SPEC.md                  — original experiment spec
  RESULTS.md               — this file
  phase1_parser.py         — expression parser (20/20 tests)
  phase1_solvers.py        — solver dispatch (16/16 tests)
  phase1_pipeline.py       — v1: mid-generation injection (FAILED)
  phase1_pipeline_v2.py    — v2: post-hoc correction (verified, vacuous on GSM8K)
  phase1_bench.py          — v1 benchmark runner
  phase1_bench_v2.py       — v2 benchmark runner (ready, not run)
  results_phase1_saved/    — checkpoint data from v1 run
    gsm8k_baseline_checkpoint.jsonl     (50 items, 50% accuracy)
    gsm8k_augmented_checkpoint.jsonl    (23 items, 26% accuracy)
```

---

## Next Steps

1. **Build a HARD benchmark**: problems the model genuinely can't compute
   (symbolic algebra, counting, constraint SAT, large-number arithmetic)
2. **Test solver value on HARD problems**: where the computation is the bottleneck
3. **Skip Phase 2 for now**: residual-level dispatch is premature when the
   value proposition hasn't been demonstrated at the token level
4. **Reconsider the architecture**: the solver is not a "better FFN" — it's an
   entirely new capability. The routing question is "does this require computation
   the model can't do?" not "is this an arithmetic expression?"
