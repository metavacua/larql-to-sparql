# `larql confidence` (slot-bits) — Specification

**Status:** Draft.
**Source experiments:** Shannon Exp 30 (cross-entropy baseline),
Exp 32 (template vs slot bits), Exp 33 (resolution map + surface-alias
correction).
**Crates touched:** `larql-cli` (new top-level subcommand), `larql-inference`
(read-only consumer of forward-pass logits), `larql-lql` (optional new
function — see §6).

## 1. Purpose

Expose Shannon-arc slot bits as a label-free retrieval-confidence score for
LARQL's "model-as-database" surface. Given a prompt and an answer, the
command returns:

```
bits[i] = -log2 p( answer_token[i] | prompt + answer_token[<i] )
```

The aggregate is the cost (in bits) of the prompt narrowing the model's
distribution onto the supplied answer. Low bits = confidently retrieved.

Exp 32 calibration on Gemma 3 4B:

```
geography/capitals     slot mean = 3.53 bits/slot
chemistry/formulas     slot mean = 2.81 bits/slot
authors                slot mean = 5.77 bits/slot   (broader, less canonical)
template ambient mean  slot mean = 4.15 bits/slot   (across categories)
```

This is **not** a correctness oracle. Exp 32 explicitly confirmed wrong-but-
confident is reachable. The score measures *how confidently the prefix
selects the supplied answer*, not whether the answer is true. That
distinction is part of the spec.

This `confidence` is distinct from per-edge extraction confidence
(`docs/confidence.md`); the latter scores `c_in × c_out` for edges in the
vindex, the former scores `-log2 p(answer | prompt)` for a live forward
pass. Same word, different unit. The spec uses **"slot bits"** internally and
"confidence" only on the user-facing CLI surface.

## 2. CLI

```
larql confidence <vindex>                            \
    --prompt   "The capital of France is"            \
    --answer   " Paris"                              \
   [--surface-aliases " Paris"," PARIS","Paris"]    \
   [--report   text|json]                           \
   [--baseline none|template-completion]
```

Behaviour:

1. Tokenise `prompt + answer` using the model's tokenizer.
2. Run forward (single pass).
3. Score every token in `answer`:

   ```
   bits[i] = -log2 softmax(logits[prompt_end + i - 1])[answer_token[i]]
   ```

4. Emit per-token bits, sum, and the aggregate label.

### 2.1 Output (text)

```
prompt          : "The capital of France is"
answer          : " Paris"           (1 token: " Paris")
total bits      : 0.62
contract        : low (< 3.5 bits — confidently retrieved)
per-token bits  : [0.62]
surface aliases : 1 used (" Paris" matched first)
```

### 2.2 Contracts (label thresholds)

```
total/N < 1.0   :  very-low  (one-token cost of an in-distribution next word)
1.0 ≤ x < 3.5   :  low       (canonical fact territory; capitals 3.53)
3.5 ≤ x < 6.0   :  moderate  (less canonical; multi-surface answers)
6.0 ≤ x         :  high      (no clear retrieval; treat as not-indexed)
```

These bands are reported, not enforced. The command always exits 0 on
successful inference.

## 3. Surface-alias handling (Exp 33 correction)

Exp 33's strongest-looking finding ("late-stack `work_author` override")
mostly evaporated under valid-surface controls. The discipline is part of
the spec:

- The user may supply alternative surface forms via `--surface-aliases`.
- The command scores each surface form independently, then reports the
  cheapest. The reported `bits` is **min over surfaces**, with the chosen
  surface noted.
- If `--surface-aliases` is omitted, the command checks the tokenizer for
  obvious leading-space variants of the supplied answer (`"Paris"` vs
  `" Paris"`) and reports if a different surface would have been
  cheaper, but does not silently switch.

This is a small amount of work that prevents the "Flaubert vs Gustave
Flaubert" failure mode the experiment surfaced.

## 4. Optional template baseline

`--baseline template-completion` runs the prompt without the slot
("The capital of France is …"), measuring how the model would have
completed it freely, and reports:

```
slot bits      : 0.62                # supplied answer
template top-1 : " Paris"  (0.59 bits)
delta          : +0.03 bits
agreement      : YES — supplied answer matches the model's top-1
```

This is the "is the supplied answer also the model's preferred answer"
control. Useful but not the headline number; defaults to `--baseline none`.

## 5. JSON output

```
{
  "version": "slot-bits-spec/1",
  "prompt": "...",
  "answer": "...",
  "total_bits": 0.62,
  "per_token_bits": [0.62],
  "surface": " Paris",
  "surface_aliases_tried": [" Paris", "Paris"],
  "label": "low",
  "baseline": null
}
```

`--baseline template-completion` adds a `baseline` object with `top1`,
`top1_bits`, `delta_bits`, `agreement` (`true | false`).

Schema is versioned. Future fields may be added; existing fields will not
silently change shape.

## 6. LQL surface (optional, deferred)

A natural extension is exposing slot bits as an LQL expression:

```sql
SELECT CONFIDENCE("The capital of France is", " Paris");
-- 0.62
```

This is **deferred** until the CLI lands. The CLI is the primary surface;
the LQL function is a thin wrapper over the same `larql-inference`
entrypoint and adds parser/AST work in `larql-lql`. Not a blocker for v1.

## 7. Determinism

Two runs of the same `(vindex, prompt, answer, surface-aliases)` produce
byte-identical output, including per-token bits. The command does not
sample, does not use temperature, and does not mutate the vindex.

The forward pass uses the production decode path. Prefill and decode share
the same numerical path; this is enforced by reusing `larql-inference`'s
existing decode entry point with `seq_len = len(prompt + answer)`.

## 8. Out of scope

- **Multi-token templates with internal computation** (`"5 + 7 ="` →
  scoring `"12"`). The slot-bits unit still makes sense, but
  template-vs-slot is no longer the right partition. Use
  `larql confidence` if the answer is a contiguous trailing surface;
  otherwise reach for `larql dev fidelity`.
- **Slot bits inside `INFER` output.** Worth doing later. Not in v1; keeps
  the public CLI envelope minimal.
- **Cross-model comparisons.** Slot bits depend on the model's vocab and
  tokenizer; comparing capitals' bits across Gemma 3 4B and Llama 2 7B is
  meaningful, but the spec does not invent a normalisation. Users compare
  apples to apples.
- **Truth verification.** §1: confidence is not correctness.

## 9. Test plan

```
1. Sanity: known canonical fact ("The capital of France is" / " Paris")
   on gemma-3-4b reports total_bits within ±0.5 of the Exp 32 capitals
   mean (3.53), per-token vector matches manually computed -log2 p.
2. Determinism: two runs of the same args produce identical JSON.
3. Surface-alias: supplying a worse surface as primary (e.g. "PARIS")
   plus a better one in --surface-aliases reports the cheaper surface
   and notes the substitution.
4. Template baseline: --baseline template-completion on the same
   capital fact reports agreement=true and small delta (< ±0.2 bits)
   on capitals; agreement=false on a deliberately mis-supplied answer.
5. Wrong-but-confident: a known model-confident hallucination reports
   low bits — this is expected and asserts §1 (confidence ≠ truth).
6. Empty answer: errors with a clear message; does not score the prompt
   tokens.
```

## 10. References

- `~/chris-source/chris-experiments/shannon/30_shannon/results.md`
- `~/chris-source/chris-experiments/shannon/32_template_slot_bits/results.md`
- `~/chris-source/chris-experiments/shannon/33_resolution_map/CORRECTION.md`
- `docs/confidence.md` (per-edge extraction confidence — distinct unit,
  noted in §1 to avoid name confusion)
- Memory: `project_shannon_experiments` — Gemma 3 4B at ~1.02 bits/char
  on Frankenstein, slot-bits tightness 4× gzip on the 4KB closure.
