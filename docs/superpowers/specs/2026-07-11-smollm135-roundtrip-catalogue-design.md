# SmolLM2-135M Round-Trip Difference Catalogue — Design

> **Status:** design, approved 2026-07-11. Branch: `smollm135-roundtrip-catalogue`
> (isolated, off `lql-strategy-matrix` HEAD). Next: implementation plan
> (`superpowers:writing-plans`).

## 1. Purpose

Extend the CI's observable algebra with a **model → vindex → model round-trip
difference catalogue** for SmolLM2-135M. For every produced output model, compare
it to the input model and **catalogue, classify, and categorize every
difference**, attributing each to its code-level cause.

This is a **descriptive experiment, not a validation suite.** It imposes no
correctness oracle beyond faithful description of divergence. It does **not**
reward-hack, canonicalize-to-match, or otherwise obscure a difference to make a
result look clean: if input and output are byte-identical we report that; if they
differ, we report exactly *where*, *why*, and *how much*, in fine detail.
Canonicalization is used only to **classify** cosmetic-vs-semantic divergence,
never to normalize a difference away.

The posture is **measure-and-document**. The CI runs on the assumption that
`larql` behaves as intended; wherever reality deviates, the catalogue **records**
it (cross-referenced to existing issues) rather than fixing it. Fixes are out of
scope; documented deviations become candidate issues.

## 2. Definitions (load-bearing)

- **Round-trip (byte-identity sense):** `fixed-format model → vindex → the SAME
  fixed-format model`. Byte-identity is **only** expected when the output format
  equals the input format. Going from a base model to an Instruct variant, to
  GGUF, or across any format/dtype boundary is **not a round-trip** under this
  definition and is **not** expected to be byte-identical — it is a
  *format-transform*, and byte-identity is N/A for it by definition.
- **A (passthrough):** `extract → compile with ZERO edits → compare`. Output
  *should* equal input (for a true round-trip). Every delta is unintended loss.
- **B (edit-bearing):** `extract → INSERT an edge → compile → compare`.
  Differences are expected — but conditioned on the INSERT form (§5).
- **Driver:** the model-producing surface — **LQL** (`COMPILE … INTO MODEL …
  FORMAT safetensors`) or the **`larql compile` CLI** (`--base … → model.safetensors`).

## 3. Scope

- **Models (boundary "A"):** the two **official** `HuggingFaceTB` SmolLM2-135M
  safetensors models — no others. GGUF and third-party re-quantizations are out
  of scope (there is no official 135M GGUF; and GGUF-in/safetensors-out can never
  be byte-identical, so it is not a round-trip under §2).
  - `HuggingFaceTB/SmolLM2-135M` — base, **F32**.
  - `HuggingFaceTB/SmolLM2-135M-Instruct` — **BF16** (already the `smol135` entry
    in `gen_legs.py` / `model-lifecycle`).
- **Deactivated for now:** the BitNet cluster (`bitnet2b`, `bitnetgguf`) and the
  other models (`qwen05`, `smol360`, `qwen15`, `granite1b`). Deactivation is a
  **data toggle** (§6.1), not structural surgery — re-enabling a model is a
  one-line change so the CI "just works" when models are added back.
- **Out of scope (named, not built):** a **vindex-level fidelity comparison**
  (source tensors vs the vindex's own tensor files). See §9 / N1 — the round-trip
  as built does **not** test whether the vindex stored the weights, because
  `compile` re-reads the base model. This is explicit future work.

## 4. Per-variant expected outcomes (the core insight)

`larql compile` unconditionally emits **bf16** (`write_safetensors` →
`encode_bf16`, no output-dtype flag on CLI `CompileArgs` or LQL `FORMAT`). Applied
to §2's definition, the two variants become **two different experiments**:

| Variant | In → out format | Round-trip? | Expectation | Role |
|---|---|---|---|---|
| **Instruct (BF16)** | bf16 → bf16 | **Yes** (format preserved) | **Byte-identity** — every delta is a **defect** | The byte-identity oracle |
| **Base (F32)** | F32 → bf16 | **No** (format broken by forced-bf16) | Byte-identity **not** expected | Documents the **absence** of a format-preserving round-trip |

- **Instruct/BF16** is the true byte-identity test. Deltas — Gemma3 config-stamp,
  tensor-order-vs-source, `__metadata__` loss, `lm_head` drop — are defects
  measured against the byte-identity expectation.
- **Base/F32** is a **negative result by construction**: the tool has no
  F32-preserving output path, so `F32 → vindex → F32` is impossible; the finding
  *is* "compile silently downcasts F32 → bf16." The F32 variant documents this,
  it is not byte-compared.

## 5. Comparison lattice

Per variant, cross **driver** {LQL, CLI} × **mode** {A, B} × (for B) **INSERT
form** {Knn (default), Compose}. Comparisons catalogued:

| # | comparison | what it catalogues | expected (BF16 / F32) |
|---|---|---|---|
| 1 | `input ↔ LQL·A` | LQL-driver serialization loss | identical / format-not-preserved |
| 2 | `input ↔ CLI·A` | CLI-driver serialization loss | identical / format-not-preserved |
| 3 | `LQL·A ↔ CLI·A` | **driver divergence** — do the two surfaces agree? | identical (else finding) |
| 4 | `x·B(Compose) ↔ x·A` | edit effect | **differs** (B==A ⟹ bug) |
| 5 | `x·B(Knn) ↔ x·A` | KnnStore not compiled | **B==A expected** (silent-loss headline) |
| 6 | `LQL·B ↔ CLI·B` | driver divergence under edit | identical (else finding) |

**INSERT-form conditioning of the B≠A oracle** (ast.rs `InsertMode`):
- `mode=Compose` → writes gate/up/down overlay (`PatchOp::Insert`) → compile
  materializes it → **expect B≠A**; `B==A` is a bug.
- `mode=Knn` (**default**) → stores a KnnStore residual key (`PatchOp::InsertKnn`),
  no FFN overlay → compile does not materialize it → **expect B==A**. The headline:
  *the default INSERT silently does not reach the compiled model.* Cross-ref the
  Vindexfile stub INSERT (#242) in the report; do not run it as a round-trip leg.

## 6. Architecture

Purely **additive**. New files + one new workflow. **Zero edits to
`conformance.py`** (the concurrent session's hot file). Model-level track (like
`model-lifecycle`), not the per-leg corpus — the round-trip reads weights from the
base model, so per-leg would re-introduce the "extract-twice" waste removed by
commit `2ba8a60f`.

### 6.1 Active-model gate

A single **active-model registry** is the source of truth read by `gen_legs.py`,
the `model-lifecycle` matrix, and the new round-trip job. Each model carries an
`active` flag and a **variant list** (repo + expected dtype). Default active =
`smol135` with variants `{SmolLM2-135M (F32), SmolLM2-135M-Instruct (BF16)}`;
all others `active: false`. Re-enabling = flip `active` + list variants.

### 6.2 `roundtrip_diff.py` (the differ)

A pure, testable module (mirrors the `tensor_presence.py` precedent). Two layers:

- **Structural layer — Python 3.12 stdlib only:**
  - manifest **bijection**: input files ↔ output files (orphans on either side are
    findings before any per-file comparison);
  - safetensors **header** diff: tensor set, per-tensor `dtype`/`shape`,
    `data_offsets` **order**, and `__metadata__` presence. Header is
    `u64-LE length + JSON`, parsed with `struct` + `json` (no torch/safetensors dep);
  - `config.json` **JSON structural diff** (arch rewrite, key add/remove,
    reformat);
  - tokenizer / other companion files: **raw-byte** compare.
- **Value layer — numpy:** decode tensor data buffers and compute ladder rungs
  3–5 (tensor-wise bytes → bit-exact → ε-isomorphism). Required for the B≠A oracle
  and for confirming byte-identity of retained values.

### 6.3 The identity/isomorphism ladder (classification instrument)

Per matched file, record the **strictest rung that holds**. The rung a file breaks
at *is* its category — the ladder classifies, it never normalizes to pass.

1. **raw SHA256** — whole file byte-identical.
2. **canonicalized** — identical after normalizing safetensors header key order +
   padding (a pass here with rung-1 fail ⟹ **cosmetic serialization order**).
3. **tensor-wise bytes** — every named tensor's raw bytes match.
4. **bit-exact numeric** — dequantized values bit-identical.
5. **numeric isomorphism** — equal within ε / up to a declared structural
   equivalence.

### 6.4 Command-outcome dimension (consumed, not re-implemented)

The differ **consumes** `run_matrix.py` rows (`bucket`, `err_signal`, `err_line`,
`exit_code`) for the "did the driving statement parse / run / crash / mask an
error" dimension. Parse-error, masked-error, and crash classification remain the
**concurrent session's** `conformance.py` territory — the differ never
re-implements it. A classification gap surfaced here is routed there, not patched.

## 7. Difference taxonomy + JSONL schema

**Categories** (each attributed to a code cause):
`config-arch-rewrite` (K3, Gemma3 stamp) · `config-reformat` (K3, pretty-print) ·
`dtype-downcast` (K2, F32→bf16) · `tensor-order-vs-source` (K8/N7) ·
`metadata-loss` (K10, `serialize(None)`) · `lm_head-drop` (K4, tied dedup) ·
`driver-divergence` (LQL vs CLI) · `format-not-preserved` (round-trip-validity
precondition failed, §4) · `refusal-behavior` (per level, consumed from
`run_matrix`).

**Round-trip-validity precondition:** a comparison is a *true round-trip*
(byte-identity expected) only when `in-format == out-format`; otherwise it is a
*format-transform* and its byte-identity rungs are recorded **N/A** with category
`format-not-preserved`.

**Catalogue row (JSONL)** — one per (variant, driver, mode, insert-form, file,
comparison):
```
{
  "model": "smol135", "variant": "instruct-bf16" | "base-f32",
  "driver": "lql" | "cli", "mode": "A" | "B", "insert_form": "knn" | "compose" | null,
  "comparison": "input_vs_A" | "lqlA_vs_cliA" | "B_vs_A" | ...,
  "file": "model.safetensors" | "config.json" | ...,
  "true_roundtrip": true | false,          // in-format == out-format
  "strictest_rung": 1..5 | "na",
  "categories": ["dtype-downcast", ...],
  "expected": "byte-identical" | "differs" | "B==A" | "format-not-preserved",
  "matches_expected": true | false,        // false ⟹ the finding
  "detail": "...", "cause_refs": ["K2", "#242", ...]
}
```

## 8. CI wiring — `lql-roundtrip-catalogue.yml`

Triggered on the isolated branch. Per active variant:
1. **build once** (reuse the matrix build artifact pattern);
2. **produce**: extract vindex; run both drivers × {A; B(Knn); B(Compose)};
   emit `input-manifest`, `output-manifest`, both safetensors **headers**, file
   **hashes**, and the driving statements' `run_matrix` outcome JSONL;
3. **`roundtrip_diff` job**: join per `(model, variant)`, run the differ →
   catalogue **JSONL** + aggregated **`roundtrip-catalogue.md`**.

Artifacts namespaced `roundtrip-*`, **excluded** from the `results-*` conformance
glob, retention 24 h. Resource note: extract + compile + any INFER run real
inference → GitHub-hosted runners only (uncontained locally would OOM the dev box;
`larql-probe safe` itself has an open deadlock, #246).

## 9. Testing (TDD)

- `roundtrip_diff_test.py` (pytest, Python 3.12): the differ is a **pure function
  over synthetic manifests/headers** — every category, every ladder rung, the
  round-trip-validity precondition, and the INSERT-form-conditioned oracle are
  tested with fabricated inputs, **no larql required**. The numpy value layer is
  tested on tiny synthetic tensors (F32-vs-bf16 downcast, exact-match, ε-match).
- Follows the `tensor_presence_test.py` precedent (stdlib checker, pytest-only test
  dep).

## 10. Honesty / absence inventory (baked into the report)

The `roundtrip-catalogue.md` **leads** with:
- **N1:** `compile` re-reads the **base model** (single.rs:31, patch.rs:24); the
  vindex only supplies patch edges. So this round-trip tests the **re-serialization
  pipeline**, not whether the vindex stored the weights. A true vindex-fidelity
  test (§3, out of scope) is named as future work.
- **Co-headline (§4):** `compile` has **no F32-preserving output path** (forces
  bf16) — so only the BF16/Instruct model can be byte-identity-tested; the
  F32/base model documents the absence of a format-preserving round-trip.
- **N6:** source `__metadata__` is dropped (`serialize(None)`, save.rs:113) —
  invisible to any value comparison; only the header diff reveals it.
- **N7:** tensor/header **order** vs source diverges even when values are
  bit-identical — the header key-order diff must be explicit or this category is
  invisible.

## 11. Research residual references

Key facts this design rests on (from the research phase, 2026-07-11):
`K2` forced-bf16 output · `K3` Gemma3 config-stamp + reformat · `K4` `lm_head`
tied-dedup drop · `K8` safetensors 0.7.0 deterministic serialize order (so raw-SHA
mismatch vs source = order/dtype/config, not run noise) · `K9` base 135M = F32,
Instruct = BF16 · `K10` `__metadata__` dropped · `N1` compile re-reads base.
Related issues: `#242` (Vindexfile INSERT stub), `#272` (INFER/INSERT/COMPILE
panic below `all`), `#254` (tied-embed `lm_head` duplication), `#246`
(`larql-probe safe` deadlock).
