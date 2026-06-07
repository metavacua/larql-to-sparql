# ADR-0018 — LM Synthesis Pipeline

**Status:** Proposed  
**Date:** 2026-06-06  
**Depends on:** `larql-lql`, `larql-python`, `larql-vindex`

---

## Context

The LM synthesis toolchain distills LQL knowledge into a vindex (Architecture B:
KNN distillation). It currently exists as an informal Python toolchain in
`crates/larql-wasm/tools/`. This ADR formalizes the architecture, identifies known
gaps, and defines the path to a production-quality pipeline.

---

## Architecture

The pipeline has two components:

### 1. BoW Classifier (¬L∧¬M certified)

A pure-numpy logistic regression classifier that predicts which LQL statement type
(SELECT, WALK, INFER, INSERT, etc.) is most appropriate for a given natural-language
prompt.

**Input:** Bag-of-words feature vector of the prompt.  
**Output:** LQL statement type probability distribution.  
**Weights:** 352 KB JSON file (stored in `tools/` — should be moved to external storage).

**Properties:**
- No recursion (¬L): linear scan over vocabulary, dot product, sigmoid
- No unbounded allocation (¬M): fixed-size vocabulary vector, fixed-size weight matrix
- ¬L∧¬M certifiable as a WASM32 kernel once ported to Rust

**Current accuracy:** 76.8% combined LQL + wasm statement type prediction.

### 2. KNN Distillation (Architecture B)

Stores (layer, residual_key, target_token) tuples. At inference time, if the
residual at layer `l` has cosine similarity > 0.75 with a stored key, the
corresponding `target_token` overrides the model's top-1 prediction.

**Correctness:** 100% same-prompt recall (stored prompts are always retrieved
correctly). Generalization to novel prompts is not measured.

**What was tried and falsified:** Compose mode (`install_compiled_slot`) does not
work for the KNN synthesis use case. KNN mode only.

---

## Known Gaps

| Gap | Current state | Target |
|-----|--------------|--------|
| BoW weights in git | `tools/bow_classifier_weights.json` (352 KB) | External storage (HuggingFace Hub or S3) |
| KNN store not persisted | KNN store lives in process memory; lost on exit | `knn_save` / `knn_load` Python binding |
| No Rust classifier | Classification requires Python at inference time | `larql-classify` crate or method in `larql-lql` |
| LQL coverage | 76.8% combined | Target 90%+ |
| No incremental training | BoW classifier retrained from scratch on each corpus update | Incremental SGD or online learning variant |

---

## Decisions

### knn_save / knn_load binding

Add two methods to `larql-python`'s `PyVindex`:

```python
vindex.knn_save(path: str) -> None
vindex.knn_load(path: str) -> None
```

Implementation: serialize/deserialize the `KnnStore` from `larql-inference` as a
JSON or binary file. Effort: ~10 lines of PyO3 + existing serialization infrastructure
in `larql-vindex`.

### External storage for BoW weights

Move `tools/bow_classifier_weights.json` to an external artifact store (HuggingFace
Hub under the `metavacua/larql-synthesis` repo, or an S3 bucket). Add a `tools/fetch_weights.sh`
script that downloads the weights on first use. The weights must not live in git —
they bloat the repo and change frequently during training.

### Python → Rust → wasm32v1-none translation subproject

A parallel subproject translates the Python synthesis toolchain to equivalent pure Rust
that compiles to `wasm32v1-none`. Three translation targets:

1. **BoW classifier** → `larql-classify` crate: fixed-size vocabulary embedding, dot
   product + logistic regression weights, ¬L∧¬M certifiable. Deferred until Python
   baseline stabilizes (≥ 90% accuracy).
2. **KNN store query path** → pure Rust cosine scan over a fixed-layout byte slice,
   certifiable ¬L∧¬M. Enables browser-side KNN lookup without a Python interpreter.
3. **Template expansion** → pure Rust LQL template renderer, mapping top-1 statement
   type to an LQL statement string.

The translated Rust code lands in `larql-python-wasm32v1-none` (no PyO3 dependency).
Once translated, the synthesis pipeline runs entirely in-browser as a certified
wasm32v1-none module — no Python interpreter, no server round-trip.

### Pyodide deployment route (wasm32-unknown-emscripten)

For environments that already have a Python runtime (Pyodide in the browser,
or a WASI Python host), the existing PyO3 bindings can be deployed via
`wasm32-unknown-emscripten` at the `posix` resource tier. This is handled by
the `larql-python-interface` crate (#124) — see ADR-0016 Phase 4 (emscripten probe).

The Pyodide route and the Rust translation route are complementary:
- Pyodide: users who want `import larql` in a browser Python REPL
- Rust translation: users who want a lightweight certified wasm32v1-none module with
  no Python interpreter dependency

---

## Pipeline Diagram

```
Natural-language prompt
    │
    ▼
BoW Classifier (Python / future: Rust ¬L∧¬M)
    │ top-1 statement type
    ▼
LQL template expansion
    │ LQL statement string
    ▼
larql-python executor
    │
    ├── KNN lookup (KnnStore, cosine > 0.75)
    │       │ hit: override top-1 with stored target_token
    │       └── miss: run forward pass
    │
    ▼
Response token
    │
    ▼
KNN store update (if new prompt is high-confidence)
```

---

## License

AGPL v3. Rationale: novel synthesis pipeline; original state-of-the-art
contribution.

---

## Consequences

**Positive.**

- Formalizing the pipeline makes the BoW classifier and KNN store production-ready
  artifacts, not one-off Python scripts.
- `knn_save` / `knn_load` makes the KNN distillation persistent across sessions,
  enabling incremental training.
- Once ported to Rust, the BoW classifier becomes part of the certified browser
  LQL dialect — a classifier that runs in a browser sandbox.

**Negative.**

- Moving weights to external storage requires a fetch step in the development
  workflow. Contributors must run `fetch_weights.sh` before using the classifier.
- The 76.8% accuracy floor means 23.2% of prompts are misclassified. The error
  mode is incorrect statement type selection, which degrades inference quality
  but does not cause crashes.

**Not in scope.**

- Retraining the base language model (INFER remains neural; only the statement
  type classifier is replaced).
- Multi-language synthesis (English prompts only in the current classifier).
- Active learning or human-in-the-loop labeling for the training corpus.
