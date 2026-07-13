# Round-Trip Compile Exercise on `lql-strategy-matrix.yml` — Design Spec

> Supersedes the mis-scoped `2026-07-11-smollm135-roundtrip-catalogue-design.md`
> (that arc built a parallel Python library; this one extends the existing workflow).

## 1. Metadata

```yaml
# Dublin Core Terms (dcterms)
dcterms:title:        "Round-Trip Compile Exercise on lql-strategy-matrix.yml"
dcterms:creator:      "metavacua"
dcterms:contributor:  "Claude Opus 4.8 (assistant)"
dcterms:date:         "2026-07-12"
dcterms:type:         "Text.DesignSpecification"
dcterms:identifier:   "spec/2026-07-12-roundtrip-compile-lql-strategy-matrix"
dcterms:subject:      ["larql", "lql", "GitHub Actions", "model round-trip", "compile", "SmolLM2-135M", "difference measurement"]
dcterms:description:  >-
  Extend the existing GitHub Actions workflow lql-strategy-matrix.yml to exercise
  the COMPILE half of the model→vindex→model round-trip (larql `compile` CLI and
  LQL `COMPILE … INTO MODEL`) on the vindex the workflow already extracts for
  SmolLM2-135M base (F32) and Instruct (BF16), and to compute and emit the measured
  differences between the input model and each reconstructed model.
dcterms:conformsTo:   "ISO/IEC/IEEE 29148:2018"
dcterms:relation:     ["metavacua/larql-to-sparql#271", "lql-strategy-matrix.yml", "scripts/lql_matrix/gen_legs.py"]

# Schema.org
schema:@type:         "TechArticle"
schema:headline:      "Round-Trip Compile Exercise on lql-strategy-matrix.yml"
schema:about:         "Measured byte/tensor differences of a larql model→vindex→model compile round-trip, run in CI"
schema:proficiencyLevel: "Expert"
schema:dependencies:  ["larql-cli", "safetensors 0.7.0", "numpy", "python 3.12", "GitHub Actions"]
```

## 2. Constraints (RFC 2119 / BCP 14 — normative in ALL-CAPS)

- **C-1** The deliverable MUST be edits to `.github/workflows/lql-strategy-matrix.yml` and its helper scripts under `scripts/lql_matrix/`. It MUST NOT introduce a parallel enumeration/runner/aggregate library; it MUST reuse `gen_legs.py`, the `matrix` job, and the peer-job aggregation pattern.
- **C-2** The change MUST be additive and MUST NOT modify `scripts/lql_matrix/conformance.py` (owned by the concurrent session, PR #271).
- **C-3** The round-trip step MUST run inside the `matrix` job on the runner where the leg's extracted vindex and the downloaded base model are already present; it MUST NOT re-extract the model.
- **C-4** The experiment MUST run on GitHub-hosted runners; it MUST NOT depend on the local dev host, and it MUST be exercised via commit → push → pull request → Actions.
- **C-5** The round-trip MUST exercise BOTH compile drivers — the `larql compile` CLI and the LQL `COMPILE … INTO MODEL … FORMAT safetensors` statement — and MUST compare their outputs.
- **C-6** The round-trip MUST exercise BOTH modes — passthrough (compile with no edit) and edit-bearing (LQL `INSERT` then compile) — and for the edit-bearing mode MUST run the LQL `INSERT` statement in BOTH `MODE KNN` and `MODE COMPOSE`.
- **C-7** The differ MUST emit only measured quantities and machine-checkable predicates. It MUST NOT emit `expected`, `matches_expected`, `cause`, `verdict`, `category`, or any meaning-named interpretation field.
- **C-8** The differ MUST degrade to a structured record on malformed/unreadable input and MUST NOT crash the job (matching the existing "never crash the checker" convention, commit `315087a1`).
- **C-9** Round-trip artifacts MUST be namespaced `roundtrip-*` and MUST be excluded from the `results-*` conformance/aggregate glob.
- **C-10** The value-level tensor comparison MAY use `numpy`; the structural comparison SHOULD use the Python standard library only.
- **C-11** The measure-and-document posture MUST hold: larql/lql defects surfaced by the round-trip are RECORDED (and cross-referenced to issues), not fixed in this change.

## 3. Acceptance Criteria (EARS)

- **AC-1** *(ubiquitous)* The workflow SHALL emit, per compile-capable SmolLM2-135M leg and variant, a `roundtrip-<leg>.json` file containing the measured differences between the input model and each reconstructed model.
- **AC-2** *(event-driven)* WHEN a compile-capable SmolLM2-135M leg's vindex has been produced, the round-trip step SHALL run the `larql compile` CLI and the LQL `COMPILE … INTO MODEL` statement against that vindex and the base model, and SHALL retain the output model directories on the runner.
- **AC-3** *(event-driven)* WHEN the reconstructed model directories exist, the differ SHALL compute the manifest bijection, the safetensors header diff, the `config.json`/companion-file diff, and the per-tensor value metrics, and SHALL write them to `roundtrip-<leg>.json`.
- **AC-4** *(state-driven)* WHILE a leg is not compile-capable (extract level below `inference`), the round-trip step SHALL be skipped via an `if:` guard and SHALL NOT emit a `roundtrip-*` artifact.
- **AC-5** *(unwanted behavior)* IF the differ encounters a malformed or unreadable model file, THEN it SHALL record a structured error entry for that file and SHALL continue processing the remaining files without crashing the job.
- **AC-6** *(event-driven)* WHEN all matrix legs have completed, the peer `roundtrip` job SHALL download every `roundtrip-*.json` and aggregate them into a single catalogue artifact.
- **AC-7** *(ubiquitous)* Every field the differ emits SHALL be a measured quantity or a machine-checkable predicate; the emit SHALL contain no interpretation field, verified against a declared output schema.
- **AC-8** *(optional feature)* WHERE the edit-bearing mode is exercised, the LQL `INSERT` SHALL be run in both `MODE KNN` and `MODE COMPOSE`, and each reconstructed model SHALL be compared against the passthrough reconstruction produced by the same driver.
- **AC-9** *(event-driven)* WHEN the two drivers (CLI, LQL) produce their respective output models for the same leg/mode, the differ SHALL additionally compute and emit the driver-vs-driver difference.

## 4. Architecture (Structurizr DSL)

```structurizr
workspace "lql-strategy-matrix + roundtrip" {
  model {
    dev = person "Maintainer"
    ci = softwareSystem "lql-strategy-matrix workflow" {
      plan       = container "plan job"        "enumerates legs via gen_legs.py"
      build      = container "build job"        "compiles larql-cli once → binary artifact"
      matrix     = container "matrix job"       "per-leg: extract → corpus → (NEW) round-trip" {
        extractStep   = component "extract step"    "produces out/<leg>.vindex"
        corpusStep    = component "corpus step"     "runs commands.jsonl via run_matrix"
        roundtripStep = component "round-trip step (NEW)" "compile (CLI + LQL) × (passthrough + INSERT knn/compose); keep output models"
        differ        = component "differ helper (NEW)"  "roundtrip_diff.py: structural (stdlib) + value (numpy) → roundtrip-<leg>.json"
      }
      rtAggregate = container "roundtrip job (NEW)"  "downloads roundtrip-*.json → catalogue artifact"
      aggregate   = container "aggregate job"    "results-*.jsonl → lql-matrix.md"
      conformance = container "conformance job"  "results-*.jsonl → conformance.md/json (UNCHANGED)"
      inventory   = container "inventory job"    "manifest listings → tensor_presence → q8"
    }
    hf = softwareSystem "HuggingFace Hub" "snapshot_download of base models"

    dev -> ci "push → PR → Actions"
    plan -> matrix "legs (fromJSON)"
    build -> matrix "larql binary"
    hf -> matrix "base model snapshot"
    extractStep -> roundtripStep "out/<leg>.vindex (reused, no re-extract)"
    roundtripStep -> differ "input model dir + reconstructed model dirs"
    differ -> rtAggregate "roundtrip-<leg>.json (artifact)"
  }
  views {
    container ci "containers" { include * }
    component matrix "matrix-internals" { include * }
  }
}
```

*(Render/validate with Structurizr only if that tool is granted+present; otherwise the `.dsl` source above is the authoritative model — architecture-rendering skipped, tool not granted.)*

## 5. Decisions (MADR)

### ADR-1 — Round-trip runs as a step in the `matrix` job (not a separate job)
- **Status:** accepted.
- **Context:** the round-trip needs the leg's extracted vindex AND the base model on disk.
- **Alternatives:**
  1. **Step in the `matrix` job** *(chosen)* — reuses the leg's existing extraction and the on-disk base; only a small `roundtrip-<leg>.json` leaves the job.
  2. **Separate dedicated job** — re-extracts the model per round-trip (reintroducing the per-leg extract-twice waste removed by refactor `2ba8a60f`), or requires uploading the multi-GB vindex/model as an artifact.
- **Decision:** (1). Reuse-in-place; a separate job re-extracts or ships gigabytes for no benefit.

### ADR-2 — Value comparison uses `numpy`; structural comparison is stdlib-only
- **Status:** accepted.
- **Alternatives:**
  1. **numpy value layer + stdlib structural** *(chosen)* — numpy is universally present on GitHub runners; bf16/f32 decode + tolerance compare is trivial.
  2. **stdlib-only `struct` decode** — preserves the stdlib-only checker precedent but is slower and hand-rolls bf16→f32.
- **Decision:** (1). The value layer needs efficient tensor decode; the structural layer stays stdlib (`struct`+`json` read the safetensors header directly).

### ADR-3 — Differ runs where the model files are, emitting a small JSON
- **Status:** accepted.
- **Alternatives:** (1) diff in the `matrix` job, emit small JSON *(chosen)*; (2) upload full model dirs to a downstream job to diff there (ships gigabytes; near artifact limits).
- **Decision:** (1). `[sound: models are colocated in the matrix job]`.

## 6. Glut Register

- **G-1** *(scope limitation, not a contradiction — recorded for honesty):* `larql compile` re-reads the `--base` model for weights (the vindex supplies only patch edges), so the model→vindex→model round-trip measures the **re-serialization/compile pipeline**, NOT whether the vindex losslessly stored the weights. This is a known limitation of what the round-trip can prove, surfaced as a measured finding; a vindex-level fidelity comparison (source tensors vs the vindex's own `.bin` files) is out of scope for this spec. Not tagged `in-conflict` — no two acceptance criteria contradict; it bounds the interpretation of AC-1/AC-3 results.

*(No `in-conflict` acceptance criteria at this time.)*
