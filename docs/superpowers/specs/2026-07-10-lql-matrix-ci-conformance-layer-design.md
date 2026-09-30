# CI Conformance Layer for the LQL Strategy Matrix — Design

Date: 2026-07-10 · Status: design (v1, extensible) · Scope: harness/CI only, no larql code.

## Context & motivation

The `lql-strategy-matrix` CI (`scripts/lql_matrix/`, `.github/workflows/lql-strategy-matrix.yml`)
is a *discovery* experiment: for every `(source × produce-recipe × level)` leg it
produces a vindex and runs the full LQL command corpus, recording raw mechanical
outcomes. It has no oracle — a cell can be observed but never judged wrong.

The expanded run (80 legs, 6 models) plus the tracker's V&V issue landscape show
this is now the limiting factor. Empirically:

- **Silent hollow extraction** — granite-3.0 MoE and BitNet extract to **0 features**
  (exit 0, `has_model_weights=True`). Confirms/extends the `#183` umbrella
  (arch-key-driven extraction silently drops unmapped tensors), `#147` (bitnet),
  `#153` (MoE experts). Only qwen/llama-dense extract real vindexes.
- **No fail-loud guardrail** — hollow/partial vindexes don't error; downstream ops
  *panic* instead: attention-level → `sparse_compute.rs:85` (B1); q4k×MoE →
  `weight.rs:335` (B2).
- **Diagnostic reporting is itself unreliable** — `larql lql` exits 0 on in-band
  errors (`#206` zero rejection power); a silent `--level` override on `--quant q4k`;
  a panic message that blames `--compact` when the cause is q4k×MoE; `SHOW LAYERS`
  reporting a plausible-but-wrong `0`. Failures and successes are not mechanically
  trustworthy.

The tracker already theorizes the larql-side fixes (`#183/#155/#201/#206/#216/#247`).
This design does **not** implement those. It builds the **CI-side empirical oracle**
that detects these failures automatically — the prerequisite the maintainer named:
*"the first step is discovering what fails; sorting into regressions is only meaningful
once we can detect failures at all."*

## Goal

Turn the matrix into a **conformance probe**: evaluate a catalog of principled,
no-expected-output invariants over the already-captured per-leg/per-cell data, and
report every violation in a mechanically-checkable form — without abandoning the
discovery framing (violations warn by default; a `strict` mode gates).

## Design decisions (settled)

1. **Invariants only** — no declared per-`(arch×format×op)` baseline. A failure is a
   failure; regression-vs-known sorting is deferred (extension point).
2. **Strict opt-in** — default: record violations red in the report, run stays green.
   A `strict` `workflow_dispatch` input flips violations to a non-zero run failure.
   (Resolves `#247`.)
3. **Separate consumer** — `aggregate.py` stays *descriptive*; a new `conformance.py`
   is the *oracle/gate*. Single responsibility; the invariant catalog grows
   independently of the report.

## Invariant catalog (v1 — an extensible registry)

Each invariant is a pure function over fields already in the artifacts
(`results-*.jsonl` per-cell rows, `descriptor-*.json`, `produce-*.json`). Adding an
invariant = adding a function to the registry; this catalog is explicitly a *first
cut* sized for detection coverage, not completeness.

| id | class | check | primary catches |
|---|---|---|---|
| C-1 | Completeness | produced vindex `feature_count > 0` (derived from the leg's `STATS` cell output; falls back to Completeness-violation if `STATS` itself failed) | hollow granite/bitnet (#183/#147/#153) |
| N-1 | No-crash | no `exit 101` (Rust panic) or crash (134/137/139) in `produce` or any corpus cell. A graceful in-band refusal (exit 0 + err_signal) or clean non-zero error is allowed; a crash never is | B1, B2 |
| D-1 | Descriptor self-match | produced `quant == expect_quant` | silent dequant |
| D-2 | Descriptor self-match | `family` is a recognized arch, not a silent `GenericArch` fallback | #154 |
| X-1 | Cross-check | `SHOW LAYERS` per-layer feature count == `STATS` feature count | S1 (mmap SHOW LAYERS) |
| X-2 | Cross-check | `q4k`-inline descriptor == `q4k`-posthoc descriptor; `safetensors-to-vindex` == `extract` | format/recipe invariants |
| R-1 | Diagnostic | error-text/exit coherence — a cell with `err_signal=1` that still exits 0 is "failed to error"; exit 0 with no error text but a hollow/degenerate result is "false success" | #206 (no rejection power) |
| R-2 | Diagnostic | flag-honored — a requested flag not reflected in the produced descriptor, with no warning emitted, is "failed to warn" | S2 (silent `--level` override) |
| R-3 | Diagnostic | message-attribution — error/panic text names a flag the recipe did not pass (e.g. `--compact`) → misattribution | B2 (misleading `--compact` message) |

**Diagnostic-conformance scope note.** R-* check the *mechanics* of larql's error/
success reporting (does it error when it should, warn when it should, attribute
correctly, and not claim false success). They do **not** check semantic output
correctness (is INFER's answer right?) — that requires an oracle and is out of scope.

## Architecture & data flow

```
matrix legs (produce + corpus)  ──►  artifacts:
   results-<leg>.jsonl (meta + per-cell rows: exit_code, bucket, err_signal,
                        err_line, stdout/stderr head+tail, peak_rss_kb, duration)
   descriptor-<leg>.json (family, dtype, quant, expect_quant, quant_match, hidden)
   produce-<leg>.json    (produce exit/bucket/duration/rss/vindex_mb)
        │
        ├──►  aggregate.py  ──►  lql-matrix.md      (descriptive; unchanged)
        └──►  conformance.py ──► conformance.md      (violations by class → job summary)
                                 conformance.json    (machine-readable: [{id,class,leg,
                                                       cell,detail,severity}])
```

`conformance.py`:
- input: the same `artifacts/results-*/…` glob the aggregate job already downloads.
- reads each leg's meta/cells/descriptor/produce; derives `feature_count` from the
  `STATS` cell's `Features: N` line (no new capture step, no larql change).
- runs the invariant registry over every leg/cell; collects violations.
- writes `conformance.md` (grouped by class, with the offending leg/cell + the
  captured evidence line) and `conformance.json`.
- exit code: `0` always, unless `--strict` (set from the workflow input) **and**
  violations exist → exit non-zero.
- **Tolerant**: missing/partial artifacts never crash the checker — a leg with no
  produced vindex yields a Completeness violation, a malformed line is skipped-and-counted.

**Workflow change.** Add a peer `conformance` job (`needs: matrix`, `if: always()`,
separate from `aggregate` so the descriptive report and its artifact upload are never
affected by the gate's exit). It downloads the leg artifacts, runs `conformance.py`,
appends `conformance.md` to `$GITHUB_STEP_SUMMARY`, uploads the reports (24h
retention), and honors a `strict` `workflow_dispatch` input for its exit code.

## Minimal capture additions

None to the produce/corpus steps. `feature_count` comes from the `STATS` cell that
already runs in every corpus. Requested recipe flags are already on the leg record.
(Optional hardening later: have `descriptor.py` also record `feature_count` directly
from the vindex so Completeness doesn't depend on `STATS` succeeding.)

## Testing

Extend the stub-driven `lql-matrix-smoke` workflow (no larql build, no model) with
fixtures that exercise each invariant class and assert `conformance.py` classifies
them: a hollow stub (0 features) → C-1; a panic stub (exit 101) → N-1; an
exit-0-with-`Error:` stub → R-1; a `--level`-ignored stub → R-2. Smoke-first on a
runner, per the established discipline, before wiring into the real matrix.

## Out of scope / extension points (this is v1)

- **Semantic output correctness** (oracle-requiring) — excluded.
- **Regression-vs-known sorting** (a declared `(arch×format×op)` baseline) — deferred;
  the invariant registry and `conformance.json` are shaped so a baseline can later
  reclassify a known violation from red→amber without changing the checks.
- **Growing the catalog** — the registry is the extension surface; new invariants
  (e.g. RSS ceilings, timeout budgets, more cross-checks) slot in as functions.

## Success criteria

1. The expanded run's known failures are each caught by an invariant and appear in
   `conformance.md`: granite/bitnet hollow (C-1), B1/B2 panics (N-1), SHOW LAYERS
   (X-1), silent `--level` override (R-2), the `--compact` misattribution (R-3).
2. Default run stays green; `strict` run fails with a non-zero exit listing violations.
3. `conformance.py` never crashes on missing/partial artifacts.
4. Adding a new invariant is a single-function change with a stub-smoke fixture.
