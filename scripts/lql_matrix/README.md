# LQL Strategy Matrix — empirical extraction × command experiment

**What this is:** a discovery experiment that extracts one real model at **every
extraction level** (`browse`, `attention`, `inference`, `all`) and runs the
**full LQL command corpus** against each level's vindex, recording the **raw
mechanical outcome** of every cell.

**What this is NOT:** a validation suite. It makes **no** claim about whether any
command's output is *correct*. There are no semantic oracles, no expected
values, no pass/fail-correctness. A command that breaks is **data to read**, not
a bug to fix here. Imposing "expected" semantics is how a broken thing gets
papered over or misreported — so we don't. We run it and record what actually
happened.

## Why it runs in CI, not on a dev box

Extraction (especially `--level all`) and `COMPILE`/MEMIT are minutes and
multi-GB per cell. The point is to see how the full matrix behaves under real
load across levels and (later) models — so it runs on **GitHub-hosted runners**
(ample RAM, ≤6 h budget, isolated), where a dev machine's constraints are
irrelevant. See `.github/workflows/lql-strategy-matrix.yml`.

## Axes

- **Level** (matrix job axis): `browse ⊂ attention ⊂ inference ⊂ all`. Extracted
  fresh per level — extraction itself is an observed cell.
- **Command variant** (`commands.jsonl`): every LQL statement and its variants.
  Each cell carries its **own dependency chain** (e.g. the `INSERT` cell opens a
  patch; the round-trip cell does `INSERT → SAVE → COMPILE → USE → INFER`), so a
  cell tests both the statement and the chain it needs.

Start with one model (Qwen2.5-Coder-0.5B-Instruct). Add models by extending the
workflow matrix later.

## Recording (mechanical only)

The runner is `run_matrix.py` (invoked via the `run_matrix.sh` shim). It emits a
`{"type":"meta",…}` **provenance** row per level (commit, `larql --version`,
model, runner OS, `RUST_BACKTRACE`, timestamp), then one row per cell with:
exact substituted command, `exit_code`, coarse **mechanical** bucket,
`duration_ms`, `peak_rss_kb` (via `/usr/bin/time -v`), `stdout_bytes`/
`stderr_bytes`, an 800-char head of each stream **and** an 800-char `stderr_tail`
(panics/backtraces live at the tail), plus an **error-text overlay**
(`err_signal`/`err_line`). The **full** stdout/stderr of every cell are written
to `<out_dir>/cells/<level>.<id>.{out,err}` and kept as artifacts — nothing is
captured then discarded.

Buckets: `ok` (exit 0) · `err<N>` (non-zero N) · `timeout` (per-cell cap) ·
`crash` (SIGKILL-OOM/SIGSEGV/SIGABRT = 137/139/134).

`larql lql` exits `0` even on an in-band error, so `ok` ≠ "did something
meaningful". `err_signal=1` marks a cell whose stdout/stderr carried an
`Error:`/`panicked`/`Parse error` **despite** exit 0 (a masked error or graceful
refusal); `aggregate.py` renders it as ⚠️ vs a clean ✅. We surface this, we do
not "correct" it.

I/O practices: stream content never transits the shell argv (the driver reads
the captured files itself, utf-8, truncating by codepoint); `timeout
--kill-after` bounds hung cells; artifacts are retained **24h** then auto-expire.

## Files

| file | role |
|---|---|
| `commands.jsonl` | the **vindex-level** corpus (runs per leg against its produced vindex); `{{VINDEX}}`/`{{TMP}}` placeholders |
| `commands-model.jsonl` | the **model-level** corpus (`EXTRACT MODEL` / `USE MODEL`) — run once per model by the `model-lifecycle` job, not per leg |
| `gen_legs.py` | enumerates the **legs** (native level grid + one-off transformation/encoding recipes) as JSON — the matrix columns |
| `descriptor.py` | reads a produced vindex's `index.json` → `(family, dtype, quant, …)` vs the leg's expected quant |
| `run_matrix.py` | the runner — orchestrates each cell, captures full streams + RSS + error-signal → JSONL |
| `run_matrix.sh` | thin shim → `run_matrix.py` (stable `<leg> <vindex> <corpus> <out>` API) |
| `aggregate.py` | merges per-leg JSONL → `lql-matrix.md` (descriptor conformance, per-leg tally, resource, failures detail) |
| `../../.github/workflows/lql-strategy-matrix.yml` | the CI matrix (plan legs → build once → produce vindex per leg → run corpus → aggregate → 24h artifacts + job summary) |
| `../../.github/workflows/lql-matrix-smoke.yml` | fast stub-driven smoke of the harness itself on a runner (no build/model) |

## Legs (decoupled axes)

A **leg** is one produced vindex + the vindex-level corpus run against it.
`gen_legs.py` keeps the axes separate rather than one uniform cross-product:

- **Native extraction** — every model × every level `{browse, attention, inference,
  all}` at native precision. This is the real "test each level independently".
- **Transformations** — tested **once at `level=all`**, not multiplied by the level
  grid: `--quant q4k` per model, `--f32`, post-hoc `quantize {q4k,fp4}`. A
  transformation like q4k *implies* `--level all`, so crossing it with levels only
  homogenises (that redundancy is now *asserted* via a single q4k-browse sentinel +
  the conformance cross-check, not re-run — see tracker #275).
- **Encoding / convert** — `gguf-to-vindex`, including BitNet native **I2_S ternary**
  via `--keep-quant`. GGUF legs carry `tokenizer_repo`; the workflow stages the base
  repo's `tokenizer.json` beside the `.gguf` (the `-gguf` repo ships none — #180/#277).

Model-level commands (`EXTRACT MODEL` / `USE MODEL`) don't read a produced vindex, so
they run **once per model** in the `model-lifecycle` job (`commands-model.jsonl`) —
not replicated across every leg. Each produced vindex's actual `(family, dtype,
quant, feature_count)` is recorded and checked by the conformance layer.

## Run locally (tooling smoke-test only — NOT the full experiment)

```bash
# Cheap browse cells against an existing vindex, contained on a dev box:
LARQL_BIN=target/release/larql \
WRAP="larql-probe safe --mem 2500 --" \
scripts/lql_matrix/run_matrix.sh browse <some.vindex> \
  scripts/lql_matrix/commands.jsonl out/results-browse.jsonl
python3 scripts/lql_matrix/aggregate.py "out/results-*.jsonl" lql-matrix.md
```

On CI, `WRAP` is empty (the runner is already isolated) and larql runs directly.
