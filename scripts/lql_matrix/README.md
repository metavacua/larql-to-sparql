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

Per cell, `run_matrix.sh` records: exact command, `exit_code`, `duration_ms`,
`stdout_bytes`/`stderr_bytes`, an 800-char head of each stream, and a coarse
**mechanical** bucket:

- `ok` — exit 0
- `err<N>` — non-zero exit N
- `timeout` — killed by the per-cell wall cap
- `crash` — SIGKILL(OOM)/SIGSEGV/SIGABRT (137/139/134)

Note (already observed locally): `larql lql` exits `0` even on an unknown
statement, so `ok` ≠ "did something meaningful" — the signal is in the captured
stderr text. That is itself a finding the matrix surfaces; we record it, we do
not "correct" it.

## Files

| file | role |
|---|---|
| `commands.jsonl` | the command corpus (matrix columns); `{{VINDEX}}`/`{{MODEL}}`/`{{TMP}}` placeholders |
| `run_matrix.sh` | runs a corpus against one vindex → raw JSONL outcomes |
| `aggregate.py` | merges per-level JSONL → `lql-matrix.md` (level × command table + tallies) |
| `../../.github/workflows/lql-strategy-matrix.yml` | the CI matrix (extract per level → run → aggregate → artifacts + job summary) |

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
