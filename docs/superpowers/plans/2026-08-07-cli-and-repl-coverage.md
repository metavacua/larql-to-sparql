# CLI and REPL Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the LQL strategy matrix from 4 of larql's 38 CLI subcommands and zero REPL coverage to the whole CLI surface plus the REPL, driven three ways, with dependency-ordered and permuted cells and verbatim captures.

**Architecture:** `run_matrix.py` gains a `--driver` parameter selecting how a cell is invoked (`lql` | `repl-pipe` | `repl-pty` | `cli`). Two new corpora describe CLI invocations. A new `sequence.py` orders cells by declared `needs`/`produces` and permutes the independent ones deterministically. `gen_legs.py` adds the driver axis; the workflow runs each driver and uploads every capture.

**Tech Stack:** Python 3.12 stdlib only (`subprocess`, `pty`, `json`, `argparse`, `random`), pytest for harness tests, bash + GitHub Actions for the workflow. No new dependencies.

## Task Ordering: highest failure probability first

Tasks are ordered by **how likely they are to fail**, not by dependency
convenience. Anything that could invalidate later work is attempted before that
work exists; near-certainties come last, when a failure costs a fix rather than
a redesign.

| rank | risk | where |
|---|---|---|
| 1 | 38 CLI invocations with arguments guessed from doc comments | Task 0 |
| 2 | `larql repl` on non-tty stdin — rustyline behaviour unverified | Task 0 |
| 3 | `pty.fork` + `select` + `EIO` runner — fiddly, easy to hang | Task 0 |
| 4 | `--driver` wiring without changing existing `lql` behaviour | Task 3 |
| 5 | sequencer wiring, order actually observable | Task 9 |
| — | corpus regex fix, pure functions, leg axis, consolidation | Tasks 1, 2, 5, 8, 10 |

The top three all land in Task 0, which writes **no harness code at all** — it
runs things on a runner and uploads what they print. A guessed CLI argument that
is wrong should cost a capture to discover, not a corpus, a sequencer and a
workflow branch built on top of it.

## Exhaustive Coverage of Finite Condition Sets

Where a set of conditions is tractably finite, the tests enumerate **all
acceptance and all rejection cases**, not a representative sample.

| set | size | where enumerated |
|---|---|---|
| `drivers.build(driver, cell)` — 4 drivers × {lql-cell, cli-cell} | 8 | Task 2 |
| `sequence.sequence` graph shapes — empty, single, chain, diamond, all-independent, unsatisfiable, self-cycle, mutual cycle | 8 | Task 8 |
| `corpus_lint` rejections — each required key missing, duplicate id, bad subcommand, bare BEGIN PATCH, missing deps | 8 | Tasks 1, 6 |
| larql subcommands | 38 | Tasks 0, 6, 7 |
| `ExtractLevel` values × surface (CLI, LQL) | 4 × 2 | already in the matrix |

"Reasonable threshold" means the enumeration stops where the set stops being
finite: cell *orderings* are factorial, so permutation is sampled by seed rather
than exhausted, and that sampling is declared in Task 9 rather than pretended to
be coverage.

## Demonstration Discipline

**Every task that changes runtime behaviour ends by running on a real runner
against the real binary, and the next task does not start until that output has
been read.** Unit tests here use a fake `larql` shell script; they prove the
harness calls what it means to call, and prove nothing about what larql does.

Concretely, each such task's final steps are: push, wait for the run, download
the artifact, read it, and only then proceed. Tasks 2 and 8 are pure functions
with no runtime surface and are exempt. Task 0 exists solely to answer the
riskiest unknown before anything is built on it.

This is deliberately slower per task and avoids the failure it is named for: a
long uninterrupted build whose components were only ever exercised against
stand-ins, discovering at the end that the surface they stand in for behaves
differently.

Wiring is therefore incremental too. Each driver is connected to the workflow in
the task that introduces it, not batched into a single wiring task at the end.

## Global Constraints

Copied from `docs/superpowers/specs/2026-08-07-cli-and-repl-coverage-design.md`. Every task's requirements implicitly include these.

- **The harness reports; it does not judge.** No pass/fail, no bucket, no error-signal, no first-error line, no conformance verdict. Never add one back.
- **No failure becomes a plausible success value.** No `2>/dev/null`, no `|| true`, no substituting `{}` or `""` for a crash. Record the failure and its stderr.
- **No post-processing of output.** Captures are verbatim. No splitting, no truncation, no head/tail snapshots, no stripping of terminal control sequences or `\r`.
- **No redundancy assumptions.** A CLI subcommand is never treated as covered by a same-named LQL statement or by a driver. Both run, always.
- **Rows are indexes, not descriptions.** A JSONL row may contain only facts about the process — driver, cell id, what was sent, exit code, duration, peak RSS, byte counts, capture paths. Nothing derived from output contents.
- **Nothing is skipped.** Not for an unsatisfied dependency, not for a guessed-inapplicable subcommand, not for a timeout. A cell that cannot work still runs and the result is recorded.
- Harness tests live beside the module as `scripts/lql_matrix/<module>_test.py` and run with `python3 -m pytest -q` from `scripts/lql_matrix/`.
- Existing test suite must stay green: 59 tests at plan start.

---

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/lql_matrix/commands.jsonl` | *(modify)* LQL corpus; six `BEGIN PATCH;` cells fixed |
| `scripts/lql_matrix/corpus_lint.py` | *(create)* validates a corpus file's shape; one responsibility: is this corpus well-formed |
| `scripts/lql_matrix/corpus_lint_test.py` | *(create)* tests for the above |
| `scripts/lql_matrix/drivers.py` | *(create)* builds the argv / stdin for one cell under one driver. Knows nothing about corpora or sequencing |
| `scripts/lql_matrix/drivers_test.py` | *(create)* tests for the above |
| `scripts/lql_matrix/sequence.py` | *(create)* dependency ordering + deterministic permutation. Pure; no I/O |
| `scripts/lql_matrix/sequence_test.py` | *(create)* tests for the above |
| `scripts/lql_matrix/run_matrix.py` | *(modify)* orchestration only: read corpus, sequence, invoke via driver, capture, index |
| `scripts/lql_matrix/run_matrix.sh` | *(modify)* pass through the new flag |
| `scripts/lql_matrix/cli-help.jsonl` | *(create)* one `--help` cell per subcommand |
| `scripts/lql_matrix/cli-commands.jsonl` | *(create)* real invocations with `needs`/`produces` |
| `scripts/lql_matrix/gen_legs.py` | *(modify)* driver axis + long-session legs |
| `.github/workflows/lql-strategy-matrix.yml` | *(modify)* run each driver, upload every capture |

---

### Task 0: Probe everything likely to fail, before any code exists

The three highest-risk items in this plan are attempted here, with **no harness
code**: 38 CLI invocations whose arguments are guesses, `larql repl` on non-tty
stdin, and a `pty.fork` runner. Each is something that, if wrong, invalidates
work built on top of it.

**Files:**
- Modify: `.github/workflows/lql-strategy-matrix.yml`
- Create: `scripts/lql_matrix/probe_pty.py` *(20 lines, throwaway-shaped but kept — Task 4 lifts its loop)*

**Interfaces:**
- Consumes: the `larql-bin` artifact from the existing `build` job.
- Produces: a `probe` artifact. Task 4 lifts the verified read loop from
  `probe_pty.py` into `run_matrix.run_under_pty`. Task 7's corpus is written
  from the captured argument errors rather than guessed a second time.

- [ ] **Step 1: Write the minimal pty probe**

Create `scripts/lql_matrix/probe_pty.py`:

```python
#!/usr/bin/env python3
"""Smallest thing that answers: can we give a child a real tty and read it back?

Task 4's run_under_pty is the fiddly part of this plan — pty.fork, a select
loop, and EIO-on-child-exit are each easy to get subtly wrong in a way that
hangs a runner. This proves the loop on a runner before it is embedded in the
harness. Usage: probe_pty.py <out-file> <cmd> [args...]  (stdin is forwarded)
"""
import errno, os, pty, select, signal, sys, time

def main():
    out, argv = sys.argv[1], sys.argv[2:]
    payload = sys.stdin.buffer.read()
    pid, fd = pty.fork()
    if pid == 0:
        try:
            os.execvp(argv[0], argv)
        except Exception:
            os._exit(127)
    if payload:
        os.write(fd, payload)
    deadline = time.monotonic() + 60
    with open(out, "wb") as f:
        while True:
            if time.monotonic() > deadline:
                os.kill(pid, signal.SIGKILL)
                print("TIMEOUT", file=sys.stderr)
                break
            r, _, _ = select.select([fd], [], [], 1.0)
            if not r:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    break
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError as e:
                if e.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            f.write(chunk); f.flush()
    os.close(fd)
    try:
        _, status = os.waitpid(pid, 0)
        print(f"exit={os.waitstatus_to_exitcode(status)}", file=sys.stderr)
    except ChildProcessError:
        print("exit=unknown", file=sys.stderr)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the probe locally against a known-tty program**

Run:
```bash
cd /home/metavacua/larql-vindex3-03-08-2026
printf 'hello\n' | python3 scripts/lql_matrix/probe_pty.py /tmp/p.out \
  bash -c 'if [ -t 0 ]; then echo TTY_YES; else echo TTY_NO; fi; read -r l; echo "GOT:$l"'
cat /tmp/p.out
```
Expected: `TTY_YES` and `GOT:hello`, and the command returns rather than hanging.
If it hangs, fix the loop **now** — this is the cheapest place it will ever be
diagnosed.

- [ ] **Step 3: Add the probe job**

In `.github/workflows/lql-strategy-matrix.yml`, insert this job immediately
after the `build:` job:

```yaml
  # Everything in this plan most likely to be WRONG, attempted first, with no
  # harness code. Three risks: CLI arguments guessed from doc comments; whether
  # `larql repl` reads non-tty stdin at all (run_repl goes through rustyline and
  # only falls back to the stdin reader when the editor FAILS to construct); and
  # a pty.fork read loop. Nothing here asserts. It runs things and uploads what
  # they printed.
  probe:
    needs: build
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v6
      - uses: actions/download-artifact@v4
        with:
          name: larql-bin
          path: bin
      - name: Prep
        run: |
          chmod +x bin/larql
          sudo apt-get update && sudo apt-get install -y libopenblas-dev util-linux
          mkdir -p probe
      - name: REPL — piped stdin
        run: |
          printf 'SHOW MODELS;\nSTATS;\nexit\n' \
            | timeout 60 ./bin/larql repl > probe/repl-pipe.out 2> probe/repl-pipe.err
          echo "exit=$?" >> probe/repl-pipe.err
      - name: REPL — pty via script(1)
        run: |
          printf 'SHOW MODELS;\nSTATS;\nexit\n' \
            | timeout 60 script -q -e -c './bin/larql repl' probe/repl-script.merged \
              > probe/repl-script.out 2> probe/repl-script.err
          echo "exit=$?" >> probe/repl-script.err
      - name: REPL — pty via probe_pty.py (the loop Task 4 will use)
        run: |
          printf 'SHOW MODELS;\nSTATS;\nexit\n' \
            | python3 scripts/lql_matrix/probe_pty.py probe/repl-pty.merged \
              ./bin/larql repl 2> probe/repl-pty.err
      - name: One-shot lql, for comparison
        run: |
          timeout 60 ./bin/larql lql 'SHOW MODELS; STATS;' \
            > probe/lql.out 2> probe/lql.err
          echo "exit=$?" >> probe/lql.err
      - name: All 38 subcommands — --help
        run: |
          set -uo pipefail
          for c in run chat pull model link list show slice publish rm bench \
                   dec-bench k3-ledger accuracy shannon serve repl lql extract \
                   extract-index build compile convert hf verify diag parity \
                   moe-locality recipe capabilities card query describe stats \
                   validate merge filter dev; do
            timeout 30 ./bin/larql "$c" --help > "probe/help.$c.out" 2> "probe/help.$c.err"
            echo "$c exit=$?" >> probe/help.index
          done
          cat probe/help.index
      - name: All 38 subcommands — the guessed real invocations
        run: |
          set -uo pipefail
          TMP=$(mktemp -d)
          # No vindex exists in this job. That is deliberate: it separates
          # "the argument shape is wrong" from "the input was missing", and
          # both answers are needed before Task 7 writes the corpus.
          while IFS= read -r line; do
            [ -z "$line" ] && continue
            id=${line%%|*}; rest=${line#*|}
            eval "set -- $rest"
            timeout 60 ./bin/larql "$@" > "probe/cmd.$id.out" 2> "probe/cmd.$id.err"
            echo "$id exit=$? argv=[$rest]" >> probe/cmd.index
          done <<EOF
extract|extract $TMP/nomodel -o $TMP/v.vindex --level all
extract-index|extract-index $TMP/nomodel -o $TMP/v2.vindex --level browse
convert|convert quantize q4k --input $TMP/v.vindex --output $TMP/q.vindex
build|build $TMP/Vindexfile -o $TMP/b.vindex
slice|slice $TMP/v.vindex --output $TMP/s.vindex --kind browse
compile|compile --base $TMP/nomodel --vindex $TMP/e.vlp --output $TMP/c
link|link $TMP/v.vindex
pull|pull chrishayuk/gemma-3-4b-it-vindex
model|model pull HuggingFaceTB/SmolLM2-135M
list|list
show|show $TMP/v.vindex
verify|verify $TMP/v.vindex
diag|diag $TMP/v.vindex
capabilities|capabilities
run|run $TMP/v.vindex --prompt hi --max-tokens 2
chat|chat $TMP/v.vindex --max-tokens 1
serve|serve $TMP/v.vindex --port 18080
bench|bench $TMP/v.vindex --tokens 2
dec-bench|dec-bench
k3-ledger|k3-ledger report
accuracy|accuracy $TMP/v.vindex
shannon|shannon score $TMP/v.vindex
parity|parity $TMP/v.vindex
moe-locality|moe-locality $TMP/v.vindex
publish|publish $TMP/v.vindex --repo example/does-not-exist
hf|hf upload $TMP/v.vindex --repo example/does-not-exist
recipe|recipe validate $TMP/recipe.yaml
card|card render $TMP/recipe.yaml
dev|dev
repl|repl
lql|lql "SHOW MODELS;"
query|query $TMP/graph.json --entity France
describe|describe $TMP/graph.json France
stats|stats $TMP/graph.json
validate|validate $TMP/graph.json
merge|merge $TMP/graph.json $TMP/graph.json -o $TMP/m.json
filter|filter $TMP/graph.json --min-confidence 0.5 -o $TMP/f.json
rm|rm $TMP/v.vindex
EOF
          cat probe/cmd.index
      - name: Show every capture
        if: always()
        run: |
          for f in probe/*; do
            echo "───── $f ($(wc -c < "$f") bytes)"
            head -40 "$f" || true
          done
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: probe
          retention-days: 1
          path: probe/
```

The `serve`, `chat` and `repl` rows have no terminating input and are expected
to consume their 60s timeout. That is recorded, not avoided — Task 7 needs to
know which subcommands do not self-terminate.

- [ ] **Step 4: Verify the workflow parses**

Run:
```bash
cd /home/metavacua/larql-vindex3-03-08-2026
python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/lql-strategy-matrix.yml')); print('jobs:', list(d['jobs'].keys()))"
```
Expected: the job list includes `probe`.

- [ ] **Step 5: Commit and push**

```bash
git add .github/workflows/lql-strategy-matrix.yml scripts/lql_matrix/probe_pty.py
git commit -m "ci: probe the three riskiest assumptions before building on them

38 CLI invocations with guessed arguments, larql repl on non-tty stdin,
and a pty.fork read loop. No harness code — it runs things and uploads
what they printed. A guessed argument that is wrong should cost a
capture to discover, not a corpus and a sequencer built on top of it."
git push
```

- [ ] **Step 6: Read every capture before writing any code**

```bash
gh run download <run-id> --repo metavacua/larql-to-sparql --name probe --dir /tmp/probe
cat /tmp/probe/help.index
cat /tmp/probe/cmd.index
for f in /tmp/probe/repl-*; do echo "── $f ($(wc -c < "$f") bytes)"; cat "$f"; done
```

Record the answers; each one determines later tasks:

| observation | consequence |
|---|---|
| a `--help` exits non-zero or prints nothing | a real CLI finding; the Task 6 corpus still includes it |
| `cmd.<id>` stderr says an argument or flag is unknown | **the guess was wrong** — Task 7's corpus uses the corrected argv, not the guess |
| `cmd.<id>` stderr says the input file is missing | the argument shape is right; Task 7 keeps it and declares the `needs` |
| `repl-pipe.out` has `SHOW MODELS` output | piping works; `repl-pipe` proceeds as designed |
| `repl-pipe.out` empty, exit 0 | rustyline read nothing; the driver still ships (a capture of nothing is the finding) and Task 4 expects empty |
| `repl-pipe` hit its timeout | rustyline blocked on non-tty; a real larql finding, and the driver needs its timeout |
| `repl-pty.merged` non-empty but `repl-pipe.out` empty | pipe and pty diverge — exactly why the spec runs both |
| `repl-pty.merged` empty or `probe_pty.py` printed TIMEOUT | **the read loop is wrong** — fix it here; Task 4 lifts this loop verbatim |

**Do not start Task 1 until `cmd.index`, `help.index` and the four `repl-*`
captures have been read.** The purpose of this task is that the plan's guesses
fail here, cheaply, rather than nine tasks later.

---

### Task 1: Fix the six malformed `BEGIN PATCH` cells

`parse_begin` in `crates/larql-lql/src/parser/patch.rs` calls `expect_string()`, so the path is mandatory. Six cells send `BEGIN PATCH;` and have never opened a named patch session — every one produces `Parse error: expected string literal, got Semicolon`, and because a cell is a batch, that error is the first in the batch and masks everything after it.

**Files:**
- Create: `scripts/lql_matrix/corpus_lint.py`
- Create: `scripts/lql_matrix/corpus_lint_test.py`
- Modify: `scripts/lql_matrix/commands.jsonl`

**Interfaces:**
- Consumes: nothing.
- Produces: `corpus_lint.lint_lql_corpus(path: str) -> list[str]` returning a list of human-readable problems, empty when clean. Task 5 reuses it for the CLI corpora via `lint_cli_corpus`.

- [ ] **Step 1: Write the failing test**

Create `scripts/lql_matrix/corpus_lint_test.py`:

```python
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import corpus_lint as L

HERE = os.path.dirname(__file__)


def test_bare_begin_patch_is_reported(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"id": "x", "cat": "patch", "lql": 'USE "v"; BEGIN PATCH;'}) + "\n",
                 encoding="utf-8")
    problems = L.lint_lql_corpus(str(p))
    assert any("BEGIN PATCH" in s for s in problems)


def test_named_begin_patch_is_clean(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"id": "x", "cat": "patch",
                             "lql": 'USE "v"; BEGIN PATCH "{{TMP}}/x.vlp";'}) + "\n",
                 encoding="utf-8")
    assert L.lint_lql_corpus(str(p)) == []


def test_missing_required_key_is_reported(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"id": "x"}) + "\n", encoding="utf-8")
    problems = L.lint_lql_corpus(str(p))
    assert any("lql" in s for s in problems)


def test_duplicate_id_is_reported(tmp_path):
    p = tmp_path / "c.jsonl"
    rows = [json.dumps({"id": "dup", "cat": "a", "lql": "STATS;"}),
            json.dumps({"id": "dup", "cat": "a", "lql": "STATS;"})]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert any("dup" in s for s in L.lint_lql_corpus(str(p)))


def test_the_real_lql_corpus_is_clean():
    # The shipped corpus must never regress to malformed input. A malformed
    # probe is a harness defect, not a finding about larql.
    assert L.lint_lql_corpus(os.path.join(HERE, "commands.jsonl")) == []
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest corpus_lint_test.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'corpus_lint'`

- [ ] **Step 3: Write the minimal implementation**

Create `scripts/lql_matrix/corpus_lint.py`:

```python
#!/usr/bin/env python3
"""Validate a corpus file's shape.

A malformed probe is a harness defect, not a finding about larql. Six cells
in commands.jsonl sent `BEGIN PATCH;` — the grammar requires a path
(parser/patch.rs `parse_begin` calls `expect_string()`), so those cells never
opened a named patch session, and because a cell is a batch of statements the
resulting parse error was the FIRST error and masked everything after it.

This module exists so that class of defect fails locally in milliseconds
instead of consuming a CI run and then being misread as a product finding.
"""
import json
import re

# `BEGIN PATCH` with no quoted path. Matches `BEGIN PATCH;` and `BEGIN PATCH `
# at end of statement, not `BEGIN PATCH "file.vlp";`.
_BARE_BEGIN_PATCH = re.compile(r"BEGIN\s+PATCH\s*(;|$)", re.IGNORECASE)

_LQL_REQUIRED = ("id", "cat", "lql")


def _rows(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            out.append((n, json.loads(line)))
    return out


def lint_lql_corpus(path):
    """Return a list of problems in an LQL corpus. Empty list means clean."""
    problems = []
    seen = {}
    for n, row in _rows(path):
        for key in _LQL_REQUIRED:
            if key not in row:
                problems.append(f"{path}:{n}: missing required key {key!r}")
        cid = row.get("id")
        if cid in seen:
            problems.append(f"{path}:{n}: duplicate id {cid!r} (first at line {seen[cid]})")
        elif cid is not None:
            seen[cid] = n
        lql = row.get("lql", "")
        if _BARE_BEGIN_PATCH.search(lql):
            problems.append(
                f"{path}:{n}: cell {cid!r} sends `BEGIN PATCH` with no path — "
                "the grammar requires one, so this cell can never open a patch "
                'session. Use BEGIN PATCH "{{TMP}}/<name>.vlp";')
    return problems
```

- [ ] **Step 4: Run the tests — four pass, the real-corpus one fails**

Run: `cd scripts/lql_matrix && python3 -m pytest corpus_lint_test.py -q`
Expected: 4 passed, 1 failed — `test_the_real_lql_corpus_is_clean` reports six cells.

- [ ] **Step 5: Fix the six cells**

Run this rewrite, which gives each cell its own `.vlp` under the substituted tmp dir so cells cannot collide:

```bash
cd /home/metavacua/larql-vindex3-03-08-2026
python3 - <<'PY'
import json, re
p = "scripts/lql_matrix/commands.jsonl"
out = []
fixed = []
for line in open(p, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    new = re.sub(r"BEGIN\s+PATCH\s*;",
                 'BEGIN PATCH "{{TMP}}/%s.vlp";' % d["id"], d["lql"])
    if new != d["lql"]:
        fixed.append(d["id"])
        d["lql"] = new
    out.append(json.dumps(d))
open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("fixed:", fixed)
PY
```

Expected output: `fixed: ['insert.in_patch', 'patch.begin', 'patch.begin_insert_save', 'compile.into_vindex', 'compile.into_model', 'roundtrip.insert_compile_infer']`

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest -q`
Expected: PASS — 64 tests (59 existing + 5 new).

- [ ] **Step 7: Commit**

```bash
git add scripts/lql_matrix/corpus_lint.py scripts/lql_matrix/corpus_lint_test.py scripts/lql_matrix/commands.jsonl
git commit -m "harness: fix six cells that sent BEGIN PATCH with no path

parse_begin calls expect_string(), so the path is mandatory. These six
cells have never opened a named patch session; each produced a parse
error that, being first in a multi-statement batch, masked every later
error in the cell. corpus_lint catches the class in milliseconds instead
of a CI run."
```

---

### Task 2: Driver abstraction — argv and stdin for one cell

**Files:**
- Create: `scripts/lql_matrix/drivers.py`
- Create: `scripts/lql_matrix/drivers_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `drivers.DRIVERS: tuple[str, ...]` = `("lql", "repl-pipe", "repl-pty", "cli")`
  - `drivers.build(driver: str, cell: dict, larql: str) -> tuple[list[str], bytes | None]` returning `(argv, stdin_bytes)`. `stdin_bytes` is `None` when the driver writes no stdin. Task 3 calls this from `run_matrix.py`.
  - `drivers.split_statements(lql: str) -> list[str]` — splits a cell's `lql` on `;` at top level, preserving quoted strings, each returned statement ending in `;`.

- [ ] **Step 1: Write the failing test**

Create `scripts/lql_matrix/drivers_test.py`:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import drivers as D

CELL_LQL = {"id": "c", "cat": "x", "lql": 'USE "v"; STATS;'}
CELL_CLI = {"id": "c", "cat": "x", "argv": ["verify", "{{VINDEX}}"]}


def test_split_statements_basic():
    assert D.split_statements('USE "v"; STATS;') == ['USE "v";', 'STATS;']


def test_split_statements_preserves_semicolon_in_string():
    assert D.split_statements('WALK "a; b" TOP 5;') == ['WALK "a; b" TOP 5;']


def test_split_statements_adds_missing_trailing_semicolon():
    assert D.split_statements("STATS") == ["STATS;"]


def test_lql_driver_is_one_shot_batch():
    argv, stdin = D.build("lql", CELL_LQL, "/bin/larql")
    assert argv == ["/bin/larql", "lql", 'USE "v"; STATS;']
    assert stdin is None


def test_repl_pipe_sends_statements_on_stdin_one_per_line():
    argv, stdin = D.build("repl-pipe", CELL_LQL, "/bin/larql")
    assert argv == ["/bin/larql", "repl"]
    assert stdin == b'USE "v";\nSTATS;\n'


def test_repl_pty_appends_exit_because_a_terminal_has_no_eof():
    argv, stdin = D.build("repl-pty", CELL_LQL, "/bin/larql")
    assert argv == ["/bin/larql", "repl"]
    assert stdin == b'USE "v";\nSTATS;\nexit\n'


def test_cli_driver_uses_argv_verbatim():
    argv, stdin = D.build("cli", CELL_CLI, "/bin/larql")
    assert argv == ["/bin/larql", "verify", "{{VINDEX}}"]
    assert stdin is None


def test_unknown_driver_raises():
    import pytest
    with pytest.raises(ValueError):
        D.build("nope", CELL_LQL, "/bin/larql")


# ── The full 4 drivers x 2 cell-shapes matrix. Four accept, four reject.
#
#              lql-cell (has "lql")      cli-cell (has "argv")
#   lql        accept                    reject (KeyError)
#   repl-pipe  accept                    reject (KeyError)
#   repl-pty   accept                    reject (KeyError)
#   cli        reject (KeyError)         accept
#
# The four acceptances are the tests above. The four rejections follow.
# A driver must never fabricate a missing field: a fabricated invocation
# would be captured and read as a real result.

def test_lql_cell_under_cli_driver_raises():
    import pytest
    with pytest.raises(KeyError):
        D.build("cli", CELL_LQL, "/bin/larql")


def test_cli_cell_under_lql_driver_raises():
    import pytest
    with pytest.raises(KeyError):
        D.build("lql", CELL_CLI, "/bin/larql")


def test_cli_cell_under_repl_pipe_driver_raises():
    import pytest
    with pytest.raises(KeyError):
        D.build("repl-pipe", CELL_CLI, "/bin/larql")


def test_cli_cell_under_repl_pty_driver_raises():
    import pytest
    with pytest.raises(KeyError):
        D.build("repl-pty", CELL_CLI, "/bin/larql")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest drivers_test.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'drivers'`

- [ ] **Step 3: Write the implementation**

Create `scripts/lql_matrix/drivers.py`:

```python
#!/usr/bin/env python3
"""Build the argv and stdin for one cell under one driver.

Four drivers over the same corpora, because a driver is a distinct user
surface and none covers another:

  lql        larql lql "<statements>"   one-shot batch
  repl-pipe  larql repl, statements on stdin, non-tty
  repl-pty   larql repl under a pseudo-terminal
  cli        the subcommand invoked directly

repl-pty appends `exit` because a terminal has no EOF: without it the session
would run to the cell timeout every time.

This module knows nothing about corpora, sequencing, or capture. It maps a
cell to an invocation and stops.
"""

DRIVERS = ("lql", "repl-pipe", "repl-pty", "cli")


def split_statements(lql):
    """Split on top-level `;`, preserving quoted strings. Each returned
    statement ends with `;`. Mirrors larql's own splitter in
    crates/larql-lql/src/repl.rs so the REPL drivers send exactly the
    statements the batch driver would."""
    stmts, cur = [], []
    in_string, quote = False, '"'
    for ch in lql:
        if in_string:
            cur.append(ch)
            if ch == quote:
                in_string = False
        elif ch in ('"', "'"):
            in_string, quote = True, ch
            cur.append(ch)
        elif ch == ";":
            cur.append(ch)
            stmts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        stmts.append(tail if tail.endswith(";") else tail + ";")
    return [s for s in stmts if s]


def build(driver, cell, larql):
    """Return (argv, stdin_bytes). stdin_bytes is None when nothing is written.

    Raises ValueError on an unknown driver and KeyError when the cell lacks
    the field the driver needs — never fabricates an invocation, because a
    fabricated one would be captured and read as a real result.
    """
    if driver == "lql":
        return [larql, "lql", cell["lql"]], None
    if driver in ("repl-pipe", "repl-pty"):
        lines = split_statements(cell["lql"])
        if driver == "repl-pty":
            lines = lines + ["exit"]
        return [larql, "repl"], ("\n".join(lines) + "\n").encode("utf-8")
    if driver == "cli":
        return [larql, *cell["argv"]], None
    raise ValueError(f"unknown driver {driver!r}; expected one of {DRIVERS}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest drivers_test.py -q`
Expected: PASS — 12 tests: 3 splitter, 4 acceptances and 4 rejections covering
the whole driver x cell-shape matrix, plus the unknown-driver rejection.

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/drivers.py scripts/lql_matrix/drivers_test.py
git commit -m "harness: driver abstraction — argv and stdin per cell

Four drivers over the same corpora: lql one-shot batch, repl piped,
repl under a pty, and the CLI subcommand direct. A driver is a distinct
user surface and none is assumed to cover another. repl-pty appends
exit because a terminal has no EOF."
```

---

### Task 3: Wire the driver into `run_matrix.py`, `lql` behaviour unchanged

**Files:**
- Modify: `scripts/lql_matrix/run_matrix.py`
- Modify: `scripts/lql_matrix/run_matrix.sh`
- Create: `scripts/lql_matrix/run_matrix_test.py`

**Interfaces:**
- Consumes: `drivers.build(driver, cell, larql)`, `drivers.DRIVERS` (Task 2).
- Produces: `run_matrix.py <level> <vindex> <corpus> <out> [--driver NAME]`, default `lql`. Capture files become `<cells>/<level>.<driver>.<cell_id>.{out,err}`. Rows gain `"driver"`. Task 4 adds the pty path; Tasks 6–8 add the `cli` driver and sequencing.

- [ ] **Step 1: Write the failing test**

Create `scripts/lql_matrix/run_matrix_test.py`:

```python
import json
import os
import stat
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FAKE_LARQL = """#!/usr/bin/env bash
# Fake larql: echoes its argv and any stdin, so the harness can be tested
# without building the real binary.
echo "ARGV: $*"
if [ ! -t 0 ]; then while IFS= read -r l; do echo "STDIN: $l"; done; fi
echo "to stderr" >&2
exit 3
"""


def _fake_bin(tmp_path):
    p = tmp_path / "larql"
    p.write_text(FAKE_LARQL, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _corpus(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(json.dumps({"id": "c1", "cat": "x", "lql": 'USE "{{VINDEX}}"; STATS;'}) + "\n",
                 encoding="utf-8")
    return str(p)


def _run(tmp_path, driver):
    out = tmp_path / "results.jsonl"
    env = dict(os.environ, LARQL_BIN=_fake_bin(tmp_path), CELL_TIMEOUT="30")
    argv = [sys.executable, os.path.join(HERE, "run_matrix.py"),
            "leg1", "/nonexistent.vindex", _corpus(tmp_path), str(out)]
    if driver:
        argv += ["--driver", driver]
    subprocess.run(argv, env=env, check=True, capture_output=True)
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("type") != "meta"], tmp_path


def test_default_driver_is_lql_and_row_records_it(tmp_path):
    rows, _ = _run(tmp_path, None)
    assert len(rows) == 1
    assert rows[0]["driver"] == "lql"
    assert rows[0]["exit_code"] == 3


def test_capture_files_are_named_by_driver_and_contain_full_output(tmp_path):
    rows, tp = _run(tmp_path, "lql")
    out_path = tp / rows[0]["stdout"]
    assert out_path.exists()
    text = out_path.read_text()
    assert "ARGV: lql" in text
    assert (tp / rows[0]["stderr"]).read_text().strip() == "to stderr"


def test_row_carries_no_derived_opinion(tmp_path):
    rows, _ = _run(tmp_path, "lql")
    forbidden = {"bucket", "err_signal", "err_line", "stdout_head",
                 "stderr_head", "stderr_tail", "status", "ok", "passed"}
    assert forbidden.isdisjoint(rows[0].keys())


def test_repl_pipe_writes_statements_to_stdin(tmp_path):
    rows, tp = _run(tmp_path, "repl-pipe")
    text = (tp / rows[0]["stdout"]).read_text()
    assert "ARGV: repl" in text
    assert 'STDIN: USE "/nonexistent.vindex";' in text
    assert "STDIN: STATS;" in text


def test_unknown_driver_is_rejected(tmp_path):
    out = tmp_path / "r.jsonl"
    env = dict(os.environ, LARQL_BIN=_fake_bin(tmp_path))
    r = subprocess.run([sys.executable, os.path.join(HERE, "run_matrix.py"),
                        "leg1", "/v", _corpus(tmp_path), str(out), "--driver", "nope"],
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest run_matrix_test.py -q`
Expected: FAIL — `run_matrix.py` does not accept `--driver` (`ValueError: too many values to unpack`).

- [ ] **Step 3: Rewrite `run_matrix.py`'s argument handling and cell loop**

Replace the whole `def main() -> None:` body in `scripts/lql_matrix/run_matrix.py` with:

```python
def main() -> None:
    ap = argparse.ArgumentParser(description="Run a corpus against one vindex and capture everything.")
    ap.add_argument("level")
    ap.add_argument("vindex")
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--driver", default="lql", choices=drivers.DRIVERS)
    ns = ap.parse_args()
    level, vindex, corpus, out, driver = ns.level, ns.vindex, ns.corpus, ns.out, ns.driver

    larql = os.environ.get("LARQL_BIN", "target/release/larql")
    model = os.environ.get("MODEL_ID", "")
    tmproot = os.environ.get("TMPROOT") or tempfile.mkdtemp()
    wrap = shlex.split(os.environ.get("WRAP", ""))
    cell_timeout = os.environ.get("CELL_TIMEOUT", "900")
    os.makedirs(tmproot, exist_ok=True)

    out_path = pathlib.Path(out)
    cells_dir = out_path.parent / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    time_bin = "/usr/bin/time" if pathlib.Path("/usr/bin/time").exists() else None

    def subst(s: str) -> str:
        return (s.replace("{{VINDEX}}", vindex)
                 .replace("{{MODEL}}", model)
                 .replace("{{TMP}}", tmproot))

    try:
        ver = subprocess.run([larql, "--version"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
    except Exception as e:
        ver = f"<--version failed: {type(e).__name__}: {e}>"
    meta = {
        "type": "meta", "level": level, "driver": driver, "larql_version": ver,
        "commit": os.environ.get("GITHUB_SHA", ""),
        "model": model, "runner_os": os.environ.get("RUNNER_OS", ""),
        "vindex": vindex, "backtrace": os.environ.get("RUST_BACKTRACE", ""),
        "time_bin": bool(time_bin), "wrap": " ".join(wrap),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")

    n = 0
    with open(corpus, encoding="utf-8") as cf:
        cells = [json.loads(l) for l in cf if l.strip()]

    for c in cells:
        cid, cat = c["id"], c.get("cat", "")
        cell = dict(c)
        if "lql" in cell:
            cell["lql"] = subst(cell["lql"])
        if "argv" in cell:
            cell["argv"] = [subst(a) for a in cell["argv"]]

        argv_cmd, stdin_bytes = drivers.build(driver, cell, larql)

        outf = cells_dir / f"{level}.{driver}.{cid}.out"
        errf = cells_dir / f"{level}.{driver}.{cid}.err"
        timef = cells_dir / f"{level}.{driver}.{cid}.time"

        argv = ["timeout", "--kill-after=10", cell_timeout, *wrap]
        if time_bin:
            argv += [time_bin, "-v", "-o", str(timef)]
        argv += argv_cmd

        t0 = time.monotonic()
        with outf.open("wb") as so, errf.open("wb") as se:
            rc = subprocess.run(argv, stdout=so, stderr=se,
                                input=stdin_bytes).returncode
        dur_ms = int((time.monotonic() - t0) * 1000)

        peak_rss_kb = ""
        if time_bin and timef.exists():
            m = RSS_RE.search(timef.read_text("utf-8", "replace"))
            if m:
                peak_rss_kb = int(m.group(1))
            timef.unlink()

        row = {
            "level": level, "driver": driver, "id": cid, "cat": cat,
            "sent": cell.get("lql") or cell.get("argv"),
            "exit_code": rc, "duration_ms": dur_ms,
            "peak_rss_kb": peak_rss_kb,
            "stdout": str(outf.relative_to(out_path.parent)),
            "stderr": str(errf.relative_to(out_path.parent)),
            "stdout_bytes": outf.stat().st_size,
            "stderr_bytes": errf.stat().st_size,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(f"[{level}/{driver}] {cid} -> exit={rc} {dur_ms}ms "
              f"rss={peak_rss_kb} out={row['stdout_bytes']}B "
              f"err={row['stderr_bytes']}B", file=sys.stderr)
        n += 1

    print(f"wrote {n} rows + provenance to {out}", file=sys.stderr)
```

Then add to the import block at the top of the file, after `import time`:

```python

import drivers
```

and add `import argparse` to the alphabetical import list (before `import datetime`).

Finally, add this immediately above `def main() -> None:` so the module resolves its siblings when invoked by path:

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

- [ ] **Step 4: Update the shim's documented API**

In `scripts/lql_matrix/run_matrix.sh`, replace the line reading
`#   LARQL_BIN  MODEL_ID  TMPROOT  WRAP  CELL_TIMEOUT` with:

```bash
#   LARQL_BIN  MODEL_ID  TMPROOT  WRAP  CELL_TIMEOUT
# Trailing flags pass through, e.g. --driver repl-pipe (default lql).
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest -q`
Expected: PASS — 70 tests (64 + 6 new).

- [ ] **Step 6: Commit**

```bash
git add scripts/lql_matrix/run_matrix.py scripts/lql_matrix/run_matrix.sh scripts/lql_matrix/run_matrix_test.py
git commit -m "harness: run_matrix takes a --driver, default lql

Capture files are named by driver so the same cell under two drivers
does not collide, and the row records which driver ran. A test asserts
the row carries none of bucket/err_signal/err_line/stdout_head — the
derived fields whose removal this harness depends on."
```

- [ ] **Step 7: Demonstrate on a runner — the `lql` driver must be unchanged**

This task's whole claim is that adding `--driver` changed nothing for the
existing path. Prove it against the real binary before building on it.

In the workflow's `Run LQL command corpus` step, append `--driver lql` to the
`run_matrix.sh` invocation, and rename the output to
`out/results-${NAME}.lql.jsonl`; widen the upload glob to
`out/results-${{ matrix.leg.name }}.*.jsonl`. Then:

```bash
git add .github/workflows/lql-strategy-matrix.yml
git commit -m "ci: pass --driver lql explicitly; capture files gain the driver segment"
git push
```

Wait for the run, then download one leg and confirm the captures are the same
shape as before the change:

```bash
gh run download <run-id> --repo metavacua/larql-to-sparql --name results-smol135.native.all --dir /tmp/d
python3 -c "
import json,glob
rows=[json.loads(l) for f in glob.glob('/tmp/d/results-*.jsonl') for l in open(f) if l.strip()]
cells=[r for r in rows if r.get('type')!='meta']
print(len(cells),'cells; drivers:',{r['driver'] for r in cells})
assert all('bucket' not in r and 'err_line' not in r for r in cells), 'a derived field came back'
print('capture files present:', len(glob.glob('/tmp/d/cells/*')))"
```

Expected: 64 cells, `{'lql'}`, no derived fields, capture files named
`<leg>.lql.<cell>.{out,err}`. **Do not start Task 4 until this has run.**

---

### Task 4: The `repl-pty` driver

`subprocess` cannot give the child a tty. This task adds a pty runner used only when `--driver repl-pty`.

**Files:**
- Modify: `scripts/lql_matrix/run_matrix.py`
- Modify: `scripts/lql_matrix/run_matrix_test.py`

**Interfaces:**
- Consumes: `drivers.build("repl-pty", cell, larql)` returning stdin bytes ending in `exit\n` (Task 2).
- Produces: `run_matrix.run_under_pty(argv: list[str], stdin_bytes: bytes, merged_path: pathlib.Path, timeout_s: int) -> int` returning the child's exit status. Writes the single merged stream a terminal produces to `merged_path`.

- [ ] **Step 1: Write the failing test**

Append to `scripts/lql_matrix/run_matrix_test.py`:

```python
def test_run_under_pty_gives_the_child_a_tty_and_captures_merged(tmp_path):
    sys.path.insert(0, HERE)
    import pathlib
    import run_matrix as R
    script = tmp_path / "istty"
    script.write_text("#!/usr/bin/env bash\n"
                      "if [ -t 0 ]; then echo TTY_YES; else echo TTY_NO; fi\n"
                      "read -r line; echo \"GOT:$line\"\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    merged = tmp_path / "m.merged"
    rc = R.run_under_pty([str(script)], b"hello\n", pathlib.Path(merged), 30)
    text = merged.read_text(errors="replace")
    assert "TTY_YES" in text
    assert "GOT:hello" in text
    assert rc == 0


def test_repl_pty_driver_writes_a_merged_capture(tmp_path):
    rows, tp = _run(tmp_path, "repl-pty")
    assert rows[0]["driver"] == "repl-pty"
    assert "merged" in rows[0]
    assert (tp / rows[0]["merged"]).exists()
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest run_matrix_test.py -q -k pty`
Expected: FAIL — `AttributeError: module 'run_matrix' has no attribute 'run_under_pty'`

- [ ] **Step 3: Add the pty runner**

In `scripts/lql_matrix/run_matrix.py`, add `import errno`, `import fcntl`, `import pty`, `import select`, `import signal`, `import struct`, `import termios` to the import block, and add this function immediately above `def main() -> None:`:

```python
def run_under_pty(argv, stdin_bytes, merged_path, timeout_s):
    """Run argv with a pseudo-terminal on stdin/stdout/stderr, write everything
    the terminal carried to merged_path, and return the exit status.

    A pty is required because `larql repl` goes through rustyline, and rustyline
    on a pipe is not the same code path as rustyline on a terminal. Whether they
    behave identically is exactly what the repl-pipe and repl-pty legs exist to
    show, so the harness must be able to produce a real terminal.

    A terminal has ONE stream: stdout and stderr are interleaved as a user sees
    them. Control sequences and \\r are written through verbatim — stripping
    them would be post-processing, and what the terminal received is part of
    what happened.
    """
    pid, fd = pty.fork()
    if pid == 0:  # child
        try:
            os.execvp(argv[0], argv)
        except Exception:
            os._exit(127)

    # 80x24 so wrapping is deterministic across runners rather than inherited.
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    except OSError:
        pass

    if stdin_bytes:
        os.write(fd, stdin_bytes)

    deadline = time.monotonic() + timeout_s
    with open(merged_path, "wb") as mf:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.kill(pid, signal.SIGKILL)
                break
            r, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not r:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    break
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError as e:
                if e.errno == errno.EIO:   # pty closed: child exited
                    break
                raise
            if not chunk:
                break
            mf.write(chunk)
            mf.flush()

    os.close(fd)
    try:
        _, status = os.waitpid(pid, 0)
    except ChildProcessError:
        return -1
    return os.waitstatus_to_exitcode(status)
```

- [ ] **Step 4: Branch the cell loop onto the pty runner**

In `main()`, replace the block from `t0 = time.monotonic()` through the `dur_ms = ...` line with:

```python
        mergedf = cells_dir / f"{level}.{driver}.{cid}.merged"
        t0 = time.monotonic()
        if driver == "repl-pty":
            rc = run_under_pty(argv_cmd, stdin_bytes, mergedf, int(cell_timeout))
            outf.write_bytes(b"")   # a terminal has one stream; these exist so
            errf.write_bytes(b"")   # every row indexes the same three paths
        else:
            with outf.open("wb") as so, errf.open("wb") as se:
                rc = subprocess.run(argv, stdout=so, stderr=se,
                                    input=stdin_bytes).returncode
        dur_ms = int((time.monotonic() - t0) * 1000)
```

Then add to the `row` dict, immediately after the `"stderr_bytes"` entry:

```python
            "merged": str(mergedf.relative_to(out_path.parent)) if driver == "repl-pty" else None,
            "merged_bytes": mergedf.stat().st_size if driver == "repl-pty" else None,
```

Note the pty path does not use the `timeout`/`time` wrapper argv, because it
execs the child directly; `run_under_pty` enforces the timeout itself and peak
RSS is unavailable for this driver (the row records `""`, which is the existing
"not measured" value).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest -q`
Expected: PASS — 72 tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/lql_matrix/run_matrix.py scripts/lql_matrix/run_matrix_test.py
git commit -m "harness: repl-pty driver — a real pseudo-terminal

larql repl goes through rustyline, and rustyline on a pipe is not the
same code path as rustyline on a terminal. Whether they agree is what
the two repl legs exist to show, so the harness must be able to produce
a real tty. Control sequences and \\r are written through verbatim."
```

- [ ] **Step 7: Demonstrate both REPL drivers on a runner**

The fake binary cannot tell you what rustyline does. Run the two new drivers
against the real one on **a single leg** before extending to all of them.

In the workflow step edited in Task 3, replace the single invocation with a loop
over the three LQL drivers, hardcoded for this demonstration only (Task 5
replaces the hardcoded list with `matrix.leg.drivers`):

```yaml
          for DRV in lql repl-pipe repl-pty; do
            LARQL_BIN=./bin/larql MODEL_ID="${CORPUS_MODEL}" \
            TMPROOT="$(mktemp -d)" CELL_TIMEOUT=900 \
            scripts/lql_matrix/run_matrix.sh \
              "${NAME}" "out/${NAME}.vindex" scripts/lql_matrix/commands.jsonl \
              "out/results-${NAME}.${DRV}.jsonl" --driver "$DRV"
          done
```

```bash
git add .github/workflows/lql-strategy-matrix.yml
git commit -m "ci: run all three LQL drivers on every leg"
git push
```

Read one leg's captures and compare the three drivers on the same cell:

```bash
gh run download <run-id> --repo metavacua/larql-to-sparql --name results-smol135.native.all --dir /tmp/d
for drv in lql repl-pipe repl-pty; do
  echo "══════ $drv"
  ls /tmp/d/cells/ | grep "\.$drv\." | head -3
  f=$(ls /tmp/d/cells/*.$drv.describe.basic.* 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "── $f ($(wc -c < "$f") bytes)"; head -20 "$f"; }
done
```

What to check, and what each answer means:

- **All three produced non-empty captures for `describe.basic`** — the drivers
  work; proceed.
- **`repl-pipe` captures are empty while `lql` is not** — consistent with the
  Task 0 probe; the driver still ships, and this is a finding about larql's
  non-interactive REPL, not a harness bug to fix.
- **`repl-pty` capture is empty or the leg hit its timeout** — the pty runner is
  wrong. Return to Step 3 of this task; do not proceed to Task 5.
- **`repl-pty` contains `\r` and escape sequences** — correct and expected. They
  stay verbatim; stripping them is a Global Constraint violation.

**Do not start Task 5 until these captures have been read.**

---

### Task 5: Driver axis in `gen_legs.py`

**Files:**
- Modify: `scripts/lql_matrix/gen_legs.py`
- Modify: `scripts/lql_matrix/gen_legs_test.py`

**Interfaces:**
- Consumes: `drivers.DRIVERS` (Task 2).
- Produces: every leg dict gains `"drivers": list[str]` and `"long_session": bool`. Task 10's workflow reads both.

- [ ] **Step 1: Write the failing test**

Append to `scripts/lql_matrix/gen_legs_test.py`:

```python
def test_every_leg_declares_its_drivers():
    legs = G.build_legs()
    assert all(lg["drivers"] for lg in legs)
    assert all(set(lg["drivers"]) <= {"lql", "repl-pipe", "repl-pty", "cli"} for lg in legs)


def test_native_legs_run_all_three_lql_drivers():
    legs = {lg["name"]: lg for lg in G.build_legs()}
    assert set(legs["smol135.native.all"]["drivers"]) == {"lql", "repl-pipe", "repl-pty"}


def test_there_is_exactly_one_long_session_leg_per_model():
    legs = [lg for lg in G.build_legs() if lg["long_session"]]
    models = [lg["name"].split(".", 1)[0] for lg in legs]
    assert len(models) == len(set(models)), "one long-session leg per model"
    assert set(models) == {m for m, _ in G.SAFETENSORS}


def test_long_session_legs_use_a_repl_driver():
    # A long session is only meaningful where state persists; the one-shot
    # batch driver starts a fresh Session per invocation.
    for lg in G.build_legs():
        if lg["long_session"]:
            assert "lql" not in lg["drivers"]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest gen_legs_test.py -q`
Expected: FAIL — `KeyError: 'drivers'`

- [ ] **Step 3: Add the axis**

In `scripts/lql_matrix/gen_legs.py`, change the `leg()` signature and body to:

```python
def leg(name, hf, op, level="all", flags="",
        source_kind="safetensors", corpus_model=None, tokenizer_repo="",
        roundtrip=False, lql_with="", drivers=("lql",), long_session=False):
    return {"name": name, "hf": hf, "corpus_model": corpus_model or hf,
            "source_kind": source_kind, "op": op, "level": level, "flags": flags,
            "tokenizer_repo": tokenizer_repo,
            "roundtrip": roundtrip, "lql_with": lql_with,
            "drivers": list(drivers), "long_session": long_session}
```

In `build_legs()`, change the native-extraction loop to request all three LQL drivers:

```python
    # 1. NATIVE EXTRACTION — full level grid per model, native precision (f16).
    for mid, hf in SAFETENSORS:
        for lv in LEVELS:
            rt = mid in ("smol135", "smol135base") and lv in ("inference", "all")
            legs.append(leg(f"{mid}.native.{lv}", hf, "extract", level=lv,
                            roundtrip=rt,
                            drivers=("lql", "repl-pipe", "repl-pty")))
```

Then append, immediately before `return legs`:

```python
    # 4. LONG SESSION — the whole corpus through ONE repl session per model.
    #    The only thing that exercises state accumulating across commands.
    #    Ordering is load-bearing here and cells contaminate each other by
    #    design; that is the point, and it is why this is a separate leg
    #    rather than a replacement for the per-cell backbone.
    for mid, hf in SAFETENSORS:
        legs.append(leg(f"{mid}.longsession", hf, "extract", level="all",
                        drivers=("repl-pipe", "repl-pty"), long_session=True))

    # 5. CLI SURFACE — every subcommand, help and real invocation. Driven per
    #    model because most subcommands need a produced vindex.
    for mid, hf in SAFETENSORS:
        legs.append(leg(f"{mid}.cli", hf, "extract", level="all",
                        drivers=("cli",)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest -q`
Expected: PASS — 76 tests.

- [ ] **Step 5: Verify the leg count under the active filter**

Run:
```bash
cd scripts/lql_matrix && LQL_MATRIX_ONLY="smol135,smol135base" python3 gen_legs.py \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d),'legs'); [print(' ',l['name'],l['drivers'],'long' if l['long_session'] else '') for l in d]"
```
Expected: 25 legs — the 21 existing plus `smol135.longsession`, `smol135base.longsession`, `smol135.cli`, `smol135base.cli`.

- [ ] **Step 6: Commit**

```bash
git add scripts/lql_matrix/gen_legs.py scripts/lql_matrix/gen_legs_test.py
git commit -m "harness: driver axis, long-session legs, cli legs

Native legs now declare all three LQL drivers. One long-session leg per
model sends the whole corpus through a single repl session — the only
thing that exercises cross-command state, and excluded from the lql
driver because a one-shot batch starts a fresh Session each time."
```

---

> **Pause point.** Tasks 1–5 deliver working REPL coverage on their own. Tasks 6–10 add the CLI surface and dependency sequencing.

---

### Task 6: `cli-help.jsonl` — every subcommand's help

**Files:**
- Create: `scripts/lql_matrix/cli-help.jsonl`
- Modify: `scripts/lql_matrix/corpus_lint.py`
- Modify: `scripts/lql_matrix/corpus_lint_test.py`

**Interfaces:**
- Consumes: `corpus_lint._rows` (Task 1).
- Produces: `corpus_lint.lint_cli_corpus(path: str, require_deps: bool) -> list[str]` and `corpus_lint.SUBCOMMANDS: tuple[str, ...]` — the 38 lowercase subcommand names. Task 7 reuses both.

- [ ] **Step 1: Write the failing test**

Append to `scripts/lql_matrix/corpus_lint_test.py`:

```python
def test_cli_corpus_requires_argv(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"id": "x", "cat": "cli"}) + "\n", encoding="utf-8")
    assert any("argv" in s for s in L.lint_cli_corpus(str(p), require_deps=False))


def test_cli_help_corpus_covers_every_subcommand():
    path = os.path.join(HERE, "cli-help.jsonl")
    assert L.lint_cli_corpus(path, require_deps=False) == []
    seen = {json.loads(l)["argv"][0] for l in open(path, encoding="utf-8") if l.strip()}
    assert seen == set(L.SUBCOMMANDS), f"missing: {set(L.SUBCOMMANDS) - seen}"


def test_subcommand_list_matches_the_rust_source():
    # If someone adds a subcommand to main.rs, this fails until the corpus
    # covers it. A subcommand nothing invokes is a subcommand nothing tests.
    import re
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    src = open(os.path.join(root, "crates/larql-cli/src/main.rs"), encoding="utf-8").read()
    body = re.search(r"enum Commands \{(.*?)\n\}", src, re.S).group(1)
    variants = re.findall(r"^\s{4}([A-Z]\w+)", body, re.M)
    def kebab(v):
        return re.sub(r"(?<!^)(?=[A-Z])", "-", v).lower()
    assert sorted(kebab(v) for v in variants) == sorted(L.SUBCOMMANDS)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest corpus_lint_test.py -q`
Expected: FAIL — `AttributeError: module 'corpus_lint' has no attribute 'lint_cli_corpus'`

- [ ] **Step 3: Add the CLI lint and the subcommand list**

Append to `scripts/lql_matrix/corpus_lint.py`:

```python
# Every subcommand in crates/larql-cli/src/main.rs `enum Commands`, in clap's
# kebab-case form. Kept here as data so a test can compare it against the Rust
# source: a subcommand nothing invokes is a subcommand nothing tests.
SUBCOMMANDS = (
    "run", "chat", "pull", "model", "link", "list", "show", "slice", "publish",
    "rm", "bench", "dec-bench", "k3-ledger", "accuracy", "shannon", "serve",
    "repl", "lql", "extract", "extract-index", "build", "compile", "convert",
    "hf", "verify", "diag", "parity", "moe-locality", "recipe", "capabilities",
    "card", "query", "describe", "stats", "validate", "merge", "filter", "dev",
)

_CLI_REQUIRED = ("id", "cat", "argv")


def lint_cli_corpus(path, require_deps):
    """Return a list of problems in a CLI corpus. Empty list means clean.

    require_deps=True additionally demands `needs` and `produces` lists, which
    the sequencer uses to order cells.
    """
    problems = []
    seen = {}
    for n, row in _rows(path):
        for key in _CLI_REQUIRED:
            if key not in row:
                problems.append(f"{path}:{n}: missing required key {key!r}")
        cid = row.get("id")
        if cid in seen:
            problems.append(f"{path}:{n}: duplicate id {cid!r} (first at line {seen[cid]})")
        elif cid is not None:
            seen[cid] = n
        argv = row.get("argv")
        if isinstance(argv, list) and argv and argv[0] not in SUBCOMMANDS:
            problems.append(f"{path}:{n}: {argv[0]!r} is not a larql subcommand")
        if require_deps:
            for key in ("needs", "produces"):
                if not isinstance(row.get(key), list):
                    problems.append(f"{path}:{n}: cell {cid!r} missing {key!r} list")
    return problems
```

- [ ] **Step 4: Generate the help corpus**

Run:
```bash
cd /home/metavacua/larql-vindex3-03-08-2026
python3 - <<'PY'
import json, sys
sys.path.insert(0, "scripts/lql_matrix")
from corpus_lint import SUBCOMMANDS
rows = [{"id": f"help.{c}", "cat": "cli-help", "argv": [c, "--help"]} for c in SUBCOMMANDS]
with open("scripts/lql_matrix/cli-help.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows), "help cells")
PY
```
Expected: `38 help cells`

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest -q`
Expected: PASS — 79 tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/lql_matrix/cli-help.jsonl scripts/lql_matrix/corpus_lint.py scripts/lql_matrix/corpus_lint_test.py
git commit -m "harness: --help cell for all 38 subcommands

A test compares the subcommand list against enum Commands in main.rs,
so adding a subcommand fails the harness until a cell covers it. The
matrix previously drove 4 of 38."
```

- [ ] **Step 7: Demonstrate 38 real `--help` invocations on a runner**

This is the cheapest possible real exercise of the `cli` driver — no vindex, no
model, no network — so it isolates "does the cli driver invoke correctly" from
"do the subcommands work".

Add to the workflow step, inside the driver loop added in Task 4:

```yaml
            if [ "$DRV" = "cli" ]; then
              scripts/lql_matrix/run_matrix.sh "${NAME}" "out/${NAME}.vindex" \
                scripts/lql_matrix/cli-help.jsonl \
                "out/results-${NAME}.cli.help.jsonl" --driver cli
            fi
```

and add `cli` to the hardcoded driver list for this demonstration.

```bash
git add .github/workflows/lql-strategy-matrix.yml
git commit -m "ci: run the 38 --help cells through the cli driver"
git push
```

Read the result:

```bash
gh run download <run-id> --repo metavacua/larql-to-sparql --name results-smol135.native.all --dir /tmp/d
python3 -c "
import json,glob
rows=[json.loads(l) for f in glob.glob('/tmp/d/results-*.cli.help.jsonl') for l in open(f) if l.strip()]
cells=[r for r in rows if r.get('type')!='meta']
print(len(cells),'help cells')
bad=[(r['id'],r['exit_code']) for r in cells if r['exit_code']!=0]
print('non-zero exits:',bad)
empty=[r['id'] for r in cells if r['stdout_bytes']==0]
print('empty stdout:',empty)"
```

Expected: 38 cells. A subcommand whose `--help` exits non-zero or prints nothing
is a real finding about the CLI — record it, do not "fix" the cell to make it
pass. **Do not start Task 7 until these have been read.**

---

### Task 7: `cli-commands.jsonl` — real invocations with declared dependencies

**Files:**
- Create: `scripts/lql_matrix/cli-commands.jsonl`
- Modify: `scripts/lql_matrix/corpus_lint_test.py`

**Interfaces:**
- Consumes: `corpus_lint.lint_cli_corpus(path, require_deps=True)`, `corpus_lint.SUBCOMMANDS` (Task 6).
- Produces: a corpus whose cells each carry `needs: list[str]` and `produces: list[str]` over the artifact vocabulary `{"vindex", "cache-entry", "graph-file", "recipe", "vlp"}`. Task 8's sequencer consumes it.

- [ ] **Step 1: Write the failing test**

Append to `scripts/lql_matrix/corpus_lint_test.py`:

```python
def test_cli_commands_corpus_is_clean_and_covers_every_subcommand():
    path = os.path.join(HERE, "cli-commands.jsonl")
    assert L.lint_cli_corpus(path, require_deps=True) == []
    seen = {json.loads(l)["argv"][0] for l in open(path, encoding="utf-8") if l.strip()}
    assert seen == set(L.SUBCOMMANDS), f"never invoked: {sorted(set(L.SUBCOMMANDS) - seen)}"


def test_cli_commands_artifacts_come_from_a_closed_vocabulary():
    path = os.path.join(HERE, "cli-commands.jsonl")
    vocab = {"vindex", "cache-entry", "graph-file", "recipe", "vlp"}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        assert set(d["needs"]) <= vocab, f"{d['id']}: unknown needs {d['needs']}"
        assert set(d["produces"]) <= vocab, f"{d['id']}: unknown produces {d['produces']}"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest corpus_lint_test.py -q -k cli_commands`
Expected: FAIL — `FileNotFoundError: cli-commands.jsonl`

- [ ] **Step 3: Write the corpus**

Create `scripts/lql_matrix/cli-commands.jsonl` with one line per subcommand.

**Start from Task 0's `cmd.index` and `cmd.*.err` captures, not from the rows
below.** Task 0 ran every one of these argument shapes against the real binary
precisely so this corpus is written from what larql said rather than guessed a
second time. For each subcommand:

- stderr named an unknown flag or argument → **use the corrected argv**, taken
  from that subcommand's `help.<name>.out`
- stderr named a missing input file → the shape is right; keep it and declare
  the `needs`
- it consumed its timeout → keep it, and note it in `cat` as non-terminating

The rows below are the shapes Task 0 probed. Where a capture showed the shape
was wrong, the corrected row replaces it here — the guess does not survive into
the corpus.

(`{{VINDEX}}`, `{{MODEL}}` and `{{TMP}}` are substituted by `run_matrix.py`.)

```json
{"id": "cli.extract", "cat": "produce", "argv": ["extract", "{{MODEL}}", "-o", "{{TMP}}/cli.vindex", "--level", "all"], "needs": [], "produces": ["vindex"]}
{"id": "cli.extract-index", "cat": "produce", "argv": ["extract-index", "{{MODEL}}", "-o", "{{TMP}}/cli-idx.vindex", "--level", "browse"], "needs": [], "produces": ["vindex"]}
{"id": "cli.convert", "cat": "produce", "argv": ["convert", "quantize", "q4k", "--input", "{{VINDEX}}", "--output", "{{TMP}}/cli-q4k.vindex"], "needs": ["vindex"], "produces": ["vindex"]}
{"id": "cli.build", "cat": "produce", "argv": ["build", "{{TMP}}/Vindexfile", "-o", "{{TMP}}/cli-build.vindex"], "needs": [], "produces": ["vindex"]}
{"id": "cli.slice", "cat": "produce", "argv": ["slice", "{{VINDEX}}", "--output", "{{TMP}}/cli-slice.vindex", "--kind", "browse"], "needs": ["vindex"], "produces": ["vindex"]}
{"id": "cli.compile", "cat": "produce", "argv": ["compile", "--base", "{{MODEL}}", "--vindex", "{{TMP}}/empty.vlp", "--output", "{{TMP}}/cli-compiled"], "needs": ["vlp"], "produces": []}
{"id": "cli.link", "cat": "register", "argv": ["link", "{{VINDEX}}"], "needs": ["vindex"], "produces": ["cache-entry"]}
{"id": "cli.pull", "cat": "register", "argv": ["pull", "chrishayuk/gemma-3-4b-it-vindex"], "needs": [], "produces": ["cache-entry"]}
{"id": "cli.model", "cat": "register", "argv": ["model", "pull", "{{MODEL}}"], "needs": [], "produces": []}
{"id": "cli.list", "cat": "consume", "argv": ["list"], "needs": ["cache-entry"], "produces": []}
{"id": "cli.show", "cat": "consume", "argv": ["show", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.verify", "cat": "consume", "argv": ["verify", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.diag", "cat": "consume", "argv": ["diag", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.capabilities", "cat": "consume", "argv": ["capabilities"], "needs": [], "produces": []}
{"id": "cli.run", "cat": "consume", "argv": ["run", "{{VINDEX}}", "--prompt", "Paris is the capital of", "--max-tokens", "4"], "needs": ["vindex"], "produces": []}
{"id": "cli.chat", "cat": "consume", "argv": ["chat", "{{VINDEX}}", "--max-tokens", "1"], "needs": ["vindex"], "produces": []}
{"id": "cli.serve", "cat": "consume", "argv": ["serve", "{{VINDEX}}", "--port", "18080"], "needs": ["vindex"], "produces": []}
{"id": "cli.bench", "cat": "consume", "argv": ["bench", "{{VINDEX}}", "--tokens", "4"], "needs": ["vindex"], "produces": []}
{"id": "cli.dec-bench", "cat": "consume", "argv": ["dec-bench", "--help-me-fail"], "needs": [], "produces": []}
{"id": "cli.k3-ledger", "cat": "consume", "argv": ["k3-ledger", "report"], "needs": [], "produces": []}
{"id": "cli.accuracy", "cat": "consume", "argv": ["accuracy", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.shannon", "cat": "consume", "argv": ["shannon", "score", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.parity", "cat": "consume", "argv": ["parity", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.moe-locality", "cat": "consume", "argv": ["moe-locality", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.publish", "cat": "consume", "argv": ["publish", "{{VINDEX}}", "--repo", "example/does-not-exist"], "needs": ["vindex"], "produces": []}
{"id": "cli.hf", "cat": "consume", "argv": ["hf", "upload", "{{VINDEX}}", "--repo", "example/does-not-exist"], "needs": ["vindex"], "produces": []}
{"id": "cli.recipe", "cat": "consume", "argv": ["recipe", "validate", "{{TMP}}/recipe.yaml"], "needs": ["recipe"], "produces": []}
{"id": "cli.card", "cat": "consume", "argv": ["card", "render", "{{TMP}}/recipe.yaml"], "needs": ["recipe"], "produces": []}
{"id": "cli.dev", "cat": "consume", "argv": ["dev", "--help"], "needs": [], "produces": []}
{"id": "cli.repl", "cat": "consume", "argv": ["repl"], "needs": [], "produces": []}
{"id": "cli.lql", "cat": "consume", "argv": ["lql", "SHOW MODELS;"], "needs": [], "produces": []}
{"id": "cli.query", "cat": "graph", "argv": ["query", "{{TMP}}/graph.json", "--entity", "France"], "needs": ["graph-file"], "produces": []}
{"id": "cli.describe", "cat": "graph", "argv": ["describe", "{{TMP}}/graph.json", "France"], "needs": ["graph-file"], "produces": []}
{"id": "cli.stats", "cat": "graph", "argv": ["stats", "{{TMP}}/graph.json"], "needs": ["graph-file"], "produces": []}
{"id": "cli.validate", "cat": "graph", "argv": ["validate", "{{TMP}}/graph.json"], "needs": ["graph-file"], "produces": []}
{"id": "cli.merge", "cat": "graph", "argv": ["merge", "{{TMP}}/graph.json", "{{TMP}}/graph.json", "-o", "{{TMP}}/merged.json"], "needs": ["graph-file"], "produces": ["graph-file"]}
{"id": "cli.filter", "cat": "graph", "argv": ["filter", "{{TMP}}/graph.json", "--min-confidence", "0.5", "-o", "{{TMP}}/filtered.json"], "needs": ["graph-file"], "produces": ["graph-file"]}
{"id": "cli.rm", "cat": "destroy", "argv": ["rm", "{{VINDEX}}"], "needs": ["cache-entry"], "produces": []}
```

Several will still fail after correction — `cli.publish`/`cli.hf` target a
nonexistent repo, `cli.recipe`/`cli.card` need a recipe file no cell produces,
the graph cells need a graph file no cell produces, `cli.serve`/`cli.chat` do
not self-terminate. **That is intended and is a different thing from a wrong
argument.** Per the spec nothing is skipped for an unsatisfied dependency or a
guessed-inapplicable command; the capture records what larql does when asked,
which is the measurement. Removing such a cell would be the harness deciding
what is worth asking. Correcting a *wrong argument* is the opposite — that is
the harness asking the question it meant to ask.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest -q`
Expected: PASS — 81 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/cli-commands.jsonl scripts/lql_matrix/corpus_lint_test.py
git commit -m "harness: a real invocation for every one of the 38 subcommands

Each cell declares needs/produces over a closed artifact vocabulary so
the sequencer can order them. Cells that cannot succeed in CI still run
— an unsatisfied dependency or a missing remote is recorded, not
skipped, because what larql does when asked is the measurement."
```

- [ ] **Step 6: Demonstrate the real invocations on a runner**

Wire `cli-commands.jsonl` in beside `cli-help.jsonl` in the same workflow
branch, push, and read every capture:

```bash
git add .github/workflows/lql-strategy-matrix.yml
git commit -m "ci: run the 38 real CLI invocations"
git push
gh run download <run-id> --repo metavacua/larql-to-sparql --name results-smol135.native.all --dir /tmp/d
python3 -c "
import json,glob
rows=[json.loads(l) for f in glob.glob('/tmp/d/results-*.cli.commands.jsonl') for l in open(f) if l.strip()]
cells=[r for r in rows if r.get('type')!='meta']
for r in sorted(cells,key=lambda x:x['id']):
    print(f\"{r['id']:<22} exit={r['exit_code']:<5} out={r['stdout_bytes']:>7}B err={r['stderr_bytes']:>7}B\")"
```

Then read the stderr of every non-zero cell. Expect many failures — wrong flags
in this plan's guesses, missing recipe and graph files, unreachable repos,
`serve`/`chat` hitting the timeout. Classify each into exactly one of:

- **the cell's arguments are wrong** — fix the cell, this is a harness defect
- **larql rejected a valid request** — leave the cell, this is the finding
- **the dependency genuinely is not there** — leave the cell; Task 8's sequencer
  will record it as `unsatisfied` and it still runs

Do not resolve the third category by deleting cells. **Do not start Task 8 until
every non-zero cell has been placed in one of these three buckets.**

---

### Task 8: Dependency ordering and deterministic permutation

**Files:**
- Create: `scripts/lql_matrix/sequence.py`
- Create: `scripts/lql_matrix/sequence_test.py`

**Interfaces:**
- Consumes: cells with `id`, `needs`, `produces` (Task 7).
- Produces: `sequence.sequence(cells: list[dict], seed: int) -> list[dict]` returning the cells reordered, each gaining `"order_index": int` and `"unsatisfied": list[str]`. Task 9 calls it from `run_matrix.py`.

- [ ] **Step 1: Write the failing test**

Create `scripts/lql_matrix/sequence_test.py`:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import sequence as S

A = {"id": "produce", "needs": [], "produces": ["vindex"]}
B = {"id": "consume1", "needs": ["vindex"], "produces": []}
C = {"id": "consume2", "needs": ["vindex"], "produces": []}
D = {"id": "orphan", "needs": ["graph-file"], "produces": []}


def _ids(cells):
    return [c["id"] for c in cells]


def test_producer_precedes_its_consumers():
    out = _ids(S.sequence([B, C, A], seed=0))
    assert out.index("produce") < out.index("consume1")
    assert out.index("produce") < out.index("consume2")


def test_same_seed_gives_the_same_order():
    assert _ids(S.sequence([B, C, A], 7)) == _ids(S.sequence([B, C, A], 7))


def test_different_seeds_can_permute_independent_cells():
    orders = {tuple(_ids(S.sequence([B, C, A], s))) for s in range(12)}
    assert len(orders) > 1, "independent cells must actually permute"


def test_every_cell_appears_exactly_once():
    out = _ids(S.sequence([B, C, A, D], 3))
    assert sorted(out) == sorted(["produce", "consume1", "consume2", "orphan"])


def test_unsatisfiable_cell_still_runs_and_is_marked():
    out = S.sequence([A, D], 0)
    orphan = [c for c in out if c["id"] == "orphan"][0]
    assert orphan["unsatisfied"] == ["graph-file"]
    satisfied = [c for c in out if c["id"] == "produce"][0]
    assert satisfied["unsatisfied"] == []


def test_order_index_is_dense_and_matches_position():
    out = S.sequence([B, C, A, D], 5)
    assert [c["order_index"] for c in out] == list(range(len(out)))


def test_input_cells_are_not_mutated():
    original = dict(A)
    S.sequence([A, B], 0)
    assert A == original


# ── The eight graph shapes. Empty, single, chain and diamond are the
# acceptance cases; all-independent, unsatisfiable, self-cycle and mutual
# cycle are the rejection-shaped ones. None of the four may hang, drop a
# cell, or raise — an unorderable cell still runs, with what it lacked
# recorded, because "this breaks when its input is absent" is an
# observation the harness exists to capture.

def test_empty_input_returns_empty():
    assert S.sequence([], 0) == []


def test_single_cell_is_returned_with_index_zero():
    out = S.sequence([A], 0)
    assert _ids(out) == ["produce"] and out[0]["order_index"] == 0


def test_diamond_orders_root_first_and_join_last():
    root = {"id": "root", "needs": [], "produces": ["a"]}
    left = {"id": "left", "needs": ["a"], "produces": ["b"]}
    right = {"id": "right", "needs": ["a"], "produces": ["c"]}
    join = {"id": "join", "needs": ["b", "c"], "produces": []}
    out = _ids(S.sequence([join, left, right, root], 0))
    assert out[0] == "root" and out[-1] == "join"


def test_all_independent_cells_all_run_and_none_is_unsatisfied():
    cells = [{"id": f"c{i}", "needs": [], "produces": []} for i in range(5)]
    out = S.sequence(cells, 0)
    assert len(out) == 5
    assert all(c["unsatisfied"] == [] for c in out)


def test_self_cycle_still_runs_and_records_what_it_lacked():
    # A cell that needs what it itself produces can never be satisfied.
    selfdep = {"id": "loop", "needs": ["x"], "produces": ["x"]}
    out = S.sequence([selfdep], 0)
    assert _ids(out) == ["loop"]
    assert out[0]["unsatisfied"] == ["x"]


def test_mutual_cycle_still_runs_both_and_records_both():
    p = {"id": "p", "needs": ["y"], "produces": ["x"]}
    q = {"id": "q", "needs": ["x"], "produces": ["y"]}
    out = S.sequence([p, q], 0)
    assert sorted(_ids(out)) == ["p", "q"]
    assert out[0]["unsatisfied"] and out[1]["unsatisfied"]


def test_a_cycle_terminates():
    # Regression guard: the loop must not spin when nothing is ever ready.
    import threading
    done = threading.Event()
    threading.Thread(
        target=lambda: (S.sequence([{"id": "a", "needs": ["z"], "produces": []}], 0),
                        done.set()),
        daemon=True).start()
    assert done.wait(5), "sequence() did not terminate on an unsatisfiable graph"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest sequence_test.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sequence'`

- [ ] **Step 3: Write the implementation**

Create `scripts/lql_matrix/sequence.py`:

```python
#!/usr/bin/env python3
"""Order cells by declared dependencies, permuting the independent ones.

Cells are not an unordered set. A consumer run before its producer measures
nothing about the consumer — `verify` on a vindex that does not exist yet
reports a missing directory, not anything about verify.

Where two cells are independent under the dependency relation, their relative
order is a free variable, and free variables get varied rather than frozen:
`seed` selects a permutation. It is deterministic, so an ordering-dependent
failure is re-runnable.

A cell whose needs are never produced is NOT dropped. It runs last with
`unsatisfied` recording what was missing, because "this command breaks when
its input is absent" is an observation worth capturing. Ordering is never
used to decide correctness.

Pure: no I/O, no subprocess, no logging.
"""
import random


def sequence(cells, seed):
    """Return cells reordered, each a copy carrying `order_index` and
    `unsatisfied`. Input dicts are not mutated."""
    rng = random.Random(seed)
    remaining = [dict(c) for c in cells]
    satisfied = set()
    ordered = []

    while remaining:
        ready = [c for c in remaining if set(c.get("needs", [])) <= satisfied]
        if not ready:
            # Nothing can be satisfied: emit the rest in a seeded order with
            # their missing artifacts recorded. Never skipped.
            rest = list(remaining)
            rng.shuffle(rest)
            for c in rest:
                c["unsatisfied"] = sorted(set(c.get("needs", [])) - satisfied)
                ordered.append(c)
            remaining = []
            break
        pick = ready[rng.randrange(len(ready))]
        pick["unsatisfied"] = []
        ordered.append(pick)
        satisfied.update(pick.get("produces", []))
        remaining.remove(pick)

    for i, c in enumerate(ordered):
        c["order_index"] = i
        c.setdefault("unsatisfied", [])
    return ordered
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest sequence_test.py -q`
Expected: PASS — 14 tests, covering all eight graph shapes plus determinism,
permutation, non-mutation and a termination guard.

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/sequence.py scripts/lql_matrix/sequence_test.py
git commit -m "harness: dependency ordering with deterministic permutation

A consumer run before its producer measures nothing about the consumer.
Independent cells are permuted by seed rather than frozen in file order,
deterministically so an ordering-dependent failure is re-runnable. A
cell whose needs are never produced still runs, with the missing
artifacts recorded — nothing is skipped."
```

---

### Task 9: Sequence the corpus in `run_matrix.py` and record the order

**Files:**
- Modify: `scripts/lql_matrix/run_matrix.py`
- Modify: `scripts/lql_matrix/run_matrix_test.py`

**Interfaces:**
- Consumes: `sequence.sequence(cells, seed)` (Task 8).
- Produces: `run_matrix.py` accepts `--order-seed N` (default `0`); rows gain `"order_index"`, `"unsatisfied"`, and the meta row gains `"order_seed"`.

- [ ] **Step 1: Write the failing test**

Append to `scripts/lql_matrix/run_matrix_test.py`:

```python
def _corpus_with_deps(tmp_path):
    p = tmp_path / "dep.jsonl"
    rows = [
        {"id": "consumer", "cat": "c", "argv": ["verify", "{{VINDEX}}"],
         "needs": ["vindex"], "produces": []},
        {"id": "producer", "cat": "p", "argv": ["extract", "m", "-o", "v"],
         "needs": [], "produces": ["vindex"]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(p)


def test_producer_runs_before_consumer_and_order_is_recorded(tmp_path):
    out = tmp_path / "r.jsonl"
    env = dict(os.environ, LARQL_BIN=_fake_bin(tmp_path), CELL_TIMEOUT="30")
    subprocess.run([sys.executable, os.path.join(HERE, "run_matrix.py"),
                    "leg1", "/v", _corpus_with_deps(tmp_path), str(out),
                    "--driver", "cli", "--order-seed", "1"],
                   env=env, check=True, capture_output=True)
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    meta = [r for r in rows if r.get("type") == "meta"][0]
    cells = [r for r in rows if r.get("type") != "meta"]
    assert meta["order_seed"] == 1
    ids = [c["id"] for c in cells]
    assert ids.index("producer") < ids.index("consumer")
    assert [c["order_index"] for c in cells] == [0, 1]
    assert all(c["unsatisfied"] == [] for c in cells)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest run_matrix_test.py -q -k order`
Expected: FAIL — `unrecognized arguments: --order-seed`

- [ ] **Step 3: Wire the sequencer in**

In `scripts/lql_matrix/run_matrix.py`:

1. Add `import sequence` next to `import drivers`.
2. Add to the argument parser, after the `--driver` line:

```python
    ap.add_argument("--order-seed", type=int, default=0,
                    help="permutation seed for cells independent under needs/produces")
```

3. After `level, vindex, corpus, out, driver = ...`, add:

```python
    order_seed = ns.order_seed
```

4. Add `"order_seed": order_seed,` to the `meta` dict, immediately after `"driver": driver,`.

5. Replace the corpus read

```python
    with open(corpus, encoding="utf-8") as cf:
        cells = [json.loads(l) for l in cf if l.strip()]
```

with

```python
    with open(corpus, encoding="utf-8") as cf:
        cells = [json.loads(l) for l in cf if l.strip()]
    cells = sequence.sequence(cells, order_seed)
```

6. Add to the `row` dict, immediately after `"cat": cat,`:

```python
            "order_index": c.get("order_index"),
            "unsatisfied": c.get("unsatisfied", []),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest -q`
Expected: PASS — 89 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/lql_matrix/run_matrix.py scripts/lql_matrix/run_matrix_test.py
git commit -m "harness: sequence the corpus by dependencies, record the order

--order-seed selects the permutation of independent cells; the seed is
in the meta row and each cell carries order_index and unsatisfied, so a
capture can be tied back to the ordering that produced it."
```

- [ ] **Step 6: Demonstrate that ordering changes outcomes, on a runner**

Permutation is only worth having if order actually matters. Prove it by running
the CLI corpus twice with different seeds and diffing the outcomes.

Add `--order-seed "${ORDER_SEED}"` to the cli invocations and push twice, once
with `ORDER_SEED: 1` and once with `ORDER_SEED: 2`, then:

```bash
python3 -c "
import json,glob,sys
def outcomes(d):
    rows=[json.loads(l) for f in glob.glob(d+'/results-*.cli.commands.jsonl') for l in open(f) if l.strip()]
    return {r['id']:r['exit_code'] for r in rows if r.get('type')!='meta'}
a,b=outcomes('/tmp/seed1'),outcomes('/tmp/seed2')
diff={k:(a[k],b[k]) for k in a if a.get(k)!=b.get(k)}
print('cells whose outcome changed with ordering:',diff or 'none')"
```

Either answer is informative and neither blocks progress: a non-empty diff means
ordering is load-bearing and permutation is earning its cost; an empty diff for
these seeds means these cells are order-independent, which is worth knowing and
does not mean permutation should be removed. Record which you saw.

---

### Task 10: Consolidate the workflow onto the declared driver list

Tasks 3, 4, 6, 7 and 9 each wired their piece in and demonstrated it, so the
workflow currently drives a **hardcoded** driver list. This task replaces that
with `matrix.leg.drivers` from Task 5, so a leg runs exactly the drivers it
declares and the long-session and cli legs take effect. It is the only task
whose whole content is cleanup, and by construction nothing here is unproven —
every driver it references has already produced captures on a runner.

**Files:**
- Modify: `.github/workflows/lql-strategy-matrix.yml`

**Interfaces:**
- Consumes: `matrix.leg.drivers`, `matrix.leg.long_session` (Task 5); `run_matrix.sh … --driver … --order-seed …` (Tasks 3, 9); `cli-help.jsonl`, `cli-commands.jsonl` (Tasks 6, 7).
- Produces: captures for every declared driver, uploaded under `results-<leg>`.

- [ ] **Step 1: Replace the hardcoded driver list with the declared one**

In `.github/workflows/lql-strategy-matrix.yml`, replace the step named
`Run LQL command corpus (${{ matrix.leg.name }})` in the `matrix` job with:

```yaml
      - name: Run corpora per driver (${{ matrix.leg.name }})
        run: |
          set -uo pipefail
          VINDEX="out/${NAME}.vindex"
          for DRV in $DRIVERS; do
            case "$DRV" in
              cli)
                for CORPUS in cli-help cli-commands; do
                  echo "=== ${NAME} / ${DRV} / ${CORPUS} ==="
                  LARQL_BIN=./bin/larql MODEL_ID="${CORPUS_MODEL}" \
                  TMPROOT="$(mktemp -d)" CELL_TIMEOUT=900 \
                  scripts/lql_matrix/run_matrix.sh \
                    "${NAME}" "$VINDEX" "scripts/lql_matrix/${CORPUS}.jsonl" \
                    "out/results-${NAME}.${DRV}.${CORPUS}.jsonl" \
                    --driver "$DRV" --order-seed "${ORDER_SEED}"
                done ;;
              *)
                echo "=== ${NAME} / ${DRV} ==="
                LARQL_BIN=./bin/larql MODEL_ID="${CORPUS_MODEL}" \
                TMPROOT="$(mktemp -d)" CELL_TIMEOUT=900 \
                scripts/lql_matrix/run_matrix.sh \
                  "${NAME}" "$VINDEX" scripts/lql_matrix/commands.jsonl \
                  "out/results-${NAME}.${DRV}.jsonl" \
                  --driver "$DRV" --order-seed "${ORDER_SEED}" ;;
            esac
          done
```

- [ ] **Step 2: Add the driver and seed env vars**

In the `matrix` job's `env:` block, after the `LQL_WITH:` line, add:

```yaml
      DRIVERS: ${{ join(matrix.leg.drivers, ' ') }}
      ORDER_SEED: ${{ github.run_number }}
```

`ORDER_SEED` is the run number so successive runs explore different orderings
while any single run stays reproducible from the seed recorded in its meta row.

- [ ] **Step 3: Widen the upload glob**

In the `Upload leg results` step, replace the line
`            out/results-${{ matrix.leg.name }}.jsonl`
with:

```yaml
            out/results-${{ matrix.leg.name }}.*.jsonl
```

- [ ] **Step 4: Verify the workflow parses and the driver join renders**

Run:
```bash
cd /home/metavacua/larql-vindex3-03-08-2026
python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/lql-strategy-matrix.yml')); print('YAML OK, jobs:', list(d['jobs'].keys()))"
grep -n 'DRIVERS:\|ORDER_SEED:\|--driver\|results-.*\*\.jsonl' .github/workflows/lql-strategy-matrix.yml
```
Expected: `YAML OK`, and the grep shows `DRIVERS`, `ORDER_SEED`, two `--driver` uses and the widened glob.

- [ ] **Step 5: Confirm no suppression was introduced**

Run:
```bash
grep -n '2>/dev/null\|>/dev/null\|| true' .github/workflows/lql-strategy-matrix.yml | grep -v '^\s*[0-9]*:\s*#'
```
Expected: no output. Any hit is a Global Constraint violation and must be removed before committing.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/lql-strategy-matrix.yml
git commit -m "ci: run every driver and both CLI corpora per leg

Each leg runs the drivers it declares; cli legs additionally run the
help and invocation corpora. ORDER_SEED is the run number, so runs
explore different permutations while each run stays reproducible from
the seed in its meta row."
```

---

## Self-Review

**Spec coverage.** Every section of `2026-08-07-cli-and-repl-coverage-design.md` maps to a task: principles → Global Constraints (and asserted by `test_row_carries_no_derived_opinion`, Task 3); four drivers → Tasks 2–4; session granularity → Task 5 (`long_session` legs); capture → Tasks 3–4; the three corpora → Tasks 1, 6, 7; WIP surfaces → Task 7 (`build`, `show`, `dev` cells reach them); sequencing and permutation → Tasks 8–9; termination and hangs → Task 2 (`exit` for pty) and Task 4 (pty timeout); scale → no task needed, it is an accepted consequence.

**Deliberately deferred:** the spec's `.merged` capture is produced only for `repl-pty` (Task 4), which is the only driver that has one — `repl-pipe` keeps separate `.out`/`.err`, as the spec's capture table describes.

**Placeholder scan.** No TBDs, no "add error handling", no "similar to Task N". Every code step carries the code to type; every test step names the command and the expected result.

**Type consistency.** `drivers.build(driver, cell, larql) -> (argv, stdin_bytes)` is defined in Task 2 and called identically in Tasks 3 and 4. `sequence.sequence(cells, seed) -> list[dict]` is defined in Task 8 and called in Task 9. `corpus_lint.lint_lql_corpus(path)` (Task 1) and `lint_cli_corpus(path, require_deps)` (Task 6) share `_rows`. `SUBCOMMANDS` is defined once in Task 6 and consumed by Tasks 6 and 7. Row keys `driver`, `order_index`, `unsatisfied`, `merged` are introduced in Tasks 3, 9, 9, 4 and read only after.

**Demonstration coverage.** Every task that changes runtime behaviour ends on a real runner against the real binary, and names what to read and what each outcome means before the next task starts: Task 0 (does the REPL accept piped stdin at all), Task 3 (the `lql` driver is unchanged), Task 4 (both REPL drivers produce captures), Task 6 (38 real `--help` invocations), Task 7 (38 real invocations, each non-zero classified), Task 9 (does ordering change outcomes). Tasks 1, 2, 5 and 8 are corpus data or pure functions with no runtime surface.

**Ordering by failure probability.** The three riskiest items — 38 guessed CLI argument shapes, `larql repl` on non-tty stdin, and a `pty.fork` read loop — are all attempted in Task 0, which writes no harness code. Task 7 then writes its corpus from Task 0's captures rather than guessing a second time, and Task 4 lifts a read loop already proven on a runner. The near-certainties (corpus regex, pure functions, leg axis, consolidation) come last, where a failure costs a fix rather than a redesign.

**Exhaustive coverage of the finite sets.** `drivers.build` is enumerated over all 4 drivers × 2 cell shapes — four acceptances and four rejections, Task 2. `sequence.sequence` is enumerated over all eight graph shapes including self-cycle and mutual cycle, with a termination guard, Task 8. `corpus_lint` rejects each missing required key, duplicate ids, non-subcommands and bare `BEGIN PATCH`, Tasks 1 and 6. All 38 subcommands appear in Tasks 0, 6 and 7. Cell *orderings* are factorial and therefore sampled by seed rather than exhausted — Task 9 declares that sampling rather than presenting it as coverage.

**Assumption this plan makes that Task 0 may invalidate.** Tasks 2–4 assume `larql repl` reads piped stdin. If Task 0 shows it does not, `drivers.build` and the pty runner are unchanged — a capture of nothing is still the finding — but Task 4's demonstration expectations and Task 10's outcome change. That is why Task 0 runs first and why its step 4 is a gate rather than a note.
