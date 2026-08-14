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

> **Result (run 31246014726, see `docs/diagnoses/task0-cli-repl-probe.md`).**
> The READ half of the loop is verified; the WRITE half is refuted. Writing all
> statements up front and never again loses them: under `script(1)` one of
> three vanished silently, and under `probe_pty.py` two did and the driver hit
> its deadline. Task 4 must **not** lift `probe_pty.py`'s single up-front
> `os.write` — it has to write each statement in response to a prompt, and
> re-measure both pty mechanisms afterwards. The design's line that a
> `pty`-module driver is "an acceptable alternative" to `script(1)` is
> measured false as written: they lost different amounts of the same input.
>
> Task 7's argv corrections for the 10 wrong rows are tabulated in that
> diagnosis. Piped stdin executes statements correctly and needs no change.

- [x] **Step 1: Write the minimal pty probe**

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

- [x] **Step 2: Verify the probe locally against a known-tty program**

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

- [x] **Step 3: Add the probe job**

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
    needs: [gate, build]
    if: needs.gate.outputs.probe == 'true'
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      # We are not testing HuggingFace. `pull`, `model pull`, `publish` and
      # `hf upload` would otherwise put HF's availability, rate limiter and
      # auth into the measurement, none of which is larql's surface. Pointing
      # the hub at a closed port makes them fail at the network boundary in
      # milliseconds, so what the capture shows is how the CLI parsed the argv
      # and what it did next — which is the whole question this probe asks.
      #
      # This is the in-tree mechanism, not a trick: larql-vindex honours
      # HF_ENDPOINT in format/huggingface/download/mod.rs:283, and its own
      # test at :576 points it at a non-existent server for exactly this.
      # hf-hub 0.5 reads it too (api/mod.rs:14).
      #
      # KNOWN ESCAPE: k3_ledger/fetch.rs:36 hardcodes https://huggingface.co
      # and ignores HF_ENDPOINT. `k3-ledger report` should die at clap first
      # (`report` is not among its subcommands), but if a capture shows hub
      # traffic anyway, that is a larql finding — a hardcoded URL inconsistent
      # with the rest of the tree — and NOT a harness defect to route around.
      HF_ENDPOINT: http://127.0.0.1:1
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
          # GitHub's default shell is `bash --noprofile --norc -eo pipefail`.
          # The `set -uo pipefail` in the later steps does NOT clear -e. Left
          # alone, the first nonzero exit kills the step and every capture
          # after it is lost — the harness would destroy precisely the data it
          # exists to collect, and a probe whose whole purpose is to provoke
          # failures fails on the first one. `|| true` is not the fix (it
          # converts a failure into a success value); `continue-on-error` is
          # not either (-e still aborts the script inside the step, it just
          # paints the job green). Turn -e off and record the code.
          set +e
          printf 'SHOW MODELS;\nSTATS;\nexit\n' \
            | timeout 60 ./bin/larql repl > probe/repl-pipe.out 2> probe/repl-pipe.err
          # BOTH halves, separately. Under pipefail a pipeline's $? is whichever
          # member failed. If larql exits without draining stdin — plausible,
          # and the exact unknown this probe exists to settle — printf takes
          # SIGPIPE and $? is 141 for PRINTF, which reads as larql having died
          # on a signal. One lying exit code already cost this project three
          # reversed claims about COMPILE; do not let the probe mint another.
          echo "printf_exit=${PIPESTATUS[0]} larql_exit=${PIPESTATUS[1]}" \
            >> probe/repl-pipe.err
      - name: REPL — pty via script(1)
        run: |
          set +e
          printf 'SHOW MODELS;\nSTATS;\nexit\n' \
            | timeout 60 script -q -e -c './bin/larql repl' probe/repl-script.merged \
              > probe/repl-script.out 2> probe/repl-script.err
          # script -e propagates the child's exit, so PIPESTATUS[1] is larql's.
          echo "printf_exit=${PIPESTATUS[0]} script_exit=${PIPESTATUS[1]}" \
            >> probe/repl-script.err
      - name: REPL — pty via probe_pty.py (the loop Task 4 will use)
        run: |
          set +e
          printf 'SHOW MODELS;\nSTATS;\nexit\n' \
            | python3 scripts/lql_matrix/probe_pty.py probe/repl-pty.merged \
              ./bin/larql repl 2> probe/repl-pty.err
          # probe_pty.py already prints the child's `exit=` to its own stderr;
          # this records whether the DRIVER itself survived, which is a
          # different question and the one Task 4 depends on.
          echo "printf_exit=${PIPESTATUS[0]} driver_exit=${PIPESTATUS[1]}" \
            >> probe/repl-pty.err
      - name: One-shot lql, for comparison
        run: |
          set +e
          timeout 60 ./bin/larql lql 'SHOW MODELS; STATS;' \
            > probe/lql.out 2> probe/lql.err
          echo "exit=$?" >> probe/lql.err
      - name: All 38 subcommands — --help
        run: |
          set +e -u
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
          set +e -u
          TMP=$(mktemp -d)
          # Neither a vindex nor a reachable hub exists in this job. Both
          # absences are deliberate and they do the same work: they separate
          # "the argument shape is wrong" from "the input was missing" and
          # from "the network answered", and all three answers are needed
          # before Task 7 writes the corpus. See the HF_ENDPOINT note on the
          # job for why the hub is closed rather than warm.
          # A bash array, NOT a heredoc. A heredoc body has to sit at column 0
          # and its terminator likewise, and column 0 ends a YAML block scalar
          # — the workflow would not parse. Every line here stays inside the
          # scalar's indentation.
          CMDS=(
            "extract|extract $TMP/nomodel -o $TMP/v.vindex --level all"
            "extract-index|extract-index $TMP/nomodel -o $TMP/v2.vindex --level browse"
            "convert|convert quantize q4k --input $TMP/v.vindex --output $TMP/q.vindex"
            "build|build $TMP/Vindexfile -o $TMP/b.vindex"
            "slice|slice $TMP/v.vindex --output $TMP/s.vindex --kind browse"
            "compile|compile --base $TMP/nomodel --vindex $TMP/e.vlp --output $TMP/c"
            "link|link $TMP/v.vindex"
            "pull|pull chrishayuk/gemma-3-4b-it-vindex"
            "model|model pull HuggingFaceTB/SmolLM2-135M"
            "list|list"
            "show|show $TMP/v.vindex"
            "verify|verify $TMP/v.vindex"
            "diag|diag $TMP/v.vindex"
            "capabilities|capabilities"
            "run|run $TMP/v.vindex --prompt hi --max-tokens 2"
            "chat|chat $TMP/v.vindex --max-tokens 1"
            "serve|serve $TMP/v.vindex --port 18080"
            "bench|bench $TMP/v.vindex --tokens 2"
            "dec-bench|dec-bench"
            "k3-ledger|k3-ledger report"
            "accuracy|accuracy $TMP/v.vindex"
            "shannon|shannon score $TMP/v.vindex"
            "parity|parity $TMP/v.vindex"
            "moe-locality|moe-locality $TMP/v.vindex"
            "publish|publish $TMP/v.vindex --repo example/does-not-exist"
            "hf|hf upload $TMP/v.vindex --repo example/does-not-exist"
            "recipe|recipe validate $TMP/recipe.yaml"
            "card|card render $TMP/recipe.yaml"
            "dev|dev"
            "repl|repl"
            "lql|lql \"SHOW MODELS;\""
            "query|query $TMP/graph.json --entity France"
            "describe|describe $TMP/graph.json France"
            "stats|stats $TMP/graph.json"
            "validate|validate $TMP/graph.json"
            "merge|merge $TMP/graph.json $TMP/graph.json -o $TMP/m.json"
            "filter|filter $TMP/graph.json --min-confidence 0.5 -o $TMP/f.json"
            "rm|rm $TMP/v.vindex"
          )
          for line in "${CMDS[@]}"; do
            id=${line%%|*}; rest=${line#*|}
            # eval so the row's own quoting survives into argv: the lql row
            # must arrive as TWO arguments (`lql` and `SHOW MODELS;`), and an
            # unquoted split would make it three and mask the real behaviour.
            eval "set -- $rest"
            timeout 60 ./bin/larql "$@" > "probe/cmd.$id.out" 2> "probe/cmd.$id.err"
            echo "$id exit=$? argc=$# argv=[$rest]" >> probe/cmd.index
          done
          cat probe/cmd.index
      - name: Show every capture
        if: always()
        run: |
          # set +e rather than `|| true` on the head: same protection against
          # -e killing the listing part-way, without writing a success value
          # over a failure anywhere. This step only echoes; the artifact below
          # is the actual record.
          set +e
          for f in probe/*; do
            echo "───── $f ($(wc -c < "$f") bytes)"
            head -40 "$f"
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

The `pull`, `model pull`, `publish` and `hf upload` rows are expected to fail
at the network boundary against the closed `HF_ENDPOINT` port, in milliseconds.
That is the intended result, not a degraded one: what Task 7 needs from them is
whether clap accepted the argv and how far `run()` got, and a real fetch would
answer neither while putting HF's rate limiter into the measurement. Whether
larql's *downloading* works is a different question, already covered by the
`prefetch`-warmed legs, and is not this probe's subject.

- [x] **Step 4: Verify the workflow parses**

Run:
```bash
cd /home/metavacua/larql-vindex3-03-08-2026
python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/lql-strategy-matrix.yml')); print('jobs:', list(d['jobs'].keys()))"
```
Expected: the job list includes `probe`.

- [x] **Step 5: Commit and push**

```bash
git add .github/workflows/lql-strategy-matrix.yml scripts/lql_matrix/probe_pty.py
git commit -m "ci: probe the three riskiest assumptions before building on them

38 CLI invocations with guessed arguments, larql repl on non-tty stdin,
and a pty.fork read loop. No harness code — it runs things and uploads
what they printed. A guessed argument that is wrong should cost a
capture to discover, not a corpus and a sequencer built on top of it."
git push
```

- [x] **Step 6: Read every capture before writing any code**

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

### Task 1: Fix the corpus cells that cannot parse — **DONE, approach changed**

`parse_begin` in `crates/larql-lql/src/parser/patch.rs` calls `expect_string()`, so the path is mandatory. Six cells send `BEGIN PATCH;` and have never opened a named patch session — every one produces `Parse error: expected string literal, got Semicolon`, and because a cell is a batch, that error is the first in the batch and masks everything after it.

> **Approach changed during execution, on instruction: do not write custom
> parsers or Python scripts — LQL and LARQL already have a lexer and parser.**
>
> The planned `corpus_lint.py` was a regex that approximated the LQL grammar.
> It was written, and it was wrong twice over: it produced a false positive on
> `BEGIN PATCH;` appearing inside a string literal, and — decisively — it knew
> only about `BEGIN PATCH`, so it would have declared the corpus clean while a
> seventh cell was still malformed. It was deleted unused.
>
> Replaced by `crates/larql-lql/tests/matrix_corpus_wellformed.rs`, which reads
> the shipped corpus and runs every cell through LQL's own `split_statements`
> feeding `parser::parse` — the exact pair `run_batch` uses, so the test checks
> the same decomposition that actually runs. `split_statements` was made `pub`
> for this; a batch's statement boundaries are not derivable from outside
> without reimplementing it, since `;` inside a string literal does not end a
> statement.
>
> **This rule binds Tasks 6 and 7.** Their planned `lint_cli_corpus`,
> `SUBCOMMANDS` and `_rows` do not exist and must not be written as a Python
> re-implementation of the CLI's argument grammar. clap already owns that
> grammar; validate a CLI corpus by invoking the real binary, exactly as Task 0
> did to correct ten guessed argv rows.

**Files:**
- Create: `crates/larql-lql/tests/matrix_corpus_wellformed.rs`
- Modify: `crates/larql-lql/src/repl.rs`, `crates/larql-lql/src/lib.rs` — export `split_statements`
- Modify: `scripts/lql_matrix/commands.jsonl`

**Interfaces:**
- Consumes: `larql_lql::{parse, split_statements}`.
- Produces: `larql_lql::split_statements(&str) -> Vec<String>`, now public.

- [x] **Step 1: Write the test against the real parser**

`crates/larql-lql/tests/matrix_corpus_wellformed.rs`, two tests:
`every_shipped_lql_cell_parses` walks `commands.jsonl` and reports every
statement the parser rejects, with cell id, line, the statement text and the
real `ParseError`; `bare_begin_patch_is_rejected_by_the_grammar` pins *why* the
corpus rule exists, and fails if the grammar ever stops requiring the path.

Cells whose id begins `neg.` or `error.` are exempt — the corpus deliberately
carries negative cells, and a test demanding everything parse would delete the
coverage proving larql rejects bad input.

- [x] **Step 2: Run it against the unfixed corpus**

Run: `cargo test -p larql-lql --test matrix_corpus_wellformed`
Result: FAILED, **7** of 64 cells (149 statements), all
`Parse error: expected string literal, got Semicolon` — one root-cause class,
a mandatory string operand omitted.

- [x] **Step 3: Root-cause the seventh cell**

The plan predicted six. The real parser found a seventh the regex could not
have: cell `merge`, line 47, `USE "{{VINDEX}}"; MERGE;`. `parse_merge`
(`parser/mutation.rs:154`) calls `expect_string()` for `source`, mandatory just
as `parse_begin` does for its path; the grammar's own tests all feed
`MERGE "source.vindex";`. It is the only `MERGE` cell in the corpus, so **the
MERGE surface has never once been exercised by this harness.**

- [x] **Step 4: Fix the seven cells**

Six get their own `.vlp` under the substituted tmp dir, named for the cell so
they cannot collide: `BEGIN PATCH "{{TMP}}/<id>.vlp";`. `merge` gets the
grammar's minimal accepted form, `MERGE "{{VINDEX}}";`. Nothing else changed —
7 lines of 64, verified by `git diff --numstat`.

- [x] **Step 5: Verify**

`cargo test -p larql-lql --test matrix_corpus_wellformed` passes; the full
`cargo test -p larql-lql` suite passes across its targets so
exporting `split_statements` broke nothing; `cargo fmt --check` clean and
`cargo clippy --tests` at 0 warnings.

Whether a now-parseable cell *succeeds* at runtime is not asserted anywhere.
That is the run's business and belongs in the captures.


### Task 2: Statements become data; drivers map a cell to an invocation

> **Approach changed during execution, same instruction as Task 1: no custom
> parsers, no Python re-implementations.**
>
> As planned, `drivers.py` carried its own `split_statements` whose docstring
> said it "mirrors larql's own splitter in crates/larql-lql/src/repl.rs". Two
> implementations of one grammar, kept in agreement by hand. Deleted before it
> was written.
>
> The splitter is not moved — it is **removed**. A corpus cell's `lql` becomes
> a LIST of statements, so the boundaries are authored data rather than
> something any code has to re-derive. The one-shot `lql` driver joins with a
> space (byte-identical to the current single-string cell), and the REPL
> drivers join with newlines. No splitter exists in the harness at all.
>
> Measured facts behind the choice: 0 of 64 cells contain `;` inside a string
> literal; cells hold 1–9 statements, mean 2.3; and `larql_lql::parse` rejects
> a multi-statement line outright (`unexpected trailing token`), so a REPL
> driver genuinely must send one statement per line — sending the whole cell
> would turn every multi-statement cell into a single parse error.

**Files:**
- Create: `scripts/lql_matrix/drivers.py`, `scripts/lql_matrix/drivers_test.py`
- Modify: `scripts/lql_matrix/commands.jsonl` — `lql` becomes a list
- Modify: `crates/larql-lql/tests/matrix_corpus_wellformed.rs` — list shape, plus the round-trip guard
- Modify: `scripts/lql_matrix/run_matrix.py` — join the list at the use site

**Interfaces:**
- Consumes: `larql_lql::split_statements` (migration and guard only, never at run time).
- Produces:
  - `drivers.DRIVERS: tuple[str, ...]` = `("lql", "repl-pipe", "repl-pty", "cli")`
  - `drivers.build(driver: str, cell: dict, larql: str) -> tuple[list[str], bytes | None]`
    returning `(argv, stdin_bytes)`; `stdin_bytes` is `None` when the driver
    writes no stdin. Task 3 calls this from `run_matrix.py`.
  - No `split_statements`. Deliberately.

- [x] **Step 1: Migrate the corpus with the real splitter, once**

A throwaway Rust test emits the list-shaped corpus using
`larql_lql::split_statements`, trimming each entry (the splitter returns
`" STATS;"` with the leading space intact). Run it, take the output, delete the
test. Splitting 64 cells by hand or in Python would reintroduce exactly the
second implementation this task exists to remove.

- [x] **Step 2: Update the corpus test, and add the round-trip guard**

`matrix_corpus_wellformed.rs` reads `lql` as an array and parses each entry —
simpler than before, since it no longer splits. Then the guard that keeps
Task 1's `pub` export honest:

```rust
// The one-shot `lql` driver hands the SPACE-JOINED cell to run_batch, which
// splits it again with this same function. If that split ever disagrees with
// the authored list, the three LQL drivers stop exercising identical
// statement sequences and the design's driver-parity property is silently
// gone. Pin it with the real splitter rather than trusting the migration.
let joined = entries.join(" ");
let resplit: Vec<String> = larql_lql::split_statements(&joined)
    .iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
assert_eq!(resplit, entries, "cell {id:?} does not survive join/split");
```

- [x] **Step 3: Keep the existing matrix leg runnable**

`run_matrix.py` does `subst(c["lql"])` and would break on a list — invisibly,
because the matrix legs are currently gated off. One line: join the list at the
use site. Every commit leaves the harness runnable.

- [x] **Step 4: Write the failing driver test**

`scripts/lql_matrix/drivers_test.py` — no splitter tests, because there is no
splitter. The full 4 drivers x 2 cell-shapes matrix, four acceptances and four
rejections, plus the unknown-driver rejection:

```python
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import drivers as D

CELL_LQL = {"id": "c", "cat": "x", "lql": ['USE "v";', "STATS;"]}
CELL_CLI = {"id": "c", "cat": "x", "argv": ["verify", "{{VINDEX}}"]}


def test_lql_driver_joins_with_spaces_into_one_shot_batch():
    argv, stdin = D.build("lql", CELL_LQL, "/bin/larql")
    assert argv == ["/bin/larql", "lql", 'USE "v"; STATS;']
    assert stdin is None


def test_repl_pipe_sends_one_statement_per_line():
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
    with pytest.raises(ValueError):
        D.build("nope", CELL_LQL, "/bin/larql")


# A driver must never fabricate a missing field: a fabricated invocation would
# be captured and read as a real result.
@pytest.mark.parametrize("driver", ["lql", "repl-pipe", "repl-pty"])
def test_cli_cell_under_an_lql_driver_raises(driver):
    with pytest.raises(KeyError):
        D.build(driver, CELL_CLI, "/bin/larql")


def test_lql_cell_under_cli_driver_raises():
    with pytest.raises(KeyError):
        D.build("cli", CELL_LQL, "/bin/larql")


def test_a_string_lql_cell_is_rejected_not_silently_iterated():
    # A str is iterable, so a missed migration would splice a cell into
    # single characters and produce 60 nonsense statements rather than fail.
    with pytest.raises(TypeError):
        D.build("repl-pipe", {"id": "c", "cat": "x", "lql": 'USE "v"; STATS;'},
                "/bin/larql")
```

- [x] **Step 5: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest drivers_test.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'drivers'`

- [x] **Step 6: Write the implementation**

```python
#!/usr/bin/env python3
"""Build the argv and stdin for one cell under one driver.

Four drivers over the same corpora, because a driver is a distinct user
surface and none covers another:

  lql        larql lql "<statements>"   one-shot batch
  repl-pipe  larql repl, statements on stdin, non-tty
  repl-pty   larql repl under a pseudo-terminal
  cli        the subcommand invoked directly

A cell's `lql` is a LIST of statements. There is no splitter here and there
must not be one: LQL's grammar lives in crates/larql-lql, a second
implementation of it in Python would drift, and statement boundaries are
authored data anyway. `larql_lql::parse` rejects a multi-statement line
(`unexpected trailing token`), which is why the REPL drivers send one
statement per line rather than the whole cell.

repl-pty appends `exit` because a terminal has no EOF: without it the session
would run to the cell timeout every time.

This module knows nothing about corpora, sequencing, or capture. It maps a
cell to an invocation and stops.
"""

DRIVERS = ("lql", "repl-pipe", "repl-pty", "cli")


def _statements(cell):
    stmts = cell["lql"]
    if isinstance(stmts, str):
        raise TypeError(
            f"cell {cell.get('id')!r}: `lql` is a str, expected a list of "
            "statements. A str is iterable, so accepting one here would "
            "splice the cell into single characters instead of failing.")
    return stmts


def build(driver, cell, larql):
    """Return (argv, stdin_bytes). stdin_bytes is None when nothing is written.

    Raises ValueError on an unknown driver, KeyError when the cell lacks the
    field the driver needs, TypeError on an unmigrated string cell — never
    fabricates an invocation, because a fabricated one would be captured and
    read as a real result.
    """
    if driver == "lql":
        return [larql, "lql", " ".join(_statements(cell))], None
    if driver in ("repl-pipe", "repl-pty"):
        lines = list(_statements(cell))
        if driver == "repl-pty":
            lines.append("exit")
        return [larql, "repl"], ("\n".join(lines) + "\n").encode("utf-8")
    if driver == "cli":
        return [larql, *cell["argv"]], None
    raise ValueError(f"unknown driver {driver!r}; expected one of {DRIVERS}")
```

- [x] **Step 7: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest drivers_test.py -q`
Expected: PASS. (Counts in this plan have drifted as later work added tests —
run the suite for the current number rather than trusting a figure written
before the tests existed.)

- [x] **Step 8: Commit**

Corpus migration, Rust guard, `run_matrix.py` join and the driver module go in
together: each alone leaves the harness in a state where a cell shape and its
readers disagree.


### Task 3: Wire the driver into `run_matrix.py`, `lql` behaviour unchanged

**Files:**
- Modify: `scripts/lql_matrix/run_matrix.py`
- Modify: `scripts/lql_matrix/run_matrix.sh`
- Create: `scripts/lql_matrix/run_matrix_test.py`

**Interfaces:**
- Consumes: `drivers.build(driver, cell, larql)`, `drivers.DRIVERS` (Task 2).
- Produces: `run_matrix.py <level> <vindex> <corpus> <out> [--driver NAME]`, default `lql`. Capture files become `<cells>/<level>.<driver>.<cell_id>.{out,err}`. Rows gain `"driver"`. Task 4 adds the pty path; Tasks 6–8 add the `cli` driver and sequencing.

- [x] **Step 1: Write the failing test**

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

- [x] **Step 2: Run it to make sure it fails**

Run: `cd scripts/lql_matrix && python3 -m pytest run_matrix_test.py -q`
Expected: FAIL — `run_matrix.py` does not accept `--driver` (`ValueError: too many values to unpack`).

- [x] **Step 3: Rewrite `run_matrix.py`'s argument handling and cell loop**

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

- [x] **Step 4: Update the shim's documented API**

In `scripts/lql_matrix/run_matrix.sh`, replace the line reading
`#   LARQL_BIN  MODEL_ID  TMPROOT  WRAP  CELL_TIMEOUT` with:

```bash
#   LARQL_BIN  MODEL_ID  TMPROOT  WRAP  CELL_TIMEOUT
# Trailing flags pass through, e.g. --driver repl-pipe (default lql).
```

- [x] **Step 5: Run the tests to verify they pass**

Run: `cd scripts/lql_matrix && python3 -m pytest -q`
Expected: PASS. (Do not pin a count here — the suite has since grown past it.)

- [x] **Step 6: Commit**

```bash
git add scripts/lql_matrix/run_matrix.py scripts/lql_matrix/run_matrix.sh scripts/lql_matrix/run_matrix_test.py
git commit -m "harness: run_matrix takes a --driver, default lql

Capture files are named by driver so the same cell under two drivers
does not collide, and the row records which driver ran. A test asserts
the row carries none of bucket/err_signal/err_line/stdout_head — the
derived fields whose removal this harness depends on."
```

- [x] **Step 7: Demonstrate on a runner — the `lql` driver must be unchanged**

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

> **Step 7 result (run 31268145895, all 21 legs + model-lifecycle green).**
> Leg `smol135.native.all`: 64 cells, `drivers == {'lql'}`, no derived field
> present, every capture on disk, captures named
> `smol135.native.all.lql.<cell>.{out,err}` — 128 files, 110 297 bytes. The
> `lql` path is unchanged by the driver parameter. Task 4 is unblocked.
>
> The first attempt at this step failed 21/21 on two defects of mine; see
> `docs/diagnoses/task3-matrix-demonstration.md`.

---

> **Simplification note — Tasks 4–10 rewritten 2026-08-09.** The coverage
> requirements are unchanged: 38 subcommands × (`--help` + a real invocation),
> 4 drivers with no redundancy assumptions, per-cell REPL sessions plus a
> long-session leg per model, dependency-ordered and permuted cells with the
> order in the row, and verbatim `.out`/`.err`/`.merged` captures. What went is
> mechanism:
>
> - **`corpus_lint.lint_cli_corpus` and the hardcoded Python `SUBCOMMANDS`
>   tuple** — a Python re-implementation of clap's argument grammar, forbidden
>   by the same rule that deleted `corpus_lint.py` in Task 1 and the Python
>   splitter in Task 2. Replaced by a clap-owned test inside `larql-cli`
>   (Tasks 6–7), which is strictly *more* coverage: it rejects a wrong argv
>   shape with clap's own error instead of merely checking `argv[0]` is a known
>   name.
> - **`sequence`'s `unsatisfied` field** and the six single-shape tests that
>   asserted its contents. The spec asks that a capture be tied to the order
>   that produced it — `order_index` plus the meta row's `order_seed` does that.
>   `unsatisfied` was a harness-computed opinion about a cell, and rows carry
>   facts about the process only. The eight graph shapes survive as one
>   parametrized test.
> - **`run_under_pty(argv, stdin_bytes, …)`** — the static-payload shape Task 0
>   measured as lossy. Statements now pass through as a list and are written one
>   per prompt (Task 4).
> - **Restated code and argued rationale** — `drivers.build`, `run_matrix.py`'s
>   cell loop, `probe_pty.py`'s read loop and the existing tests are referenced
>   by name rather than re-specified.
>
> Three references *before* Task 4 are now stale and were deliberately left
> untouched: the File Structure rows for `corpus_lint.py`/`corpus_lint_test.py`
> (neither exists; Task 1 records why), and the Exhaustive Coverage row for
> `corpus_lint` rejections. The row for `sequence.sequence`'s eight graph shapes
> is still accurate.

### Task 4: The `repl-pty` driver

`subprocess` cannot give a child a tty, and Task 0 measured that a terminal
loses statements written to it up front — `repl-pty` lost `STATS;` *and*
`exit`, `script(1)` lost `STATS;`. So this driver writes **one statement per
prompt**.

That supersedes one Task 2 interface detail, recorded here so the change is
auditable: `build("repl-pty", …)` returned `(argv, b"…\nexit\n")`, and a
prompt-paced writer would have had to split that payload back apart, making the
`\n` a contract nothing declared or tested. The statement list passes through
instead, so boundaries stay authored data.

**Files:**
- Modify, all under `scripts/lql_matrix/`: `drivers.py`, `drivers_test.py`, `run_matrix.py`, `run_matrix_test.py`

**Interfaces:**
- Consumes: `probe_pty.write_all` and the read loop it proves on a runner (Task 0).
- Produces:
  - `drivers.pty_lines(cell) -> list[str]` — the cell's statements plus `exit`; `build("repl-pty", …)` now returns `(argv, None)`.
  - `run_matrix.run_under_pty(argv, lines, merged_path, timeout_s) -> tuple[int, list[str]]` — exit status and the lines it *actually* wrote.
  - `repl-pty` in `IMPLEMENTED_DRIVERS`; rows for this driver gain `merged` / `merged_bytes`.

- [ ] **Step 1: Write the failing tests**

In `drivers_test.py`, replace `test_repl_pty_appends_exit_because_a_terminal_has_no_eof`
with the same assertion against `pty_lines` (`== ['USE "v";', "STATS;", "exit"]`)
plus `build` returning `stdin is None`, and extend the existing KeyError and
TypeError rejection cases to cover `pty_lines`.

Append to `run_matrix_test.py`:

```python
def test_run_under_pty_writes_one_line_per_prompt_and_captures_merged(tmp_path):
    sys.path.insert(0, HERE)
    import pathlib
    import run_matrix as R
    # Stands in for `larql repl`: a tty check, then a prompt before EVERY read.
    # Writing all lines up front is what Task 0 measured as lossy, so a driver
    # that did it would drop lines here exactly as it did on the runner.
    script = tmp_path / "fakerepl"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "if [ -t 0 ]; then echo TTY_YES; else echo TTY_NO; fi\n"
        "while true; do printf 'larql> '; read -r l || exit 0;\n"
        "  [ \"$l\" = exit ] && { echo Goodbye.; exit 0; }; echo \"RAN:$l\"; done\n",
        encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    merged = tmp_path / "m.merged"
    rc, written = R.run_under_pty([str(script)], ["A;", "B;", "exit"],
                                  pathlib.Path(merged), 30)
    text = merged.read_text(errors="replace")
    assert "TTY_YES" in text
    assert "RAN:A;" in text and "RAN:B;" in text     # neither statement lost
    assert "Goodbye." in text
    assert written == ["A;", "B;", "exit"]
    assert rc == 0


def test_repl_pty_driver_writes_a_merged_capture(tmp_path):
    rows, tp = _run(tmp_path, "repl-pty")
    assert rows[0]["driver"] == "repl-pty"
    assert rows[0]["merged_bytes"] == (tp / rows[0]["merged"]).stat().st_size
```

Run: `cd scripts/lql_matrix && python3 -m pytest -q -k pty`
Expected: FAIL — no `run_matrix.run_under_pty`, and `--driver repl-pty` rejected by `IMPLEMENTED_DRIVERS`.

- [ ] **Step 2: Implement the driver**

In `drivers.py`:

```python
def pty_lines(cell):
    """The lines a pty session sends, in order. `exit` is appended because a
    terminal has no EOF — without it the session runs to the cell timeout."""
    return [*_statements(cell), "exit"]
```

and `build("repl-pty", …)` returns `([larql, "repl"], None)`: it writes no
stdin, because the pty runner does.

`probe_pty.py` already carries the read loop, verified on a runner: `write_all`
looping over short writes, `select` on a 1 s tick, `os.waitpid(WNOHANG)` on a
quiet fd **keeping** the status in `reaped`, `EIO` as the pty's EOF, and
`killpg` rather than `kill` on timeout. Read its docstrings before touching
this — each paragraph records a measurement. Only its up-front
`write_all(fd, payload)` is replaced.

In `run_matrix.py`: add `errno`, `fcntl`, `pty`, `select`, `signal`, `struct`,
`termios` to the imports and `from probe_pty import write_all` beside
`import drivers`; add `"repl-pty"` to `IMPLEMENTED_DRIVERS`, deleting the
paragraph of its comment saying the driver does not exist yet; and add above
`main()`:

```python
PROMPT = b"larql> "  # printed before every read — the design's Capture section


def run_under_pty(argv, lines, merged_path, timeout_s):
    """Run argv on a pseudo-terminal, releasing one line per prompt observed.

    rustyline on a pipe is not the code path rustyline takes on a terminal, and
    whether they agree is what the two REPL legs exist to show. Task 0 measured
    that statements written up front are LOST under a terminal, so this counts
    prompts in the accumulated stream and writes the next line each time a new
    one appears. That is input pacing on a live stream, not post-processing:
    merged_path still receives every byte, control sequences and \\r included.

    Returns (exit status, lines actually written) — if larql dies mid-cell the
    tail was never sent, and the row must say what was sent, not what was meant.
    """
    pending, written, answered, buf = list(lines), [], 0, b""
    pid, fd = pty.fork()
    if pid == 0:
        try:
            os.execvp(argv[0], argv)
        except Exception:
            os._exit(127)
    # 80x24 so wrapping is a property of the harness, not of the runner.
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    except OSError:
        pass
    # ... read loop lifted from probe_pty.main() with the up-front write gone,
    # and this run after each `f.write(chunk); f.flush()`:
    #
    #     buf += chunk              # accumulated, because a prompt can straddle
    #     seen = buf.count(PROMPT)  # two reads and a per-chunk count misses it
    #     while seen > answered and pending:
    #         line = pending.pop(0)
    #         write_all(fd, (line + "\n").encode("utf-8"))
    #         written.append(line)
    #         answered += 1
    #
    # The deadline, killpg, `reaped`, EIO and the final waitpid are unchanged.
    # Return (exitcode, written) instead of printing.
```

Then in the cell loop, branch the `t0 = time.monotonic()` … `dur_ms = …` block
on `driver == "repl-pty"`: call `run_under_pty(argv_cmd, drivers.pty_lines(cell),
mergedf, int(cell_timeout))`, set `stdin_repr` to `"".join(l + "\n" for l in
written)`, and write empty `outf`/`errf` so every row indexes the same three
paths. The other branch keeps `subprocess.run` and derives `stdin_repr` from
`stdin_bytes` as today. The row's `"stdin"` becomes `stdin_repr`, and
`merged` / `merged_bytes` go after `"stderr_bytes"` (`None` for other drivers).

The pty path does not use the `timeout` / `/usr/bin/time` wrapper argv — it
execs the child directly and enforces the deadline itself — so `peak_rss_kb`
stays `""`, the existing not-measured value.

- [ ] **Step 3: Verify and commit**

Run: `cd scripts/lql_matrix && python3 -m pytest -q` — expected PASS.

```bash
git add scripts/lql_matrix/drivers.py scripts/lql_matrix/drivers_test.py \
        scripts/lql_matrix/run_matrix.py scripts/lql_matrix/run_matrix_test.py
git commit -m "harness: repl-pty driver — a real pty, one statement per prompt

Task 0 measured that statements written to a terminal up front are lost:
repl-pty lost STATS; and exit, script(1) lost STATS;. The runner now
releases one line per prompt it observes and records the lines it
actually wrote. Control sequences and \\r are captured verbatim."
```

- [ ] **Step 4: Demonstrate both REPL drivers on a runner**

The fake binary cannot tell you what rustyline does. In the workflow step Task 3
edited, loop the three LQL drivers (hardcoded here; Task 10 replaces the list
with `matrix.leg.drivers`):

```yaml
          for DRV in lql repl-pipe repl-pty; do
            LARQL_BIN=./bin/larql MODEL_ID="${CORPUS_MODEL}" \
            TMPROOT="$(mktemp -d)" CELL_TIMEOUT=900 \
            scripts/lql_matrix/run_matrix.sh \
              "${NAME}" "out/${NAME}.vindex" scripts/lql_matrix/commands.jsonl \
              "out/results-${NAME}.${DRV}.jsonl" --driver "$DRV"
          done
```

Push, then read `cells/*.repl-pty.*.merged` for a multi-statement cell against
the same cell's `.lql.*.out`:

- **every statement in the cell appears in `.merged`** — the pacing fixed what
  Task 0 measured. This is the specific regression: in the probe, `STATS;`
  produced no output and no error under either pty mechanism.
- **a statement is still missing, or the leg timed out** — the pacing is wrong.
  Return to Step 2; do not proceed.
- **`.merged` is full of `\r` and `^[[?2004h`** — correct, and they stay.
- **`repl-pipe` differs from `lql`** — a finding about larql, not a harness bug.

**Do not start Task 5 until these captures have been read.**

---

### Task 5: Driver axis in `gen_legs.py`

**Files:**
- Modify: `scripts/lql_matrix/gen_legs.py`, `scripts/lql_matrix/gen_legs_test.py`

**Interfaces:**
- Produces: every leg dict gains `"drivers": list[str]` and `"long_session": bool`. Task 10's workflow reads both.

- [ ] **Step 1: Write the failing tests**

Append to `scripts/lql_matrix/gen_legs_test.py`:

```python
def test_every_leg_declares_drivers_from_the_known_set():
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


def test_every_model_has_a_cli_leg():
    cli = {lg["name"].split(".", 1)[0] for lg in G.build_legs() if lg["drivers"] == ["cli"]}
    assert cli == {m for m, _ in G.SAFETENSORS}
```

Run: `cd scripts/lql_matrix && python3 -m pytest gen_legs_test.py -q` — expected FAIL, `KeyError: 'drivers'`.

- [ ] **Step 2: Add the axis**

Give `leg()` two more keyword parameters, `drivers=("lql",)` and
`long_session=False`, and put `"drivers": list(drivers)` and
`"long_session": long_session` in the dict it returns. Then in `build_legs()`:

- section 1 (native extraction) passes `drivers=("lql", "repl-pipe", "repl-pty")`;
- immediately before `return legs`, append two loops over `SAFETENSORS`:

```python
    # 4. LONG SESSION — the whole corpus through ONE repl session per model.
    #    The only thing that exercises state accumulating across commands.
    #    Cells contaminate each other by design; that is the point, and it is
    #    why this is a separate leg rather than a replacement for the per-cell
    #    backbone.
    for mid, hf in SAFETENSORS:
        legs.append(leg(f"{mid}.longsession", hf, "extract", level="all",
                        drivers=("repl-pipe", "repl-pty"), long_session=True))

    # 5. CLI SURFACE — every subcommand, help and real invocation. Driven per
    #    model because most subcommands need a produced vindex.
    for mid, hf in SAFETENSORS:
        legs.append(leg(f"{mid}.cli", hf, "extract", level="all", drivers=("cli",)))
```

- [ ] **Step 3: Verify and commit**

```bash
cd scripts/lql_matrix && python3 -m pytest -q
LQL_MATRIX_ONLY="smol135,smol135base" python3 gen_legs.py \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d),'legs'); [print(' ',l['name'],l['drivers'],'long' if l['long_session'] else '') for l in d]"
```

Expected: PASS, and 25 legs — the 21 existing plus `smol135.longsession`,
`smol135base.longsession`, `smol135.cli`, `smol135base.cli`.

```bash
git add scripts/lql_matrix/gen_legs.py scripts/lql_matrix/gen_legs_test.py
git commit -m "harness: driver axis, long-session legs, cli legs

Native legs declare all three LQL drivers. One long-session leg per
model sends the whole corpus through a single repl session — the only
thing that exercises cross-command state, and excluded from the lql
driver because a one-shot batch starts a fresh Session each time."
```

---

> **Pause point.** Tasks 1–5 deliver working REPL coverage on their own. Tasks 6–10 add the CLI surface and dependency sequencing.

---

### Task 6: `cli-help.jsonl` — every subcommand's help, validated by clap

The corpus is validated by **clap**, which owns the argument grammar, in a test
module inside `larql-cli` — the same shape as Task 1's
`matrix_corpus_wellformed.rs`, which validates the LQL corpus with LQL's own
parser. There is no Python subcommand list and no Python argv checker; a second
implementation of clap's grammar would drift from it, and Task 0 spent ten
captures discovering that guessed argv shapes are wrong.

**Files:**
- Create: `scripts/lql_matrix/cli-help.jsonl`
- Create: `crates/larql-cli/src/matrix_cli_corpus.rs`
- Modify: `crates/larql-cli/src/main.rs` — one `#[cfg(test)] mod` line
- Modify: `scripts/lql_matrix/run_matrix.py` — `IMPLEMENTED_DRIVERS`

**Interfaces:**
- Consumes: `crate::Cli` via `clap::CommandFactory` (private to the crate root, visible to a child module — the existing `trampoline_tests` module is the precedent).
- Produces: `cargo test -p larql-cli --bin larql matrix_cli_corpus`, and helpers Task 7 extends.

- [ ] **Step 1: Write the failing test**

Add `#[cfg(test)]\nmod matrix_cli_corpus;` beside the existing `mod` lines in
`crates/larql-cli/src/main.rs`, and create
`crates/larql-cli/src/matrix_cli_corpus.rs`:

```rust
//! The CLI corpora are well-formed according to CLAP, not according to a copy
//! of clap. `Cli::command()` is the exact grammar the binary parses with, so a
//! wrong argv shape fails here with clap's own error instead of costing a CI
//! leg — which is what the ten corrected rows in
//! docs/diagnoses/task0-cli-repl-probe.md cost the first time.
//!
//! Asserts nothing about what a command DOES. Whether an accepted invocation
//! succeeds, errors or times out is the run's business, in the captures.
use clap::CommandFactory;
use std::collections::BTreeSet;
use std::path::PathBuf;

struct Row { line: usize, id: String, argv: Vec<String>, raw: serde_json::Value }

/// Every row of a corpus under `scripts/lql_matrix/`, read the way
/// `crates/larql-lql/tests/matrix_corpus_wellformed.rs` reads the LQL corpora.
/// `{{VINDEX}}`/`{{MODEL}}`/`{{TMP}}` stay unsubstituted — clap parses them as
/// ordinary strings, so keep placeholders out of typed positions (a numeric
/// value_parser rejects `{{TMP}}`).
fn corpus(name: &str) -> Vec<Row> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/lql_matrix").join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()).map(|(i, l)| {
        let raw: serde_json::Value = serde_json::from_str(l)
            .unwrap_or_else(|e| panic!("{name}:{}: not JSON: {e}", i + 1));
        let id = raw["id"].as_str().expect("cell has no string `id`").to_string();
        let argv = raw["argv"].as_array()
            .unwrap_or_else(|| panic!("{name}:{}: cell {id:?} has no `argv` array", i + 1))
            .iter().map(|v| v.as_str().expect("argv entry is not a string").to_string())
            .collect();
        Row { line: i + 1, id, argv, raw }
    }).collect()
}

/// Every subcommand clap knows, minus clap's own auto-generated `help`.
fn subcommands() -> BTreeSet<String> {
    crate::Cli::command().get_subcommands()
        .map(|s| s.get_name().to_string()).filter(|n| n != "help").collect()
}

/// The argv the binary would see.
fn full_argv(r: &Row) -> Vec<String> {
    std::iter::once("larql".to_string()).chain(r.argv.iter().cloned()).collect()
}

/// The subcommands a corpus invokes. One it never invokes is one nothing tests.
fn covered(rows: &[Row]) -> BTreeSet<String> {
    rows.iter().map(|r| r.argv[0].clone()).collect()
}

#[test]
fn cli_help_corpus_covers_every_subcommand_and_every_row_renders_help() {
    let rows = corpus("cli-help.jsonl");
    let (want, got) = (subcommands(), covered(&rows));
    assert_eq!(got, want, "uncovered: {:?}", want.difference(&got));
    let cmd = crate::Cli::command();
    for r in &rows {
        match cmd.clone().try_get_matches_from(full_argv(r)) {
            Err(e) if e.kind() == clap::error::ErrorKind::DisplayHelp => {}
            other => panic!("cli-help.jsonl:{}: cell {:?} did not render help: {:?}",
                            r.line, r.id, other.err().map(|e| e.to_string())),
        }
    }
}
```

Run: `cargo test -p larql-cli --bin larql matrix_cli_corpus` — expected FAIL,
`cannot read …/cli-help.jsonl`.

- [ ] **Step 2: Generate the help corpus**

The 38 names in clap's kebab-case form. They are exact values; the test above
guards them against drift, so nothing hand-maintains a second copy.

```bash
cd /home/metavacua/larql-vindex3-03-08-2026
python3 - <<'PY'
import json
SUBCOMMANDS = """run chat pull model link list show slice publish rm bench
dec-bench k3-ledger accuracy shannon serve repl lql extract extract-index
build compile convert hf verify diag parity moe-locality recipe capabilities
card query describe stats validate merge filter dev""".split()
assert len(SUBCOMMANDS) == 38, len(SUBCOMMANDS)
with open("scripts/lql_matrix/cli-help.jsonl", "w", encoding="utf-8") as f:
    for c in SUBCOMMANDS:
        f.write(json.dumps({"id": f"help.{c}", "cat": "cli-help",
                            "argv": [c, "--help"]}) + "\n")
print(len(SUBCOMMANDS), "help cells")
PY
```

Expected: `38 help cells`, and the cargo test above now passes. If it reports an
uncovered name, clap has a subcommand this list does not — add it; that is the
drift the test exists to catch.

- [ ] **Step 3: Make `--driver cli` reachable, then commit**

Add `"cli"` to `run_matrix.IMPLEMENTED_DRIVERS` and delete the paragraph of its
comment that says the `cli-*.jsonl` corpora do not exist yet.

```bash
cd scripts/lql_matrix && python3 -m pytest -q
cd /home/metavacua/larql-vindex3-03-08-2026 && cargo test -p larql-cli --bin larql matrix_cli_corpus
git add scripts/lql_matrix/cli-help.jsonl scripts/lql_matrix/run_matrix.py \
        crates/larql-cli/src/matrix_cli_corpus.rs crates/larql-cli/src/main.rs
git commit -m "harness: a --help cell for all 38 subcommands, checked by clap

The corpus is validated with Cli::command(), the same grammar the binary
parses with, so a name clap does not know fails locally instead of on a
runner. Adding a subcommand fails the test until a cell covers it. The
matrix previously drove 4 of 38."
```

- [ ] **Step 4: Demonstrate 38 real `--help` invocations on a runner**

The cheapest real exercise of the `cli` driver — no vindex, no model, no
network — so it isolates "does the driver invoke correctly" from "do the
subcommands work". Inside the driver loop from Task 4:

```yaml
            if [ "$DRV" = "cli" ]; then
              scripts/lql_matrix/run_matrix.sh "${NAME}" "out/${NAME}.vindex" \
                scripts/lql_matrix/cli-help.jsonl \
                "out/results-${NAME}.cli.help.jsonl" --driver cli
            fi
```

Add `cli` to the hardcoded driver list, push, and read the 38 rows' exit codes
and `stdout_bytes`. Task 0 measured 38 of 38 at `exit=0`; a subcommand whose
`--help` now exits non-zero or prints nothing is a real finding about the CLI —
record it, do not edit the cell to make it pass. **Do not start Task 7 until
these have been read.**

---

### Task 7: `cli-commands.jsonl` — real invocations with declared dependencies

**Files:**
- Create: `scripts/lql_matrix/cli-commands.jsonl`
- Modify: `crates/larql-cli/src/matrix_cli_corpus.rs`

**Interfaces:**
- Consumes: `corpus`, `subcommands`, `covered` (Task 6).
- Produces: a corpus whose cells each carry `needs: list[str]` and `produces: list[str]` over the vocabulary `{"vindex", "cache-entry", "graph-file", "recipe", "vlp"}`. Task 8's sequencer consumes it.

- [ ] **Step 1: Write the failing test**

Append to `crates/larql-cli/src/matrix_cli_corpus.rs`:

```rust
#[test]
fn cli_commands_corpus_invokes_every_subcommand_with_argv_clap_accepts() {
    let rows = corpus("cli-commands.jsonl");
    let (want, got) = (subcommands(), covered(&rows));
    assert_eq!(got, want, "never invoked: {:?}", want.difference(&got));
    let cmd = crate::Cli::command();
    for r in &rows {
        // The six rows Task 0 could not settle ship with a marker, which clap
        // would happily accept as a free-form string value. Fail on it here.
        for a in &r.argv {
            assert!(!a.contains("TAKE FROM PROBE"),
                    "cli-commands.jsonl:{}: cell {:?} still has a placeholder: {a:?}", r.line, r.id);
        }
        // Ok(_) is the bar, not "clap did not crash". A row that only reaches
        // DisplayHelp or MissingRequiredArgument is not a real invocation —
        // exactly the shapes Task 0 measured as wrong (shannon, dec-bench,
        // dev, card, hf, k3-ledger).
        if let Err(e) = cmd.clone().try_get_matches_from(full_argv(r)) {
            panic!("cli-commands.jsonl:{}: cell {:?} is not a real invocation:\n{e}", r.line, r.id);
        }
    }
}

#[test]
fn cli_commands_artifacts_come_from_a_closed_vocabulary() {
    const VOCAB: [&str; 5] = ["vindex", "cache-entry", "graph-file", "recipe", "vlp"];
    for r in corpus("cli-commands.jsonl") {
        for key in ["needs", "produces"] {
            let list = r.raw[key].as_array()
                .unwrap_or_else(|| panic!("cli-commands.jsonl:{}: {:?} has no {key:?} list", r.line, r.id));
            for v in list {
                let a = v.as_str().expect("artifact is not a string");
                assert!(VOCAB.contains(&a),
                        "cli-commands.jsonl:{}: {:?} unknown {key} {a:?}", r.line, r.id);
            }
        }
    }
}
```

Run: `cargo test -p larql-cli --bin larql matrix_cli_corpus` — expected FAIL,
`cannot read …/cli-commands.jsonl`.

- [ ] **Step 2: Write the corpus**

One line per subcommand, minimum; more where a subcommand has distinct modes.
`{{VINDEX}}`, `{{MODEL}}` and `{{TMP}}` are substituted by `run_matrix.py` and
must stay out of typed positions, since the test above parses them literally.

**Six rows below are marked `TAKE FROM PROBE` because nobody knows their shape
yet.** Task 0 recorded only "enumerate subcommands" for them. Do not guess a
third time: read the argument shape out of the probe artifact's
`help.<name>.out` (regenerate with a `lql-strategy-matrix` dispatch at
`scope: probe` if it has expired), and let the Step 1 test confirm it — clap
rejects a wrong shape locally now, which is what makes this cheap.

```json
{"id": "cli.extract", "cat": "produce", "argv": ["extract", "{{MODEL}}", "-o", "{{TMP}}/cli.vindex", "--level", "all"], "needs": [], "produces": ["vindex"]}
{"id": "cli.extract-index", "cat": "produce", "argv": ["extract-index", "{{MODEL}}", "-o", "{{TMP}}/cli-idx.vindex", "--level", "browse"], "needs": [], "produces": ["vindex"]}
{"id": "cli.convert", "cat": "produce", "argv": ["convert", "quantize", "q4k", "--input", "{{VINDEX}}", "--output", "{{TMP}}/cli-q4k.vindex"], "needs": ["vindex"], "produces": ["vindex"]}
{"id": "cli.build", "cat": "produce", "argv": ["build", "{{TMP}}/Vindexfile", "-o", "{{TMP}}/cli-build.vindex"], "needs": [], "produces": ["vindex"]}
{"id": "cli.slice", "cat": "produce", "argv": ["slice", "{{VINDEX}}", "--output", "{{TMP}}/cli-slice.vindex", "--preset", "TAKE FROM PROBE: help.slice.out lists --parts/--preset values"], "needs": ["vindex"], "produces": ["vindex"]}
{"id": "cli.compile", "cat": "produce", "argv": ["compile", "--base", "{{MODEL}}", "--vindex", "{{TMP}}/empty.vlp", "--output", "{{TMP}}/cli-compiled"], "needs": ["vlp"], "produces": []}
{"id": "cli.link", "cat": "register", "argv": ["link", "{{VINDEX}}"], "needs": ["vindex"], "produces": ["cache-entry"]}
{"id": "cli.pull", "cat": "register", "argv": ["pull", "chrishayuk/gemma-3-4b-it-vindex"], "needs": [], "produces": ["cache-entry"]}
{"id": "cli.model", "cat": "register", "argv": ["model", "pull", "{{MODEL}}"], "needs": [], "produces": []}
{"id": "cli.list", "cat": "consume", "argv": ["list"], "needs": ["cache-entry"], "produces": []}
{"id": "cli.show", "cat": "consume", "argv": ["show", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.verify", "cat": "consume", "argv": ["verify", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.diag", "cat": "consume", "argv": ["diag", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.capabilities", "cat": "consume", "argv": ["capabilities"], "needs": [], "produces": []}
{"id": "cli.run", "cat": "consume", "argv": ["run", "{{VINDEX}}", "Paris is the capital of", "--max-tokens", "4"], "needs": ["vindex"], "produces": []}
{"id": "cli.chat", "cat": "consume", "argv": ["chat", "{{VINDEX}}", "--max-tokens", "1"], "needs": ["vindex"], "produces": []}
{"id": "cli.serve", "cat": "consume", "argv": ["serve", "{{VINDEX}}", "--port", "18080"], "needs": ["vindex"], "produces": []}
{"id": "cli.bench", "cat": "consume", "argv": ["bench", "{{VINDEX}}", "--tokens", "4"], "needs": ["vindex"], "produces": []}
{"id": "cli.dec-bench", "cat": "consume", "argv": ["dec-bench", "TAKE FROM PROBE: help.dec-bench.out — bare invocation only printed help"], "needs": [], "produces": []}
{"id": "cli.k3-ledger", "cat": "consume", "argv": ["k3-ledger", "TAKE FROM PROBE: help.k3-ledger.out — 'report' is not a subcommand"], "needs": [], "produces": []}
{"id": "cli.accuracy", "cat": "consume", "argv": ["accuracy", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.shannon", "cat": "consume", "argv": ["shannon", "TAKE FROM PROBE: help.shannon.out — shannon <COMMAND>"], "needs": ["vindex"], "produces": []}
{"id": "cli.parity", "cat": "consume", "argv": ["parity", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.moe-locality", "cat": "consume", "argv": ["moe-locality", "{{VINDEX}}"], "needs": ["vindex"], "produces": []}
{"id": "cli.publish", "cat": "consume", "argv": ["publish", "{{VINDEX}}", "--repo", "example/does-not-exist"], "needs": ["vindex"], "produces": []}
{"id": "cli.hf", "cat": "consume", "argv": ["hf", "TAKE FROM PROBE: help.hf.out — 'upload' is not a subcommand"], "needs": ["vindex"], "produces": []}
{"id": "cli.recipe", "cat": "consume", "argv": ["recipe", "validate", "{{TMP}}/recipe.yaml"], "needs": ["recipe"], "produces": []}
{"id": "cli.card", "cat": "consume", "argv": ["card", "TAKE FROM PROBE: help.card.out — card <COMMAND>"], "needs": ["recipe"], "produces": []}
{"id": "cli.dev", "cat": "consume", "argv": ["dev", "TAKE FROM PROBE: help.dev.out — needs a subcommand"], "needs": [], "produces": []}
{"id": "cli.repl", "cat": "consume", "argv": ["repl"], "needs": [], "produces": []}
{"id": "cli.lql", "cat": "consume", "argv": ["lql", "SHOW MODELS;"], "needs": [], "produces": []}
{"id": "cli.query", "cat": "graph", "argv": ["query", "--graph", "{{TMP}}/graph.json", "France"], "needs": ["graph-file"], "produces": []}
{"id": "cli.describe", "cat": "graph", "argv": ["describe", "--graph", "{{TMP}}/graph.json", "France"], "needs": ["graph-file"], "produces": []}
{"id": "cli.stats", "cat": "graph", "argv": ["stats", "{{TMP}}/graph.json"], "needs": ["graph-file"], "produces": []}
{"id": "cli.validate", "cat": "graph", "argv": ["validate", "{{TMP}}/graph.json"], "needs": ["graph-file"], "produces": []}
{"id": "cli.merge", "cat": "graph", "argv": ["merge", "{{TMP}}/graph.json", "{{TMP}}/graph.json", "-o", "{{TMP}}/merged.json"], "needs": ["graph-file"], "produces": ["graph-file"]}
{"id": "cli.filter", "cat": "graph", "argv": ["filter", "{{TMP}}/graph.json", "--min-confidence", "0.5", "-o", "{{TMP}}/filtered.json"], "needs": ["graph-file"], "produces": ["graph-file"]}
{"id": "cli.rm", "cat": "destroy", "argv": ["rm", "{{VINDEX}}"], "needs": ["cache-entry"], "produces": []}
```

Several rows will still fail at run time — `cli.publish`/`cli.hf` target a
nonexistent repo, `cli.recipe`/`cli.card` need a recipe file no cell produces,
the graph cells need a graph file no cell produces, `cli.serve`/`cli.chat` do
not self-terminate. **That is intended, and is a different thing from a wrong
argument.** Nothing is skipped for an unsatisfied dependency or a
guessed-inapplicable command; what larql does when asked is the measurement.
Correcting a *wrong argument* is the opposite — that is the harness asking the
question it meant to ask.

Also present by design and not to be pruned: the WIP surfaces the design names
(`build --compile`, `show` on an unimplemented programme, `vector-extract`
components), and `repl`/`lql` as plain subcommands even though they are their
own drivers.

- [ ] **Step 3: Verify and commit**

```bash
cargo test -p larql-cli --bin larql matrix_cli_corpus
cd scripts/lql_matrix && python3 -m pytest -q
git add scripts/lql_matrix/cli-commands.jsonl crates/larql-cli/src/matrix_cli_corpus.rs
git commit -m "harness: a real invocation for every one of the 38 subcommands

Every row must reach Ok(_) through clap's own parser, so a shape that
only prints help or reports a missing required argument fails locally —
the ten shapes Task 0 had to correct on a runner. Cells that cannot
succeed in CI still run: an unsatisfied dependency or an unreachable
remote is recorded, not skipped."
```

- [ ] **Step 4: Demonstrate the real invocations on a runner**

Wire `cli-commands.jsonl` in beside `cli-help.jsonl`, push, and read the stderr
of every non-zero cell. Expect many. Place each in exactly one bucket:

- **the cell's arguments are wrong** — fix the cell; a harness defect.
- **larql rejected a valid request** — leave it; this is the finding.
- **the dependency genuinely is not there** — leave it; the cell still runs.

Do not resolve the third bucket by deleting cells. **Do not start Task 8 until
every non-zero cell has been placed in a bucket.**

---

### Task 8: Dependency ordering and deterministic permutation

**Files:**
- Create: `scripts/lql_matrix/sequence.py`, `scripts/lql_matrix/sequence_test.py`

**Interfaces:**
- Consumes: cells with `id`, `needs`, `produces` (Task 7).
- Produces: `sequence.sequence(cells: list[dict], seed: int) -> list[dict]` — the cells reordered, each a copy carrying `"order_index": int`. Task 9 calls it from `run_matrix.py`.

There is no `unsatisfied` field. The spec asks that a capture be tied to the
order that produced it; `order_index` plus the meta row's `order_seed` does
that. A per-cell list of what the graph could not supply is the harness forming
an opinion about a cell, and rows carry facts about the process only.

- [ ] **Step 1: Write the failing test**

Create `scripts/lql_matrix/sequence_test.py`:

```python
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import sequence as S

A = {"id": "produce", "needs": [], "produces": ["vindex"]}
B = {"id": "consume1", "needs": ["vindex"], "produces": []}
C = {"id": "consume2", "needs": ["vindex"], "produces": []}
D = {"id": "orphan", "needs": ["graph-file"], "produces": []}


def _ids(cells):
    return [c["id"] for c in cells]


# All eight graph shapes. Empty, single, chain and diamond are the acceptance
# cases; all-independent, unsatisfiable, self-cycle and mutual cycle are the
# rejection-shaped ones. None may hang, drop a cell or raise — an unorderable
# cell still runs, because "this breaks when its input is absent" is an
# observation the harness exists to capture.
SHAPES = {
    "empty": [],
    "single": [A],
    "chain": [B, A],
    "diamond": [{"id": "join", "needs": ["b", "c"], "produces": []},
                {"id": "left", "needs": ["a"], "produces": ["b"]},
                {"id": "right", "needs": ["a"], "produces": ["c"]},
                {"id": "root", "needs": [], "produces": ["a"]}],
    "independent": [{"id": f"c{i}", "needs": [], "produces": []} for i in range(5)],
    "unsatisfiable": [A, D],
    "self-cycle": [{"id": "loop", "needs": ["x"], "produces": ["x"]}],
    "mutual-cycle": [{"id": "p", "needs": ["y"], "produces": ["x"]},
                     {"id": "q", "needs": ["x"], "produces": ["y"]}],
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_shape_terminates_emitting_each_cell_once_with_a_dense_index(name):
    import threading
    cells = SHAPES[name]
    box = []
    t = threading.Thread(target=lambda: box.append(S.sequence(cells, 3)), daemon=True)
    t.start()
    t.join(5)
    assert box, f"sequence() did not terminate on the {name} graph"
    out = box[0]
    assert sorted(_ids(out)) == sorted(_ids(cells))
    assert [c["order_index"] for c in out] == list(range(len(out)))


def test_producer_precedes_its_consumers():
    out = _ids(S.sequence([B, C, A], seed=0))
    assert out.index("produce") < out.index("consume1")
    assert out.index("produce") < out.index("consume2")


def test_diamond_orders_root_first_and_join_last():
    out = _ids(S.sequence(SHAPES["diamond"], 0))
    assert out[0] == "root" and out[-1] == "join"


def test_same_seed_gives_the_same_order():
    assert _ids(S.sequence([B, C, A], 7)) == _ids(S.sequence([B, C, A], 7))


def test_different_seeds_permute_independent_cells():
    orders = {tuple(_ids(S.sequence([B, C, A], s))) for s in range(12)}
    assert len(orders) > 1, "independent cells must actually permute"


def test_input_cells_are_not_mutated():
    original = dict(A)
    S.sequence([A, B], 0)
    assert A == original
```

Run: `cd scripts/lql_matrix && python3 -m pytest sequence_test.py -q` — expected FAIL, `ModuleNotFoundError: No module named 'sequence'`.

- [ ] **Step 2: Write the implementation**

Create `scripts/lql_matrix/sequence.py`:

```python
#!/usr/bin/env python3
"""Order cells by declared dependencies, permuting the independent ones.

A consumer run before its producer measures nothing about the consumer —
`verify` on a vindex that does not exist yet reports a missing directory, not
anything about verify. Where two cells are independent, their relative order is
a free variable, and free variables get varied rather than frozen: `seed`
selects the permutation, deterministically, so an ordering-dependent failure is
re-runnable.

A cell whose needs are never produced is NOT dropped — it runs at the end.
Ordering is never used to decide correctness, and this module records no
opinion about a cell beyond where it ran.

Pure: no I/O, no subprocess, no logging.
"""
import random


def sequence(cells, seed):
    """Return cells reordered, each a copy carrying `order_index`.
    Input dicts are not mutated."""
    rng = random.Random(seed)
    remaining = [dict(c) for c in cells]
    satisfied = set()
    ordered = []

    while remaining:
        ready = [c for c in remaining if set(c.get("needs", [])) <= satisfied]
        if not ready:
            # Nothing can ever become ready (a cycle, or an artifact no cell
            # produces). Emit the rest in a seeded order rather than looping.
            rng.shuffle(remaining)
            ordered.extend(remaining)
            break
        pick = ready[rng.randrange(len(ready))]
        ordered.append(pick)
        satisfied.update(pick.get("produces", []))
        remaining.remove(pick)

    for i, c in enumerate(ordered):
        c["order_index"] = i
    return ordered
```

- [ ] **Step 3: Verify and commit**

Run: `cd scripts/lql_matrix && python3 -m pytest -q` — expected PASS.

```bash
git add scripts/lql_matrix/sequence.py scripts/lql_matrix/sequence_test.py
git commit -m "harness: dependency ordering with deterministic permutation

A consumer run before its producer measures nothing about the consumer.
Independent cells are permuted by seed rather than frozen in file order,
deterministically so an ordering-dependent failure is re-runnable. A
cell whose needs are never produced still runs — nothing is skipped."
```

---

### Task 9: Sequence the corpus in `run_matrix.py` and record the order

**Files:**
- Modify: `scripts/lql_matrix/run_matrix.py`, `scripts/lql_matrix/run_matrix_test.py`

**Interfaces:**
- Consumes: `sequence.sequence(cells, seed)` (Task 8).
- Produces: `run_matrix.py` accepts `--order-seed N` (default `0`); rows gain `"order_index"` and the meta row gains `"order_seed"`.

- [ ] **Step 1: Write the failing test**

Append to `scripts/lql_matrix/run_matrix_test.py`:

```python
def test_producer_runs_before_consumer_and_the_order_is_recorded(tmp_path):
    out = tmp_path / "r.jsonl"
    corpus = tmp_path / "dep.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in [
        {"id": "consumer", "cat": "c", "argv": ["verify", "{{VINDEX}}"],
         "needs": ["vindex"], "produces": []},
        {"id": "producer", "cat": "p", "argv": ["extract", "m", "-o", "v"],
         "needs": [], "produces": ["vindex"]},
    ]) + "\n", encoding="utf-8")
    env = dict(os.environ, LARQL_BIN=_fake_bin(tmp_path), CELL_TIMEOUT="30")
    subprocess.run([sys.executable, os.path.join(HERE, "run_matrix.py"),
                    "leg1", "/v", str(corpus), str(out),
                    "--driver", "cli", "--order-seed", "1"],
                   env=env, check=True, capture_output=True)
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    meta = [r for r in rows if r.get("type") == "meta"][0]
    cells = [r for r in rows if r.get("type") != "meta"]
    assert meta["order_seed"] == 1
    ids = [c["id"] for c in cells]
    assert ids.index("producer") < ids.index("consumer")
    assert [c["order_index"] for c in cells] == [0, 1]
```

Run: `cd scripts/lql_matrix && python3 -m pytest run_matrix_test.py -q -k order` — expected FAIL, `unrecognized arguments: --order-seed`.

- [ ] **Step 2: Wire the sequencer in**

In `run_matrix.py`: `import sequence` beside `import drivers`; add
`ap.add_argument("--order-seed", type=int, default=0, help="permutation seed
for cells independent under needs/produces")`; put `"order_seed": ns.order_seed`
in the `meta` dict after `"driver"`; and add `"order_index": c.get("order_index")`
to the `row` dict after `"cat"`.

The cell loop currently streams the corpus line by line. Sequencing needs the
whole list first, so read it up front and hand it to the sequencer:

```python
    with open(corpus, encoding="utf-8") as cf:
        cells = sequence.sequence(
            [json.loads(l) for l in cf if l.strip()], ns.order_seed)
    for c in cells:
```

- [ ] **Step 3: Verify and commit**

Run: `cd scripts/lql_matrix && python3 -m pytest -q` — expected PASS.

```bash
git add scripts/lql_matrix/run_matrix.py scripts/lql_matrix/run_matrix_test.py
git commit -m "harness: sequence the corpus by dependencies, record the order

--order-seed selects the permutation of independent cells; the seed is
in the meta row and each cell carries order_index, so a capture can be
tied back to the ordering that produced it."
```

- [ ] **Step 4: Demonstrate that ordering changes outcomes, on a runner**

Add `--order-seed "${ORDER_SEED}"` to the cli invocations and run the CLI corpus
twice, with `ORDER_SEED: 1` and `ORDER_SEED: 2`, then diff `{id: exit_code}`
between the two downloads. A non-empty diff means ordering is load-bearing and
permutation is earning its cost; an empty diff means these cells are
order-independent for these seeds, which is worth knowing and is not a reason to
remove permutation. Record which you saw; neither answer blocks Task 10.

---

### Task 10: Consolidate the workflow onto the declared driver list

Tasks 3, 4, 6, 7 and 9 each wired their piece in and demonstrated it, so the
workflow drives a **hardcoded** driver list. This replaces it with
`matrix.leg.drivers` from Task 5, so a leg runs exactly the drivers it declares
and the long-session and cli legs take effect. Nothing here is unproven — every
driver it references has already produced captures on a runner.

**Files:**
- Modify: `.github/workflows/lql-strategy-matrix.yml`

**Interfaces:**
- Consumes: `matrix.leg.drivers`, `matrix.leg.long_session` (Task 5); `run_matrix.sh … --driver … --order-seed …` (Tasks 3, 9); `cli-help.jsonl`, `cli-commands.jsonl` (Tasks 6, 7).
- Produces: captures for every declared driver, uploaded under `results-<leg>`.

- [ ] **Step 1: Replace the hardcoded driver list with the declared one**

Replace the step named `Run LQL command corpus (${{ matrix.leg.name }})` in the
`matrix` job with:

```yaml
      - name: Run corpora per driver (${{ matrix.leg.name }})
        run: |
          set -uo pipefail
          VINDEX="out/${NAME}.vindex"
          for DRV in $DRIVERS; do
            case "$DRV" in
              cli) CORPORA="cli-help cli-commands" ;;
              *)   CORPORA="commands" ;;
            esac
            for CORPUS in $CORPORA; do
              echo "=== ${NAME} / ${DRV} / ${CORPUS} ==="
              LARQL_BIN=./bin/larql MODEL_ID="${CORPUS_MODEL}" \
              TMPROOT="$(mktemp -d)" CELL_TIMEOUT=900 \
              scripts/lql_matrix/run_matrix.sh \
                "${NAME}" "$VINDEX" "scripts/lql_matrix/${CORPUS}.jsonl" \
                "out/results-${NAME}.${DRV}.${CORPUS}.jsonl" \
                --driver "$DRV" --order-seed "${ORDER_SEED}"
            done
          done
```

- [ ] **Step 2: Add the driver and seed env vars**

In the `matrix` job's `env:` block, after `LQL_WITH:`:

```yaml
      DRIVERS: ${{ join(matrix.leg.drivers, ' ') }}
      ORDER_SEED: ${{ github.run_number }}
```

`ORDER_SEED` is the run number so successive runs explore different orderings
while any single run stays reproducible from the seed in its meta row.

- [ ] **Step 3: Verify the workflow, and that no suppression came in**

The `Upload leg results` glob is already `out/results-${{ matrix.leg.name
}}.*.jsonl` (widened in Task 3), so it covers the new names unchanged.

```bash
cd /home/metavacua/larql-vindex3-03-08-2026
python3 -c "import yaml; d=yaml.safe_load(open('.github/workflows/lql-strategy-matrix.yml')); print('YAML OK, jobs:', list(d['jobs'].keys()))"
grep -n 'DRIVERS:\|ORDER_SEED:\|--driver\|--order-seed' .github/workflows/lql-strategy-matrix.yml
grep -n '2>/dev/null\|>/dev/null\||| true' .github/workflows/lql-strategy-matrix.yml | grep -v ':\s*#'
```

Expected: `YAML OK`; `DRIVERS`, `ORDER_SEED`, `--driver` and `--order-seed`
present; and **no output** from the third grep. Any hit there is a Global
Constraint violation and must be removed before committing.

- [ ] **Step 4: Commit**

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

**Spec coverage.** Every section of `2026-08-07-cli-and-repl-coverage-design.md` maps to a task: principles → Global Constraints (and asserted by `test_row_carries_no_derived_opinion`, Task 3); four drivers → Tasks 2–4; session granularity → Task 5 (`long_session` legs); capture → Tasks 3–4; the three corpora → Tasks 1, 6, 7; WIP surfaces → Task 7 (`build`, `show` cells reach them); sequencing and permutation → Tasks 8–9; termination and hangs → Task 4 (`exit` as the last pty line, plus the pty deadline) and the existing `timeout --kill-after=10` for the other drivers; scale → no task needed, it is an accepted consequence.

**Deliberately deferred:** the spec's `.merged` capture is produced only for `repl-pty` (Task 4), which is the only driver that has one — `repl-pipe` keeps separate `.out`/`.err`, as the spec's capture table describes.

**Where grammars are owned.** No Python re-implements a grammar anything else already owns. The LQL corpus is checked by LQL's own parser (`crates/larql-lql/tests/matrix_corpus_wellformed.rs`, Task 1); the CLI corpora are checked by clap's own `Cli::command()` (`crates/larql-cli/src/matrix_cli_corpus.rs`, Tasks 6–7); statement boundaries are authored data with no splitter anywhere in the harness (Task 2). The `corpus_lint.py` this plan originally specified does not exist and must not be written.

**Type consistency.** `drivers.build(driver, cell, larql) -> (argv, stdin_bytes)` is defined in Task 2 and called in Tasks 3–4; `drivers.pty_lines(cell) -> list[str]` is added in Task 4 and called only from the pty branch. `sequence.sequence(cells, seed) -> list[dict]` is defined in Task 8 and called in Task 9. Row keys `driver`, `merged`/`merged_bytes` and `order_index` are introduced in Tasks 3, 4 and 9 and read only after.

**Demonstration coverage.** Every task that changes runtime behaviour ends on a real runner against the real binary, naming what to read and what each outcome means: Task 0 (does the REPL accept piped stdin), Task 3 (the `lql` driver is unchanged), Task 4 (both REPL drivers produce captures, and `STATS;` is no longer lost), Task 6 (38 real `--help` invocations), Task 7 (38 real invocations, each non-zero bucketed), Task 9 (does ordering change outcomes). Tasks 1, 2, 5 and 8 are corpus data, leg data or pure functions with no runtime surface.

**Ordering by failure probability.** The three riskiest items — 38 guessed CLI argument shapes, `larql repl` on non-tty stdin, and a `pty.fork` read loop — were all attempted in Task 0, which wrote no harness code. Task 7 writes its corpus from Task 0's captures rather than guessing again, and clap now rejects a wrong shape locally, so the six rows Task 0 could not settle cost a test run rather than a CI leg. Task 4 lifts a read loop already proven on a runner.

**Exhaustive coverage of the finite sets.** `drivers.build` is enumerated over all 4 drivers × 2 cell shapes — four acceptances and four rejections, Task 2. `sequence.sequence` is enumerated over all eight graph shapes including self-cycle and mutual cycle, with a termination guard, in one parametrized test, Task 8. All 38 subcommands appear in Tasks 0, 6 and 7, and clap enforces that the set is complete rather than a hand-maintained list. Cell *orderings* are factorial and therefore sampled by seed rather than exhausted — Task 9 declares that sampling rather than presenting it as coverage.

**Assumption Task 0 settled.** Tasks 2–4 assumed `larql repl` reads piped stdin; Task 0 measured that it does. What Task 0 also measured — that a terminal loses statements written ahead of the prompt — is why Task 4 passes the statement list through and paces it against the prompt rather than writing one static payload.
