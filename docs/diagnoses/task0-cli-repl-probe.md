# Task 0 probe — what the CLI and REPL actually did

**Run:** `lql-strategy-matrix` #31246014726, job `probe`, ubuntu-latest
**Commit:** c9601ad4 · **Artifact:** `probe` (163 files)
**Plan:** `docs/superpowers/plans/2026-08-07-cli-and-repl-coverage.md`, Task 0

This records what the captures show. It does not grade anything. Every claim
below names the file it came from, so it can be checked against the artifact
rather than against this summary.

**This run's artifact was uploaded at `retention-days: 1` and is gone after
2026-08-09.** Retention is now 14 days for later runs; to regenerate this one,
dispatch `lql-strategy-matrix` with `scope: probe`.

## The known unknown is resolved: piped stdin works

The design flagged one thing it refused to settle by reasoning — whether
`larql repl` executes anything on non-tty stdin, given `run_repl` builds a
`rustyline::DefaultEditor` and only falls back to the stdin-native
`run_repl_basic` when that constructor *fails*.

It executes. `repl-pipe.out` has the banner, `SHOW MODELS;` output, and
`Goodbye.`; `repl-pipe.err` has `Error: No backend loaded. Run USE
"path.vindex" first.` from `STATS;`. Both statements ran, in order, and
`larql_exit=0`.

The capture does *not* establish that `exit` was read: `run_repl` prints
`Goodbye.` on `ReadlineError::Eof` as well as on `exit`, and stdin closed at
the same moment, so the two are indistinguishable here. What is established is
that piped statements execute — which is the question the design asked.

## The three drivers disagree, on identical input

Same three statements — `SHOW MODELS;`, `STATS;`, `exit` — to three drivers:

| driver | SHOW MODELS | STATS | exit | outcome |
|---|---|---|---|---|
| `repl-pipe` | ran | ran (error) | indistinguishable from EOF | `larql_exit=0` |
| `repl-script` (pty via `script -q -e`) | ran | **absent** | ran | `script_exit=0` |
| `repl-pty` (`probe_pty.py`) | ran | **absent** | **absent** | `TIMEOUT`, `exit=-9` |

`repl-script.out` shows the tty echoing all three lines up front, then one
`larql> ` prompt taking `SHOW MODELS;`, then a second prompt that receives
nothing visible and goes straight to `Goodbye.` — `STATS;` produced no output
and no error. `repl-pty.merged` ends mid-prompt after the `SHOW MODELS;` table,
and the driver hit its 60s deadline.

So under a terminal, statements written ahead of the prompt are lost, and how
many are lost depends on which pty mechanism wrote them. Whether that is
rustyline's bracketed-paste handling (`^[[?2004h` / `^[[?2004l` bracket every
prompt in the capture), the line discipline, or larql, this probe does not say.

Two consequences for the plan:

- Running `repl-pipe` and `repl-pty` as separate legs was load-bearing. A
  single pty leg would have produced a clean capture missing a statement, with
  nothing to compare it against.
- The design called a `pty`-module driver "an acceptable alternative" to
  `script(1)`. They are not interchangeable — they lost different amounts.
  Task 4 has to write statements *in response to* the prompt rather than
  up front, and both mechanisms need re-measuring after that change.

## `larql lql` exits 0 on an in-band error

`lql.out` ends with `Error: No backend loaded. Run USE "path.vindex" first.`
and `lql.err` records `exit=0`. Corroborates what the harness already assumed;
recorded here because it is now measured on this binary rather than inherited.

## All 38 `--help` calls exit 0

`help.index`: 38 of 38 at `exit=0`. No clap misconfiguration, no drift between
the declared surface and the built binary.

## 10 of 38 real invocations had the wrong argv

`--help` is what corrects them, so Task 7 writes the corpus from this table
rather than guessing a second time. `exit=2` is clap rejecting the shape;
`exit=1` is the command parsing and then failing on missing input, which is the
intended outcome in this job and not an error to fix.

| cmd | guessed | clap said | correct shape from `--help` |
|---|---|---|---|
| `slice` | `--kind browse` | unexpected `--kind` | `slice [OPTIONS] --output <OUT> <SOURCE>`; use `--parts`/`--preset` |
| `run` | `--prompt hi` | unexpected `--prompt` | `run [OPTIONS] <MODEL> [PROMPT]` — prompt is positional |
| `query` | `--entity France` | unexpected `--entity` | `query --graph <GRAPH> <SUBJECT> [RELATION]` |
| `describe` | `graph.json France` | unexpected `France` | `describe --graph <GRAPH> <ENTITY>` |
| `shannon` | `shannon score <v>` | missing required args | `shannon <COMMAND>` — enumerate subcommands |
| `card` | `card render <f>` | unexpected `<f>` | `card <COMMAND>` |
| `hf` | `hf upload …` | unrecognized `upload` | enumerate `hf` subcommands |
| `k3-ledger` | `k3-ledger report` | unrecognized `report` | enumerate subcommands |
| `dec-bench` | bare | printed help | needs a subcommand/args |
| `dev` | bare | printed help | needs a subcommand |

`query` and `describe` both take `--graph <GRAPH>` documented as "Path to graph
file (.larql.json or .larql.bin)". That is direct support for the design's
principle-4 claim: these are not the LQL statements of the same name, and they
do not operate on a vindex.

## HuggingFace stayed out of it

`HF_ENDPOINT=http://127.0.0.1:1` held. No capture mentions `huggingface.co`.
`cmd.model.err` and `cmd.publish.err` both end in `request error: io:
Connection refused`, and `cmd.pull.err` fails to fetch `index.json`. Each row
still shows that clap accepted the argv and how far `run()` got, which is what
was wanted from them.

The recorded escape did not fire: `k3_ledger/fetch.rs:36` hardcodes
`https://huggingface.co` and ignores `HF_ENDPOINT`, but `k3-ledger report` died
at clap first (`unrecognized subcommand 'report'`), so no request was made. The
escape remains live for whatever the real subcommand turns out to be.

## Incidental

- `cmd.publish.err`: `failed to download index.json from
  hf:///tmp/tmp.…/v.vindex` — a local path was turned into an `hf://` ref.
- `cmd.repl` exited 0 immediately rather than timing out — **with stdin closed**,
  which is the condition in that step, not a general property. Given a tty it
  plainly does not self-terminate: the `repl-pty` leg in this same run hit its
  60s deadline. `serve`, `chat` and `run` are still unanswered on this point;
  they exited 1 on the missing vindex before reaching the question.

## Harness defects this run had to fix first

Both were in the plan text and would have destroyed the run:

- The job's YAML could not parse — the 38-row heredoc body and its terminator
  sat at column 0, which ends a YAML block scalar. Now a bash array.
- GitHub's default `run:` shell is `bash -eo pipefail` and `set -uo pipefail`
  does not clear `-e`, so the 38-command loop would have aborted at the first
  non-zero exit with every later capture lost. Each capture step now clears
  `-e` and records the code. All 38 rows are present in `cmd.index`, which is
  the evidence the fix worked.

Piped invocations record both `PIPESTATUS` entries separately, so a `printf`
that takes SIGPIPE cannot be read as larql dying on a signal. In this run
`printf_exit=0` throughout, so none did — but the field is what makes that
statement checkable instead of assumed.
