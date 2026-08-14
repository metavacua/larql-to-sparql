# CLI and REPL coverage in the LQL strategy matrix

**Date:** 2026-08-07
**Status:** design, approved for planning
**Branch:** `xml-groundwork/fidelity-measure`
**Applies to:** `.github/workflows/lql-strategy-matrix.yml`, `scripts/lql_matrix/`

## Purpose

Run larql's and LQL's command surfaces on GitHub-hosted runners and report what
happens. Today the matrix drives **4 of 38** CLI subcommands (`extract`,
`convert gguf-to-vindex`, `convert quantize`, `lql`) and reaches the REPL not at
all. This design extends coverage to the whole CLI surface and to the REPL as a
first-class surface.

The harness provokes and exposes breakage. It does not prevent, divert, or
soften it.

## Principles

These are load-bearing, not preamble. Each has been violated in this harness and
each violation produced a wrong conclusion.

1. **The harness reports; it does not judge.** No pass/fail, no conformance
   oracle, no bucket, no error-signal, no first-error line. Verdicts about the
   data are formed outside the workflow by a human.
2. **No failure is converted into a plausible success value.** No `2>/dev/null`,
   no `|| true`, no substituting `{}` or `""` for a crash. A failure is recorded
   as a failure, with its stderr.
3. **No post-processing of output.** Captures are verbatim. Splitting, head
   truncation and first-error extraction are all forbidden — each destroyed
   attribution the last time it was used.
4. **No redundancy assumptions.** A CLI subcommand is never treated as covered
   by a same-named LQL statement, or vice versa. Both run, always.
5. **A malformed probe is a harness defect, not a finding.** The corpus sends
   valid input; when it does not, that is fixed, not rationalised.

### Why principle 4 is not theoretical

Two same-named pairs have already been measured as different operations:

- `larql extract --level` produces `dtype=f16`; LQL `EXTRACT MODEL … WITH …`
  produces `dtype=f32` (`extract.rs:55` passes `StorageDtype::F32` as a literal,
  and the LQL grammar has no dtype syntax at all).
- `larql compile --base` requires the base checkpoint and treats the vindex as
  optional; LQL `COMPILE … INTO MODEL` loads from the vindex's own split files.

Five further pairs collide by name and have **separate implementations** —
`query_cmd::run`, `describe_cmd::run`, `stats_cmd::run`, `merge_cmd::run`,
`filter_cmd::run`, none of which dispatch to `larql_lql`. Their doc comments say
"graph file", so they may not even operate on the same artifact as their LQL
namesakes.

## Drivers

One shared LQL corpus, invoked four ways. The statement list is identical across
the three LQL drivers, so a divergence between them is observable rather than
inferred.

| driver | invocation | surface |
|---|---|---|
| `lql` | `larql lql "<statements>"` | one-shot batch (exists today) |
| `repl-pipe` | `printf '%s\n' … \| larql repl` | non-interactive / scripted REPL |
| `repl-pty` | `larql repl` under a pseudo-terminal, statements written to it | interactive REPL as a user meets it |
| `cli` | each subcommand invoked directly | the CLI surface, its own corpus |

**How statements reach the pty is left to the plan**, but the requirement is
explicit: `larql repl` must see a tty on stdin, and the statements must arrive
as if typed. `script -q -c 'larql repl' <typescript>` with statements on its
stdin is the expected mechanism (`util-linux`, preinstalled on ubuntu runners);
a small `pty`-module driver is an acceptable alternative. Whichever is used, the
captured typescript will contain terminal control sequences and `\r` — these are
kept verbatim and not stripped, because stripping is post-processing and because
what the terminal actually received is part of what happened.

## Session granularity

**Backbone: one REPL session per corpus cell.** Open a session, send that cell's
statements in order, close it. Cells stay isolated from one another, as they are
today.

**Plus: one long-session leg per model.** The entire corpus through a single
session. This is the only thing that exercises state accumulating across
commands, and nothing tests it today. Ordering is load-bearing here and cells
contaminate each other by design — that is the point of the leg, and it is why
it is separate from the backbone rather than replacing it.

## Capture

Verbatim. No splitting, no truncation, no derived fields.

Per session or invocation:

- `.out` — complete stdout
- `.err` — complete stderr
- `.merged` — for `repl-pty`, the single stream a terminal produces

The `larql> ` prompt is printed before every read, so statement boundaries are
**visible in the capture**. The harness does not parse them. Attribution is the
reader's job, performed on complete data.

The JSONL row is an index, not a description: driver, cell id, statements sent,
exit code, duration, peak RSS, byte counts, and the capture paths. Every field
is a fact about the process, none is derived from output contents.

Exit codes are recorded but carry no special weight. `larql lql` exits 0 on
in-band errors and `run_repl` returns nothing at all, so an exit code is one
datum among several and never a verdict.

## Corpora

Three files, all under `scripts/lql_matrix/`.

### 1. `commands.jsonl` (exists) — the LQL corpus

64 cells, run through all three LQL drivers.

**Fix required:** six cells send `BEGIN PATCH;`. `parse_begin` calls
`expect_string()`, so the path is mandatory and these have never once opened a
named patch session. They become `BEGIN PATCH "{{TMP}}/<cell>.vlp";`.

### 2. `cli-help.jsonl` (new) — every subcommand's help

One cell per subcommand: `larql <cmd> --help`. 38 cells that always run,
independent of any vindex. Catches clap misconfiguration and drift between the
declared and actual surface.

### 3. `cli-commands.jsonl` (new) — real invocations

One or more real invocations per subcommand — one minimum, more where a
subcommand has distinct modes (e.g. `convert` has `gguf-to-vindex` and
`quantize`; `slice` has several slice kinds). Every subcommand appears at least
once; none appears zero times.

**No pre-filtering by guessed applicability.** Subcommands needing a server,
network, or a graph file rather than a vindex still run; their failure is the
finding. This explicitly includes `Bench`, `DecBench`, `K3Ledger`, `Accuracy`,
`Shannon`, `Serve`, `Publish`, `Pull`, `Hf`, `Dev`, `Recipe`, `Card`,
`MoeLocality`, `Parity`, `Diag`.

`Repl` and `Lql` appear here too, invoked as plain subcommands. They are also
their own drivers above; that is duplication on purpose, since "the driver
covers it" is a redundancy assumption.

Long-running subcommands (`Serve`, `Chat`, and `Run` with no prompt) are given
input that terminates them, or run under the per-cell timeout and are recorded
as timeouts. Neither is skipped.

The full subcommand list (38; `ExtractIndex` is documented as an alias for
`Extract` and is run anyway, because "identical behavior" is a claim in a doc
comment, not a measurement):

```
Run  Chat  Pull  Model  Link  List  Show  Slice  Publish  Rm  Bench  DecBench
K3Ledger  Accuracy  Shannon  Serve  Repl  Lql  Extract  ExtractIndex  Build
Compile  Convert  Hf  Verify  Diag  Parity  MoeLocality  Recipe  Capabilities
Card  Query  Describe  Stats  Validate  Merge  Filter  Dev
```

### WIP surfaces

Three known work-in-progress paths that the matrix reaches today only by
accident, and which the CLI corpus reaches deliberately:

- `larql build --compile <fmt>` → `"(compile not yet implemented — built vindex saved at …)"` (`build_cmd.rs:102`)
- `larql show` on a programme the binary lacks → `"[programme not implemented by this binary]"` (`show_cmd.rs:105`)
- `vector-extract` components → `"{component}: skipped (not yet implemented)"` (`vector_extract_cmd.rs:152`)

These are recorded, not skipped. A WIP surface that says so is a useful
observation; a WIP surface the harness routes around is invisible.

## Sequencing and permutation

Cell count is not a constraint; each cell runs in a couple of minutes and they
parallelise. What matters is that cells are **ordered by their real
dependencies** and that independent orderings are **permuted** rather than fixed.

### Dependencies are a DAG, not a list

The CLI corpus has genuine producer → consumer → destroyer edges. A consumer run
before its producer measures nothing about the consumer:

- **producers:** `extract`, `extract-index`, `convert`, `build`, `slice`,
  `compile` — each yields an artifact others read.
- **registrars:** `link`, `pull` — put an artifact in the cache, which is what
  `list`, `show`, `run`, `rm` resolve against.
- **consumers:** `verify`, `show`, `list`, `run`, `diag`, `capabilities`,
  `parity`, `bench`, `serve`, `publish`, `hf`, `card`, `slice`, `shannon`,
  `accuracy` — need an artifact, and for the cache-resolving ones, a
  registration.
- **destroyer:** `rm` — invalidates what the consumers resolve, so it is ordered
  last within its chain rather than excluded.
- **graph-file commands:** `query`, `describe`, `stats`, `validate`, `merge`,
  `filter` — depend on a graph file, not a vindex, and form their own chain.

Each cell declares what it needs and what it produces. The runner sequences
from that declaration; it does not rely on file order in the corpus.

### Permutation

Where two cells are independent under the DAG, their relative order is a free
variable, and free variables get varied rather than frozen:

- Permute independent consumers against each other within a leg.
- Permute the position of `rm` and other destructive/registrational commands
  among the consumers, so "consumer after `rm`" is exercised as well as before.
- The long-session REPL leg (whole corpus, one session) is itself a permutation
  axis: state accumulates, so ordering is load-bearing there by construction.

A permutation is identified in the row (a seed or an explicit order id) so a
capture can be tied back to the order that produced it. The permutation set is
declared and reproducible, not randomised per run — an ordering-dependent
failure has to be re-runnable.

Order is **not** used to decide correctness. A cell that fails because its
producer failed is recorded exactly as it happened; nothing is skipped for
having an unsatisfied dependency, because "this consumer breaks when its input
is missing" is itself an observation worth capturing.

## Termination and hangs

- **pipe:** stdin closes; `run_repl` breaks on `ReadlineError::Eof`.
- **pty:** a terminal has no EOF, so the statement list ends with `exit`.
- Both run under the existing per-cell `timeout --kill-after=10`. A hang is
  recorded as a timeout with its partial capture retained, never silently
  retried or dropped.

## Known unknown, deliberately not designed around

**Whether piping into `larql repl` executes anything.** `run_repl` constructs a
`rustyline::DefaultEditor` and only falls back to the stdin-native
`run_repl_basic` when that constructor *fails*. Behaviour on non-tty stdin is
rustyline's, and I have not verified it.

This is not resolved by reasoning. The `repl-pipe` leg runs and the capture
shows what happens — statements executed, nothing read, or a hang. If piping
does not work, that is a finding about larql's non-interactive story, not a
harness problem to route around with a PTY.

Running `repl-pipe` and `repl-pty` as separate legs is what makes the difference
visible. A single leg that silently read nothing would produce clean empty
captures indistinguishable from "ran fine, no errors."

## Scale

Per leg: 38 help cells + ~38 invocation cells + 64 LQL cells × 3 drivers ≈ **270
cells**, before permutation multiplies the independent orderings. Against 64
today.

**Cell count is explicitly not a constraint.** Each cell runs in a couple of
minutes and cells parallelise across the matrix. The binding requirements are
dependency ordering and permutation coverage, not economy. A design that reduced
cell count by dropping permutations or pruning "probably redundant" subcommands
would be optimising the wrong quantity.

## Out of scope

- Any verdict, score, threshold, or pass/fail gate.
- Fixing the larql and LQL defects the matrix surfaces. The harness reports
  them; repair is separate work against separate issues.
- Per-statement splitting of REPL captures.
- The `lql-matrix-smoke.yml` workflow.

## Changes required

**`scripts/lql_matrix/`**
- `commands.jsonl` — fix the six `BEGIN PATCH;` cells.
- `cli-help.jsonl`, `cli-commands.jsonl` — new corpora.
- `run_matrix.py` — accept a driver parameter (`lql` | `repl-pipe` | `repl-pty`
  | `cli`); emit `.merged` for the pty driver; keep rows as indexes.
- `gen_legs.py` — add the driver axis and the long-session legs.

**`.github/workflows/lql-strategy-matrix.yml`**
- Drive the corpora per driver; upload every capture.
- No new suppression: no `2>/dev/null`, no `|| true`, no fabricated defaults.
