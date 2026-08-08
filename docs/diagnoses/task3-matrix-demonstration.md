# Task 3 demonstration — the driver parameter, and two defects it exposed

**Failed run:** `lql-strategy-matrix` #31267330863 — 21/21 legs and
model-lifecycle failed
**Green run:** #31268145895 — 21 legs, model-lifecycle, probe, all green
**Plan:** `docs/superpowers/plans/2026-08-07-cli-and-repl-coverage.md`, Task 3
Step 7

The plan gates Task 4 on proving that adding `--driver` changed nothing for the
existing `lql` path, measured against the real binary. It did not change it —
but getting there took two fixes, both to defects I had introduced earlier in
this session, and neither reachable from the local test suite.

## Defect 1 — a step that fails when nothing is wrong

Commit `cdfcde7a`, "harness: stop converting failures into plausible success
values", replaced `2>/dev/null || echo '{}'` with a real stderr capture and
ended the step with:

```bash
[ -s "$MDIR/listing.err" ] && { echo "listing stderr:"; cat "$MDIR/listing.err"; }
```

GitHub runs `run:` blocks under `bash -e {0}`, and a step's exit status is its
**last** command's. When `listing.err` is empty — that is, when nothing went
wrong — `[ -s ]` returns 1 and the step fails. Every leg died on it.

`-e` is not the mechanism, and it is worth being precise: bash exempts
non-final commands of an `&&` list from `-e`, so the identical guard earlier in
the same script is harmless. It is *being last* that matters.

Demonstrated in a shell rather than argued:

| form | file | exit |
|---|---|---|
| `[ -s f ] && { ...; }` | empty | **1** |
| `if [ -s f ]; then ...; fi` | empty | 0 |
| `if [ -s f ]; then ...; fi` | non-empty | 0, content printed |

Both guards became `if` blocks. Every step in the workflow was then swept for a
last command that can legitimately return non-zero; there are no others.

The irony is the useful part: a commit whose whole purpose was to stop turning
failures into successes turned a success into a failure, in the same edit.

## Defect 2 — a corpus the test could not see

`model-lifecycle` failed separately:

```
TypeError: cell 'use.model': `lql` is a str, expected a list of statements —
a str is iterable, so accepting one would splice the cell into single
characters.
```

`commands-model.jsonl` was never migrated to the statements-as-a-list shape in
Task 2. The guard behaved exactly as designed — it named the cell and the
reason instead of silently producing a screenful of single-character
statements.

What failed was the corpus test. It read `commands.jsonl` **by name**, so it
could only ever check the file its author remembered. It now discovers
`commands*.jsonl` and asserts it found at least two. Verified in both
directions: reverting the model corpus reproduces the runner's error exactly;
restoring it passes.

Both migrations were performed by `larql_lql::split_statements` through a
throwaway test, never by hand.

## Step 7 result

Leg `smol135.native.all` from the green run:

```
cells:            64
drivers:          ['lql']
meta driver:      lql   larql: larql 0.1.0
derived fields:   none
row keys:         cat driver duration_ms exit_code id level peak_rss_kb
                  sent stderr stderr_bytes stdout stdout_bytes
missing captures: none
captured bytes:   110297   (128 files)
```

Captures are named `smol135.native.all.lql.<cell>.{out,err}`. The `lql` path is
unchanged by the driver parameter, so Task 4 is unblocked.

## What Task 1's corpus fixes bought

The seven cells that had never parsed now reach the executor. Their captures,
first-hand:

- **`merge`** — `46066 features merged, 0 skipped (strategy: KeepSource)`. The
  only `MERGE` cell in the corpus, so this is the first time the statement has
  ever executed under this harness.
- **`patch.begin`** — `Patch session started: …/patch.begin.vlp`.
- **`compile.into_vindex`** — `Compiled … → …/.compiled.vindex.tmp.3608`,
  `Features: 46080`, `Size: 263.8 MB`, exit 0.
- **`compile.into_model`** — `Error: Execution error: failed to write model: IO
  error: No such file or directory (os error 2)`.
- **`roundtrip.insert_compile_infer`** — runs through INFER, patch, DESCRIBE
  and COMPILE INTO VINDEX.

So on this binary and this vindex, **`COMPILE … INTO VINDEX` works and
`COMPILE … INTO MODEL` fails**, and the failure is specifically a write to a
path that does not exist. This is the distinction earlier work got wrong three
times in a row by reading derived fields; here it comes from two capture files.

Two further observations, recorded rather than acted on:

- `compile.into_model` reports **`exit_code=0`** while printing an error.
  `larql lql`'s exit code remains unusable as a verdict.
- That error is on **stdout**, not stderr — `.err` is 0 bytes. `run_batch`
  collects `Error: {e}` into its output vector, so in-band errors leave by
  stdout. Anything reading stderr alone would see a clean run.

## Bearing

Three defects in this task's work — the probe's `-e`, stdin inheritance in
`run_matrix.py`, and the `[ -s ]` guard — were all guards that only misfire in
an environment the local suite does not reproduce. Local tests passed before
each one. What caught all three was running the real thing on a runner, which
is what the plan's demonstration steps are for.
