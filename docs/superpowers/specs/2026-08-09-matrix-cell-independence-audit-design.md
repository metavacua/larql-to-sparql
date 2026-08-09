# Matrix cell independence audit

**Date:** 2026-08-09
**Status:** design, approved for planning
**Branch:** `xml-groundwork/fidelity-measure`
**Applies to:** `.github/workflows/lql-strategy-matrix.yml`, `scripts/lql_matrix/`

## Purpose

The strategy matrix's harness has been debugged one command at a time this
session — `BEGIN PATCH`, `MERGE`, the `[ -s ]` exit-status bug, stdin
inheritance, the `[skip ci]` marker overriding a dispatch — each found by
manual, forensic, one-command-at-a-time investigation. Every one of them was a
**harness** defect, not a LARQL defect: the instrument was lying about what it
tested, not reporting a true finding.

That method doesn't scale to the full command surface, and it can't rule out
that more of the same class of bug is sitting undiscovered in the other 50-odd
commands nobody has looked at yet. This spec replaces one-at-a-time manual
audits with **57 independent, parallel, per-command audits**, each isolated
from the others, run through the same mechanism this session already trusts:
real GitHub-hosted runner execution, not source-reading, not local inference.

**The goal is not to fix LARQL.** The goal is to get the CI to a point where
its reports about LARQL can be trusted at all — for every command, not just
the handful anyone happened to look at by hand.

## The independence hypothesis

Each of the 57 units is dispatched **blind to what the others are doing**: no
shared list of known findings, no briefing on which commands are suspected
broken. Every agent starts from the same premise — "your command might be
fine, might be lying, you don't know yet, go find out."

If several units, working independently, converge on the same file, the same
function, or the same root cause — that convergence **is the result**. It
means the bug is systemic, the way `BEGIN PATCH`'s masking effect wasn't a
`BEGIN PATCH` problem at all, it was a "first error in a batch hides the rest"
problem that could have surfaced from any multi-statement cell. Divergence is
also informative: if two units touch the same file in **incompatible** ways,
that's a real seam in the harness worth knowing about before it's papered over
by a hasty merge.

This is why context is withheld, not out of thoroughness for its own sake. A
pre-digested findings list would collapse exactly the measurement this audit
exists to take.

## Scope boundary — CI/workflow/harness layer only

**In scope, fixable:** `.github/workflows/lql-strategy-matrix.yml`,
`scripts/lql_matrix/run_matrix.py`, `drivers.py`, `gen_legs.py`,
`commands.jsonl`, `commands-model.jsonl`, and any other file under
`scripts/lql_matrix/`.

**Out of scope, report-only:** LARQL's Rust source (`crates/`). If a unit
root-causes its command down to a real Rust-level defect — the way `COMPILE …
INTO MODEL`'s missing `index.json` write was found this session — it **does
not fix it**. It reports the finding, precisely, with the evidence that
established it, and stops there. A real, uncorrected LARQL defect is not a gap
in this audit's coverage — it is the test signal that proves the harness can
correctly detect and report a real problem instead of masking it. Fixing it
would remove exactly the case this audit needs to verify the CI against. This
is not a staged or eventual exclusion: fixing LARQL/LQL is not this audit's
job, on this pass or on any pass.

A unit that finds nothing wrong reports that too, honestly, rather than
manufacturing a finding to justify the dispatch.

## Roster — 57 units

**LQL-side, 20 units.** One agent per top-level verb; each agent is
responsible for exhaustively covering all of that verb's documented sub-forms
as its own matrix cell (fragmenting sub-forms across separate agents would
duplicate setup for no signal):

`USE` · `SHOW` (9 sub-forms) · `DESCRIBE` (8) · `WALK` (3) · `INFER` (5) ·
`SELECT` (8) · `EXPLAIN` (INFER/WALK) · `STATS` · `TRACE` (3) · `DIFF` ·
`INSERT` · `DELETE` · `UPDATE` · `MERGE` · `REBALANCE` (3) · **PATCH
lifecycle** (BEGIN/SAVE/APPLY/REMOVE PATCH grouped — testing APPLY without
BEGIN+SAVE first is meaningless, so one agent owns the whole stateful
sequence) · `COMPACT` (4) · `COMPILE` (INTO VINDEX / INTO MODEL) · `EXTRACT` ·
**negative-rejection coverage** (`NOT A VALID STATEMENT`, `FOOBAR`, `SHOW
FOOBAR` — verifying the grammar durably rejects bad input, and extending the
corpus if it finds a gap).

**CLI-side, 37 units.** One per `clap` subcommand, confirmed against `enum
Commands` in `crates/larql-cli/src/main.rs` (37 variants, not the 38 an
earlier hand-typed list carried):

`Run · Chat · Pull · Model · Link · List · Show · Slice · Publish · Rm · Bench
· DecBench · K3Ledger · Accuracy · Shannon · Serve · Repl · Lql · Extract ·
ExtractIndex · Build · Compile · Convert · Hf · Verify · Diag · Parity ·
MoeLocality · Recipe · Capabilities · Card · Query · Describe · Stats ·
Validate · Merge · Filter · Dev`

Where a subcommand has its own sub-subcommand grammar (`Hf`, `K3Ledger`,
`Recipe`, `Card`, `Convert`), the agent discovers it via `--help` and covers
every sub-mode found — Task 0 already showed `dec-bench` and `dev` refusing to
run bare, and `k3-ledger report` / `hf upload` being rejected as unrecognized
subcommands, meaning their real grammars were never enumerated.

No unit gets a pre-digested "here's what's already known about your command"
briefing — see Independence Hypothesis above.

## Mechanism — GitHub-hosted runners, not the dev machine

Each unit's `git` isolation is a worktree (`isolation: 'worktree'`), branched
from the current head of `xml-groundwork/fidelity-measure`, named
`matrix-audit/<unit-slug>`. The worktree is a **staging area for commits
only** — the actual research, iteration, and verification loop is:

```
edit (if warranted) → push its own branch → gh workflow run
lql-strategy-matrix.yml --ref <its-branch> [-f scope=... / inputs] →
poll → gh api .../artifacts → download → read the real capture →
root-cause → (fix if in-scope) → re-dispatch to verify → commit → report
```

No local `cargo build --release`, no local `larql extract/run/infer`, beyond
a fast `cargo check` for syntax sanity. This is not a style preference — it's
what "prefer GitHub-hosted runners over clobbering the dev machine" requires,
and it also mostly retires the CLAUDE.md wrapped-execution constraint for this
exercise: that policy protects *this host*, and a GitHub-hosted runner isn't
this host. It remains a backstop: if a unit does run `larql run/extract/infer`
locally for some reason, it must go through `larql-probe safe`, no exception.

**No new workflow files.** Every unit works within the existing
`lql-strategy-matrix.yml` and the existing `scripts/lql_matrix/` files,
versioned from the same root, inside its own isolated branch. This is
deliberate: it's what makes convergence/divergence at synthesis time visible
instead of hidden behind 57 non-overlapping files.

## Available tools and skills — need-to-know, not a briefing

The distinction that matters: **infrastructure is shared, findings are not.**
Every unit is told about the tools; none is told what any other unit found.

- **`systematic-debugging` (mandatory).** Root cause before any fix, full
  stop. This is the actual method, not a suggestion.
- **`writing-plans`' discipline, not necessarily the artifact.** Each unit's
  fix (if any) follows the bite-sized TDD rhythm — reproduce the failure via
  a real dispatched run first, confirm it fails for the stated reason,
  implement the minimal fix, confirm the same real-run reproduction now
  passes, commit. A unit whose command turns out to need genuinely multi-step
  work is free to invoke `writing-plans` formally; most won't need to.
- **`requesting-code-review`'s rubric, wired at the workflow-script level.**
  Not self-invoked by each unit — reviewing your own diff via a nested agent
  dispatch is unreliable to depend on and invisible to the controller. It runs
  as a second pipeline stage per unit (below), authored once, applied to every
  unit's diff.
- **`/simplify`'s rubric, wired at the workflow-script level, same reasoning
  as `requesting-code-review`.** `/simplify` is a real command — not
  `code-simplifier`, which is a separate plugin agent scoped to a different
  project path and unavailable here. `/simplify`'s own instructions are
  explicit that it is a *post-fix cleanup pass, not a bug hunt*: four parallel
  reviewers (Reuse, Simplification, Efficiency, Altitude) over a diff, findings
  deduped and applied, explicitly not looking for correctness issues — that's
  `requesting-code-review`'s job. It runs as a third pipeline stage, after the
  fix and after correctness review, over the same diff, and only when there is
  a diff to look at — a unit that changed nothing has nothing to simplify.
- **Existing tools, read as reference, not copied blind:** the
  `compile-evidence` job in `lql-strategy-matrix.yml` (a worked example of
  wiring a new `workflow_dispatch` job correctly — it already demonstrates and
  documents the two YAML footguns this session found: `build`'s `if:` needing
  every new gate scope, and `[skip ci]` silently overriding a dispatch),
  `scripts/lql_matrix/fs-snapshot.sh` (whole-subtree diffing — the tool that
  caught `COMPILE`'s staging-directory behavior after two wrong local
  readings), `crates/larql-lql/tests/ast_dump.rs` (run the real parser, don't
  infer from source), `drivers.py` / `run_matrix.py` (the four-driver
  abstraction), and `crates/larql-lql/tests/matrix_corpus_wellformed.rs` (the
  corpus well-formedness gate).

## Per-unit deliverable

A unit's **report is its Workflow `agent()` return value** — a schema-shaped
object, not a file it has to remember to write to some shared location. Real
commits, if any, live on its own branch and are inspected via `git log`/`git
diff` at synthesis time, not copied anywhere.

Report shape: what was tested, what was found, what (if anything) was fixed
and where, how the fix was verified (must cite a real dispatched run, not
local reasoning), and anything noticed outside this unit's assigned command —
including Rust-source-level findings — reported without being acted on.

## Execution model

A single Workflow script, `pipeline()` over the roster, three stages per unit:

1. **Diagnose + fix** — `haiku`, `isolation: 'worktree'`, the unit's assigned
   command, the constraints above, nothing else. Returns a structured report
   including whether it produced a real diff.
2. **Correctness review** — skipped if stage 1 produced no diff. A second
   agent applies `requesting-code-review`'s rubric, scaled to a single-command
   fix. Findings here do not trigger a fix loop (no budget for that at this
   scale); they're carried into the unit's final report as a caveat.
3. **Simplify pass** — skipped if stage 1 produced no diff. `/simplify`'s own
   four-angle parallel pattern (Reuse / Simplification / Efficiency /
   Altitude) over the same diff, findings deduped and applied directly, same
   as `/simplify` does when run interactively.

`pipeline()`, not `parallel()`-then-barrier, because units are independent
end-to-end — one unit's review stage starting while another is still
diagnosing costs nothing and wastes no time.

**Synthesis is a barrier** at the end, deliberately: it needs every unit's
result at once to detect cross-cutting patterns, which a per-unit stage
cannot see.

## Successive pilots, not one pilot

This exact mechanism — a unit branching, pushing, dispatching a real
`workflow_dispatch` run, reading a real artifact, fixing, re-verifying via
another real run, then a 2-stage cleanup pass — has never been exercised
end-to-end before. Rather than one fixed-size pilot, the roster is ramped in
stages, each validated before the next:

1. **One unit** (`STATS` — the simplest, least stateful command) to prove the
   raw mechanics: worktree creation, push, `gh workflow run --ref`, artifact
   read, commit, structured return, and that the review/simplify stages
   correctly skip when there's nothing to review.
2. **Four units** spanning distinct classes: the stateful PATCH lifecycle, a
   CLI subcommand with its own sub-grammar (`Hf` or `K3Ledger`), `COMPILE`
   (calibration — its harness-layer terrain is already partly known), plus one
   more plain verb. Validates the pattern holds across genuinely different
   shapes of work, not just the easy case.
3. **A larger slice** (roughly a third of the roster) to observe real
   cross-unit convergence/divergence for the first time at nontrivial scale,
   and to catch anything that only shows up under real concurrency.
4. **The full roster**, once 1–3 have each come back clean or with understood,
   non-structural issues.

Each stage's results are read before the next stage is dispatched — this is
not four fire-and-forget batches, it's a gate at every step.

## Synthesis

Once all units land (pilot, then full run), the controller — no further agent
dispatch, this stays bounded:

1. Groups findings by root-cause pattern across units.
2. Explicitly flags every case where independent units touched the same file
   or function — the independence-hypothesis result, whether it shows
   convergence, divergence, or outright conflict.
3. Verifies each unit's claimed fix against its actual branch commits and
   actual dispatched-run evidence — a report's claim is not taken at face
   value any more than any other capture-derived claim has been this session.
4. Produces one ranked document: confirmed harness fixes ready to merge,
   contested/overlapping fixes needing reconciliation, and out-of-scope
   Rust-level findings recorded as evidence that the harness can surface a real
   defect — not as a backlog for this or any other effort to act on.

**No auto-merge.** 57 independently-produced branches get presented, not
merged — the same standing rule this session has followed for every
risky/irreversible action.

## Out of scope

- Fixing LARQL's Rust source. Report-only, not staged, not deferred — a real
  uncorrected LARQL defect is the signal this audit exists to detect and
  verify against, not a queue of work for this or any future CI or LARQL
  effort to clear.
- New `.github/workflows/*.yml` files — everyone works in the existing file.
- Auto-merging any unit's branch.
- A fix-loop / re-review cycle beyond the one review pass per unit — findings
  from review are carried as caveats, not chased to resolution, at this scale.
