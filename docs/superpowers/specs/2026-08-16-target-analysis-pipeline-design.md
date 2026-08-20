# Generalized target-analysis pipeline — design

Date: 2026-08-16
Branch: `experiment/cuda-nvptx-lint-build` (off `metavacua/larql-to-sparql`)

## Purpose

Generalize the ad hoc `experiment-cuda-nvptx.yml` workflow (built target-by-target,
crate-by-crate, over the course of the nvptx64-nvidia-cuda investigation) into a
uniform, target-parameterized pipeline: given a target triple and a root crate, produce
a complete, sound, valid map of every place gating or fixing would be necessary to
build and runtime-test for that target — success and failure alike, not just failure.

This is a direct continuation of the original `larql-cli-target-sampling` initiative
(`docs/superpowers/specs/2026-08-13-larql-cli-target-sampling-design.md`): that design
established the philosophy (failures are the analysis data; standalone real-CI
experiments over local simulation; documentation is not trusted over observed runner
behavior) for a fixed, hand-built target graph (wasm family + kani). This spec keeps
that philosophy and removes the fixed-graph constraint — the CUDA experiment is the
proof case, not the end goal. Its purpose, stated directly by the person who
commissioned it: force critical interaction with CI workflows and less-common tools,
specifically to surface gaps and inadequacies in how this project researches, designs,
and builds its own analysis tooling, and to use those gaps to drive the tooling's own
development. If an industry-standard tool is ever found that does all of this simply,
the hand-built parts of this pipeline retire in its favor.

## Foundational framework

Two independent-looking stacks turn out to be dependent, not orthogonal, and getting
this wrong caused real, repeated errors during the investigation this spec is built on.

**Hosting layers** (who executes on/in what): the dev machine (persistent, stateful,
shared with the human operator, real and lasting blast radius from anything that runs
on it); the coding agent, running *on* the dev machine as literal processes but
reasoning *about* everything, including things it never directly touches (it has never
once, in this project's history, executed directly on a GitHub-hosted runner — only
mutated the dev machine's git state, pushed, and read results after the fact);
GitHub-hosted runners (ephemeral, a fresh VM per job from a known image, isolated from
and inconsequential to the dev machine, free at this project's tier); GitHub workflows
(version-controlled YAML, judgment-free at runtime, more durable than any single runner
instance since the same file re-executes on a fresh runner every time).

**Semantic levels** (whose claims are epistemically prior): the object language (L0) —
the actual source, manifests, and target capabilities under study; the metalanguage
(L1) — deterministic tool/workflow output that makes claims *about* L0
(compiler diagnostics, `target-spec-json`, JSON build output); agent-interpretation
(L2) — judgment about what L1's output means. Valid reasoning has to proceed
L0 → L1 → L2 — L1 has to be exhaustively gathered before L2 is licensed to form a
hypothesis — never the reverse. This project's own debugging history is the evidence
for why: repeatedly, a hypothesis was formed first (L2), then tooling was reached for
to confirm it (L1), rather than the other way around, and each time this produced a
fix that measurably changed nothing.

These two stacks are **not orthogonal**. L0 has no existence independent of the
hosting stack — it is bytes on the dev machine's disk or a runner's checked-out copy,
and both the agent (hosting layer 2) and the workflow (hosting layer 4) directly
*write* it, not merely describe it (every mutation stage in the Secondary layer below
is exactly this). What *does* hold: hosting layer does not determine semantic level.
A curated ban list living inside a workflow file (`deny-nvptx.toml`'s `[[bans.deny]]`
entries) is still L2 — the judgment of which crates are "real" blockers was made by a human
or agent, not derived mechanically from anything. The test for L1 status is
reproducibility from source alone: would this exact output result regardless of who or
what authored the logic, given the same L0 input? If not, it's L2, wherever it lives,
and has to be labeled as such.

## Standing design principles

Established through repeated, specific correction over the course of this design's
development — not defaults, each one is load-bearing:

1. **No exclusion based on local dev-machine tool availability.** GitHub-hosted
   runners are ephemeral and can install anything; "not installed on the dev machine"
   is informative about the dev machine and irrelevant to CI design. The only real
   constraint is an empirically-verified GitHub Actions platform limit — never assumed
   in either direction, always checked by actually trying it.
2. **No agent-mediated filtering, summarizing, or relevance-judgment inside the
   pipeline.** Full raw deterministic tool output, preserved and retrievable, always.
   An agent's judgment about what's "redundant" is presumptively untrustworthy — this
   was demonstrated directly and repeatedly (treating `cargo tree`'s display as
   sufficient, assuming `-Z avoid-dev-deps` worked because it was silently accepted).
3. **Exhaustive, unconditional fan-out.** Every relevant probe runs every time; nothing
   is skipped because an earlier result seems to already explain things. This governs
   *probe types*, not membership in the target universe itself: real evidence from an
   unscoped run against all 331 real `rustc --print target-list` entries showed 212 of
   them (64%) are rustc's own tier 3 ("community-maintained, may or may not build"),
   most lacking `host_tools` — building `larql-cli` against these is near-certain to
   fail before revealing anything specific to larql-cli's own no_std-readiness, at real,
   recurring cost given the Secondary layer's many-round recursive design. The default,
   `push`-triggered target universe is therefore scoped to `tier <= 2` (rustc's own
   "guaranteed to build" classification, sourced directly from `target-spec-json`'s
   `metadata.tier` field) — this is still principle 3 in spirit, not a violation of it:
   the scoping rule is mechanically grounded in rustc's own emitted classification, not
   an agent's judgment about what "seems relevant" (principle 2). A `workflow_dispatch`
   request for one specific target always bypasses this default scope, including for a
   tier-3 target — an explicit request is real intent, never silently narrowed.
4. **Distinct, unmerged per-probe verdicts.** Disagreement between two independently
   mechanically-grounded sources is the valuable signal (this is how the
   `cargo tree` vs. `cargo build --unit-graph` gap was eventually found), so no
   aggregation step is allowed to collapse one source's claim into another's.
5. **The nvptx64-nvidia-cuda target is a standing canary.** Since it's known with
   certainty to be permanently incapable of full success (`std: false`,
   `only-cdylib: true`, confirmed via `rustc --print target-spec-json`), any probe
   reporting an unexpected clean/success signal against it is presumptively a bug in
   that probe, not progress.
6. **Fix-attempts (Secondary layer) serve two purposes, not one:** validating that the
   Primary layer's instrumentation actually catches what a mutation changes, and
   deliberately excavating issues currently hidden behind today's outermost blocker —
   tracked as depth, since a crate can have multiple independent, stacked problems that
   only surface once something clears the way.
7. **Over-collection is correctable; under-collection is not.** An agent can prune a
   pile of data competently; it cannot reliably notice an absence it has no evidence
   pointing to. The only sound default is maximal collection.
8. **Mechanical sequencing is required wherever a real prerequisite exists, and is
   distinct from judgment-based skipping.** `needs:`/ordered steps express genuine
   dependencies (Stage B needs Stage A's file changes; a known dependency-graph cycle
   involving `larql-inference`/`larql-kv`/a third crate, discovered via their tests,
   has to be detected and reported, not silently survived). This is never license to
   skip a probe because it seems unnecessary.
9. **Both the discovery step and the indexing/aggregation step must be fully
   autonomous.** Real `run:` steps executed by the runner itself, producing live
   results on every run, with zero human or agent intervention beyond the initial
   trigger. Interactive research (by a human or an agent, outside the workflow)
   validates that a mechanism is real; it is never itself part of the deliverable.

## Architecture

Two layers, explicit priority order:

- **Primary layer — observability.** Everything that captures ground truth about a
  target and a build attempt, success or failure, without modifying source. This is
  the actual deliverable: "change the target parameter, get the map." Runs
  unconditionally, to completion, every time.
- **Secondary layer — fix-attempts.** The existing `nostd-fix-attempt`-style pipeline
  (mechanical rewrite, no_std scaffold, dependency-feature patch, workspace-member
  isolation), generalized. Instrumental, not terminal: it exists to keep producing
  before/after contrasts that stress-test the Primary layer's own completeness, and to
  get far enough past today's blockers to reveal what's currently hidden behind them.
  Runs after a full Primary-layer baseline exists for the same target.

## Components

**Discovery job** (autonomous, runs first, feeds everything downstream via
`fromJSON()`-driven dynamic matrices): queries `rustc --print target-list` (or a
specific target if one was given as input); for the default (no explicit target
requested) path, additionally queries `rustc --print target-spec-json`'s
`metadata.tier` for every one of those targets and filters to `tier <= 2` before
anything downstream sees the list (Standing Principle 3) — a `workflow_dispatch`
request for one specific target always skips this filter. Also queries crates.io's
real search API and GitHub's own dependency-graph/SBOM API for ecosystem discovery,
`rustup target list`/`component list` for toolchain availability — all as real `run:`
steps on the runner, every run, not pre-computed and frozen into the workflow file. A
target-family registry (e.g. `os: cuda` → CUDA toolkit tooling) supplements this as
one more queried, explicitly-labeled-as-curated (L2) source, never the sole source of
truth.

**Target-capability probes** (per target, crate-independent — one job per target
regardless of how many crates get tested against it): `target-spec-json` (full
structured JSON — `std`, `only-cdylib`, `panic-strategy`, `tier`, `os`, `arch`),
`--print cfg --target T`, `--print target-list | grep -qx T` as a sanity check, and
`--print supported-crate-types --target T` — the real, per-target, empirically
authoritative crate-type list, discovered while writing this spec: it does *not*
match what `only-cdylib: true`'s name suggests. For `nvptx64-nvidia-cuda` it reports
`bin, cdylib, lib, rlib, staticlib` — not "only cdylib." No field-name inference is
trusted over this print option again; the build-attempt probes below use its output
directly, per target, rather than any crate-type list decided in advance.

**Dependency-graph probes** (per crate × target × feature-config): `cargo metadata
--filter-platform`, `cargo tree` (`-e features`, `-i`, `--duplicates`) *and*
`cargo build --unit-graph` run together, deliberately not one replacing the other, so
their disagreement stays visible; a cycle-detection sub-probe walking the metadata
resolve graph including dev-dependency edges, reporting every cycle by crate name; the
native-link reachability scan (crates with a real `links` field, verified against
actual `build.rs` source, not assumed from the field's presence alone — `rayon-core`
and `prettyplease` are confirmed false positives, `openssl-sys`/`protobuf-src`/
`onig_sys`/`ring` confirmed real, all by reading source, not guessing); `cargo-deny`
checks against a target-specific config (`deny-nvptx.toml`), explicitly labeled as
curated/L2, checkable against the raw scan.

**Build-attempt probes**: every plausible `-Zbuild-std` mode (`none`/prebuilt, `std`,
`core,alloc`, `core`) × `{check, clippy, build}` × `{default-features,
--no-default-features}` × every crate-type the target-capability probe's
`--print supported-crate-types --target T` actually listed for that target — not a
crate-type list chosen in advance from reading a field name, and not narrowed to
whichever subset seems relevant. All attempted unconditionally regardless of what
`target-spec-json` claims about `std` either, for the same reason: the actual
attempt's result is real L1 data even where the outcome is predictable. All modes ×
configs × crate-types, `--keep-going`, full uncapped `--message-format=json` to an
artifact (never truncated
to fit a step-summary size limit — that's what silently dropped ~5MB of real Stage C
output once already). Clippy's lint set is comprehensive (`clippy::all`, `pedantic`,
`nursery`, `cargo`, rustc's own full set), not a hand-picked subset.

**Runtime-test probes**: where a runner exists (wasmtime for the wasm family, native
execution for host targets), actual test execution. Where none exists — nvptx
currently, no free GPU CI anywhere for this project — explicitly recorded as
`"blocked: no runner available, reason: <cited>"`, never silently omitted.

**Secondary-layer stages**, sequenced by real dependency, not linear default order:
Stage A (mechanical `std`→`core`/`alloc` rewrite via `clippy --fix`, host target — has
to run first, rustc cannot resolve `std::` paths once the sysroot is core+alloc-only,
so the fixer needs real `std` present to have anything to suggest against) must
precede Stage B (`#![no_std]` + `extern crate alloc;` scaffold injection — inserted
after any existing leading `//!`/`#![` block, never before it, since a real item like
`extern crate alloc;` closes the module's inner-attribute preamble and any `//!` after
it becomes a parse error). Stage B2 (patches every `crates/*/Cargo.toml` matching
`serde = { workspace = true[...] }`, not just the exact-string variant) and Stage B3
(trims the root `Cargo.toml`'s `members`/`default-members` to the target crate's real
reachable subtree, removing unrelated workspace crates from Cargo's
feature-unification scope entirely) touch disjoint files from each other and from A/B.
**Corrected 2026-08-20 (Task 16 pre-dispatch check):** this document previously claimed
they "run as `background: true` steps concurrent with A/B, joined by a `wait`" — checked
directly against GitHub's own workflow-syntax documentation (WebFetch, not assumed) and
against every real workflow file in this repository (`grep -rn "background:\|wait-all:\|wait:"
.github/workflows/*.yml`, zero matches anywhere): there is no `background`/`wait`/
`wait-all`/`parallel` step-level key documented anywhere in GitHub Actions. Steps within
a job execute strictly sequentially; concurrency is only controllable at the workflow
and job level, never the step level. This claim was never validated by an actual run
(this document's own Validation approach section requires exactly that), and is now
ruled out. Stages A, B, B2, and B3 run as four plain sequential steps within the
`secondary-mutate` job — real wall-clock cost is negligible regardless (a `clippy --fix`
pass, one `sed`, and one short Python script), so sequential execution costs nothing
worth optimizing away. Every stage's effect is captured as a
before/after diff of the Primary layer's `--unit-graph` and target-capability output,
not inferred from Stage C's pass/fail alone — that diff is the validation signal, and
whatever new findings appear at that depth are attributed to having cleared the prior
layer specifically.

## Data flow

**Triggers**: `push` alone, scoped to the relevant branch pattern. `workflow_dispatch`
is available as a secondary, manual single-target convenience path, understood to only
become usable after the branch has already fired once via `push` (GitHub Actions does
not register `workflow_dispatch` on a non-default branch until it has run via a
non-dispatch trigger — discovered empirically in this project's own prior work, not
assumed). `pull_request` was considered and rejected: it adds nothing here (no
external-fork contributors, main is disposable, GitHub already attaches check runs to
the commit SHA regardless of which event triggered them) and would cause duplicate
runs whenever a branch happens to have an open PR.

**Fan-out**: two real prerequisite edges exist, both mechanical (Principle 8), and both
have to be declared as `needs:`, not treated as if everything were flat. First: when a
specific target is given as `workflow_dispatch` input, the crate tree for the root
crate (cycle-aware reachability traversal) and target-capability probes for that one
target have no real prerequisite and can start immediately alongside the Discovery job.
Second: when the pipeline is determining the full set of targets/crates itself (the
normal `push`-triggered path), every downstream job's `strategy.matrix` is populated
via `fromJSON()` from the Discovery job's own output — that *is* a genuine mechanical
dependency, not optional parallelism, and is expressed as `needs: [discovery]` on every
job whose matrix depends on it. Within a single target's fan-out, once the matrix
exists, target-capability, dependency-graph, and build-attempt probes have no
prerequisite *on each other*. **Corrected 2026-08-20** (same correction as above,
same real evidence): this paragraph's original "`parallel:`/`background`+`wait` steps
within jobs, sharing one checkout" framing assumed a step-level concurrency mechanism
that does not exist in GitHub Actions. As actually implemented (Tasks 7-9), each probe
is its own separate batched job (`target-capability`, `dependency-graph`,
`build-attempt`), each with its own `needs: [discovery]` (plus `target-capability` for
`build-attempt`, which needs its batch's capability artifact) and no `needs:` edge
against each other — GitHub Actions schedules jobs with no `needs:` relationship between
them concurrently on its own, which is the real mechanism this design relies on, not an
intra-job step feature. The checkout/toolchain-install cost is paid per job because jobs
are the real isolation unit; there is no mechanism to share it across separate jobs
short of a cache (explicitly rejected elsewhere in this document) — build-attempt probes
specifically never wait on what the dependency-graph probes found, they always run
regardless. Beyond `needs: [discovery]`,
`needs:` appears only within the Secondary layer's own internal stage ordering, and at
its boundary with the Primary layer (`needs: [primary layer jobs]`,
`if: !cancelled()`, since those are expected to genuinely fail).

**Storage**: every probe's complete raw output to `actions/upload-artifact`, one
artifact per (probe, tool, crate, target, mode, feature-config), named on a fixed,
parseable scheme.

**Indexing**: a final job, gated only on completion (`!cancelled()`), runs a
checked-in, version-controlled script (not ad hoc logic decided at run time) that
performs purely structural extraction — error counts by `target.name`, boolean
artifact-presence, direct field pass-through, coded contradiction rules (e.g.
`target-spec.std == false AND build[std-mode].errors == []` → flag
`unexpected-clean-std-build`) — producing a navigation index, not a digest. The
completeness of this index is itself checked: the expected artifact set is enumerable
in advance from the declared probes and matrix, and the indexing step fails loudly if
any expected artifact is missing, rather than silently indexing whatever happened to
show up.

Both the discovery job and the indexing job are executed entirely by the runner, as
plain `run:` steps, autonomously, every run — this is not relaxable. If every human
and every agent who ever worked on this pipeline disappeared, a fresh push to the
branch would still produce the same structural guarantees.

**Open question, not resolved by anything in this design:** GitHub Actions artifacts
expire (90 days by default) and this pipeline is meant to be re-run per target over
time as the codebase evolves. Whether cross-run history needs to be preserved/diffable
(longer retention, or committing the index itself for longitudinal comparison) is
still an open decision, not assumed either way.

## Error handling

Execution failure (a probe couldn't run to completion — a tool crashed, a flag was
silently non-functional, an external API timed out, a platform limit was hit) and a
finding (the probe ran fine; its content happens to be negative) are different
categories, and conflating them recreates the green/red camouflage problem this
project already fixed once elsewhere. Job-level conclusion means different things by
probe type accordingly: observability-probe jobs (target-capability, dependency-graph,
discovery, indexing) reflect execution success only — a job that correctly determines
`std: false` has succeeded at its actual task and should be green regardless of how
negative that finding is. Build-attempt-probe jobs reflect the build's own pass/fail,
via the honest-result pattern already proven (step-level `continue-on-error`, no
job-level masking, a final unmasked assertion step checking real `.outcome`).

**Corrected 2026-08-20 (final whole-branch review):** this document's "a final
unmasked assertion step checking real `.outcome`" was never actually built this
way, and Task 8's own text explicitly adjudicated the deviation rather than
missing it: `fmt` and `build-attempt`'s honest-result steps read the real
`.outcome` and emit an `::notice::`, but never assert/fail on it. The reasoning
(recorded in Task 8's plan text): the uploaded raw JSON is strictly more
informative than a boolean pass/fail, `if: always()` already guarantees it's
never lost to a masked failure, and a human or the indexing job's own
completeness check is the actual consumer, not a job-level gate. This is a
real, deliberate, evidenced choice, not an oversight — this section is
corrected to match what was actually built and validated by real CI, per this
document's own Validation-approach standard.

Retries are narrow: only external network calls (crates.io, GitHub's API) get a
bounded, logged retry, since a rate limit or timeout isn't information about the
target. Actual `cargo build`/`clippy` attempts are never retried — deterministic
against the given source and target, a retry produces the same answer and would only
obscure a real, reproducible finding as if it might have been a fluke.

Platform-limit hits get their own explicit finding category
(`platform-limit-hit`), distinct from both "clean" and "genuine compile error" — the
precedent being the step-summary 1024KB cap silently dropping ~5MB of real Stage C
output before this was caught and fixed.

Contradictions get privileged placement in any index or report output — never mixed
in among routine findings where they could be scrolled past, since a contradiction
between two independently mechanically-grounded sources is the single highest-value
output this design exists to produce.

## Testing

**The nvptx canary** (Standing Principle 5) is the primary, always-on validation
mechanism for the Primary layer.

**Cross-checking against the original target-sampling pipeline's already-obtained
results** (`wasm32v1-none` specifically has real, historical output from the simpler,
original design) — agreement is expected; new findings the old pipeline missed are the
new pipeline earning its complexity; unexplained disagreement is a bug in the new
pipeline to chase before trusting it on a target with nothing to check against.

**Indexing/discovery scripts get ordinary unit tests** against fixed, synthetic JSON
fixtures with known expected extraction results — standard software-testing discipline
applied to the pipeline's own L1 components, independent of anything about larql-cli.

**Reproducibility is a directly testable, mechanical property**: run any probe or
script twice against byte-identical input, assert byte-identical output. Failing this
is itself proof a component claiming L1 status isn't actually L1.

**Autonomy is checkable structurally**: no step requiring manual approval or paused
input, confirmed by inspection of the workflow file and by the fact that every real
run this design is built on has already run unattended via `push`.

**Completeness is checked directly, not just declared**: the indexing step's
loud-failure-on-missing-artifact behavior (see Data flow) is itself the completeness
test, turning "under-running is the real risk" from a principle into an automatic
check.

**The Secondary layer's testing dimension is distinct and harder**, since its whole
purpose is exploring past the frontier of what's currently known — there is no
pre-existing "correct depth-3 finding" to check a given mutation's output against
directly. Four things are testable regardless:

- **Noise floor, established before trusting any depth attribution.** Run the same,
  *unmutated* state twice; confirm findings are stable across the two runs before
  treating any depth-N → depth-(N+1) difference as caused by the mutation rather than
  by ordinary run-to-run variance (registry ordering, patch-version drift,
  `--keep-going` output ordering).
- **Blast-radius containment, per stage.** `git diff` the checkout immediately before
  and after each stage; assert the changed files are exactly the stage's declared
  scope. A stage touching something outside its intended files is a meta-error
  contaminating everything built on top of it.
- **Golden fixtures with a deliberately known, planted outcome** — generalizing what
  `serde-nostd-probe` already was: a minimal, synthetic crate with a specific,
  known-in-advance issue, run through the full Stage A→C pipeline, asserting the
  pipeline reports the known result. This tests the mutation-and-report machinery
  itself, not larql-cli.
- **Ephemerality and non-leakage, checked structurally**: no stage invokes `git
  commit`/`git push` (mechanically greppable); a fresh run starts from genuinely
  unmutated source, confirmed by the fact that mutations from run N never appear in
  run N+1's baseline.
- **Cross-target and cross-native comparison of the same underlying finding.** The
  native build/test result is existing, trusted ground truth for whether a mutation
  preserved real correctness, independent of whether the target ever succeeds at all —
  running the same mutated source against native, not just the target, answers "did
  this mutation break something real" directly. Once the broader target matrix exists
  (wasm family, native, nvptx, others), the same underlying finding (e.g., does
  `serde_core`'s `de::value` module have this gap) becomes checkable across targets
  using data the pipeline is already producing as a byproduct — agreement across
  no_std-constrained targets reframes a finding as general rather than target-specific;
  disagreement narrows the hypothesis space toward something target-specific, either
  way from existing data rather than new investigation built solely to answer this one
  question.

## Explicitly not doing

- No caching (`actions/cache`, `Swatinem/rust-cache`) — matches the original
  target-sampling design's own reasoning: runner minutes aren't a constraint at this
  project's tier, and there's no data yet on what's actually slow.
- No commits from CI, ever, anywhere in either layer — matches the original design's
  same rule, extended to the Secondary layer's mutations, which are explicitly
  ephemeral to a single job's checkout.
- No agent-authored curation presented as if it were mechanically-grounded (L1) — every
  curated list (`deny-nvptx.toml`, the target-family tooling registry) is explicitly
  labeled as such, sourced, and checkable against raw, uncurated scan output.
- No resolution of the cross-run artifact retention question (Data flow) — left open,
  not assumed.

## Validation approach

Same as the original target-sampling design: validated exclusively by running for real
on GitHub-hosted runners, never by local simulation. This extends to the design's own
claims about GitHub Actions mechanics (`fromJSON()` dynamic matrices, reusable-workflow
`jobs.<id>.uses`) — verified against the actual current documentation source (not a
summarized/model-processed version of it, which was directly caught omitting real
content during this design's own development) before being relied on, and ultimately
proven by an actual run, not by the documentation alone.

**Corrected 2026-08-20:** this section previously included "the `background`/`wait`/
`parallel` step family" in the list of mechanics claims described as verified. That
was itself an unverified claim slipping past its own standard — a direct WebFetch of
GitHub's workflow-syntax documentation, done as part of Task 16's pre-dispatch check,
found no such step-level keys documented anywhere, and a repository-wide grep found
zero real usages in any workflow file this project has ever written. This is exactly
the failure mode this section exists to prevent (an assumed mechanics claim, never
actually run-proven), caught by applying the section's own rule retroactively. See the
Secondary-layer stages and Fan-out sections above for the corrected design (plain
sequential steps within `secondary-mutate`; separate concurrent jobs, not intra-job
step concurrency, for target-capability/dependency-graph/build-attempt).
