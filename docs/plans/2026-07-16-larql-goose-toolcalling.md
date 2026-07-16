# Implementation Plan — Tool-Calling for the larql-driven Goose Coding Agent

- **Source design**: `docs/specs/2026-07-16-larql-goose-toolcalling-design.md`
- **Source residual**: `docs/specs/2026-07-16-larql-goose-toolcalling-research-residual.md`
- **Manifest**: `rdloop.toml`
- **Date**: 2026-07-16
- **Repos touched**: `metavacua/larql-to-sparql` (**Repo A**, checked out at cwd — the 12-leg CI
  workflow lives here) and `metavacua/goose` (**Repo B**, local clone at
  `~/work/metavacua-goose`, `feat/larql-local-inference-backend` branch — the new tool-call
  emulation module lives here). Every task states which repo its Files live in.

## Global Constraints (RFC 2119, copied verbatim from the design doc)

- The strategy matrix MUST include at least one approach requiring zero model-weight changes and
  zero new parsing theory as the baseline (harness-level emulation).
- The strategy matrix MUST treat the three LARQL-native approaches as discovery/measurement legs,
  not pass/fail gates; a negative/inconclusive result MUST NOT block merge.
- The CI strategy matrix MUST cap concurrent GitHub-hosted runners at 12 (`max-parallel: 12`).
- Any new harness-level parser MUST produce `MessageContent::ToolRequest{id, tool_call:
  Ok(CallToolRequestParams)}` — MUST NOT introduce a parallel dispatch path.
- A new harness-level parser MUST NOT depend on the `mlx` feature or any `pub(super)`-scoped
  `llamacpp::` module.
- LARQL-native approaches MUST NOT mutate a base vindex directly — all edits MUST flow through
  `PatchedVindex` overlays / `.vlp` patches.
- Every strategy-matrix leg MUST run inside the existing QEMU/KVM VM boundary when it exercises a
  live `goose run` coding task.
- The design SHOULD hold out the literal test-task answer from any few-shot example used to prime
  tool-call emission.

## Task Graph

```json
{
  "tasks": [
    {"id": "T1", "depends_on": []},
    {"id": "T2", "depends_on": ["T1"]},
    {"id": "T3", "depends_on": ["T1"]},
    {"id": "T4", "depends_on": []},
    {"id": "T5", "depends_on": []},
    {"id": "T6", "depends_on": []},
    {"id": "T7", "depends_on": ["T2", "T3", "T4", "T5", "T6"]},
    {"id": "T8", "depends_on": ["T7"], "on_fail": {"precondition": "any of legs 1-6 (VM-based) times out repeatedly on GitHub-hosted runners", "route": "escalate-to-user"}},
    {"id": "T9", "depends_on": ["T7"]}
  ]
}
```

### Parallelization Table (wave -> members)

| Wave | Members |
|---|---|
| 1 | T1, T4, T5, T6 |
| 2 | T2, T3 |
| 3 | T7 |
| 4 | T8, T9 |

T1 (the emulation parser module) gates T2/T3 (its two config variants) since they're the same
module with a mode toggle. T4 (native-template measurement), T5 (LARQL-native CLI measurement
scripts), and T6 (CI workflow skeleton) are independent of T1 and of each other. T7 (assemble the
full 12-leg matrix) needs all of T2/T3/T4/T5/T6 done. T8 (run it) and T9 (aggregate/report) both
need T7 but not each other, so they parallelize on GitHub's side (the workflow's own
`matrix`/`aggregate` jobs), not in this task graph.

### Serialization Table (task -> depends_on)

| Task | Depends on |
|---|---|
| T2 | T1 |
| T3 | T1 |
| T7 | T2, T3, T4, T5, T6 |
| T8 | T7 |
| T9 | T7 |

### Contingency Table

| Task | on_fail precondition | Route |
|---|---|---|
| T8 | Any of legs 1-6 (VM-based) times out repeatedly on GitHub-hosted runners | escalate-to-user (matches this project's established pattern for VM/CI infra faults it can't root-cause without more runner budget) |

## Task 1 — `larql_tool_emulation.rs`: streaming-adapted emulation parser (approach `emulate-stream-harness`)

**Repo**: B. **AC**: AC-1, AC-2. **ADR**: ADR-2.
**Files**: new `crates/goose-local-inference/src/larql_tool_emulation.rs`;
`crates/goose-local-inference/src/lib.rs` (add `mod larql_tool_emulation;`, ungated);
`crates/goose-local-inference/src/larql.rs` (wire into `generate()`).
**Interfaces**: a `LarqlEmulatorParser` type with a `push_line(&mut self, line: &str) ->
Option<EmulatedAction>` method (buffered-per-line, not per-token, per residual K39) mirroring
`llamacpp/inference_emulated_tools.rs`'s `EmulatorAction::{ShellCommand,ExecuteCode}` shape;
`build_larql_emulator_tool_description(tools: &[Tool]) -> String` reusing
`tool_parsing::compact_tools_json` (already ungated, K41); a `send_emulator_action` equivalent
constructing `MessageContent::tool_request(Uuid::new_v4(),
Ok(CallToolRequestParams::new(...).with_arguments(...)))` exactly matching
`reply_parts.rs::categorize_tool_requests`'s expected shape (K40).

- [ ] Write a failing unit test in `larql_tool_emulation.rs`'s own `#[cfg(test)]` block: given the
      line `$ ls -1 /tmp | wc -l`, `push_line` returns
      `Some(EmulatedAction::ShellCommand("ls -1 /tmp | wc -l"))`; given an ordinary text line, it
      returns `None`. Run it, confirm it fails (module doesn't exist yet).
- [ ] Implement `LarqlEmulatorParser` and `build_larql_emulator_tool_description`, adapting
      `StreamingEmulatorParser`'s pattern-matching logic (not its per-token buffering, which
      doesn't apply here) to whole-line input.
- [ ] Wire into `larql.rs::generate()`: when `request.tools` is non-empty, append
      `build_larql_emulator_tool_description(request.tools)` to the tiny-model prompt (T1 of the
      2026-07-15 phase already established `tiny_model_prompt()`), and route each response line
      through `LarqlEmulatorParser::push_line` before constructing the `Message` sent over
      `tx.blocking_send` — on a match, send a tool-request message instead of plain text.
- [ ] Run the test, confirm green.
- [ ] `cargo test -p goose-local-inference` (full crate), confirm no regression.
- [ ] `cargo check -p goose-cli --no-default-features --features local-inference,rustls-tls`
      (matches this project's established CI build config), confirm it still compiles.
- [ ] Commit (Repo B, `feat/larql-local-inference-backend` branch).

## Task 2 — `shell-conv` config variant (matrix leg 1)

**Repo**: B. **AC**: AC-1, AC-2. **ADR**: ADR-2, ADR-6 (leg 1).
**Files**: `crates/goose-local-inference/src/larql_tool_emulation.rs` (mode selection).
**Interfaces**: `LarqlEmulatorParser::new(EmulatorConvention::ShellCommand)`.

- [ ] Confirm T1's default/only implemented convention already matches this leg's config
      (`$ command` lines) — if so, this task is "already satisfied by T1," record that plainly
      rather than duplicating code, and add the leg-specific integration test: a fixed prompt
      through `LarqlBackend::generate()` (mocked child process, not a live `larql chat`) producing
      a `ToolRequest` for `developer__shell`.
- [ ] Commit only if a real code change was needed; otherwise note "no-op, satisfied by T1" in the
      commit that closes T7.

## Task 3 — `fenced-tool` config variant (matrix leg 2)

**Repo**: B. **AC**: AC-1, AC-2. **ADR**: ADR-2, ADR-6 (leg 2).
**Files**: `crates/goose-local-inference/src/larql_tool_emulation.rs` (second convention).
**Interfaces**: `EmulatorConvention::FencedJson`, matching ` ```tool_call\n{...}\n``` ` blocks.

- [ ] Write a failing unit test: given a 3-line sequence (fence-open, one JSON line, fence-close)
      fed through `push_line` one line at a time, the parser accumulates and returns
      `Some(EmulatedAction::ExecuteCode(...))`-equivalent (or a new `EmulatedAction::ToolCallJson`
      variant, whichever keeps `larql.rs`'s dispatch code simplest) only after the closing fence.
- [ ] Implement the `FencedJson` convention as a second small state machine alongside T1's
      shell-command one (same file, same `LarqlEmulatorParser` type, an enum discriminating which
      convention is active).
- [ ] Run the test, confirm green; `cargo test -p goose-local-inference` full-crate regression.
- [ ] Commit.

## Task 4 — Native chat-template support measurement (matrix legs 5-6)

**Repo**: B. **AC**: AC-4. **ADR**: ADR-1 (approach `native-template-wiring`), closes Q5.
**Files**: new small standalone binary or `#[test]` in `crates/goose-local-inference/` — this task
is a measurement, not a production-code change (per the design's honest framing: legs 5-6 exist to
answer "does SmolLM2's template even support native tool-calling," not to ship a feature whose
premise is unconfirmed).
**Interfaces**: reuses `llamacpp/mod.rs`'s existing `template_result_supports_native_tool_calling`
predicate logic (read-only call, no wiring into `larql.rs` itself yet — that would be a follow-on
task gated on this measurement's answer).

- [ ] Write a small harness (test or example binary) that renders SmolLM2-135M-Instruct's own
      `tokenizer_config.json` chat template (already confirmed to exist and be a real ChatML
      template, per this session's earlier direct verification) against a synthetic tool-definition
      list, once with `compact_tools_json` (leg 5) and once with a full JSON-schema tool
      definition (leg 6), and evaluates `template_result_supports_native_tool_calling`'s predicate
      against the rendered output.
- [ ] Run it, record the boolean result for each config as this task's actual deliverable — a
      grounded answer to Q5, not a guess.
- [ ] Commit the harness code (kept small and clearly labeled as a measurement tool, not
      production wiring) regardless of which way the answer comes out.

## Task 5 — LARQL-native measurement scripts (matrix legs 7-12)

**Repo**: A. **AC**: AC-4. **ADR**: ADR-3, ADR-6 (legs 7-12).
**Files**: new `scripts/toolcalling_matrix/` directory: `run_patch_chain.sh`,
`run_patch_ensemble.sh`, `run_introspect_patch.sh` (one script per approach, parameterized for
each leg's specific config per the ADR-6 table).
**Interfaces**: each script invokes real, existing LARQL CLI commands (`larql lql 'INSERT INTO
EDGES ... MODE compose'`, `larql walk`, `larql circuit-discover`, `larql dev ov-rd`) against a
SmolLM2-135M-Instruct vindex and writes a JSON result record (`{"leg_id", "outcome":
"installed"|"skipped"|"error", "detail": "..."}`) — no script may report `"outcome": "installed"`
without the underlying LQL/CLI command itself having exited 0 and the described edit actually
being present in the resulting vindex (checked via `larql show`/`larql walk` post-edit, not
assumed from the command's own exit code alone — closes this project's own N1-style "silent no-op
reported as success" concern, carried forward from the 2026-07-15 design).

- [ ] `run_patch_chain.sh`: for legs 7-8, chain the described single-token `INSERT ... MODE
      compose` calls (opening-tag tokens only for leg 7; extended to JSON-key tokens for leg 8),
      then verify via `larql walk` that each installed feature is actually present and its target
      token's probability is inside the balancer's configured band — record `installed` only if
      true for every chained slot, `partial` if some but not all installs verify, `error`
      otherwise.
- [ ] `run_patch_ensemble.sh`: for legs 9-10, run the 5-prompt ensemble (shared slot for leg 9,
      distinct slots for leg 10), then measure post-edit generation against all 5 phrasings (not
      just the install prompt) to test generalization — record the fraction of the 5 phrasings
      that actually trigger the edited behavior, not just whether the `INSERT` succeeded.
- [ ] `run_introspect_patch.sh`: for leg 11, run the contrastive prompt set through `larql walk`
      only and emit the ranked feature-correlation table (no patch applied — this leg is
      explicitly cheapest/discovery-only per ADR-6). For leg 12, additionally run
      `circuit-discover` clustering and `dev ov-rd`'s ablation/replacement on the top-ranked
      candidates, then apply the winning feature via the existing `PatchedVindex` overlay and
      verify the same way `run_patch_chain.sh` does.
- [ ] Smoke-test each script locally against the already-extracted local SmolLM2-135M-Instruct
      vindex (confirmed present on this dev machine earlier this session) before wiring into CI.
- [ ] Commit.

## Task 6 — `goose-larql-toolcalling-matrix.yml` skeleton (plan/build/matrix/aggregate DAG)

**Repo**: A. **AC**: AC-3, AC-6. **ADR**: ADR-4.
**Files**: new `.github/workflows/goose-larql-toolcalling-matrix.yml`.
**Interfaces**: `plan` job emitting the hand-authored 12-leg JSON (ADR-6's table, literal — no
generator script, per ADR-4's rationale); `build` job compiling `larql-cli` and `goose-cli` once
each, uploaded as artifacts; `matrix` job with `strategy.max-parallel: 12`, `fail-fast: false`,
`matrix.leg: fromJSON(needs.plan.outputs.legs)`; `aggregate` job rendering
`$GITHUB_STEP_SUMMARY`.

- [ ] Author the `plan` job's leg JSON matching ADR-6's table exactly (12 entries, fields:
      `leg_id`, `approach_id`, `kind` [`vm-coding-task` for legs 1-6, `cli-measurement` for legs
      7-12], `config`).
- [ ] Author `build` (reuses this session's existing `build-larql`/`build-goose` job bodies from
      `goose-larql-vm-pipeline.yml` verbatim where possible, to avoid drifting conventions).
- [ ] Author `matrix`: legs where `kind == vm-coding-task` run the existing VM bake/boot/assert
      steps (parameterized by `leg.config` for prompt/emulator-mode selection via env vars); legs
      where `kind == cli-measurement` run the matching Task 5 script directly on the runner host
      (no VM needed — these never start an inference-serving process, only bounded LARQL CLI
      subcommands over an already-extracted vindex).
- [ ] Author `aggregate`: descriptive-only markdown table (leg_id, approach_id, kind, outcome),
      matching `aggregate.py`'s "no pass/fail-correctness judgement" stance for the `cli-measurement`
      legs; `vm-coding-task` legs' pass/fail (did the assert step find a real `ToolRequest`
      dispatched) is reported as-is since those legs DO have a real correctness bar (AC-5).
- [ ] Validate: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/goose-larql-toolcalling-matrix.yml'))"`
      exits 0.
- [ ] Commit.

## Task 7 — Assemble and dry-run the full matrix

**Repo**: A (workflow), B (module, already committed by T1-T4).
**AC**: AC-3, AC-5, AC-6. **ADR**: ADR-4, ADR-6.
**Files**: `.github/workflows/goose-larql-toolcalling-matrix.yml` (from T6, now wired to the real
T1-T5 artifacts/scripts).

- [ ] `workflow_dispatch` a first real run on the `feat/larql-goose-selfplay` branch.
- [ ] Confirm the `plan` job's leg count is exactly 12 and `matrix`'s concurrent-job count never
      exceeds 12 (`gh run view --json jobs` timestamps, or the Actions UI's own concurrency
      display).
- [ ] If any leg fails for an infra reason (timeout, OOM, missing runtime lib) rather than a
      genuine leg-specific finding, root-cause and fix it the same way this session already fixed
      four such bugs in `goose-larql-vm-pipeline.yml` (GLIBC mismatch, disk space, HF pre-download,
      chat template) — do not mask an infra bug as a "leg finding."

## Task 8 — Run to completion, capture results

**Repo**: A. **AC**: AC-3, AC-4, AC-5, AC-6. **ADR**: ADR-3, ADR-4.

- [ ] Re-run after T7's fixes until all 12 legs complete (not necessarily all "pass" — legs 7-12
      completing with an honest negative/partial finding satisfies AC-4).
- [ ] Download and review the `aggregate` job's `$GITHUB_STEP_SUMMARY` output.

## Task 9 — Update the research residual with matrix findings

**Repo**: A. **AC**: AC-4. **ADR**: ADR-3.
**Files**: `docs/specs/2026-07-16-larql-goose-toolcalling-research-residual.md`.

- [ ] Append new K-entries (K46+) recording each leg's actual outcome, resolving Q5-Q8 with
      empirical answers (or narrowing them, if a leg's result is itself inconclusive — recorded as
      such, not papered over).
- [ ] If `emulate-stream-harness` (legs 1-2) demonstrably works end-to-end (a real `ToolRequest`
      dispatched from a live `goose run` coding task), record this as the project's first working
      tool-calling capability and note it in the main `2026-07-15` design doc's own "Open items"
      section as resolved.

## Requirements Traceability Matrix

| AC | Task(s) | Test(s) |
|---|---|---|
| AC-1 (ungated, mlx-independent parser) | T1 | T1's unit test + `cargo check --features local-inference,rustls-tls` |
| AC-2 (correct `ToolRequest` shape, no bespoke dispatch) | T1, T2, T3 | T1/T3's unit tests; T2's integration test |
| AC-3 (exactly 12 legs enumerated) | T6, T7 | T7's `plan` job leg-count assertion |
| AC-4 (LARQL-native legs recorded as discovery, not gated) | T5, T8, T9 | T5's script-level `outcome` field; T9's residual update |
| AC-5 (real dispatch verified, liveness bounded) | T2, T7, T8 | T2's integration test; T7's dry-run assertion |
| AC-6 (`fail-fast: false`, one leg's failure doesn't block siblings) | T6, T7 | T7's dry-run confirms sibling legs complete despite one failure |
| AC-7 (parser is model-agnostic) | T1 | *(not directly tested this phase — carried forward as a residual open item if a second model is ever swapped in)* |

## Self-review

- Every task with a code change (T1-T5) has a write-failing-test-first step. T6-T9 are
  infra/aggregation/measurement tasks without new production logic of their own, so they use a
  validate/dry-run/record step instead, stated plainly rather than a fabricated TDD step. ✔
- Every task cites at least one AC and the design doc's ADR. ✔
- No task says "TBD." T4 and T5 are explicitly framed as measurement tasks whose *answer* is the
  deliverable, not a fixed implementation outcome — this is intentional per ADR-1/ADR-3, not an
  unresolved placeholder. ✔
- AC-7 has no dedicated test this phase — recorded honestly in the RTM rather than a fabricated
  citation, and carried forward as a residual open item. ✔
