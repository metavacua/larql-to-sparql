# Implementation Plan — Self-Playing, Self-Learning LARQL Coding Agent

- **Source design**: `docs/specs/2026-07-15-larql-goose-selfplay-design.md`
- **Source residual**: `docs/specs/2026-07-15-larql-goose-selfplay-research-residual.md`
- **Manifest**: `rdloop.toml`
- **Date**: 2026-07-15
- **Repos touched**: `metavacua/larql-to-sparql` (**Repo A**, checked out at cwd) and
  `metavacua/goose` (**Repo B**, fork — NOT yet checked out locally; T0 provisions it). Every
  task below states which repo its Files live in. A delegated subagent MUST be told which repo's
  working tree to operate in before starting a task.

## Global Constraints (RFC 2119, copied verbatim from the design doc)

- The self-play loop MUST score trajectories using an evaluator that does not share mutable state
  with the actor.
- The system MUST NOT serve the vindex via `larql-server` as the Goose transport.
- The system MUST NOT invoke any local-model-loading command unwrapped by an active containment
  boundary; because the Goose provider links LARQL in-process, the containment unit MUST be the
  whole `goose` process, not individual CLI calls.
- All actual execution of `larql-cli`/REPL/`goose` commands MUST be delegated to a subagent; the
  orchestrating loop MUST NOT run them inline.
- Base vindexes MUST remain immutable; all mutation MUST flow through `PatchedVindex`
  overlays/`.vlp` patches.
- A self-learning round that installs zero effective edits MUST be distinguishable, in its own
  logged outcome, from a round that installed N>0 edits.
- Any GitHub branch/PR/issue write MUST stay within `[security].write_allowed` in `rdloop.toml`.
- The coding-task suite SHOULD default to difficulty scoped under SmolLM2-135M's actual capability
  ceiling.
- The design SHOULD measure a behavioral before/after delta before claiming a learning round
  changed anything.

## Task Graph

```json
{
  "tasks": [
    {"id": "T0",  "depends_on": []},
    {"id": "T1",  "depends_on": []},
    {"id": "T2",  "depends_on": []},
    {"id": "T3",  "depends_on": []},
    {"id": "T4",  "depends_on": []},
    {"id": "T5",  "depends_on": []},
    {"id": "T8",  "depends_on": []},
    {"id": "T10", "depends_on": []},
    {"id": "T16", "depends_on": []},
    {"id": "T17", "depends_on": []},
    {"id": "T6",  "depends_on": ["T5"]},
    {"id": "T9",  "depends_on": ["T10", "T17"], "on_fail": {"precondition": "peak RSS during extraction exceeds the declared ceiling", "route": "escalate-to-user"}},
    {"id": "T11", "depends_on": ["T0"]},
    {"id": "T14", "depends_on": ["T10"]},
    {"id": "T7",  "depends_on": ["T6"]},
    {"id": "T12", "depends_on": ["T11"]},
    {"id": "T13", "depends_on": ["T12"]},
    {"id": "T15", "depends_on": ["T1", "T2", "T3", "T4", "T6", "T7", "T8", "T9", "T10", "T13", "T14", "T16", "T17"]}
  ]
}
```

### Parallelization Table (wave -> members)

| Wave | Members |
|---|---|
| 1 | T0, T1, T2, T3, T4, T5, T8, T10, T16, T17 |
| 2 | T6, T9, T11, T14 |
| 3 | T7, T12 |
| 4 | T13 |
| 5 | T15 |

Wave 1 is now a 10-way fork — dispatch via `Workflow`/concurrent `Agent` calls, not sequentially.
T1-T4 touch four disjoint files (`compose.rs`, `tuning.rs`, `plan.rs`, `executor/mod.rs`); T16
touches `larql-cli`'s `main()` + `larql-compute::cpu::spin_pool`; T17 touches `larql-cli`'s
`main()` (a different function/startup hook than T16, but same file — coordinate order of
insertion, or land sequentially within the pair, to avoid a trivial merge conflict even though
there's no logical conflict). T0/T5/T8/T10 touch no files any of T1-T4/T16/T17 touch.

### Serialization Table (task -> depends_on)

| Task | Depends on |
|---|---|
| T6 | T5 |
| T9 | T10, T17 |
| T11 | T0 |
| T14 | T10 |
| T7 | T6 |
| T12 | T11 |
| T13 | T12 |
| T15 | T1, T2, T3, T4, T6, T7, T8, T9, T10, T13, T14, T16, T17 |

### Contingency Table

| Task | Precondition | Route |
|---|---|---|
| T9 | peak RSS during extraction exceeds the declared ceiling | escalate-to-user |

(T10's original `flyctl` contingency no longer applies — ADR-2 moved to a local QEMU/KVM boundary,
already installed and verified booting; see Task 10 below.)

---

## Task 0 — Provision `metavacua/goose` as a local worktree

**Repo**: B. **AC**: none directly (infrastructure prerequisite for T11-T13).
**Files**: new local clone path (e.g. `~/work/metavacua-goose/`, outside this repo).
**Interfaces**: none (setup only).

- [ ] `git clone https://github.com/metavacua/goose ~/work/metavacua-goose` (or `gh repo clone`)
- [ ] Smoke check (this task's "test", since there's no code yet): `cargo check -p goose-provider-types -p goose-local-inference --manifest-path ~/work/metavacua-goose/Cargo.toml` exits 0 — confirms the workspace builds and the two reference crates (trait definition, FFI-precedent provider) actually resolve before any new code depends on them. A non-zero exit is a real failure, not a skip condition — report verbatim, do not silently proceed.
- [ ] Record the clone path in `rdloop.toml` under `[x.goose_fork] path = "..."` for later tasks to reference.

## Task 1 — Fix #261: silent capacity-collision skip

**Repo**: A. **AC**: AC-3. **ADR**: ADR-3.
**Files**: `crates/larql-lql/src/executor/mutation/insert/compose.rs`
**Interfaces**: `find_free_feature(...) -> Option<usize>` call site at `compose.rs:102-106`; must
change the `continue`-on-`None` branch to return a typed outcome distinguishing "installed" from
"skipped: capacity".

- [ ] Write a failing test in `crates/larql-lql/tests/` (or the module's own `#[cfg(test)]` block)
      asserting: given a layer whose feature slots are all occupied, `INSERT`/`COMPOSE` returns an
      explicit `Skipped(CapacityCollision)` (or equivalent `Err`/enum variant) rather than
      succeeding silently with zero effect. Run it, confirm it fails for the right reason (current
      code returns `Ok`/no signal at all, not a wrong-value assertion failure).
- [ ] Implement the minimal change: replace the bare `continue` with a recorded skip reason
      returned to the caller.
- [ ] Run the test, confirm green.
- [ ] Run `cargo test -p larql-lql` (full crate) to confirm no regression.
- [ ] Commit.

## Task 2 — Fix #237: `alpha_mul` miscalibration for small `hidden_dim`

**Repo**: A. **AC**: AC-4. **ADR**: ADR-3.
**Files**: `crates/larql-lql/src/executor/tuning.rs`
**Interfaces**: `DEFAULT_INSERT_ALPHA_MUL: f32 = 0.1` (`tuning.rs:34-38`) becomes a function of
`hidden_size` (or an explicit override surfaced to the self-play driver), not a fixed constant.

- [ ] Write a failing test asserting that for a `hidden_size` representative of SmolLM2-135M (per
      its `config.json`, fetched in T9 or hardcoded from the known HF config), the computed alpha
      differs from the Gemma-3-4B-calibrated constant by a bounded, intentional ratio (not simply
      "any different value" — assert the actual scaling formula's output).
- [ ] Implement the scaling function.
- [ ] Run, confirm green; run `cargo test -p larql-lql`.
- [ ] Commit.

## Task 3 — Fix #238: multi-subtoken targets don't chain

**Repo**: A. **AC**: AC-3. **ADR**: ADR-3.
**Files**: `crates/larql-lql/src/executor/mutation/insert/plan.rs` (`plan.rs:107-124`)

- [ ] Write a failing test: `INSERT` with a target string that tokenizes to >1 subtoken asserts all
      subtokens are chained into the down vector (not just the first), by checking the resulting
      patch's operation list covers each subtoken position.
- [ ] Implement chaining across all subtokens of the target.
- [ ] Run, confirm green; run `cargo test -p larql-lql`.
- [ ] Commit.

## Task 4 — Fix #252: `BEGIN`/`SAVE`-created patches are unremovable

**Repo**: A. **AC**: AC-3. **ADR**: ADR-3.
**Files**: `crates/larql-lql/src/executor/mod.rs` (`exec_save_patch` ~391-400, `REMOVE PATCH`
lookup ~502-505)

- [ ] Write a failing test: `BEGIN; ...; SAVE PATCH 'x';` followed by `REMOVE PATCH 'x';` asserts
      the patch is actually removed (currently fails because `description` is hardcoded `None`).
- [ ] Implement: persist the real description/path on save so the removal lookup matches.
- [ ] Run, confirm green; run `cargo test -p larql-lql`.
- [ ] Commit.

## Task 5 — `RoundOutcome` logging schema

**Repo**: A. **AC**: AC-3. **ADR**: ADR-3.
**Files**: new `crates/larql-lql/src/selfplay/outcome.rs` (new module; wire into `lib.rs`)
**Interfaces**:
```rust
pub struct RoundOutcome {
    pub learned: bool,
    pub op_count: usize,
    pub reason: Option<String>, // required when learned=false and op_count==0
}
```

- [ ] Write a failing test: constructing a `RoundOutcome` with `op_count == 0` and `learned: true`
      is rejected (e.g. a `debug_assert!`/constructor invariant, or a `TryFrom` that errors) — the
      global constraint "zero-op rounds MUST NOT be logged as success" must be enforced in the
      type, not just by convention.
- [ ] Implement the struct + constructor invariant.
- [ ] Run, confirm green.
- [ ] Commit.

## Task 6 — Self-play driver skeleton

**Repo**: A. **AC**: AC-2, AC-7. **ADR**: ADR-4.
**Files**: new `crates/larql-cli/src/commands/selfplay/mod.rs` (+ wired into `Commands` enum in
`crates/larql-cli/src/main.rs`, sibling to `Repl`/`Lql` per the design doc's placement note)
**Interfaces**:
```rust
pub struct Task { pub id: String, pub reference_test_cmd: Option<Vec<String>> }
pub fn score_task(task: &Task) -> Option<bool> // None = excluded (AC-7), Some(exit_code == 0) = AC-2
```

- [ ] Write a failing test: `score_task` on a `Task` with `reference_test_cmd: None` returns `None`
      (excluded, per AC-7), never a fabricated pass/fail.
- [ ] Write a second failing test: `score_task` on a `Task` with a real (fixture) command returns
      `Some(true)`/`Some(false)` matching that command's actual exit code — the driver must not
      consult anything from the actor/model in computing this value (assert via a fixture where
      the "actor's own opinion" and the exit code deliberately disagree, confirming exit code wins).
- [ ] Implement `Task`/`score_task`.
- [ ] Run both, confirm green.
- [ ] Commit.

## Task 7 — Before/after behavioral eval around `COMPILE INTO VINDEX`

**Repo**: A. **AC**: AC-4. **ADR**: ADR-3.
**Files**: `crates/larql-cli/src/commands/selfplay/mod.rs` (extends T6), calling existing
`COMPILE`/`COMPACT` executor paths.
**Interfaces**: `pub fn compile_with_delta(task_subset: &[Task]) -> (RoundOutcome, f64 /* score delta */)`

- [ ] Write a failing test with a fixture vindex + fixture task subset: asserts the function scores
      the subset once before compiling, once after, and returns a delta that is `0.0` when the
      compile step is a no-op — i.e. this test must fail today because no such function exists,
      and must distinguish "compiled" from "compiled AND behavior changed" once implemented.
- [ ] Implement `compile_with_delta` calling T6's `score_task` before/after `COMPILE INTO VINDEX`.
- [ ] Run, confirm green.
- [ ] Commit.

## Task 8 — Curate a scoped coding-task suite (resolves Q2)

**Repo**: A. **AC**: AC-2, AC-7. **ADR**: ADR-4.
**Files**: new `selfplay-tasks/<task-id>/{task.md,solution_scaffold/,test.sh}` (one directory per
task; at least 5 tasks to start, e.g. "add two numbers function", "fix an off-by-one loop bound" —
deliberately scoped under SmolLM2-135M's capability per the design doc's SHOULD).
**Interfaces**: `test.sh` in each task dir MUST exit 0 on a correct solution and non-zero on an
incorrect one — this is the task's own machine-readable pass/fail contract.

- [x] Author >=5 task directories with real, runnable `test.sh` reference tests. **Done
      2026-07-15**: `selfplay-tasks/{add-two-numbers,is-even,max-of-two,reverse-string,
      fix-off-by-one-sum}/`, each with `task.md`, `solution_scaffold/solution.py` (stub or, for
      the last task, a deliberately buggy implementation — intentional task-shape variety per
      residual Q2), `reference_solution.py`, `test.sh`. Verified for real: all 5 scaffolds exit 1
      (4 via `NotImplementedError`, 1 via wrong output), all 5 reference solutions exit 0 when
      swapped in, scaffolds restored afterward (confirmed no leftover state).
- [ ] Write the suite-lint test (`crates/larql-cli/src/commands/selfplay/suite_lint.rs` or a shell
      test under `tests/`) asserting every directory under `selfplay-tasks/` has `task.md` +
      `test.sh` (executable) — still open; the manual verification above substitutes for now but
      the automated check is not yet written.
- [ ] Commit.

## Task 9 — Measure SmolLM2-135M resource fit (resolves Q5)

**Repo**: A. **AC**: none directly (feeds T15's viability). **ADR**: ADR-2 (must run inside the Fly
Machine boundary per C2 — extraction loads a local model).
**Files**: new `scripts/measure_extract_footprint.sh` (delegated to a subagent, executed inside the
Fly Machine from T10 — never run directly on the local host per this plan's containment rule).
**Interfaces**: script exits 0 and prints peak RSS in MB; exits 1 if peak RSS exceeds a declared
ceiling passed as `$1`.

- [ ] Write the script such that it deliberately fails first (e.g. run against a placeholder
      ceiling of `1` MB) to confirm the failure path actually triggers and reports correctly —
      this is the TDD-equivalent "red" step for a measurement script.
- [ ] Run for real inside the Fly Machine: `larql extract HuggingFaceTB/SmolLM2-135M --level
      inference` under the script, with a realistic ceiling derived from the Fly Machine's
      provisioned memory (not the local host's constrained 2.3GB — that number only motivated
      ADR-2, it is not the ceiling to test against once execution has actually moved off-host).
- [ ] Record the actual peak RSS as a new K entry in the research residual (empirical, dated).
- [ ] Commit the script (not the vindex artifact itself — too large for the repo).

## Task 10 — Local QEMU/KVM microVM provisioning

**Repo**: A. **AC**: AC-1. **ADR**: ADR-2 (revised — local KVM microVM, not Fly Machine; see ADR-2's
revision history in the design doc for why Fly/Docker were each tried and superseded).
**Files**: new `scripts/selfplay-vm/{selfplay-start.sh,cloud-init/user-data,cloud-init/meta-data}`;
`rdloop.toml` already updated (`qemu-system-x86_64`/`qemu-img`/`cloud-localds`/`kvm-ok` granted in
`[capabilities].subprocess`; `cloud.debian.org` granted in `[capabilities].network`).
**Interfaces**: `selfplay-start.sh` boots the guest (with a caller-supplied `runcmd` override or a
fixed default), tears it down, and exits with the guest command's exit code — same contract as
originally specified, mechanism changed from a remote Fly Machine to a local guest.

- [x] **Substantially done 2026-07-15**, ahead of formal TDD sequencing (delegated subagent
      smoke test, not yet wrapped in a reusable script/test harness — that packaging is the
      remaining checkbox below). Confirmed working end-to-end: `qemu-system-x86_64` 7.2.22 +
      `qemu-img`/`cloud-localds` installed (`sudo apt-get install qemu-system-x86 qemu-utils
      cloud-image-utils`, user-authorized); `kvm-ok` confirms real KVM acceleration; a Debian 12
      genericcloud qcow2 + 4G overlay + cloud-init seed booted with `-enable-kvm -m 768M -smp 1
      -cpu host -display none -serial file:console.log -no-reboot`, ran a `runcmd` writing a
      marker file, and shut down cleanly in 134s wall-clock (residual K29/K30 — full console
      transcript evidence).
- [ ] Write a failing test for the *reusable* form: `scripts/selfplay-vm/selfplay-start.sh --
      echo ok` before the script exists (fails: command not found) — the ad hoc smoke test above
      proved the mechanism; this step turns it into the plan's actual `selfplay-start.sh`
      interface (parameterized command, not a hardcoded marker-file `runcmd`).
- [ ] Implement `selfplay-start.sh` generalizing the proven boot command: build a fresh overlay
      per invocation (never reuse/mutate the base qcow2 or a prior overlay — avoids state leaking
      between self-play rounds), inject the caller's command via cloud-init `runcmd`, capture
      guest stdout via the serial log, propagate the guest command's exit code as the script's own
      exit code (need a convention for the guest to signal its real exit code back, e.g. writing
      it to a known marker file the host script reads after shutdown — cloud-init alone doesn't
      forward exit codes automatically, this is a real design detail to work out, not hand-waved).
- [ ] Run `selfplay-start.sh -- echo ok`, confirm exit 0 and `ok` observed.
- [ ] Run `selfplay-start.sh -- false`, confirm the script propagates a non-zero exit — proves the
      boundary reports real failures, not just successes (AC-6 at the infrastructure layer).
- [ ] Decide and implement the guest-to-host communication channel referenced in ADR-2's
      consequences (`vsock`/SSH over a host-only network device/9p-virtiofs) if the marker-file/
      serial-log approach above proves too limited for driving a real interactive `goose` session
      (vs. this smoke test's one-shot `runcmd`) — record the actual choice, don't leave it
      implicit.
- [ ] Commit (scripts + cloud-init templates; not the multi-hundred-MB base image, which stays
      outside the repo, documented via a download-on-demand step instead).

## Task 11 — `larql.rs` `LocalInferenceBackend` scaffold (spawn + error path)

**Repo**: B. **AC**: AC-1, AC-6. **ADR**: ADR-1 (revised — subprocess adapter, not in-process
`Provider`; see design doc's ADR-1 revision history for why).
**Files**: `crates/goose-local-inference/src/larql.rs` (new, sibling to `llamacpp.rs`/`mlx.rs`).
**Interfaces**: `struct LarqlBackend; impl LocalInferenceBackend for LarqlBackend { fn id(&self) ->
&'static str; fn load_model(&self, model_id: &str, resolved: &ResolvedModelPaths, settings:
&ModelSettings) -> Result<Box<dyn BackendLoadedModel>, ProviderError>; fn generate(&self, loaded:
&mut dyn BackendLoadedModel, request: LocalGenerationRequest<'_>) -> Result<(), ProviderError>; fn
available_memory_bytes(&self) -> u64; }` (exact trait per residual K34).

- [ ] Write a failing test asserting `LarqlBackend.id() == "larql"` (or the agreed backend id) —
      fails today (module doesn't exist).
- [ ] Write a second failing test: `load_model` against a deliberately invalid/missing vindex path
      returns `Err(ProviderError::...)` rather than panicking or hanging — this is AC-6's test. Also
      cover: the spawned `larql chat` child exiting immediately (nonzero exit before producing any
      output) is surfaced as an `Err`, not silently treated as "loaded."
- [ ] Implement the scaffold: `load_model` spawns `larql chat <resolved_vindex_path>` as a child
      process with piped stdin/stdout/stderr (`std::process::Command`, following the piped-I/O
      pattern OpenFang's `process_manager.rs` uses — residual K24 — for lifecycle bookkeeping, not
      its literal API), stores the child handle in the returned `BackendLoadedModel`; `generate()`
      can `todo!()`/return `NotImplemented` at this stage (T12 fills it in).
- [ ] Run both tests, confirm green.
- [ ] Commit on a new branch in the `metavacua/goose` fork (per the user's authorized branch/PR
      scope) — do not push to `main` directly.

## Task 12 — `larql.rs` real `generate()` implementation (drives the spawned `larql chat` child)

**Repo**: B (drives Repo A's existing, unmodified `larql chat`/`larql run` process — no new Repo A
code required for this task specifically, per residual K35's confirmed stdin/stdout framing).
**AC**: AC-6. **ADR**: ADR-1.
**Files**: same as T11, `generate()` body implemented.

- [ ] Write a failing integration test: against a tiny fixture vindex (checked into the goose
      fork's test fixtures, or referenced from Repo A's `test_fixtures` behind the `test-utils`
      feature per `AGENTS.md`), calling `generate()` with a trivial prompt writes the expected line
      to the child's stdin and reads a non-empty response from its stdout before the turn-boundary
      signal — fails today (`generate()` isn't implemented).
- [ ] Implement `generate()`: write the prompt (translated from `LocalGenerationRequest`'s
      `messages`/`system`) as a line to the child's stdin per `run_chat`'s confirmed framing
      (residual K35 — one line in, response streamed to stdout, boundary signaled on stderr's next
      `"> "`); read stdout, push chunks onto the `StreamSender` as they arrive. If stderr-based
      turn-boundary detection proves too fragile in practice, make the small additive change to
      `run_chat` (Repo A) to also emit an explicit stdout marker — if this Repo-A change is needed,
      it becomes its own TDD sub-step here (failing test on the marker's presence, then implement
      it in `run_cmd.rs`), not a silent assumption.
- [ ] Run, confirm green.
- [ ] Commit.

## Task 13 — Backend registration

**Repo**: B. **AC**: AC-1. **ADR**: ADR-1.
**Files**: `crates/goose-local-inference/src/lib.rs`'s `InferenceRuntime::get_or_init()` (residual
K34 — where `llamacpp`/`mlx` are inserted into the `backends` HashMap today).

- [ ] Write a failing test: after `InferenceRuntime::get_or_init()`, assert a `"larql"` key is
      present in the runtime's backend map (or equivalent public accessor — check what's actually
      exposed for testing at implementation time).
- [ ] Implement the `backends.insert(LARQL_BACKEND_ID, Arc::new(LarqlBackend::new()) as Arc<dyn
      LocalInferenceBackend>)` call site.
- [ ] Run, confirm green; run the fork's own `cargo test -p goose-local-inference` (narrower than
      the whole `goose` workspace) to confirm no regression.
- [ ] Open a PR in `metavacua/goose` from the branch (per user-authorized scope) — do not merge
      without separate review.

## Task 14 — Liveness/heartbeat monitor for delegated `goose` sessions

**Repo**: A. **AC**: AC-5. **ADR**: none new (closes residual N4/N5 directly).
**Files**: new `scripts/selfplay_watchdog.sh` (or a small Rust binary if a shell script proves too
fragile for the timeout-vs-liveness distinction needed).
**Interfaces**: given a PID/machine-id and a timeout, exits 0 if a liveness signal (heartbeat file
touch, or Fly Machine status API poll) was observed within the window, exits 1 (fault) otherwise.

- [ ] Write a failing test: simulate a hung process (a script that never touches its heartbeat
      file) and assert the watchdog exits 1 after the declared timeout — must actually wait out a
      short test timeout (e.g. 2s) to prove the fault path fires, not just that the code compiles.
- [ ] Write a second test: simulate a live process (touches its heartbeat file periodically) and
      assert the watchdog exits 0 and does not fire early.
- [ ] Implement the watchdog.
- [ ] Run both, confirm green.
- [ ] Commit.

## Task 15 — End-to-end demonstration (Phase 3/4 integration point)

**Repo**: A (orchestrates both). **AC**: all of AC-1 through AC-7 jointly. **ADR**: all four.
**Files**: none new — this task wires T1-T14's outputs together and is verified by execution, not
new source.

- [ ] Inside the Fly Machine (T10), with the registered provider (T13) and the curated suite (T8),
      run one full self-play round: Goose (using the `larql` provider) attempts one task from
      `selfplay-tasks/`, T6 scores it via the task's own `test.sh`, and on PASS the driver proposes
      an `INSERT`/`COMPOSE` patch.
- [ ] Confirm the round's `RoundOutcome` (T5) is logged correctly for both a PASS-with-effective-
      edit case and a case that exercises T1's fixed capacity-collision path (`learned=false,
      reason=...`, not silently "success").
- [ ] Run `compile_with_delta` (T7) at the end of the round and record the actual before/after
      score delta — a delta of exactly `0.0` here is a valid, reportable outcome (per N3), not a
      failure to hide.
- [ ] Confirm the watchdog (T14) observed the session's full duration with no false fault.
- [ ] Report full verbatim output (session transcript, `RoundOutcome`, delta) as the plan's
      Definition-of-Done evidence — this is the actual "demonstration" the task set out to produce.

---

## Task 16 — CLI-level CPU governor (issue #185)

**Repo**: A. **AC**: none directly — closes residual K33/#185; de-risks every other task that runs
real `larql` commands (T9, T10's guest workload, T15). **ADR**: ADR-5.
**Files**: `crates/larql-cli/src/main.rs` (startup, before subcommand dispatch),
`crates/larql-compute/src/cpu/spin_pool.rs` (or wherever `spin_pool`'s sizing logic actually lives
— confirm exact path at implementation time).
**Interfaces**: a `main()`-startup call installing a global rayon thread-pool cap; `spin_pool`'s
default sizing changed to respect that cap rather than `rayon::current_num_threads()` pre-cap.

- [ ] Write a failing test asserting: when `RAYON_NUM_THREADS` is unset and the host reports `N`
      cores, the installed global rayon pool size is `max(1, N - 1)` — fails today (no such
      installation exists, per #185's problem statement).
- [ ] Write a second failing test asserting `spin_pool`'s default sizing does not exceed the capped
      pool size (i.e. it reads the capped thread count, not raw `num_cpus`).
- [ ] Implement the `main()`-startup rayon-pool cap (skip installation, i.e. respect the user's
      explicit choice, when `RAYON_NUM_THREADS` is already set — per #185's "explicit opt-outs
      remain, not the only path to safety").
- [ ] Implement the `spin_pool` sizing fix.
- [ ] Run both tests, confirm green; run `cargo test -p larql-cli -p larql-compute` for regressions.
- [ ] Commit.

## Task 17 — CLI-level memory governor (issue #211)

**Repo**: A. **AC**: none directly — closes residual K33/#211; de-risks T9 specifically (T9's own
`on_fail` contingency, "peak RSS during extraction exceeds the declared ceiling," becomes a
graceful abort with a diagnostic instead of a bare OOM-kill once this lands). **ADR**: ADR-5.
**Files**: `crates/larql-cli/src/main.rs` (startup watchdog installation), new
`crates/larql-cli/src/resource_governor.rs` (or similar — houses the anon-RSS polling + ceiling
logic).
**Interfaces**: `pub fn install_memory_governor(ceiling_fraction: f64) -> GovernorHandle` (or
equivalent), reading `/proc/self/status` `VmRSS`, aborting with a clear diagnostic when RSS exceeds
`available_ram_at_startup * ceiling_fraction` (default `0.85` per #211).

- [ ] Write a failing test: a harness that allocates memory in a loop until it should breach a
      deliberately tiny test ceiling (e.g. a few MB, not real system RAM) asserts the governor
      aborts the process (or returns a breach signal the test harness intercepts before an actual
      process exit, depending on how the governor is structured to stay testable) with the
      expected diagnostic message — fails today (no governor exists).
- [ ] Write a second failing test: confirms LARQL's own mmap'd vindex file-backed pages do NOT
      count toward the anon-RSS ceiling (i.e. the governor reads `VmRSS`'s anon-specific component,
      not total RSS, per #211's "observed OOMs were anon-rss only, file-rss ~0" — this test must
      actually mmap a file and confirm it doesn't move the tracked figure, not just assert the
      field name is right).
- [ ] Implement the watchdog (periodic poll on a background thread, or checked at allocation
      checkpoints — pick per #211's suggested ~100ms cadence or an equivalent, document the choice)
      plus the `LARQL_MEMORY_LIMIT` env-var opt-out named in #211's acceptance criteria.
- [ ] Run both tests, confirm green; run `cargo test -p larql-cli` for regressions.
- [ ] Commit.

---

## Requirements Traceability Matrix

| AC | Task(s) | Test(s) |
|---|---|---|
| AC-1 (execution boundary, orchestrator-killable) | T10, T11, T15 | T10's `selfplay-start.sh -- echo ok` / `-- false` smoke tests (mechanism proven 2026-07-15, script packaging still open); T15's E2E run |
| AC-2 (evaluator is sole pass/fail signal) | T6, T15 | T6's `score_task` disagreement-fixture test |
| AC-3 (zero-op rounds never reported as success) | T1, T3, T4, T5, T15 | T1/T3/T4's regression tests; T5's constructor-invariant test |
| AC-4 (before/after delta on compile) | T2, T7, T15 | T2's scaling-formula test; T7's delta test |
| AC-5 (liveness timeout -> fault, not silent hang) | T14, T15 | T14's hung/live simulation tests |
| AC-6 (`ProviderError` surfaced, not panic) | T11, T12 | T11's invalid-path error test |
| AC-7 (tasks without reference suite excluded) | T6, T8, T15 | T6's `None`-returning test; T8's suite-lint test |
| *(no AC — infrastructure hardening, ADR-5)* | T16, T17 | T16's pool-size/spin_pool tests; T17's ceiling-breach/anon-rss tests |

## Self-review

- Every task above has a write-failing-test step before its implementation step. ✔
- Every task cites at least one `AC-n` and the design doc's `ADR-n` that motivates it, except T0,
  T9/T14 (explicitly "infrastructure prerequisite, no direct AC" / "closes a residual N entry
  rather than an AC"), and T16/T17 (explicitly "no AC — cites ADR-5 and a specific pre-existing
  GitHub issue instead, since this work predates and is broader than this design's own AC set") —
  stated plainly rather than a fabricated citation. ✔
- No task says "TBD" — T11/T12's earlier open question (new crate vs. extending
  `goose-local-inference`) is now resolved by reading the actual code (K34: extend, via the
  existing `LocalInferenceBackend` HashMap seam), but T12 still names one remaining open detail
  (stderr- vs. stdout-based turn-boundary detection) and requires a TDD sub-step if it needs a
  small Repo-A change, rather than silently assuming it away; T10 similarly names its own
  remaining open design detail (guest exit-code propagation convention). ✔
- Q2 and Q5 from the residual are scheduled as T8 (done) and T9 respectively, not silently assumed. ✔
