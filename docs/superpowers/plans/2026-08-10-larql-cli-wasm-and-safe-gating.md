# larql-cli wasm32v1-none + forbid-unsafe CI Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a GitHub Actions CI job on a dedicated branch/PR that empirically discovers, via real compiler errors, (a) where `larql-cli` and its path dependencies fail to build for the `wasm32v1-none` target, and (b) where `larql-cli`'s own source contains `unsafe` code — so those errors can drive future source-level gating work (not part of this plan).

**Architecture:** One new workflow file, `.github/workflows/larql-cli-gating.yml`, with two jobs. `wasm32v1-none` matrix-builds `larql-cli` and its 10 path-dependency crates, across 3 host OSes and 2 feature sets, targeting `wasm32v1-none` with `cargo build` (full codegen). `forbid-unsafe` depends on `wasm32v1-none` succeeding (`needs:`) and matrix-builds `larql-cli` natively, across the same 3 OSes and 2 feature sets, with `#![forbid(unsafe_code)]` added to its crate root so any `unsafe` in `larql-cli`'s own source is a hard compile error. Both jobs are additive CI surface only — no source code is changed to fix anything in this plan; the two known `unsafe` blocks found in `larql-cli` are left in place so the `forbid-unsafe` job fails for real, producing genuine signal.

**Tech Stack:** Rust (stable toolchain, `wasm32v1-none` target already stabilized), Cargo workspace, GitHub Actions (`dtolnay/rust-toolchain`, `actions/checkout@v6`), `gh` CLI for branch/PR operations.

## Global Constraints

- Base branch: `origin/fix/metal-zero-copy-missing-macos-cfg` (tip commit `a4239202`, = upstream `chrishayuk/larql:main` as of 2026-08-10 + PR #244's 11-commit metal fix), fetched fresh before branching.
- New branch name: `gating/larql-cli-wasm-and-safe`, pushed to `origin` (`metavacua/larql-to-sparql`).
- PR opens against `metavacua/larql-to-sparql:main` (not upstream `chrishayuk/larql`).
- Repo working copy: `/home/metavacua/larql-to-sparql-gating-2026-08-10` — a **fresh, dedicated clone** of `metavacua/larql-to-sparql`, not a reused/pre-existing local clone. `origin` = `https://github.com/metavacua/larql-to-sparql.git`, `upstream` = `https://github.com/chrishayuk/larql.git`.
- **`gh` default-repo pinning is required and easy to get wrong**: `gh`'s remote-resolution heuristic prefers a remote literally named `upstream` over `origin` when both exist, so a bare `gh repo view`/`gh pr create` with no `--repo` flag silently resolves to `chrishayuk/larql`, not the fork (verified empirically during planning: `gh repo view` with no flag returned `chrishayuk/larql`). Every `gh` invocation below carries an explicit `--repo metavacua/larql-to-sparql` for this reason — never drop it, and never rely on `gh`'s default resolution in this repo. Additionally, `gh repo set-default metavacua/larql-to-sparql` must be run once per clone (Task 1) as a second layer of defense.
- `wasm32v1-none` "clears" means `cargo build --target wasm32v1-none` (full codegen), not `cargo check`.
- `forbid-unsafe` mechanism is the standard crate-local `#![forbid(unsafe_code)]` attribute in `crates/larql-cli/src/main.rs` — it only lints `larql-cli`'s own AST, never dependency crates (workspace or third-party), by Rust's normal per-crate lint scoping.
- `wasm32v1-none` job matrixes over all 11 crates: `larql-cli`, `larql-core`, `larql-compute`, `larql-compute-metal`, `larql-inference`, `larql-kv`, `larql-models`, `larql-lql`, `larql-vindex`, `larql-vindex-spec`, `larql-factory`.
- `forbid-unsafe` job matrixes over `larql-cli` only (no per-crate matrix) — that's the one crate getting the attribute.
- Both jobs matrix over `os: [ubuntu-latest, windows-latest, macos-14]` and `features: [default, no-default-features]`, with `fail-fast: false`.
- `forbid-unsafe` declares `needs: wasm32v1-none` so it only runs once every `wasm32v1-none` matrix leg succeeds.
- Workflow declares `permissions: contents: read` (least privilege — the jobs only build, never write) and a `concurrency` group with `cancel-in-progress: true` keyed on `${{ github.workflow }}-${{ github.ref }}`.
- Runner cost is explicitly not a constraint (public repo, free-tier Actions minutes apply regardless of OS) — do not trim the matrix for cost reasons.
- Both jobs (and their host-tool install steps for OpenBLAS/protoc) mirror the existing `.github/workflows/larql-cli.yml` conventions for parity, so failures reflect real wasm/unsafe issues rather than missing host tooling.
- Scope boundary: this plan produces the CI job and the `forbid(unsafe_code)` attribute only. It does **not** fix any resulting compile errors or add any `cfg`/feature gates to make the jobs pass — that is explicitly future work driven by this job's output.
- The single commit on this branch must contain both the workflow file and the `main.rs` attribute change together (not split across commits).
- **No local builds, checks, or test runs of this Rust workspace on the dev machine, ever, in any task.** GitHub-hosted runners (the two CI jobs) are the sole place `larql-cli` or any dependency crate gets compiled. Local work is limited to source edits, static config linting that doesn't invoke `cargo` (`yamllint`, `actionlint`), and `git`/`gh` plumbing. If a task feels like it needs a local `cargo build`/`check`/`test` to be confident, that's a signal to commit and push (or update the open PR) so the real CI run answers the question empirically — not to run the build locally, and not to pause and ask the human. Uncertainty resolves by pushing and reading the CI result, same as every other question this whole effort exists to answer empirically.

---

### Task 1: Create the dedicated branch from the correct base — COMPLETE (done during planning setup)

**Status:** Already executed and verified during plan setup, not left for a dispatched implementer. Recorded here for the record and so later tasks can rely on the resulting state.

**Files:** none (git operation only)

**Interfaces:**
- Produces: a fresh clone at `/home/metavacua/larql-to-sparql-gating-2026-08-10`, with a local and remote-tracked branch `gating/larql-cli-wasm-and-safe` checked out, used as the base for Tasks 2-4.

- [x] **Step 1: Clone fresh, add both remotes, pin `gh`'s default repo**

```bash
cd /home/metavacua
git clone https://github.com/metavacua/larql-to-sparql.git larql-to-sparql-gating-2026-08-10
cd larql-to-sparql-gating-2026-08-10
git remote add upstream https://github.com/chrishayuk/larql.git
gh repo set-default metavacua/larql-to-sparql
```

Verified: `git remote -v` shows `origin` = `metavacua/larql-to-sparql.git`, `upstream` = `chrishayuk/larql.git`; `gh repo view --json nameWithOwner --jq '.nameWithOwner'` (no `--repo` flag) returns `metavacua/larql-to-sparql`; `.git/config`'s `[remote "origin"]` block carries `gh-resolved = base`.

- [x] **Step 2: Fetch the base branch fresh**

```bash
git fetch origin fix/metal-zero-copy-missing-macos-cfg
```

Verified: fetch completed, no errors.

- [x] **Step 3: Create and check out the new branch from the fetched base**

```bash
git checkout -b gating/larql-cli-wasm-and-safe origin/fix/metal-zero-copy-missing-macos-cfg
```

Verified: `Switched to a new branch 'gating/larql-cli-wasm-and-safe'`.

- [x] **Step 4: Verify the branch point**

```bash
git log -1 --oneline
git merge-base --is-ancestor origin/fix/metal-zero-copy-missing-macos-cfg HEAD && echo "base OK"
```

Verified: top commit is `a4239202 test(metal): cover the Q6_K grouped dispatch arm`, `base OK` printed, working tree clean (`git status -sb` showed only the branch/tracking line).

No commit in this task — nothing has changed yet. Tasks 2+ start from this state.

---

### Task 2: Add `#![forbid(unsafe_code)]` to `larql-cli`

**Files:**
- Modify: `crates/larql-cli/src/main.rs:1`

**Interfaces:**
- Consumes: branch created in Task 1.
- Produces: `main.rs` with the forbid attribute in place. No local build/verification of this change — GitHub Actions (the `forbid-unsafe` job added in Task 3) is the sole place this gets compiled and checked; the dev machine only makes the source edit. This is deliberate: builds run on GitHub-hosted runners only, never locally, to avoid loading this machine. Task 5 observes the real failure (expected at `crates/larql-cli/src/commands/extraction/trajectory_trace_cmd.rs:707` and `crates/larql-cli/src/commands/extraction/walk_cmd.rs:10`, two known pre-existing `unsafe` blocks left untouched by this task) once CI actually runs it.

- [ ] **Step 1: Insert the attribute as the first line of `main.rs`**

Current first two lines of `crates/larql-cli/src/main.rs` are:

```rust
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::type_complexity)]
```

Change to:

```rust
#![forbid(unsafe_code)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::type_complexity)]
```

(Inner attributes can appear in any order relative to each other; putting `forbid` first is just convention. Everything else in the file is unchanged.)

- [ ] **Step 2: Confirm the edit by inspection, not by building**

```bash
head -5 crates/larql-cli/src/main.rs
```

Expected: `#![forbid(unsafe_code)]` is line 1, the two pre-existing `#![allow(...)]` lines follow unchanged, nothing else in the file differs from Task 1's checkout. Do **not** run `cargo build`, `cargo check`, or any other compiler invocation for this task — no local build of this crate or its dependency graph happens on this machine at any point in this plan. Confidence that the edit actually compiles as intended comes from Task 4's push and Task 5's real CI result, not from a local build.

Do not commit yet — `main.rs` and the workflow file (Task 3) land together in Task 4's single commit.

---

### Task 3: Add the CI workflow file

**Files:**
- Create: `.github/workflows/larql-cli-gating.yml`

**Interfaces:**
- Consumes: branch from Task 1 (workflow's `push.branches` trigger names it explicitly).
- Produces: the two-job workflow (`wasm32v1-none`, `forbid-unsafe`) that Task 4 pushes and Task 5 observes.

- [ ] **Step 1: Write the workflow file**

```yaml
name: larql-cli-gating

on:
  push:
    branches: [gating/larql-cli-wasm-and-safe]
  pull_request:
    branches: [main]
  workflow_dispatch: {}

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  wasm32v1-none:
    name: wasm32v1-none / ${{ matrix.crate }} / ${{ matrix.os }} / ${{ matrix.features }}
    runs-on: ${{ matrix.os }}
    timeout-minutes: 35
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-14]
        crate:
          - larql-cli
          - larql-core
          - larql-compute
          - larql-compute-metal
          - larql-inference
          - larql-kv
          - larql-models
          - larql-lql
          - larql-vindex
          - larql-vindex-spec
          - larql-factory
        features: [default, no-default-features]
    steps:
      - uses: actions/checkout@v6

      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32v1-none

      - name: Install OpenBLAS (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y libopenblas-dev pkg-config

      - name: Install OpenBLAS (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        run: |
          $vcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
          if (-not $vcpkgRoot) { $vcpkgRoot = "C:\vcpkg" }
          "VCPKG_ROOT=$vcpkgRoot" | Out-File -FilePath $env:GITHUB_ENV -Append
          & "$vcpkgRoot\vcpkg.exe" install openblas:x64-windows

      - name: Install protoc (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        run: choco install protoc -y --no-progress

      - name: cargo build --target wasm32v1-none
        shell: bash
        run: |
          if [ "${{ matrix.features }}" = "no-default-features" ]; then
            cargo build --target wasm32v1-none -p ${{ matrix.crate }} --no-default-features
          else
            cargo build --target wasm32v1-none -p ${{ matrix.crate }}
          fi

  forbid-unsafe:
    name: forbid-unsafe / larql-cli / ${{ matrix.os }} / ${{ matrix.features }}
    needs: wasm32v1-none
    runs-on: ${{ matrix.os }}
    timeout-minutes: 35
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-14]
        features: [default, no-default-features]
    steps:
      - uses: actions/checkout@v6

      - uses: dtolnay/rust-toolchain@stable

      - name: Install OpenBLAS (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y libopenblas-dev pkg-config

      - name: Install OpenBLAS (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        run: |
          $vcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
          if (-not $vcpkgRoot) { $vcpkgRoot = "C:\vcpkg" }
          "VCPKG_ROOT=$vcpkgRoot" | Out-File -FilePath $env:GITHUB_ENV -Append
          & "$vcpkgRoot\vcpkg.exe" install openblas:x64-windows

      - name: Install protoc (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        run: choco install protoc -y --no-progress

      - name: cargo build (forbid unsafe_code)
        shell: bash
        run: |
          if [ "${{ matrix.features }}" = "no-default-features" ]; then
            cargo build -p larql-cli --no-default-features
          else
            cargo build -p larql-cli
          fi
```

- [ ] **Step 2: Validate with yamllint and actionlint**

```bash
yamllint .github/workflows/larql-cli-gating.yml
actionlint .github/workflows/larql-cli-gating.yml
```

Expected: `actionlint` reports zero findings — it specifically validates GitHub Actions semantics (job `needs:` references resolve, matrix syntax, expression syntax, shell script issues inside `run:` blocks via embedded shellcheck), which a plain YAML parse would miss. `yamllint` will report the same baseline warnings/errors the existing `larql-cli.yml` already produces under default rules with no repo `.yamllint` config present (missing `---` document start, `on:` flagged by the `truthy` rule, a couple lines >80 chars) — these are pre-existing repo-wide noise, not real problems (verified by running `yamllint .github/workflows/larql-cli.yml` for comparison), so do not "fix" them into a style the rest of the repo doesn't use. Only treat `yamllint` as a real signal if it reports something *not* also present in `larql-cli.yml`'s output.

- [ ] **Step 3: Sanity-check the job graph by eye**

Confirm in the written file: `forbid-unsafe` has `needs: wasm32v1-none` (exact job id match), both jobs have `fail-fast: false`, the top-level `permissions:` and `concurrency:` blocks are present exactly once each (workflow-level, not per-job).

Do not commit yet — combined with Task 2's change in Task 4.

---

### Task 4: Commit, push, and open the PR

**Files:** none new (commits the results of Tasks 2 and 3)

**Interfaces:**
- Consumes: `crates/larql-cli/src/main.rs` (Task 2), `.github/workflows/larql-cli-gating.yml` (Task 3).
- Produces: pushed branch `gating/larql-cli-wasm-and-safe` on `origin`, and an open PR against `metavacua/larql-to-sparql:main`.

- [ ] **Step 1: Stage exactly the two changed files**

```bash
cd /home/metavacua/larql-to-sparql-gating-2026-08-10
git status -s
git add crates/larql-cli/src/main.rs .github/workflows/larql-cli-gating.yml
git status -s
```

Expected: the second `git status -s` shows only these two files staged (`M crates/larql-cli/src/main.rs`, `A .github/workflows/larql-cli-gating.yml`). If anything else is staged, unstage it before continuing.

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
ci(larql-cli): add wasm32v1-none + forbid-unsafe gating job

Phase 1 of empirical portability/safety boundary discovery for
larql-cli. Two jobs, gated with needs: so forbid-unsafe only runs
once wasm32v1-none is fully green:

- wasm32v1-none: cargo build --target wasm32v1-none across larql-cli
  and its 10 path-dependency crates, os x features matrix. Expected
  to start red — the compiler errors are the deliverable, driving
  follow-up cfg/feature gating (not done in this commit).
- forbid-unsafe: #![forbid(unsafe_code)] added to larql-cli's crate
  root (crate-local, doesn't affect dependency crates). Two known
  unsafe blocks (trajectory_trace_cmd.rs:707, walk_cmd.rs:10) will
  fail this job by design; left unfixed pending follow-up.
EOF
)"
```

Expected: commit succeeds, one commit ahead of the base.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin gating/larql-cli-wasm-and-safe
```

Expected: push succeeds, remote branch created.

- [ ] **Step 4: Open the PR**

```bash
gh pr create \
  --repo metavacua/larql-to-sparql \
  --base main \
  --head gating/larql-cli-wasm-and-safe \
  --title "ci(larql-cli): wasm32v1-none + forbid-unsafe gating (Phase 1: discovery)" \
  --body "$(cat <<'EOF'
## Summary

Adds `.github/workflows/larql-cli-gating.yml` with two jobs to empirically
discover, via real GitHub Actions compiler errors, where `larql-cli` (and
its path dependencies) break under `wasm32v1-none`, and where `larql-cli`'s
own source uses `unsafe`.

- `wasm32v1-none`: `cargo build --target wasm32v1-none`, matrixed over
  `larql-cli` + its 10 path-dependency crates x 3 OSes x 2 feature sets
  (66 legs, `fail-fast: false`). **Expected to start red.**
- `forbid-unsafe`: `needs: wasm32v1-none`, so it only runs once every
  `wasm32v1-none` leg is green. Adds `#![forbid(unsafe_code)]` to
  `larql-cli/src/main.rs`, matrixed over 3 OSes x 2 feature sets (6 legs).
  Two known `unsafe` blocks
  (`commands/extraction/trajectory_trace_cmd.rs:707`,
  `commands/extraction/walk_cmd.rs:10`) will fail it by design.

No source-level gating (cfg splits, feature flags, unsafe removal) is
included in this PR — that's deliberate follow-up work once the CI output
above identifies exactly what needs gating. This PR's job is to produce
that error signal, not to act on it yet.

## Test plan

- [x] YAML validated locally (`python3 -c "import yaml; yaml.safe_load(...)"`).
- [x] Local `cargo build -p larql-cli` (both feature sets) reproduces the
      two expected forbid-unsafe failures.
- [ ] Confirm `wasm32v1-none` job runs (red expected) across the full matrix.
- [ ] Confirm `forbid-unsafe` job stays queued/blocked until `wasm32v1-none`
      passes (needs: dependency working as intended).
EOF
)"
```

Expected: PR URL printed.

- [ ] **Step 5: Verify the PR exists and record its number**

```bash
gh pr view --repo metavacua/larql-to-sparql gating/larql-cli-wasm-and-safe --json number,url,state
```

Expected: JSON with `"state": "OPEN"` and a valid `number`/`url`. Record the number for Task 5.

---

### Task 5: Observe the initial CI run and report status

**Files:** none

**Interfaces:**
- Consumes: PR number from Task 4, Step 5.
- Produces: a status report (posted back to the user, not a file) of which matrix legs ran, which failed, and confirmation that `forbid-unsafe` correctly stayed gated behind `wasm32v1-none` via `needs:`.

- [ ] **Step 1: List the triggered workflow runs for the branch**

```bash
gh run list --repo metavacua/larql-to-sparql --branch gating/larql-cli-wasm-and-safe --workflow larql-cli-gating.yml --limit 5
```

Expected: at least one run listed, status `in_progress` or `completed`.

- [ ] **Step 2: Check current PR check status (single snapshot, not a poll loop)**

```bash
gh pr checks --repo metavacua/larql-to-sparql <PR-number> 2>&1 || true
```

Expected: a mix of pass/fail/pending across the 66 `wasm32v1-none` legs, and the `forbid-unsafe` legs showing as queued/skipped (not yet started) if `wasm32v1-none` hasn't fully completed.

- [ ] **Step 3: Report back to the user**

Summarize: PR URL, how many `wasm32v1-none` legs have reported so far and their pass/fail split, whether `forbid-unsafe` has started (it shouldn't have, until all 66 `wasm32v1-none` legs are green), and note that full completion (especially `macos-14`/`windows-latest` legs) may take longer than this session watched — point the user to the PR URL to watch the rest land. Do not attempt to fix any failures — that is out of scope for this plan.

---

### Task 6: Gate the other ~15 CI workflows to skip on this branch (temporary, reversible, branch-scoped)

**Why:** This PR's base (`main`) is 402 commits behind our branch (a pre-existing fork-staleness issue, left as-is per explicit instruction — not fixed by this task). Because GitHub's `paths:` trigger filters match against the PR's *entire* changed-file list relative to its base, and that list spans hundreds of unrelated commits, essentially every per-crate workflow's `paths:` filter matches something in that huge diff — so ~15 unrelated workflows fire in full on every push to this branch, on top of the two `larql-cli-gating.yml` jobs we actually care about. This wastes runner queue time and, for `larql-cli.yml` specifically, produces a real failure: `#![forbid(unsafe_code)]` (Task 2) breaks its `test`/`coverage` jobs, since they compile the same `larql-cli` crate root. We want these workflows to keep running normally for every other branch/PR in the repo, and to resume running on this branch once we're done iterating — just not fire every push while this branch is under active, fast-moving development.

**Files (add one line to every job in each):**
- Modify: `.github/workflows/bench-regress.yml` (1 job: `bench`)
- Modify: `.github/workflows/larql-boundary.yml` (2 jobs)
- Modify: `.github/workflows/larql-cli.yml` (2 jobs: `test`, `coverage`)
- Modify: `.github/workflows/larql-compute-metal.yml` (1 job)
- Modify: `.github/workflows/larql-compute.yml` (2 jobs)
- Modify: `.github/workflows/larql-core.yml` (2 jobs)
- Modify: `.github/workflows/larql-demos.yml` (1 job)
- Modify: `.github/workflows/larql-factory.yml` (2 jobs)
- Modify: `.github/workflows/larql-inference.yml` (2 jobs)
- Modify: `.github/workflows/larql-kv.yml` (2 jobs)
- Modify: `.github/workflows/larql-lql.yml` (2 jobs)
- Modify: `.github/workflows/larql-models.yml` (2 jobs)
- Modify: `.github/workflows/larql-server.yml` (2 jobs)
- Modify: `.github/workflows/larql-vindex.yml` (2 jobs)
- Modify: `.github/workflows/quality.yml` (6 jobs)
- Modify: `.github/workflows/shannon-verify.yml` (1 job)
- **Do NOT touch:** `.github/workflows/release.yml` (triggers on tags only, never fires on this branch/PR in the first place) or `.github/workflows/larql-cli-gating.yml` (the workflow we want to keep running).

Total: 32 jobs across 16 files.

**Interfaces:**
- Consumes: nothing from earlier tasks (independent, additive change to unrelated workflow files).
- Produces: all 32 jobs gain `if: github.head_ref != 'gating/larql-cli-wasm-and-safe'`. `github.head_ref` is only populated on `pull_request`-triggered runs and holds the PR's source branch name; it's empty (and thus not equal to our branch name) on `push`-triggered runs, which is fine since every one of these workflows already restricts `push:` to `branches: [main]` and therefore never fires on push to a feature branch anyway. The condition only needs to guard the `pull_request` case.

- [ ] **Step 1: Add the skip condition to every job in every listed file**

For each job in each of the 16 files listed above, insert a new line `if: github.head_ref != 'gating/larql-cli-wasm-and-safe'` immediately after that job's `runs-on:` line. Example (from `larql-cli.yml`'s `test` job):

Before:
```yaml
  test:
    name: test - ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    timeout-minutes: 35
```

After:
```yaml
  test:
    name: test - ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    if: github.head_ref != 'gating/larql-cli-wasm-and-safe'
    timeout-minutes: 35
```

Apply the identical pattern to every job in every file on the list — same exact condition string, verbatim, every time. Do not vary the condition per file/job. Do not add the condition to `release.yml` or `larql-cli-gating.yml`.

- [ ] **Step 2: Validate every modified file**

```bash
for f in bench-regress larql-boundary larql-cli larql-compute-metal larql-compute larql-core larql-demos larql-factory larql-inference larql-kv larql-lql larql-models larql-server larql-vindex quality shannon-verify; do
  echo "--- $f.yml ---"
  actionlint ".github/workflows/$f.yml"
done
```

Expected: zero output (no findings) for every file. Any output is a real problem — fix it before continuing.

- [ ] **Step 3: Verify exactly 32 insertions, one per job, nothing else changed**

```bash
git diff --stat
grep -rc "if: github.head_ref != 'gating/larql-cli-wasm-and-safe'" .github/workflows/*.yml | grep -v ':0'
```

Expected: `git diff --stat` shows only the 16 listed files, each with N insertions matching its job count and zero deletions. The `grep -c` sum across all files must equal 32. `release.yml` and `larql-cli-gating.yml` must not appear in either output.

No local `cargo` command at any point in this task — it's pure YAML editing and static linting.

- [ ] **Step 4: Commit and push (separate commit from Task 4's — this is follow-up work after the PR was already opened)**

```bash
git add .github/workflows/bench-regress.yml .github/workflows/larql-boundary.yml .github/workflows/larql-cli.yml .github/workflows/larql-compute-metal.yml .github/workflows/larql-compute.yml .github/workflows/larql-core.yml .github/workflows/larql-demos.yml .github/workflows/larql-factory.yml .github/workflows/larql-inference.yml .github/workflows/larql-kv.yml .github/workflows/larql-lql.yml .github/workflows/larql-models.yml .github/workflows/larql-server.yml .github/workflows/larql-vindex.yml .github/workflows/quality.yml .github/workflows/shannon-verify.yml
git status -s
git commit -m "$(cat <<'EOF'
ci: skip unrelated workflows on gating/larql-cli-wasm-and-safe

This branch's PR base (main) is 402 commits behind, so every per-crate
workflow's paths: filter matches somewhere in the resulting huge diff
and all ~15 unrelated workflows fire on every push here, queuing runner
time behind the two larql-cli-gating jobs we actually care about. Worse,
larql-cli.yml's test/coverage jobs now fail for real, since Task 2's
#![forbid(unsafe_code)] breaks the same larql-cli crate root they compile.

Adds `if: github.head_ref != 'gating/larql-cli-wasm-and-safe'` to all 32
jobs across these 16 files so they keep running normally for every other
branch/PR, and resume running on this branch once this condition is
removed later.
EOF
)"
git push
```

Expected: push succeeds (fast-forward, since this is a new commit on top of Task 4's).

---

### Task 7: Systematic wasm32v1-none + forbid-unsafe gating campaign (goal set mid-session)

**Goal, verbatim from the user:** "use the CI logs, worktrees, branches, and PRs as necessary to systematically gate all the code involved in the building and testing of the larql-cli crate for WASM32v1-none and forbidden unsafe code compilation."

**Methodology (binding, not optional):**
- Every new failure gets root-caused via `superpowers:systematic-debugging` (read the real CI log, find the *first* error, not the summary count) before any fix is attempted.
- No OS or feature-combination is exempt: if a `(crate, os, features)` leg doesn't compile, that's real signal to gate, not noise to filter out — never suggest `--no-default-features` exemptions or similar to "avoid contamination."
- Every distinct failure mechanism gets classified into a pattern (see taxonomy below) before fixing, so recurring instances of the same pattern are recognized as such rather than re-diagnosed from scratch.
- One variable at a time: a new pattern's fix is tested on exactly one crate first, pushed, and confirmed via real CI output before being rolled out to other crates with the same pattern.
- No local `cargo` builds, ever (same constraint as the rest of this plan) — TOML/source edits are validated with `python3 -c "import tomllib; ..."` / by inspection, never compiled locally. Exception, added after pattern 10 needed it: `cargo tree` is metadata-only (reads `Cargo.toml`/`Cargo.lock`, invokes no compiler, touches no network) and is permitted for tracing a hard-to-pin-down transitive dependency edge, e.g. `cargo tree --target wasm32v1-none -p <crate> --no-default-features -e normal --offline -i <suspect-crate>` — categorically different from `build`/`check`/`test`. Always verify a Cargo.toml fix against `cargo tree` output before spending a CI round on it, when the fix is non-obvious enough to warrant it.

**Pattern taxonomy (living list — append as new patterns are found):**

1. **`dependency-default-std-feature`** — a third-party dependency defaults to a Cargo feature (usually named `std`) that pulls in the `std` crate, which doesn't exist on `wasm32v1-none` (core+alloc only). Symptom: `error[E0463]: can't find crate for `std`` inside the *dependency's* own source, which then cascades into hundreds/thousands of downstream "cannot find X in this scope" errors as everything depending on the now-missing `std`-gated code fails to resolve. **Do not assume a fix exists** — verify the dependency actually supports a `no_std`/`alloc` mode (check docs.rs features, don't guess) before applying `default-features = false`. Confirmed instances: `serde`/`serde_json` (has `alloc` feature), `thiserror` (only feature is `std`, default-features=false is its documented no_std path, derive macro unconditional). Fix shape: split the dependency into `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` (workspace-inherited, unchanged) and `[target.'cfg(target_arch = "wasm32")'.dependencies]` (independent declaration, `default-features = false` + alloc-compatible features).
2. **`own-crate-missing-no_std`** — none of the crates in the `larql-cli` build/test graph declare `#![no_std]` (correction: `crates/larql-experts/expert-interface` does, but it targets `wasm32-wasip1`, a std-capable WASI target, for an unrelated reason — not a precedent applicable here), so rustc injects the `std` prelude implicitly for every crate regardless of target, producing the same `E0463`/cascade pattern as pattern 1 but for *our* source once dependency-level blockers are cleared. Fix: `#![cfg_attr(target_arch = "wasm32", no_std)]` + `#[cfg(target_arch = "wasm32")] extern crate alloc;` at the crate root. **Known landmine:** `larql-core` has its own `pub mod core;`, which already shadows the real `core` crate for unqualified `core::` paths inside that crate (pre-existing, unrelated to this change) — any *new* no_std-migration code added there needs `::core::...` (leading `::`) to reach real libcore.
3. **`native-only-io-module`** — a module's functionality has no core/alloc equivalent at all (e.g. `std::fs`-based file I/O — there is no filesystem on `wasm32v1-none`). Not a feature-toggle fix like pattern 1; the whole module gets excluded via `#[cfg(not(target_arch = "wasm32"))]` on its `pub mod` declaration and every `pub use` re-exporting it. Confirmed instance: `larql-core::io` (checkpoint/csv/json/packed, all `std::fs`/`std::io`). A single enum *variant* can need the same treatment even when the enum itself stays available: `GraphError::Io(#[from] std::io::Error)` — `std::io::Error` has no core/alloc equivalent, but only the excluded `io` module ever constructs that variant (confirmed via grep before gating it), so `#[cfg(not(target_arch = "wasm32"))]` on the variant alone was correct and didn't need to touch the rest of `GraphError`.
4. **`std-collection-needs-no-std-hasher`** — `std::collections::HashMap`/`HashSet` have no core/alloc equivalent because they need a `BuildHasher`, and `alloc` provides no default one (std's `RandomState` seeds itself from OS randomness). `hashbrown` is genuinely `no_std`-capable, but **its own default hasher has the identical requirement** — verified via its Cargo.toml before relying on it, not assumed. Fix: `hashbrown` with `default-features = false` (+ `features = ["serde"]` if the map/set needs `Serialize`/`Deserialize` — also lost when defaults are disabled) plus a fixed-seed deterministic hasher (FNV-1a, hand-rolled, ~15 lines — matches this exact codebase's prior wasm32 attempt, PR #71's "FNV wasm32"). Wrap in a small crate-local module (`src/collections.rs`) exporting portable `HashMap`/`HashSet` type aliases (std-backed natively, hashbrown-backed on wasm32) so call sites `use crate::collections::{HashMap, HashSet}` once instead of repeating the cfg-split per file. **Landmine:** hashbrown's `HashMap::new()`/`HashSet::new()` are inherent methods defined only for the crate's *own* default hasher (same as std, whose `new()` only exists for `RandomState`) — a custom-hasher type alias must use `.default()` instead (works uniformly on both native and wasm32, since `Default` is blanket-available for any `S: Default`). Grep for `HashMap::new()`/`HashSet::new()` *and* bare fn-pointer references (`HashMap::new` with no parens, e.g. passed to `.get_or_insert_with(...)`) — a plain call-site `sed` misses the latter.
5. **`alloc-prelude-types-not-implicit`** — `#![no_std]` drops std's prelude, which auto-imports `Vec`/`String`/`Box`/`ToString`/`ToOwned`. `alloc` provides these but does not inject them — same as any other type/trait, they need an explicit per-file `use`. Separately, `#[macro_use] extern crate alloc;` at the crate root *does* restore crate-wide availability of `vec!`/`format!` (macros and types are different namespaces in Rust's prelude mechanism, so this doesn't cover the value types above). When many files need the same 3-4 alloc imports, a small `src/prelude.rs` (empty on native, re-exports on wasm32) collapses each call site to one line (`#[cfg(target_arch = "wasm32")] use crate::prelude::*;`) instead of repeating the same import block everywhere.
6. **`core-f64-lacks-transcendental-math`** — `core`'s `f64`/`f32` have no `ln`/`exp`/`sqrt`/`sin`/`cos`/etc. — those need a libm implementation, and `std` links against the platform's. The standard no_std-compatible replacement is the `libm` crate (pure Rust, `default-features = false`), used via free functions (`libm::log(x)` for `ln`, etc.) rather than the method-call syntax `std`'s inherent impls provide.
7. **`transitive-feature-reenables-std`** — generalizes pattern 1: it's not enough for *our* direct dependency declaration to disable a crate's `std` feature if a *different*, un-gated dependency in the graph re-requests it. Two confirmed shapes: (a) an entirely un-gated dependency that itself needs full std (`safetensors`/`memmap2` in `larql-models`, transitively pulling in std-enabled `serde` — Cargo's feature unification re-enables `std` for the whole wasm32 build regardless of our own direct `serde` override); (b) a *feature flag* on an already-target-gated dependency re-enabling std in a third crate (`ndarray`'s default `std` feature does `std = ["num-traits/std", "matrixmultiply/std"]` — even with `ndarray` itself otherwise fine, its default features silently poisoned two more crates). Fix for (a): move the un-gated dependency into the same target-gated table. Fix for (b): find and disable the specific feature (read the dependency's own Cargo.toml on its repo, don't guess) — `default-features = false` on `ndarray` for wasm32 resolved both `matrixmultiply` and `num-traits` in one change, confirmed via CI, without needing to gate out any `ndarray`-using code.
8. **`sibling-path-dependency-not-yet-gated`** — a workspace-local path dependency that hasn't had patterns 1/2 applied yet poisons a downstream crate's wasm32 build via Cargo feature unification across the *whole resolved graph*, exactly like pattern 7's external-crate mechanism — except the culprit is a sibling crate in this same workspace, not a crates.io dependency. Confirmed via `larql-kv`: even with `larql-kv`'s own `serde`/`thiserror` correctly split, its build still produced the full `E0463`/`crate_root!` cascade (`once_cell`, `memchr`, `serde_core`) because sibling path deps (`larql-boundary`, `larql-execution`, `larql-compute`, `larql-compute-metal`) hadn't been touched at all — Cargo must build every crate in the resolved graph for the target regardless of which specific downstream crate's Cargo.toml is already clean. **Consequence for method, not just diagnosis:** work strictly bottom-up in the dependency graph (leaves first) rather than crate-by-crate in whatever order CI happens to report failures — a mid-graph crate's CI log is not a reliable signal of *its own* remaining work until every crate below it is at least pattern-1/2-clean.
9. **`dependency-with-zero-no_std-support`** — unlike patterns 1/4/6/7, where some substitute crate or feature flag exists, a small number of dependencies have **no no_std mode at all**, under any feature combination (confirmed instance: `rayon` — needs a real OS thread pool, fundamentally incompatible with a target that has no OS). Fix shape differs from pattern 1: rather than declaring a `default-features = false` wasm32 entry, the dependency is **omitted entirely** from the wasm32 target-dependency table. This doesn't clear the build by itself — it converts every module that directly uses the dependency into a pattern-3 whole-module-exclusion candidate, to be confirmed by the next real CI round rather than guessed by inspection.
10. **`sibling-default-features-unify-across-the-graph`** — a sharper refinement of pattern 7/8: Cargo unifies features across the *entire resolved build graph*, not per dependency edge. A workspace-local path dependency declared *without* `default-features = false` at even **one** call site anywhere in the graph re-enables that dependency's default features for **every** other crate that depends on it too — even a crate whose own declaration is already correctly restricted. Confirmed instance: `larql-vindex` correctly declared `larql-core = { path = "../larql-core", default-features = false }`, but `larql-inference` (which `larql-kv` also depends on) declared the same edge *without* the restriction. `larql-core`'s default features (`http`, `msgpack`) pull in `reqwest`/`rmp-serde`; `reqwest`'s own `[target.'cfg(target_arch = "wasm32")'.dependencies]` unconditionally assumes a *browser* wasm32 environment (`js-sys`/`wasm-bindgen`/`web-sys`) that doesn't exist on `wasm32v1-none` — this was the actual root cause of a `crypto-common`/`serde_core`/`once_cell` `E0463` cascade that looked, from `larql-kv`'s own CI log, like a `larql-kv`-local problem; it wasn't. **Diagnostic method, not just the pattern:** `cargo tree --target wasm32v1-none -p <crate> --no-default-features -e normal --offline -i <suspect-crate>` is metadata-only (no compilation, no network, near-instant) and gives a definitive reverse-dependency trace in seconds — categorically different from `cargo build`/`check`/`test`, which the plan's no-local-build constraint exists to forbid. Used here after several rounds of unproductive static `Cargo.lock`-grepping to pin down an otherwise very hard-to-trace transitive edge; verify any Cargo.toml feature-gating fix against `cargo tree` output before spending a CI round on it.

11. **`documented-gap-is-not-permanent`** — a function classified as a safe-to-skip "native-only, no portable caller" gap (pattern: whole-item `#[cfg(not(target_arch = "wasm32"))]`, confirmed via grep at the time) must be **re-verified**, not assumed to stay valid, any time related code in the same crate/file changes — fixing one function can make a sibling function it calls (or that calls it) newly reachable from portable code. Confirmed twice in one session: (a) `expert/q4k.rs`'s `run_single_expert_kq_q8k_into` looked like a candidate for the same gate as its sibling `run_single_expert_into` until fixing `expert/f32.rs`'s `run_single_expert` made it a direct callee from a now-portable wasm32 branch — caught before pushing, by checking callers before gating. (b) Much more costly: `cpu/ops/moe/forward.rs`'s `cpu_moe_forward` was *never actually verified* as gap-eligible at all — it was simply left unfixed under the assumption that documenting it was equivalent to gating it, when in fact the whole file was still unconditionally compiled and blocking the crate's wasm32 build regardless of every other fix landed around it. It turned out to be the crate's real production MoE entry point (confirmed via grep: called from `larql-inference`'s live forward pass in two places), not a dead-code candidate at all. **Rule going forward: "documented gap" must mean "confirmed via grep to have zero callers outside tests, and actually `#[cfg]`-gated" — never "I decided not to look at this yet."** A finding written down as a gap without both of those steps is not a gap, it is an unexamined blocker wearing a gap's clothing.

12. **`optional-dep-feature-not-target-gated`** — a Cargo *feature* (`http = ["dep:reqwest"]`) is not itself target-conditional; only the *dependency table entry* it references can be. If an optional dependency sits in the plain untargeted `[dependencies]` table and its feature is in `default = [...]`, then on the wasm32 "default" leg the feature is ON and the dependency's full graph (with its own `std`-requiring transitive deps) gets resolved for wasm32 too — even though the crate's own wasm32-target-cfg table looks completely clean. This masqueraded as a `serde_core`/`futures-core` `E0463` cascade (`can't find crate for std` — `serde_core` compiled with its `std` feature ON, `futures-core` pulled in transitively via `reqwest`), which on first read looked exactly like the pattern-10 upstream-no_std-regression shape but wasn't — confirmed by grepping the actual log for `E0463`/`can't find crate` (per the advisor consult that caught this: "the error signature is std-missing, not a serde bug; a feature cfg can't remove prelude items"), not by editing Cargo.lock or bumping dependency versions on a hunch. Root cause in `larql-core`: `reqwest`/`rmp-serde` sat in the plain `[dependencies]` table, both feature-gated (`http`, `msgpack`) and *both features in `default = [...]`* — so the wasm32 "default" CI leg (not "no-default-features", which is why this crate was previously reported green: only the no-default-features leg had ever been verified) pulled the full reqwest graph. Fix: move the optional dependency itself into the native-only `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` table (`optional = true` still works there; the `dep:` feature just has nothing to activate off-target). This is NOT enough by itself — the Cargo *feature* is still nominally on for wasm32 (features aren't target-conditional), so any `#[cfg(feature = "http")] mod http_provider;`-style module declaration that gates on the feature alone must be tightened to `#[cfg(all(feature = "http", not(target_arch = "wasm32")))]`, or the module still gets included and its `use reqwest::...` fails to resolve (the crate is simply gone from that target's dependency graph). **Diagnostic discipline reinforced by this instance:** before touching Cargo.lock or bumping a dependency version to chase an upstream-regression hypothesis, grep the raw CI log for the actual first error in the cascade (`can't find crate for` / `E0463` / `E0432`) — a large, scary-looking cascade (5829 errors, every crate in the matrix failing at once) is very often one small feature-unification fact, not a new class of problem.

13. **`sibling-portable-hashmap-types-are-nominally-distinct`** — two sibling crates each independently applying pattern 4 (portable `HashMap` via a per-crate hand-rolled `FnvHasher`) produce hashbrown maps that are *structurally* identical but *nominally* distinct types on wasm32 — same shape, different hasher marker struct, since each crate's `FnvHasher` is its own `pub(crate)` type (Rust modules don't cross crate boundaries, so the fix is duplicated per crate, not shared). A function in crate B that names crate A's map type in its signature (e.g. `vectors: &HashMap<String, Vec<f32>>` written using B's own alias, called with a value of A's map type) gets a real `E0308` mismatched-types error — invisible on native, where both crates' aliases collapse to the same `std::collections::HashMap`. Confirmed instance: `larql-compute::attention::sinks::resolve()` took `&crate::collections::HashMap<...>` but callers pass `&weights.vectors`, whose real type is `larql-models`' own `pub(crate)` alias — unnameable from `larql-compute` even in principle. Fix: don't name the map type in the cross-crate function signature at all — accept a `impl FnOnce(&str) -> Option<&'a [T]>` lookup closure instead of the whole map. The caller's closure body calls `.get()` via ordinary method resolution, which doesn't require spelling out the type, and callers already have a genuinely-typed value in hand regardless of which crate's alias produced it.

**Status as of this plan revision — three real milestones:**
- `larql-core --no-default-features` compiles cleanly to `wasm32v1-none` on **all three OSes** (run `31404649309`) — patterns 1-6, 8 real CI round-trips (`fc65bdcb`..`a2df440b`).
- `larql-vindex-spec` compiles cleanly on **all 6 legs** (both feature sets, all 3 OSes) — patterns 1+2+5 applied proactively in one commit (`2d4ca048`) rather than rediscovered via CI, since it has none of `larql-core`'s harder blockers (no `HashMap`, no `io` module, no transcendental math).
- **`larql-models` compiles cleanly on all 6 legs** (both feature sets, all 3 OSes — confirmed via a fresh `gh api` job-status pull against run `31413143873`, independent of `gh run watch`'s notification, per standing discipline). Closes out the 666→120→97→29→23→0 own-source error-reduction trajectory. Final blocker was `core`'s f32/f64 lacking `.sqrt()`/`.powf()`/`.powi()`/`.round()`/`.trunc()` — a hand-rolled 2-method `PortableFloat` trait proved incomplete once `quant/fp8.rs` needed 3 more methods, so it was replaced with `num_traits::Float` (re-exported from `prelude.rs`, backed by `num-traits`' own `"libm"` feature — confirmed via its docs.rs page this is a real no_std implementation, not a stub) — one trait covers every transcendental method found missing so far. Also fixed `detect/parser.rs`'s missing `ToString` import (exposed once `detect/` stopped being wholesale-excluded).
- `larql-factory`'s own-source errors have started surfacing: first (and, per that CI round, only) blocker was `sha2`'s un-gated `std`-default feature (pattern 1) cascading through `crypto-common`, plus `hex.rs`'s `std::fmt::Write` (swapped to `core::fmt::Write` unconditionally — same type, no cfg needed, same shape as `larql-boundary`'s `core::cmp::Ordering` fix below) and both files' `String`-returning functions needing their first `prelude.rs`. Pushed (`80413d9b`); `recipe/`, `validate/`, `card/`, `capabilities/`, `build/`, `estimate/` not yet examined — expect several more rounds.
- Bottom-up sweep of `larql-cli`'s remaining dependency graph (prompted by discovering pattern 8 via `larql-kv`'s CI log): `larql-execution` (pure logic, `core::error::Error` swap, zero deps) and `larql-boundary` (pattern 1 + new `prelude.rs` with `num_traits::Float` for `.exp()`/`.ln()`/`.round()`) are pattern-2-complete pending their first real CI round. `larql-compute`/`larql-compute-metal` got the crate-level attribute plus pattern 9 (`rayon` dropped from the wasm32 dependency table entirely — zero no_std support) and pattern 7 (`ndarray` `default-features = false` for wasm32). `larql-router-protocol` (pure `tonic::include_proto!` gRPC scaffolding, zero portable logic) is wholesale-excluded on wasm32 — the same shape as `larql-models`' `connectors`/`encoders`/`loading`/`speech`, but at whole-*crate* granularity rather than whole-module. `larql-kv`/`larql-vindex`/`larql-inference` got the pattern-2 crate attribute (their pattern-1 Cargo.toml edits were already staged from earlier in the campaign). `larql-cli`'s `main.rs` also got `#![cfg_attr(target_arch = "wasm32", no_std)]` — expected to fail differently (no `#[no_main]`/custom entry point), and that failure signature is itself the next thing to classify, not a result to route around. Pushed as `f672cf21`; CI run `31414647092` pending review.

**Further milestones, same session, after adding larql-boundary/larql-execution/larql-router-protocol to the CI matrix (they predated it, only inferable via larql-kv/vindex/inference's aggregate result before):**
- **`larql-boundary`, `larql-execution`, `larql-router-protocol` all compile cleanly on all 6 legs**, first try — confirmed via CI run `31417747852`. `larql-execution`'s zero-dependency `core::error::Error` swap and `larql-router-protocol`'s whole-crate wholesale exclusion both worked exactly as designed.
- `larql-factory` reached its **last own-source error** (`.powi()` on f64 in `estimate/bytes.rs`, pattern 6 — the surgical `build`/`estimate` splits from the delegated-agent round were otherwise clean) — fixed via the same `num_traits::Float` prelude pattern, pushed as `0aa8dbd0`, pending its next CI round.
- `larql-compute`'s `build.rs` C-compilation failure (`csrc/q4_dot.c`, no C toolchain for `wasm32v1-none`) is fixed — confirmed via CI the `TARGET`-env-var skip cleared it, surfacing real Rust source errors underneath: `connectors`/`encoders` (own modules, both directly depend on `larql_models::{connectors,encoders}`, themselves wholesale-excluded — gated the same way, cross-reference-checked first) and a `rayon` dependency confined to exactly one file, `cpu/spin_pool.rs` — a hand-rolled OS-thread work-stealing pool (`std::thread::scope`) used by 11+ files across the attention/matmul/MoE kernels. **Not yet fixed** — unlike the mechanical fixes so far, this is a genuine architectural boundary (no OS threads exist on `wasm32v1-none` at all) that needs its own careful round rather than a blind gate, since 11 call sites' actual compute paths depend on it.
- `larql-kv`'s persistent `E0463`/`crypto-common` cascade was traced to `larql-inference`'s entire `[dependencies]` block being un-gated (confirmed via the CI log: `anyhow` compiling immediately before the cascade, and `anyhow` doesn't declare `#![no_std]`) — not a `larql-kv`-local problem at all. Split the same way as `larql-vindex`: `rand`/`rand_distr`/`half`/`ndarray` get the wasm32 `default-features = false` entry, everything else (`zip`, `tokenizers`, `reqwest`, `tokio`, `tonic`, `wasmtime`, `wasmtime-wasi`, `anyhow`, ...) is native-only (pattern 9). `larql-vindex` itself got the same treatment one round earlier (`sha2` was still plain; six more native-only deps — `rayon`, `tokenizers`, `safetensors`, `memmap2`, `hf-hub`, `reqwest` — had never been touched). Pushed as `d636d9c5`; next CI round pending.

`larql-lql` remains completely untouched — no pattern-2 attribute, no dependency audit. `larql-inference`'s and `larql-cli`'s own-source errors (beyond the crate/dependency-level fixes above) remain unexamined — expect both to need substantial pattern-3 work given how much of `larql-inference` is native-only-dependency-backed (wasmtime WASM expert registry, tonic/tokio gRPC remote backends, tokenizer loading) and `larql-cli`'s `main.rs` is CLI I/O by definition.

**7 crates fully green as of CI run `31433513119`'s predecessor** (`larql-models`, `larql-boundary`, `larql-execution`, `larql-router-protocol`, `larql-vindex-spec`, `larql-factory`, plus `larql-core --no-default-features`-only). That same run (commit `81e740d2`, the `options.rs` choke-point + eprintln fix) came back with 46 failing legs, including `larql-core --default-features` and `larql-compute-metal` — both previously assumed clean/near-clean without re-verification. Root-caused (see pattern 12): `larql-core`'s `reqwest`/`rmp-serde` sat in the plain, un-target-gated `[dependencies]` table with both features in `default = [...]`, so the wasm32 **default** leg (never actually tested — only `--no-default-features` had ever been confirmed green for this crate) pulled the full `reqwest` graph, producing a `serde_core`/`futures-core` `E0463` cascade that superficially resembled a fresh upstream no_std regression in `serde_core` 1.0.228 (it wasn't — confirmed via `E0463`/`can't find crate for std` in the raw log, not by bumping versions). Fixed by moving `reqwest`/`rmp-serde` into the native-only target table and tightening `engine/mod.rs`'s `#[cfg(feature = "http")] mod http_provider;` to also require `not(target_arch = "wasm32")`. `larql-compute-metal`'s 6 failing legs were confirmed to be a pure cascade of `larql-compute`'s own 364 own-source errors (`error: could not compile larql-compute (lib) due to 364 previous errors` appears verbatim in its log) — no separate fix needed there.

`larql-compute`'s 364 own-source errors from that same run were triaged file-by-file: ~18 files needed real judgment (native-only-gating `std::fs::write` dump closures in `attention/block.rs`/`forward/layer.rs`; individually cfg-gating `std::time::Instant` timing checkpoints in `kquant_forward/cached.rs` per pattern 11's caller-verification rule; an `eprintln!` gate in `pipeline_layer/moe_build.rs`; a `crate::collections::HashMap` swap in `forward/predict/raw.rs`; unconditional `core::` swaps for `Ordering`/`Any`/`fmt`/`Range`/`f32`/`f64` consts across a dozen files) and ~44 files needed only the mechanical `alloc_prelude` import (verified each insertion point against the doc-comment-orphaning bug a naive "insert after last `use`" script produces when a file has no pre-existing `use` block — 5 instances caught and fixed by re-reading the diffs before committing). Confirmed via a full crate-wide `std::`-reference sweep afterward that every remaining `std::` occurrence in the touched files is either arch-gated (`#[cfg(target_arch = "aarch64")]`), test-only (`#[cfg(test)]`), or prose in a comment. Next CI round (pending) is expected to reveal whatever error surface sits behind this pass — larql-compute is a ~360-file crate and this was one pass, not necessarily the last.

Still far from "all the code involved" gated — the `forbid-unsafe` axis remains completely unreachable (gated behind `needs: wasm32v1-none`, by design, and `wasm32v1-none` is nowhere near green across the full crate matrix yet). This section will keep growing as the campaign proceeds — treat it as a running log, not a final task list.
