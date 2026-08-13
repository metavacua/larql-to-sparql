# larql-cli Target-Sampling CI Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, via a series of standalone real-CI experiments, a GitHub Actions workflow that builds and tests `larql-cli` across `wasm32v1-none`, `wasm32-unknown-unknown`, the WASI targets, Emscripten, and a native Kani verification pass — surfacing (not fixing) portability breakage as analysis data, with `clippy --fix` output propagated between jobs as diff artifacts only.

**Architecture:** Each task adds one standalone, `workflow_dispatch`-triggered experiment workflow file, pushed and run for real on GitHub-hosted runners (`gh workflow run` / `gh run watch` — never simulated locally). Later tasks reuse patterns validated by earlier ones. The final task assembles everything into one consolidated workflow (`larql-cli-target-sampling.yml`) matching the approved target graph, and removes the now-superseded standalone experiment files.

**Tech Stack:** GitHub Actions (`dtolnay/rust-toolchain`, `actions/checkout@v7`, `actions/upload-artifact@v4`/`download-artifact@v4`), Rust stable (edition 2021, rust-version 1.88 per `Cargo.toml`), `wasmtime` CLI, `kani-verifier`, Emscripten SDK, `gh` CLI for triggering/observing runs.

**Spec:** `docs/superpowers/specs/2026-08-13-larql-cli-target-sampling-design.md`

## Global Constraints

- All jobs run on `ubuntu-latest` — nothing in this workflow needs macOS/Windows (unlike native `larql-cli.yml`, which multi-OSes only for the macOS-only `gpu`/Metal feature).
- Every target/kani job uses `--no-default-features` (the `gpu` feature requires Metal, unavailable on Linux/wasm) and `--no-deps` on clippy (keeps lint scope to `larql-cli`'s own code, matching the existing convention in `.github/workflows/larql-cli.yml:122`).
- Clippy always runs with `-D warnings` (matches existing repo convention).
- No caching (`actions/cache`/`Swatinem/rust-cache`) — deferred until iteration time is a demonstrated problem.
- **No commits from CI, ever.** Fix propagation between jobs is via `actions/upload-artifact`/`download-artifact` diff files scoped to a single workflow run, never `git commit`/`git push` from a job.
- Step-level `continue-on-error: true` on every clippy/fix/build/test step; job-level `continue-on-error: true` on every job except `fmt-check` — confirmed via Ken Muse's writeup (linked in the spec) that this makes `needs.<job>.result` report `"success"` downstream even when the job's own steps failed, so the `needs:` chain is never blocked by expected breakage.
- No `wasm32-wali-linux-musl` or native musl targets (excluded per spec).
- No summary/report job.
- Standalone experiment workflows use `workflow_dispatch: {}` plus `push: branches: [feat/larql-cli-target-sampling]`. `workflow_dispatch`-only does not work on a non-default branch — GitHub Actions only registers a workflow as dispatchable once it has run via a non-dispatch trigger (or exists on the default branch); this was discovered empirically in Task 1 (see ledger), not assumed from docs. The `push` trigger is scoped to this one branch so pushing a later task's file doesn't silently mass-retrigger every earlier experiment workflow beyond what's expected. The final consolidated workflow (Task 10) uses `push`/`pull_request` on `main` plus `workflow_dispatch`, matching the existing `larql-cli.yml` convention.
- Repo: `metavacua/larql-to-sparql`, branch `feat/larql-cli-target-sampling`, working tree at `/home/metavacua/larql-upstream-2026-08-13`, remote `fork`.
- `larql-cli` package name: `larql-cli`; binary: `larql`; path: `crates/larql-cli`.

---

### Task 1: fmt-check standalone experiment

**Files:**
- Create: `.github/workflows/experiment-fmt-check.yml`

**Interfaces:**
- Produces: confirmation that `dtolnay/rust-toolchain@stable` + `components: rustfmt` + `cargo fmt -p larql-cli -- --check` works via `workflow_dispatch` on this branch — the toolchain-install pattern every later task reuses.

- [ ] **Step 1: Write the experiment workflow**

```yaml
name: experiment-fmt-check

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  fmt-check:
    name: fmt-check
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v7

      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt

      - name: Format check
        run: cargo fmt -p larql-cli -- --check
```

- [ ] **Step 2: Commit and push**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
git add .github/workflows/experiment-fmt-check.yml
git commit -m "ci(experiment): add standalone fmt-check workflow"
git push fork feat/larql-cli-target-sampling
```

- [ ] **Step 3: Trigger and watch the run**

```bash
sleep 8
RUN_ID=$(gh run list --workflow=experiment-fmt-check.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" -R metavacua/larql-to-sparql --exit-status
```

Expected: the run completes (pass or fail on the actual `cargo fmt --check` result — either is fine; what matters is the *mechanism* ran without an Actions-level error like an unknown action or bad YAML). If the run fails to even start or errors on the toolchain-install step itself, that's an Actions-mechanics problem to fix in this file and re-push before moving on — everything downstream reuses this exact toolchain-install pattern.

- [ ] **Step 4: If step 3 required fixes, commit them**

```bash
git add .github/workflows/experiment-fmt-check.yml
git commit -m "ci(experiment): fix fmt-check workflow mechanics"
git push fork feat/larql-cli-target-sampling
```

---

### Task 2: wasmtime install method — technical-choice comparison

**Files:**
- Create: `.github/workflows/experiment-wasmtime-install-taiki.yml`
- Create: `.github/workflows/experiment-wasmtime-install-script.yml`
- Create: `.cargo/config.toml`

**Interfaces:**
- Consumes: toolchain-install pattern from Task 1.
- Produces: the winning wasmtime-install method (one `run:`/`uses:` step) and a working `.cargo/config.toml` `runner` entry for `wasm32v1-none`, both reused starting in Task 3.

- [ ] **Step 1: Write variant A — `taiki-e/install-action`**

```yaml
name: experiment-wasmtime-install-taiki

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  install-wasmtime:
    name: install-wasmtime (taiki-e/install-action)
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v7

      - name: Install wasmtime via taiki-e/install-action
        uses: taiki-e/install-action@wasmtime

      - name: Verify wasmtime is on PATH and runs
        run: wasmtime --version
```

- [ ] **Step 2: Write variant B — official install script**

```yaml
name: experiment-wasmtime-install-script

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  install-wasmtime:
    name: install-wasmtime (official script)
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v7

      - name: Install wasmtime via official install script
        run: |
          curl https://wasmtime.dev/install.sh -sSf | bash
          echo "$HOME/.wasmtime/bin" >> "$GITHUB_PATH"

      - name: Verify wasmtime is on PATH and runs
        run: wasmtime --version
```

- [ ] **Step 3: Commit, push, trigger both, watch both**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
git add .github/workflows/experiment-wasmtime-install-taiki.yml .github/workflows/experiment-wasmtime-install-script.yml
git commit -m "ci(experiment): compare wasmtime install methods"
git push fork feat/larql-cli-target-sampling

sleep 8
RUN_A=$(gh run list --workflow=experiment-wasmtime-install-taiki.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
RUN_B=$(gh run list --workflow=experiment-wasmtime-install-script.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_A" -R metavacua/larql-to-sparql --exit-status
gh run watch "$RUN_B" -R metavacua/larql-to-sparql --exit-status
```

Expected: at least one variant prints a real `wasmtime` version and exits 0. If `taiki-e/install-action@wasmtime` fails because `wasmtime` isn't a tool it supports (its tool catalog is not something to assume from memory — this is exactly the kind of uncertainty the experiment resolves), the official install script is the fallback; if both work, prefer `taiki-e/install-action` for consistency with the existing `cargo-llvm-cov` install pattern in `.github/workflows/larql-cli.yml:153`.

- [ ] **Step 4: Delete the losing variant's workflow file**

```bash
# Replace <losing-file> with whichever variant did not win
git rm .github/workflows/experiment-wasmtime-install-<losing-suffix>.yml
```

- [ ] **Step 5: Write `.cargo/config.toml` with the wasm32v1-none runner**

```toml
[target.wasm32v1-none]
runner = "wasmtime run"
```

- [ ] **Step 6: Commit and push**

```bash
git add .cargo/config.toml .github/workflows/
git commit -m "ci(experiment): settle on wasmtime install method, add wasm32v1-none runner config"
git push fork feat/larql-cli-target-sampling
```

---

### Task 3: wasm32v1-none full standalone experiment

**Files:**
- Create: `.github/workflows/experiment-wasm32v1-none.yml`

**Interfaces:**
- Consumes: winning wasmtime-install step from Task 2; `.cargo/config.toml` runner entry from Task 2.
- Produces: the validated per-target job body (clippy → conditional fix → diff upload → build → test-via-wasmtime) that Tasks 5–9 copy and adapt.

- [ ] **Step 1: Write the experiment workflow**

(This uses the winning install method from Task 2 — shown here as `taiki-e/install-action@wasmtime`; substitute the official-script step if that's what won.)

```yaml
name: experiment-wasm32v1-none

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  wasm32v1-none:
    name: wasm32v1-none
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7

      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32v1-none
          components: clippy

      - name: Install wasmtime
        uses: taiki-e/install-action@wasmtime

      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32v1-none -- -D warnings

      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32v1-none --allow-dirty --allow-staged -- -D warnings

      - name: Capture cumulative diff
        if: always()
        run: git diff > wasm32v1-none.patch

      - name: Upload diff artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: diff-wasm32v1-none
          path: wasm32v1-none.patch
          retention-days: 7

      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32v1-none

      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32v1-none
```

- [ ] **Step 2: Commit, push, trigger, watch**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
git add .github/workflows/experiment-wasm32v1-none.yml
git commit -m "ci(experiment): add wasm32v1-none standalone workflow"
git push fork feat/larql-cli-target-sampling

sleep 8
RUN_ID=$(gh run list --workflow=experiment-wasm32v1-none.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" -R metavacua/larql-to-sparql --exit-status || true
gh run view "$RUN_ID" -R metavacua/larql-to-sparql --log > /tmp/wasm32v1-none-run.log
```

Expected: the job completes (its own steps may fail — that's the analysis signal the spec wants). What must work mechanically: the toolchain+target install, the wasmtime install, the diff capture/upload steps executing without an Actions-level error. Read `/tmp/wasm32v1-none-run.log` for exactly where/why `clippy`/`build`/`test` broke on real upstream code — record this; it's the first piece of actual analysis data this whole workflow exists to produce, not something to silently discard.

- [ ] **Step 3: If the workflow mechanics (not larql-cli's code) failed, fix and re-push**

Iterate on the workflow YAML only for Actions-level failures (bad step syntax, wrong action input names, PATH issues). Leave `larql-cli` build/clippy/test failures alone — those are expected.

```bash
git add .github/workflows/experiment-wasm32v1-none.yml
git commit -m "ci(experiment): fix wasm32v1-none workflow mechanics"
git push fork feat/larql-cli-target-sampling
```

---

### Task 4: kani install method — technical-choice comparison + standalone experiment

**Files:**
- Create: `.github/workflows/experiment-kani-action.yml`
- Create: `.github/workflows/experiment-kani-manual.yml`

**Interfaces:**
- Produces: the winning kani-install method, reused in Task 10's consolidated `kani` job.

- [ ] **Step 1: Write variant A — official Kani GitHub Action**

```yaml
name: experiment-kani-action

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  kani:
    name: kani (model-checking/kani-github-action)
    runs-on: ubuntu-latest
    timeout-minutes: 30
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7

      - name: Run Kani
        uses: model-checking/kani-github-action@v1
        with:
          args: -p larql-cli --no-default-features
```

- [ ] **Step 2: Write variant B — manual install**

```yaml
name: experiment-kani-manual

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  kani:
    name: kani (manual install)
    runs-on: ubuntu-latest
    timeout-minutes: 30
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7

      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable

      - name: Install kani-verifier
        run: cargo install kani-verifier

      - name: kani setup
        run: cargo kani setup

      - name: Run Kani
        run: cargo kani -p larql-cli --no-default-features
```

- [ ] **Step 3: Commit, push, trigger both, watch both**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
git add .github/workflows/experiment-kani-action.yml .github/workflows/experiment-kani-manual.yml
git commit -m "ci(experiment): compare kani install methods"
git push fork feat/larql-cli-target-sampling

sleep 8
RUN_A=$(gh run list --workflow=experiment-kani-action.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
RUN_B=$(gh run list --workflow=experiment-kani-manual.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_A" -R metavacua/larql-to-sparql --exit-status || true
gh run watch "$RUN_B" -R metavacua/larql-to-sparql --exit-status || true
```

Expected: `larql-cli` almost certainly has zero `#[kani::proof]` harnesses yet, so both variants should run to completion doing effectively nothing (Kani reports "0 verified harnesses" or similar) — that's fine, this task validates the *tooling installs and runs*, not that proofs exist yet. What matters: does the action/manual-install sequence complete without an Actions-mechanics error (missing CBMC backend, install timeout, wrong action version).

- [ ] **Step 4: Delete the losing variant, commit**

```bash
git rm .github/workflows/experiment-kani-<losing-suffix>.yml
git commit -m "ci(experiment): settle on kani install method"
git push fork feat/larql-cli-target-sampling
```

---

### Task 5: wasm32v1-none → wasm32-unknown-unknown sub-chain (artifact hand-off validation)

**Files:**
- Create: `.github/workflows/experiment-chain-v1none-unknown.yml`
- Modify: `.cargo/config.toml`

**Interfaces:**
- Consumes: validated single-job body from Task 3 (duplicated here — standalone experiment files don't share includes at this stage).
- Produces: confirmation that `actions/upload-artifact` → `actions/download-artifact` → `git apply` → re-diff correctly carries a cumulative patch across a real `needs:` edge, and that job-level `continue-on-error: true` really does let the second job run even when the first job's steps fail.

- [ ] **Step 1: Write the two-job workflow**

```yaml
name: experiment-chain-v1none-unknown

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  wasm32v1-none:
    name: wasm32v1-none
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7

      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32v1-none
          components: clippy

      - name: Install wasmtime
        uses: taiki-e/install-action@wasmtime

      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32v1-none -- -D warnings

      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32v1-none --allow-dirty --allow-staged -- -D warnings

      - name: Capture cumulative diff
        if: always()
        run: git diff > wasm32v1-none.patch

      - name: Upload diff artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: diff-wasm32v1-none
          path: wasm32v1-none.patch
          retention-days: 7

      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32v1-none

      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32v1-none

  wasm32-unknown-unknown:
    name: wasm32-unknown-unknown
    needs: wasm32v1-none
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7

      - name: Download upstream diff artifact
        uses: actions/download-artifact@v4
        with:
          name: diff-wasm32v1-none

      - name: Apply upstream diff
        run: |
          if [ -s wasm32v1-none.patch ]; then
            git apply --allow-empty wasm32v1-none.patch
          fi

      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
          components: clippy

      - name: Install wasmtime
        uses: taiki-e/install-action@wasmtime

      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-unknown-unknown -- -D warnings

      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-unknown-unknown --allow-dirty --allow-staged -- -D warnings

      - name: Capture cumulative diff
        if: always()
        run: git diff > wasm32-unknown-unknown.patch

      - name: Upload diff artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: diff-wasm32-unknown-unknown
          path: wasm32-unknown-unknown.patch
          retention-days: 7

      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-unknown-unknown

      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32-unknown-unknown
```

Add the matching runner entry to `.cargo/config.toml`:

```toml
[target.wasm32-unknown-unknown]
runner = "wasmtime run"
```

- [ ] **Step 2: Commit, push, trigger, watch**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
git add .github/workflows/experiment-chain-v1none-unknown.yml .cargo/config.toml
git commit -m "ci(experiment): validate wasm32v1-none -> wasm32-unknown-unknown artifact hand-off"
git push fork feat/larql-cli-target-sampling

sleep 8
RUN_ID=$(gh run list --workflow=experiment-chain-v1none-unknown.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" -R metavacua/larql-to-sparql --exit-status || true
gh run view "$RUN_ID" -R metavacua/larql-to-sparql --json jobs --jq '.jobs[] | {name, conclusion}'
```

Expected, specifically: (a) `wasm32-unknown-unknown` job actually starts and runs even if `wasm32v1-none` job's own conclusion is `failure` — this is the concrete test of the `needs.<job>.result == "success"` claim from the spec; (b) `git apply` on the downloaded patch succeeds without conflict on a fresh checkout; (c) `download-artifact`/`upload-artifact` version `v4` semantics work as expected (single-run scope, correct file retrieval).

- [ ] **Step 3: If the chaining mechanism itself is broken (job 2 skipped, or apply/download failed), fix and re-push**

```bash
git add .github/workflows/experiment-chain-v1none-unknown.yml .cargo/config.toml
git commit -m "ci(experiment): fix artifact hand-off mechanics"
git push fork feat/larql-cli-target-sampling
```

---

### Task 6: wasm32-wasip1 standalone experiment

**Files:**
- Create: `.github/workflows/experiment-wasm32-wasip1.yml`
- Modify: `.cargo/config.toml`

**Interfaces:**
- Consumes: validated job pattern from Task 5's `wasm32-unknown-unknown` job (copy, retarget); `diff-wasm32-unknown-unknown` artifact name from Task 5.

- [ ] **Step 1: Write the experiment workflow**

This file runs standalone (`workflow_dispatch` only, no real `needs:` on the Task 5 workflow — GitHub Actions artifacts don't cross workflow-file boundaries by job name alone, and downloading Task 5's artifact by cross-workflow `run-id` is unreliable to script blind). It skips the upstream-diff download/apply and just verifies `wasm32-wasip1` builds/tests correctly starting from a clean checkout — the real cross-job artifact hand-off only needs to work *within* a single workflow run, which Task 5 already validated and Task 10 exercises for real across the full graph.

```yaml
name: experiment-wasm32-wasip1

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  wasm32-wasip1:
    name: wasm32-wasip1
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7

      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-wasip1
          components: clippy

      - name: Install wasmtime
        uses: taiki-e/install-action@wasmtime

      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip1 -- -D warnings

      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip1 --allow-dirty --allow-staged -- -D warnings

      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-wasip1

      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32-wasip1
```

- [ ] **Step 2: Add the runner entry**

```toml
[target.wasm32-wasip1]
runner = "wasmtime run --dir=."
```

- [ ] **Step 3: Commit, push, trigger, watch**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
git add .github/workflows/experiment-wasm32-wasip1.yml .cargo/config.toml
git commit -m "ci(experiment): add wasm32-wasip1 standalone workflow"
git push fork feat/larql-cli-target-sampling

sleep 8
RUN_ID=$(gh run list --workflow=experiment-wasm32-wasip1.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" -R metavacua/larql-to-sparql --exit-status || true
```

Expected: mechanism (toolchain/target install, wasmtime install/runner invocation) completes without an Actions-level error; record whatever `larql-cli` build/clippy/test results actually are.

- [ ] **Step 4: If mechanics failed, fix and re-push**

```bash
git add .github/workflows/experiment-wasm32-wasip1.yml .cargo/config.toml
git commit -m "ci(experiment): fix wasm32-wasip1 workflow mechanics"
git push fork feat/larql-cli-target-sampling
```

---

### Task 7: wasm32-wasip2 standalone experiment

**Files:**
- Create: `.github/workflows/experiment-wasm32-wasip2.yml`
- Modify: `.cargo/config.toml`

**Interfaces:**
- Consumes: pattern from Task 6.

- [ ] **Step 1: Write the experiment workflow**

```yaml
name: experiment-wasm32-wasip2

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  wasm32-wasip2:
    name: wasm32-wasip2
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7

      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-wasip2
          components: clippy

      - name: Install wasmtime
        uses: taiki-e/install-action@wasmtime

      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip2 -- -D warnings

      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip2 --allow-dirty --allow-staged -- -D warnings

      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-wasip2

      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32-wasip2
```

- [ ] **Step 2: Add the runner entry**

```toml
[target.wasm32-wasip2]
runner = "wasmtime run -S preview2 --dir=."
```

- [ ] **Step 3: Commit, push, trigger, watch**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
git add .github/workflows/experiment-wasm32-wasip2.yml .cargo/config.toml
git commit -m "ci(experiment): add wasm32-wasip2 standalone workflow"
git push fork feat/larql-cli-target-sampling

sleep 8
RUN_ID=$(gh run list --workflow=experiment-wasm32-wasip2.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" -R metavacua/larql-to-sparql --exit-status || true
```

Expected: same mechanics-vs-code-failure distinction as Task 6. The `-S preview2` wasmtime flag is a best-effort guess at current `wasmtime run` syntax for WASI preview2 — if `wasmtime --version`'s help output (from Task 2) shows different flag syntax, correct it here based on the real CLI, not assumption.

- [ ] **Step 4: If mechanics failed, fix and re-push**

```bash
git add .github/workflows/experiment-wasm32-wasip2.yml .cargo/config.toml
git commit -m "ci(experiment): fix wasm32-wasip2 workflow mechanics"
git push fork feat/larql-cli-target-sampling
```

---

### Task 8: wasm32-wasip1-threads standalone experiment

**Files:**
- Create: `.github/workflows/experiment-wasm32-wasip1-threads.yml`
- Modify: `.cargo/config.toml`

**Interfaces:**
- Consumes: pattern from Task 6.

- [ ] **Step 1: Write the experiment workflow**

```yaml
name: experiment-wasm32-wasip1-threads

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  wasm32-wasip1-threads:
    name: wasm32-wasip1-threads
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7

      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-wasip1-threads
          components: clippy

      - name: Install wasmtime
        uses: taiki-e/install-action@wasmtime

      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip1-threads -- -D warnings

      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip1-threads --allow-dirty --allow-staged -- -D warnings

      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-wasip1-threads

      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32-wasip1-threads
```

- [ ] **Step 2: Add the runner entry**

```toml
[target.wasm32-wasip1-threads]
runner = "wasmtime run -W threads=y --dir=."
```

- [ ] **Step 3: Commit, push, trigger, watch**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
git add .github/workflows/experiment-wasm32-wasip1-threads.yml .cargo/config.toml
git commit -m "ci(experiment): add wasm32-wasip1-threads standalone workflow"
git push fork feat/larql-cli-target-sampling

sleep 8
RUN_ID=$(gh run list --workflow=experiment-wasm32-wasip1-threads.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" -R metavacua/larql-to-sparql --exit-status || true
```

Expected: same mechanics-vs-code-failure distinction. The `-W threads=y` flag is a best-effort guess — correct against the real `wasmtime run --help` output if wrong.

- [ ] **Step 4: If mechanics failed, fix and re-push**

```bash
git add .github/workflows/experiment-wasm32-wasip1-threads.yml .cargo/config.toml
git commit -m "ci(experiment): fix wasm32-wasip1-threads workflow mechanics"
git push fork feat/larql-cli-target-sampling
```

---

### Task 9: wasm32-unknown-emscripten — emsdk technical-choice comparison + standalone experiment

**Files:**
- Create: `.github/workflows/experiment-emscripten-setup-action.yml`
- Create: `.github/workflows/experiment-emscripten-manual.yml`

**Interfaces:**
- Produces: the winning emsdk-install method, reused in Task 10's consolidated `wasm32-unknown-emscripten` job. Build-only — no wasmtime runner entry (per spec, Emscripten output expects a JS host, not wasmtime).

- [ ] **Step 1: Write variant A — `mymindstorm/setup-emsdk`**

```yaml
name: experiment-emscripten-setup-action

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  wasm32-unknown-emscripten:
    name: wasm32-unknown-emscripten (setup-emsdk action)
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7

      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-emscripten
          components: clippy

      - name: Install Emscripten SDK
        uses: mymindstorm/setup-emsdk@v14

      - name: Verify emcc
        run: emcc --version

      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-unknown-emscripten -- -D warnings

      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-unknown-emscripten --allow-dirty --allow-staged -- -D warnings

      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-unknown-emscripten
```

- [ ] **Step 2: Write variant B — manual emsdk clone + activate**

```yaml
name: experiment-emscripten-manual

on:
  workflow_dispatch: {}
  push:
    branches: [feat/larql-cli-target-sampling]

jobs:
  wasm32-unknown-emscripten:
    name: wasm32-unknown-emscripten (manual emsdk)
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7

      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-emscripten
          components: clippy

      - name: Install Emscripten SDK manually
        run: |
          git clone https://github.com/emscripten-core/emsdk.git "$HOME/emsdk"
          cd "$HOME/emsdk"
          ./emsdk install latest
          ./emsdk activate latest

      - name: Verify emcc
        run: |
          source "$HOME/emsdk/emsdk_env.sh"
          emcc --version
          echo "$HOME/emsdk" >> "$GITHUB_PATH"
          echo "$HOME/emsdk/upstream/emscripten" >> "$GITHUB_PATH"

      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-unknown-emscripten -- -D warnings

      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-unknown-emscripten --allow-dirty --allow-staged -- -D warnings

      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-unknown-emscripten
```

- [ ] **Step 3: Commit, push, trigger both, watch both**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
git add .github/workflows/experiment-emscripten-setup-action.yml .github/workflows/experiment-emscripten-manual.yml
git commit -m "ci(experiment): compare emsdk install methods"
git push fork feat/larql-cli-target-sampling

sleep 8
RUN_A=$(gh run list --workflow=experiment-emscripten-setup-action.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
RUN_B=$(gh run list --workflow=experiment-emscripten-manual.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_A" -R metavacua/larql-to-sparql --exit-status || true
gh run watch "$RUN_B" -R metavacua/larql-to-sparql --exit-status || true
```

Expected: whichever variant gets `emcc --version` to print a real version wins — that's the concrete, observable pass/fail signal for this comparison, independent of whatever `larql-cli`'s own clippy/build steps report afterward.

- [ ] **Step 4: Delete the losing variant, commit**

```bash
git rm .github/workflows/experiment-emscripten-<losing-suffix>.yml
git commit -m "ci(experiment): settle on emsdk install method"
git push fork feat/larql-cli-target-sampling
```

---

### Task 10: Assemble the consolidated `larql-cli-target-sampling.yml`

**Files:**
- Create: `.github/workflows/larql-cli-target-sampling.yml`
- Delete: `.github/workflows/experiment-*.yml` (all remaining standalone experiment files)

**Interfaces:**
- Consumes: every validated job body from Tasks 1, 3, 4 (winner), 5, 6, 7, 8, 9 (winner).
- Produces: the full target graph from the spec, wired with real `needs:` and job-level `continue-on-error: true`.

- [ ] **Step 1: Write the consolidated workflow**

```yaml
name: larql-cli-target-sampling

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch: {}

jobs:
  fmt-check:
    name: fmt-check
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v7
      - name: Install stable Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - name: Format check
        run: cargo fmt -p larql-cli -- --check

  wasm32v1-none:
    name: wasm32v1-none
    needs: fmt-check
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32v1-none
          components: clippy
      - uses: taiki-e/install-action@wasmtime
      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32v1-none -- -D warnings
      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32v1-none --allow-dirty --allow-staged -- -D warnings
      - name: Capture cumulative diff
        if: always()
        run: git diff > wasm32v1-none.patch
      - name: Upload diff artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: diff-wasm32v1-none
          path: wasm32v1-none.patch
          retention-days: 7
      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32v1-none
      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32v1-none

  wasm32-unknown-unknown:
    name: wasm32-unknown-unknown
    needs: wasm32v1-none
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7
      - uses: actions/download-artifact@v4
        with:
          name: diff-wasm32v1-none
      - name: Apply upstream diff
        run: |
          if [ -s wasm32v1-none.patch ]; then
            git apply --allow-empty wasm32v1-none.patch
          fi
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
          components: clippy
      - uses: taiki-e/install-action@wasmtime
      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-unknown-unknown -- -D warnings
      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-unknown-unknown --allow-dirty --allow-staged -- -D warnings
      - name: Capture cumulative diff
        if: always()
        run: git diff > wasm32-unknown-unknown.patch
      - name: Upload diff artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: diff-wasm32-unknown-unknown
          path: wasm32-unknown-unknown.patch
          retention-days: 7
      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-unknown-unknown
      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32-unknown-unknown

  wasm32-wasip1:
    name: wasm32-wasip1
    needs: wasm32-unknown-unknown
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7
      - uses: actions/download-artifact@v4
        with:
          name: diff-wasm32-unknown-unknown
      - name: Apply upstream diff
        run: |
          if [ -s wasm32-unknown-unknown.patch ]; then
            git apply --allow-empty wasm32-unknown-unknown.patch
          fi
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-wasip1
          components: clippy
      - uses: taiki-e/install-action@wasmtime
      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip1 -- -D warnings
      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip1 --allow-dirty --allow-staged -- -D warnings
      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-wasip1
      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32-wasip1

  wasm32-wasip2:
    name: wasm32-wasip2
    needs: wasm32-unknown-unknown
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7
      - uses: actions/download-artifact@v4
        with:
          name: diff-wasm32-unknown-unknown
      - name: Apply upstream diff
        run: |
          if [ -s wasm32-unknown-unknown.patch ]; then
            git apply --allow-empty wasm32-unknown-unknown.patch
          fi
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-wasip2
          components: clippy
      - uses: taiki-e/install-action@wasmtime
      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip2 -- -D warnings
      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip2 --allow-dirty --allow-staged -- -D warnings
      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-wasip2
      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32-wasip2

  wasm32-wasip1-threads:
    name: wasm32-wasip1-threads
    needs: wasm32-unknown-unknown
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7
      - uses: actions/download-artifact@v4
        with:
          name: diff-wasm32-unknown-unknown
      - name: Apply upstream diff
        run: |
          if [ -s wasm32-unknown-unknown.patch ]; then
            git apply --allow-empty wasm32-unknown-unknown.patch
          fi
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-wasip1-threads
          components: clippy
      - uses: taiki-e/install-action@wasmtime
      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip1-threads -- -D warnings
      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-wasip1-threads --allow-dirty --allow-staged -- -D warnings
      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-wasip1-threads
      - name: Test (via wasmtime)
        continue-on-error: true
        run: cargo test -p larql-cli --no-default-features --target wasm32-wasip1-threads

  wasm32-unknown-emscripten:
    name: wasm32-unknown-emscripten
    needs: wasm32-unknown-unknown
    runs-on: ubuntu-latest
    timeout-minutes: 20
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7
      - uses: actions/download-artifact@v4
        with:
          name: diff-wasm32-unknown-unknown
      - name: Apply upstream diff
        run: |
          if [ -s wasm32-unknown-unknown.patch ]; then
            git apply --allow-empty wasm32-unknown-unknown.patch
          fi
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-emscripten
          components: clippy
      - uses: mymindstorm/setup-emsdk@v14
      - name: Clippy
        id: clippy
        continue-on-error: true
        run: cargo clippy -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-unknown-emscripten -- -D warnings
      - name: Clippy --fix
        if: steps.clippy.outcome == 'failure'
        continue-on-error: true
        run: cargo clippy --fix -p larql-cli --bins --tests --no-default-features --no-deps --target wasm32-unknown-emscripten --allow-dirty --allow-staged -- -D warnings
      - name: Build
        continue-on-error: true
        run: cargo build -p larql-cli --bins --no-default-features --target wasm32-unknown-emscripten

  kani:
    name: kani
    needs: fmt-check
    runs-on: ubuntu-latest
    timeout-minutes: 30
    continue-on-error: true
    steps:
      - uses: actions/checkout@v7
      - uses: model-checking/kani-github-action@v1
        with:
          args: -p larql-cli --no-default-features
```

(Substitute Task 4's and Task 9's actual winning variants if they differed from the `model-checking/kani-github-action`/`mymindstorm/setup-emsdk` guesses shown here.)

- [ ] **Step 2: Delete the standalone experiment workflows**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
git rm .github/workflows/experiment-*.yml
```

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/larql-cli-target-sampling.yml
git commit -m "ci: assemble consolidated larql-cli-target-sampling workflow

Combines the fmt-check gate, wasm32v1-none -> wasm32-unknown-unknown ->
{wasip1, wasip2, wasip1-threads, emscripten} fan-out, and the native kani
job into one workflow, using the job patterns validated by the standalone
experiment workflows this replaces."
git push fork feat/larql-cli-target-sampling
```

---

### Task 11: Open a PR and validate the full graph on real hosted runners

**Files:** none (no code changes — validation only)

**Interfaces:**
- Consumes: Task 10's consolidated workflow.
- Produces: a real, observed end-to-end run confirming the full graph's `needs:`/`continue-on-error` chaining behaves as designed under actual (expected) breakage.

- [ ] **Step 1: Open the PR**

```bash
cd /home/metavacua/larql-upstream-2026-08-13
gh pr create -R metavacua/larql-to-sparql \
  --base main \
  --head feat/larql-cli-target-sampling \
  --title "ci: larql-cli target-sampling workflow" \
  --body "Adds a GitHub Actions workflow sampling larql-cli across wasm32v1-none, wasm32-unknown-unknown, the WASI targets, Emscripten, and a native Kani pass. Breakage is expected and is the point — see docs/superpowers/specs/2026-08-13-larql-cli-target-sampling-design.md."
```

- [ ] **Step 2: Watch the triggered run**

```bash
sleep 10
RUN_ID=$(gh run list --workflow=larql-cli-target-sampling.yml --branch=feat/larql-cli-target-sampling -R metavacua/larql-to-sparql --limit=1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" -R metavacua/larql-to-sparql --exit-status || true
gh run view "$RUN_ID" -R metavacua/larql-to-sparql --json jobs --jq '.jobs[] | {name, conclusion, needs: .steps | length}'
```

- [ ] **Step 3: Confirm the graph behaved as designed**

Check specifically:
- Every job ran (none show `conclusion: "skipped"`), including the fan-out jobs even if `wasm32v1-none` and/or `wasm32-unknown-unknown` failed.
- `kani` ran independently of the wasm chain's outcome.
- Each target job's diff artifact is present (`gh run view "$RUN_ID" -R metavacua/larql-to-sparql --json artifacts`).

Record the actual build/clippy/test outcomes per target in a follow-up note (not part of this plan) — that data is the deliverable this whole workflow exists to produce for the later migration/refactoring work referenced in the spec's "Purpose" section.

- [ ] **Step 4: Leave the PR open for review**

No merge action is part of this plan — merging is a separate decision outside this implementation plan's scope.
