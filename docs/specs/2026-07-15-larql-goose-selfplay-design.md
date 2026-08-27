# Design Spec — Self-Playing, Self-Learning LARQL Coding Agent (SmolLM2-135M via Goose)

## Metadata (Dublin Core / Schema.org)

- **dc:title**: Self-Playing, Self-Learning LARQL Coding Agent
- **dc:creator**: RDL loop (Ralph-loop session, agent-authored, human-steered)
- **dc:date**: 2026-07-15
- **dc:source**: `docs/specs/2026-07-15-larql-goose-selfplay-research-residual.md`
- **schema:softwareVersion**: targets `larql-to-sparql` main, SmolLM2-135M, `metavacua/goose` fork
- **status**: DRAFT — Phase 1 (Brainstorm) output, feeds Phase 2 (`rdl-writing-plans`)

## 1. Problem framing (why this shape, not the naive one)

A prior failure mode in this org's own RDL history — a "self-play" `evolve()` layer that passed
12/12 tests while being a structural misunderstanding of AlphaZero-style self-play — makes the
framing itself the first design decision, not an afterthought. AlphaZero-style self-play requires
**two independent things in tension**: a policy that acts, and an evaluator whose verdict the
policy does not control. A single process that generates output and also scores its own output is
not self-play; it is unsupervised self-scoring, and it degrades into exactly the kind of silent
no-op this design must guard against (residual N1).

**This design's actual shape:**

- **Actor**: SmolLM2-135M, compiled into a LARQL vindex, driving Goose as a coding agent (writes
  code, runs tools, edits files) against small, scoped coding tasks.
- **Evaluator**: an external, independent test suite per task — pass/fail is not adjudicated by
  the model, by Goose, or by any component that shares state with the actor. This is what makes
  it self-play rather than self-scoring: the actor cannot move the goalposts.
- **Learning**: trajectory outcomes translate into LQL `INSERT`/`COMPOSE` operations against the
  agent's own vindex (facts observed to correlate with success), periodically consolidated via
  `COMPACT MAJOR` (MEMIT closed-form edit) and `COMPILE ... INTO VINDEX` (pure structural
  weight-matrix edit) — no gradient descent, consistent with this repo's own thesis that "the
  model *is* the database" (`AGENTS.md`).

## 2. RFC 2119 Constraints

- The self-play loop **MUST** score trajectories using an evaluator that does not share mutable
  state with the actor (test suite exit code, not model/agent self-assessment). *(residual C4, N1)*
- The system **MUST NOT** serve the vindex via `larql-server` (`larql serve`, its OpenAI-compatible
  HTTP route, or gRPC) as the Goose transport. *(residual C1, user directive)*
- The system **MUST NOT** invoke any local-model-loading command unwrapped by an active
  containment boundary. *(residual C2)* Because Goose's `larql.rs` backend spawns `larql chat` as a
  child process for the duration of a session (residual K22, ADR-1), the containment unit **MUST**
  be the whole `goose` process **plus** its spawned `larql chat` child, not individual isolated CLI
  invocations.
- All actual execution of `larql-cli`/REPL/`goose` commands **MUST** be delegated to a subagent;
  the orchestrating loop **MUST NOT** run them inline. *(residual C3, user directive)*
- Base vindexes **MUST** remain immutable; all self-learning mutation **MUST** flow through
  `PatchedVindex` overlays / `.vlp` patches, per this repo's existing invariant (`AGENTS.md`) —
  this design introduces no exception to it.
- A self-learning round that installs zero effective edits (e.g. issue #261's silent
  capacity-collision skip) **MUST** be distinguishable, in the round's own logged outcome, from a
  round that installed N>0 edits — silent no-ops **MUST NOT** be reported as successful learning.
  *(residual N1)*
- Any GitHub branch/PR/issue write **MUST** stay within `[security].write_allowed` in
  `rdloop.toml`. *(residual C5)*
- The design **SHOULD** default the coding-task suite to difficulty deliberately scoped under
  SmolLM2-135M's actual capability ceiling, not an unscoped standard benchmark. *(residual Q2)*
- The design **SHOULD** measure a behavioral before/after delta (not just "compile succeeded")
  before claiming a learning round changed anything. *(residual N3)*

## 3. EARS Acceptance Criteria

- **AC-1** (ubiquitous): The system SHALL run every actual `larql`/`goose` process inside an
  isolated execution boundary (local QEMU/KVM microVM; §5 ADR-2) that the orchestrator can
  terminate independently of the process's own cooperation.
- **AC-2** (event-driven): WHEN a self-play round completes a Goose coding-agent session against a
  task, THE system SHALL record the task's independent test-suite exit code as the round's sole
  pass/fail signal, before any LQL mutation is proposed.
- **AC-3** (event-driven): WHEN a round is scored PASS, THE system SHALL propose one or more LQL
  `INSERT`/`COMPOSE` operations derived from the successful trajectory, and SHALL log the resulting
  patch's operation count; a round producing zero operations SHALL be logged as
  `learned=false, reason=<cause>`, never silently folded into a generic "done."
  *(closes N1 — no silent no-op path may report success.)*
- **AC-4** (event-driven): WHEN `COMPILE ... INTO VINDEX` runs at the end of a consolidation cycle,
  THE system SHALL re-run the same before/after task subset through the evaluator and record the
  delta, so "compiled" and "measurably changed behavior" are never conflated. *(closes N3.)*
- **AC-5** (state-driven): WHILE a `goose` process (running the `larql.rs` backend, with its spawned
  `larql chat` child) is active inside its execution boundary, THE orchestrator SHALL poll or
  receive a liveness signal at a bounded interval, and SHALL treat the absence of that signal past
  a declared timeout as a fault, not as "still working." *(closes N4/N5 — no unbounded silent
  hang.)*
- **AC-6** (unwanted behavior): IF the `larql.rs` backend's `generate()`/`load_model()` call panics,
  errors, the spawned `larql chat` child exits unexpectedly, or the underlying vindex fails to
  load, THEN Goose SHALL surface a `ProviderError` through the normal `LocalInferenceBackend` error
  path (not a raw process crash the orchestrator has to infer from silence).
- **AC-7** (optional feature): WHERE a task's reference test suite is unavailable or ambiguous, THE
  system SHALL exclude that task from the scored set rather than approximate a score.

## 4. Structurizr DSL — architecture model

```dsl
workspace "LARQL Self-Play Coding Agent" "SmolLM2-135M self-play via Goose, self-learning via LQL patches" {

  model {
    developer = person "Developer / Operator" "Reviews rounds, approves patch consolidation, owns the local VM boundary"

    larqlSystem = softwareSystem "LARQL" "Model-as-database engine; vindex + LQL" {
      vindex = container "SmolLM2-135M vindex" "mmap'd, browse+inference level, PatchedVindex overlay" "vindex files + .vlp patches"
      lql = container "larql-lql" "LQL parser/executor; run_statement/run_batch library entry points" "Rust crate"
      larqlChat = container "larql chat / larql run" "Existing, unmodified interactive CLI process — plain stdin/stdout chat loop" "Rust binary (larql-cli), spawned as a child process"
      selfPlayDriver = container "self-play driver (new)" "Orchestrates rounds: task selection, patch proposal, COMPACT/COMPILE cycles" "Rust, new crate or larql-cli subcommand"
    }

    gooseFork = softwareSystem "metavacua/goose (fork)" "Coding agent harness" {
      gooseCore = container "goose" "Agent loop: plans, calls tools, streams provider output" "Rust binary"
      larqlBackend = container "larql.rs (new)" "LocalInferenceBackend impl; spawns + pipes larql chat via stdin/stdout, no HTTP, no in-process Cargo link" "Rust, new file in goose-local-inference"
    }

    evaluator = softwareSystem "Task evaluator" "Independent pass/fail signal — MUST NOT share state with the actor" "External test suite per coding task"

    vm = deploymentEnvironment "Local QEMU/KVM microVM" "Isolated execution boundary; KVM-accelerated, verified booting on this host (ADR-2)"

    developer -> selfPlayDriver "selects task batch, reviews consolidation cycles"
    selfPlayDriver -> gooseCore "launches a session per task (delegated subagent, inside vm)"
    gooseCore -> larqlBackend "generate() calls, per LocalInferenceBackend trait"
    larqlBackend -> larqlChat "spawns as child process; writes stdin, reads stdout per turn"
    larqlChat -> lql "same in-process LQL executor larql-cli always uses"
    lql -> vindex "reads (inference) / writes (INSERT/COMPOSE via overlay)"
    gooseCore -> evaluator "submits candidate patch/solution"
    evaluator -> selfPlayDriver "pass/fail exit code (independent signal)"
    selfPlayDriver -> vindex "proposes INSERT/COMPOSE on PASS; COMPACT MAJOR + COMPILE INTO VINDEX on consolidation"
  }

  views {
    systemContext larqlSystem "SystemContext" {
      include *
      autoLayout
    }
    container larqlSystem "LarqlContainers" {
      include *
      autoLayout
    }
    container gooseFork "GooseContainers" {
      include *
      autoLayout
    }
  }
}
```

## 5. MADR Decision Records

### ADR-1: Goose transport is a subprocess `LocalInferenceBackend` adapter driving unmodified `larql chat`, not `larql-server` and not in-process Cargo linking

- **Status**: Accepted (2026-07-15), revised from an earlier version of this ADR. Revision
  trigger: the user proposed a simpler shape directly — LARQL keeps running as its own native
  CLI/REPL process inside the VM (ADR-2), and Goose's *existing* "local model" mechanism gets an
  adapter for it, rather than LARQL being linked into the `goose` binary as a Cargo dependency.
  Checking the actual code confirmed this is not just simpler but a better structural fit than the
  original plan.
- **Context**: `larql-server`'s OpenAI-compatible route has two open, unfixed bugs blocking Goose
  (#266, #268), discovered by `metavacua/babel-harness`'s own CI; the user separately vetoed the
  `larql-server` method outright (C1). The originally-accepted version of this ADR proposed
  implementing the full async `Provider` trait directly, depending on `larql-lql`/`larql-inference`
  as an in-process Cargo crate inside `goose` — modeled on `goose-local-inference`'s `llama-cpp-2`
  FFI dependency. Reading the actual `goose-local-inference` crate (residual K34) shows a better
  seam already exists one layer down: `InferenceRuntime` holds a `HashMap<&'static str, Arc<dyn
  LocalInferenceBackend>>` (currently `llamacpp`, `mlx`), and `LocalInferenceBackend` is a
  **synchronous** trait (`load_model`/`generate`/`available_memory_bytes`) — a much more natural
  fit for a subprocess-I/O adapter (blocking write-then-read) than the fully async `Provider`
  trait. Separately, LARQL already has an interactive chat loop (`larql chat`/`larql run` with no
  prompt — residual K35) that is a plain, unmodified CLI process with simple text-in/text-out
  framing (confirmed by reading `run_chat`, `run_cmd.rs:396-430`: prompt written to stderr, one
  line read from stdin per turn, response streamed to stdout) — not `larql-server`, so no conflict
  with C1.
- **Decision**: Implement a new `LocalInferenceBackend` (e.g. `larql.rs`, alongside `llamacpp.rs`/
  `mlx.rs` in `goose-local-inference`) that spawns `larql chat <vindex>` (or `larql run <vindex>`)
  as a child process per loaded model, drives it via piped stdin/stdout (write a line, read the
  streamed response), and registers it in `InferenceRuntime::get_or_init()`'s backend map under a
  new `LARQL_BACKEND_ID`. LARQL itself needs no new library-linking surface and no change to its
  own crate boundaries — it stays exactly what it already is, a CLI binary. The only two-sided
  coupling is the adapter's understanding of `run_chat`'s existing stdin/stdout framing (plus,
  possibly, one small additive change to `run_chat` to emit an explicit stdout turn-boundary
  marker, since today's boundary signal is on stderr — a design detail for T11/T12, not yet
  decided).
- **Autonomous startup ordering inside the guest** (clarified by the user 2026-07-15): Goose cannot
  usefully start without a model backend available, so inside the VM the correct boot sequence is
  LARQL-first, Goose-second, both starting **autonomously** — not driven step-by-step from the
  host over an interactive channel. The self-play loop is meant to be a closed loop *inside* the
  VM once it boots, not an external orchestrator issuing commands into the guest one round at a
  time. Concretely: `larql.rs`'s `load_model()` already spawns `larql chat` on demand (satisfying
  the ordering at the single-generate-call level), but the **guest's own boot sequence** should
  independently guarantee LARQL/its vindex are staged and ready (e.g. a systemd unit or init
  script that waits for the 9p-mounted model files, per T10's file-provisioning step) before the
  `goose` process starts, with a `goose.service`-style unit declaring `After=`/`Requires=` on
  whatever prepares the model — not assumed to "just work" because `larql.rs` happens to spawn its
  own child on first use. This is a T10/T15 design point (guest init sequencing), not a change to
  `larql.rs` itself.
- **Consequences**: No HTTP/SSE/schema-validation bugs to route around (avoids #266/#268 by
  construction, same as the original version of this ADR). No new Cargo cross-repo dependency
  needed (`larql-lql` is never imported by `goose`) — the coupling is at the process-I/O level, not
  the type-system level, which is weaker coupling in the sense that matters here (independent
  release cadence, independent crash domains: a LARQL panic kills the child process, not the whole
  `goose` binary — arguably a containment *improvement* over the original in-process design, where
  a LARQL panic would have taken `goose` down with it). The whole `goose` process **plus** its
  spawned `larql chat` child are still what the VM boundary (ADR-2) must contain (residual K22
  still holds — it's now two processes instead of one, not zero). OpenFang's `process_manager.rs`
  *pattern* (residual K24 — persistent piped process, `alive`/`uptime_secs` tracking) is a directly
  applicable reference for how `load_model`/`generate` manage the spawned `larql chat` child's
  lifecycle, though `LocalInferenceBackend`'s actual method signatures (K34) are what the
  implementation must satisfy, not `process_manager.rs`'s own API shape.
- **Alternatives considered**: the original in-process `Provider`-trait-implementing, Cargo-linked
  design (superseded — the `LocalInferenceBackend` seam is a better fit and was simply not found
  until this revision); `larql-server` + OpenAI provider (rejected, E1); export to GGUF + real
  Ollama (rejected — reintroduces a different server dependency and conversion step neither this
  nor the superseded design needed).

### ADR-2: Execution boundary is a local, KVM-accelerated QEMU microVM (genuinely local, genuinely a VM)

- **Status**: Accepted (2026-07-15), third and final revision of this ADR. History, for
  traceability: (1) originally a Fly Machine (real Firecracker VM, but remote/cloud); (2) briefly
  revised to a local Docker sandbox after finding `metavacua/openfang`'s `docker_sandbox.rs`
  (K23/K24) — **withdrawn** once the user clarified "VM" meant genuine kernel isolation, not a
  shared-kernel container, and a follow-up search of OpenFang (707 files) + 8 other `metavacua`
  forks found zero hypervisor/microVM code anywhere in the org; (3) reverted to the Fly Machine —
  **also withdrawn** once the user clarified execution must be **local**, not cloud-offloaded,
  which Fly Machines are not.
- **Context**: `larql-probe safe` deadlocks (issue #246) — its cgroup/`systemd-run --user` approach
  is also independently known (user's global CLAUDE.md) to not enforce real resource caps on this
  host at all, which is arguably the deeper reason it needed replacing, not just the deadlock bug.
  A genuine local VM sidesteps that class of problem entirely: hardware-virtualization resource
  limits (vCPU count, guest RAM ceiling) are enforced by the hypervisor, independent of the host's
  own cgroup/scheduler quirks. This host **can** do this: `/dev/kvm` exists, `VT-x` reports "full"
  support, `qemu-system-x86` has an installable apt candidate (not yet installed), and user
  `metavacua` is already in the `kvm` group (K27/K28) — the earlier "no local VM tooling available"
  conclusion (K16/K19) was checking only already-*installed* tools, not installable ones; this was
  a real gap in that research pass, corrected here rather than papered over.
- **Decision**: Provision a local QEMU/KVM microVM (`qemu-system-x86_64 -enable-kvm`, sized with an
  explicit `-m <RAM>`/`-smp <vCPUs>` ceiling) as the execution boundary for any `goose` session
  running the `larql.rs` backend and its spawned `larql chat` child (K22 — both processes, not
  per-CLI-call). This is the only
  option evaluated so far that is simultaneously (a) local, (b) a real kernel-isolated VM, and
  (c) free of new cloud dependency/billing.
- **What survives from the two prior revisions**: Fly Machines (`deploy/fly/`) remain documented
  as a legitimate option for a *future*, genuinely remote/production deployment — not this design's
  dev-time self-play loop. `process_manager.rs`'s liveness-monitoring *pattern* (persistent piped
  process, `alive`/`uptime_secs`) remains a valid design reference for T14 regardless of the
  isolation mechanism, adapted here to monitor a QEMU child process / guest-agent heartbeat instead
  of a Docker container or remote Fly exec channel.
- **Consequences**: Requires `sudo apt-get install qemu-system-x86 qemu-utils` (one-time, elevated —
  flagged for explicit confirmation before T10 runs it, same as any other new-software install)
  plus a guest disk image capable of running `larql`/`goose` (a minimal Linux guest — Alpine or
  Debian cloud image — provisioned via `cloud-init` or a hand-built qcow2, decided at T10
  implementation time). Guest-to-host communication (starting sessions, retrieving results) needs
  an explicit channel — `virtio-serial`/`vsock`, SSH over a host-only network device, or a shared
  9p/virtiofs mount — not yet chosen; T10 must record the actual choice, not leave it implicit.
  Slower to iterate than a container (real guest boot time) but that cost buys the actual isolation
  strength requested.
- **Alternatives considered and rejected**: Fly Machine (rejected — remote, not local, per explicit
  clarification, despite being a real VM); local Docker sandbox (rejected — real isolation class
  mismatch, shared kernel); `nix`-provisioned qemu (rejected only as *redundant* — apt already gives
  a working path without first installing Nix); resource-bounded subagent with ulimits only
  (rejected — no independent kill switch, same shape as the already-rejected `larql-probe`).

### ADR-3: Self-learning mutation path is LQL `INSERT`/`COMPOSE` → `COMPACT MAJOR` → `COMPILE INTO VINDEX`

- **Status**: Accepted (2026-07-15)
- **Context**: This repo's core invariant is "edits are structural patches, not fine-tuning."
  `COMPILE INTO VINDEX` is confirmed gradient-free (pure `down_weights.bin` column rewrite via
  `install_edge`); `COMPACT MAJOR` is confirmed gradient-free (MEMIT closed-form). No extract-level
  gate currently blocks these at `browse` level (K6), so no new capability grant is required to run
  them.
- **Decision**: Reuse these primitives as-is rather than inventing a new mutation mechanism. The
  self-play driver's only new responsibility is *deciding what facts to INSERT* from a successful
  trajectory and *when to consolidate* — not building new weight-editing machinery.
- **Consequences**: Directly inherits the four confirmed open bugs on this path (#237 alpha_mul
  miscalibration, #261 silent capacity-collision skip, #238 multi-subtoken non-chaining, #252
  unremovable patches) as **blocking prerequisites**, not implementation risk to discover later —
  they are already known and must be fixed or explicitly worked around before Phase 3 can produce
  a trustworthy "learned" claim (AC-3, AC-4 exist specifically to make failures of these bugs
  visible rather than silently swallowed).
- **Alternatives considered**: gradient fine-tuning of a SmolLM2-135M copy outside LARQL entirely
  (rejected — abandons this repo's entire thesis and the reason LARQL is the vehicle for this task
  at all).

### ADR-4: Evaluator is an external test suite, never the model/agent's own judgment

- **Status**: Accepted (2026-07-15)
- **Context**: residual C4 / E3 — this org's own RDL history already produced one "self-play" design
  that was a structural misunderstanding (single-process self-scoring dressed up as self-play).
- **Decision**: Every scored round's pass/fail comes from running an independent test command
  (exit code 0/non-zero) against the task's own reference suite; the actor (SmolLM2-135M/Goose)
  never sees or influences the scoring logic.
- **Consequences**: Requires curating or selecting a task set with real, runnable, independent
  tests (residual Q2) — this is nontrivial authoring/curation work, not a placeholder to defer
  indefinitely.
- **Alternatives considered**: Shannon bits/char self-consistency score as a proxy reward (rejected
  — residual K3 confirms no such score currently reaches any mutation path, and reusing it would
  just be self-scoring under a different name, the exact failure mode this ADR exists to avoid).

### ADR-5: `larql-cli`/`larql-repl` gain an intrinsic Layer-1 resource governor, as a complement to (never a substitute for) ADR-2's VM boundary

- **Status**: Accepted (2026-07-15). Trigger: the user separately flagged "optimizing larql-cli
  and larql repl for resource awareness" as important — research surfaced that this is not a new
  idea to invent but an already-triaged, evidence-backed cluster of >=20 open issues in this repo
  (residual K31-K33), anchored by issue #182, whose stated invariant ("no command may crash the
  host — intrinsic, default, no external controls") is close to a verbatim match for what the user
  asked for.
- **Context**: The whole reason ADR-2 exists is that `larql-probe` (an *external* control) failed
  (#246) and this host is demonstrably resource-constrained (K15). #182's own argument is that
  external controls are the wrong layer *in general*, not just that this particular one is broken —
  safety should be a property of `larql` itself, applying to every command "including commands not
  yet individually hardened, and commands not yet written." This is a stronger, more durable fix
  than any external wrapper (VM, container, or probe script) can be on its own: a VM boundary stops
  a runaway process from taking the *host* down, but a command can still fail ungracefully *inside*
  the boundary (OOM-killed inside the guest, no diagnostic) without an intrinsic governor.
- **Decision**: Implement issue #185 (CPU governor: default rayon pool to `nproc - 1`, cap/disable
  default busy-spin `spin_pool` sizing) and issue #211 (memory governor: anon-RSS watchdog on
  `/proc/self/status`, self-derived ceiling `available_ram * 0.85`, graceful abort with a clear
  diagnostic on breach, optional `setrlimit(RLIMIT_AS, ...)` backstop tuned to not false-trip on
  LARQL's own mmap'd vindex files) as new, additive `larql-cli` `main()`-startup behavior — both
  are Layer 1 of #182's two-layer design and are scoped, evidenced, and unimplemented, i.e. ready
  to build directly rather than needing further design work here.
- **Consequences**: This work is genuinely independent of the goose/self-play track (touches
  `larql-cli`'s `main()` and `larql-compute::cpu::spin_pool`, not `larql-lql`'s patch/compose path
  or anything goose-fork-side) and directly de-risks T9 (SmolLM2-135M resource-fit measurement,
  which is exactly the "COMPOSE INSERT on a memory-constrained host" scenario issue #239 describes)
  and every other Phase 3 task that runs `larql` commands, inside or outside the VM. Layer 2
  (per-command preflight/streaming — #167/#170/#178/#180/#189 etc.) is explicitly out of scope for
  this ADR; it is real, valuable, already-triaged work but is a much larger surface than this
  design needs to close to make its own self-play loop safe to run, and is left to its own separate
  effort rather than silently expanding this plan's scope.
- **Alternatives considered**: rely solely on ADR-2's VM boundary and skip intrinsic governance
  (rejected — the VM protects the *host*, not the *guest's own diagnostics*; a governor-less
  `larql` OOM-killed inside the guest with no diagnostic is still a debugging dead end, just a
  contained one); implement the full Layer 2 per-command streaming fixes now (rejected as
  out-of-scope-for-this-design — real work, but belongs to issues #166/#192's own umbrella tracking,
  not bundled into the self-play plan).

## 5b. Host toolchain inventory (VM/guest-build side)

Requested explicitly (2026-07-15): identify the full toolchain up front rather than discover it
one blocker at a time. All entries below are host-side packages, small (largest single addition
so far ~1GB for `libguestfs-tools`, which pulls in a full kernel package as a dependency), and
installed autonomously per the user's standing authorization ("absent of them being many GBs in
size, the toolchain is permissible to install... autonomously... without setting up new accounts
or changing permissions or messing with cgroups [outside the VM]").

| Tool | Status | Purpose |
|---|---|---|
| `qemu-system-x86`, `qemu-utils` | Installed | The hypervisor itself (KVM-accelerated) + `qemu-img` for overlay/disk management |
| `cloud-image-utils` (`cloud-localds`) | Installed | Builds cloud-init seed ISOs for guest first-boot config |
| `cpu-checker` (`kvm-ok`) | Installed | Confirms KVM acceleration is actually usable (residual K29) |
| `libclang-dev` | Installed | Needed transitively — `goose-local-inference`'s unconditional `llama-cpp-sys-2` dependency needs `libclang.so` for its bindgen build step, even though our `larql.rs` backend doesn't use llama.cpp itself |
| `libguestfs-tools` (`virt-customize`, `guestfish`, `virt-copy-in`) | Installed (~1GB, pulls in a full `linux-image-amd64` kernel as a dependency of its internal appliance) | **Primary guest-provisioning mechanism** (user preference, 2026-07-15): bakes files directly into a guest qcow2 offline, before boot, via libguestfs's own KVM-backed appliance — sidesteps both the guest's missing 9p kernel modules (confirmed absent from the Debian 12 `cloud-amd64` kernel flavor, residual finding this session) and the need for any runtime network/SSH/credential mechanism |
| `dosfstools`, `mtools` | Installed (superseded) | An earlier attempt (raw FAT data-disk populated via `mcopy`, attached as a second virtio-blk device) — abandoned in favor of `virt-customize` per stated preference, but left installed since they're small and harmless; not part of the primary path going forward |
| Rust stable toolchain, `cargo` | Already present | Builds both `larql-cli` (this repo) and the `metavacua/goose` fork |

**Still needed, not yet installed/built (identify now, don't discover later):**

| Tool/artifact | Purpose | Note |
|---|---|---|
| A full `goose` binary built from `metavacua/goose` with our `larql.rs` backend registered | The actual coding-agent binary to run inside the guest | So far only `cargo check -p goose-local-inference` has been run (confirms `larql.rs` compiles) — a full `cargo build --release -p goose-cli` (or whatever the actual bin crate is) has not been attempted; this is a much bigger build than `larql-cli`'s (the goose workspace has ~12 crates) and needs its own resource-aware, delegated build pass |
| A way to get that `goose` binary + its runtime assets into the guest | Same mechanism as `larql`/vindex — `virt-customize --copy-in` once built | Not yet needed until the goose binary itself exists |
| Guest-side `systemd` unit(s) for the autonomous LARQL-then-Goose boot sequence (design note, §"Autonomous startup ordering" above) | Closes the loop so the VM boots directly into a running self-play session with no host-side interactive driving | Not yet authored |
| `flyctl` | **Not needed** — ADR-2 no longer uses Fly Machines | Explicitly dropped, listed here only so it isn't silently re-added later |

## 6. Open items carried into Phase 2 (planning)

- Q2 (task-suite sourcing/curation) is resolved (plan T8, done — see plan doc). Q5 (135M resource
  fit) remains genuinely open.
- ADR-2 is now a locally-verified, working QEMU/KVM boundary (residual K29/K30) — the `flyctl`
  prerequisite this section originally named no longer applies; superseded by K27-K30.
- The four inherited patch-path bugs (#237/#238/#252/#261) need a fix-or-workaround decision per
  bug before AC-3/AC-4 can be verified — Phase 2 must sequence these ahead of any self-play round
  that depends on them.
- ADR-5's two governor issues (#185, #211) are new plan tasks, sequenced early (they de-risk T9
  and every other Phase-3 task that runs real `larql` commands) — see plan doc T16/T17.
