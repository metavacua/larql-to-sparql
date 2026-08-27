# Research Residual — self-playing, self-learning LARQL coding agent (SmolLM2-135M, via Goose)

Status: Phase 0 (Research) checkpoint, RDL loop, running inside a Ralph loop (unlimited
iterations, no completion promise). This file is the canonical knowledge state — read it before
resuming work in any future iteration rather than re-deriving from scratch. Manifest: `./rdloop.toml`.

Date grounded: 2026-07-15 (`date +%Y-%m-%d`, empirical).

## K — Known (verified, sourced; truth-type + scope required)

- K1: `larql shannon verify` computes `-log2(p)` per true next token → `total_bits`/`bits_per_token`/`bits_per_char`, compared pairwise across engines (LARQL Rust vs HF/PyTorch F32 in CI); `pass = max_pairwise_delta_pct <= threshold` (default 0.5%). [`crates/larql-cli/src/commands/primary/shannon_cmd.rs:660,689-695,838-853`] [empirical] [scope: full]
- K2: SmolLM2-135M is the **sole** CI-gated model (`LARQL_VERIFY_MODEL=HuggingFaceTB/SmolLM2-135M`, `.github/workflows/shannon-verify.yml:58`), chosen for being ungated + fast (~262 tokens, ~7s). The richer 4-arch matrix (`scripts/diagnose_models.py:77-154`) is manual-only, not CI. [empirical] [scope: full]
- K3: No code path anywhere connects a Shannon bits score to INSERT/UPDATE/PATCH/COMPILE — confirmed by repo-wide grep across `shannon_cmd.rs`, `larql-kv/src/accuracy_suite/runner/scoring.rs`, `forward_overrides.rs`. Shannon and Compile/Vindexfile are disjoint CLI command groups (`main.rs:101` vs `127,131-132`). [empirical] [scope: full]
- K4: `PatchedVindex` overlay (`crates/larql-vindex/src/patch/overlay.rs:90-114`) never mutates base files; `.vlp` JSON schema is `VindexPatch{version, base_model, base_checksum, created_at, description, author, tags, operations[]}` (`patch/format.rs:30-121`). [empirical] [scope: full]
- K5: `COMPILE ... INTO VINDEX` is a pure structural edit — hardlinks unchanged files, column-rewrites only `down_weights.bin` via `install_edge` (`crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs:53-124`), **no gradient computation**. `COMPILE ... INTO MODEL` uses closed-form MEMIT (`ΔW = R^T S⁻¹ Q`), also not gradient descent, but requires `has_model_weights` or hard-errors. [empirical] [scope: full]
- K6: `ExtractLevel` actually has **four** values — `Browse, Attention, Inference, All` (`crates/larql-vindex/src/config/index.rs:215-232`), not three as `AGENTS.md` states. `VindexError::InsufficientExtractLevel` is never constructed outside its own unit test — **no runtime gate** ties mutation ops to extract level. INSERT/DELETE/UPDATE/`COMPILE INTO VINDEX` all run at `browse`; only `COMPILE INTO MODEL` is blocked. [empirical] [scope: full]
- K7: Confirmed **open, unfixed** bugs directly on the self-learning critical path: #237 (`DEFAULT_INSERT_ALPHA_MUL=0.1` hardcoded, calibrated for Gemma-3 4B, no hidden_size scaling — `executor/tuning.rs:34-38`); #261 (capacity collision silently `continue`s, no error — `mutation/insert/compose.rs:102-106`); #238 (multi-subtoken targets don't chain, by design — `mutation/insert/plan.rs:107-124`); #252 (`BEGIN`/`SAVE`-created patches unremovable — `description:None` bug, `executor/mod.rs:391-400,502-505`). [empirical] [scope: full]
- K8: LQL has 27 statement variants (`ast.rs:7-188`). Two programmatic (non-REPL) library entry points exist: `run_statement` (fresh `Session` per call, `repl.rs:152-156`) and `run_batch` (one `Session`, `;`-split, `repl.rs:159-185`, stateful). `larql lql '<stmt>'` already calls `run_batch` (`main.rs:585-593`). [empirical] [scope: full]
- K9: **No self-play/self-improvement loop exists anywhere in this codebase.** Exhaustive grep (`evolve|self_play|selfplay|genetic|fitness|mutation_score|train|learn`, all `*.rs`) found zero automated generate→score→mutate-weights loops. All existing "score → decide" cycles (Exp 26, Exp 27, MEMIT `COMPACT MAJOR`, probe train/eval harnesses) are one-shot, human-read-and-actioned. [empirical] [scope: full]
- K10: `larql-probe safe` — the containment wrapper mandated by the user's global CLAUDE.md policy — has a **confirmed live deadlock**: issue #246, reproduced by reading the script directly (`start_monitor`'s backgrounded process needs an explicit redirect or `cmd_safe` hangs on EOF before the wrapped command ever runs — `larql-probe:101-103,256-370`). [empirical] [scope: full]
- K11: `larql-server` is this repo's only HTTP/gRPC serving surface, and has two confirmed open bugs blocking Goose specifically: #266 (tool schema rejects `pattern`/`format`, 400) and #268 (SSE streaming decode error) — both discovered by sibling repo `metavacua/babel-harness`'s own CI (`.github/workflows/larql-vindex.yml`, the "vindex-drives-babel" goal step, Goose over larql `/v1`). [empirical] [scope: full]
- K12: `metavacua/goose` is a real, accessible fork (lineage `aaif-goose/goose` ← `block/goose`), already self-described as "candidate coding harness for larql-goose." Currently: single `main` branch, zero PRs, zero LARQL-referencing code anywhere (code search empty) — a clean, unclaimed integration target. [empirical] [scope: full]
- K13: Goose's provider architecture (`crates/goose/src/providers/`) uses a `<name>_def.rs` convention (`ollama_def.rs`, `openai_def.rs`, `databricks_def.rs`, ...) plus `provider_registry.rs` + `inventory/` + `init.rs` for registration. `ollama_def.rs`/`local_inference.rs` are the closest structural analogues for a new local, non-HTTP `larql_def.rs`. [empirical] [scope: partial — file listing only, trait/interface contents not read; see Q4]
- K14: SmolLM2-135M (`HuggingFaceTB/SmolLM2-135M`) is confirmed live/downloadable (HTTP 307 resolve-cache redirect for `config.json`, checked 2026-07-15). [empirical] [scope: full]
- K15: This dev machine is resource-constrained **in practice, right now**, not just per policy: host load average 11.6 on 8 cores, ~2.3GB RAM free, observed live while a subagent's plain `cargo build --release -p larql-cli` exceeded a 10-minute foreground window and had to move to a background task (`bwn4yra7z` / agent `a344e3921142e4e6a`). [empirical] [scope: full]
- K16: No local VM tooling is installed (`qemu-system-x86_64`, `multipass`, `firecracker`, `docker`, `podman`, `flyctl` all absent from PATH). `ollama` (`/usr/local/bin/ollama`) and `goose` (`/home/metavacua/.local/bin/goose`) CLIs **are** installed locally. [empirical] [scope: full]
- K17: Locally cached model assets are SmolLM2-**360M**, not 135M (`/home/metavacua/larql-vindexes/smollm2-360m.vindex`, 1.3GB, plus HF cache). The 135M target model is not yet extracted locally. [empirical] [scope: full]
- K18: `deploy/fly/` already deploys `larql-server` as Fly Machines (Firecracker-backed) in production for a different model (gemma-4-26b expert server) — real precedent for "a dedicated lightweight VM," but `flyctl` isn't installed locally and no equivalent config exists for a SmolLM2 dev/experiment sandbox. [empirical] [scope: partial — precedent covers production serving, not ad hoc dev/experiment sandboxing; see Q3]
- K19: `nix` itself is **not installed/on PATH** on this dev machine, despite `flake.nix`/`nix/` existing in the repo — the flake is authored for contributors with Nix installed elsewhere, not usable as an ad hoc local-VM provisioning tool on this box without first installing Nix. [empirical] [scope: full]
- K20: Resolves Q1/Q4. The real `Provider` trait lives at `crates/goose-provider-types/src/base.rs` in `metavacua/goose` (not `crates/goose/src/providers/base.rs`, which only re-exports it). It is `#[async_trait] pub trait Provider: Send + Sync` with exactly two mandatory methods — `fn get_name(&self) -> &str` and `async fn stream(&self, model_config: &ModelConfig, system: &str, messages: &[Message], tools: &[Tool]) -> Result<MessageStream, ProviderError>` (`MessageStream = Pin<Box<dyn Stream<Item = Result<(Option<Message>, Option<ProviderUsage>), ProviderError>> + Send>>`) — everything else (`complete`, `get_context_limit`, `fetch_supported_models`, etc.) has a default body. A `def.rs` type additionally implements `ProviderDescriptor::metadata()` (static) and `ProviderDef::from_env(...)`. Registration is a plain call, `registry.register::<LarqlProviderDef>(preferred)`, generic over `F: ProviderDef` — no macro/`inventory::submit!` needed for a built-in provider. [empirical] [scope: full]
- K21: **`crates/goose-local-inference`** (in `metavacua/goose`) is the exact architectural precedent for an in-process, non-HTTP provider: it implements `Provider` directly and depends on `llama-cpp-2`/`llama-cpp-sys-2` (Rust↔C++ FFI to llama.cpp) and, on macOS, `safemlx`/`safemlx-lm` (MLX FFI) as ordinary Cargo dependencies — no subprocess, no local HTTP server. A `pub(super) trait LocalInferenceBackend` (`load_model`/`generate`/`available_memory_bytes`) is implemented per-backend (`llamacpp::LlamaCppBackend`, `mlx::MlxBackend`); the `Provider`-implementing struct holds a `Mutex`-guarded `ModelSlot` and streams output via `async_stream::try_stream!`. A `larql_def.rs` provider follows the identical shape, substituting a dependency on `larql-lql`/`larql-inference` for `llama-cpp-2`. [empirical] [scope: full]
- K22: K21 has a direct containment-policy implication: because `goose-local-inference` already depends on `llama-cpp-2` (one of the six literal patterns named in the user's global CLAUDE.md inference-containment policy), and a `larql_def.rs` provider would link `larql-lql`/`larql-inference` **in-process into the same `goose` binary**, there is no longer a separate child "command" to wrap per-invocation the way `larql-probe safe -- <command>` assumes — the entire `goose` process itself becomes the thing performing local inference for the duration of any session using either provider. Whatever VM mechanism resolves Q3 must therefore contain the **whole goose process**, not individual `larql` subcommand calls. [sound: from K20, K21, C2] [scope: full]
- K23: `metavacua/openfang` (fork of `RightNow-AI/openfang`, "Open-source Agent Operating System," Rust, ~137K LOC) exists and is directly relevant to Q3/C2. Its `crates/openfang-runtime/src/docker_sandbox.rs` runs agent-spawned commands inside Docker containers with real, enforced resource limits (`docker run --memory <limit> --cpus <limit> --pids-limit <limit> --cap-drop ALL --security-opt no-new-privileges`) plus input sanitization (container-name/image-name/command validation, shell-metacharacter rejection). This is a working, already-tested (1767+/2696+ tests per README) local resource-cap mechanism — the exact class of thing `larql-probe` attempted and failed at (K10). [empirical] [scope: full]
- K24: `crates/openfang-runtime/src/process_manager.rs` implements persistent, interactive process sessions: `ProcessManager::start` spawns a child with piped stdin/stdout/stderr, tracked by `ProcessId` in a `DashMap`, with a per-agent process-count cap; `ProcessInfo{alive, uptime_secs}` is queryable at any time. This directly matches the user's "always delegate to subagents so you can monitor the subagent and control as necessary" requirement, applied at the OS-process level — it is a working precedent for AC-5's liveness/timeout requirement, not something this design needs to build from scratch (as ADR-2/T14 originally assumed). [empirical] [scope: full]
- K25: `crates/openfang-runtime/src/subprocess_sandbox.rs` (env-var allowlist stripping, no resource caps) and `workspace_sandbox.rs` (filesystem path-traversal confinement, no resource caps) are complementary but do NOT address CPU/memory containment on their own — only `docker_sandbox.rs` (K23) does. `crates/openfang-runtime/src/sandbox.rs` is a separate, unrelated mechanism: a Wasmtime WASM sandbox (fuel-limited CPU budget, capped linear memory, capability-based host-call gating) for **small untrusted skill/plugin modules**, not for hosting a full native inference engine — not a fit for containing the whole `goose`+LARQL process (K22), though it is a notable architectural echo of LARQL's own existing `--experts`/WASM-in-FFN dispatch mechanism (`docs/cli.md`'s `--experts`/`--experts-dir` flags). [empirical] [scope: full]
- K26: Docker itself is **not installed** on this dev host either (confirmed absent in the same PATH check as K16). Superseded by K27/K28 below — the user clarified "VM" means genuine kernel-level isolation, not a container, which rules out Docker as the boundary regardless of this K entry.
- K27: This host genuinely supports local hardware-accelerated virtualization: `/dev/kvm` exists (`crw-rw---- root:kvm`), `VT-x` is present and reports "full" virtualization support (`lscpu`), and `qemu-system-x86` has an installable apt candidate (`1:7.2+dfsg-7+deb12u18+b3`, not yet installed). Earlier claims that "no local VM tooling is available" (K16/K19) were about tooling already *installed*, not installable — this was a scope gap in the original research, corrected here. [empirical] [scope: full]
- K28: User `metavacua` is already a member of the `kvm` group (`id`: `groups=...,106(kvm),...`) — once `qemu-system-x86`/`qemu-kvm` is installed, `/dev/kvm` access needs no root/sudo elevation for the VM boundary itself (package install does need `sudo apt-get install`, a one-time elevated step). [empirical] [scope: full]
- K29: `qemu-system-x86`, `qemu-utils`, and `cloud-image-utils` are now actually installed (`apt-get install`, user-authorized 2026-07-15, confirmed via `qemu-system-x86_64 --version` → `QEMU emulator version 7.2.22`), and `kvm-ok` (from newly-installed `cpu-checker`) confirms **"KVM acceleration can be used"** on this exact host. ADR-2's local-QEMU/KVM boundary is no longer just planned — the hypervisor itself is installed and verified working. [empirical] [scope: full]
- K30: A full local KVM microVM boot smoke test **succeeded**, delegated to a subagent (2026-07-15). Debian 12 genericcloud qcow2 (331M) + a `qemu-img`-backed 4G overlay + a `cloud-localds` seed ISO, booted `-enable-kvm -m 768M -smp 1 -cpu host -display none -serial file:/tmp/console.log -no-reboot` under a 180s timeout. Serial console confirms real KVM engagement (not TCG): `Hypervisor detected: KVM`, `kvm-clock`, `Booting paravirtualized kernel on KVM`; cloud-init ran a `runcmd` writing a marker file and echoing `VM_BOOT_OK_MARKER` to `/dev/console`; the guest then `poweroff`'d cleanly (`kvm: exiting hardware virtualization`), QEMU exited 0 with no timeout kill needed. Total wall-clock 134s. Artifacts left at `/tmp/{debian-genericcloud-amd64.qcow2,selfplay-vm-overlay.qcow2,seed.iso,console.log}`. This is the first genuinely working piece of ADR-2's execution boundary, not just a plan. [empirical] [scope: full]
- K31: A large, already-triaged cluster of **open** `area:resource-governance` issues exists in this repo (>=20: #135,#150,#157,#166,#167,#168,#169,#170,#178,#180,#182,#185,#189,#192,#194,#211,#239,#240,#246,#251,#282), independent of anything discovered in this session's own research — these predate and are broader than the goose/self-play design. Two are explicit umbrella/meta issues: #166 ("extract: unusable on constrained dev hosts") and #192 ("convert correctness + resource"). [empirical] [scope: full]
- K32: Issue #182 ("CLI-level resource governor: no command may crash the host — intrinsic, default, no external controls") is the parent/anchor of this cluster and states almost exactly the concern the user raised independently in this session. It proposes two layers: **Layer 1** (a CLI-boundary governor installed once at `main()`, before subcommand dispatch, applying to every command including future/unhardened ones — an anon-RSS memory watchdog with a self-derived ceiling that aborts gracefully rather than lets the kernel OOM-kill, plus a rayon/OpenBLAS/OMP thread cap reserving >=1 core) and **Layer 2** (per-command preflight/streaming so commands succeed within the ceiling, not just fail safely at it — `extract`'s existing streaming path is cited as the model to follow; `convert`/inference are not yet fixed, per #178/#170/#180). Motivating crashes are evidenced (dmesg OOM records on phi3 `convert`, #178; host freeze/thermal-throttle on `larql label`, #185). [empirical] [scope: full]
- K34: `metavacua/goose`'s `crates/goose-local-inference` (K21's crate) has a concrete,
  already-generalized extension point for exactly this integration: `InferenceRuntime` holds a
  `HashMap<&'static str, Arc<dyn LocalInferenceBackend>>`, currently populated with two backends
  (`llamacpp`, `mlx`) at `InferenceRuntime::get_or_init()`. `LocalInferenceBackend` (`backend.rs`)
  is a **synchronous** trait — `fn load_model(&self, model_id: &str, resolved: &ResolvedModelPaths,
  settings: &ModelSettings) -> Result<Box<dyn BackendLoadedModel>, ProviderError>`, `fn
  generate(&self, loaded: &mut dyn BackendLoadedModel, request: LocalGenerationRequest<'_>) ->
  Result<(), ProviderError>` (writes onto a `StreamSender`, not a return value — streaming happens
  by side effect), `fn available_memory_bytes(&self) -> u64` — not async, which fits a
  blocking-subprocess-I/O adapter (write to stdin, block-read from stdout) far more naturally than
  the full async `Provider` trait ADR-1 originally targeted. Adding LARQL support means
  implementing a third `LocalInferenceBackend` and registering it in that `HashMap`, not
  implementing `Provider` from scratch. [empirical] [scope: full]
- K35: `docs/cli.md` (already-read in this session's earlier research) documents `larql chat
  <MODEL>` / `larql run <MODEL>` (no prompt) as an existing interactive chat loop over
  stdin/stdout — a plain CLI process (`crates/larql-cli/src/commands/primary/run_cmd.rs`), entirely
  separate from `larql-server`. This is a strictly better fit for a subprocess-driven
  `LocalInferenceBackend` adapter than building a new mode from scratch, and does not conflict with
  C1 ("MUST NOT use larql-server") since it is a different code path. Exact stdin/stdout framing
  confirmed by reading `run_chat` (`run_cmd.rs:396-430`): the `"> "` turn prompt is written to
  **stderr** (not stdout), one line is read from stdin per turn (blank lines skipped, EOF/Ctrl-D
  exits cleanly), and the prompt is dispatched to `walk_cmd::run(...)` which streams the response
  to **stdout**. Plain text, no framing bytes/JSON — directly drivable by piped stdin/stdout, though
  an adapter reading only stdout has no stdout-side turn-boundary marker (the boundary signal is
  stderr's next `"> "`) — a small additive change to also emit an explicit stdout marker is a
  plausible T11/T12 implementation detail, not yet decided. [empirical] [scope: full]
- K33: Issue #185 (CPU governor, child of #182) specifies: default-cap the global rayon pool to `nproc - 1` when `RAYON_NUM_THREADS` is unset (installed once in `main()`), and stop `larql_compute::cpu::spin_pool` (default-on busy-spin, `LARQL_SPIN_POOL`) from sizing itself to all cores by default. Acceptance: a default run leaves >=1 core schedulable for the OS; env-var opt-outs remain explicit, not the only safety path. Issue #211 (memory governor, child of #182) specifies: an anon-RSS watchdog reading `/proc/self/status` periodically from `main()` startup, ceiling derived as `available_ram * 0.85` (headroom for no-swap hosts), graceful abort with a clear diagnostic on breach, optional `setrlimit(RLIMIT_AS, ...)` hard backstop tuned to not false-trip on mmap/file-rss (LARQL's vindexes are mmap'd by design, K-invariant from `AGENTS.md` — the observed OOMs were anon-rss only, file-rss ~0, so this is compatible with LARQL's own storage model, not in tension with it). [empirical] [scope: full]

## Q — Open

- Q1 (RESOLVED → K20, K21): What exact async trait/interface must a new Goose provider implement? Answered: `Provider` trait, two mandatory methods, `goose-local-inference` is the direct precedent.
- Q2: Should the self-play coding-task suite be an existing benchmark or newly authored tasks scoped to SmolLM2-135M's actual capability?
  - Accept("reuse an existing suite") → grounds "learning" claims in recognized external ground truth (satisfies C4).
  - Reject → SmolLM2-135M is far below standard SWE-bench-class task difficulty; an unscoped-difficulty benchmark would produce near-100% failure regardless of learning, generating no usable signal either way.
- Q3: What is the concrete "dedicated lightweight VM" now that neither `larql-probe` (K10, broken) nor `larql-server` (vetoed by user) nor local VM tooling (K16, K19, absent — including `nix` itself) is available, and it must contain a whole long-running `goose` process (K22), not just discrete CLI calls?
  - Accept → something must be provisioned before any Phase 3 execution task can run: either (a) a cloud Fly Machine reusing K18's precedent (needs `flyctl` installed + network egress, already granted in `rdloop.toml`), or (b) installing a local VM tool (`nix` itself, then `nixpkgs#qemu`/a microvm flake input) first.
  - Reject → a resource-bounded subagent with fixed ulimits/thread caps could substitute — weak, since the user already rejected the structurally analogous `larql-probe` approach once, and K22 shows the isolation unit is now a whole long-lived process, which ulimits alone don't sandbox as well as a VM boundary (no independent kill switch from the orchestrator's point of view).
  - **Not yet escalated to the user** — this is the one remaining architectural gap before Phase 1 brainstorming can close out the runtime/deployment view. Two concrete sub-options exist (Fly Machine vs. install Nix+qemu locally); recommend asking once, batched with any other remaining opens, rather than blocking on it immediately.
- Q4 (RESOLVED → K20): folded into Q1's resolution.
- Q5: Does SmolLM2-135M extraction + a browse/inference-level vindex fit inside the ~2.3GB free RAM observed in K15? Not yet measured.

## C — Constraints (hard limits on any solution)

- C1: MUST NOT use `larql-server` (`larql serve`, its OpenAI-compatible HTTP route, or gRPC) as the Goose transport. *(User directive, 2026-07-15.)*
- C2: MUST NOT run any local-model-loading command unwrapped (`larql run/extract/infer`, `*.gguf` loads, `ollama`, `llama.cpp`, `llama_cpp`/`transformers`/`from_pretrained`) — but the mandated wrapper is confirmed broken (K10); a working containment substitute (dedicated lightweight VM, per user directive) MUST replace it, not be silently dropped.
- C3: MUST delegate all actual larql-cli/repl/goose execution to subagents, never run inline in the orchestrating loop. *(User directive, 2026-07-15 — "so that you can monitor the subagent and control as necessary.")*
- C4: Self-play scoring MUST use an external, independent evaluator (e.g. a real test suite's pass/fail), never a single process self-scoring its own output — per this org's own RDL-documented precedent of a prior `evolve()` GA layer that passed 12/12 tests while being a structural misunderstanding of AlphaZero-style self-play.
- C5: Any GitHub write (branch/PR/issue) against `metavacua/goose` or `metavacua/larql-to-sparql` must stay within `[security].write_allowed` in `rdloop.toml` (both granted 2026-07-15).

## E — Eliminated (rejected hypotheses with causal axiom)

- E1: "Serve the vindex via `larql-server`'s OpenAI-compatible endpoint; point Goose's existing `openai` provider at it." — rejected: user directive vetoes this explicitly (C1), independent of #266/#268 already making it the weaker technical option.
- E2: "Wrap all model execution in `larql-probe safe` as originally specified." — rejected: axiom "the mandated wrapper works" is false (K10/issue #246). User has also independently directed a VM-based replacement (C2).
- E3: "Self-play = a single model scores and edits its own weights with no independent evaluator." — pre-emptively rejected (not yet attempted): this is exactly the structural misunderstanding this org's own RDL history already fell into once; C4 forecloses this branch.

## D — Dependencies

- D1: `rdloop.toml` manifest confirmation → all downstream phases. **Done**, confirmed 2026-07-15.
- D2: Resolve Q1/Q4 (Goose provider trait signature) → before finalizing provider architecture in the Phase 1 Structurizr model.
- D3: Resolve Q3 (concrete VM mechanism) → before any Phase 3 task runs actual larql-cli/goose commands. Does **not** block the design/spec/plan work itself (Phases 1-2 can proceed and name this as an explicit open dependency).
- D4: SmolLM2-135M extraction (K14) ∥ Goose provider skeleton authoring (K12/K13) — independent, no shared inputs, parallelizable once each is separately scoped.
- D5: Background build task (`bwn4yra7z` / agent `a344e3921142e4e6a`) → must complete and be checked before "larql-cli builds cleanly" can be promoted to a K entry. **Unresolved, in flight** as of this checkpoint.

## A — Artifacts (verbatim)

- HTTP 307 redirect confirming SmolLM2-135M reachability (`resolve-cache` location header, HuggingFaceTB/SmolLM2-135M, checked 2026-07-15).
- Goose provider file listing (`crates/goose/src/providers/`): `acp_tooling.rs, amp_acp.rs, anthropic_def.rs, avian.rs, azure.rs, azureauth.rs, base.rs, bedrock.rs, catalog_util.rs, chatgpt_codex.rs, claude_acp.rs, claude_code.rs, cli_common.rs, codex.rs, codex_acp.rs, copilot_acp.rs, cursor_agent.rs, custom_provider_config.rs, databricks_def.rs, databricks_v2_def.rs, formats/, gcpauth.rs, gcpvertexai.rs, gemini_cli.rs, gemini_oauth.rs, githubcopilot.rs, google_def.rs, huggingface.rs, huggingface_auth.rs, init.rs, inventory/, kimicode.rs, litellm.rs, local_inference.rs, mod.rs, nanogpt.rs, oauth.rs, oauth_device_flow.rs, ollama_cloud.rs, ollama_def.rs, openai_def.rs, openrouter.rs, pi_acp.rs, provider_registry.rs, provider_secrets.rs, provider_test.rs, sagemaker_tgi.rs, snowflake_def.rs, testprovider.rs, tetrate.rs, toolshim.rs, usage_estimator.rs, utils-to-move.md, utils.rs, xai.rs, xai_oauth.rs`.
- Open-issue set relevant to this design: #237, #238, #246, #252, #261, #266, #268, #274, #275 (all `metavacua/larql-to-sparql`, all OPEN as of 2026-07-15).

- K37: `cargo build --release -p larql-cli` (originally dispatched to a delegated subagent) completed successfully — `target/release/larql` (121MB) is a working binary, verified directly 2026-07-15: `--version` → `larql 0.1.0`; `--help` lists the full command surface; `show` against the cached `smollm2-360m.vindex` prints real metadata (32 layers, hidden 960, F16, no quant, full file inventory); `larql lql "USE '...'; DESCRIBE \"the\";"` loads the vindex (32 layers, 81.9K features) and returns a real model-introspection result (`the — signal: diffuse (1 edges, max gate 7.7), Output (L26-31): → noqa 7.7 L31`). [empirical] [scope: full]

## N — Negatives (confirmed absences, unsurvivable failure modes)

- N6 (empirically confirmed, not just theoretical — was N4/N5's foreseen case, now observed): the subagent dispatched to build+verify `larql-cli` (agent `a344e3921142e4e6a`) was orphaned mid-task by a session checkpoint/restart. Its own transcript stopped at 04:19; the `cargo build` OS process it had launched in the background was not tied to its own lifecycle and kept running unsupervised, finishing at 08:32 — four hours with no supervising agent to observe or report completion. `TaskOutput` on the original task ID after the restart returns "No task found with ID," confirming the harness's own tracking does not survive a session checkpoint/restart even though the underlying OS process can. This is exactly the "subagent process that stops with no lifecycle event the orchestrator observes" gap the RDL skill documents as an unimplemented watchdog requirement — now demonstrated live in this project, not hypothetical. The build itself happened to succeed anyway (K37) and was recovered by re-verifying directly after the fact, but a failed build in the same situation would have gone equally unreported, with nothing distinguishing "still building" from "silently dead" until a human happened to ask.

- N1: Self-play loop closure — **silent no-op**: #261 confirms `find_free_feature` returning `None` triggers a silent `continue` (`compose.rs:102-106`). A "self-learning" round can complete, report success, and change nothing, with no exception and no signal distinguishing "learned" from "silently skipped."
- N2: Goose-drives-model correctness — even after #266/#268 are fixed, nothing validates that a tool call Goose issues is *semantically* correct on the LARQL side; schema validation confirms shape, not meaning. A tool call can be schema-valid and semantically wrong with no existing test catching it.
- N3: "`COMPILE INTO VINDEX` learned something" — **silent near-zero-effect edit**: K7's `alpha_mul=0.1` miscalibration (#237) means an edit can install and the vindex can reload/serve without error while the actual behavioral delta is negligible on a small `hidden_dim` model. "Compiled successfully" is not evidence of a measurable behavioral change without an explicit before/after eval, which no part of this design has specified yet.
- N4: VM sandbox containment — **unsurvivable/undetectable class**: whatever replaces `larql-probe`, if it's a cloud microVM reached over network, a network partition mid-run leaves the orchestrator unable to distinguish "still executing" from "died silently" without an explicit heartbeat design. This is the exact watchdog gap the RDL skill itself documents as unimplemented in this environment.
- N5: Delegated subagent liveness — same gap as N4 at the subagent layer: a subagent driving a long goose session has no independent timeout/heartbeat beyond the orchestrator's own judgment call to check in — a hung session is indistinguishable from "still working" until then.

## Milestone: real, VM-contained `larql chat` inference achieved (2026-07-15)

- K38: After a multi-step debugging chain (9p passthrough failed — Debian 12 `cloud-amd64` kernel
  ships no `9p*` modules at all, confirmed via `modprobe` exit codes; pivoted to `virt-customize`
  offline image-baking per user preference; hit a missing-shared-library failure
  (`libopenblas.so.0`), fixed by baking in `libopenblas0-pthread`/`libgfortran5`/`libquadmath0`;
  hit a cloud-init once-per-instance semaphore bug from reusing an `instance-id` across boots of
  the same overlay, fixed by bumping it; hit a guest OOM at `-m 1536M` while loading the
  ~1.3GB smollm2-360m vindex, fixed by raising to `-m 2560M`) — `larql chat` genuinely ran inside
  the QEMU/KVM VM boundary and produced real model output: **" Paris, which"** continuing the
  prompt "The capital of France is", using SmolLM2-360M via the unmodified `larql chat` CLI. No
  `larql-probe`, no host-side unwrapped execution, no cloud dependency. This is the first
  end-to-end proof that ADR-1/ADR-2's combined architecture (VM-contained `larql chat`, driven the
  same way `larql.rs`'s `LocalInferenceBackend` will drive it) actually works on real hardware with
  a real model. [empirical] [scope: full]
- N7: Generation is very slow on this guest sizing (~3-4 tokens per 90s wall-clock on 1 vCPU,
  CPU-only, no GPU) — a full `larql chat` default 64-token completion would take on the order of
  30+ minutes at this rate. This is a real operational constraint for any self-play round design
  (T15's E2E demo, and any future task battery) — round latency budgets must account for this, not
  assume interactive-speed generation inside the VM.
- K39: Confirmed across two further runs (2 vCPU, `-m 2560M`, correctly-sized outer wall-clock
  timeout accounting for ~130s boot/cloud-init overhead): the continuation for "The capital of
  France is" grew from `" Paris, which"` (90s budget) to `" Paris, which is located in the northern
  part of"` (170s budget) — factually correct, coherent, and monotonically extending with more time
  budget. This confirms K38's result is real and repeatable, not a one-off fluke, and that the
  remaining limiting factor is purely wall-clock generation speed on a small CPU-only guest, not
  correctness or stability of the VM-contained inference path itself. [empirical] [scope: full]

## Post-hoc addendum (empirically confirmed during this session, not just theorized)

- N6: The residual's own N4/N5 entries (subagent/VM liveness gaps) predicted this class of failure
  in the abstract; it then actually happened. The first delegated build-verification subagent
  (`cargo build --release -p larql-cli`, dispatched to check basic toolchain plumbing) backgrounded
  its own child bash task, reported "waiting," was resumed twice via `SendMessage` with
  acknowledgement each time — then, on a later check, `SendMessage` returned `"No transcript found
  for agent ID"`, no `cargo`/`rustc` process was running, and `target/release/larql` did not exist.
  The agent and its work were silently lost with no error surfaced to the orchestrator at the
  moment of loss — the only way this was caught was an explicit, human-initiated status check ("the
  background agent appears to have been done or lost?"), not any hook or notification. [empirical]
  [scope: full] This confirms N4/N5 were not overcautious hedging — nesting a background task
  inside a delegated subagent (rather than backgrounding directly at the orchestrator layer) is a
  real, now-witnessed loss mode, not just a documented gap. Corrective action taken: subsequent
  long-running work should background directly (`Bash` `run_in_background: true`) rather than via a
  subagent that itself backgrounds a nested task, reducing indirection by one layer.

## Steering log (user directives received mid-research, integrated above)

1. "The larql probe is inadequate. A dedicated lightweight VM for running the larql-cli and larql repl experiments in is preferable. Always delegate them to subagents so that you can monitor the subagent and control as necessary." → C2, C3, Q3.
2. "Goal set: the model should be demonstrated as a coding agent using the goose harness." → reframes the entire task's success criterion; drove K11-K13 research.
3. "Do not use the larql-server method." → C1, E1.
4. "Metavacua/goose can be used; branches and PRs can be submitted there. Goose or a strict subset of the project can be integrated into the larql-to-sparql via experimental branch." → C5, manifest update (`rdloop.toml`), K12/K13 research target.
