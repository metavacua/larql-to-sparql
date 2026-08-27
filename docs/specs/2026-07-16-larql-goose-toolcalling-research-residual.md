# Research Residual — tool-calling for the larql-driven Goose coding agent

Status: Phase 0 (Research) checkpoint, RDL loop, continuation of
`docs/specs/2026-07-15-larql-goose-selfplay-research-residual.md`. This file is the canonical
knowledge state for the tool-calling sub-project specifically — read it before resuming work.
Manifest: `./rdloop.toml`.

Date grounded: 2026-07-16 (empirical, four parallel research agents over the local
`larql-canonical`, `larql-main`, and `metavacua-goose` checkouts).

## K — Known (verified, sourced; truth-type + scope required)

- K36: `LarqlBackend` (`crates/goose-local-inference/src/larql.rs` in `metavacua/goose`,
  `feat/larql-local-inference-backend`) receives `LocalGenerationRequest.tools: &[Tool]` on every
  `generate()` call but ignores it entirely — no tool-call parsing or dispatch exists in this
  backend today. [empirical] [scope: full]
- K37: Goose's `llamacpp` backend (`crates/goose-local-inference/src/llamacpp/mod.rs:460-520`)
  decides between native and emulated tool-calling via `should_use_native_tool_calling`
  (mode + a chat-template dry-run through `supports_native_tool_calling`/
  `template_result_supports_native_tool_calling`, requiring `parse_tool_calls` plus a non-empty
  parser string). When emulation is chosen (`use_emulator = !native_tool_calling &&
  !tools.is_empty()`), it swaps `request.system` for `load_tiny_model_prompt() +
  build_emulator_tool_description(tools, code_mode_enabled)` — the same substitution this
  project already applied to `larql.rs` this session (see main design doc's Phase-3 fix). [empirical] [scope: full]
- K38: Two **independent** text-based tool-call emulation implementations exist in
  `goose-local-inference`, not one shared parser:
  1. `llamacpp/inference_emulated_tools.rs` (ungated, ordinary crate-internal visibility) —
     `StreamingEmulatorParser` (153-300), a 3-state machine (`Normal`/`InCommand`/
     `InExecuteBlock`) matching `\n$ command\n` and ` ```execute_typescript` fences in
     **streamed, per-token** text, holding back a small tail buffer so a pattern can't split
     across chunks. Matches become `EmulatorAction::{ShellCommand,ExecuteCode}` →
     `send_emulator_action` (302-351) builds
     `MessageContent::tool_request(Uuid::new_v4(), Ok(CallToolRequestParams::new(...)
     .with_arguments(...)))`.
  2. `tool_emulation.rs` + `native_tool_parsing.rs` (top-level, both `#[cfg(feature =
     "mlx")]`-gated at `lib.rs:16-20`, used only by `mlx.rs`) — `native_tool_parsing.rs` is
     **buffered, whole-response** (not streaming): `message_from_native_tool_text` (10-45) tries
     OpenAI `{"tool_calls":[...]}` JSON, then bare tool-call arrays/objects, then
     `<function=name><parameter=k>v</parameter></function>` XML via regex; `tool_call_content`
     (155-218) converts any match into the same `CallToolRequestParams` shape, validating names
     (`is_valid_function_name`) and coercing string args (`safely_parse_json`).
  Neither is directly reusable from `larql.rs` without new code: (1) is `mlx`-gated and this
  backend's build doesn't enable `mlx` (Apple-only); (2) is `pub(super)`-scoped inside
  `llamacpp::`, not reachable from a crate-root sibling module. [empirical] [scope: full]
- K39: `larql chat`'s `run_chat` (`crates/larql-cli/src/commands/primary/run_cmd.rs:381-418` in
  `larql-main`) is **whole-response-per-turn**, not streaming: one line in via stdin, one
  fully-generated response printed + explicitly flushed to stdout, then the next `"> "` prompt on
  stderr. This is a strictly simpler input shape than `llama.cpp`'s per-token streaming callback
  design that `StreamingEmulatorParser` (K38.1) was built for, and closer to
  `native_tool_parsing.rs`'s buffered whole-response model (K38.2) — but K38.2 is unreachable
  without de-gating or forking. [empirical] [scope: full]
- K40: The real dispatch consumer, `goose/src/agents/reply_parts.rs`'s
  `categorize_tool_requests` (424-563), only requires well-formed
  `MessageContent::ToolRequest{id, tool_call: Ok(CallToolRequestParams)}` in the response
  content — nothing backend-specific. Any new larql-side parser that produces this shape plugs
  into existing, already-tested dispatch unchanged. [empirical] [scope: full]
- K41: `tool_parsing::compact_tools_json` (`tool_parsing.rs:4-18`) — a minimal `{name,
  description}` tool-list serializer — is already ungated and crate-visible, usable by a new
  larql-side implementation without modification. [empirical] [scope: full]
- K42: LARQL's structural-patch mechanism (`PatchedVindex` overlay,
  `crates/larql-vindex/src/patch/overlay.rs:90` in `larql-canonical`; `.vlp` format,
  `patch/format.rs:32`; LQL `INSERT INTO EDGES ... MODE compose`,
  `crates/larql-lql/src/executor/mutation/insert/compose.rs`) is built end-to-end around
  **one-entity-to-one-output-token factual edits**: a single canonical prompt template
  (`"The {relation} of {entity} is"`, `executor/tuning.rs:195-198`) drives a balancer that scales
  a synthesized down-vector until one target token's probability lands in a `[floor, ceiling]`
  band. There is no multi-token target support, no output-template/grammar notion, and `COMPILE`
  bakes existing patches into weights rather than synthesizing new behaviors. [empirical] [scope: full]
- K43: Using K42's mechanism for tool-call-syntax behavior-shaping (rather than single-fact
  correction, its designed purpose) would require either (a) reconstructing a multi-token fixed
  template by chaining several single-token installs across different `(layer,feature)` slots —
  unvalidated, and in tension with `find_free_feature`'s slot-collision avoidance, which exists
  specifically to keep unrelated single-fact edits from clobbering each other — or (b) deriving a
  trigger condition from a *class* of tool-requesting prompts rather than one canonical prompt —
  also unvalidated, since gate vectors are single directions found from one prompt's residual,
  not a class boundary. No eval harness exists anywhere in the codebase for "does the model
  reliably emit valid tool-call syntax" (only per-prompt probability-band checks). [empirical] [scope: full]
- K44: LARQL's introspection tooling — `walk` (per-prompt, per-layer top-K FFN gate feature
  attribution with token attribution via `down_meta`, `crates/larql-cli/src/commands/extraction/
  walk_cmd.rs` in `larql-main`), `circuit-discover` (OV→gate coupling + head clustering,
  `circuit_discover_cmd.rs`), `attention-capture` (cross-prompt attention diffing), and the
  experimental `larql dev ov-rd` harness (per-head ablation/replacement + KL/top-k causal
  evaluation, scoped to one L0H6 research line, explicitly not a stable verb) — covers per-input
  attribution and unsupervised structural clustering, but **no cross-example, labeled-behavior
  correlation tool exists** (confirmed via targeted grep across `larql-vindex`, `larql-cli`,
  `larql-inference` for instruction-following/format-adherence/fenced-code correlation logic:
  zero hits). A "find which feature correlates with correct vs incorrect output-format adherence,
  then patch it" pipeline would need to be built from these primitives, not just invoked. [empirical] [scope: full]
- K45: This repo's established CI strategy-matrix convention
  (`.github/workflows/lql-strategy-matrix.yml` in `larql-canonical`) is: a `plan` job enumerates
  legs as JSON via a Python generator (`scripts/lql_matrix/gen_legs.py`, one dict per leg with a
  fixed field schema), a `build` job compiles the binary once, a `matrix` job runs
  `strategy.matrix.leg: fromJSON(...)` with **`max-parallel: 12`** and `fail-fast: false`
  ("every leg runs even when one breaks — breakage is data"), followed by independent
  `aggregate`/`conformance`/`inventory`-style jobs (`needs: matrix`, `if: always()`) that consume
  uploaded per-leg artifacts and render a `$GITHUB_STEP_SUMMARY` report. `conformance.py` is
  descriptive by default (exit 0) and only gates (`--strict`) on explicit `workflow_dispatch`
  input. [empirical] [scope: full]

## Q — Open questions carried forward

- Q5: Does `SmolLM2-135M-Instruct`'s own chat template satisfy
  `template_result_supports_native_tool_calling`'s predicate (`parse_tool_calls` non-empty) at
  all? Unknown without an empirical CI leg (`smol135.native-template.*` in the design's strategy
  matrix). If not, native-template-wiring is dead on arrival for this model and only the
  emulation approaches remain viable.
- Q6: Does a chained multi-token LQL patch install (K43a) survive past the first 1-2 installed
  `(layer,feature)` slots, or does cross-slot interference make it unreliable in practice? No
  precedent in the codebase either way.
- Q7: Does the balancer's floor/ceiling band, tuned for single-fact recall accuracy, generalize
  to a *class* of tool-requesting prompts (K43b) sharing one slot, or does it just average out to
  unreliable firing on any individual member of the class?
- Q8: Would a contrastive introspection→patch pipeline (K44) be better built as a new `larql-cli`
  subcommand, or as a Python layer over exported `walk`/`circuit-discover`/`ov-rd` JSON artifacts,
  per `ov-rd`'s own README guidance that "feature/code correlation scans" belong outside LARQL
  itself? Affects effort estimate for the `introspect-then-patch` approach materially.

## N — Next actions (bridges to design/plan docs)

- N1: Design doc (`2026-07-16-larql-goose-toolcalling-design.md`) formalizes six candidate
  approaches (K37-K44) as RFC 2119 constraints + EARS acceptance criteria + MADR decision
  records, selecting `emulate-stream-harness` (K38.1-adapted) as Phase-1 implementation per the
  research synthesis's recommendation (lowest risk, zero new parsing theory, zero model/weight
  changes, reuses K40/K41 unchanged).
- N2: Plan doc (`2026-07-16-larql-goose-toolcalling.md`) sequences implementation of the
  Phase-1 leg plus the 12-leg CI strategy matrix (`goose-larql-toolcalling-matrix.yml`,
  `max-parallel: 12` per K45) covering all six approaches.
- N3: `patch-chain-single-token`, `patch-ensemble-trigger`, and `introspect-then-patch` legs
  (K42-K44) are explicitly discovery/measurement legs, not expected-to-pass gates — matching
  K45's `conformance.py` "descriptive by default" convention. Their CI job outcome is data, not
  a merge blocker.
