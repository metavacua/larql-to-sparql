# Design Spec — Tool-Calling for the larql-driven Goose Coding Agent

## Metadata (Dublin Core / Schema.org)

- **dc:title**: Tool-Calling Strategy Matrix for the LARQL Goose `LocalInferenceBackend`
- **dc:creator**: RDL loop (Ralph-loop session, agent-authored, human-steered)
- **dc:date**: 2026-07-16
- **dc:source**: `docs/specs/2026-07-16-larql-goose-toolcalling-research-residual.md`
- **schema:softwareVersion**: targets `metavacua/goose` (`feat/larql-local-inference-backend`),
  `larql-to-sparql` main, SmolLM2-135M-Instruct
- **status**: DRAFT — Phase 1 (Brainstorm) output, feeds Phase 2 (`rdl-writing-plans`)

## 1. Problem framing

`LarqlBackend` (the `LocalInferenceBackend` implementation driving `larql chat` as a subprocess,
built earlier in this same RDL loop) can now run a real coding task end-to-end through the
QEMU/KVM VM boundary and produce coherent, on-topic text — the immediately preceding fix
(swapping Goose's full agentic system prompt for a lean tiny-model prompt, per this repo's
`docs/specs/2026-07-15-larql-goose-selfplay-design.md`) confirmed that empirically. What it
cannot do at all is **tool-calling**: `request.tools` is accepted and ignored (residual K36); no
code path parses a model response for a tool-call pattern, and no code path constructs a
`MessageContent::ToolRequest` for Goose's dispatch to consume.

This matters for the project's stated goal ("the model should be demonstrated as a coding agent
using the goose harness") because a coding *agent* — as opposed to a one-shot code-completion
call — is expected to run shell commands, read/write files, and iterate on tool output, not just
emit a single text answer. Goose already has a working, general answer to "how does a model
without native structured tool-calling get to act like an agent" (residuals K37-K41): the
`llamacpp`/`mlx` backends implement text-based emulation, parsing a fixed textual convention
(`$ command` lines, ` ```execute_typescript` fences, or buffered JSON/XML) out of the model's
plain-text response and converting matches into real `ToolRequest` messages that dispatch through
the same `reply_parts.rs` consumer every other backend uses. `LarqlBackend` was simply never
extended to do the same thing.

Separately, the user has posed a genuinely open research question: LARQL's own structural-patch
system (`PatchedVindex`/`.vlp`, residual K42) and introspection tooling (`walk`/`circuit-discover`/
`ov-rd`, residual K44) exist independently of Goose's harness-level tool-calling machinery, and
the model itself could *in principle* be durably edited to emit tool-call syntax more reliably,
rather than relying purely on prompting. This is a materially different kind of approach — riskier,
research-heavy, with no existing eval harness — and the research residual is explicit that
repurposing a single-fact editor for multi-token output-format shaping is unvalidated (K43) and
that a "find the responsible feature, then patch it" pipeline does not exist today (K44).

This design does not pick one approach and discard the rest. It formalizes **six** distinct,
independently-motivated technical approaches (harness-level emulation, two variants; native
chat-template wiring; and three LARQL-native approaches spanning patch-chaining, patch-ensemble
trigger generalization, and introspect-then-patch) and commits to testing all of them empirically,
in parallel, via a CI strategy matrix — following this repo's own established convention
(`lql-strategy-matrix.yml`, residual K45) rather than inventing a new experimental harness shape.
The lowest-risk approach is also implemented as working code in this same phase, so the project
has a real, demonstrable tool-calling capability regardless of how the higher-risk legs turn out.

## 2. RFC 2119 Constraints

- The strategy matrix **MUST** include at least one approach requiring zero changes to model
  weights and zero new parsing theory (harness-level emulation, ported from Goose's own existing,
  working pattern) as the baseline every other approach is compared against. *(residual K38.1,
  synthesis `recommended_first_implementation`)*
- The strategy matrix **MUST** treat the three LARQL-native approaches (patch-chaining,
  patch-ensemble, introspect-then-patch) as discovery/measurement legs, not pass/fail gates — per
  this repo's own `conformance.py` convention (descriptive by default, `--strict` opt-in only,
  residual K45) and because no eval harness for "valid tool-call emission rate" exists yet
  (residual K43/K44). A leg reporting a negative or inconclusive result **MUST NOT** block merge
  of the harness-level implementation.
- The CI strategy matrix **MUST** cap concurrent GitHub-hosted runners at **12**
  (`strategy.max-parallel: 12`), matching both this repo's existing convention (residual K45) and
  the user's explicit instruction this session.
- Any new harness-level parser **MUST** produce exactly the message shape
  `MessageContent::ToolRequest{id, tool_call: Ok(CallToolRequestParams)}` that
  `reply_parts.rs::categorize_tool_requests` already consumes (residual K40) — **MUST NOT**
  introduce a parallel/bespoke dispatch path.
- A new harness-level parser **MUST NOT** depend on the `mlx` Cargo feature (Apple-only,
  residual K38.2) or on any module currently scoped `pub(super)` inside `llamacpp::` (residual
  K38.2) — it **MUST** be new, ungated, crate-visible code, consistent with how `larql.rs`'s
  existing `tiny_model_prompt()` helper was already kept independent of both for the same reason.
- LARQL-native approaches **MUST NOT** mutate a base vindex directly — all edits **MUST** flow
  through `PatchedVindex` overlays / `.vlp` patches, per this repo's existing invariant
  (`AGENTS.md`, carried forward from `docs/specs/2026-07-15-larql-goose-selfplay-design.md`'s own
  constraint of the same shape).
- Every strategy-matrix leg **MUST** run inside the existing QEMU/KVM VM boundary (ADR-2 of the
  2026-07-15 design doc) when it exercises a live `goose run` coding task — **MUST NOT** run
  unwrapped local inference on the CI runner's own host process, consistent with this project's
  standing containment posture.
- The design **SHOULD** hold out the literal test-task answer from any few-shot example used to
  prime tool-call emission (mirroring the `multiply`-not-`add` pattern already applied to
  `larql_tiny_model_system.md` this session) so matrix results measure generalization, not
  memorization.

## 3. EARS Acceptance Criteria

- **AC-1** (ubiquitous): The system SHALL implement a harness-level tool-call emulation parser in
  `goose-local-inference` that is reachable from `larql.rs` without enabling the `mlx` feature.
- **AC-2** (event-driven): WHEN `larql chat`'s response (one line, per residual K39's
  whole-response-per-turn framing) matches the configured tool-call textual convention, THE
  system SHALL emit a `MessageContent::ToolRequest` with a validated tool name and coerced
  arguments, and SHALL NOT emit the matched raw text as plain assistant prose.
- **AC-3** (event-driven): WHEN the strategy-matrix's `plan` job runs, THE system SHALL enumerate
  exactly the 12 legs defined in §5 ADR-6's table as JSON, consumed by `fromJSON` in the `matrix`
  job, matching this repo's `gen_legs.py`-style leg schema (residual K45).
  *(closes the "about a dozen runners per push" resource constraint.)*
- **AC-4** (event-driven): WHEN any of the three LARQL-native matrix legs (patch-chaining,
  patch-ensemble, introspect-then-patch) completes, THE system SHALL record its raw mechanical
  outcome (success/partial/failure plus a free-text observation) without converting that outcome
  into a CI pass/fail gate, consistent with `conformance.py`'s descriptive-by-default behavior.
  *(closes the "discovery, not gate" constraint.)*
- **AC-5** (state-driven): WHILE the `smol135.emulate-stream.shell-conv` leg (the Phase-1
  implementation) runs a live coding task inside the VM, THE system SHALL assert both that a real
  `ToolRequest` was dispatched (not just that text resembling a shell command appeared in the
  transcript) and that the underlying `larql chat` process is bounded by the existing
  `GENERATE_TIMEOUT`/liveness mechanisms already present in `larql.rs`.
- **AC-6** (unwanted behavior): IF a matrix leg's `goose run` invocation hangs past its configured
  timeout, THEN THE system SHALL fail only that leg (`fail-fast: false`, residual K45) and SHALL
  NOT block or cancel sibling legs.
- **AC-7** (optional feature): WHERE a future model/vindex is swapped in for SmolLM2-135M-Instruct,
  THE emulation parser SHOULD continue to function unmodified, since its contract (line-buffered
  text in, `ToolRequest` out) does not depend on any SmolLM2-specific behavior.

## 4. Structurizr DSL — architecture model

```dsl
workspace "larql-goose-toolcalling" "Tool-calling for the larql-driven Goose coding agent" {

  model {
    goose = softwareSystem "Goose CLI" "Coding agent harness (metavacua/goose fork)" {
      larqlBackend = container "LarqlBackend" "LocalInferenceBackend impl; spawns `larql chat`" "Rust"
      toolEmulator = container "larql tool-call emulator (NEW)" "Buffered per-line parser: text -> ToolRequest" "Rust"
      replyParts = container "reply_parts::categorize_tool_requests" "Existing dispatch consumer, backend-agnostic" "Rust"
    }
    larqlChat = softwareSystem "larql chat" "Unmodified LARQL CLI subprocess, one line in / one response out per turn"
    matrix = softwareSystem "goose-larql-toolcalling-matrix.yml" "GitHub Actions strategy matrix, 12 legs, max-parallel 12"
    patchTooling = softwareSystem "LARQL patch/introspection tooling" "PatchedVindex overlay, walk/circuit-discover/ov-rd (existing, in larql-canonical)"

    larqlBackend -> larqlChat "writes prompt line to stdin, reads full response from stdout"
    larqlBackend -> toolEmulator "hands each response line to the parser"
    toolEmulator -> replyParts "emits MessageContent::ToolRequest on match"
    matrix -> larqlBackend "exercises emulate-stream / emulate-buffered / native-template legs"
    matrix -> patchTooling "exercises patch-chain / patch-ensemble / introspect-then-patch legs"
  }

  views {
    systemContext goose {
      include *
      autoLayout
    }
  }
}
```

## 5. MADR Decision Records

### ADR-1: Six candidate approaches are formalized and matrixed, not narrowed to one upfront

- **Status**: Accepted
- **Context**: The user asked to test "a strategy matrix of approaches for utilizing and
  improving these larql features," not to pick a single approach unilaterally. The research
  residual surfaces both harness-level and LARQL-native technical directions with genuinely
  different risk profiles (K37-K44).
- **Decision**: Six approaches are matrixed: `emulate-stream-harness`, `emulate-buffered-harness`,
  `native-template-wiring`, `patch-chain-single-token`, `patch-ensemble-trigger`,
  `introspect-then-patch`. See ADR-6's leg table for the full 12-leg breakdown.
- **Consequences**: More CI surface area than a single-approach plan, bounded by the 12-runner
  cap (ADR-6). Three of the six approaches (the LARQL-native ones) are explicitly
  discovery-only and may report negative/inconclusive results — this is expected and is itself
  the deliverable (residual K43/K44 already establish these mechanisms were never validated for
  this use case).
- **Alternatives considered**: Implement only the lowest-risk approach and stop. Rejected — it
  answers "can we make this specific model call tools" but not the user's actual question, which
  is about LARQL's own tool-calling-relevant capabilities in general.

### ADR-2: `emulate-stream-harness` is implemented as real, working code in this same phase

- **Status**: Accepted
- **Context**: Every approach except `emulate-stream-harness` either depends on an unproven
  premise (`native-template-wiring`: does SmolLM2's own chat template even support native
  tool-calling — Q5) or repurposes machinery never built for this (the three LARQL-native
  approaches — K43/K44). Without one grounded, working implementation, the strategy matrix would
  be all measurement and no demonstrated capability.
- **Decision**: Port `llamacpp/inference_emulated_tools.rs`'s `StreamingEmulatorParser` pattern
  (K38.1) into a new, ungated module in `goose-local-inference`, adapted for `larql chat`'s
  whole-response-per-line framing (K39) rather than per-token streaming — a strictly simpler
  input model. Wire it into `larql.rs`'s `generate()` loop, reusing the already-ungated
  `tool_parsing::compact_tools_json` (K41) for the tool-list prompt text and the existing
  `tx.blocking_send` channel contract unchanged.
- **Consequences**: Delivers a real, demonstrable tool-calling capability for the larql backend
  immediately, independent of how the other five approaches' CI legs turn out. Establishes the
  message-shape contract (`MessageContent::ToolRequest`) the matrix's other harness-level legs
  (`emulate-buffered-harness`) can be compared against on equal footing.
- **Alternatives considered**: De-gating `native_tool_parsing.rs` (K38.2) directly instead of a
  new module. Rejected as the *first* implementation because it requires modifying
  `mlx`-shared code before any leg has empirically shown whether buffered-JSON/XML parsing
  actually outperforms streaming-shell-command parsing against this specific model's output
  style — exactly what the `emulate-buffered-harness` matrix legs are for. It remains a strong
  second-approach candidate, not discarded.

### ADR-3: LARQL-native approaches are legs in the matrix, not out-of-scope

- **Status**: Accepted
- **Context**: The user's framing — "larql has the capability theoretically to allow the
  goose-larql coding agent to add tool calling features to itself" — points specifically at
  LARQL's own patch/introspection machinery, not just at harness-level engineering. Declining to
  test this direction at all would leave the user's actual question unanswered.
- **Decision**: Three LARQL-native approaches are included (`patch-chain-single-token`,
  `patch-ensemble-trigger`, `introspect-then-patch`), each scoped as an explicit
  discovery/measurement leg per ADR-1, run against the research residual's own honest risk
  assessment (K43/K44) rather than a manufactured optimistic one.
- **Consequences**: These legs may return "does not work" as their finding, and that is a valid,
  useful outcome — it converts an open research question into a grounded residual (new K-entries)
  the same way this project's other RDL phases have (e.g. the 2026-07-15 residual's K10 finding
  that `larql-probe` was broken, which directly changed that design's ADR-2).
- **Alternatives considered**: Defer LARQL-native approaches to a later phase pending
  harness-level success. Rejected — the research cost of testing them (a few CI legs) is small
  relative to the value of a grounded answer, and nothing about harness-level success or failure
  changes whether the patch mechanism generalizes to multi-token templates.

### ADR-4: Strategy-matrix shape follows `lql-strategy-matrix.yml` exactly, not a new pattern

- **Status**: Accepted
- **Context**: This repo already has one battle-tested strategy-matrix CI convention
  (`plan`→`build`→`matrix[max-parallel:12,fail-fast:false]`→`aggregate`/`conformance`, residual
  K45). Inventing a second, different shape for this project would fragment conventions for no
  benefit.
- **Decision**: `goose-larql-toolcalling-matrix.yml` reuses the same job DAG shape: a `plan` job
  emits the 12-leg JSON (hand-authored list, not a generator script, since the leg count is fixed
  and small — see ADR-6), a `build` job compiles `larql-cli` and `goose-cli` once, a `matrix` job
  runs each leg with `max-parallel: 12` / `fail-fast: false`, and an `aggregate` job renders a
  `$GITHUB_STEP_SUMMARY` report, matching `aggregate.py`'s descriptive-only stance.
- **Consequences**: Contributors already familiar with `lql-strategy-matrix.yml` can read this
  workflow without learning a new shape. `max-parallel: 12` directly satisfies both the repo
  convention and the user's explicit "about a dozen runners per push" instruction as one
  constraint, not two.
- **Alternatives considered**: A generator script (`gen_legs.py`-style) instead of a hand-authored
  JSON leg list. Rejected for this phase — 12 is a small, fixed, hand-reviewable set; a generator
  is worth its complexity once the leg space grows past what's comfortable to eyeball, which this
  isn't yet.

### ADR-5: Held-out few-shot examples, not the literal test task

- **Status**: Accepted
- **Context**: This session already found (empirically, in the immediately preceding phase) that
  a 135M model given no code-output example drifted into listing test cases; the fix added a
  `multiply`-function few-shot example specifically *not* matching the `add`-function test task,
  to keep the demonstration honest rather than teaching to the test.
- **Decision**: Every matrix leg that primes tool-call emission via a few-shot example (both
  `emulate-*` approaches) MUST use example tool calls distinct from whatever the leg's actual
  coding-task assertion checks for.
- **Consequences**: Matrix results measure the parser's/prompt's ability to generalize a pattern,
  not memorized recall — consistent with the project's existing commitment to honest
  demonstration over benchmark gaming.
- **Alternatives considered**: None seriously — this follows directly from the precedent already
  set and accepted implicitly by continuing the project.

### ADR-6: The 12-leg matrix table

| # | leg_id | approach_id | config |
|---|---|---|---|
| 1 | `smol135.emulate-stream.shell-conv` | emulate-stream-harness | `$ command` convention (verbatim llamacpp pattern), `ForceEmulated` |
| 2 | `smol135.emulate-stream.fenced-tool` | emulate-stream-harness | ` ```tool_call{...}``` ` fenced-JSON convention instead |
| 3 | `smol135.emulate-buffered.openai-json` | emulate-buffered-harness | de-gated `message_from_native_tool_text`, OpenAI `tool_calls` JSON branch, whole-response parse |
| 4 | `smol135.emulate-buffered.xml-function` | emulate-buffered-harness | same de-gated parser, `<function=name>` XML branch |
| 5 | `smol135.native-template.compact-tools` | native-template-wiring | `compact_tools_json`, `ToolCallingMode::Auto`, measures Q5 |
| 6 | `smol135.native-template.full-schema` | native-template-wiring | full JSON-schema tool defs instead of compact form |
| 7 | `smol135.patch-chain.tool-open-tag` | patch-chain-single-token | chain targets only the opening `<tool_call>` tag tokens |
| 8 | `smol135.patch-chain.json-key-tokens` | patch-chain-single-token | chain extends to following `{"name":` tokens |
| 9 | `smol135.patch-ensemble.trigger-5prompt-shared-slot` | patch-ensemble-trigger | 5 phrasings, one shared `(layer,feature)` slot |
| 10 | `smol135.patch-ensemble.trigger-multi-layer-slots` | patch-ensemble-trigger | same 5 phrasings, one distinct slot each |
| 11 | `smol135.introspect-patch.walk-rank-only` | introspect-then-patch | contrastive `walk` ranking only, no patch applied (cheapest) |
| 12 | `smol135.introspect-patch.circuit-ablate-apply` | introspect-then-patch | full pipeline: `circuit-discover` + `ov-rd` ablation + overlay apply |

Legs 1-6 assert against a live `goose run` coding task inside the VM boundary (ADR-2 of the
2026-07-15 design doc). Legs 7-12 are LARQL-CLI-only measurement legs (no VM, no Goose) — they
exercise `larql lql`/`larql walk`/`larql circuit-discover`/`larql dev ov-rd` directly against a
SmolLM2-135M-Instruct vindex, per ADR-3's discovery-only framing, and do not require the VM
boundary since they never run an inference-serving process outside contained `larql-cli`
subcommands.

## 6. Open items carried into Phase 2 (planning)

- Q5-Q8 (research residual) remain open; the plan doc sequences the matrix legs that resolve
  them empirically rather than resolving them by further reading.
- The plan doc must decide concrete pass/observe criteria text for each of legs 7-12's "raw
  mechanical outcome" logging (AC-4), analogous to `lql-strategy-matrix.yml`'s bucket taxonomy
  (`ok`/`timeout`/`crash`/`err`) — this design intentionally leaves the exact bucket vocabulary to
  the plan phase rather than over-specifying it here.
- Whether `emulate-buffered-harness` (legs 3-4) forks `native_tool_parsing.rs` verbatim or
  reimplements its parsing logic independently to avoid any `mlx`-feature entanglement is an
  implementation-level decision for the plan's task breakdown, not a design-level one.
