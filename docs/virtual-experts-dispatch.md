# Virtual Experts: Turning Local Tool Use into Expert Routing

> Virtual experts are not normal tool calls. They are bounded routing
> decisions into typed, sandboxed compute units.

A writeup of the work to take the WASM-experts subsystem from "scaffolding
present, no production wiring" to "end-to-end tool dispatch through
`larql run --experts`, validated on Mistral 7B Instruct v0.3."

This document captures *what was built*, *why*, and *what we learned along
the way* — particularly the production findings about model capacity and
prompt design that aren't obvious from the code alone.

## TL;DR

`larql run --experts` now performs real end-to-end tool dispatch: a model
emits a structured op-call, the host parses it, resolves it through a
dispatcher, a WASM expert executes it under wasmtime, and the result is
returned to stdout.

The load-bearing lesson is that reliable local tool use isn't a prompting
problem. It depends on five things working together — correct chat-template
wrapping, scoped op vocabularies, visible argument schemas, tolerant
parsing, and (for weak models) constrained decoding. Mistral 7B Instruct
v0.3 Q4K works end-to-end today with focused op subsets; smaller Q4K
models hit the constrained-decode wall. See
[§Production findings: what actually mattered](#production-findings-what-actually-mattered)
before betting on this in a deployment.

```
user prompt
  → ChatTemplate (Gemma/Mistral/Llama/ChatML/Plain auto-detected from vindex)
  → ExpertSession::build_prompt with arg-schema-aware system prompt
  → tokenize + Metal Q4K decode (or CPU Q4K, or CPU F32)
  → parse_op_call (handles Mistral comma drops, fullwidth punctuation,
    code fences, escaped quotes inside string args)
  → ExpertSession::dispatch via Dispatcher trait
  → wasmtime → WASM expert → ExpertResult
```

Mentally: the model is not being asked to use a tool; it is being asked to
route into a small advertised expert table. That turns tool use from an
open-ended generation problem into a bounded selection-and-argument
problem.

End-to-end demonstration:

```
$ echo "What is the GCD of 144 and 60?" \
    | larql run <vindex> --experts --metal --ops gcd,is_prime,factorial,to_roman
{"args":{"a":144,"b":60},"expert_id":"arithmetic","op":"gcd","value":12}
```

13 seconds to load + decode + dispatch on Mistral 7B Instruct v0.3 Q4K via
Metal on M-series Mac. 867 lib tests + 96 CLI tests pass; 2 integration
tests gated behind `--ignored` exercise the full path against a real model.

## What this is, beyond `larql run`

The shape of the work is "tool use as expert routing, not best-effort
prose parsing." We're still parsing — but we're parsing a deliberately
constrained op-call format, not retrofitting structure onto whatever
freeform text the model produced. The model picks an operation from a
bounded vocabulary the host advertises, and the host owns parsing,
dispatch decision, sandbox, and result formatting.

Read this way, WASM experts sit closer to MoE experts than to
traditional chat-style tools — they're callable, typed, sandboxed, and
swappable behind a single dispatch trait. The routing substrate is
analogous, even though the expert implementation is not: neural experts
are weight shards selected by a learned router, while WASM experts are
host-executed programs selected by op name. Both are forms of routing
decisions to specialised callable units. The fact that
`crates/larql-inference/src/ffn/moe_remote.rs` (MoE weight sharding) and
`crates/larql-inference/src/experts/` (WASM compute experts) now coexist
cleanly under disambiguated names — Phase 3 of this work — makes that
parallel structurally explicit.

**Why WASM as the boundary?** Because experts should be portable,
sandboxed modules with an explicit ABI. The model shouldn't need to know
how an op is implemented; it only needs to select the op and supply the
advertised args. WASM is a good fit for that boundary: deterministic
execution, host-controlled memory, low language coupling (today every
expert is Rust, but a Zig, C, TinyGo, or other WASI-compatible language
could plug in behind the same ABI), and trust + validation + resource
control land back on the host where they belong.

The `Dispatcher` trait is therefore the load-bearing abstraction.
Anything that resolves an op-name + args to a result can plug in:

  - `ExpertRegistry` — local WASM experts (today)
  - `FilteredDispatcher` — narrowed allowlist (today)
  - `Box<dyn Dispatcher>` — runtime composition (today)
  - `ConstrainedDispatcher` — vocabulary-masked decode lift (next)
  - `CachedDispatcher` — memoise pure ops
  - `AuditedDispatcher` — log every call for replay
  - `RateLimitedDispatcher` — quota / cost guards
  - `RemoteDispatcher` — RPC to a sandboxed worker pool

That gives the expert layer the same middleware shape HTTP clients have.
It's the right substrate for a local virtual-expert runtime — the rest
of this document is the production work to make the trait load-bearing
instead of theoretical.

## Starting state

Going in:

- **`crates/larql-experts/`** — a nested workspace with 19 WASM cdylibs
  (arithmetic, conway, date, …) targeting `wasm32-wasip1`, sharing the
  `expert-interface` crate. Each cdylib advertised metadata as a flat
  `Vec<String>` of op names.
- **`crates/larql-inference/src/experts/`** — `ExpertRegistry` with
  wasmtime + lazy instantiation + `.cwasm` cache. A handful of dispatch
  test files (`test_expert_dispatch`, `test_constrained_dispatch`,
  `test_llm_dispatch`, `test_trie_dispatch`) duplicating an `extract_json`
  helper four different ways.

Four architectural issues:

1. **Name collision.** `larql_inference::experts::ExpertRegistry` (WASM
   compute experts) and `larql_inference::ffn::RemoteExpertBackend` (MoE
   weight sharding) shared the word "expert" in the same crate. Grepping
   was painful and code review was confused.
2. **Zero production wiring.** Despite a 1.3K-line `test_experts.rs`
   exercising 175 ops, no CLI subcommand instantiated a registry. The only
   way to use the system was to write a Rust test.
3. **External path leak.** `test_trie_dispatch.rs:49` read its probe from
   `../../lazarus-play/experiments/cascade_trie_<slug>_probe.json` — a
   sibling-repo path with no documentation or skip-on-missing UX.
4. **Duplicated parser.** Three test files reimplemented JSON extraction
   from model output. Only the trie-test version handled Mistral's
   missing-comma-before-`"args"` quirk and fullwidth-punctuation
   normalization.

## Phase 1–4: foundational cleanups

### Phase 1 — `parse_op_call`

Extracted to `larql_inference::experts::parser`. Returns
`Option<OpCall { op, args }>` so callers don't all reimplement the
"validate `op` is a string" pattern.

The implementation is brace-depth-aware (skips `{` inside string values,
respects `\"` escapes), normalises fullwidth `，:` to ASCII `,:`, and
patches `…"value"args":` (Mistral) by inserting the missing comma. It
walks multiple top-level `{...}` blocks and returns the first one with a
valid string `op` field — so models that emit a preamble or a code-fence
wrapper still parse cleanly.

17 unit tests cover happy paths, every malformation we've seen in the
wild, and explicit reject paths (no object, no `op`, non-string `op`,
unbalanced braces).

### Phase 2 — Cascade trie probe

The probe artefact is per-model and 1.8–2.9 MB. Vendoring all three
(Gemma, Llama, Mistral) into git would add ~7 MB; gating behind git-lfs
felt heavy for one test. Solution: refactor `CascadeTrie::find` to consult
a precedence chain — `LARQL_PROBE_PATH` → `LARQL_PROBE_DIR` →
caller-supplied search dirs. Add a gitignored `tests/data/` directory with
a README explaining how to populate it and where probes are exported from
the sibling `lazarus-play` repo. The test then skips with regen
instructions when no probe is found.

A pure `find_with_env` variant takes env-var values as parameters so the
precedence chain can be unit-tested without env mutation (which would race
with parallel tests). 5 unit tests cover all four precedence outcomes.

### Phase 3 — MoE rename

`ffn/remote_expert.rs` → `ffn/moe_remote.rs`. Type renames:
`RemoteExpertBackend` → `RemoteMoeBackend`, `RemoteExpertError` →
`RemoteMoeError`, `generate_with_remote_experts` →
`generate_with_remote_moe`, `examples/expert_grid_generate.rs` →
`examples/moe_grid_generate.rs`. Module doc explicitly disambiguates from
`crate::experts`.

Side-effect of running the rename: caught two pre-existing build breakages
where `MoeRouterWeights` had grown new fields (`router_norm_parameter_free`,
`router_input_scalar`) without their callers being updated, and
`rms_norm_no_weight` was referenced but undefined. Both fixed inline so
the workspace builds clean.

### Phase 4 — CLI wiring

Added `larql_inference::prompt::ChatTemplate` (5 variants:
Gemma/Mistral/Llama/ChatML/Plain) with two resolution paths:
`for_model_id(&str)` for HF-style identifiers and `for_family(&str)` when
a `ModelArchitecture` is in scope.

Added `larql_inference::experts::ExpertSession` that owns a registry,
builds the system prompt, wraps with a chat template, and dispatches
parsed op-calls. Returns a structured `Result<DispatchOutcome,
DispatchSkip>` so callers can distinguish "model didn't try"
(`NoOpCall`), "model named a missing op" (`UnknownOp`), and "expert
declined the args" (`ExpertDeclined`).

Wired through `larql-cli/src/commands/primary/run_cmd.rs` as
`larql run --experts`. A `Strategy` enum picks between three decode paths:

| vindex quant | `--metal` | strategy                       | why                                   |
|--------------|-----------|--------------------------------|---------------------------------------|
| Q4_K         | yes       | `layer_graph::generate`        | Metal prefill + KV-cached decode      |
| Q4_K         | no        | `vindex::generate_q4k_cpu`     | per-step `predict_q4k` loop, no KV cache → O(N²) |
| f32          | any       | `forward::generate_cached`     | CPU F32, KV-cached                    |

Plus chat mode (REPL on stdin) when no prompt is given. Loads the model
once, dispatches per turn.

## Test coverage hardening

After Phase 4 shipped, an honest audit surfaced five gaps where critical
code wasn't really covered. All five closed:

1. **`pick_strategy`** was a private impure function (called
   `default_backend()` internally). Refactored into `metal_ready_for_q4`
   (impure) + `pick_strategy(quant, metal_ready)` (pure). 4 tests cover
   the 2×2 quant × metal-ready matrix.

2. **`resolve_experts_dir` precedence** — same approach.
   `resolve_experts_dir_inner(arg_dir, env_dir, exe_path)` takes the
   inputs directly; the public wrapper just plumbs from process state.
   5 tests cover arg-valid, arg-invalid, env-fallthrough, workspace-walk,
   all-fail.

3. **`CascadeTrie::find` env paths** — already factored as
   `find_with_env` in Phase 2. 5 additional tests cover env_path-wins,
   env_path-falls-through, env_dir-wins, env_dir-falls-through, all-empty.

4. **MoE `router_norm_parameter_free=true`** — new codepath added to
   `MoeRouterWeights::route` that calls `rms_norm_no_weight`. Direct test
   covers HF Gemma 4 codepath. Bonus test for `router_input_scalar`
   non-1.0 to prove the scalar actually multiplies through.

5. **`ExpertSession` mock** — the previous tests all required the WASM
   build dir on disk and skipped otherwise, so a fresh checkout had ~0%
   session coverage. The dispatch path is now built around a small
   `Dispatcher` trait so the same `ExpertSession` can be composed with
   filtering, mocking, and (eventually) auditing/caching/rate-limiting
   middleware. Introduced the trait, made `ExpertSession` generic over
   it (with `Default = ExpertRegistry` for backwards compat), and added a
   `MockDispatcher` in tests with canned responses + call recording.
   10 mock-backed tests run unconditionally.

## Integration tests

Two end-to-end tests added, both `#[ignore]`d by default with skip-on-
missing-prerequisites for clean CI behaviour:

- **`test_generate_q4k_cpu`** (`larql-inference`) — loads a real Q4K
  vindex, runs `generate_q4k_cpu` for 4 tokens, asserts non-empty output.
  Validated against Gemma 3 4B Q4K: 4 tokens in 393s on CPU (98s/tok,
  expected for the O(N²) per-step path).

- **`experts_chat_mode_dispatches_via_stdin`** (`larql-cli`) — spawns
  `larql run --experts` with no prompt arg, pipes a prompt over stdin,
  asserts dispatch evidence appears in stdout/stderr. Validated against
  Mistral 7B Instruct v0.3 Q4K: 13s end-to-end including model load.

Both honour `LARQL_TEST_VINDEX=<path>` for explicit override.

## The args-schema epic

The chat-mode test ran end-to-end on the first try — but Mistral 7B
emitted `{"op":"gcd","args":{"144":144,"60":60}}` instead of
`{"a":144,"b":60}`. The pipeline correctly extracted the call,
correctly dispatched, and the expert correctly declined because the keys
didn't match. The system worked; the model didn't know the parameter
names because the system prompt only listed op names, not signatures.

Fix: extend the WASM ABI to advertise per-op argument schemas.

### ABI change

`ExpertMetadata::ops` changed from `Vec<String>` to `Vec<OpSpec>` where
`OpSpec { name: String, args: Vec<String> }`. The `expert_exports!` macro
grew new syntax:

```rust
ops = [
    ("gcd",      ["a", "b"]),
    ("is_prime", ["n"]),
    ("to_roman", ["n"]),
]
```

This is a breaking ABI change. All 19 expert crates were migrated.
~250 individual arg names enumerated by reading each expert's dispatch
function and extracting the `args.get("...")` calls.

Host-side `caller.rs` mirrored the change. `ExpertRegistry::op_specs()`
returns `Vec<&OpSpec>` sorted by name. The `Dispatcher` trait grew an
`op_specs()` method (and `MockDispatcher` was updated accordingly).

### System prompt redesign

The first attempt rendered ops as a multi-line list — `gcd(a, b)\n` per
line, ~3 KB total at 126 ops. Models collapsed into degenerate output
(Gemma 3 4B emitted `kennisk... ` — Dutch for "knowledge", repeated;
Mistral 7B Instruct emitted `1111111...`). The format was too verbose
and gave the model too many simultaneous choices.

The fix was to mirror the format already proven to work in
`test_llm_dispatch.rs`: dense, single-line, no example.

```
Respond with ONLY a JSON object {"op":"...","args":{...}}.
ops: factorial{"n"}, gcd{"a","b"}, is_leap_year{"year"}, is_prime{"n"}, to_roman{"n"}
No extra text.
```

Under 2 KB even with 100+ ops.

### `FilteredDispatcher` + `--ops` flag

Even with the dense format, 126 ops is too many choices for small models.
Real production users will want to scope: a math-chatbot wants
`gcd,lcm,factorial,is_prime,...` not all 126.

`FilteredDispatcher<D>` wraps any `Dispatcher` and exposes only an
allowlist of ops. Calls to non-allowed ops short-circuit to `None` (which
the session surfaces as `UnknownOp`). The CLI exposes this via
`--ops <CSV>`.

To let the CLI pick raw vs. filtered at runtime without duplicating
generation code, added `impl Dispatcher for Box<dyn Dispatcher>`. The CLI
holds `Box<dyn Dispatcher>` and the `ExpertSession` is generic enough to
own it.

## The `detect_template` bug

After all the above, Gemma 3 4B was *still* emitting garbage. Verbose
logging revealed `template: plain` — no chat template wrapping at all.

Root cause: `detect_template` called
`larql_models::detect_architecture(vindex_path)`, which looks for
`config.json`. Vindexes ship `index.json` instead (the `model_dir →
config.json` convention is for raw safetensors directories). So every
vindex was getting `ChatTemplate::Plain`, which is a passthrough.

Fix: read `vindex_path/index.json` directly and consume the `family`
field. Fall back to `model` for the substring heuristic, then to
`detect_architecture` for genuine safetensors dirs, then to `Plain`.

This was the root cause of "model produces gibberish." With the
detection fixed, the prompt fixed, the schema in place, and `--ops`
narrowing the choices, Mistral 7B Instruct v0.3 Q4K dispatched correctly
on the first try.

**Generalisation worth remembering.** This is the kind of bug you get
when a workspace grows two file-layout conventions in parallel —
safetensors dirs (`config.json`) and vindexes (`index.json`) — and the
older shared utility (`detect_architecture`) only knows about one.
The same shape will recur: tokenizer-config detection, lm-head metadata,
quant-format probing. The right long-term fix is a single
`ModelLayout` resolver that knows both conventions; the short-term fix
in this work was just a vindex-aware shortcut in the consumer.

## Production findings: what actually mattered

The single most useful takeaway from this work: reliable local tool use
isn't a prompting problem. It's a small set of things that have to work
together, and missing any one of them collapses the whole pipeline:

  1. **Chat template correctness** — without family-correct wrapping
     every model degrades to garbage. See the `detect_template` debugging
     arc above.
  2. **Op vocabulary scoping** — 126 ops overwhelms small models; 5–15
     ops reliably narrows their decision. The `--ops` flag is a feature,
     not a workaround.
  3. **Argument schema visibility** — without per-op arg keys advertised
     in the prompt, models hallucinate keys. See the args-schema epic
     above.
  4. **Parser tolerance** — production model output is ragged: code
     fences, fullwidth punctuation, missing commas, escaped quotes inside
     string args. `parse_op_call` handles all of these without
     configuration.
  5. **Constrained decode** — the unlock for weak models. Already
     prototyped in `tests/test_constrained_dispatch.rs` and
     `tests/test_trie_dispatch.rs`; not yet wired into `larql run`. See
     "What's still loose."

Orthogonal to those five — sitting under all of them — is **model capacity
+ instruction tuning**: at Q4K, 7B+ instruct works today, base models
don't, smaller instruct models need #5.

What follows are the empirical observations that produced that list.
They aren't visible from the code; they're what we learned by running
real models against the pipeline.

- **Q4K small models cannot do free-form tool use reliably *without
  constrained decode*.** Gemma 3 4B Q4K and Gemma 4 E2B Q4K both emit
  structurally-valid JSON but hallucinate op names (`gcdd`, `to_number`,
  `toRoman`) and fabricate arg keys (`base`, `output`, `maxLen`). This
  behaved like a model-capacity / instruction-following issue rather than
  a dispatch or parser issue — the prompts arrived correctly and the
  parser handled the malformed JSON cleanly. Constrained decoding
  (already prototyped in
  `tests/test_constrained_dispatch.rs` and `tests/test_trie_dispatch.rs`)
  forces character-level masking onto the op vocabulary and is the right
  unlock for weak models; wiring it into the CLI is the highest-leverage
  next step.
- **7B+ instruct models work.** Mistral 7B Instruct v0.3 Q4K dispatches
  correctly with focused op subsets.
- **The `--ops` filter is a feature, not a workaround.** Even strong
  models do better with 5–15 ops than 126. Production deployments should
  always scope.
- **At Q4K, the base-vs-instruct gap is the dominant signal.** The
  local `mistral-7b-v0.1-q4k.vindex` (base, not instruct) was unusable
  for tool dispatch; the `mistral-7b-instruct-v0.3-q4k.vindex` worked
  perfectly. Don't assume a base-model vindex will follow instructions
  even with a good prompt — quantization isn't the issue, instruction
  tuning is. (We don't have a non-Q4K comparison point to claim Q4K
  amplifies the gap; that would need a separate experiment.)
- **Chat templates matter enormously.** Sending the prompt without
  template wrapping degraded all models to garbage output. Detection from
  the vindex's metadata is non-optional.

## API surface added

```rust
// expert-interface (WASM ABI):
pub struct OpSpec { pub name: String, pub args: Vec<String> }

// larql_inference::experts:
pub struct OpCall { pub op: String, pub args: Value }
pub fn parse_op_call(text: &str) -> Option<OpCall>
pub trait Dispatcher {
    fn op_specs(&self) -> Vec<OpSpec>;
    fn call(&mut self, op: &str, args: &Value) -> Option<ExpertResult>;
}
pub struct ExpertSession<D: Dispatcher = ExpertRegistry>
pub struct FilteredDispatcher<D: Dispatcher>
pub enum DispatchSkip { NoOpCall, UnknownOp(String), ExpertDeclined { op, args } }

// larql_inference::prompt:
pub enum ChatTemplate { Gemma, Mistral, Llama, ChatML, Plain }
impl ChatTemplate {
    pub fn for_family(&str) -> Self
    pub fn for_model_id(&str) -> Self
    pub fn wrap(&self, user_prompt: &str) -> String
}

// larql_inference::vindex:
pub fn generate_q4k_cpu(weights, tokenizer, prompt_ids, max_tokens, index)
    -> Vec<(String, u32)>
pub fn is_end_of_turn(token: &str) -> bool

// larql_inference::trie:
impl CascadeTrie {
    pub fn slug(model_id: &str) -> String
    pub fn filename_for(model_id: &str) -> String
    pub fn find(model_id: &str, extra_dirs: I) -> Option<PathBuf>
    pub fn find_with_env(model_id, env_path, env_dir, extra_dirs) -> Option<PathBuf>
}

// larql-cli:
larql run <model> --experts [--experts-dir <DIR>] [--ops <CSV>]
```

## Test inventory

| Suite                                | `cargo test` default | With `-- --ignored` |
|--------------------------------------|----------------------|---------------------|
| `larql-inference` lib                | 867 pass             | 867 pass            |
| `larql-cli` (lib + integration)      | 96 pass, 1 ignored   | 97 pass             |
| `larql-inference --test test_generate_q4k_cpu` | 0 pass, 1 ignored | 1 pass     |
| **Total**                            | **963 pass**         | **965 pass**        |

`cargo test` in default config completes in ~3 seconds. The two
`#[ignore]`d tests load a real 4B/7B model and take 30s–7min depending
on backend; explicitly opt-in via `--ignored`.

## What's still loose

- **Constrained decode is not wired into `larql run --experts`.** The
  `test_constrained_dispatch.rs` and `test_trie_dispatch.rs` files
  demonstrate vocabulary-masked decoding that forces models to emit valid
  op names character-by-character — this would make tool dispatch work
  reliably even on weak models. Plumbing it through the CLI is a future
  task.
- **The cascade trie probe path** is documented + skip-aware but not
  vendored. CI runs skip; local runs with the probe present exercise the
  full pipeline.
- **Args validation in WASM dispatch.** The `expert_interface` exposes
  the schema but doesn't validate at the WASM boundary — bad args still
  go to the dispatch function and fail there. A schema check in
  `larql_call` would surface earlier with better errors.
- **Multi-turn context.** `ExpertSession` is single-shot per call — it
  doesn't accumulate conversation history. Real chat use cases will need
  a small `ConversationState` wrapper that threads prior op calls into
  the prompt.
- **`Runtime::generate` isn't generic over the dispatcher strategy.**
  So the CLI currently reaches for `Box<dyn Dispatcher>` to swap raw vs.
  filtered. That's the trait being used correctly as a middleware seam,
  which is fine for now — but when a third dispatcher (cached, audited,
  rate-limited) appears, the right move is to push the generic down
  into `Runtime::generate` and drop the box.

## Files touched

```
crates/larql-experts/expert-interface/src/lib.rs       # ABI: OpSpec
crates/larql-experts/experts/*/src/lib.rs              # 19 files: ops = [(name, args)]
crates/larql-inference/src/experts/{caller,registry,parser,session,mod}.rs
crates/larql-inference/src/prompt.rs                   # ChatTemplate
crates/larql-inference/src/trie/mod.rs                 # find_with_env
crates/larql-inference/src/vindex/{q4k_forward,mod}.rs # generate_q4k_cpu
crates/larql-inference/src/ffn/{moe_remote,mod}.rs     # rename + new fields
crates/larql-inference/src/layer_graph/grid.rs         # router_norm_parameter_free
crates/larql-inference/src/lib.rs                      # re-exports
crates/larql-inference/tests/{data/,test_generate_q4k_cpu,test_*_dispatch}.rs
crates/larql-inference/examples/moe_grid_generate.rs   # renamed
crates/larql-cli/src/commands/primary/run_cmd.rs       # --experts wiring
crates/larql-cli/src/main.rs                           # ChatArgs ↔ RunArgs
crates/larql-cli/tests/test_run_experts.rs             # CLI integration tests
crates/larql-server/tests/test_expert_endpoint.rs      # rename callers
```

## How to use it

```sh
# Build the WASM modules once.
cd crates/larql-experts
cargo build --target wasm32-wasip1 --release
cd ../..

# Run a focused tool-use session. In practice, always scope ops via --ops —
# even strong models do better with 5–15 options than 126.
larql run ~/.cache/larql/local/mistral-7b-instruct-v0.3-q4k.vindex \
    --experts \
    --metal \
    --ops gcd,is_prime,factorial,to_roman \
    "What is the GCD of 144 and 60?"
```

For chat mode, omit the prompt. For non-Metal CPU decode, omit `--metal`;
it works with any quant, but expect roughly minute-scale responses on 4B
models.
