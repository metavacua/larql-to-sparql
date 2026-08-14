# Roadmap — larql-cli

For shipped work, see [CHANGELOG.md](CHANGELOG.md).

## Current state (verified 2026-08-04)

**Command surface.** `main.rs` declares 38 top-level subcommands, one of which
(`dev`) fans out to 25 more. Grouped by purpose:

| Group | Verbs |
|---|---|
| Primary | `run`, `chat`, `pull`, `model`, `link`, `list`, `show`, `slice`, `publish`, `rm`, `serve`, `repl` |
| Bench / measurement | `bench`, `dec-bench`, `k3-ledger`, `accuracy`, `shannon` |
| Build / extraction | `extract`, `extract-index`, `build`, `compile`, `convert`, `hf`, `verify`, `recipe`, `card` |
| Diagnostics | `diag`, `parity`, `moe-locality`, `capabilities` |
| Graph / query | `lql`, `query`, `describe`, `stats`, `validate`, `merge`, `filter` |
| Legacy research | `dev <subcmd>` — 25 commands kept for backwards-compat |

**Caching.** Dual cache (HuggingFace hub + `~/.cache/larql/local/`) with
shorthand resolution (`larql run gemma3-4b-it-vindex`), resolver in
`commands/primary/cache.rs`.

**Multi-modal.** `--image` + `--mm-weights` on `larql run`, prefix-only vision
captioning. Phase 1 (PR #143, 2026-05-24): Gemma 3 + SigLIP,
`TokenBudget::Fixed(256)`. Phase 2 (PR #144, 2026-05-25): Granite Vision +
SigLIP2 + MLP GELU connector + `TokenBudget::PerTile{729}` with AnyRes tiling
(`anyres_tiler.rs`). `prepare_multimodal_input` dispatches on budget type via
trait objects. Image decode/resize in `image_input.rs`, plan assembly in
`run_cmd_image.rs`. Engine capability check (`supports_multimodal()`) fires
before the encoder runs. Q4K vindex dispatch supported. 3-image regression test
in `tests/multimodal_e2e.rs` (`#[ignore]`, NOT FOR CI).

**Cross-engine verification.** `larql shannon verify` (added 2026-05-16) is a
bits/char correctness check that orchestrates the LARQL Rust forward
(in-process) against HF/PyTorch and MLX reference scorers (subprocesses driving
`scripts/shannon_score_{hf,mlx}.py`). Prints a delta table, exits non-zero if
any pair-wise delta exceeds `--threshold` (default 0.5%). Its first serious
application surfaced four config-loading bugs in `larql-models` (rms_norm_eps
not parsed; Gemma 3 per-layer-type rope_scaling missing; llama3 rope_scaling
missing; StarCoder2 norm_epsilon alias). CI gate:
[`.github/workflows/shannon-verify.yml`](../../.github/workflows/shannon-verify.yml)
runs it on every PR. Per-arch sweep:
[`scripts/diagnose_models.py`](../../scripts/diagnose_models.py). See
[`docs/cli.md#cross-engine-verify`](../../docs/cli.md#cross-engine-verify) and
[`docs/diagnoses/shannon-cross-engine-divergence.md`](../../docs/diagnoses/shannon-cross-engine-divergence.md).

**Tests.** 574 `#[test]` sites across `src/` and `tests/`; two integration files
(`multimodal_e2e.rs`, `test_run_experts.rs`).

**Coverage.** Enforced total floor is **7%** — this is a binary crate whose bulk
is clap wiring and I/O against live models and servers. The policy narrows the
90% per-file default to the `bench/` and `dec_bench/` subtrees, excluding the
`*_runtime.rs` I/O wrappers and top-level orchestrators. See
`coverage-policy.json`, whose `policy_note` is the rationale of record.

**Lint.** Clippy clean under both feature sets as of 2026-06-03; `make lint`
runs `cargo clippy --workspace --tests -- -D warnings`.

---

## Open defects

Both raised by the 2026-05-28 whole-codebase review; neither is fixed.

- **P1 — user-facing panic** on multimodal input against a non-multimodal
  model. The lone reachable unwrap in the crate.
- **P2 — NaN `partial_cmp().unwrap()`** at `diagnostics/parity.rs:1126` (the
  review cited `:1119`; the line has since moved, the unwrap has not). Route
  through the shared NaN-safe helper — workspace-wide cleanup; `larql-core` has
  five sites of the same defect. Note `parity.rs:284` already uses
  `unwrap_or(Ordering::Equal)`, so only the logit sort is exposed.

---

## P1: Generation UX

### Sampling flags
**Status**: Not started
**Files**: `src/commands/primary/run_cmd.rs`
Add `--temperature F`, `--top-p F`, `--top-k N`, `--repetition-penalty F` to
the `run` / `chat` subcommands. Values are threaded through to `generate.rs`
logit post-processing (tracked in larql-inference P0).

### `--max-context N`
**Status**: Not started
**Files**: `src/commands/primary/run_cmd.rs`
Expose `--max-context N` (default 8192). Thread through to `KVCache::new_per_layer`
in `generate.rs`. `larql chat` should also respect this for multi-turn state.

### Auto-extract on `larql run hf://`
**Status**: Not started
**Files**: `src/commands/primary/cache.rs` (resolver)
If the shorthand looks like `hf://owner/name` and no cached vindex is found, offer
to run `larql extract` inline (confirm prompt or `--yes`). Collapses the three-step
`extract → link → run` flow to one command. Today only **vindex** `hf://` paths
resolve via the cache; raw HF model paths still need an explicit `extract`.

### OpenAI-compatible surface — CLI side
**Status**: Not started
**Files**: `src/commands/primary/run_cmd.rs`
After the server-side `/v1/chat/completions` endpoint lands (larql-server P0),
expose `larql run --openai-url URL` to send prompts to any OpenAI-compatible
endpoint (including the local `larql serve` instance). Useful for round-trip
testing without a client library.

---

## P2: parity polish

`larql parity` is wired and shipping (see CHANGELOG 2026-05-10). Remaining
open scoping work from the original 2026-04-27 design:

### `--json` output
**Files**: `src/commands/diagnostics/parity.rs`
Human-readable table by default; `--json` emits machine-parseable diff records
for CI consumption (`max_diff`, `index_of_first_divergence`, `checkpoint_name`).

### `--from-recording <path>` replay
**Files**: `src/commands/diagnostics/parity.rs`
Replay a previously captured trace without reloading the model. Useful for
repeated diffs against the same recorded reference run; pairs naturally with
HF sidecar captures once those exist.

### Per-component tolerance defaults
**Files**: `src/commands/diagnostics/parity.rs`
`forward` after 30 layers will accumulate to ~1e-2 even for "correct"
backends; `--tolerance` should default per-component instead of a single
`1e-3`.

### Trace-point infrastructure (larql-inference side)
**Files**: `larql-inference/src/diagnostics/` (new module)
Today `parity` runs each backend end-to-end and compares outputs. The
designed-but-unbuilt extension is named trace points (`post_pre_norm`,
`post_router_softmax`, `post_gate_matmul`, `post_activation`,
`post_down_matmul`, `post_combine`, `post_post_norm`) emitted to a
registered `TraceSink`. Walking the merged traces would let the diagnostic
print the **first divergence** with full surrounding context. Gated on a
`diagnostics` cargo feature in `larql-inference` so release builds pay zero
overhead. Scoped here because the CLI is the primary consumer; the
underlying work belongs to larql-inference.

### `hf` backend for parity
**Files**: `tools/hf_capture.py` + `src/commands/diagnostics/parity.rs`
A Python sidecar that runs `model.forward` with intermediate captures and
writes `.safetensors`; Rust harness loads and compares. The third backend
column (after `reference` and `cpu`/`metal`).

---

## P2: MoE / expert routing

### `--experts` flag (sampling, not WASM)
**Status**: Not started
**Files**: `src/commands/primary/run_cmd.rs`, the `serve` glue
`larql run --experts '0-31=http://host1,32-63=http://host2'` — MoE counterpart
to `--ffn URL`. Maps expert ID ranges to remote URLs; passed through to
`RemoteExpertBackend` in larql-inference. Distinct from the existing
`--experts` flag in `run_cmd.rs` which gates WASM-op dispatch (gcd, base64,
…). Naming overlap to be resolved when this lands. See also
`larql-lql/ROADMAP.md` Phase 3 for the LQL grammar surface.
