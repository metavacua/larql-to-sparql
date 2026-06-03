# Changelog — larql-cli

All notable changes to `larql-cli` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) conventions,
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

Entries migrated from ROADMAP.md on 2026-05-10; pre-2026-05-10 entries
preserve the date and voice they were originally written in.

## [2026-05-10] — `diag` and `parity` wired to clap; warning sweep

Two existing diagnostic modules became reachable from the CLI:

- **`larql diag <vindex> [--probe] [--probe-tokens N]`** — engine
  diagnostic. Loads a vindex through the production path and prints
  which kernel paths the loader picks (lm_head fast/slow, attn
  fused/per-proj), validates Q4_K/Q6_K manifest strides against the
  canonical 144-byte GGUF layout, and surfaces silent-slowdown
  classes (stale 148-byte stride, `vocab_size=0`) at a glance.
  Implementation in `src/commands/primary/diag_cmd.rs` predated the
  wiring; this dated entry records the clap surface landing.
- **`larql parity <vindex> --component <C>`** — cross-backend
  numerical parity diff (reference / cpu / metal). Components:
  `moe-expert`, `moe-block`, `lm-head`, `layer`. The full
  implementation in `src/commands/diagnostics/parity.rs` predated
  the wiring; this dated entry records the clap surface landing.

Both are grouped under the **Build** help heading next to `verify`.

**Warning cleanup**: 63 → 0 build warnings in `larql-cli`. Removed dead
`Proposal` struct + `pairwise_proposals` fn from
`commands/dev/ov_rd/induce_program/proposal.rs`; pruned three stale import
blocks from `synthesize_program.rs`; underscore-prefixed three unused
variables; module-level `#![allow(dead_code)]` on the four research
diagnostic-capture files (`induce_program/{context,evaluate,localize}.rs`,
`synthesize_program.rs`) with header comments explaining the suppression
is for accumulated debug fields awaiting a viewer; per-item
`#[allow(dead_code)]` on five orphan re-exports / helpers
(`program/mod.rs` re-exports of `smoke`/`strict`/`ProgramSize`/
`MAX_FIXED_POINT_ITERS`, `Program::pq_config`, `ProgramRule::complexity`,
`ProgramCache::num_codes`, `program::context::strata` constants).

## [2026-04-30] — `larql parity --component layer` extended to dense models

Was MoE-only via `LARQL_DUMP_RESIDUALS`; now also handles dense by
setting `LARQL_METAL_DUMP_LAYERS` and reading per-layer
`metal_layer_NN_h_out.f32` / `metal_layer_NN_h_post_attn.f32`. Used to
confirm Gemma 4 31B dense matches between CPU and Metal at every layer
(cos ≥ 0.9999), which localised the bug to chat-template / sampling
rather than the math.

## [2026-04-30] — `larql parity --component lm-head` works on dense vindexes

The MoE-only gate (`is_hybrid_moe()` check) only fires for `moe-expert` /
`moe-block` now; `lm-head` is backend-agnostic (Q4_K matvec vs f32
reference) and works on any vindex with an lm_head.

## [2026-04-30] — Dense Metal path applies chat templates

`walk_cmd::run_predict_q4k` was sending the raw user prompt to
`encode_prompt`; chat-template wrapping only happened for the
`--moe-shards` / `--moe-units-manifest` paths. Both paths now go
through `larql_inference::chat::render_user_prompt`. Fixes "The answer
is:" looping on Gemma 4 31B dense and the "more questions instead of
answers" frame on Gemma 3.

## [2026-04-30] — Auto-injected default system prompt for Gemma 4

Gemma 4 needs a system prompt to enter answer mode (all variants).
`LARQL_NO_DEFAULT_SYSTEM=1` opts out, `LARQL_SYSTEM=<text>` overrides.
