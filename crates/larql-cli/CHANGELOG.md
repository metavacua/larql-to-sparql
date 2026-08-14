# Changelog — larql-cli

All notable changes to `larql-cli` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/) conventions,
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

Entries migrated from ROADMAP.md on 2026-05-10; pre-2026-05-10 entries
preserve the date and voice they were originally written in.

## [2026-08-01] — `k3-ledger freq-mass`: grade a resident expert set on events, not support

**`larql k3-ledger freq-mass --pool <capture>`** — frequency-mass coverage from
a `dec-bench capture --routing` pool. Answers what fraction of routing *events*
a resident set of C symbols per layer serves, which is the question a cache
actually poses; support (which symbols get touched at all) is a different
quantity and the two had been conflated. No checkpoint and no network: it
short-circuits before the geometry fetch so it keeps working on non-K3 traces.

Four estimators, all scored on the same held-out window so only the
residency-ranking key differs — static (`λ=1`, pooled prior, leave-one-out),
causal (`λ=0`), shrinkage (interior `λ`), oracle, and a marginal-preserving
null. `--per-symbol` additionally emits the measured per-symbol mass vector
(raw counts *and* probabilities) that DEC-8.4's precision allocator consumes.

New modules, both pure and both **symbol-agnostic** — `SelectionTrace` never
learns whether its symbols are MoE experts or FFN feature rows, so the same
conventions serve DEC-8.4 and DEC-8.1:

- `selection_trace.rs` — generic `[session][step][stratum][width]` trace, a
  capture-pool adapter, support, and the uniform-support null. 100% covered.
- `freqmass.rs` — the five arms, `miss_ratios`, `mass_per_residency`. 98% covered.
- `symbol_mass.rs` — per-symbol census; cold-tail counts always emitted, rows
  opt-in. 99% covered.

Findings recorded in [`docs/dec-funnel.md`](../../docs/dec-funnel.md), all
**8-of-128, 6.25%-activation** figures on a 64-prompt domain mixture (R2/R10):
a 6.25% resident set covers 42% of routing events (not 6.25% — uniform routing
understates coverage ~6.8× and overstates support 3.4×, now standing rule R9);
session-adaptive population buys **1.03×** on the miss stream at that operating
point against a 1.41× oracle bound; 16.1% of layer-expert pairs went unobserved,
with per-layer spread 3–68 of 128.

`k3-ledger touch` now annotates its uniform hit count as a null rather than a
forecast, since DEC-8.6's "~1 of 16 hit/layer" inherited the understatement.

The whole `k3-ledger` verb is documented in [`docs/cli.md`](../../docs/cli.md)
for the first time.

## [2026-06-03] — Clippy clean under both feature sets (gpu-on and gpu-off)

Closes the hygiene half of the 2026-05-28 hardening entry below. The 2
default-build nits (unused `ProjectorWeights` import, dead `total_tiles` field)
are fixed, plus the 41 `--no-default-features` (gpu-off) warnings:
`diagnostics/parity.rs` gets a gpu-off `#![cfg_attr(.., allow(dead_code))]`, and
the `walk_cmd`/`shannon_cmd` `--metal`-requires-gpu stubs route through a
cfg-split `metal_backend_box()?` helper instead of a diverging `let` (which had
poisoned downstream code as unreachable). `make lint`
(`cargo clippy --workspace --tests -- -D warnings`) is green.

Coverage: per the crate `coverage-policy.json` the enforced total floor is
**7%** (binary crate, mostly command wiring; most files excluded) — currently
~12–14%, passing. The per-file 90% default applies only to the non-excluded
modules (e.g. `bench/ollama.rs` at 91%).

## [2026-05-28] — Hardening findings from the whole-codebase review

From the whole-codebase review ([`docs/audits/codebase-review-2026-05-28.md`](../../docs/audits/codebase-review-2026-05-28.md)):

- **P1 — user-facing panic** on multimodal input against a non-multimodal model (lone reachable unwrap in the crate).
- **P2 — NaN `partial_cmp().unwrap()`** at `parity.rs:1119` → shared NaN-safe helper.
- **Hygiene** — 43 clippy warnings across the two feature sets. Fixed 2026-06-03, see the entry above.

The two panic findings were recorded, not fixed. They remain open — see
[`ROADMAP.md`](ROADMAP.md) §"Open defects".

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
