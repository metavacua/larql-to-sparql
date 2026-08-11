//! Accuracy suite result types + `KvEngine`-trait-based drivers.
//!
//! Produces a per-engine summary with both top-1 and Shannon scores,
//! split by [`KnowledgeSource`](super::prompts::KnowledgeSource) so
//! parametric correctness and in-context recall are reported
//! separately:
//!
//! ```text
//!                     Param Top-1   Param bits   InCtx Top-1   InCtx bits   Needle@32K
//! Standard KV         100%          0.42         100%          1.10         100%
//! Markov RS           100%          0.43          88%          3.71          85%
//! TurboQuant 4-bit     99%          0.55         100%          1.45          95%
//! ```
//!
//! Three drivers populate the table:
//! - [`evaluate_parametric`] — short-cue completions from
//!   [`super::prompts`]. Scores `top1_match` + `bits_per_token`.
//! - [`evaluate_in_context`] — needle-in-haystack from
//!   [`super::needle`]. Same metrics, but the answer is planted in the
//!   prompt; the K/V state has to preserve it.
//! - [`evaluate_conflict`] — [`super::conflict`] prompts where the
//!   in-context claim contradicts pretraining. Scores
//!   `followed_context` (correct) vs `parametric_fallback` (compressed
//!   K/V lost the steering).
//!
//! All three accept a `build_engine` closure so each prompt gets a
//! fresh engine; needle tests grow K/V state to ~32K tokens and need
//! the reset between cases.
//!
//! # Module layout
//!
//! This module was a single 2,000-line `runner.rs` until 2026-05-24,
//! when it grew past the point where a flat file was navigable. The
//! sub-modules are organized by responsibility:
//!
//! - [`types`] — the public data structures
//!   ([`ScoreOutcome`], [`EvalLabels`], [`PromptScore`],
//!   [`ConflictScore`], [`StrategySplit`]).
//! - [`scoring`] — per-prompt scoring helpers ([`score_one`]
//!   internals, Shannon bits math). Private to the runner module.
//! - [`drivers`] — public corpus drivers ([`evaluate_parametric`] /
//!   [`evaluate_in_context`] / [`evaluate_conflict`]).
//! - [`aggregate`] — [`compute_strategy_split`] + the
//!   [`format_strategy_split`] table renderer + their helpers.
//! - [`legacy`] — older result types
//!   ([`StrategyAccuracy`], [`AccuracySuiteResult`], [`PromptResult`]
//!   and friends) kept for back-compat with downstream code that
//!   predates the split-axis runner.

pub mod aggregate;
// All 3 drivers (evaluate_parametric/evaluate_in_context/evaluate_conflict)
// take `tokenizer: &larql_inference::tokenizers::Tokenizer` unconditionally;
// `tokenizers` is not(wasm32)-gated upstream — no portable pocket.
#[cfg(not(target_arch = "wasm32"))]
pub mod drivers;
pub mod legacy;
pub mod scoring;
pub mod types;

pub use aggregate::{compute_strategy_split, format_strategy_split};
#[cfg(not(target_arch = "wasm32"))]
pub use drivers::{evaluate_conflict, evaluate_in_context, evaluate_parametric};
pub use legacy::{
    compute_strategy_accuracy, format_accuracy_table, format_category_breakdown,
    AccuracySuiteResult, PromptResult, StrategyAccuracy,
};
pub use types::{ConflictScore, EvalLabels, PromptScore, ScoreOutcome, StrategySplit};
