//! Primary user-facing verbs: `run`, `pull`, `list`, `show`, `rm`.
//!
//! These wrap the lower-level `extraction::*` commands behind a slimmer
//! flag set and ollama-style ergonomics. Research/power-user tooling lives
//! under `larql dev <subcmd>`.

// Per-module `forbid(unsafe_code)` (pattern 18) -- see
// commands/dev/ov_rd/mod.rs for the rationale. `shannon_cmd` calls
// `ndarray::s![...]`, CI-confirmed via workflow run 31464601274.
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod accuracy_cmd;
#[forbid(unsafe_code)]
pub mod bench;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod cache;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod capabilities_cmd;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod card_cmd;
#[forbid(unsafe_code)]
pub mod dec_bench;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod diag_cmd;
#[forbid(unsafe_code)]
pub mod k3_ledger;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod link_cmd;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod list_cmd;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod model_cmd;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod publish_cmd;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod pull_cmd;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod recipe_cmd;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod rm_cmd;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod run_cmd;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod run_cmd_image;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod run_cmd_speak;
// `shannon_cmd` is now wholly native (clap Args, std::fs/Instant/indicatif,
// tokenizers::Tokenizer-by-value CLI driver code) -- its portable Shannon
// math core was split out to the ungated `shannon_math` module below by
// this round's gating pass, so gating the whole `shannon_cmd` declaration
// no longer strands any portable code (unlike gating it before the split
// would have). No `#[forbid(unsafe_code)]` here, matching its pre-split
// status: `ndarray::s![...]` (score_token_range et al.).
#[cfg(not(target_arch = "wasm32"))]
pub mod shannon_cmd;
// Portable Shannon-scoring math extracted out of `shannon_cmd` (see above).
// No `#[forbid(unsafe_code)]`: `logits_for_row` uses `ndarray::s![...]`,
// same rationale as `shannon_cmd`'s own exemption.
pub mod shannon_math;
#[forbid(unsafe_code)]
pub mod shannon_trace;
#[forbid(unsafe_code)]
#[cfg(not(target_arch = "wasm32"))]
pub mod show_cmd;
#[forbid(unsafe_code)]
pub mod slice_cmd;
