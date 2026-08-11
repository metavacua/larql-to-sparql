//! Primary user-facing verbs: `run`, `pull`, `list`, `show`, `rm`.
//!
//! These wrap the lower-level `extraction::*` commands behind a slimmer
//! flag set and ollama-style ergonomics. Research/power-user tooling lives
//! under `larql dev <subcmd>`.

// Per-module `forbid(unsafe_code)` (pattern 18) -- see
// commands/dev/ov_rd/mod.rs for the rationale. `shannon_cmd` calls
// `ndarray::s![...]`, CI-confirmed via workflow run 31464601274.
#[forbid(unsafe_code)]
pub mod accuracy_cmd;
#[forbid(unsafe_code)]
pub mod bench;
#[forbid(unsafe_code)]
pub mod cache;
#[forbid(unsafe_code)]
pub mod capabilities_cmd;
#[forbid(unsafe_code)]
pub mod card_cmd;
#[forbid(unsafe_code)]
pub mod dec_bench;
#[forbid(unsafe_code)]
pub mod diag_cmd;
#[forbid(unsafe_code)]
pub mod k3_ledger;
#[forbid(unsafe_code)]
pub mod link_cmd;
#[forbid(unsafe_code)]
pub mod list_cmd;
#[forbid(unsafe_code)]
pub mod model_cmd;
#[forbid(unsafe_code)]
pub mod publish_cmd;
#[forbid(unsafe_code)]
pub mod pull_cmd;
#[forbid(unsafe_code)]
pub mod recipe_cmd;
#[forbid(unsafe_code)]
pub mod rm_cmd;
#[forbid(unsafe_code)]
pub mod run_cmd;
#[forbid(unsafe_code)]
pub mod run_cmd_image;
#[forbid(unsafe_code)]
pub mod run_cmd_speak;
pub mod shannon_cmd;
#[forbid(unsafe_code)]
pub mod shannon_trace;
#[forbid(unsafe_code)]
pub mod show_cmd;
#[forbid(unsafe_code)]
pub mod slice_cmd;
