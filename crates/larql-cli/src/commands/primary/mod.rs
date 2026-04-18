//! Primary user-facing verbs: `run`, `pull`, `list`, `show`, `rm`.
//!
//! These wrap the lower-level `extraction::*` commands behind a slimmer
//! flag set and ollama-style ergonomics. Research/power-user tooling lives
//! under `larql dev <subcmd>`.

pub mod cache;
pub mod link_cmd;
pub mod list_cmd;
pub mod pull_cmd;
pub mod rm_cmd;
pub mod run_cmd;
pub mod show_cmd;
