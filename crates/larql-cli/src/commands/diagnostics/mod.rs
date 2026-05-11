//! Diagnostic / parity tools — `larql parity` and friends.
//!
//! Cross-backend numerical diff tooling. Used to catch silent regressions
//! between the CPU, Metal, and (eventually) HuggingFace reference paths
//! when refactoring quantisation, activations, norms, or expert routing.
//!
//! See `crates/larql-cli/ROADMAP.md` P0 → "`larql parity`" for the design.

pub mod parity;
