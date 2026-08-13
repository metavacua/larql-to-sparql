//! Diagnostic / parity tools — `larql parity` and friends.
//!
//! Cross-backend numerical diff tooling. Used to catch silent regressions
//! between the CPU, Metal, and (eventually) HuggingFace reference paths
//! when refactoring quantisation, activations, norms, or expert routing.
//!
//! What shipped is in `crates/larql-cli/CHANGELOG.md` (2026-05-10); the
//! remaining open scoping work is `ROADMAP.md` → "P2: parity polish".

pub mod moe_locality;
pub mod parity;
