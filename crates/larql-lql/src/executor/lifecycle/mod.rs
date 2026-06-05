//! Lifecycle executor: USE, STATS, EXTRACT, COMPILE, DIFF, EXPORT PATCH.
//!
//! Each verb lives in its own file; this module is a pure re-export
//! point, so `Session::exec_*` method lookups resolve unchanged.

mod compile;
mod diff;
mod export;
mod extract;
mod stats;
mod use_cmd;
