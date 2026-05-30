//! Helpers for the `build_vindex` extraction pipeline.
//!
//! Each function is a discrete pipeline stage or utility used by
//! `super::build::build_vindex`:
//!
//! - [`chrono_now`]              — ISO-8601 timestamp without `chrono`.
//! - [`build_whole_word_vocab`]  — reduce the vocab to whole-word tokens
//!                                 + matching embedding rows.
//! - [`compute_gate_top_tokens`] — per-feature top whole-word token (the
//!                                 "what activates this feature" label).
//! - [`compute_offset_direction`]— normalised `embed[output] - embed[input]`
//!                                 direction; the relation vector for
//!                                 clustering.
//! - [`ClusterData`]             — collected cluster inputs.
//! - [`run_clustering_pipeline`] — k-means + label + write
//!                                 `relation_clusters.json` /
//!                                 `feature_clusters.jsonl`.
//!
//! Module layout (round-6 split, 2026-05-10):
//! - `timestamp`  — `chrono_now`
//! - `vocab`      — `build_whole_word_vocab`
//! - `gate_tops`  — `compute_gate_top_tokens`
//! - `offset`     — `compute_offset_direction`
//! - `clustering` — `ClusterData`, `run_clustering_pipeline`
//! - `test_support` (test-only) — shared test fixtures.

mod clustering;
mod gate_tops;
mod offset;
#[cfg(test)]
mod test_support;
mod timestamp;
mod vocab;

#[cfg(not(target_arch = "wasm32"))]
// Re-exports preserving the pre-split path (`super::build_helpers::*`).
pub(crate) use clustering::{run_clustering_pipeline, ClusterData};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use gate_tops::compute_gate_top_tokens;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use offset::compute_offset_direction;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use timestamp::chrono_now;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use vocab::build_whole_word_vocab;
