//! Unlimited Context Engine — re-exported from `larql_kv::unlimited_context`.
//!
//! The implementation now lives in larql-inference. This module is a thin
//! re-export so existing benchmark code continues to compile unchanged.

pub use larql_kv::unlimited_context::{
    empty_prior, rs_extend_from_checkpoint, CheckpointStore, EngineStats, ExtendOutput,
    TokenArchive, UnlimitedContextEngine,
};

#[doc(hidden)]
pub use larql_kv::unlimited_context::empty_prior as __empty_prior_for_test;
