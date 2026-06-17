//! Markov Residual Stream strategy — delegates to `larql_kv::markov_residual`.
//!
//! This module is a thin re-export / compat shim so the benchmark runner
//! continues to work while the implementation lives in larql-inference.

pub use larql_kv::accuracy::compare_hidden as compare_hidden_states;
pub use larql_kv::markov_residual::{
    kv_memory_bytes_for_seq, recompute_kv, rs_decode_step, rs_prefill, MarkovResidualEngine,
    RsPrefillResult, RsStore,
};
