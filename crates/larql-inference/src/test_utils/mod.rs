//! Synthetic test fixtures for engine and layer-graph unit tests.
//!
//! The generic `make_test_weights()` ModelWeights builder lives in
//! [`larql_models::test_fixtures`] (gated behind the `test-utils`
//! feature) so downstream test crates — `larql-compute`, this crate,
//! and others — can construct realistic fixtures without disk I/O.
//! Inference re-exports it here so existing
//! `crate::test_utils::make_test_weights` callers don't change.
//!
//! Inference-specific helpers stay here:
//! - `make_test_vindex(weights)` — in-memory VectorIndex with random gate vectors
//! - `make_test_tokenizer(vocab_size)` — WordLevel tokenizer mapping token N to "[N]"
//! - `make_test_q4k_*`, `make_gemma3_*`, `make_starcoder2_*` etc. — arch-specific
//!   fixtures that pull in vindex/tokenizer machinery.
//!
//! Dimensions for `make_test_weights`: vocab=32, hidden=16,
//! intermediate=32, 2 q-heads, 1 kv-head, head_dim=8, 2 layers.
//! Forward pass ≈ 10 ms on CPU.

pub use larql_models::test_fixtures::make_test_weights;

mod fixtures;
mod model_dir;
mod q4k;
mod vindex;

#[allow(unused_imports)]
pub use fixtures::*;
#[allow(unused_imports)]
pub use model_dir::*;
#[allow(unused_imports)]
pub use q4k::*;
#[allow(unused_imports)]
pub use vindex::*;

#[cfg(test)]
mod synthetic_model_dir_tests;

#[cfg(test)]
mod mock_gpu_backend_tests;
