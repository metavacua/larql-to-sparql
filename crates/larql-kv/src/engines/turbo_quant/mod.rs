//! TurboQuantEngine — WHT + Lloyd-Max K/V cache compression.
//!
//! Sub-modules provide the low-level codec primitives; `engine` contains
//! the `TurboQuantEngine` implementation and the `TurboQuant` codec struct.

pub mod codebooks;
// `index: &VectorIndex` unconditional plus `std::time::Instant::now()`
// in non-test code.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod dispatch;
pub mod engine;
pub mod lloyd_max;
pub mod packing;
pub mod rotation;

// `TurboQuant` (stateless codec struct) is portable; `TurboQuantEngine`
// wraps it in `impl KvEngine` (upstream-gated) and also calls
// `decompress_matrix`, which is native via
// `larql_compute::cpu::spin_pool::par_chunks_mut` (rayon-class,
// not(wasm32)-gated in larql-compute) even though it touches no
// VectorIndex/Tokenizer itself.
pub use engine::TurboQuant;
#[cfg(not(target_arch = "wasm32"))]
pub use engine::TurboQuantEngine;
