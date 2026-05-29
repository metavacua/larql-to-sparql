//! TurboQuantEngine — WHT + Lloyd-Max K/V cache compression.
//!
//! Sub-modules provide the low-level codec primitives; `engine` contains
//! the `TurboQuantEngine` implementation and the `TurboQuant` codec struct.

pub mod codebooks;
pub mod engine;
pub mod lloyd_max;
pub mod packing;
pub mod rotation;

pub use engine::{TurboQuant, TurboQuantEngine};
