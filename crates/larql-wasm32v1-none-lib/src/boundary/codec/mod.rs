//! Residual compression codecs.
//!
//! | Module  | Bytes (d=2560) | Ratio | Contract |
//! |---------|----------------|-------|----------|
//! | [`bf16`]  | 5 120         | 1×    | Exact    |
//! | [`int8`]  | 2 564         | 2×    | D-       |

pub mod bf16;
pub mod int8;
