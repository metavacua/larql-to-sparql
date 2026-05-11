//! Phase 1 — residual compression codecs.
//!
//! Two schemes ship in v0.1:
//!
//! | Module  | Bytes (d=2560) | Ratio | Contract |
//! |---------|----------------|-------|----------|
//! | [`bf16`]  | 5 120         | 1×    | Exact    |
//! | [`int8`]  | 2 564         | 2×    | D-       |
//!
//! All functions are pure: no allocations beyond the returned value.
//! No model or MLX dependency.

pub mod bf16;
pub mod int8;
