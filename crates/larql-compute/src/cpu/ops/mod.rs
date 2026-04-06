//! CPU operation dispatch — one file per operation type.
//!
//! Mirrors the Metal ops/ structure for consistent API across backends.
//! Each module handles dispatch for one category of compute operation.

pub mod f32_matmul;
pub mod q4_matvec;
pub mod q4_vecmat;
pub mod q4_common;
pub mod q8_matvec;
pub mod vector;
pub mod attention;
pub mod geglu;
