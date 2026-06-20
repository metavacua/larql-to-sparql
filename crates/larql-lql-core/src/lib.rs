//! LQL lexer, parser, and AST — pure Tier-0 (wasm32v1-none safe).
//!
//! This crate contains no I/O, no networking, no filesystem access, and no
//! dependency on `std`. It compiles for `wasm32v1-none` and any other
//! `no_std` target.

#![no_std]
extern crate alloc;

pub mod ast;
pub mod error;
pub mod lexer;
pub mod parser;

pub use ast::Statement;
pub use error::LqlError;
pub use parser::parse;
