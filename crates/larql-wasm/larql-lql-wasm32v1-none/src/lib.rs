#![cfg_attr(target_arch = "wasm32", no_std)]
#[macro_use]
extern crate alloc;

pub mod ast;
pub mod error;
pub mod executor;
pub(crate) mod lexer;
pub mod parser;
pub mod relations;
pub mod repl;

pub use ast::Statement;
pub use error::LqlError;
#[cfg(not(target_arch = "wasm32"))]
pub use executor::Session;
#[cfg(not(target_arch = "wasm32"))]
pub use parser::parse;
#[cfg(not(target_arch = "wasm32"))]
pub use repl::{run_batch, run_repl, run_statement};
