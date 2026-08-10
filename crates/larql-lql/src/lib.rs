// See crates/larql-core/src/lib.rs for the pattern-2 rationale. repl.rs
// is entirely rustyline-based (interactive terminal line editing, no
// wasm32 story at all) and executor/backend.rs's Remote backend uses
// reqwest -- both likely need pattern-3 whole-module/item exclusion,
// but applying only the confirmed-safe crate-level attribute here and
// letting the next real CI round show the exact boundary rather than
// guessing across executor/'s internal structure.
#![cfg_attr(target_arch = "wasm32", no_std)]
#[cfg(target_arch = "wasm32")]
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
pub use executor::Session;
pub use parser::parse;
pub use repl::{run_batch, run_repl, run_statement};
