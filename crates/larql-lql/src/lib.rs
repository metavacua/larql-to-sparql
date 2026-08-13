// See crates/larql-core/src/lib.rs for the pattern-2 rationale. The real
// wasm32 boundary (confirmed by a full-crate survey): ast/error/lexer/parser
// are a portable AST+tokeniser+parser core with no I/O; executor/ (Session
// has a bare PathBuf field with no core/alloc equivalent, and Backend's
// non-None variants all hold already-native-gated types: larql_vindex's
// PatchedVindex/RouterIndex, larql_inference's Tokenizer, or reqwest),
// repl.rs (rustyline + std::io, plus transitively via Session), and
// relations.rs (RelationClassifier is std::fs-only-constructible, plus
// larql_inference::tokenizers::Tokenizer) are all wholesale native and
// gated below.
#![cfg_attr(target_arch = "wasm32", no_std)]
#[cfg(target_arch = "wasm32")]
#[macro_use]
extern crate alloc;

mod alloc_prelude;
mod collections;

pub mod ast;
pub mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod executor;
pub(crate) mod lexer;
pub mod parser;
#[cfg(not(target_arch = "wasm32"))]
pub mod relations;
#[cfg(not(target_arch = "wasm32"))]
pub mod repl;

pub use ast::Statement;
pub use error::LqlError;
#[cfg(not(target_arch = "wasm32"))]
pub use executor::Session;
pub use parser::parse;
#[cfg(not(target_arch = "wasm32"))]
pub use repl::{run_batch, run_repl, run_statement};
