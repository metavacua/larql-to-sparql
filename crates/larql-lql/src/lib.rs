// AST, lexer, and parser live in larql-lql-core (WASM-pure).
// Re-exported here for backward compatibility.
pub use larql_lql_core::ast;
pub use larql_lql_core::error;
pub use larql_lql_core::parser;
pub use larql_lql_core::Statement;
pub use larql_lql_core::LqlError;
pub use larql_lql_core::parse;

pub mod executor;
pub mod relations;
pub mod repl;

pub use executor::Session;
pub use repl::{run_batch, run_repl, run_statement};
