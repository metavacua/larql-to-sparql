// Pure parser layer — lives in larql-lql-core (Tier 0).
// Re-exported here so callers and internal executor code have zero churn.
pub use larql_lql_core::ast;
pub use larql_lql_core::error;
pub use larql_lql_core::lexer;
pub use larql_lql_core::parser;
pub use larql_lql_core::{LqlError, Statement};
pub use larql_lql_core::parser::parse;

// IO-dependent executor layer stays here.
pub mod executor;
pub mod repl;
pub mod relations;
pub use executor::Session;
pub use repl::{run_batch, run_repl, run_statement};
