pub mod edge_template;
pub mod java_queries;
pub mod lql_adapter;
pub mod matrix;
pub mod python_queries;
pub mod rust_queries;
pub mod ts_queries;

pub use edge_template::{EdgeTemplate, Endpoint, LanguageQueries};
pub use java_queries::JAVA_QUERIES;
pub use lql_adapter::lql_to_edges;
pub use matrix::{MatrixEntry, MatrixStatus, WasmTarget, COMPILATION_MATRIX};
pub use python_queries::PYTHON_QUERIES;
pub use rust_queries::RUST_QUERIES;
pub use ts_queries::{ASSEMBLYSCRIPT_QUERIES, TS_QUERIES};
