pub mod java_lang;
pub mod python_lang;
pub mod query_extractor;
pub mod rust_lang;
pub mod sparql_extractor;
pub mod ts_lang;
pub use java_lang::java_extractor;
pub use python_lang::PythonExtractor;
pub use query_extractor::QueryExtractor;
pub use rust_lang::RustExtractor;
pub use sparql_extractor::SparqlExtractor;
pub use ts_lang::TsExtractor;

use larql_core::core::{edge::Edge, enums::SourceType, graph::Graph};

pub(super) fn ast_edge(s: &str, r: &str, o: &str) -> Edge {
    let mut e = Edge::new(s, r, o);
    e.source = SourceType::Ast;
    e.confidence = 1.0;
    e
}

pub trait LanguageExtractor: Send + Sync {
    fn extensions(&self) -> &[&'static str];
    fn extract(&self, source: &str, path: &str, graph: &mut Graph);
}
