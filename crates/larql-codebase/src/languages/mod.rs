pub mod python_lang;
pub mod rust_lang;
pub mod ts_lang;
pub use python_lang::PythonExtractor;
pub use rust_lang::RustExtractor;
pub use ts_lang::TsExtractor;

use larql_core::core::graph::Graph;

pub trait LanguageExtractor: Send + Sync {
    fn extensions(&self) -> &[&'static str];
    fn extract(&self, source: &str, path: &str, graph: &mut Graph);
}
