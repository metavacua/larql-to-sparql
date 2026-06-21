use larql_codebase_core::languages::RUST_QUERIES;

use super::QueryExtractor;

/// Rust source extractor — delegates to the generic `QueryExtractor`
/// with the Tier-0 `RUST_QUERIES` table.
pub fn rust_extractor() -> QueryExtractor {
    QueryExtractor::new(&RUST_QUERIES, || tree_sitter_rust::LANGUAGE.into())
}

/// Preserve the public name for backward compatibility.
pub struct RustExtractor;

impl RustExtractor {
    pub fn new() -> QueryExtractor {
        rust_extractor()
    }
}

#[cfg(test)]
mod tests {
    use larql_core::core::graph::Graph;

    use super::*;
    use crate::languages::LanguageExtractor;

    #[test]
    fn extracts_function_def_edge() {
        let source = r#"fn hello_world() { println!("hi"); }"#;
        let mut g = Graph::new();
        rust_extractor().extract(source, "src/lib.rs", &mut g);
        let entities = g.list_entities();
        assert!(
            entities.iter().any(|n| n.contains("hello_world")),
            "expected hello_world, got: {:?}",
            entities
        );
    }

    #[test]
    fn use_statement_produces_imports_edge() {
        let source = "use std::collections::HashMap;";
        let mut g = Graph::new();
        rust_extractor().extract(source, "src/main.rs", &mut g);
        assert!(
            g.edges().iter().any(|e| e.relation == "imports"),
            "expected an 'imports' edge"
        );
    }
}
