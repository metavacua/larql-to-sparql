use larql_codebase_core::languages::TS_QUERIES;

use super::QueryExtractor;

pub fn ts_extractor() -> QueryExtractor {
    QueryExtractor::new(&TS_QUERIES, || {
        tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()
    })
}

/// Preserve the public name for backward compatibility.
pub struct TsExtractor;

impl TsExtractor {
    pub fn new() -> QueryExtractor {
        ts_extractor()
    }
}

#[cfg(test)]
mod tests {
    use larql_core::core::graph::Graph;

    use super::*;
    use crate::languages::LanguageExtractor;

    #[test]
    fn ts_function_produces_defined_in() {
        let src = "function greet(name: string): string { return name; }";
        let mut g = Graph::new();
        ts_extractor().extract(src, "src/greet.ts", &mut g);
        assert!(
            g.list_entities().iter().any(|n| n.contains("greet")),
            "expected 'greet' in entities"
        );
    }

    #[test]
    fn ts_import_produces_imports_edge() {
        let src = "import { foo } from './bar';";
        let mut g = Graph::new();
        ts_extractor().extract(src, "src/main.ts", &mut g);
        assert!(
            g.edges().iter().any(|e| e.relation == "imports"),
            "expected an 'imports' edge"
        );
    }
}
