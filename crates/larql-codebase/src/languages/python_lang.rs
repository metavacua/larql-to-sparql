use larql_codebase_core::languages::PYTHON_QUERIES;

use super::QueryExtractor;

pub fn python_extractor() -> QueryExtractor {
    QueryExtractor::new(&PYTHON_QUERIES, || tree_sitter_python::LANGUAGE.into())
}

/// Preserve the public name for backward compatibility.
pub struct PythonExtractor;

impl PythonExtractor {
    pub fn new() -> QueryExtractor {
        python_extractor()
    }
}

#[cfg(test)]
mod tests {
    use larql_core::core::graph::Graph;

    use super::*;
    use crate::languages::LanguageExtractor;

    #[test]
    fn python_def_produces_defined_in() {
        let src = "def compute(x):\n    return x * 2\n";
        let mut g = Graph::new();
        python_extractor().extract(src, "utils.py", &mut g);
        assert!(
            g.list_entities().iter().any(|n| n.contains("compute")),
            "expected 'compute' in entities"
        );
    }

    #[test]
    fn python_import_produces_imports_edge() {
        let src = "import os\nfrom pathlib import Path\n";
        let mut g = Graph::new();
        python_extractor().extract(src, "main.py", &mut g);
        assert!(
            g.edges().iter().any(|e| e.relation == "imports"),
            "expected an 'imports' edge"
        );
    }
}
