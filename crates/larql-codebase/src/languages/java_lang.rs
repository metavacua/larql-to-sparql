use larql_codebase_core::languages::JAVA_QUERIES;

use super::QueryExtractor;

/// Java source extractor — delegates to the generic `QueryExtractor`
/// with the Tier-0 `JAVA_QUERIES` table.
pub fn java_extractor() -> QueryExtractor {
    QueryExtractor::new(&JAVA_QUERIES, || tree_sitter_java::LANGUAGE.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::languages::LanguageExtractor;
    use larql_core::core::graph::Graph;

    #[test]
    fn java_method_produces_defined_in() {
        let src = r#"
public class Greeter {
    public String greet(String name) {
        return "Hello " + name;
    }
}
"#;
        let mut g = Graph::new();
        java_extractor().extract(src, "Greeter.java", &mut g);
        assert!(
            g.list_entities().iter().any(|n| n.contains("greet")),
            "expected 'greet' in entities, got: {:?}",
            g.list_entities()
        );
    }

    #[test]
    fn java_import_produces_imports_edge() {
        let src = "import java.util.List;\nclass Foo {}";
        let mut g = Graph::new();
        java_extractor().extract(src, "Foo.java", &mut g);
        assert!(
            g.edges().iter().any(|e| e.relation == "imports"),
            "expected an 'imports' edge"
        );
    }
}
