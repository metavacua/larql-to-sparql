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

    #[test]
    fn java_class_produces_has_class() {
        let mut g = Graph::new();
        java_extractor().extract("class Greeter { }", "Greeter.java", &mut g);
        assert!(
            g.edges().iter().any(|e| e.relation == "has_class"),
            "Expected 'has_class' edge for class declaration, got: {:?}",
            g.edges()
        );
    }

    #[test]
    fn java_interface_produces_has_interface() {
        let mut g = Graph::new();
        java_extractor().extract(
            "interface Printable { void print(); }",
            "Printable.java",
            &mut g,
        );
        assert!(
            g.edges().iter().any(|e| e.relation == "has_interface"),
            "Expected 'has_interface' edge for interface declaration, got: {:?}",
            g.edges()
        );
    }

    #[test]
    fn java_method_invocation_produces_calls() {
        let mut g = Graph::new();
        java_extractor().extract(
            "class A { void run() { doSomething(); } }",
            "A.java",
            &mut g,
        );
        assert!(
            g.edges().iter().any(|e| e.relation == "calls"),
            "Expected 'calls' edge for method invocation, got: {:?}",
            g.edges()
        );
    }
}
