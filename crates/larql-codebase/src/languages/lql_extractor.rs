use larql_codebase_core::languages::lql_to_edges;
use larql_core::core::graph::Graph;

use super::{ast_edge, LanguageExtractor};

pub struct LqlExtractor;

impl LanguageExtractor for LqlExtractor {
    fn extensions(&self) -> &[&'static str] {
        &["lql"]
    }

    fn extract(&self, source: &str, _path: &str, graph: &mut Graph) {
        for (entity, relation, target, confidence) in lql_to_edges(source) {
            let mut edge = ast_edge(&entity, &relation, &target);
            edge.confidence = confidence;
            graph.add_edge(edge);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::graph::Graph;

    #[test]
    fn lql_insert_emits_edge() {
        // SQL-style INSERT syntax verified against larql-lql-core parser
        let src = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Alice", "KNOWS", "Bob");"#;
        let mut g = Graph::new();
        LqlExtractor.extract(src, "test.lql", &mut g);
        assert!(
            !g.edges().is_empty(),
            "Expected at least one edge from LQL INSERT"
        );
        assert_eq!(g.edges()[0].subject, "Alice");
        assert_eq!(g.edges()[0].relation, "KNOWS");
        assert_eq!(g.edges()[0].object, "Bob");
    }

    #[test]
    fn lql_insert_confidence_propagated() {
        let src = r#"INSERT INTO EDGES (entity, relation, target) VALUES ("Alice", "lives-in", "Colchester") CONFIDENCE 0.8;"#;
        let mut g = Graph::new();
        LqlExtractor.extract(src, "test.lql", &mut g);
        assert!(!g.edges().is_empty(), "Expected at least one edge");
        assert!(
            (g.edges()[0].confidence - 0.8).abs() < 1e-6,
            "Expected confidence 0.8, got {}",
            g.edges()[0].confidence
        );
    }

    #[test]
    fn lql_invalid_source_emits_no_edges() {
        let src = "this is not valid LQL !!!@#$";
        let mut g = Graph::new();
        LqlExtractor.extract(src, "test.lql", &mut g);
        assert!(g.edges().is_empty(), "Invalid LQL should produce no edges");
    }

    #[test]
    fn lql_extensions() {
        assert!(LqlExtractor.extensions().contains(&"lql"));
    }
}
