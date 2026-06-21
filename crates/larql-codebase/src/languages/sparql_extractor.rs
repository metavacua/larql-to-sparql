use larql_core::core::graph::Graph;
use spargebra::algebra::GraphPattern;
use spargebra::term::{NamedNodePattern, TermPattern};
use spargebra::{Query, SparqlParser};

use super::{ast_edge, LanguageExtractor};

pub struct SparqlExtractor;

impl LanguageExtractor for SparqlExtractor {
    fn extensions(&self) -> &[&'static str] {
        &["sparql", "rq"]
    }

    fn extract(&self, source: &str, path: &str, graph: &mut Graph) {
        let query: Query = match SparqlParser::new().parse_query(source) {
            Ok(q) => q,
            Err(_) => return,
        };

        let pattern: &GraphPattern = match &query {
            Query::Select { pattern, .. } => pattern,
            Query::Construct { pattern, .. } => pattern,
            Query::Ask { pattern, .. } => pattern,
            Query::Describe { pattern, .. } => pattern,
        };

        extract_pattern(pattern, path, graph);
    }
}

fn extract_pattern(pattern: &GraphPattern, path: &str, graph: &mut Graph) {
    match pattern {
        GraphPattern::Bgp { patterns } => {
            for tp in patterns {
                let s = term_str(&tp.subject);
                let p = named_node_str(&tp.predicate);
                let o = term_str(&tp.object);
                if let (Some(s), Some(p), Some(o)) = (s, p, o) {
                    graph.add_edge(ast_edge(&s, &p, &o));
                }
            }
        }
        GraphPattern::Join { left, right }
        | GraphPattern::LeftJoin { left, right, .. }
        | GraphPattern::Union { left, right }
        | GraphPattern::Minus { left, right } => {
            extract_pattern(left, path, graph);
            extract_pattern(right, path, graph);
        }
        GraphPattern::Filter { inner, .. }
        | GraphPattern::Extend { inner, .. }
        | GraphPattern::OrderBy { inner, .. }
        | GraphPattern::Project { inner, .. }
        | GraphPattern::Distinct { inner }
        | GraphPattern::Reduced { inner }
        | GraphPattern::Slice { inner, .. }
        | GraphPattern::Group { inner, .. } => {
            extract_pattern(inner, path, graph);
        }
        GraphPattern::Graph { inner, .. } => extract_pattern(inner, path, graph),
        GraphPattern::Service { name, inner, .. } => {
            if let NamedNodePattern::NamedNode(nn) = name {
                graph.add_edge(ast_edge(path, "uses_service", nn.as_str()));
            }
            extract_pattern(inner, path, graph);
        }
        _ => {}
    }
}

fn term_str(term: &TermPattern) -> Option<String> {
    match term {
        TermPattern::NamedNode(n) => Some(n.as_str().to_owned()),
        TermPattern::Variable(v) => Some(format!("?{}", v.as_str())),
        TermPattern::BlankNode(b) => Some(format!("_:{}", b.as_str())),
        TermPattern::Literal(_) => None,
    }
}

fn named_node_str(nn: &NamedNodePattern) -> Option<String> {
    match nn {
        NamedNodePattern::NamedNode(n) => Some(n.as_str().to_owned()),
        NamedNodePattern::Variable(v) => Some(format!("?{}", v.as_str())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::graph::Graph;

    const WIKIDATA_LOGICS: &str = r#"
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

CONSTRUCT {
  ?logic rdfs:label ?label .
  ?logic wdt:P31 wd:Q8078 .
}
WHERE {
  ?logic wdt:P31 wd:Q8078 .
  ?logic rdfs:label ?label .
  FILTER(LANG(?label) = "en")
}
LIMIT 10
"#;

    #[test]
    fn sparql_bgp_produces_triple_edges() {
        let mut g = Graph::new();
        SparqlExtractor.extract(WIKIDATA_LOGICS, "wikidata-logics.rq", &mut g);
        assert!(
            !g.edges().is_empty(),
            "Expected at least one edge from SPARQL BGP"
        );
    }

    #[test]
    fn sparql_extensions() {
        assert!(SparqlExtractor.extensions().contains(&"sparql"));
        assert!(SparqlExtractor.extensions().contains(&"rq"));
    }

    #[test]
    fn sparql_select_produces_edges() {
        let sparql = r#"SELECT * WHERE { ?s <http://ex.org/p> ?o }"#;
        let mut g = Graph::new();
        SparqlExtractor.extract(sparql, "test.rq", &mut g);
        assert!(
            g.edges()
                .iter()
                .any(|e| e.relation == "http://ex.org/p"),
            "Expected an edge with predicate http://ex.org/p"
        );
    }
}
