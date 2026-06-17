//! Predicate-based edge filtering.

use crate::core::edge::Edge;
use crate::core::enums::SourceType;
use crate::core::graph::Graph;

/// Comparison operator for metadata predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataCompare {
    Eq,
    NotEq,
    Gt,
    Gte,
    Lt,
    Lte,
}

impl MetadataCompare {
    fn matches_ordering<T: PartialOrd + PartialEq>(self, actual: T, expected: T) -> bool {
        match self {
            Self::Eq => actual == expected,
            Self::NotEq => actual != expected,
            Self::Gt => actual > expected,
            Self::Gte => actual >= expected,
            Self::Lt => actual < expected,
            Self::Lte => actual <= expected,
        }
    }
}

/// Predicate over an edge metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataPredicate {
    U64 {
        key: String,
        op: MetadataCompare,
        value: u64,
    },
    F64 {
        key: String,
        op: MetadataCompare,
        value: f64,
    },
    String {
        key: String,
        op: MetadataCompare,
        value: String,
    },
    Bool {
        key: String,
        value: bool,
    },
}

impl MetadataPredicate {
    pub fn u64_min(key: impl Into<String>, value: u64) -> Self {
        Self::U64 {
            key: key.into(),
            op: MetadataCompare::Gte,
            value,
        }
    }

    pub fn u64_max(key: impl Into<String>, value: u64) -> Self {
        Self::U64 {
            key: key.into(),
            op: MetadataCompare::Lte,
            value,
        }
    }

    pub fn f64_min(key: impl Into<String>, value: f64) -> Self {
        Self::F64 {
            key: key.into(),
            op: MetadataCompare::Gte,
            value,
        }
    }

    pub fn f64_max(key: impl Into<String>, value: f64) -> Self {
        Self::F64 {
            key: key.into(),
            op: MetadataCompare::Lte,
            value,
        }
    }

    fn matches(&self, edge: &Edge) -> bool {
        let Some(metadata) = edge.metadata.as_ref() else {
            return false;
        };

        match self {
            Self::U64 { key, op, value } => metadata
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|actual| op.matches_ordering(actual, *value)),
            Self::F64 { key, op, value } => metadata
                .get(key)
                .and_then(serde_json::Value::as_f64)
                .is_some_and(|actual| op.matches_ordering(actual, *value)),
            Self::String { key, op, value } => metadata
                .get(key)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|actual| op.matches_ordering(actual, value.as_str())),
            Self::Bool { key, value } => metadata
                .get(key)
                .and_then(serde_json::Value::as_bool)
                .is_some_and(|actual| actual == *value),
        }
    }
}

/// Configuration for filtering graph edges.
///
/// All fields are optional. An edge must pass ALL set predicates.
#[derive(Debug, Default)]
pub struct FilterConfig {
    pub min_confidence: Option<f64>,
    pub max_confidence: Option<f64>,
    pub metadata: Vec<MetadataPredicate>,
    pub relations: Option<Vec<String>>,
    pub exclude_relations: Option<Vec<String>>,
    pub sources: Option<Vec<SourceType>>,
    pub subject_contains: Option<String>,
    pub object_contains: Option<String>,
}

impl FilterConfig {
    pub fn with_metadata(mut self, predicate: MetadataPredicate) -> Self {
        self.metadata.push(predicate);
        self
    }
}

impl FilterConfig {
    /// Check whether an edge passes all configured predicates.
    pub fn matches(&self, edge: &Edge) -> bool {
        if let Some(min) = self.min_confidence {
            if edge.confidence < min {
                return false;
            }
        }
        if let Some(max) = self.max_confidence {
            if edge.confidence > max {
                return false;
            }
        }

        if let Some(ref rels) = self.relations {
            if !rels.iter().any(|r| r == &edge.relation) {
                return false;
            }
        }
        if let Some(ref excl) = self.exclude_relations {
            if excl.iter().any(|r| r == &edge.relation) {
                return false;
            }
        }

        if let Some(ref srcs) = self.sources {
            if !srcs.contains(&edge.source) {
                return false;
            }
        }

        if let Some(ref pat) = self.subject_contains {
            if !edge.subject.contains(pat.as_str()) {
                return false;
            }
        }
        if let Some(ref pat) = self.object_contains {
            if !edge.object.contains(pat.as_str()) {
                return false;
            }
        }

        for predicate in &self.metadata {
            if !predicate.matches(edge) {
                return false;
            }
        }

        true
    }
}

/// Filter a graph, returning a new graph with only matching edges.
/// Preserves schema and graph metadata.
pub fn filter_graph(graph: &Graph, config: &FilterConfig) -> Graph {
    let mut result = Graph::new().with_schema(graph.schema.clone());
    result.metadata = graph.metadata.clone();

    for edge in graph.edges() {
        if config.matches(edge) {
            result.add_edge(edge.clone());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::enums::SourceType;

    fn test_edge(subj: &str, rel: &str, obj: &str, conf: f64) -> Edge {
        Edge::new(subj, rel, obj).with_confidence(conf)
    }

    fn test_edge_with_meta(
        subj: &str,
        rel: &str,
        obj: &str,
        conf: f64,
        layer: usize,
        selectivity: f64,
    ) -> Edge {
        Edge::new(subj, rel, obj)
            .with_confidence(conf)
            .with_source(SourceType::Parametric)
            .with_metadata("layer", serde_json::json!(layer))
            .with_metadata("selectivity", serde_json::json!(selectivity))
            .with_metadata("c_in", serde_json::json!(5.0))
            .with_metadata("c_out", serde_json::json!(8.0))
    }

    fn build_test_graph() -> Graph {
        let mut g = Graph::new();
        g.add_edge(test_edge_with_meta(
            "France",
            "capital-of",
            "Paris",
            0.9,
            26,
            0.8,
        ));
        g.add_edge(test_edge_with_meta(
            "Germany",
            "capital-of",
            "Berlin",
            0.7,
            26,
            0.6,
        ));
        g.add_edge(test_edge_with_meta(
            "France",
            "language-of",
            "French",
            0.5,
            10,
            0.3,
        ));
        g.add_edge(test_edge("Japan", "continent", "Asia", 1.0).with_source(SourceType::Document));
        g
    }

    #[test]
    fn test_no_filters_passes_all() {
        let g = build_test_graph();
        let config = FilterConfig::default();
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 4);
    }

    #[test]
    fn test_min_confidence() {
        let g = build_test_graph();
        let config = FilterConfig {
            min_confidence: Some(0.6),
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        // France/capital(0.9), Germany/capital(0.7), Japan/continent(1.0) pass; France/language(0.5) fails
        assert_eq!(filtered.edge_count(), 3);
    }

    #[test]
    fn test_max_confidence() {
        let g = build_test_graph();
        let config = FilterConfig {
            max_confidence: Some(0.7),
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 2);
    }

    #[test]
    fn test_relation_whitelist() {
        let g = build_test_graph();
        let config = FilterConfig {
            relations: Some(vec!["capital-of".to_string()]),
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 2);
    }

    #[test]
    fn test_relation_blacklist() {
        let g = build_test_graph();
        let config = FilterConfig {
            exclude_relations: Some(vec!["capital-of".to_string()]),
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 2);
    }

    #[test]
    fn test_source_filter() {
        let g = build_test_graph();
        let config = FilterConfig {
            sources: Some(vec![SourceType::Parametric]),
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 3);
    }

    #[test]
    fn test_min_layer() {
        let g = build_test_graph();
        let config = FilterConfig {
            metadata: vec![MetadataPredicate::u64_min("layer", 20)],
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 2);
    }

    #[test]
    fn test_max_layer() {
        let g = build_test_graph();
        let config = FilterConfig {
            metadata: vec![MetadataPredicate::u64_max("layer", 15)],
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 1);
    }

    #[test]
    fn test_min_selectivity() {
        let g = build_test_graph();
        let config = FilterConfig {
            metadata: vec![MetadataPredicate::f64_min("selectivity", 0.5)],
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 2);
    }

    #[test]
    fn test_subject_contains() {
        let g = build_test_graph();
        let config = FilterConfig {
            subject_contains: Some("France".to_string()),
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 2);
    }

    #[test]
    fn test_object_contains() {
        let g = build_test_graph();
        let config = FilterConfig {
            object_contains: Some("Paris".to_string()),
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 1);
    }

    #[test]
    fn test_combined_filters() {
        let g = build_test_graph();
        let config = FilterConfig {
            min_confidence: Some(0.6),
            relations: Some(vec!["capital-of".to_string()]),
            metadata: vec![MetadataPredicate::u64_min("layer", 20)],
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 2);
    }

    #[test]
    fn test_missing_metadata_fails_predicate() {
        let g = build_test_graph();
        let config = FilterConfig {
            metadata: vec![MetadataPredicate::f64_min("unknown_score", 0.1)],
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 0);
    }

    #[test]
    fn test_preserves_metadata() {
        let mut g = Graph::new();
        g.metadata
            .insert("source".to_string(), serde_json::json!("test"));
        g.add_edge(test_edge("A", "rel", "B", 0.8));
        let config = FilterConfig::default();
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.metadata.get("source"), g.metadata.get("source"));
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph::new();
        let config = FilterConfig {
            min_confidence: Some(0.5),
            ..Default::default()
        };
        let filtered = filter_graph(&g, &config);
        assert_eq!(filtered.edge_count(), 0);
    }
}
