//! DESCRIBE types — DescribeEdge and LabelSource.
//!
//! These represent the output of a DESCRIBE operation on an entity.
//! The actual DESCRIBE logic lives in the executor (larql-lql), but these
//! types are vindex-level so they can be shared across consumers.

/// Source of a relation label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelSource {
    /// Model inference confirmed this feature encodes this relation.
    Probe,
    /// Cluster-based matching (Wikidata, WordNet, pattern detection).
    Cluster,
    /// Entity pattern detection (country, language, month, number).
    Pattern,
    /// TF-IDF fallback — no confirmed label (no tag shown in output).
    None,
    /// Architecture B: inserted via KNN store.
    KnnStore,
}

impl std::fmt::Display for LabelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Probe => write!(f, "probe"),
            Self::Cluster => write!(f, "cluster"),
            Self::Pattern => write!(f, "pattern"),
            Self::None => write!(f, ""),
            Self::KnnStore => write!(f, "knn"),
        }
    }
}

/// A single edge from a DESCRIBE result.
#[derive(Debug, Clone)]
pub struct DescribeEdge {
    /// Relation label (e.g., "capital", "language"). None for unlabelled edges.
    pub relation: Option<String>,
    /// Where the label came from.
    pub source: LabelSource,
    /// Target token (what the feature outputs).
    pub target: String,
    /// Gate activation score.
    pub gate_score: f32,
    /// Lowest layer this edge appears in.
    pub layer_min: usize,
    /// Highest layer this edge appears in.
    pub layer_max: usize,
    /// Number of features across layers that contribute to this edge.
    pub count: usize,
    /// Additional output tokens from the strongest feature (for context).
    pub also_tokens: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_source_display_all_variants() {
        assert_eq!(LabelSource::Probe.to_string(), "probe");
        assert_eq!(LabelSource::Cluster.to_string(), "cluster");
        assert_eq!(LabelSource::Pattern.to_string(), "pattern");
        assert_eq!(LabelSource::None.to_string(), "");
        assert_eq!(LabelSource::KnnStore.to_string(), "knn");
    }

    #[test]
    fn label_source_equality() {
        assert_eq!(LabelSource::Probe, LabelSource::Probe);
        assert_ne!(LabelSource::Probe, LabelSource::Cluster);
    }

    #[test]
    fn describe_edge_fields_accessible() {
        let edge = DescribeEdge {
            relation: Some("capital".into()),
            source: LabelSource::Cluster,
            target: "Paris".into(),
            gate_score: 0.95,
            layer_min: 14,
            layer_max: 20,
            count: 3,
            also_tokens: vec!["city".into()],
        };
        assert_eq!(edge.relation.as_deref(), Some("capital"));
        assert_eq!(edge.target, "Paris");
        assert_eq!(edge.layer_min, 14);
        assert_eq!(edge.layer_max, 20);
        assert_eq!(edge.count, 3);
        assert_eq!(edge.also_tokens.len(), 1);
    }

    #[test]
    fn describe_edge_none_relation() {
        let edge = DescribeEdge {
            relation: None,
            source: LabelSource::None,
            target: "the".into(),
            gate_score: 0.1,
            layer_min: 0,
            layer_max: 0,
            count: 1,
            also_tokens: vec![],
        };
        assert!(edge.relation.is_none());
        assert_eq!(edge.source, LabelSource::None);
    }
}
