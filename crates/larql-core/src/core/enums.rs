use serde::{Deserialize, Serialize};

/// Origin of a knowledge edge.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    Parametric,
    Document,
    Installed,
    Wikidata,
    Manual,
    #[default]
    Unknown,
}

impl SourceType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Parametric => "parametric",
            Self::Document => "document",
            Self::Installed => "installed",
            Self::Wikidata => "wikidata",
            Self::Manual => "manual",
            Self::Unknown => "unknown",
        }
    }
}

/// Strategy for merging duplicate edges during deduplication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Keep the edge with the highest confidence.
    MaxConfidence,
    /// Keep the first occurrence.
    Union,
    /// Caller pre-sorts by source priority.
    SourcePriority,
}
