//! Per-layer attention policy as the graph records it.

use larql_models::config::PositionPolicy;
use serde::{Deserialize, Serialize};

/// Attention span kind of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionSpan {
    /// Attends to the last `window` positions only.
    Sliding,
    /// Attends to the whole prefix.
    Full,
}

/// One layer's attention policy: span, window, and positional encoding.
/// This is architectural liveness information — a KV planner reading it
/// knows that positions beyond `window` on a sliding layer are
/// *architecturally* dead, before any semantic analysis runs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttentionLayerPolicy {
    pub span: AttentionSpan,
    /// Window size when [`AttentionSpan::Sliding`]; `None` on full layers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
    /// How the layer encodes position — including intentional absence.
    pub position: PositionPolicy,
}
