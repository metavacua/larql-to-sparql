//! Cross-component hidden-state interfaces.

use serde::{Deserialize, Serialize};

/// A declared flow of hidden states from one component's layers into
/// another component.
///
/// The edge describes **logical flow** — which residual states cross the
/// component boundary. The tensor that implements the consumer side (a
/// fusion projector) is a separate [`LogicalObject`](super::LogicalObject)
/// referenced by [`consumer_object`](Self::consumer_object): the interface
/// is not the tensor, and the tensor is not the interface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiddenStateEdge {
    /// Component whose hidden states are read.
    pub producer_component: String,
    /// Layers tapped, in declaration order (e.g. `[1, 13, 25, 37, 49]`).
    pub producer_layers: Vec<usize>,
    /// Component consuming the states.
    pub consumer_component: String,
    /// The logical object implementing consumption (a feature projector).
    pub consumer_object: String,
    /// Tokens proposed per consumer forward, when the interface declares a
    /// block protocol (`block_size`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_size: Option<usize>,
}
