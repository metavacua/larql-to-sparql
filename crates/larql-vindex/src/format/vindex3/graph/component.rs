//! Components: the executable units of a model system.

use larql_models::config::PositionPolicy;
use serde::{Deserialize, Serialize};

use super::policy::AttentionLayerPolicy;

/// Execution role a component plays in the system. Roles are *derived from
/// evidence* (a declared tap interface, patch geometry), never from a
/// family name — see `build.rs` for the derivation rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentRole {
    /// The primary autoregressive text model.
    PrimaryText,
    /// A perception encoder (vision/audio tower + its adapter).
    Perception,
    /// A speculative drafter consuming another component's hidden states.
    Drafter,
}

/// One executable component: identity, role, and the topology facts the
/// execution planner needs (depth, width, per-layer attention policy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Component {
    /// Stable conceptual id (`target`, `vision`, `draft`) — role-derived,
    /// never a directory or tensor name.
    pub id: String,
    pub role: ComponentRole,
    /// Artifact (inventory) this component came from — physical
    /// traceability, not identity.
    pub source_artifact: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    /// Per-layer attention policy, in layer order. Present for components
    /// whose per-layer resolution is known (the text/drafter path); a
    /// perception tower carries `None` until its resolution exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attention: Option<Vec<AttentionLayerPolicy>>,
    /// The execution surface: everything generic operations read beyond
    /// topology and the per-layer table (V3-G5a). `None` only in graphs
    /// persisted before the surface existed — execution-completeness
    /// gates treat that as incomplete, never as defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<super::surface::ExecutionSurface>,
}

impl Component {
    /// Position policy of one layer, when the attention table carries it.
    pub fn position_policy(&self, layer: usize) -> Option<PositionPolicy> {
        self.attention
            .as_ref()
            .and_then(|table| table.get(layer))
            .map(|policy| policy.position)
    }
}
