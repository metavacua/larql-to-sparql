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

/// A non-text input modality.
///
/// Named for the **input**, never for how a checkpoint happens to transform
/// it. Gemma 4 31B encodes an image with a 27-layer transformer tower;
/// Gemma 4 12B (`gemma4_unified_vision`) has no tower at all and projects
/// patches straight into the language embedding space. Both are
/// [`Self::Image`]. Spelling this `VisionTower` would name one
/// implementation and exclude the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Image,
    Audio,
}

/// Geometry of a perception component that owns an internal representation.
///
/// Every field is `Option` for ONE reason: this build could not read the
/// value. It never means the property does not apply — an encoder has a
/// depth whether or not the parser knows its spelling. Qwen3.8 declares its
/// depth as `vision_config.depth` and this build reads `num_hidden_layers`,
/// so `depth` is `None` while the tower plainly exists in
/// `model.visual.blocks.*`. That is a vocabulary gap, and it must not be
/// confused with [`PerceptionTransform::DirectProjection`], for which depth
/// is not a property at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncoderGeometry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_heads: Option<usize>,
}

/// Geometry of a perception component that projects modality-native input
/// straight into the language embedding space.
///
/// Deliberately modest. Gemma 4 12B declares `mm_embed_dim`,
/// `num_soft_tokens`, `model_patch_size` and `pooling_kernel_size`, and it
/// is not yet known which of those generalise to the next encoder-free
/// architecture. The win here is saying *what the object is*, not
/// specifying every future perception design — fields arrive when
/// execution needs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ProjectionGeometry {
    /// Width of the language embedding space this projects into, when the
    /// component declares it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_width: Option<usize>,
}

/// How a modality becomes language-space embeddings.
///
/// Two species, decided from **tensor evidence** rather than declared
/// geometry: a component whose tensors carry an indexed stack owns an
/// internal representation; one whose tensors are a flat projection does
/// not. Config depth cannot decide it — Qwen3.8 (a real tower) and
/// Gemma 4 12B (no tower) both resolve `num_layers: None`, the first
/// because the spelling is unread and the second because there is nothing
/// to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerceptionTransform {
    Encoder(EncoderGeometry),
    DirectProjection(ProjectionGeometry),
}

/// What a perception component perceives, and how.
///
/// Additive on [`Component`]: absent on every container written before this
/// existed, so those still read. Where present it is **authoritative** —
/// the legacy `num_layers`/`hidden_size` fields are never the canonical
/// source for perception semantics, only the input to a one-way legacy
/// reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptionComponent {
    pub modality: Modality,
    pub transform: PerceptionTransform,
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
    /// What this component perceives and how, when its role is
    /// [`ComponentRole::Perception`].
    ///
    /// Absent on a container written before perception had an ontology;
    /// readers reconstruct those from the legacy fields rather than
    /// requiring regeneration. Absent on every non-perception component,
    /// permanently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perception: Option<PerceptionComponent>,
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
