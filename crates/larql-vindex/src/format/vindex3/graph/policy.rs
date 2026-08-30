//! Per-layer attention policy as the graph records it.

use larql_models::config::{
    PositionPolicy, LAYER_TYPE_FULL_ATTENTION, LAYER_TYPE_LINEAR_ATTENTION,
    LAYER_TYPE_SLIDING_ATTENTION, LAYER_TYPE_WINDOW_ATTENTION,
};
use serde::{Deserialize, Serialize};

/// Which attention-class operator a layer runs.
///
/// Separate from [`AttentionSpan`] on purpose, and this is the whole point
/// of the type: a span answers *how far back this layer's softmax
/// attends*, and consumers read it as **KV liveness**. A Gated DeltaNet
/// layer has no answer — nothing it retains is indexed by position, so
/// there is no prefix to bound. Spelling `linear_attention` as a span
/// would hand a KV planner a number that looks like liveness and is not,
/// which is exactly the defect this rung exists to remove: before it,
/// every one of Qwen3.8's 48 recurrent layers resolved to
/// [`AttentionSpan::Full`] and the graph reported a 64-layer full-attention
/// tower.
///
/// The op plan's `LayerAttention` is the executable form of this same
/// distinction; this is the graph's, carrying the kind without the
/// operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerOperator {
    /// Scaled dot-product attention over a per-position key/value cache.
    /// The only operator containers written before this field existed
    /// could describe, which is why it is the deserialisation default.
    #[default]
    Softmax,
    /// Gated DeltaNet recurrence: one dense `Dk × Dv` state per value
    /// head, no per-position key or value, no span, no softmax.
    GatedDelta,
}

impl LayerOperator {
    /// Whether this is the default operator — used to keep containers
    /// that predate the field serialising byte-identically.
    fn is_softmax(&self) -> bool {
        matches!(self, Self::Softmax)
    }
}

/// Attention span kind of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionSpan {
    /// Attends to the last `window` positions only.
    Sliding,
    /// Attends to the whole prefix.
    Full,
    /// Attends within a bounded region the component's own geometry
    /// defines — a perception tower's spatial window — rather than a
    /// trailing sequence window. No `window` count applies, because the
    /// extent is not a position count and the config does not declare
    /// one.
    ///
    /// Distinct from [`Self::Sliding`] on purpose. Aliasing the two would
    /// let a KV planner infer that positions beyond a window are dead,
    /// which is true of a sequence window and not of a spatial one;
    /// aliasing to [`Self::Full`] would erase the distinction the
    /// checkpoint actually declares (Muse-Glimmer's vision tower splits
    /// 37/13).
    Windowed,
}

impl AttentionSpan {
    /// The span a declared `layer_types` entry names, or `None` when the
    /// vocabulary does not contain it.
    ///
    /// Fail-closed by construction: an unrecognised spelling answers
    /// `None` so the caller refuses, rather than resolving to a
    /// behavioural default. That is the [§4.7.8] shape — `layer_types`
    /// was once parsed and validated but never consulted, so every model
    /// ran full attention on every layer — and the same shape one level
    /// up is what a "not sliding, therefore full" rule would reintroduce
    /// for any new spelling.
    ///
    /// [§4.7.8]: ../../../../../docs/k3-funnel.md
    pub fn from_declared(entry: &str) -> Option<Self> {
        if entry.eq_ignore_ascii_case(LAYER_TYPE_SLIDING_ATTENTION) {
            Some(Self::Sliding)
        } else if entry.eq_ignore_ascii_case(LAYER_TYPE_FULL_ATTENTION) {
            Some(Self::Full)
        } else if entry.eq_ignore_ascii_case(LAYER_TYPE_WINDOW_ATTENTION) {
            Some(Self::Windowed)
        } else {
            None
        }
    }

    /// The `layer_types` spelling this span corresponds to — the inverse
    /// of [`Self::from_declared`], used to compare what the graph carries
    /// against what the checkpoint declared.
    pub fn declared_name(self) -> &'static str {
        match self {
            Self::Sliding => LAYER_TYPE_SLIDING_ATTENTION,
            Self::Full => LAYER_TYPE_FULL_ATTENTION,
            Self::Windowed => LAYER_TYPE_WINDOW_ATTENTION,
        }
    }
}

/// One layer's attention policy: span, window, and positional encoding.
/// This is architectural liveness information — a KV planner reading it
/// knows that positions beyond `window` on a sliding layer are
/// *architecturally* dead, before any semantic analysis runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionLayerPolicy {
    /// Which attention-class operator this layer runs.
    ///
    /// Defaulted to [`LayerOperator::Softmax`] on deserialisation and
    /// skipped on serialisation when it is, so every container written
    /// before hybrid stacks existed reads and re-writes byte-identically.
    #[serde(default, skip_serializing_if = "LayerOperator::is_softmax")]
    pub operator: LayerOperator,
    /// How far back this layer's softmax attends.
    ///
    /// `None` exactly when no span exists to state — a
    /// [`LayerOperator::GatedDelta`] layer, or a declared spelling outside
    /// this schema's executable vocabulary. Deliberately an absence rather
    /// than a stand-in value: a consumer planning KV must handle "this
    /// layer has no prefix" instead of receiving a fabricated `Full`.
    ///
    /// `Some(x)` serialises exactly as the bare `x` did before this field
    /// became optional, so existing containers are unaffected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<AttentionSpan>,
    /// Window size when [`AttentionSpan::Sliding`]; `None` on full and
    /// windowed layers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window: Option<usize>,
    /// How the layer encodes position — including intentional absence.
    pub position: PositionPolicy,
    /// This layer's head geometry when the family varies it by layer
    /// (Gemma 4: `head_dim` 256 / 8 KV heads on sliding layers,
    /// `global_head_dim` 512 / 2 KV heads on full layers). `None` = the
    /// container predates per-layer geometry and every layer has the
    /// component surface's geometry — an absence with one meaning, not
    /// a default: the graph builder always records `Some` today, so a
    /// `None` on a fresh encode is a bug, not a fallback.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<HeadGeometry>,
    /// The value projection IS the key projection on this layer (Gemma 4
    /// `attention_k_eq_v`, full layers only): no V operand exists and V is
    /// the raw K projection, before the key's norm and rotation. Closure
    /// pairs it both ways — a V operand on such a layer is a stray, a
    /// missing V on any other layer is missing. Defaults for containers
    /// written before it was recorded.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub v_from_k: bool,
    /// The checkpoint's own `layer_types` entry for this layer, verbatim,
    /// when it declares one. `None` when the config states no per-layer
    /// array.
    ///
    /// Carried alongside [`Self::span`], never folded into it: `span` is
    /// this schema's *executable* three-way vocabulary, and a checkpoint
    /// declaring a spelling outside it (a hybrid linear-attention layer)
    /// still needs its raw declaration recorded rather than silently
    /// collapsed to whatever `span` defaulted to. Consumers that need to
    /// know whether `span` is a genuine resolution or a fallback default
    /// compare this field against `span` via [`AttentionSpan::from_declared`]
    /// — see `plan::carriage::probe_layer_types`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_span: Option<String>,
}

impl AttentionLayerPolicy {
    /// The `layer_types` spelling this layer's resolved policy corresponds
    /// to, or `None` when the schema holds no vocabulary for what it
    /// resolved.
    ///
    /// Operator first: a recurrence is `linear_attention` whatever a span
    /// would have said, because it has no span. Only a softmax layer
    /// defers to [`AttentionSpan::declared_name`]. `None` is the
    /// fail-closed answer — a caller comparing against the checkpoint's
    /// own array must refuse rather than invent a spelling.
    pub fn declared_name(&self) -> Option<&'static str> {
        match self.operator {
            LayerOperator::GatedDelta => Some(LAYER_TYPE_LINEAR_ATTENTION),
            LayerOperator::Softmax => self.span.map(AttentionSpan::declared_name),
        }
    }

    /// Whether this layer's carried policy round-trips to the spelling the
    /// checkpoint declared for it.
    ///
    /// A layer that declares nothing is vacuously faithful — there is no
    /// claim to contradict. A layer whose declaration this schema cannot
    /// express answers `false`, never "close enough".
    pub fn matches_declaration(&self) -> bool {
        match self.declared_span.as_deref() {
            None => true,
            Some(raw) => self
                .declared_name()
                .is_some_and(|name| raw.eq_ignore_ascii_case(name)),
        }
    }
}

/// Decide one layer's operator and span from the checkpoint's own
/// `layer_types` entry plus the resolved sliding/full boolean.
///
/// The single place this decision is made. `build.rs` records it into the
/// graph and `plan::compare` grades the checkpoint against it; two
/// implementations of the same rule would be free to drift, and the fact
/// they must agree on is exactly the one this rung is repairing.
///
/// Note what does **not** happen here: the span of a softmax layer is
/// taken from `resolved_sliding` — the boolean the parser derived — and
/// never from `declared`. Sourcing both sides of the comparison from the
/// declared array would make `plan::compare::layer_types_finding`
/// tautological, and a gate that cannot fail is not a gate.
pub fn resolve_layer_kind(
    declared: Option<&str>,
    resolved_sliding: bool,
) -> (LayerOperator, Option<AttentionSpan>) {
    match declared {
        // A declared recurrence. No span exists, and saying so is the
        // repair.
        Some(raw) if raw.eq_ignore_ascii_case(LAYER_TYPE_LINEAR_ATTENTION) => {
            (LayerOperator::GatedDelta, None)
        }
        // A declared softmax spelling this vocabulary knows, or one it
        // does not. Either way the span comes from the resolved boolean,
        // so the comparison downstream stays meaningful; an unrecognised
        // spelling is caught there rather than silently absorbed here.
        Some(_) | None => (
            LayerOperator::Softmax,
            Some(if resolved_sliding {
                AttentionSpan::Sliding
            } else {
                AttentionSpan::Full
            }),
        ),
    }
}

/// One layer's attention head geometry. Query-head count is a component
/// fact (no judged family varies it by layer); the KV side and the head
/// width are what Gemma 4 varies, so those are the per-layer facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadGeometry {
    pub head_dim: usize,
    pub num_kv_heads: usize,
}
