//! Execution completeness: can this graph's components be executed from
//! the graph alone? (V3-G5a gate)
//!
//! The contract derives from the **operations a component's objects
//! imply**, never from what any architecture declares:
//!
//! ```text
//! DecoderStack / PerceptionTower  →  attention + ffn + norm surface
//! Embedding / OutputHead          →  head surface
//! ```
//!
//! This check runs on a [`SystemGraph`] wherever one lives — the builder's
//! output at plan time (where a defect blocks encoding) and a container's
//! persisted graph at inspect time (`--execution-complete`), so a missing
//! `intermediate_size` is a refusal up front, never a mid-forward-pass
//! discovery.

use serde::Serialize;

use super::object::ObjectKind;
use super::SystemGraph;

/// A component that cannot be executed from the graph alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum CompletenessDefect {
    /// Executable objects exist but the component has no surface at all.
    MissingSurface { component: String },
    /// Embedding/head objects exist but the surface has no head group.
    MissingHeadSurface { component: String },
    /// A decoder stack exists but the surface records no norm placement —
    /// count is not semantics, placement is, and a plan cannot be built
    /// without it.
    MissingNormPlacement { component: String },
}

impl std::fmt::Display for CompletenessDefect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingSurface { component } => write!(
                f,
                "component `{component}` has executable objects but no execution surface"
            ),
            Self::MissingHeadSurface { component } => write!(
                f,
                "component `{component}` owns embedding/head objects but its surface has no head group"
            ),
            Self::MissingNormPlacement { component } => write!(
                f,
                "component `{component}` owns a decoder stack but its surface records no norm placement"
            ),
        }
    }
}

/// Object kinds implying the transformer op set (attention + ffn + norm).
fn implies_stack_ops(kind: ObjectKind) -> bool {
    matches!(kind, ObjectKind::DecoderStack | ObjectKind::PerceptionTower)
}

/// Object kinds implying the embedding/head op set.
fn implies_head_ops(kind: ObjectKind) -> bool {
    matches!(kind, ObjectKind::Embedding | ObjectKind::OutputHead)
}

/// Check every component intended for execution against the ops its
/// objects imply. An empty result licenses exactly one claim: no generic
/// operation implied by this graph will find its surface absent.
pub fn execution_completeness(graph: &SystemGraph) -> Vec<CompletenessDefect> {
    let mut defects = Vec::new();
    for component in &graph.components {
        let kinds: Vec<ObjectKind> = graph
            .objects
            .iter()
            .filter(|o| o.component == component.id)
            .map(|o| o.kind)
            .collect();
        let needs_stack = kinds.iter().copied().any(implies_stack_ops);
        let needs_head = kinds.iter().copied().any(implies_head_ops);
        let has_decoder_stack = kinds.contains(&ObjectKind::DecoderStack);
        match &component.execution {
            None if needs_stack || needs_head => defects.push(CompletenessDefect::MissingSurface {
                component: component.id.clone(),
            }),
            Some(surface) => {
                if needs_head && surface.head.is_none() {
                    defects.push(CompletenessDefect::MissingHeadSurface {
                        component: component.id.clone(),
                    });
                }
                if has_decoder_stack && surface.norm.placement.is_none() {
                    defects.push(CompletenessDefect::MissingNormPlacement {
                        component: component.id.clone(),
                    });
                }
            }
            _ => {}
        }
    }
    defects
}
