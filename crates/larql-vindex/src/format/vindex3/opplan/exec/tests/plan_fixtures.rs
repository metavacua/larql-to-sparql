//! Synthetic plans built from a real one.
//!
//! Constructing a `LayerPlan` field by field would mean inventing a norm
//! op, an FFN op and an attention op, and any of those could drift from
//! what the builder actually produces. Templating from the dense fixture's
//! own plan keeps every field exactly as the builder writes it, and varies
//! only the two things a topology test needs: how many layers there are and
//! what geometry they attend at.

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::fixtures::dense_f32_model;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, LayerAttention};

/// A wholly-softmax plan of `layers` identical layers at the given KV
/// geometry.
pub(super) fn softmax_plan_with_layers(
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
) -> ComponentOpPlan {
    let dir = tempfile::tempdir().unwrap();
    dense_f32_model(dir.path());
    let inventory = larql_models::inventory::build_inventory(dir.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("dense".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let mut plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();

    let mut template = plan.layers[0].clone();
    if let LayerAttention::Softmax(op) = &mut template.attention {
        op.num_kv_heads = kv_heads;
        op.head_dim = head_dim;
    }
    plan.layers = (0..layers)
        .map(|index| {
            let mut layer = template.clone();
            layer.layer = index;
            layer
        })
        .collect();
    plan
}
