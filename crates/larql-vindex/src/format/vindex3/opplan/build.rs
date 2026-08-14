//! Construct the generic operation plan for one component — or refuse
//! with the itemised closure defects.
//!
//! Two passes: closure first (classify every stack tensor, check every
//! implied op has its operands with the right geometry, and every operand
//! an implied op), then plan construction, which runs only when closure
//! holds. Nothing here reads a family name, an HF tensor name, or a layer
//! pattern — arguments come from the surface, the policy table and the
//! roles, or the plan is not built.
//!
//! Scope (5b-1): the decoder-stack text program — embedding, layers,
//! final norm, output head. A `FeatureProjector` object belongs to the
//! cross-component edge program (5e) and a perception component to the
//! perception op set (5d); their closure is deferred with their rungs.

use std::collections::BTreeMap;
use std::path::Path;

use larql_models::config::FfnType;

use super::super::encode::segment::{read_segment_header, SegmentTensor};
use super::super::encode::REPRESENTATION_ID_SEP;
use super::super::graph::roles::classify_stack_tensor;
use super::super::graph::surface::ExecutionSurface;
use super::super::graph::{LogicalObject, NormPlacement, ObjectKind, OperandRole};
use super::super::inspect::SystemInspection;
use super::{
    AttentionOp, ClosureDefect, ComponentOpPlan, EmbeddingOp, FfnOp, GateOp, LayerPlan, NormOp,
    OpPlanOutcome, OperandRef, OutputOp, QkNormOp,
};
use crate::error::VindexError;

/// Build the operation plan for `component_id` from a container's
/// inspection plus its segment tables. I/O failures are hard errors;
/// every semantic shortfall is a [`ClosureDefect`].
pub fn plan_component_ops(
    inspection: &SystemInspection,
    root: &Path,
    component_id: &str,
) -> Result<OpPlanOutcome, VindexError> {
    let graph = &inspection.graph;
    let Some(component) = graph.components.iter().find(|c| c.id == component_id) else {
        return Err(VindexError::Parse(format!(
            "no component `{component_id}` in the container's graph"
        )));
    };
    let mut defects: Vec<ClosureDefect> = Vec::new();

    let surface = match &component.execution {
        Some(surface) if surface.norm.placement.is_some() => surface,
        _ => {
            return Ok(OpPlanOutcome {
                plan: None,
                defects: vec![ClosureDefect::MissingSurface {
                    component: component.id.clone(),
                }],
            })
        }
    };
    let placement = surface.norm.placement.expect("checked above");
    let Some(attention_table) = component
        .attention
        .as_ref()
        .filter(|t| t.len() == component.num_layers)
    else {
        return Ok(OpPlanOutcome {
            plan: None,
            defects: vec![ClosureDefect::MissingAttentionTable {
                component: component.id.clone(),
            }],
        });
    };

    let objects: Vec<&LogicalObject> = graph
        .objects
        .iter()
        .filter(|o| o.component == component.id)
        .collect();
    let mut tables: BTreeMap<ObjectKind, (&LogicalObject, Vec<SegmentTensor>)> = BTreeMap::new();
    for object in &objects {
        if matches!(
            object.kind,
            ObjectKind::DecoderStack
                | ObjectKind::Embedding
                | ObjectKind::FinalNorm
                | ObjectKind::OutputHead
        ) {
            tables.insert(
                object.kind,
                (object, object_tensors(inspection, root, object)?),
            );
        }
    }

    // ── Stack closure ──
    let hidden = component.hidden_size;
    let attn = &surface.attention;
    let q_rows = attn.num_q_heads * attn.head_dim;
    let kv_rows = attn.num_kv_heads * attn.head_dim;
    let inter = surface.ffn.intermediate_size;
    let gated_ffn = surface.ffn.ffn_type == FfnType::Gated;

    let mut by_layer: BTreeMap<usize, BTreeMap<OperandRole, SegmentTensor>> = BTreeMap::new();
    if let Some((stack, tensors)) = tables.get(&ObjectKind::DecoderStack) {
        for tensor in tensors {
            match classify_stack_tensor(&tensor.name) {
                None => defects.push(ClosureDefect::UnclassifiedOperand {
                    object: stack.id.clone(),
                    tensor: tensor.name.clone(),
                }),
                Some((layer, role)) => {
                    let slot = by_layer.entry(layer).or_default();
                    if slot.insert(role, tensor.clone()).is_some() {
                        defects.push(ClosureDefect::DuplicateOperand { layer, role });
                    }
                }
            }
        }

        for layer in 0..component.num_layers {
            let present = by_layer.get(&layer);
            for role in required_roles(placement, gated_ffn, attn.output_gate.is_some()) {
                if present.is_none_or(|slot| !slot.contains_key(&role)) {
                    defects.push(ClosureDefect::MissingOperand { layer, role });
                }
            }
            let Some(slot) = present else { continue };
            // QK norms travel as a pair.
            match (
                slot.contains_key(&OperandRole::AttnQNorm),
                slot.contains_key(&OperandRole::AttnKNorm),
            ) {
                (true, false) => defects.push(ClosureDefect::MissingOperand {
                    layer,
                    role: OperandRole::AttnKNorm,
                }),
                (false, true) => defects.push(ClosureDefect::MissingOperand {
                    layer,
                    role: OperandRole::AttnQNorm,
                }),
                _ => {}
            }
            for (role, tensor) in slot {
                // An operand whose op the surface does not carry.
                if let Some(primitive) =
                    absent_op(*role, placement, gated_ffn, attn.output_gate.is_some())
                {
                    defects.push(ClosureDefect::OperandImpliesAbsentOp {
                        object: stack.id.clone(),
                        tensor: tensor.name.clone(),
                        required_primitive: primitive.to_string(),
                    });
                    continue;
                }
                if let Some(expected) = expected_shape(
                    *role,
                    hidden,
                    q_rows,
                    kv_rows,
                    inter,
                    attn.head_dim,
                    attn.qk_norm_scope,
                ) {
                    if tensor.shape != expected {
                        defects.push(ClosureDefect::GeometryMismatch {
                            tensor: format!("{}/{}", stack.id, tensor.name),
                            expected,
                            actual: tensor.shape.clone(),
                        });
                    }
                }
            }
        }
    }

    // ── Single-tensor objects ──
    let single = |kind: ObjectKind,
                  expected: Option<Vec<usize>>,
                  defects: &mut Vec<ClosureDefect>|
     -> Option<(String, SegmentTensor)> {
        let (object, tensors) = tables.get(&kind)?;
        if tensors.len() != 1 {
            defects.push(ClosureDefect::ObjectShape {
                object: object.id.clone(),
                detail: format!("expected exactly 1 tensor, found {}", tensors.len()),
            });
            return None;
        }
        let tensor = tensors[0].clone();
        if let Some(expected) = expected {
            if tensor.shape != expected {
                defects.push(ClosureDefect::GeometryMismatch {
                    tensor: format!("{}/{}", object.id, tensor.name),
                    expected,
                    actual: tensor.shape.clone(),
                });
            }
        }
        Some((object.id.clone(), tensor))
    };

    let vocab = surface.head.as_ref().map(|h| h.vocab_size);
    let embedding_tensor = single(
        ObjectKind::Embedding,
        vocab.map(|v| vec![v, hidden]),
        &mut defects,
    );
    let final_norm_tensor = single(ObjectKind::FinalNorm, Some(vec![hidden]), &mut defects);
    let head_tensor = single(
        ObjectKind::OutputHead,
        vocab.map(|v| vec![v, hidden]),
        &mut defects,
    );
    if (embedding_tensor.is_some() || head_tensor.is_some()) && surface.head.is_none() {
        defects.push(ClosureDefect::MissingSurface {
            component: component.id.clone(),
        });
    }

    if !defects.is_empty() {
        return Ok(OpPlanOutcome {
            plan: None,
            defects,
        });
    }

    // ── Plan construction (closure holds; lookups are now total) ──
    let operand = |object: &str, tensor: &SegmentTensor| OperandRef {
        object: object.to_string(),
        tensor: tensor.name.clone(),
        dtype: tensor.dtype.clone(),
        shape: tensor.shape.clone(),
    };
    let norm_op =
        |surface: &ExecutionSurface, eps: f64, object: &str, tensor: &SegmentTensor| NormOp {
            kind: surface.norm.kind,
            eps,
            weight_offset: surface.norm.weight_offset,
            weight: operand(object, tensor),
        };

    let stack_id = tables
        .get(&ObjectKind::DecoderStack)
        .map(|(o, _)| o.id.clone())
        .unwrap_or_default();
    let mut layers = Vec::with_capacity(component.num_layers);
    for layer in 0..component.num_layers {
        let slot = &by_layer[&layer];
        let get = |role: OperandRole| &slot[&role];
        let policy = &attention_table[layer];
        let qk_norm = slot
            .contains_key(&OperandRole::AttnQNorm)
            .then(|| QkNormOp {
                scope: attn.qk_norm_scope,
                weight_offset: attn.qk_norm_weight_offset,
                q: operand(&stack_id, get(OperandRole::AttnQNorm)),
                k: operand(&stack_id, get(OperandRole::AttnKNorm)),
            });
        // Placement decides which operand feeds the pre-FFN norm: the
        // dedicated one under four-norm, the overloaded
        // `post_attention_layernorm` under two-norm.
        let (post_attention_norm, pre_ffn_role, post_ffn_norm) = match placement {
            NormPlacement::PrePost => (
                Some(norm_op(
                    surface,
                    surface.norm.post_norm_eps,
                    &stack_id,
                    get(OperandRole::PostAttentionNorm),
                )),
                OperandRole::PreFfnNorm,
                Some(norm_op(
                    surface,
                    surface.norm.post_norm_eps,
                    &stack_id,
                    get(OperandRole::PostFfnNorm),
                )),
            ),
            NormPlacement::PreOnly => (None, OperandRole::PostAttentionNorm, None),
        };
        let consumed = slot.len();
        layers.push(LayerPlan {
            layer,
            pre_attention_norm: norm_op(
                surface,
                surface.norm.eps,
                &stack_id,
                get(OperandRole::PreAttentionNorm),
            ),
            attention: AttentionOp {
                num_q_heads: attn.num_q_heads,
                num_kv_heads: attn.num_kv_heads,
                head_dim: attn.head_dim,
                query_scale: attn.query_scale,
                score_scale: attn.score_scale,
                logit_softcapping: attn.logit_softcapping,
                span: policy.span,
                window: policy.window,
                position: policy.position,
                qk_norm,
                parameter_free_qk_norm: attn.parameter_free_qk_norm,
                q: operand(&stack_id, get(OperandRole::AttnQ)),
                k: operand(&stack_id, get(OperandRole::AttnK)),
                v: operand(&stack_id, get(OperandRole::AttnV)),
                o: operand(&stack_id, get(OperandRole::AttnO)),
                output_gate: attn.output_gate.map(|spec| GateOp {
                    spec,
                    projection: operand(&stack_id, get(OperandRole::AttnOutputGate)),
                }),
            },
            post_attention_norm,
            pre_ffn_norm: norm_op(surface, surface.norm.eps, &stack_id, get(pre_ffn_role)),
            ffn: FfnOp {
                intermediate_size: inter,
                activation: surface.ffn.activation,
                gate: gated_ffn.then(|| operand(&stack_id, get(OperandRole::FfnGate))),
                up: operand(&stack_id, get(OperandRole::FfnUp)),
                down: operand(&stack_id, get(OperandRole::FfnDown)),
            },
            post_ffn_norm,
            operands_accounted: consumed,
            operands_present: consumed,
        });
    }

    let plan = ComponentOpPlan {
        component: component.id.clone(),
        embedding: embedding_tensor.map(|(object, tensor)| EmbeddingOp {
            table: operand(&object, &tensor),
            scale: surface.head.as_ref().map(|h| h.embed_scale).unwrap_or(1.0),
            vocab_size: vocab.unwrap_or(0),
        }),
        layers,
        final_norm: final_norm_tensor
            .map(|(object, tensor)| norm_op(surface, surface.norm.eps, &object, &tensor)),
        output: head_tensor.map(|(object, tensor)| OutputOp {
            projection: operand(&object, &tensor),
            multiplier: surface
                .head
                .as_ref()
                .map(|h| h.output_multiplier)
                .unwrap_or(1.0),
            softcapping: surface
                .head
                .as_ref()
                .and_then(|h| h.final_logit_softcapping),
        }),
    };
    Ok(OpPlanOutcome {
        plan: Some(plan),
        defects,
    })
}

/// Tensor table of one object's canonical segment.
fn object_tensors(
    inspection: &SystemInspection,
    root: &Path,
    object: &LogicalObject,
) -> Result<Vec<SegmentTensor>, VindexError> {
    let Some(representation) = object.representations.first() else {
        return Err(VindexError::Parse(format!(
            "object `{}` carries no representation",
            object.id
        )));
    };
    let id = format!(
        "{}{REPRESENTATION_ID_SEP}{}",
        object.id, representation.encoding
    );
    let entry = inspection.index.representations.get(&id).ok_or_else(|| {
        VindexError::Parse(format!("no directory entry for representation `{id}`"))
    })?;
    let (header, _) = read_segment_header(&root.join(&entry.segment))?;
    Ok(header.tensors)
}

/// Roles every layer must supply, given the surface's ops.
fn required_roles(
    placement: NormPlacement,
    gated_ffn: bool,
    output_gate: bool,
) -> Vec<OperandRole> {
    let mut roles = vec![
        OperandRole::PreAttentionNorm,
        OperandRole::PostAttentionNorm,
        OperandRole::AttnQ,
        OperandRole::AttnK,
        OperandRole::AttnV,
        OperandRole::AttnO,
        OperandRole::FfnUp,
        OperandRole::FfnDown,
    ];
    if placement == NormPlacement::PrePost {
        roles.push(OperandRole::PreFfnNorm);
        roles.push(OperandRole::PostFfnNorm);
    }
    if gated_ffn {
        roles.push(OperandRole::FfnGate);
    }
    if output_gate {
        roles.push(OperandRole::AttnOutputGate);
    }
    roles
}

/// The primitive a found operand requires when the surface does not carry
/// its op. `None` when the operand is consumed by a declared op.
fn absent_op(
    role: OperandRole,
    placement: NormPlacement,
    gated_ffn: bool,
    output_gate: bool,
) -> Option<&'static str> {
    match role {
        OperandRole::AttnOutputGate if !output_gate => {
            Some("attention output gate (judged semantics)")
        }
        OperandRole::FfnGate if !gated_ffn => Some("gated FFN"),
        OperandRole::PreFfnNorm | OperandRole::PostFfnNorm
            if placement == NormPlacement::PreOnly =>
        {
            Some("four-norm placement")
        }
        _ => None,
    }
}

/// Expected stored shape per role, from the surface's geometry. `None`
/// for roles whose shape contract is not yet pinned.
fn expected_shape(
    role: OperandRole,
    hidden: usize,
    q_rows: usize,
    kv_rows: usize,
    intermediate: usize,
    head_dim: usize,
    qk_scope: larql_models::config::QkNormScope,
) -> Option<Vec<usize>> {
    use larql_models::config::QkNormScope;
    match role {
        OperandRole::AttnQ => Some(vec![q_rows, hidden]),
        OperandRole::AttnK | OperandRole::AttnV => Some(vec![kv_rows, hidden]),
        OperandRole::AttnO => Some(vec![hidden, q_rows]),
        OperandRole::PreAttentionNorm
        | OperandRole::PostAttentionNorm
        | OperandRole::PreFfnNorm
        | OperandRole::PostFfnNorm => Some(vec![hidden]),
        OperandRole::AttnQNorm | OperandRole::AttnKNorm => match qk_scope {
            QkNormScope::PerHead => Some(vec![head_dim]),
            // Full-projection shape contract unpinned until a real
            // instance is judged.
            QkNormScope::FullProjection => None,
        },
        OperandRole::FfnGate | OperandRole::FfnUp => Some(vec![intermediate, hidden]),
        OperandRole::FfnDown => Some(vec![hidden, intermediate]),
        // Linear(hidden -> q_heads*head_dim), per the judged spec.
        OperandRole::AttnOutputGate => Some(vec![q_rows, hidden]),
    }
}
