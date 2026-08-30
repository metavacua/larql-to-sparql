//! `larql vindex3 ops` — print the generic operation plan (G5b-1).

use larql_vindex::format::vindex3::opplan::{
    plan_component_ops, AttentionOp, GatedDeltaOp, LayerAttention, LayerPlan, NormOp,
};

use super::optional_op::scalar;
use super::OpsArgs;

pub(super) fn run_ops(args: OpsArgs) -> Result<(), Box<dyn std::error::Error>> {
    let inspection =
        larql_vindex::format::vindex3::inspect::inspect_container(&args.container, false)?;
    let outcome = plan_component_ops(&inspection, &args.container, &args.component)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&outcome)?);
    } else if let Some(plan) = &outcome.plan {
        if let Some(embedding) = &plan.embedding {
            println!(
                "embedding: {} vocab {} scale {}",
                embedding.table.object,
                embedding.vocab_size,
                scalar(embedding.scale)
            );
        }
        match args.layer {
            Some(layer) => match plan.layers.iter().find(|l| l.layer == layer) {
                Some(layer_plan) => print_layer(&plan.component, layer_plan),
                None => return Err(format!("no layer {layer} in the plan").into()),
            },
            None => {
                for layer_plan in &plan.layers {
                    match &layer_plan.attention {
                        LayerAttention::Softmax(attention) => println!(
                            "layer {:3}: {:?}{} position {:?}  {}/{} operands accounted",
                            layer_plan.layer,
                            attention.span,
                            attention
                                .window
                                .map(|w| format!("({w})"))
                                .unwrap_or_default(),
                            attention.position,
                            layer_plan.operands_accounted,
                            layer_plan.operands_present,
                        ),
                        LayerAttention::GatedDelta(op) => println!(
                            "layer {:3}: GatedDelta({}k/{}v heads) state {} elems  \
                             {}/{} operands accounted",
                            layer_plan.layer,
                            op.num_key_heads,
                            op.num_value_heads,
                            op.state_elements(),
                            layer_plan.operands_accounted,
                            layer_plan.operands_present,
                        ),
                    }
                }
            }
        }
        if let Some(output) = &plan.output {
            println!(
                "output: {} multiplier {}{}",
                output.projection.object,
                scalar(output.multiplier),
                output
                    .softcapping
                    .map(|c| format!(" softcap {c}"))
                    .unwrap_or_default(),
            );
        }
        eprintln!(
            "plan closed: {} layer(s), every executable operand accounted",
            plan.layers.len()
        );
    }
    if outcome.closed() {
        Ok(())
    } else {
        for defect in &outcome.defects {
            println!("defect: {defect}");
        }
        Err(format!(
            "operand closure failed: {} defect(s)",
            outcome.defects.len()
        )
        .into())
    }
}

fn print_layer(component: &str, layer: &LayerPlan) {
    println!("{component}.layer[{}]", layer.layer);
    let norm = |op: &NormOp, site: &str| {
        println!("  {:?}({site}, eps {:e})", op.kind, op.eps);
    };
    norm(&layer.pre_attention_norm, "pre_attention");
    match &layer.attention {
        LayerAttention::Softmax(op) => print_softmax(op),
        LayerAttention::GatedDelta(op) => print_gated_delta(op),
    }
    println!("  residual");
    if let Some(op) = &layer.post_attention_norm {
        norm(op, "post_attention");
    }
    norm(&layer.pre_ffn_norm, "pre_ffn");
    match &layer.ffn {
        larql_vindex::format::vindex3::opplan::LayerFfn::Dense(ffn) => println!(
            "  {}FFN({:?}, {})",
            if ffn.gate.is_some() { "Gated" } else { "" },
            ffn.activation,
            ffn.intermediate_size
        ),
        larql_vindex::format::vindex3::opplan::LayerFfn::Routed(ffn) => println!(
            "  RoutedFFN({} experts, top-{}, {:?}, {:?}, {}, {:?}{}) router={}/{}, bank={}",
            ffn.experts,
            ffn.top_k,
            ffn.routing_policy,
            ffn.gate_policy,
            ffn.expert_intermediate_size,
            ffn.expert_format,
            if ffn.router_bias.is_some() {
                ", router bias"
            } else {
                ""
            },
            ffn.router.object,
            ffn.router.tensor,
            ffn.gate_up.weights.object,
        ),
        larql_vindex::format::vindex3::opplan::LayerFfn::Hybrid(ffn) => {
            println!(
                "  HybridFFN: dense {}FFN({:?}, {}) → post_dense_norm  +  routed({} experts, \
                 top-{}, {:?}, {:?}, {}, {:?}{}) over pre_experts_norm(residual) → \
                 post_experts_norm; router={}/{}, bank={}",
                if ffn.dense.gate.is_some() {
                    "Gated"
                } else {
                    ""
                },
                ffn.dense.activation,
                ffn.dense.intermediate_size,
                ffn.routed.experts,
                ffn.routed.top_k,
                ffn.routed.router_kind,
                ffn.routed.gate_policy,
                ffn.routed.expert_intermediate_size,
                ffn.routed.expert_format,
                if ffn.routed.router_scale.is_some() {
                    ", router scale + per-expert scale"
                } else {
                    ""
                },
                ffn.routed.router.object,
                ffn.routed.router.tensor,
                ffn.routed.gate_up.weights.object,
            );
        }
    }
    if let Some(op) = &layer.post_ffn_norm {
        norm(op, "post_ffn");
    }
    println!("  residual");
    if let Some(scale) = &layer.layer_scale {
        println!("  × layer_scale {}/{}", scale.object, scale.tensor);
    }
}

/// The softmax attention section of one layer.
fn print_softmax(attention: &AttentionOp) {
    println!("  Attention");
    println!(
        "    geometry: {}q / {}kv, head_dim {}",
        attention.num_q_heads, attention.num_kv_heads, attention.head_dim
    );
    println!(
        "    query_scale {} score_scale {}",
        scalar(attention.query_scale),
        attention.score_scale
    );
    if attention.parameter_free_qk_norm.q || attention.parameter_free_qk_norm.k {
        println!(
            "    parameter_free_qk_norm q={} k={}",
            attention.parameter_free_qk_norm.q, attention.parameter_free_qk_norm.k
        );
    }
    println!(
        "    span {:?}{}",
        attention.span,
        attention
            .window
            .map(|w| format!("({w})"))
            .unwrap_or_default()
    );
    println!("    position {:?}", attention.position);
    if let Some(qk) = &attention.qk_norm {
        println!("    qk_norm {:?}", qk.scope);
    }
    for (name, operand) in [
        ("q", &attention.q),
        ("k", &attention.k),
        ("v", &attention.v),
        ("o", &attention.o),
    ] {
        println!(
            "    {name} = {}/{} {:?}",
            operand.object, operand.tensor, operand.shape
        );
    }
    if let Some(gate) = &attention.output_gate {
        println!(
            "    output_gate {:?} = {}/{}",
            gate.spec.activation, gate.projection.object, gate.projection.tensor
        );
    }
}

/// The Gated DeltaNet section of one layer.
///
/// Deliberately does NOT reuse the softmax vocabulary: there is no span,
/// no window and no KV head count to print, and the one number a reader
/// most needs — the recurrent state's size — has no softmax counterpart.
fn print_gated_delta(op: &GatedDeltaOp) {
    println!("  GatedDeltaNet");
    println!(
        "    geometry: {}k/{}v heads, key_head_dim {}, value_head_dim {}",
        op.num_key_heads, op.num_value_heads, op.key_head_dim, op.value_head_dim
    );
    println!(
        "    conv kernel {}  qkv channels {}",
        op.conv_kernel,
        op.qkv_channels()
    );
    println!(
        "    state: {} elements/layer at {} — constant in sequence length",
        op.state_elements(),
        op.state_dtype
            .map(|d| d.declared_name())
            .unwrap_or("undeclared")
    );
    for (name, operand) in [
        ("in_proj_qkv", &op.in_proj_qkv),
        ("in_proj_a", &op.in_proj_a),
        ("in_proj_b", &op.in_proj_b),
        ("in_proj_z", &op.in_proj_z),
        ("conv1d", &op.conv1d),
        ("a_log", &op.a_log),
        ("dt_bias", &op.dt_bias),
        ("norm", &op.norm),
        ("out_proj", &op.out_proj),
    ] {
        println!(
            "    {name} = {}/{} {:?}",
            operand.object, operand.tensor, operand.shape
        );
    }
}
