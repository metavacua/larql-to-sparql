//! The execution surface: what generic operations need in order to
//! execute a component (V3-G5a).
//!
//! [`super::Component`] answers *what part of the system this is*; the
//! surface answers *what the generic ops need to run it*. Fields are
//! grouped **by operation**, because the completeness contract derives
//! from the operations a component's objects imply — never from what any
//! particular architecture happens to declare:
//!
//! ```text
//! DecoderStack / PerceptionTower object  →  attention, ffn, norm
//! Embedding / OutputHead object          →  head
//! ```
//!
//! Every value is **fully resolved**: defaulting decisions (an absent
//! `hidden_act`, a post-norm epsilon shared with `norm_eps`, canonical
//! 1/√d attention scaling) are applied at build/judgment time and
//! persisted. A generic executor reads these fields; it never defaults an
//! absent one — absence is a completeness defect that refuses encoding,
//! not a branch at run time.

use larql_models::config::{
    Activation, AttentionGateSpec, FfnType, NormType, ParameterFreeQkNorm, QkNormScope,
};
use larql_models::inventory::components::ComponentTopology;
use larql_models::inventory::ArchitectureInventory;
use serde::{Deserialize, Serialize};

use super::object::LogicalObject;

/// Tensor-name fragments evidencing a gated FFN under a binding. Evidence,
/// not a family fact: presence of gate weights decides, whoever ships
/// them. One definition, shared by the builder and the G4 re-derivation.
const GATE_TENSOR_FRAGMENTS: &[&str] = &["gate_proj", "gate_up"];

/// Whether any tensor bound by `object` carries gate-FFN evidence.
pub fn gate_evidence(inventory: &ArchitectureInventory, object: &LogicalObject) -> bool {
    object.source_bindings.iter().any(|binding| {
        inventory
            .tensors
            .tensors
            .iter()
            .filter(|t| t.name.starts_with(&binding.tensor_prefix))
            .any(|t| {
                GATE_TENSOR_FRAGMENTS
                    .iter()
                    .any(|fragment| t.name.contains(fragment))
            })
    })
}

/// What the attention op reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionSurface {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// Multiplier on the (normalised) query states before position
    /// encoding — a declared factor, or 1.0. Kept separate from
    /// [`Self::score_scale`]: folding them is algebra-equivalent but not
    /// fp-equivalent, and the executor must place each multiply where
    /// the judged semantics put it.
    pub query_scale: f64,
    /// Canonical score-time multiplier on QK^T.
    pub score_scale: f64,
    /// Attention-logit softcap; `None` = the op is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_softcapping: Option<f32>,
    /// QK-norm scope, read when QK-norm weights exist in the stack.
    pub qk_norm_scope: QkNormScope,
    pub qk_norm_weight_offset: f32,
    /// Parameter-free QK normalisation (no weight tensors) — judged
    /// semantics no tensor evidence can reveal.
    #[serde(default)]
    pub parameter_free_qk_norm: ParameterFreeQkNorm,
    /// Judged attention-output-gate semantics; `None` = no judgment
    /// exists — **never "no gate"**. A stack shipping an
    /// [`OperandRole::AttnOutputGate`](super::roles::OperandRole) operand
    /// while this is `None` fails operand closure — the primitive exists
    /// in the IR, but its semantics for that model have not been judged,
    /// and closure refuses rather than guessing an activation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_gate: Option<AttentionGateSpec>,
}

/// What the FFN op reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FfnSurface {
    pub intermediate_size: usize,
    pub activation: Activation,
    pub ffn_type: FfnType,
}

/// What the norm op reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormSurface {
    pub kind: NormType,
    pub eps: f64,
    /// Post-norm epsilon; equals `eps` unless the source declared its own.
    pub post_norm_eps: f64,
    /// Norm placement around attention/FFN, judged from operand evidence
    /// ([`super::roles::norm_placement_evidence`]) — never from a family
    /// default, which is exactly the fact the generic fallback got wrong
    /// on the first real four-norm stack. Count is not semantics;
    /// placement is. `None` only for components with no decoder stack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<super::roles::NormPlacement>,
    /// Offset added to norm weights at runtime.
    pub weight_offset: f32,
}

/// What embedding lookup and the output head read. Present iff the
/// component owns embedding/output-head objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadSurface {
    pub vocab_size: usize,
    /// Multiplier applied to embeddings after lookup; 1.0 = identity.
    pub embed_scale: f32,
    /// Multiplier applied before the vocabulary projection; 1.0 = identity.
    pub output_multiplier: f64,
    /// Final-logit softcap; `None` = the op is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f32>,
}

/// The complete per-component execution surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionSurface {
    pub attention: AttentionSurface,
    pub ffn: FfnSurface,
    pub norm: NormSurface,
    /// Present iff the component owns embedding/output-head objects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<HeadSurface>,
}

/// Build the surface for a text-path component (target/drafter) from its
/// inventory's resolution. Returns the missing source facts when the
/// surface cannot be completed — the caller turns those into blocking
/// findings, never into defaults.
pub fn surface_from_resolved(
    inventory: &ArchitectureInventory,
) -> Result<ExecutionSurface, Vec<String>> {
    let resolved = &inventory.resolved;
    let Some(execution) = &resolved.execution else {
        return Err(vec![
            "resolved.execution (pre-v3 inventory — re-run inspect-hf)".to_string(),
        ]);
    };
    // The surface carries one head geometry for the whole component; a
    // per-layer variation is a schema gap to surface, not to average away.
    let mut missing = Vec::new();
    for layer in &resolved.layers {
        if layer.head_dim != resolved.head_dim || layer.num_kv_heads != resolved.num_kv_heads {
            missing.push(format!(
                "uniform head geometry (layer {} has head_dim {} / kv {}, component says {} / {})",
                layer.layer,
                layer.head_dim,
                layer.num_kv_heads,
                resolved.head_dim,
                resolved.num_kv_heads
            ));
            break;
        }
    }
    if !missing.is_empty() {
        return Err(missing);
    }
    Ok(ExecutionSurface {
        attention: AttentionSurface {
            num_q_heads: resolved.num_q_heads,
            num_kv_heads: resolved.num_kv_heads,
            head_dim: resolved.head_dim,
            query_scale: execution.query_scale,
            score_scale: execution.score_scale,
            logit_softcapping: execution.attn_logit_softcapping,
            qk_norm_scope: execution.qk_norm_scope,
            qk_norm_weight_offset: execution.qk_norm_weight_offset,
            parameter_free_qk_norm: execution.parameter_free_qk_norm,
            // Judged per model; never inferred from operand presence.
            output_gate: execution.attention_output_gate,
        },
        ffn: FfnSurface {
            intermediate_size: resolved.intermediate_size,
            activation: execution.activation,
            ffn_type: execution.ffn_type,
        },
        norm: NormSurface {
            kind: execution.norm_kind,
            eps: execution.norm_eps,
            post_norm_eps: execution.post_norm_eps,
            // From operand evidence via `attach_stack_evidence`, once the
            // builder knows the component's stack object.
            placement: None,
            weight_offset: execution.norm_weight_offset,
        },
        // Attached by the builder once it knows the component's objects.
        head: None,
    })
}

/// Attach the facts only the stack's operand estate can state: norm
/// placement from the norm-role evidence across the stack's bindings.
/// Shared by the builder and the G4 re-derivation, so the two can never
/// judge the same bytes differently.
pub fn attach_stack_evidence(
    surface: &mut ExecutionSurface,
    inventory: &ArchitectureInventory,
    stack: &LogicalObject,
) -> Result<(), Vec<String>> {
    let relative: Vec<String> = stack
        .source_bindings
        .iter()
        .flat_map(|binding| {
            inventory
                .tensors
                .tensors
                .iter()
                .filter_map(|t| t.name.strip_prefix(&binding.tensor_prefix))
                .map(|rest| rest.trim_start_matches('.').to_string())
        })
        .collect();
    match super::roles::norm_placement_evidence(relative.iter().map(String::as_str)) {
        Ok(placement) => {
            surface.norm.placement = Some(placement);
            Ok(())
        }
        Err(reason) => Err(vec![format!("norm placement ({reason})")]),
    }
}

/// The head surface for a component that owns embedding/output-head
/// objects, from the same resolution.
pub fn head_from_resolved(inventory: &ArchitectureInventory) -> Result<HeadSurface, Vec<String>> {
    let resolved = &inventory.resolved;
    let mut missing = Vec::new();
    let Some(execution) = &resolved.execution else {
        return Err(vec![
            "resolved.execution (pre-v3 inventory — re-run inspect-hf)".to_string(),
        ]);
    };
    let Some(vocab_size) = resolved.vocab_size else {
        missing.push("vocab_size".to_string());
        return Err(missing);
    };
    Ok(HeadSurface {
        vocab_size,
        embed_scale: execution.embed_scale,
        output_multiplier: execution.output_multiplier,
        final_logit_softcapping: execution.final_logit_softcapping,
    })
}

/// Build the surface for a perception component from its nested config
/// reading plus tensor evidence.
///
/// A nested component has no detection trait to resolve through, so the
/// judged derivations live here, in one place: MHA (kv = q) unless
/// declared otherwise, `head_dim = hidden/heads` when divisible, canonical
/// 1/√d scaling, norm kind from which epsilon spelling the config
/// declares, and FFN gating from whether gate tensors exist under the
/// tower (`has_gate_tensors` — evidence, not a family fact).
pub fn surface_from_nested(
    nested: &ComponentTopology,
    has_gate_tensors: bool,
) -> Result<ExecutionSurface, Vec<String>> {
    let mut missing = Vec::new();
    let hidden = nested.hidden_size.unwrap_or(0);
    let heads = match nested.num_attention_heads {
        Some(h) if h > 0 => h,
        _ => {
            missing.push("num_attention_heads".to_string());
            0
        }
    };
    let head_dim = match nested.head_dim {
        Some(d) => d,
        None if heads > 0 && hidden.is_multiple_of(heads) => hidden / heads,
        None => {
            missing.push(format!(
                "head_dim (hidden {hidden} not divisible by {heads} heads)"
            ));
            0
        }
    };
    let intermediate_size = match nested.intermediate_size {
        Some(i) => i,
        None => {
            missing.push("intermediate_size".to_string());
            0
        }
    };
    let activation = match nested
        .hidden_act
        .as_deref()
        .and_then(Activation::from_hf_name)
    {
        Some(a) => a,
        None => {
            missing.push(format!(
                "hidden_act (declared {:?}, judged mapping required)",
                nested.hidden_act
            ));
            Activation::Gelu
        }
    };
    // The epsilon spelling the config declares names the norm kind; a
    // component declaring neither has no norm surface to persist.
    let (kind, eps) = match (nested.norm_kind, nested.norm_eps) {
        (Some(kind), Some(eps)) => (kind, eps),
        _ => {
            missing.push("norm_eps".to_string());
            (NormType::LayerNorm, 0.0)
        }
    };
    if !missing.is_empty() {
        return Err(missing);
    }
    Ok(ExecutionSurface {
        attention: AttentionSurface {
            num_q_heads: heads,
            num_kv_heads: nested.num_key_value_heads.unwrap_or(heads),
            head_dim,
            query_scale: 1.0,
            score_scale: (head_dim as f64).powf(-0.5),
            logit_softcapping: None,
            qk_norm_scope: QkNormScope::PerHead,
            qk_norm_weight_offset: 0.0,
            parameter_free_qk_norm: ParameterFreeQkNorm::default(),
            output_gate: None,
        },
        ffn: FfnSurface {
            intermediate_size,
            activation,
            ffn_type: if has_gate_tensors {
                FfnType::Gated
            } else {
                FfnType::Standard
            },
        },
        norm: NormSurface {
            kind,
            eps,
            post_norm_eps: eps,
            // Perception towers keep their own norm topology; placement
            // is a decoder-stack concept until the perception op set (5d)
            // defines its own.
            placement: None,
            weight_offset: 0.0,
        },
        head: None,
    })
}
