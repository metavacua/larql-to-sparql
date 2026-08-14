//! MoE semantics an architecture must **declare**, not leave to inference.
//!
//! Three defects in one week came from the same shape: a fact that decides how
//! stored bytes become mathematics, left implicit and answered by a plausible
//! default, a tensor-name pattern, or whichever code path happened to run.
//! None of them failed loudly — each produced a finite, fluent, wrong model.
//! These tests make the declarations mandatory and pin the specific inference
//! rule that broke.

use larql_models::{detect_from_json, ExpertFormat, ModelArchitecture, MoeRouterKind};
use serde_json::json;

/// The Gemma-spelled substrings the safetensors loader used to match on when
/// deciding which 3-D tensors to preserve as packed expert operands.
const GEMMA_GATE_UP_SUBSTRING: &str = "experts.gate_up_proj";
const GEMMA_DOWN_SUBSTRING: &str = "experts.down_proj";

fn granite_moe() -> Box<dyn ModelArchitecture> {
    detect_from_json(&json!({
        "model_type": "granitemoe",
        "hidden_size": 1024,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "num_local_experts": 32,
        "num_experts_per_tok": 8,
        "vocab_size": 1024,
    }))
}

fn gpt_oss() -> Box<dyn ModelArchitecture> {
    detect_from_json(&json!({
        "model_type": "gpt_oss",
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 4,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_local_experts": 32,
        "num_experts_per_tok": 4,
        "vocab_size": 1024,
    }))
}

fn gemma4_moe() -> Box<dyn ModelArchitecture> {
    detect_from_json(&json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 256,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 256,
            "enable_moe_block": true,
            "num_experts": 8,
            "top_k_experts": 2,
            "moe_intermediate_size": 128,
        }
    }))
}

/// **Admission gate.** A packed expert store has no single reading: GraniteMoE
/// splits its fused operand with `chunk(2, dim=-1)` while GPT-OSS uses
/// `[..., ::2]`. Declaring the storage format therefore does not answer how to
/// read a row, and a reader that guesses computes a different model silently.
/// Every architecture whose experts are packed must say which.
#[test]
fn packed_architectures_declare_their_fused_gate_up_layout() {
    for arch in [granite_moe(), gpt_oss(), gemma4_moe()] {
        if arch.expert_format() == ExpertFormat::PerExpert {
            continue;
        }
        assert!(
            arch.gate_up_layout().is_some(),
            "{} declares {:?} but no gate_up_layout — a packed reader cannot \
             infer this, and reading it wrong is silent",
            arch.family(),
            arch.expert_format(),
        );
    }
}

/// The two packed families genuinely disagree. If this ever collapses to one
/// value the declaration is redundant and should be deleted rather than kept
/// as ceremony — so the disagreement itself is the thing under test.
#[test]
fn the_two_packed_families_use_opposite_fused_layouts() {
    assert_ne!(
        granite_moe().gate_up_layout(),
        gpt_oss().gate_up_layout(),
        "GraniteMoE chunks; GPT-OSS interleaves"
    );
}

/// **The loader bug, pinned.** `should_keep_raw` decided whether a 3-D tensor
/// was a packed expert operand by matching Gemma's tensor spelling. Granite's
/// keys do not contain those substrings, so its experts were never preserved
/// as byte ranges — they fell through to the 3-D arm and were *dropped*. The
/// model then loaded reporting its full layer count with no experts at all.
///
/// The fix asks the architecture for its keys. This test fails if anyone
/// reintroduces the substring rule as the authority.
#[test]
fn granite_packed_keys_are_invisible_to_the_gemma_substring_rule() {
    let arch = granite_moe();
    let gate_up = arch
        .packed_experts_gate_up_key(0)
        .expect("granitemoe declares a packed gate_up key");
    let down = arch
        .packed_experts_down_key(0)
        .expect("granitemoe declares a packed down key");

    assert!(
        gate_up.contains("block_sparse_moe.input_linear"),
        "{gate_up}"
    );
    assert!(down.contains("block_sparse_moe.output_linear"), "{down}");
    assert!(
        !gate_up.contains(GEMMA_GATE_UP_SUBSTRING) && !down.contains(GEMMA_DOWN_SUBSTRING),
        "granite keys must not be reachable by Gemma's spelling — that they \
         are not is exactly why a substring rule silently dropped them"
    );
}

/// `GraniteMoeTopKGating` selects then softmaxes over the selection, so the
/// chosen weights sum to 1. The inherited default softmaxes over all experts
/// and takes the top-k as they are. On the real 1B-A400M that difference is
/// 49.762% of bits/char against the HF reference, with *identical* expert
/// selection on all 294 tokens — the weights alone.
#[test]
fn granite_declares_select_then_normalise_routing() {
    assert_eq!(
        granite_moe().moe_router_kind(),
        MoeRouterKind::TopKThenSoftmax
    );
}

/// Mixtral renormalises too, by a different spelling: `softmax` over all,
/// `topk`, then `/= sum`. That is algebraically softmax over the selected
/// logits, so it is the same declaration as GraniteMoe — and unlike Granite
/// it is asserted from the reference source rather than from a measurement,
/// because the smallest of the family is 8x7B.
#[test]
fn mixtral_declares_select_then_normalise_routing() {
    let arch = detect_from_json(&json!({
        "model_type": "mixtral",
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 1024,
    }));
    assert!(arch.is_moe());
    assert_eq!(arch.moe_router_kind(), MoeRouterKind::TopKThenSoftmax);
    assert_eq!(
        arch.expert_routing_policy(),
        larql_models::ExpertRoutingPolicy::NormalisedOverSelected
    );
    // Per-expert storage, so no fused operand and no layout to declare.
    assert_eq!(arch.expert_format(), ExpertFormat::PerExpert);
    assert!(arch.gate_up_layout().is_none());
}
