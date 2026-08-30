//! Expert-routing-policy detection: the config is the authority unless an
//! architecture owns the order outright.

use crate::config::ExpertRoutingPolicy;
use crate::detect::*;

/// A Qwen3-30B-A3B config with `norm_topk_prob` set, cleared, or absent.
fn qwen3_moe_config(norm_topk_prob: Option<bool>) -> serde_json::Value {
    let mut config = serde_json::json!({
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_experts": 128,
        "num_experts_per_tok": 8
    });
    if let Some(flag) = norm_topk_prob {
        config["norm_topk_prob"] = serde_json::json!(flag);
    }
    config
}

/// The routing order is a fact about the checkpoint, so the default reads it.
///
/// Guard for the third recurrence of `docs/k3-funnel.md` §4.7 finding 5: the
/// `norm_topk_prob` read lived on `OlmoeArch` alone, so every other MoE
/// architecture silently took `SoftmaxThenSelect` whatever its config said —
/// a rescale of the entire expert branch that still produces fluent output.
#[test]
fn routing_policy_is_read_from_config_not_assumed() {
    let policy_for = |flag| detect_from_json(&qwen3_moe_config(flag)).expert_routing_policy();

    assert_eq!(
        policy_for(Some(true)),
        ExpertRoutingPolicy::NormalisedOverSelected,
        "norm_topk_prob: true renormalises the selected weights"
    );
    assert_eq!(
        policy_for(Some(false)),
        ExpertRoutingPolicy::SoftmaxThenSelect
    );
    // Absent is the conservative reading: the checkpoint never asked for a
    // renormalisation, so none is applied.
    assert_eq!(policy_for(None), ExpertRoutingPolicy::SoftmaxThenSelect);
}

/// GPT-OSS fixes its routing order in the architecture rather than in config,
/// so its override has to survive a config that says otherwise.
#[test]
fn an_architecture_owned_routing_order_beats_the_config() {
    let config = serde_json::json!({
        "model_type": "gpt_oss",
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "num_hidden_layers": 24,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "num_local_experts": 32,
        "experts_per_token": 4,
        "norm_topk_prob": false
    });

    assert_eq!(
        detect_from_json(&config).expert_routing_policy(),
        ExpertRoutingPolicy::NormalisedOverSelected
    );
}
