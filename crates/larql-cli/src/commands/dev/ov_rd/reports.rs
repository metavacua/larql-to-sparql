#![allow(dead_code)]

use serde::{Deserialize, Serialize};

use super::types::{HeadId, PqConfig};

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct FinishedHeadStats {
    pub(super) count: u64,
    pub(super) mean_norm_sq: f64,
    pub(super) second_moment: f64,
    pub(super) variance: f64,
    pub(super) rms_norm: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct HeadReport {
    pub(super) layer: usize,
    pub(super) head: usize,
    pub(super) head_dim: usize,
    pub(super) stats: FinishedHeadStats,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) wo_visible_stats: Option<FinishedHeadStats>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct CaptureReport {
    pub(super) index: String,
    pub(super) prompt_file: String,
    pub(super) prompts_seen: usize,
    pub(super) layers: Vec<usize>,
    pub(super) max_positions: Option<usize>,
    #[serde(default)]
    pub(super) wo_visible: bool,
    pub(super) heads: Vec<HeadReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct ZeroStratumReport {
    pub(super) stratum: String,
    pub(super) prompts: usize,
    pub(super) mean_kl: f64,
    pub(super) max_kl: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct ZeroPromptReport {
    pub(super) id: String,
    pub(super) stratum: String,
    pub(super) kl: f64,
    pub(super) delta_cross_entropy_bits: f64,
    pub(super) baseline_top1: u32,
    pub(super) ablated_top1: u32,
    pub(super) top1_agree: bool,
    pub(super) baseline_top1_in_ablated_top5: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct ZeroHeadReport {
    pub(super) layer: usize,
    pub(super) head: usize,
    pub(super) ablation_kind: String,
    pub(super) patch_location: String,
    pub(super) preserved_components: Vec<String>,
    pub(super) bounded_vocab_size: Option<usize>,
    pub(super) prompts: usize,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) mean_delta_cross_entropy_bits: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
    pub(super) strata: Vec<ZeroStratumReport>,
    pub(super) worst_examples: Vec<ZeroPromptReport>,
    pub(super) per_prompt: Vec<ZeroPromptReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct ZeroAblationReport {
    pub(super) index: String,
    pub(super) prompt_file: String,
    pub(super) prompts_seen: usize,
    pub(super) selected_heads: Vec<HeadId>,
    pub(super) heads: Vec<ZeroHeadReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct StaticReplacementReport {
    pub(super) index: String,
    pub(super) prompt_file: String,
    pub(super) prompts_seen: usize,
    pub(super) train_prompts_seen: usize,
    pub(super) eval_prompts_seen: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) eval_mod: Option<usize>,
    pub(super) eval_offset: usize,
    pub(super) selected_heads: Vec<HeadId>,
    pub(super) heads: Vec<StaticHeadReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct StaticHeadReport {
    pub(super) layer: usize,
    pub(super) head: usize,
    pub(super) train_samples: u64,
    pub(super) modes: Vec<StaticModeReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct StaticModeReport {
    pub(super) replacement_kind: String,
    pub(super) patch_location: String,
    pub(super) runtime_class: String,
    pub(super) prompts: usize,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) mean_delta_cross_entropy_bits: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
    pub(super) strata: Vec<ZeroStratumReport>,
    pub(super) worst_examples: Vec<ZeroPromptReport>,
    pub(super) per_prompt: Vec<ZeroPromptReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct SanityCheckReport {
    pub(super) index: String,
    pub(super) prompt_file: String,
    pub(super) prompts_seen: usize,
    pub(super) selected_heads: Vec<HeadId>,
    pub(super) heads: Vec<SanityHeadReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct SanityHeadReport {
    pub(super) layer: usize,
    pub(super) head: usize,
    pub(super) prompts: usize,
    pub(super) noop_mean_kl: f64,
    pub(super) noop_max_kl: f64,
    pub(super) noop_max_abs_logit_diff: f64,
    pub(super) residual_delta_noop_mean_kl: f64,
    pub(super) residual_delta_noop_max_kl: f64,
    pub(super) residual_delta_noop_max_abs_logit_diff: f64,
    pub(super) zero_subtract_mean_kl: f64,
    pub(super) zero_subtract_max_kl: f64,
    pub(super) zero_subtract_max_abs_logit_diff: f64,
    pub(super) per_prompt: Vec<SanityPromptReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct SanityPromptReport {
    pub(super) id: String,
    pub(super) stratum: String,
    pub(super) noop_kl: f64,
    pub(super) noop_max_abs_logit_diff: f64,
    pub(super) residual_delta_noop_kl: f64,
    pub(super) residual_delta_noop_max_abs_logit_diff: f64,
    pub(super) zero_subtract_kl: f64,
    pub(super) zero_subtract_max_abs_logit_diff: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct OracleRoundtripReport {
    pub(super) index: String,
    pub(super) prompt_file: String,
    pub(super) prompts_seen: usize,
    pub(super) sigma_rel_cutoff: f64,
    pub(super) selected_heads: Vec<HeadId>,
    pub(super) heads: Vec<OracleRoundtripHeadReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct OracleRoundtripHeadReport {
    pub(super) layer: usize,
    pub(super) head: usize,
    pub(super) head_dim: usize,
    pub(super) rank_retained: usize,
    pub(super) sigma_max: f64,
    pub(super) sigma_min_retained: f64,
    pub(super) sigma_rel_cutoff: f64,
    pub(super) prompts: usize,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) max_abs_logit_diff: f64,
    pub(super) mean_pre_wo_l2: f64,
    pub(super) max_pre_wo_l2: f64,
    pub(super) mean_wo_visible_l2: f64,
    pub(super) max_wo_visible_l2: f64,
    pub(super) per_prompt: Vec<OracleRoundtripPromptReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OracleRoundtripPromptReport {
    pub(super) id: String,
    pub(super) stratum: String,
    pub(super) kl: f64,
    pub(super) max_abs_logit_diff: f64,
    pub(super) pre_wo_l2: f64,
    pub(super) wo_visible_l2: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct OracleLowrankReport {
    pub(super) index: String,
    pub(super) prompt_file: String,
    pub(super) prompts_seen: usize,
    pub(super) static_base: String,
    pub(super) ks: Vec<usize>,
    pub(super) sigma_rel_cutoff: f64,
    pub(super) selected_heads: Vec<HeadId>,
    pub(super) heads: Vec<OracleLowrankHeadReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct OracleLowrankHeadReport {
    pub(super) layer: usize,
    pub(super) head: usize,
    pub(super) head_dim: usize,
    pub(super) rank_retained: usize,
    pub(super) empirical_rank: usize,
    pub(super) sigma_max: f64,
    pub(super) sigma_min_retained: f64,
    pub(super) static_train_samples: u64,
    pub(super) points: Vec<OracleLowrankPointReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct OracleLowrankPointReport {
    pub(super) k: usize,
    pub(super) prompts: usize,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) mean_delta_cross_entropy_bits: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
    pub(super) mean_baseline_top1_prob: f64,
    pub(super) mean_lowrank_prob_of_baseline_top1: f64,
    pub(super) mean_baseline_top1_margin: f64,
    pub(super) mean_pre_wo_l2: f64,
    pub(super) mean_wo_visible_l2: f64,
    pub(super) per_prompt: Vec<OracleLowrankPromptReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OracleLowrankPromptReport {
    pub(super) id: String,
    pub(super) stratum: String,
    pub(super) kl: f64,
    pub(super) delta_cross_entropy_bits: f64,
    pub(super) baseline_top1: u32,
    pub(super) lowrank_top1: u32,
    pub(super) top1_agree: bool,
    pub(super) baseline_top1_in_lowrank_top5: bool,
    pub(super) baseline_top1_prob: f64,
    pub(super) baseline_top2: u32,
    pub(super) baseline_top2_prob: f64,
    pub(super) baseline_top1_margin: f64,
    pub(super) lowrank_top1_prob: f64,
    pub(super) lowrank_prob_of_baseline_top1: f64,
    pub(super) lowrank_top1_margin: f64,
    pub(super) pre_wo_l2: f64,
    pub(super) wo_visible_l2: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct OraclePqReport {
    pub(super) index: String,
    pub(super) prompt_file: String,
    pub(super) prompts_seen: usize,
    pub(super) train_prompts_seen: usize,
    pub(super) eval_prompts_seen: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) max_per_stratum: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) eval_mod: Option<usize>,
    pub(super) eval_offset: usize,
    pub(super) static_base: String,
    pub(super) configs: Vec<PqConfig>,
    pub(super) sigma_rel_cutoff: f64,
    pub(super) pq_iters: usize,
    pub(super) mode_d_check: bool,
    pub(super) address_probes: bool,
    pub(super) address_mixed_key_probe: bool,
    pub(super) address_key_group_probe: bool,
    pub(super) address_key_groups: Vec<usize>,
    pub(super) address_key_group_probe_names: Vec<String>,
    pub(super) address_majority_group_probe: bool,
    pub(super) address_majority_groups: Vec<usize>,
    pub(super) address_code_substitution_group_probe: bool,
    pub(super) address_code_substitution_groups: Vec<usize>,
    pub(super) address_code_substitution_from_codes: Vec<usize>,
    pub(super) address_code_substitution_to_codes: Vec<String>,
    pub(super) address_code_class_collapse_group_probe: bool,
    pub(super) address_code_class_collapse_groups: Vec<usize>,
    pub(super) address_code_class_collapse_specs: Vec<String>,
    pub(super) address_code_position_interaction_probe: bool,
    pub(super) address_code_position_prompt_id: String,
    pub(super) address_code_position_group: usize,
    pub(super) address_code_position_primary_codes: Vec<usize>,
    pub(super) address_code_position_secondary_codes: Vec<usize>,
    pub(super) address_code_position_target_code: usize,
    pub(super) address_code_conditional_quotient_group_probe: bool,
    pub(super) address_code_conditional_quotient_group: usize,
    pub(super) address_code_conditional_quotient_primary_codes: Vec<usize>,
    pub(super) address_code_conditional_quotient_secondary_codes: Vec<usize>,
    pub(super) address_code_conditional_quotient_target_code: usize,
    pub(super) address_code_conditional_quotient_early_position_max: usize,
    pub(super) address_code_conditional_quotient_guards: Vec<String>,
    pub(super) address_code_conditional_quotient_extra_specs: Vec<String>,
    pub(super) address_code7_bos_rule_group_probe: bool,
    pub(super) address_code7_bos_rule_groups: Vec<usize>,
    pub(super) address_code7_bos_rule_code: usize,
    pub(super) address_code7_oracle_binary_group_probe: bool,
    pub(super) address_code7_oracle_binary_groups: Vec<usize>,
    pub(super) address_code7_oracle_binary_code: usize,
    pub(super) address_code7_oracle_binary_filters: Vec<String>,
    pub(super) address_corruption_sweep: bool,
    pub(super) address_group_importance: bool,
    pub(super) address_lsh_group_probe: bool,
    pub(super) address_lsh_groups: Vec<usize>,
    pub(super) address_lsh_bits: usize,
    pub(super) address_lsh_seeds: usize,
    pub(super) address_supervised_group_probe: bool,
    pub(super) address_supervised_groups: Vec<usize>,
    pub(super) address_supervised_epochs: usize,
    pub(super) address_supervised_lr: f32,
    pub(super) address_supervised_l2: f32,
    pub(super) address_gamma_projected_group_probe: bool,
    pub(super) address_gamma_projected_groups: Vec<usize>,
    pub(super) address_gamma_projected_layers: Vec<usize>,
    pub(super) address_gamma_random_ranks: Vec<usize>,
    pub(super) address_gamma_random_seeds: Vec<u64>,
    pub(super) address_gamma_learned_ranks: Vec<usize>,
    pub(super) address_gamma_learned_epochs: usize,
    pub(super) address_gamma_learned_lr: f32,
    pub(super) address_gamma_learned_l2: f32,
    pub(super) address_gamma_learned_pca_iters: usize,
    pub(super) address_code_stability: bool,
    pub(super) address_code_stability_groups: Vec<usize>,
    pub(super) address_prev_ffn_feature_group_probe: bool,
    pub(super) address_prev_ffn_feature_groups: Vec<usize>,
    pub(super) address_prev_ffn_feature_top_k: usize,
    pub(super) address_ffn_first_feature_group_probe: bool,
    pub(super) address_ffn_first_feature_groups: Vec<usize>,
    pub(super) address_ffn_first_feature_top_k: usize,
    pub(super) address_attention_relation_group_probe: bool,
    pub(super) address_attention_relation_groups: Vec<usize>,
    pub(super) address_attention_cluster_group_probe: bool,
    pub(super) address_attention_cluster_groups: Vec<usize>,
    pub(super) address_attention_cluster_ks: Vec<usize>,
    pub(super) address_attention_cluster_probe_names: Vec<String>,
    pub(super) address_reduced_qk_cluster_group_probe: bool,
    pub(super) address_reduced_qk_cluster_groups: Vec<usize>,
    pub(super) address_reduced_qk_ranks: Vec<usize>,
    pub(super) address_reduced_qk_cluster_ks: Vec<usize>,
    pub(super) address_reduced_qk_cluster_probe_names: Vec<String>,
    pub(super) stratum_conditioned_pq_groups: Vec<usize>,
    pub(super) selected_heads: Vec<HeadId>,
    pub(super) heads: Vec<OraclePqHeadReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct CodeOccurrenceRecord {
    pub(super) prompt_id: String,
    pub(super) stratum: String,
    pub(super) layer: usize,
    pub(super) head: usize,
    pub(super) config: PqConfig,
    pub(super) group: usize,
    pub(super) code: usize,
    pub(super) position: usize,
    pub(super) token_id: u32,
    pub(super) token_text: String,
    pub(super) prev_token_id: Option<u32>,
    pub(super) prev_token_text: Option<String>,
    pub(super) attn_argmax_position: Option<usize>,
    pub(super) attn_argmax_token_id: Option<u32>,
    pub(super) attn_argmax_token_text: Option<String>,
    pub(super) attn_entropy_bits: Option<f64>,
    pub(super) attn_relation_class_key: Option<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct OraclePqHeadReport {
    pub(super) layer: usize,
    pub(super) head: usize,
    pub(super) head_dim: usize,
    pub(super) rank_retained: usize,
    pub(super) empirical_rank: usize,
    pub(super) sigma_max: f64,
    pub(super) sigma_min_retained: f64,
    pub(super) static_train_samples: u64,
    pub(super) points: Vec<OraclePqPointReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct OraclePqPointReport {
    pub(super) k: usize,
    pub(super) groups: usize,
    pub(super) bits_per_group: usize,
    pub(super) oracle_address_bits: usize,
    pub(super) coefficient_codebook_bytes_f32: usize,
    pub(super) mode_d_residual_table_bytes_bf16: usize,
    pub(super) prompts: usize,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) mean_delta_cross_entropy_bits: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
    pub(super) mean_baseline_top1_prob: f64,
    pub(super) mean_pq_prob_of_baseline_top1: f64,
    pub(super) mean_baseline_top1_margin: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mode_d_mean_kl: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mode_d_p95_kl: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mode_d_max_kl: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mode_d_top1_agreement: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mode_d_top5_contains_baseline_top1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) coeff_mode_d_max_abs_logit_diff: Option<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) address_probes: Vec<AddressProbeReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) address_corruption_sweep: Vec<AddressCorruptionReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) address_group_importance: Vec<AddressGroupImportanceReport>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) code_stability: Vec<CodeStabilityReport>,
    pub(super) mean_pre_wo_l2: f64,
    pub(super) mean_wo_visible_l2: f64,
    pub(super) per_prompt: Vec<OraclePqPromptReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct CodeStabilityReport {
    pub(super) group: usize,
    pub(super) train_positions: usize,
    pub(super) eval_positions: usize,
    pub(super) train_entropy_bits: f64,
    pub(super) eval_entropy_bits: f64,
    pub(super) train_top_code: usize,
    pub(super) train_top_code_mass: f64,
    pub(super) eval_top_code: usize,
    pub(super) eval_top_code_mass: f64,
    pub(super) train_eval_js_bits: f64,
    pub(super) by_stratum: Vec<CodeStabilityStratumReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct CodeStabilityStratumReport {
    pub(super) stratum: String,
    pub(super) train_positions: usize,
    pub(super) eval_positions: usize,
    pub(super) train_entropy_bits: f64,
    pub(super) eval_entropy_bits: f64,
    pub(super) train_top_code: usize,
    pub(super) train_top_code_mass: f64,
    pub(super) eval_top_code: usize,
    pub(super) eval_top_code_mass: f64,
    pub(super) train_eval_js_bits: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct AddressProbeReport {
    pub(super) name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) selected_group_keys: Vec<String>,
    pub(super) prompts: usize,
    pub(super) positions: usize,
    pub(super) group_accuracy: f64,
    pub(super) exact_address_accuracy: f64,
    pub(super) mean_groups_correct_per_sequence: f64,
    pub(super) mean_groups_correct_per_position: f64,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) by_stratum: Vec<AddressProbeStratumReport>,
    pub(super) worst_examples: Vec<AddressProbePromptReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct AddressProbeStratumReport {
    pub(super) stratum: String,
    pub(super) prompts: usize,
    pub(super) positions: usize,
    pub(super) group_accuracy: f64,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct AddressProbePromptReport {
    pub(super) id: String,
    pub(super) stratum: String,
    pub(super) kl: f64,
    pub(super) positions: usize,
    pub(super) groups_correct: usize,
    pub(super) groups_total: usize,
    pub(super) exact_address_match: bool,
    pub(super) top1_agree: bool,
    pub(super) baseline_top1_in_predicted_top5: bool,
}

#[derive(Debug, Serialize)]
pub(super) struct AddressCorruptionReport {
    pub(super) label: String,
    pub(super) oracle_groups_kept: usize,
    pub(super) prompts: usize,
    pub(super) positions: usize,
    pub(super) group_accuracy: f64,
    pub(super) exact_address_accuracy: f64,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
    pub(super) worst_examples: Vec<AddressProbePromptReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct AddressGroupImportanceReport {
    pub(super) replaced_group: usize,
    pub(super) prompts: usize,
    pub(super) positions: usize,
    pub(super) group_accuracy: f64,
    pub(super) exact_address_accuracy: f64,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
    pub(super) worst_examples: Vec<AddressProbePromptReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OraclePqPromptReport {
    pub(super) id: String,
    pub(super) stratum: String,
    pub(super) kl: f64,
    pub(super) delta_cross_entropy_bits: f64,
    pub(super) baseline_top1: u32,
    pub(super) pq_top1: u32,
    pub(super) top1_agree: bool,
    pub(super) baseline_top1_in_pq_top5: bool,
    pub(super) baseline_top1_prob: f64,
    pub(super) baseline_top2: u32,
    pub(super) baseline_top2_prob: f64,
    pub(super) baseline_top1_margin: f64,
    pub(super) pq_top1_prob: f64,
    pub(super) pq_prob_of_baseline_top1: f64,
    pub(super) pq_top1_margin: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mode_d_kl: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mode_d_top1: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) mode_d_top1_agree: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) baseline_top1_in_mode_d_top5: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) coeff_mode_d_max_abs_logit_diff: Option<f64>,
    pub(super) pre_wo_l2: f64,
    pub(super) wo_visible_l2: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct OracleEditCatalogReport {
    pub(super) index: String,
    pub(super) prompt_file: String,
    pub(super) prompts_seen: usize,
    pub(super) train_prompts_seen: usize,
    pub(super) eval_prompts_seen: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) max_per_stratum: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) eval_mod: Option<usize>,
    pub(super) eval_offset: usize,
    pub(super) static_base: String,
    pub(super) spaces: Vec<String>,
    pub(super) edit_counts: Vec<usize>,
    pub(super) pca_rank: usize,
    pub(super) sigma_rel_cutoff: f64,
    pub(super) kmeans_iters: usize,
    pub(super) selected_heads: Vec<HeadId>,
    pub(super) heads: Vec<OracleEditCatalogHeadReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct OracleEditCatalogHeadReport {
    pub(super) layer: usize,
    pub(super) head: usize,
    pub(super) head_dim: usize,
    pub(super) rank_retained: usize,
    pub(super) empirical_rank: usize,
    pub(super) sigma_max: f64,
    pub(super) sigma_min_retained: f64,
    pub(super) static_train_samples: u64,
    pub(super) points: Vec<OracleEditCatalogPointReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct OracleEditCatalogPointReport {
    pub(super) space: String,
    pub(super) edits: usize,
    pub(super) address_bits: usize,
    pub(super) residual_table_bytes_bf16: usize,
    pub(super) prompts: usize,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) mean_delta_cross_entropy_bits: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
    pub(super) mean_baseline_top1_prob: f64,
    pub(super) mean_catalog_prob_of_baseline_top1: f64,
    pub(super) mean_baseline_top1_margin: f64,
    pub(super) per_prompt: Vec<OracleEditCatalogPromptReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OracleEditCatalogPromptReport {
    pub(super) id: String,
    pub(super) stratum: String,
    pub(super) kl: f64,
    pub(super) delta_cross_entropy_bits: f64,
    pub(super) baseline_top1: u32,
    pub(super) catalog_top1: u32,
    pub(super) top1_agree: bool,
    pub(super) baseline_top1_in_catalog_top5: bool,
    pub(super) baseline_top1_prob: f64,
    pub(super) baseline_top2: u32,
    pub(super) baseline_top2_prob: f64,
    pub(super) baseline_top1_margin: f64,
    pub(super) catalog_top1_prob: f64,
    pub(super) catalog_prob_of_baseline_top1: f64,
    pub(super) catalog_top1_margin: f64,
}

#[derive(Debug, Serialize)]
pub(super) struct OraclePqExceptionReport {
    pub(super) index: String,
    pub(super) prompt_file: String,
    pub(super) prompts_seen: usize,
    pub(super) train_prompts_seen: usize,
    pub(super) eval_prompts_seen: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) max_per_stratum: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) eval_mod: Option<usize>,
    pub(super) eval_offset: usize,
    pub(super) static_base: String,
    pub(super) base_config: PqConfig,
    pub(super) exception_edits: Vec<usize>,
    pub(super) tail_fracs: Vec<f64>,
    pub(super) tail_selector: String,
    pub(super) exception_fit: String,
    pub(super) position_candidates_per_prompt: usize,
    pub(super) sigma_rel_cutoff: f64,
    pub(super) pq_iters: usize,
    pub(super) exception_iters: usize,
    pub(super) selected_heads: Vec<HeadId>,
    pub(super) heads: Vec<OraclePqExceptionHeadReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct OraclePqExceptionHeadReport {
    pub(super) layer: usize,
    pub(super) head: usize,
    pub(super) head_dim: usize,
    pub(super) rank_retained: usize,
    pub(super) empirical_rank: usize,
    pub(super) sigma_max: f64,
    pub(super) sigma_min_retained: f64,
    pub(super) static_train_samples: u64,
    pub(super) points: Vec<OraclePqExceptionPointReport>,
}

#[derive(Debug, Serialize)]
pub(super) struct OraclePqExceptionPointReport {
    pub(super) exception_edits: usize,
    pub(super) tail_frac: f64,
    pub(super) train_error_samples: usize,
    pub(super) train_error_samples_used: usize,
    pub(super) base_address_bits: usize,
    pub(super) exception_address_bits: usize,
    pub(super) total_address_bits: usize,
    pub(super) base_table_bytes_bf16: usize,
    pub(super) exception_table_bytes_bf16: usize,
    pub(super) total_table_bytes_bf16: usize,
    pub(super) prompts: usize,
    pub(super) mean_kl: f64,
    pub(super) p95_kl: f64,
    pub(super) max_kl: f64,
    pub(super) mean_delta_cross_entropy_bits: f64,
    pub(super) top1_agreement: f64,
    pub(super) top5_contains_baseline_top1: f64,
    pub(super) mean_baseline_top1_prob: f64,
    pub(super) mean_exception_prob_of_baseline_top1: f64,
    pub(super) mean_baseline_top1_margin: f64,
    pub(super) per_prompt: Vec<OraclePqExceptionPromptReport>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct OraclePqExceptionPromptReport {
    pub(super) id: String,
    pub(super) stratum: String,
    pub(super) kl: f64,
    pub(super) delta_cross_entropy_bits: f64,
    pub(super) baseline_top1: u32,
    pub(super) exception_top1: u32,
    pub(super) top1_agree: bool,
    pub(super) baseline_top1_in_exception_top5: bool,
    pub(super) baseline_top1_prob: f64,
    pub(super) baseline_top2: u32,
    pub(super) baseline_top2_prob: f64,
    pub(super) baseline_top1_margin: f64,
    pub(super) exception_top1_prob: f64,
    pub(super) exception_prob_of_baseline_top1: f64,
    pub(super) exception_top1_margin: f64,
}
