use std::collections::{BTreeMap, HashMap};

use super::metrics::{bool_rate, mean, percentile};
use super::reports::{
    AddressCorruptionReport, AddressGroupImportanceReport, AddressProbePromptReport,
    AddressProbeReport, AddressProbeStratumReport, CodeStabilityReport, OraclePqPointReport,
    OraclePqPromptReport,
};
use super::types::PqConfig;

#[derive(Debug)]
pub(super) struct OraclePqPointAccumulator {
    prompts: Vec<OraclePqPromptReport>,
    address_probe_accumulators: HashMap<String, AddressProbeAccumulator>,
    address_corruption_accumulators: HashMap<usize, AddressProbeAccumulator>,
    address_group_importance_accumulators: HashMap<usize, AddressProbeAccumulator>,
}

impl OraclePqPointAccumulator {
    pub(super) fn new() -> Self {
        Self {
            prompts: Vec::new(),
            address_probe_accumulators: HashMap::new(),
            address_corruption_accumulators: HashMap::new(),
            address_group_importance_accumulators: HashMap::new(),
        }
    }

    pub(super) fn add(&mut self, prompt: OraclePqPromptReport) {
        self.prompts.push(prompt);
    }

    pub(super) fn add_address_probe(
        &mut self,
        name: &str,
        selected_group_keys: &[String],
        prompt: AddressProbePromptReport,
    ) {
        self.address_probe_accumulators
            .entry(name.to_string())
            .or_insert_with(|| AddressProbeAccumulator::new_with_keys(name, selected_group_keys))
            .add(prompt);
    }

    pub(super) fn add_address_corruption(
        &mut self,
        oracle_groups_kept: usize,
        prompt: AddressProbePromptReport,
    ) {
        self.address_corruption_accumulators
            .entry(oracle_groups_kept)
            .or_insert_with(|| {
                AddressProbeAccumulator::new(&format!("oracle_groups_kept_{oracle_groups_kept}"))
            })
            .add(prompt);
    }

    pub(super) fn add_address_group_importance(
        &mut self,
        replaced_group: usize,
        prompt: AddressProbePromptReport,
    ) {
        self.address_group_importance_accumulators
            .entry(replaced_group)
            .or_insert_with(|| {
                AddressProbeAccumulator::new(&format!("replaced_group_{replaced_group}"))
            })
            .add(prompt);
    }

    pub(super) fn finish(
        self,
        config: PqConfig,
        hidden_dim: usize,
        code_stability: Vec<CodeStabilityReport>,
    ) -> OraclePqPointReport {
        let kls: Vec<f64> = self.prompts.iter().map(|p| p.kl).collect();
        let levels = 1usize << config.bits_per_group;
        let mode_d_kls = self
            .prompts
            .iter()
            .filter_map(|p| p.mode_d_kl)
            .collect::<Vec<_>>();
        let coeff_mode_d_diffs = self
            .prompts
            .iter()
            .filter_map(|p| p.coeff_mode_d_max_abs_logit_diff)
            .collect::<Vec<_>>();
        OraclePqPointReport {
            k: config.k,
            groups: config.groups,
            bits_per_group: config.bits_per_group,
            oracle_address_bits: config.groups * config.bits_per_group,
            coefficient_codebook_bytes_f32: config.groups
                * levels
                * (config.k / config.groups)
                * std::mem::size_of::<f32>(),
            mode_d_residual_table_bytes_bf16: config.groups * levels * hidden_dim * 2,
            prompts: self.prompts.len(),
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            mean_delta_cross_entropy_bits: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.delta_cross_entropy_bits)
                    .collect::<Vec<_>>(),
            ),
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts.iter().map(|p| p.baseline_top1_in_pq_top5),
            ),
            mean_baseline_top1_prob: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.baseline_top1_prob)
                    .collect::<Vec<_>>(),
            ),
            mean_pq_prob_of_baseline_top1: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.pq_prob_of_baseline_top1)
                    .collect::<Vec<_>>(),
            ),
            mean_baseline_top1_margin: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.baseline_top1_margin)
                    .collect::<Vec<_>>(),
            ),
            mode_d_mean_kl: if mode_d_kls.is_empty() {
                None
            } else {
                Some(mean(&mode_d_kls))
            },
            mode_d_p95_kl: if mode_d_kls.is_empty() {
                None
            } else {
                Some(percentile(mode_d_kls.clone(), 0.95))
            },
            mode_d_max_kl: if mode_d_kls.is_empty() {
                None
            } else {
                Some(mode_d_kls.iter().copied().fold(0.0, f64::max))
            },
            mode_d_top1_agreement: if mode_d_kls.is_empty() {
                None
            } else {
                Some(bool_rate(
                    self.prompts.iter().filter_map(|p| p.mode_d_top1_agree),
                ))
            },
            mode_d_top5_contains_baseline_top1: if mode_d_kls.is_empty() {
                None
            } else {
                Some(bool_rate(
                    self.prompts
                        .iter()
                        .filter_map(|p| p.baseline_top1_in_mode_d_top5),
                ))
            },
            coeff_mode_d_max_abs_logit_diff: if coeff_mode_d_diffs.is_empty() {
                None
            } else {
                Some(coeff_mode_d_diffs.iter().copied().fold(0.0, f64::max))
            },
            address_probes: self
                .address_probe_accumulators
                .into_values()
                .map(|acc| acc.finish())
                .collect(),
            address_corruption_sweep: self
                .address_corruption_accumulators
                .into_iter()
                .map(|(oracle_groups_kept, acc)| acc.finish_corruption(oracle_groups_kept))
                .collect(),
            address_group_importance: self
                .address_group_importance_accumulators
                .into_iter()
                .map(|(replaced_group, acc)| acc.finish_group_importance(replaced_group))
                .collect(),
            code_stability,
            mean_pre_wo_l2: mean(&self.prompts.iter().map(|p| p.pre_wo_l2).collect::<Vec<_>>()),
            mean_wo_visible_l2: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.wo_visible_l2)
                    .collect::<Vec<_>>(),
            ),
            per_prompt: self.prompts,
        }
    }
}

fn address_probe_by_stratum(
    prompts: &[AddressProbePromptReport],
) -> Vec<AddressProbeStratumReport> {
    let mut by_stratum: BTreeMap<String, Vec<&AddressProbePromptReport>> = BTreeMap::new();
    for prompt in prompts {
        by_stratum
            .entry(prompt.stratum.clone())
            .or_default()
            .push(prompt);
    }

    by_stratum
        .into_iter()
        .map(|(stratum, prompts)| {
            let kls = prompts.iter().map(|prompt| prompt.kl).collect::<Vec<_>>();
            let positions = prompts.iter().map(|prompt| prompt.positions).sum::<usize>();
            let groups_total = prompts
                .iter()
                .map(|prompt| prompt.groups_total)
                .sum::<usize>()
                .max(1);
            let groups_correct = prompts
                .iter()
                .map(|prompt| prompt.groups_correct)
                .sum::<usize>();
            AddressProbeStratumReport {
                stratum,
                prompts: prompts.len(),
                positions,
                group_accuracy: groups_correct as f64 / groups_total as f64,
                mean_kl: mean(&kls),
                p95_kl: percentile(kls.clone(), 0.95),
                max_kl: kls.iter().copied().fold(0.0, f64::max),
                top1_agreement: bool_rate(prompts.iter().map(|prompt| prompt.top1_agree)),
                top5_contains_baseline_top1: bool_rate(
                    prompts
                        .iter()
                        .map(|prompt| prompt.baseline_top1_in_predicted_top5),
                ),
            }
        })
        .collect()
}

#[derive(Debug)]
struct AddressProbeAccumulator {
    name: String,
    selected_group_keys: Vec<String>,
    prompts: Vec<AddressProbePromptReport>,
}

impl AddressProbeAccumulator {
    fn new(name: &str) -> Self {
        Self::new_with_keys(name, &[])
    }

    fn new_with_keys(name: &str, selected_group_keys: &[String]) -> Self {
        Self {
            name: name.to_string(),
            selected_group_keys: selected_group_keys.to_vec(),
            prompts: Vec::new(),
        }
    }

    fn add(&mut self, prompt: AddressProbePromptReport) {
        self.prompts.push(prompt);
    }

    fn finish(mut self) -> AddressProbeReport {
        let kls = self.prompts.iter().map(|p| p.kl).collect::<Vec<_>>();
        let positions = self.prompts.iter().map(|p| p.positions).sum::<usize>();
        let total_groups = self
            .prompts
            .iter()
            .map(|p| p.groups_total)
            .sum::<usize>()
            .max(1);
        let correct_groups = self.prompts.iter().map(|p| p.groups_correct).sum::<usize>();
        self.prompts
            .sort_by(|a, b| b.kl.partial_cmp(&a.kl).unwrap_or(std::cmp::Ordering::Equal));
        AddressProbeReport {
            name: self.name,
            selected_group_keys: self.selected_group_keys,
            prompts: self.prompts.len(),
            positions,
            group_accuracy: correct_groups as f64 / total_groups as f64,
            exact_address_accuracy: bool_rate(self.prompts.iter().map(|p| p.exact_address_match)),
            mean_groups_correct_per_sequence: mean(
                &self
                    .prompts
                    .iter()
                    .map(|p| p.groups_correct as f64)
                    .collect::<Vec<_>>(),
            ),
            mean_groups_correct_per_position: correct_groups as f64 / positions.max(1) as f64,
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts
                    .iter()
                    .map(|p| p.baseline_top1_in_predicted_top5),
            ),
            by_stratum: address_probe_by_stratum(&self.prompts),
            worst_examples: self.prompts.into_iter().take(8).collect(),
        }
    }

    fn finish_corruption(mut self, oracle_groups_kept: usize) -> AddressCorruptionReport {
        let kls = self.prompts.iter().map(|p| p.kl).collect::<Vec<_>>();
        let positions = self.prompts.iter().map(|p| p.positions).sum::<usize>();
        let total_groups = self
            .prompts
            .iter()
            .map(|p| p.groups_total)
            .sum::<usize>()
            .max(1);
        let correct_groups = self.prompts.iter().map(|p| p.groups_correct).sum::<usize>();
        self.prompts
            .sort_by(|a, b| b.kl.partial_cmp(&a.kl).unwrap_or(std::cmp::Ordering::Equal));
        AddressCorruptionReport {
            label: self.name,
            oracle_groups_kept,
            prompts: self.prompts.len(),
            positions,
            group_accuracy: correct_groups as f64 / total_groups as f64,
            exact_address_accuracy: bool_rate(self.prompts.iter().map(|p| p.exact_address_match)),
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts
                    .iter()
                    .map(|p| p.baseline_top1_in_predicted_top5),
            ),
            worst_examples: self.prompts.into_iter().take(8).collect(),
        }
    }

    fn finish_group_importance(mut self, replaced_group: usize) -> AddressGroupImportanceReport {
        let kls = self.prompts.iter().map(|p| p.kl).collect::<Vec<_>>();
        let positions = self.prompts.iter().map(|p| p.positions).sum::<usize>();
        let total_groups = self
            .prompts
            .iter()
            .map(|p| p.groups_total)
            .sum::<usize>()
            .max(1);
        let correct_groups = self.prompts.iter().map(|p| p.groups_correct).sum::<usize>();
        self.prompts
            .sort_by(|a, b| b.kl.partial_cmp(&a.kl).unwrap_or(std::cmp::Ordering::Equal));
        AddressGroupImportanceReport {
            replaced_group,
            prompts: self.prompts.len(),
            positions,
            group_accuracy: correct_groups as f64 / total_groups as f64,
            exact_address_accuracy: bool_rate(self.prompts.iter().map(|p| p.exact_address_match)),
            mean_kl: mean(&kls),
            p95_kl: percentile(kls.clone(), 0.95),
            max_kl: kls.iter().copied().fold(0.0, f64::max),
            top1_agreement: bool_rate(self.prompts.iter().map(|p| p.top1_agree)),
            top5_contains_baseline_top1: bool_rate(
                self.prompts
                    .iter()
                    .map(|p| p.baseline_top1_in_predicted_top5),
            ),
            worst_examples: self.prompts.into_iter().take(8).collect(),
        }
    }
}
