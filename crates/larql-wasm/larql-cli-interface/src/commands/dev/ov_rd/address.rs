use std::collections::HashMap;

use ndarray::{Array2, ArrayView1};

#[derive(Debug, Clone)]
pub(super) struct AddressProbeModel {
    pub(super) name: String,
    pub(super) group_majority: Vec<usize>,
    pub(super) group_maps: Vec<HashMap<String, usize>>,
    pub(super) group_train_accuracy: Vec<f64>,
    pub(super) selected_group_keys: Vec<String>,
}

impl AddressProbeModel {
    pub(super) fn predict_codes(
        &self,
        token_ids: &[u32],
        stratum: &str,
        position: usize,
    ) -> Vec<usize> {
        let key = address_feature_key(&self.name, token_ids, stratum, position);
        self.group_maps
            .iter()
            .enumerate()
            .map(|(group, map)| {
                map.get(&key)
                    .copied()
                    .unwrap_or_else(|| self.group_majority[group])
            })
            .collect()
    }

    pub(super) fn predict_codes_from_key(&self, key: &str) -> Vec<usize> {
        self.group_maps
            .iter()
            .enumerate()
            .map(|(group, map)| {
                map.get(key)
                    .copied()
                    .unwrap_or_else(|| self.group_majority[group])
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub(super) struct AddressLshGroupModel {
    pub(super) groups: Vec<usize>,
    pub(super) bits: usize,
    pub(super) group_majority: Vec<usize>,
    pub(super) group_maps: Vec<HashMap<usize, usize>>,
    pub(super) group_seeds: Vec<u64>,
    pub(super) group_train_accuracy: Vec<f64>,
}

impl AddressLshGroupModel {
    pub(super) fn selected_group_keys(&self) -> Vec<String> {
        (0..self.group_majority.len())
            .map(|group| {
                if self.groups.contains(&group) {
                    format!(
                        "lsh{}bits_seed{}_train_acc_{:.3}",
                        self.bits, self.group_seeds[group], self.group_train_accuracy[group]
                    )
                } else {
                    "majority".to_string()
                }
            })
            .collect()
    }

    pub(super) fn predict_selected_groups(
        &self,
        layer_input: &Array2<f32>,
        position: usize,
        base_codes: &[usize],
    ) -> Vec<usize> {
        let mut codes = base_codes.to_vec();
        let row = layer_input.row(position);
        for &group in &self.groups {
            let bucket = lsh_bucket(row, self.group_seeds[group], self.bits);
            codes[group] = self.group_maps[group]
                .get(&bucket)
                .copied()
                .unwrap_or(self.group_majority[group]);
        }
        codes
    }
}

#[derive(Debug, Clone)]
pub(super) struct BinaryHyperplane {
    pub(super) weights: Vec<f32>,
    pub(super) bias: f32,
}

impl BinaryHyperplane {
    fn predict_bit(&self, row: ArrayView1<'_, f32>) -> bool {
        normalized_hyperplane_logit(row, &self.weights, self.bias) >= 0.0
    }
}

#[derive(Debug, Clone)]
pub(super) struct AddressSupervisedGroupModel {
    pub(super) groups: Vec<usize>,
    pub(super) bits_per_group: usize,
    pub(super) epochs: usize,
    pub(super) lr: f32,
    pub(super) l2: f32,
    pub(super) group_majority: Vec<usize>,
    pub(super) group_hyperplanes: Vec<Vec<BinaryHyperplane>>,
    pub(super) group_train_accuracy: Vec<f64>,
}

impl AddressSupervisedGroupModel {
    pub(super) fn selected_group_keys(&self) -> Vec<String> {
        (0..self.group_majority.len())
            .map(|group| {
                if self.groups.contains(&group) {
                    format!(
                        "supervised{}bit_train_acc_{:.3}_epochs{}_lr{:.3}_l2_{:.1e}",
                        self.bits_per_group,
                        self.group_train_accuracy[group],
                        self.epochs,
                        self.lr,
                        self.l2
                    )
                } else {
                    "majority".to_string()
                }
            })
            .collect()
    }

    pub(super) fn predict_selected_groups(
        &self,
        layer_input: &Array2<f32>,
        position: usize,
        base_codes: &[usize],
    ) -> Vec<usize> {
        let mut codes = base_codes.to_vec();
        let row = layer_input.row(position);
        for &group in &self.groups {
            let mut code = 0usize;
            for (bit, hyperplane) in self.group_hyperplanes[group].iter().enumerate() {
                if hyperplane.predict_bit(row) {
                    code |= 1usize << bit;
                }
            }
            codes[group] = code;
        }
        codes
    }
}

#[derive(Debug, Clone)]
pub(super) struct AddressAttentionClusterGroupModel {
    pub(super) name: String,
    pub(super) groups: Vec<usize>,
    pub(super) qk_rank: Option<usize>,
    pub(super) centroids: Vec<Vec<f64>>,
    pub(super) group_majority: Vec<usize>,
    pub(super) group_maps: Vec<HashMap<String, usize>>,
    pub(super) selected_group_keys: Vec<String>,
}

impl AddressAttentionClusterGroupModel {
    pub(super) fn predict_selected_groups(
        &self,
        token_ids: &[u32],
        stratum: &str,
        position: usize,
        attention_weights: &[f32],
        base_codes: &[usize],
    ) -> Vec<usize> {
        let features = attention_pattern_features(attention_weights, position);
        let cluster = nearest_attention_cluster(&features, &self.centroids);
        let key = attention_cluster_key(&self.name, token_ids, stratum, position, cluster);
        let mut codes = base_codes.to_vec();
        for &group in &self.groups {
            codes[group] = self.group_maps[group]
                .get(&key)
                .copied()
                .unwrap_or(self.group_majority[group]);
        }
        codes
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct AddressMatchSummary {
    pub(super) groups_correct: usize,
    pub(super) groups_total: usize,
    pub(super) exact_address_match: bool,
}

pub(super) fn address_probe_names() -> Vec<&'static str> {
    vec![
        "position",
        "stratum",
        "position_stratum",
        "token_id",
        "prev_token_id",
        "token_bigram",
        "position_stratum_token",
    ]
}

pub(super) fn prev_ffn_feature_probe_names() -> Vec<&'static str> {
    vec![
        "prev_ffn_top1",
        "prev_ffn_top2_hash",
        "prev_ffn_top4_hash",
        "prev_ffn_top8_hash",
        "prev_ffn_top16_hash",
        "stratum_prev_ffn_top1",
        "stratum_prev_ffn_top8_hash",
        "token_prev_ffn_top1",
        "token_prev_ffn_top8_hash",
        "position_prev_ffn_top1",
        "position_prev_ffn_top8_hash",
    ]
}

pub(super) fn ffn_first_feature_probe_names() -> Vec<&'static str> {
    vec![
        "ffn_first_top1",
        "ffn_first_top2_hash",
        "ffn_first_top4_hash",
        "ffn_first_top8_hash",
        "ffn_first_top16_hash",
        "stratum_ffn_first_top1",
        "stratum_ffn_first_top8_hash",
        "token_ffn_first_top1",
        "token_ffn_first_top8_hash",
        "position_ffn_first_top1",
        "position_ffn_first_top8_hash",
    ]
}

pub(super) fn attention_relation_probe_names() -> Vec<&'static str> {
    vec![
        "attn_argmax",
        "attn_top2_hash",
        "attn_top4_hash",
        "attn_entropy_bucket",
        "attn_bos_bucket",
        "attn_distance_bucket",
        "attn_relation_class",
        "stratum_attn_relation_class",
        "token_attn_relation_class",
        "position_attn_relation_class",
    ]
}

pub(super) fn attention_cluster_probe_names(cluster_count: usize) -> Vec<String> {
    vec![
        format!("attn_cluster_{cluster_count}"),
        format!("stratum_attn_cluster_{cluster_count}"),
        format!("position_attn_cluster_{cluster_count}"),
        format!("token_attn_cluster_{cluster_count}"),
    ]
}

pub(super) fn address_feature_key(
    name: &str,
    token_ids: &[u32],
    stratum: &str,
    position: usize,
) -> String {
    let token = token_ids.get(position).copied().unwrap_or(0);
    let prev = if position == 0 {
        u32::MAX
    } else {
        token_ids.get(position - 1).copied().unwrap_or(0)
    };
    match name {
        "position" => format!("p:{position}"),
        "stratum" => format!("s:{stratum}"),
        "position_stratum" => format!("p:{position}|s:{stratum}"),
        "token_id" => format!("t:{token}"),
        "prev_token_id" => format!("pt:{prev}"),
        "token_bigram" => format!("pt:{prev}|t:{token}"),
        "position_stratum_token" => format!("p:{position}|s:{stratum}|t:{token}"),
        _ => format!("p:{position}"),
    }
}

pub(super) fn attention_relation_key(
    name: &str,
    token_ids: &[u32],
    stratum: &str,
    position: usize,
    weights: &[f32],
) -> String {
    let token = token_ids.get(position).copied().unwrap_or(0);
    let argmax = attention_argmax(weights, position);
    let top2 = attention_topk_key(weights, position, 2);
    let top4 = attention_topk_key(weights, position, 4);
    let entropy = attention_entropy_bucket(weights, position);
    let bos = attention_bos_bucket(weights.first().copied().unwrap_or(0.0));
    let distance = attention_distance_bucket(argmax, position);
    let relation = attention_relation_class(argmax, position);
    match name {
        "attn_argmax" => format!("aa:{argmax}"),
        "attn_top2_hash" => format!("at2:{top2}"),
        "attn_top4_hash" => format!("at4:{top4}"),
        "attn_entropy_bucket" => format!("ae:{entropy}"),
        "attn_bos_bucket" => format!("ab:{bos}"),
        "attn_distance_bucket" => format!("ad:{distance}"),
        "attn_relation_class" => format!("ar:{relation}"),
        "stratum_attn_relation_class" => format!("s:{stratum}|ar:{relation}"),
        "token_attn_relation_class" => format!("t:{token}|ar:{relation}"),
        "position_attn_relation_class" => format!("p:{position}|ar:{relation}"),
        _ => format!("ar:{relation}"),
    }
}

pub(super) fn attention_cluster_key(
    name: &str,
    token_ids: &[u32],
    stratum: &str,
    position: usize,
    cluster: usize,
) -> String {
    let token = token_ids.get(position).copied().unwrap_or(0);
    if name.contains("stratum_attn_cluster_") {
        format!("s:{stratum}|ac:{cluster}")
    } else if name.contains("position_attn_cluster_") {
        format!("p:{position}|ac:{cluster}")
    } else if name.contains("token_attn_cluster_") {
        format!("t:{token}|ac:{cluster}")
    } else {
        format!("ac:{cluster}")
    }
}

pub(super) fn prev_ffn_feature_key(
    name: &str,
    token_ids: &[u32],
    stratum: &str,
    position: usize,
    prev_features: &[usize],
) -> String {
    let token = token_ids.get(position).copied().unwrap_or(0);
    let top1 = prev_features
        .first()
        .map(|feature| feature.to_string())
        .unwrap_or_else(|| "none".to_string());
    let top2 = prev_features
        .iter()
        .take(2)
        .map(|feature| feature.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let top2 = if top2.is_empty() {
        "none".to_string()
    } else {
        top2
    };
    let top4 = feature_set_key(prev_features, 4);
    let top8 = feature_set_key(prev_features, 8);
    let top16 = feature_set_key(prev_features, 16);
    match name {
        "prev_ffn_top1" => format!("pf1:{top1}"),
        "prev_ffn_top2_hash" => format!("pf2:{top2}"),
        "prev_ffn_top4_hash" => format!("pf4:{top4}"),
        "prev_ffn_top8_hash" => format!("pf8:{top8}"),
        "prev_ffn_top16_hash" => format!("pf16:{top16}"),
        "stratum_prev_ffn_top1" => format!("s:{stratum}|pf1:{top1}"),
        "stratum_prev_ffn_top8_hash" => format!("s:{stratum}|pf8:{top8}"),
        "token_prev_ffn_top1" => format!("t:{token}|pf1:{top1}"),
        "token_prev_ffn_top8_hash" => format!("t:{token}|pf8:{top8}"),
        "position_prev_ffn_top1" => format!("p:{position}|pf1:{top1}"),
        "position_prev_ffn_top8_hash" => format!("p:{position}|pf8:{top8}"),
        _ => format!("pf1:{top1}"),
    }
}

pub(super) fn ffn_first_feature_key(
    name: &str,
    token_ids: &[u32],
    stratum: &str,
    position: usize,
    features: &[usize],
) -> String {
    let token = token_ids.get(position).copied().unwrap_or(0);
    let top1 = features
        .first()
        .map(|feature| feature.to_string())
        .unwrap_or_else(|| "none".to_string());
    let top2 = features
        .iter()
        .take(2)
        .map(|feature| feature.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let top2 = if top2.is_empty() {
        "none".to_string()
    } else {
        top2
    };
    let top4 = feature_set_key(features, 4);
    let top8 = feature_set_key(features, 8);
    let top16 = feature_set_key(features, 16);
    match name {
        "ffn_first_top1" => format!("ff1:{top1}"),
        "ffn_first_top2_hash" => format!("ff2:{top2}"),
        "ffn_first_top4_hash" => format!("ff4:{top4}"),
        "ffn_first_top8_hash" => format!("ff8:{top8}"),
        "ffn_first_top16_hash" => format!("ff16:{top16}"),
        "stratum_ffn_first_top1" => format!("s:{stratum}|ff1:{top1}"),
        "stratum_ffn_first_top8_hash" => format!("s:{stratum}|ff8:{top8}"),
        "token_ffn_first_top1" => format!("t:{token}|ff1:{top1}"),
        "token_ffn_first_top8_hash" => format!("t:{token}|ff8:{top8}"),
        "position_ffn_first_top1" => format!("p:{position}|ff1:{top1}"),
        "position_ffn_first_top8_hash" => format!("p:{position}|ff8:{top8}"),
        _ => format!("ff1:{top1}"),
    }
}

pub(super) fn attention_argmax(weights: &[f32], position: usize) -> usize {
    let causal_len = (position + 1).min(weights.len());
    weights
        .iter()
        .take(causal_len)
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn attention_topk_key(weights: &[f32], position: usize, k: usize) -> String {
    let causal_len = (position + 1).min(weights.len());
    let mut indexed = weights
        .iter()
        .take(causal_len)
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let key = indexed
        .into_iter()
        .take(k)
        .map(|(source, _)| source.to_string())
        .collect::<Vec<_>>()
        .join(",");
    if key.is_empty() {
        "none".to_string()
    } else {
        key
    }
}

pub(super) fn attention_entropy_bits(weights: &[f32], position: usize) -> f64 {
    let causal_len = (position + 1).min(weights.len());
    weights
        .iter()
        .take(causal_len)
        .copied()
        .filter(|&p| p > 0.0)
        .map(|p| {
            let p = p as f64;
            -p * p.log2()
        })
        .sum::<f64>()
}

fn attention_entropy_bucket(weights: &[f32], position: usize) -> usize {
    let entropy_bits = attention_entropy_bits(weights, position);
    ((entropy_bits * 2.0).floor() as usize).min(16)
}

fn attention_bos_bucket(mass: f32) -> &'static str {
    match mass {
        x if x < 0.01 => "lt001",
        x if x < 0.05 => "lt005",
        x if x < 0.10 => "lt010",
        x if x < 0.25 => "lt025",
        x if x < 0.50 => "lt050",
        _ => "ge050",
    }
}

fn attention_distance_bucket(argmax: usize, position: usize) -> &'static str {
    if argmax == 0 {
        "bos"
    } else if argmax == position {
        "self"
    } else if argmax + 1 == position {
        "prev"
    } else if argmax > position {
        "future"
    } else {
        match position - argmax {
            0 => "self",
            1 => "prev",
            2..=4 => "d2_4",
            5..=8 => "d5_8",
            9..=16 => "d9_16",
            _ => "far",
        }
    }
}

fn attention_relation_class(argmax: usize, position: usize) -> &'static str {
    if argmax == 0 {
        "bos"
    } else if argmax == position {
        "self"
    } else if argmax + 1 == position {
        "prev"
    } else if argmax > position {
        "future"
    } else {
        match position - argmax {
            0 => "self",
            1 => "prev",
            2..=4 => "local",
            5..=16 => "mid",
            _ => "far",
        }
    }
}

fn feature_set_key(prev_features: &[usize], k: usize) -> String {
    let key = prev_features
        .iter()
        .take(k)
        .map(|feature| feature.to_string())
        .collect::<Vec<_>>()
        .join(",");
    if key.is_empty() {
        "none".to_string()
    } else {
        key
    }
}

pub(super) fn top_feature_ids_from_activation_row(
    row: ArrayView1<'_, f32>,
    top_k: usize,
) -> Vec<usize> {
    let mut indexed = row.iter().copied().enumerate().collect::<Vec<_>>();
    indexed.sort_unstable_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indexed
        .into_iter()
        .take(top_k)
        .map(|(feature, _)| feature)
        .collect()
}

pub(super) fn attention_pattern_features(weights: &[f32], position: usize) -> Vec<f64> {
    let causal_len = (position + 1).min(weights.len());
    if causal_len == 0 {
        return vec![0.0; 35];
    }
    let denom = causal_len.max(1) as f64;
    let argmax = attention_argmax(weights, position);
    let max_mass = weights.get(argmax).copied().unwrap_or(0.0) as f64;
    let entropy_bits = weights
        .iter()
        .take(causal_len)
        .copied()
        .filter(|&p| p > 0.0)
        .map(|p| {
            let p = p as f64;
            -p * p.log2()
        })
        .sum::<f64>();
    let entropy_norm = if causal_len > 1 {
        entropy_bits / (causal_len as f64).log2()
    } else {
        0.0
    };

    let mut bos_mass = 0.0;
    let mut self_mass = 0.0;
    let mut prev_mass = 0.0;
    let mut local_mass = 0.0;
    let mut mid_mass = 0.0;
    let mut far_mass = 0.0;
    for (source, &mass) in weights.iter().take(causal_len).enumerate() {
        let mass = mass as f64;
        if source == 0 {
            bos_mass += mass;
        }
        if source == position {
            self_mass += mass;
        } else if source + 1 == position {
            prev_mass += mass;
        } else if source < position {
            let distance = position - source;
            if distance <= 4 {
                local_mass += mass;
            } else if distance <= 16 {
                mid_mass += mass;
            } else {
                far_mass += mass;
            }
        }
    }

    let argmax_source_norm = argmax as f64 / denom;
    let argmax_distance_norm = if argmax <= position {
        (position - argmax) as f64 / denom
    } else {
        0.0
    };

    let mut features = vec![
        bos_mass,
        self_mass,
        prev_mass,
        local_mass,
        mid_mass,
        far_mass,
        entropy_bits,
        entropy_norm,
        max_mass,
        argmax_source_norm,
        argmax_distance_norm,
    ];

    let mut indexed = weights
        .iter()
        .take(causal_len)
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for rank in 0..8 {
        if let Some((source, mass)) = indexed.get(rank).copied() {
            let source_norm = source as f64 / denom;
            let rel_distance = if source <= position {
                (position - source) as f64 / denom
            } else {
                0.0
            };
            features.push(mass as f64);
            features.push(source_norm);
            features.push(rel_distance);
        } else {
            features.push(0.0);
            features.push(0.0);
            features.push(0.0);
        }
    }

    features
}

pub(super) fn nearest_attention_cluster(features: &[f64], centroids: &[Vec<f64>]) -> usize {
    let mut best_idx = 0usize;
    let mut best_dist = f64::INFINITY;
    for (idx, centroid) in centroids.iter().enumerate() {
        let dist = features
            .iter()
            .zip(centroid.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum::<f64>();
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }
    best_idx
}

pub(super) fn lsh_bucket(row: ArrayView1<'_, f32>, seed: u64, bits: usize) -> usize {
    let mut bucket = 0usize;
    for bit in 0..bits {
        let mut sum = 0.0_f64;
        for (dim, &value) in row.iter().enumerate() {
            let hash = splitmix64(
                seed ^ ((bit as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    ^ ((dim as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
            );
            let sign = if hash & 1 == 0 { -1.0 } else { 1.0 };
            sum += value as f64 * sign;
        }
        if sum >= 0.0 {
            bucket |= 1usize << bit;
        }
    }
    bucket
}

pub(super) fn train_binary_hyperplane(
    rows: &[&[f32]],
    labels: &[bool],
    dim: usize,
    epochs: usize,
    lr: f32,
    l2: f32,
) -> BinaryHyperplane {
    let mut weights = vec![0.0_f32; dim];
    let positives = labels.iter().filter(|&&label| label).count();
    let negatives = labels.len().saturating_sub(positives);
    let mut bias = if positives == 0 {
        -4.0
    } else if negatives == 0 {
        4.0
    } else {
        ((positives as f32 + 0.5) / (negatives as f32 + 0.5)).ln()
    };

    for _ in 0..epochs {
        for (row, &label) in rows.iter().zip(labels.iter()) {
            let scale = normalized_row_scale_slice(row);
            let dot = row
                .iter()
                .zip(weights.iter())
                .map(|(&x, &w)| (x / scale) * w)
                .sum::<f32>();
            let logit = (bias + dot).clamp(-30.0, 30.0);
            let prob = 1.0 / (1.0 + (-logit).exp());
            let target = if label { 1.0 } else { 0.0 };
            let grad = prob - target;
            for (w, &x) in weights.iter_mut().zip(row.iter()) {
                *w -= lr * (grad * (x / scale) + l2 * *w);
            }
            bias -= lr * grad;
        }
    }

    BinaryHyperplane { weights, bias }
}

pub(super) fn predict_code_from_hyperplanes(
    row: &[f32],
    hyperplanes: &[BinaryHyperplane],
) -> usize {
    let scale = normalized_row_scale_slice(row);
    let mut code = 0usize;
    for (bit, hyperplane) in hyperplanes.iter().enumerate() {
        let dot = row
            .iter()
            .zip(hyperplane.weights.iter())
            .map(|(&x, &w)| (x / scale) * w)
            .sum::<f32>();
        if hyperplane.bias + dot >= 0.0 {
            code |= 1usize << bit;
        }
    }
    code
}

pub(super) fn address_match_report(
    oracle_codes_by_position: &[Vec<usize>],
    predicted_codes_by_position: &[Vec<usize>],
) -> AddressMatchSummary {
    let mut groups_correct = 0usize;
    let mut groups_total = 0usize;
    let mut exact_address_match = true;
    for (oracle, predicted) in oracle_codes_by_position
        .iter()
        .zip(predicted_codes_by_position.iter())
    {
        if oracle != predicted {
            exact_address_match = false;
        }
        for (&oracle_code, &predicted_code) in oracle.iter().zip(predicted.iter()) {
            groups_total += 1;
            if oracle_code == predicted_code {
                groups_correct += 1;
            }
        }
    }
    AddressMatchSummary {
        groups_correct,
        groups_total,
        exact_address_match,
    }
}

fn normalized_row_scale_slice(row: &[f32]) -> f32 {
    let mean_square = if row.is_empty() {
        0.0
    } else {
        row.iter()
            .map(|&value| (value as f64) * (value as f64))
            .sum::<f64>()
            / row.len() as f64
    };
    (mean_square.sqrt() as f32).max(1e-6)
}

fn normalized_row_scale_view(row: ArrayView1<'_, f32>) -> f32 {
    let mean_square = if row.is_empty() {
        0.0
    } else {
        row.iter()
            .map(|&value| (value as f64) * (value as f64))
            .sum::<f64>()
            / row.len() as f64
    };
    (mean_square.sqrt() as f32).max(1e-6)
}

fn normalized_hyperplane_logit(row: ArrayView1<'_, f32>, weights: &[f32], bias: f32) -> f32 {
    let scale = normalized_row_scale_view(row);
    let dot = row
        .iter()
        .zip(weights.iter())
        .map(|(&x, &w)| (x / scale) * w)
        .sum::<f32>();
    bias + dot
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
