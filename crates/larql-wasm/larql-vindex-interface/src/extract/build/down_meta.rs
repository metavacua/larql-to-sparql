//! Stage 3 of the build pipeline — per-layer down-projection metadata
//! and cluster-direction collection.

use larql_models::{TopKEntry, WeightArray};

use crate::error::VindexError;
use crate::extract::build_helpers::{
    build_whole_word_vocab, compute_gate_top_tokens, compute_offset_direction,
};
use crate::extract::constants::{FEATURE_PROJECTION_BATCH, FIRST_CONTENT_TOKEN_ID};
use crate::extract::stage_labels::*;

use super::{knowledge_layer_range, BuildContext};

impl<'a> BuildContext<'a> {
    /// Stage 3 — per-layer down-projection metadata + cluster collection.
    ///
    /// For each layer, project `embed @ w_down` to get vocab logits per
    /// feature, take top-k as `FeatureMeta`. Knowledge layers (L14–28)
    /// also collect `(input_token, output_token, offset_direction)` for
    /// the relation clustering stage.
    pub(super) fn write_down_meta_and_clusters(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_DOWN_META);

        let mut all_down_meta: Vec<Option<Vec<Option<crate::FeatureMeta>>>> =
            vec![None; self.num_layers];

        let knowledge_layers = knowledge_layer_range(self.weights.arch.family(), self.num_layers);

        // Build whole-word vocab once, shared across layers
        let (ww_ids_shared, ww_embed_shared) = build_whole_word_vocab(
            self.tokenizer,
            &self.weights.embed,
            self.vocab_size,
            self.hidden_size,
        );

        for (layer, layer_down_meta) in all_down_meta.iter_mut().enumerate().take(self.num_layers) {
            self.callbacks
                .on_layer_start(COMP_DOWN, layer, self.num_layers);
            let start = std::time::Instant::now();

            // Collect all down matrices for this layer (dense: 1, MoE: num_experts)
            let down_matrices: Vec<(&WeightArray, usize)> = if self.is_moe && self.n_experts > 0 {
                let mut mats = Vec::new();
                for expert in 0..self.n_experts {
                    if let Some(key) = self.weights.arch.expert_ffn_down_key(layer, expert) {
                        if let Some(w) = self.weights.tensors.get(&key) {
                            mats.push((w, expert));
                        }
                    }
                }
                if let Some(key) = self.weights.arch.shared_expert_down_key(layer) {
                    if let Some(w) = self.weights.tensors.get(&key) {
                        mats.push((w, self.n_experts));
                    }
                }
                mats
            } else {
                let down_key = self.weights.arch.ffn_down_key(layer);
                match self.weights.tensors.get(&down_key) {
                    Some(w) => vec![(w, 0)],
                    None => {
                        self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                        continue;
                    }
                }
            };

            if down_matrices.is_empty() {
                self.callbacks.on_layer_done(COMP_DOWN, layer, 0.0);
                continue;
            }

            let total_features_this_layer: usize =
                down_matrices.iter().map(|(w, _)| w.shape()[1]).sum();
            let is_knowledge_layer = knowledge_layers
                .map(|(start, end)| layer >= start && layer < end)
                .unwrap_or(false);

            // Dense models: pre-compute gate top tokens for clustering.
            // (MoE: skip — too many features.)
            let gate_top_tokens: Vec<String> = if is_knowledge_layer && !self.is_moe {
                let num_features = down_matrices[0].0.shape()[1];
                compute_gate_top_tokens(
                    self.weights,
                    self.tokenizer,
                    layer,
                    num_features,
                    &ww_ids_shared,
                    &ww_embed_shared,
                )
            } else {
                vec![]
            };

            let mut feature_offset = 0usize;
            for (w_down, _expert_id) in &down_matrices {
                let num_features = w_down.shape()[1];
                let batch_size = FEATURE_PROJECTION_BATCH;

                for batch_start in (0..num_features).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(num_features);
                    self.callbacks.on_feature_progress(
                        "down",
                        layer,
                        feature_offset + batch_start,
                        total_features_this_layer,
                    );

                    let w_chunk = w_down
                        .slice(ndarray::s![.., batch_start..batch_end])
                        .to_owned();
                    let cpu = larql_compute::CpuBackend;
                    use larql_compute::MatMul;
                    let chunk_logits = cpu.matmul(self.weights.embed.view(), w_chunk.view());

                    for feat in batch_start..batch_end {
                        let col = chunk_logits.column(feat - batch_start);
                        let mut scores: Vec<(usize, f32)> =
                            col.iter().copied().enumerate().collect();

                        let k = self.down_top_k.min(scores.len());
                        if k > 0 && k < scores.len() {
                            scores.select_nth_unstable_by(k, |a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        }
                        scores.truncate(k);
                        scores.sort_unstable_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });

                        let top_k_entries: Vec<TopKEntry> = scores
                            .into_iter()
                            .filter_map(|(idx, logit)| {
                                self.tokenizer
                                    .decode(&[idx as u32], true)
                                    .ok()
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty())
                                    .map(|token| TopKEntry {
                                        token,
                                        token_id: idx as u32,
                                        logit,
                                    })
                            })
                            .collect();

                        let (top_token, top_token_id, c_score) =
                            if let Some(first) = top_k_entries.first() {
                                (first.token.clone(), first.token_id, first.logit)
                            } else {
                                (String::new(), 0, 0.0)
                            };

                        // Collect gate→down offset direction for relation clustering.
                        // The offset = normalize(target_embed - input_embed) captures
                        // the RELATION between what activates the feature (entity)
                        // and what it outputs (target). France→Paris and
                        // Germany→Berlin share the same offset = "capital-of".
                        if is_knowledge_layer
                            && (top_token_id as usize) >= FIRST_CONTENT_TOKEN_ID
                            && !gate_top_tokens.is_empty()
                        {
                            let gate_tok = &gate_top_tokens[feat];
                            if let Some(offset) = compute_offset_direction(
                                gate_tok,
                                top_token_id as usize,
                                self.weights,
                                self.tokenizer,
                                self.hidden_size,
                                self.vocab_size,
                            ) {
                                self.cluster_directions.extend_from_slice(&offset);
                                self.cluster_features.push((layer, feat));
                                let all_tokens: Vec<String> =
                                    top_k_entries.iter().map(|e| e.token.clone()).collect();
                                self.cluster_top_tokens.push(all_tokens.join("|"));
                                self.cluster_input_tokens.push(gate_tok.clone());
                                self.cluster_output_tokens.push(top_token.clone());
                            }
                        }

                        let feat_idx = feature_offset + feat;
                        if layer_down_meta.is_none() {
                            *layer_down_meta = Some(Vec::new());
                        }
                        if let Some(ref mut metas) = layer_down_meta {
                            while metas.len() <= feat_idx {
                                metas.push(None);
                            }
                            metas[feat_idx] = Some(crate::FeatureMeta {
                                top_token,
                                top_token_id,
                                c_score,
                                top_k: top_k_entries,
                            });
                        }
                    }
                }

                feature_offset += num_features;
            }

            self.callbacks
                .on_layer_done(COMP_DOWN, layer, start.elapsed().as_secs_f64() * 1000.0);
        }

        crate::format::down_meta::write_binary(self.output_dir, &all_down_meta, self.down_top_k)?;
        self.callbacks.on_stage_done(STAGE_DOWN_META, 0.0);
        Ok(())
    }
}
