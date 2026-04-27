use ndarray::Array2;

use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use super::{LayerGraph, LayerOutput};

// ── Template detection ──

/// Known template patterns for routing.
#[derive(Clone, Debug)]
pub struct TemplatePattern {
    pub name: String,
    /// Token prefix that identifies this template (before the entity slot).
    pub prefix_tokens: Vec<u32>,
    /// Layer range for cached regime.
    pub cached_layers: std::ops::RangeInclusive<usize>,
}

/// Detect which template a token sequence matches, if any.
/// Matches by longest prefix overlap.
pub fn detect_template(token_ids: &[u32], templates: &[TemplatePattern]) -> Option<usize> {
    let mut best = None;
    let mut best_len = 0;

    for (i, tmpl) in templates.iter().enumerate() {
        let prefix = &tmpl.prefix_tokens;
        if prefix.len() > token_ids.len() { continue; }
        // Check if tokens start with this prefix (skipping BOS if present)
        let offset = if token_ids.len() > prefix.len() && token_ids[0] != prefix[0] { 1 } else { 0 };
        if offset + prefix.len() > token_ids.len() { continue; }
        let matches = prefix.iter().zip(&token_ids[offset..]).all(|(a, b)| a == b);
        if matches && prefix.len() > best_len {
            best = Some(i);
            best_len = prefix.len();
        }
    }
    best
}

// ── Template-guided walk: score only features in the template's universe ──

/// Per-template per-layer feature universe: the set of features that ever
/// fire for this template across diverse entities.
///
/// Built by running forward passes for a template with many entities,
/// capturing which features activate at each layer, and taking the union.
pub struct TemplateUniverse {
    pub name: String,
    /// layer → sorted vec of feature indices that fire for this template.
    pub features: std::collections::HashMap<usize, Vec<usize>>,
}

impl TemplateUniverse {
    /// Build by running dense forward passes for a template with multiple entities.
    /// `template`: format string with `{}` for entity slot.
    /// `entities`: list of entities to test.
    /// `activation_threshold`: minimum |activation| to count a feature as firing.
    pub fn build(
        weights: &ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        name: &str,
        template: &str,
        entities: &[&str],
        ffn: &dyn FfnBackend,
        activation_threshold: f32,
    ) -> Self {
        let all_layers: Vec<usize> = (0..weights.num_layers).collect();
        let mut layer_features: std::collections::HashMap<usize, std::collections::HashSet<usize>> =
            std::collections::HashMap::new();

        for entity in entities {
            let prompt = template.replace("{}", entity);
            let encoding = match tokenizer.encode(prompt.as_str(), true) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let trace = crate::forward::trace_forward_full(
                weights, &token_ids, &all_layers,
                true, 500, false, ffn,
            );

            for (layer, acts) in &trace.activations {
                let set = layer_features.entry(*layer).or_default();
                for (feat, act) in acts {
                    if act.abs() > activation_threshold {
                        set.insert(*feat);
                    }
                }
            }
        }

        let features = layer_features.into_iter()
            .map(|(layer, set)| {
                let mut v: Vec<usize> = set.into_iter().collect();
                v.sort_unstable();
                (layer, v)
            })
            .collect();

        Self { name: name.to_string(), features }
    }

    /// Get the feature universe for a layer.
    pub fn get(&self, layer: usize) -> Option<&[usize]> {
        self.features.get(&layer).map(|v| v.as_slice())
    }

    /// Total features across all layers.
    pub fn total_features(&self) -> usize {
        self.features.values().map(|v| v.len()).sum()
    }

    /// Print a summary.
    pub fn summary(&self) {
        let mut layers: Vec<usize> = self.features.keys().copied().collect();
        layers.sort();
        for &layer in &layers {
            let n = self.features[&layer].len();
            if n > 0 {
                print!("L{layer}:{n} ");
            }
        }
        println!();
    }
}

/// Guided walk layer graph: dense attention + walk FFN restricted to
/// the template's per-layer feature universe.
///
/// Instead of scoring all 10,240 features, scores only the ~100-400
/// that the template ever activates. Per-feature dot products + accumulations.
pub struct GuidedWalkLayerGraph<'a> {
    pub weights: &'a ModelWeights,
    pub universe: &'a TemplateUniverse,
    pub index: &'a dyn larql_vindex::GateIndex,
}

impl<'a> LayerGraph for GuidedWalkLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        // Attention: dense matmul
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block(weights, h, layer, false)?;

        // FFN: guided walk — score only template universe features
        let residual = guided_walk_ffn(weights, &h_post_attn, layer, self.universe, self.index);

        Some(LayerOutput { residual, activation: None, attention: None })
    }

    fn name(&self) -> &str { "guided-walk" }
}

/// Guided walk FFN: pre-FFN norm → gate scores for universe → GEGLU → accumulate.
///
/// Gate: scores all features (one gate_scores_batch call), but only processes
/// the template universe features for up/down. The gate call is the same cost
/// as dense, but up/down computation drops from 10,240 to ~100-400 features.
/// Up/down: per-feature dot products and scaled adds (no matmul).
fn guided_walk_ffn(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    universe: &TemplateUniverse,
    index: &dyn larql_vindex::GateIndex,
) -> Array2<f32> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let hidden = h_post_attn.shape()[1];
    let seq_len = h_post_attn.shape()[0];

    // Pre-FFN norm
    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = match pre_ffn_key {
        Some(key) => crate::forward::apply_norm(weights, h_post_attn, &key, norm_offset),
        None => crate::residual::rms_norm(h_post_attn, None, norm_offset),
    };

    // Get template universe for this layer
    let features = match universe.get(layer) {
        Some(f) if !f.is_empty() => f,
        _ => return h_post_attn.clone(),
    };

    let up_view = match index.up_layer_matrix(layer) {
        Some(v) => v,
        None => return h_post_attn.clone(),
    };
    let down_view = match index.down_layer_matrix(layer) {
        Some(v) => v,
        None => return h_post_attn.clone(),
    };

    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    // Gate scores: one batch call, then index into universe features only.
    // This is still a matmul for gate, but up/down are per-feature only.
    let gate_scores = match index.gate_scores_batch(layer, &h_ffn) {
        Some(gs) => gs,
        None => return h_post_attn.clone(),
    };

    let mut ffn_out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = h_ffn.row(s);
        let mut out_row = ffn_out.row_mut(s);

        for &feat in features {
            let gate_score = gate_scores[[s, feat]];

            let act = if is_gated {
                let up_score = up_view.row(feat).dot(&x_row);
                let activated_gate = if use_gelu {
                    crate::ffn::gelu_tanh(gate_score)
                } else {
                    gate_score * crate::ffn::sigmoid(gate_score)
                };
                activated_gate * up_score
            } else {
                let v = gate_score;
                if use_gelu { crate::ffn::gelu_tanh(v) } else { v * crate::ffn::sigmoid(v) }
            };

            if act.abs() > 1e-10 {
                let down_row = down_view.row(feat);
                out_row.scaled_add(act, &down_row);
            }
        }
    }

    // Post-FFN norm + residual
    let res_mult = arch.residual_multiplier();
    if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => crate::forward::apply_norm(weights, &ffn_out, &key, norm_offset),
            None => crate::residual::rms_norm(&ffn_out, None, norm_offset),
        };
        if res_mult != 1.0 {
            h_post_attn + &(&normed * res_mult)
        } else {
            h_post_attn + &normed
        }
    } else if res_mult != 1.0 {
        h_post_attn + &(&ffn_out * res_mult)
    } else {
        h_post_attn + &ffn_out
    }
}
