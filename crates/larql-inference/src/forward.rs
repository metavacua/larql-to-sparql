//! Full transformer forward pass.
//!
//! Runs tokens through embedding → layers → final norm → logits.
//! Uses the ModelArchitecture trait for model-specific behavior
//! and FfnBackend trait for swappable FFN computation.

use ndarray::Array2;

use crate::attention::AttentionWeights;
use crate::ffn::{FfnBackend, LayerFfnRouter, WeightFfn};
use crate::model::ModelWeights;
use larql_models::NormType;
use crate::residual::rms_norm;

/// Per-head attention pattern for the last token at one layer.
pub struct LayerAttentionCapture {
    pub layer: usize,
    /// Per-head attention weights for the last token.
    /// `heads[h][j]` = how much the last token attends to position j.
    pub weights: AttentionWeights,
}

/// Result of a forward trace — residuals and optional sparse activations.
pub struct TraceResult {
    /// (layer, residual_vector) for each capture layer.
    pub residuals: Vec<(usize, Vec<f32>)>,
    /// (layer, top-K (feature_index, activation_magnitude)) for each capture layer.
    /// Only populated if capture_activations=true.
    pub activations: Vec<(usize, Vec<(usize, f32)>)>,
    /// Per-layer attention weight captures. Only populated if capture_attention=true.
    pub attention: Vec<LayerAttentionCapture>,
}

/// Prediction result from a full forward pass.
pub struct PredictResult {
    /// Top-k predicted tokens as (token_string, probability).
    pub predictions: Vec<(String, f64)>,
}

/// Prediction result with per-layer residual capture.
pub struct PredictResultWithResiduals {
    /// Top-k predicted tokens as (token_string, probability).
    pub predictions: Vec<(String, f64)>,
    /// Per-layer residual vectors (last token position only).
    /// Index i contains the residual BEFORE layer i's FFN (i.e., after attention).
    pub residuals: Vec<Vec<f32>>,
}

/// Per-layer computation strategy.
pub enum LayerMode<'a> {
    /// Run full attention + FFN with the given backend.
    Compute(&'a dyn FfnBackend),
    /// Skip the layer entirely — just multiply the hidden state by a scalar gain.
    /// gain = norm[L+1] / norm[L] from calibration data.
    ScalarGain(f32),
    /// Run attention but skip FFN (return zeros from FFN).
    /// Attention still routes information; FFN contribution is dropped.
    AttentionOnly,
}

/// Apply the appropriate norm (RMSNorm or LayerNorm) based on architecture.
pub fn apply_norm(
    weights: &ModelWeights,
    x: &Array2<f32>,
    weight_key: &str,
    norm_offset: f32,
) -> Array2<f32> {
    match weights.arch.norm_type() {
        NormType::LayerNorm => {
            let bias_key = weight_key.replace(".weight", ".bias");
            crate::residual::layer_norm(
                x,
                weights.vectors.get(weight_key),
                weights.vectors.get(&bias_key),
            )
        }
        _ => rms_norm(x, weights.vectors.get(weight_key), norm_offset),
    }
}

/// Compute x @ w.T via BLAS.
pub fn dot_proj(x: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>, w: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>) -> Array2<f32> {
    x.dot(&w.t())
}

/// Add a 1D bias vector to each row of a 2D matrix.
pub fn add_bias(x: &mut Array2<f32>, bias: &[f32]) {
    let cols = x.shape()[1];
    let n = cols.min(bias.len());
    for mut row in x.rows_mut() {
        for j in 0..n {
            row[j] += bias[j];
        }
    }
}

/// Embed token IDs with architecture-specific scaling.
fn embed_tokens(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    embed_tokens_pub(weights, token_ids)
}

/// Public embed for use by LayerGraph.
pub fn embed_tokens_pub(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let scale = weights.arch.embed_scale();

    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok_id) in token_ids.iter().enumerate() {
        let row = weights.embed.row(tok_id as usize);
        for j in 0..hidden {
            h[[i, j]] = row[j] * scale;
        }
    }
    h
}

/// Public wrapper for run_attention (used by CachedFfn calibration).
pub fn run_attention_public(weights: &ModelWeights, h: &Array2<f32>, layer: usize) -> Option<Array2<f32>> {
    run_attention(weights, h, layer)
}

/// Run attention for a single layer. Returns the post-attention residual.
fn run_attention(weights: &ModelWeights, h: &Array2<f32>, layer: usize) -> Option<Array2<f32>> {
    let (h_post_attn, _) = run_attention_inner(weights, h, layer, false)?;
    Some(h_post_attn)
}

/// Run attention with optional per-head weight capture.
/// Delegates to the shared `run_attention_block` in attention.rs.
fn run_attention_inner(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
) -> Option<(Array2<f32>, Option<AttentionWeights>)> {
    let (h_post_attn, _attn_projected, attn_weights) =
        crate::attention::run_attention_block(weights, h, layer, capture_attention)?;
    Some((h_post_attn, attn_weights))
}

/// Run FFN for a single layer using the given backend. Returns the post-FFN residual.
pub fn run_ffn(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
) -> (Array2<f32>, Option<Array2<f32>>) {
    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;

    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = match pre_ffn_key {
        Some(key) => apply_norm(weights, h_post_attn, &key, norm_offset),
        None => rms_norm(h_post_attn, None, norm_offset),
    };

    let (ffn_out, activation) = if capture_activation {
        let (out, act) = ffn.forward_with_activation(layer, &h_ffn);
        (out, Some(act))
    } else {
        (ffn.forward(layer, &h_ffn), None)
    };

    let res_mult = arch.residual_multiplier();
    let h_out = if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => apply_norm(weights, &ffn_out, &key, norm_offset),
            None => rms_norm(&ffn_out, None, norm_offset),
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
    };

    (h_out, activation)
}

/// Apply per-layer scalar multiplier if present (e.g., Gemma 4 layer_scalar).
fn apply_layer_scalar(weights: &ModelWeights, h: &mut Array2<f32>, layer: usize) {
    if let Some(key) = weights.arch.layer_scalar_key(layer) {
        if let Some(scalars) = weights.vectors.get(&key) {
            if let Some(&scalar) = scalars.first() {
                if scalar != 1.0 {
                    *h *= scalar;
                }
            }
        }
    }
}

/// Precompute per-layer input signals from token embeddings.
///
/// Combines two streams:
///   1. Model projection: main_embeds @ per_layer_model_projection.T * 1/sqrt(hidden)
///      → reshape to [seq, num_layers, ple_dim] → RMSNorm per layer
///   2. Per-layer token embed: embed_tokens_per_layer[token_ids] * sqrt(ple_dim)
///      → reshape to [seq, num_layers, ple_dim]
///   Combined: (stream1 + stream2) * 1/sqrt(2)
///
/// Returns a Vec of [seq, ple_dim] arrays, one per layer. Empty vec if PLE is not used.
fn precompute_per_layer_inputs(
    weights: &ModelWeights,
    main_embeds: &Array2<f32>,
    token_ids: &[u32],
) -> Vec<Array2<f32>> {
    let arch = &*weights.arch;
    if !arch.has_per_layer_embeddings() {
        return Vec::new();
    }

    let ple_dim = arch.per_layer_embed_dim();
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let total_ple_dim = num_layers * ple_dim;

    // Stream 1: model projection from main embeddings
    let w_model_proj = match weights.tensors.get("per_layer_model_projection.weight") {
        Some(w) => w,
        None => return Vec::new(),
    };
    // main_embeds @ w_model_proj.T → [seq, num_layers * ple_dim]
    let projected = dot_proj(main_embeds, w_model_proj);
    let model_proj_scale = (hidden as f32).powf(-0.5); // 1/sqrt(hidden)

    // Stream 2: per-layer token embeddings
    let ple_embed = weights.tensors.get("embed_tokens_per_layer.weight");
    let embed_scale = (ple_dim as f32).sqrt(); // sqrt(ple_dim)

    // Per-layer projection norm weight
    let proj_norm_w = weights.vectors.get("per_layer_projection_norm.weight");
    let norm_offset = arch.norm_weight_offset();

    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    // Build per-layer inputs
    let mut per_layer_inputs = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let col_start = layer * ple_dim;

        let mut layer_input = Array2::<f32>::zeros((seq_len, ple_dim));

        for s in 0..seq_len {
            for d in 0..ple_dim {
                // Stream 1: projected model embedding, scaled
                let mut val = projected[[s, col_start + d]] * model_proj_scale;

                // RMSNorm per vector (stream 1)
                // Deferred — apply after filling the row
                layer_input[[s, d]] = val;
            }

            // Apply RMSNorm to stream 1 for this position
            if let Some(norm_w) = proj_norm_w {
                let mut sq_sum = 0.0f32;
                for d in 0..ple_dim {
                    sq_sum += layer_input[[s, d]] * layer_input[[s, d]];
                }
                let rms = (sq_sum / ple_dim as f32 + 1e-6).sqrt();
                let inv_rms = 1.0 / rms;
                for d in 0..ple_dim {
                    layer_input[[s, d]] *= inv_rms * (norm_offset + norm_w[d]);
                }
            }

            // Add stream 2: per-layer token embedding
            if let Some(ref embed) = ple_embed {
                let tok = token_ids[s] as usize;
                let row = embed.row(tok);
                for d in 0..ple_dim {
                    layer_input[[s, d]] += row[col_start + d] * embed_scale;
                }
            }

            // Scale combined by 1/sqrt(2)
            for d in 0..ple_dim {
                layer_input[[s, d]] *= inv_sqrt2;
            }
        }

        per_layer_inputs.push(layer_input);
    }

    per_layer_inputs
}

/// Apply Per-Layer Embeddings (PLE) to the hidden state after attention+FFN.
///
/// Runs at the end of each decoder layer (after attention and FFN residual additions):
///   gate = gelu_tanh(h @ input_gate.T)   → [seq, ple_dim]
///   gated = gate * per_layer_input        → [seq, ple_dim]
///   contribution = gated @ projection.T   → [seq, hidden]
///   normed = RMSNorm(contribution)
///   h = h + normed
fn apply_per_layer_embedding(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    per_layer_input: Option<&Array2<f32>>,
) -> Array2<f32> {
    let arch = &*weights.arch;
    let per_layer_input = match per_layer_input {
        Some(p) => p,
        None => return h.clone(),
    };

    let gate_key = match arch.per_layer_input_gate_key(layer) {
        Some(k) => k,
        None => return h.clone(),
    };
    let proj_key = match arch.per_layer_projection_key(layer) {
        Some(k) => k,
        None => return h.clone(),
    };
    let w_gate = match weights.tensors.get(&gate_key) {
        Some(w) => w,
        None => return h.clone(),
    };
    let w_proj = match weights.tensors.get(&proj_key) {
        Some(w) => w,
        None => return h.clone(),
    };

    // gate = h @ w_gate.T → [seq, ple_dim]
    let mut gate = dot_proj(h, w_gate);

    // Apply gelu_tanh activation to gate
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    for val in gate.iter_mut() {
        // gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let x = *val;
        let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
        *val = 0.5 * x * (1.0 + inner.tanh());
    }

    // gated = gate * per_layer_input (element-wise)
    let gated = &gate * per_layer_input;

    // contribution = gated @ w_proj.T → [seq, hidden]
    let contribution = dot_proj(&gated, w_proj);

    // Apply post-PLE norm then residual add
    let norm_offset = arch.norm_weight_offset();
    let normed = match arch.post_per_layer_input_norm_key(layer) {
        Some(key) => apply_norm(weights, &contribution, &key, norm_offset),
        None => contribution,
    };

    h + &normed
}

/// Run a single transformer layer with the given FFN backend.
/// PLE input is the precomputed per-layer embedding signal (None if model doesn't use PLE).
fn run_layer_with_ffn(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    ple_input: Option<&Array2<f32>>,
) -> Option<(Array2<f32>, Option<Array2<f32>>)> {
    let h_post_attn = run_attention(weights, h, layer)?;
    let (h_post_ffn, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    // PLE runs after attention+FFN, before layer_scalar
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, activation))
}

/// Run a single transformer layer, optionally capturing attention weights.
fn run_layer_with_capture(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    capture_attention: bool,
    ple_input: Option<&Array2<f32>>,
) -> Option<(Array2<f32>, Option<Array2<f32>>, Option<AttentionWeights>)> {
    let (h_post_attn, attn_weights) = run_attention_inner(weights, h, layer, capture_attention)?;
    let (h_post_ffn, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, activation, attn_weights))
}

/// Project the final hidden state to logits and return top-k predictions.
pub fn logits_to_predictions_pub(
    weights: &ModelWeights,
    h: &Array2<f32>,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
) -> PredictResult {
    logits_to_predictions(weights, h, tokenizer, top_k)
}

fn logits_to_predictions(
    weights: &ModelWeights,
    h: &Array2<f32>,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
) -> PredictResult {
    let seq_len = h.shape()[0];
    let norm_offset = weights.arch.norm_weight_offset();

    let h_final = apply_norm(weights, h, weights.arch.final_norm_key(), norm_offset);

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();

    // Single BLAS gemv: (1, hidden) @ (vocab, hidden)^T → (1, vocab)
    // Replaces vocab_size individual dot products.
    let last_2d = h_final.slice(ndarray::s![seq_len - 1..seq_len, ..]);
    let logits_raw = dot_proj(&last_2d, &weights.lm_head);
    let inv_scale = 1.0 / logits_scale;
    let logits: Vec<f32> = logits_raw
        .row(0)
        .iter()
        .map(|&v| {
            let mut logit = v * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            logit
        })
        .collect();

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = logits
        .iter()
        .map(|l| ((l - max_logit) as f64).exp())
        .sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|l| (((l - max_logit) as f64).exp() / exp_sum) as f32)
        .collect();

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    let k = top_k.min(indexed.len());
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let predictions = indexed
        .into_iter()
        .filter_map(|(idx, prob)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob as f64))
        })
        .collect();

    PredictResult { predictions }
}

// ── Public API ──

/// Run a forward pass through layers 0..=stop_layer and return the full
/// hidden state matrix (seq_len, hidden_size) at that layer.
/// This is the complete residual stream — all positions, not just last token.
pub fn forward_to_layer(
    weights: &ModelWeights,
    token_ids: &[u32],
    stop_layer: usize,
) -> Array2<f32> {
    let ffn = WeightFfn { weights };
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..=stop_layer {
        h = match run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer)) {
            Some((h_new, _)) => h_new,
            None => continue,
        };
    }
    h
}

/// Run a forward pass through layers 0..=max_layer and return the
/// last-token residual at each requested capture layer.
pub fn capture_residuals(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    let trace = trace_forward(weights, token_ids, capture_layers, false, 0);
    trace.residuals
}

/// Run a forward pass and capture both residuals and sparse activations.
pub fn trace_forward(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
) -> TraceResult {
    let ffn = WeightFfn { weights };
    trace_forward_with_ffn(
        weights,
        token_ids,
        capture_layers,
        capture_activations,
        activation_top_k,
        &ffn,
    )
}

/// Run a forward pass with a custom FFN backend.
pub fn trace_forward_with_ffn(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
    ffn: &dyn FfnBackend,
) -> TraceResult {
    trace_forward_full(
        weights, token_ids, capture_layers, capture_activations,
        activation_top_k, false, ffn,
    )
}

/// Run a forward pass capturing residuals, activations, and optionally attention weights.
pub fn trace_forward_full(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
    capture_attention: bool,
    ffn: &dyn FfnBackend,
) -> TraceResult {
    let seq_len = token_ids.len();
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);

    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut results = Vec::new();
    let mut activations: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
    let mut attention_captures: Vec<LayerAttentionCapture> = Vec::new();

    for layer in 0..=max_layer {
        let is_capture_layer = capture_layers.contains(&layer);
        let need_activation = capture_activations && is_capture_layer;
        let need_attention = capture_attention && is_capture_layer;

        let (h_new, activation, attn_weights) =
            match run_layer_with_capture(weights, &h, layer, ffn, need_activation, need_attention, ple_inputs.get(layer)) {
                Some(result) => result,
                None => continue,
            };
        h = h_new;

        if is_capture_layer {
            let last_row = h.row(seq_len - 1);
            results.push((layer, last_row.to_vec()));

            if let Some(act) = activation {
                let act_row = act.row(seq_len - 1);
                let mut indexed: Vec<(usize, f32)> = act_row.iter().copied().enumerate().collect();
                indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                indexed.truncate(activation_top_k);
                activations.push((layer, indexed));
            }

            if let Some(weights) = attn_weights {
                attention_captures.push(LayerAttentionCapture {
                    layer,
                    weights,
                });
            }
        }
    }

    TraceResult {
        residuals: results,
        activations,
        attention: attention_captures,
    }
}

/// Run a full forward pass and return the top-k next token predictions.
/// Uses dense WeightFfn (ground truth).
pub fn predict(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
) -> PredictResult {
    let ffn = WeightFfn { weights };
    predict_with_ffn(weights, tokenizer, token_ids, top_k, &ffn)
}

/// Run a full forward pass with a custom FFN backend for all layers.
pub fn predict_with_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..num_layers {
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer)) {
            Some((h_new, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k)
}

/// Prediction result with per-layer attention captures and logit lens.
pub struct PredictResultWithAttention {
    pub predictions: Vec<(String, f64)>,
    pub attention: Vec<LayerAttentionCapture>,
    /// Per-layer residual vectors (last token position) for logit lens projection.
    pub residuals: Vec<(usize, Vec<f32>)>,
}

/// Run a full forward pass with a custom FFN backend, capturing attention weights
/// and per-layer residuals for logit lens.
pub fn predict_with_ffn_attention(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResultWithAttention {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut attention = Vec::with_capacity(num_layers);
    let mut residuals = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        match run_layer_with_capture(weights, &h, layer, ffn, false, true, ple_inputs.get(layer)) {
            Some((h_new, _, attn_weights)) => {
                h = h_new;
                // Capture last-token residual for logit lens
                residuals.push((layer, h.row(seq_len - 1).to_vec()));
                if let Some(w) = attn_weights {
                    attention.push(LayerAttentionCapture { layer, weights: w });
                }
            }
            None => continue,
        }
    }

    let result = logits_to_predictions(weights, &h, tokenizer, top_k);
    PredictResultWithAttention {
        predictions: result.predictions,
        attention,
        residuals,
    }
}

/// Project a single residual vector through final norm + lm_head to get top-1 prediction.
/// This is the "logit lens" — what would the model predict if it stopped at this layer?
pub fn logit_lens_top1(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    residual: &[f32],
) -> Option<(String, f64)> {
    let hidden = weights.hidden_size;
    if residual.len() != hidden { return None; }

    let h = Array2::from_shape_vec((1, hidden), residual.to_vec()).ok()?;
    let result = logits_to_predictions(weights, &h, tokenizer, 1);
    result.predictions.into_iter().next()
}

/// Forward pass with residual capture — returns predictions + per-layer residuals.
/// Captures the residual at the last token position after attention (before FFN)
/// at each layer. This is what gate_knn sees during inference.
pub fn predict_with_ffn_trace(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResultWithResiduals {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut residuals = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        // Capture the residual at the last token position BEFORE this layer
        let last_pos = h.shape()[0] - 1;
        residuals.push(h.row(last_pos).to_vec());

        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer)) {
            Some((h_new, _)) => h_new,
            None => continue,
        };
    }

    let result = logits_to_predictions(weights, &h, tokenizer, top_k);
    PredictResultWithResiduals {
        predictions: result.predictions,
        residuals,
    }
}

/// Run a full forward pass with per-layer FFN backend selection.
pub fn predict_with_router(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    router: &LayerFfnRouter,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..num_layers {
        let ffn = router.get(layer);
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer)) {
            Some((h_new, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k)
}

/// Run a forward pass with per-layer strategy: full compute or scalar gain bypass.
pub fn predict_with_strategy(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    strategy: &[LayerMode],
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..num_layers {
        match &strategy[layer] {
            LayerMode::Compute(ffn) => {
                h = match run_layer_with_ffn(weights, &h, layer, *ffn, false, ple_inputs.get(layer)) {
                    Some((h_new, _)) => h_new,
                    None => continue,
                };
            }
            LayerMode::ScalarGain(gain) => {
                h *= *gain;
            }
            LayerMode::AttentionOnly => {
                // Run attention but skip FFN — residual gets attention contribution only.
                if let Some(h_post_attn) = run_attention(weights, &h, layer) {
                    h = h_post_attn;
                }
            }
        }
    }

    logits_to_predictions(weights, &h, tokenizer, top_k)
}

/// Resume a forward pass from a pre-computed hidden state at a given start layer.
/// Runs layers start_layer..num_layers, then projects to logits.
/// The hidden state `h` should be shaped (seq_len, hidden_size).
pub fn predict_from_hidden(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h_init: &Array2<f32>,
    start_layer: usize,
    top_k: usize,
) -> PredictResult {
    let ffn = WeightFfn { weights };
    predict_from_hidden_with_ffn(weights, tokenizer, h_init, start_layer, top_k, &ffn, &[])
}

/// Resume a forward pass from a pre-computed hidden state with a custom FFN backend.
/// `token_ids` is needed for models with per-layer embeddings (PLE). Pass empty if unavailable.
pub fn predict_from_hidden_with_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h_init: &Array2<f32>,
    start_layer: usize,
    top_k: usize,
    ffn: &dyn FfnBackend,
    token_ids: &[u32],
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = h_init.clone();
    // PLE requires embeddings from token IDs — empty if not available
    let ple_inputs: Vec<Array2<f32>> = if token_ids.is_empty() {
        Vec::new()
    } else {
        let embeds = embed_tokens(weights, token_ids);
        precompute_per_layer_inputs(weights, &embeds, token_ids)
    };

    for layer in start_layer..num_layers {
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer)) {
            Some((h_new, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k)
}

/// Calibrate scalar gains from a forward pass: compute norm[L+1] / norm[L] at each layer.
pub fn calibrate_scalar_gains(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Vec<f32> {
    let all_layers: Vec<usize> = (0..weights.num_layers).collect();
    let trace = trace_forward(weights, token_ids, &all_layers, false, 0);

    let mut gains = Vec::with_capacity(weights.num_layers);
    for i in 0..trace.residuals.len() {
        let norm_curr: f32 = trace.residuals[i].1.iter().map(|x| x * x).sum::<f32>().sqrt();
        if i + 1 < trace.residuals.len() {
            let norm_next: f32 = trace.residuals[i + 1].1.iter().map(|x| x * x).sum::<f32>().sqrt();
            gains.push(if norm_curr > 1e-12 { norm_next / norm_curr } else { 1.0 });
        } else {
            gains.push(1.0);
        }
    }
    gains
}
