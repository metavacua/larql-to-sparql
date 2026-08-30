//! Corpus scoring: the forward pass, per-token bits, and the summary print.

use super::*;

pub(super) fn run_score(args: ScoreArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_window(args.context, args.stride)?;
    let text = read_text(&args.corpus, args.bytes)?;
    let model = load_model(&args.model)?;
    let ids = encode_prompt(model.tokenizer(), &*model.weights().arch, &text)?;
    if ids.len() < 2 {
        return Err("corpus must tokenize to at least one scored token".into());
    }

    eprintln!(
        "scoring {} target tokens over {} bytes...",
        ids.len() - 1,
        text.len()
    );
    let summary = score_token_range(
        model.weights(),
        &ids,
        1..ids.len(),
        args.context,
        args.stride,
        Some("scoring"),
    )?;

    print_score_summary(&summary, text.len(), text.chars().count());
    Ok(())
}

pub(crate) fn load_model(model: &str) -> Result<InferenceModel, Box<dyn std::error::Error>> {
    eprintln!("loading {model}...");
    let start = Instant::now();
    let loaded = InferenceModel::load(model)?;
    eprintln!(
        "loaded. {} layers, hidden_size={} ({:.1}s)",
        loaded.num_layers(),
        loaded.hidden_size(),
        start.elapsed().as_secs_f64()
    );
    Ok(loaded)
}

pub(crate) fn read_text(
    path: &PathBuf,
    limit_bytes: Option<usize>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut text = fs::read_to_string(path)?;
    if let Some(limit) = limit_bytes {
        if text.len() > limit {
            let mut end = limit;
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            text.truncate(end);
        }
    }
    Ok(text)
}

pub(super) fn validate_window(
    context: usize,
    stride: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if context < 2 {
        return Err("--context must be at least 2 for scoring".into());
    }
    if stride == 0 {
        return Err("--stride must be at least 1".into());
    }
    if stride >= context {
        return Err("--stride must be smaller than --context so every target has a prefix".into());
    }
    Ok(())
}

pub(super) fn ensure_token_prefix(
    prefix: &[u32],
    full: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    if full.len() < prefix.len() || full[..prefix.len()] != *prefix {
        return Err(
            "answer did not tokenize as a suffix of prefix+answer; add explicit boundary whitespace"
                .into(),
        );
    }
    Ok(())
}

#[derive(Debug, Default)]
pub(super) struct ScoreSummary {
    pub(super) total_bits: f64,
    pub(super) token_bits: Vec<f64>,
}

impl ScoreSummary {
    pub(super) fn bits_per_token(&self) -> f64 {
        self.total_bits / self.token_bits.len().max(1) as f64
    }
}

pub(super) fn score_token_range(
    weights: &ModelWeights,
    ids: &[u32],
    range: Range<usize>,
    context: usize,
    stride: usize,
    progress: Option<&str>,
) -> Result<ScoreSummary, Box<dyn std::error::Error>> {
    if range.start == 0 || range.end > ids.len() || range.start > range.end {
        return Err("invalid scoring token range".into());
    }
    let mut summary = ScoreSummary::default();
    let pb = progress.map(|label| progress_bar((range.end - range.start) as u64, label));
    let mut target_start = range.start;
    while target_start < range.end {
        let target_end = (target_start + stride).min(range.end);
        let prefix_start = target_end
            .saturating_sub(context)
            .min(target_start.saturating_sub(1));
        let chunk_ids = &ids[prefix_start..target_end];
        let hidden = forward_hidden(weights, chunk_ids)?;
        let hidden = final_norm(weights, &hidden);

        let row_start = target_start - prefix_start - 1;
        let row_end = target_end - prefix_start - 1;
        let rows = hidden.slice(s![row_start..row_end, ..]);
        let raw_logits = dot_proj(&rows, &weights.lm_head);
        for (offset, target_pos) in (target_start..target_end).enumerate() {
            let bits = bits_for_raw_row(weights, raw_logits.row(offset), ids[target_pos])?;
            summary.total_bits += bits;
            summary.token_bits.push(bits);
            if let Some(pb) = &pb {
                pb.inc(1);
            }
        }
        target_start = target_end;
    }
    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
    Ok(summary)
}

pub(super) fn print_score_summary(summary: &ScoreSummary, bytes: usize, chars: usize) {
    let chars = chars.max(1) as f64;
    let bytes = bytes.max(1) as f64;
    println!("done.");
    println!("tokens scored:  {:>10}", summary.token_bits.len());
    println!("bits/token:     {:>10.3}", summary.bits_per_token());
    println!("bits/char:      {:>10.3}", summary.total_bits / chars);
    println!("bits/byte:      {:>10.3}", summary.total_bits / bytes);
    println!("total bits:     {:>10.1}", summary.total_bits);
}

/// Pick the FFN backend that matches this model's architecture.
///
/// The scorer used to hardcode [`WeightFfn`], which resolves the *dense*
/// `ffn_{gate,up,down}_key`. A mixture-of-experts model has no such tensors,
/// so scoring one panicked with a misleading "this is a `--compact` vindex"
/// hint — the tensors were not missing, they never existed for that
/// architecture. That made `shannon verify` unusable on every MoE model, which
/// is the entire model class the K3 ladder is built on (GPT-OSS, Kimi Linear,
/// K3). See `docs/k3-funnel.md` §4.7.
///
/// `expert_weights_resolvable` gates on the weights actually being present, so
/// a MoE architecture whose experts are packed rather than per-expert f32 still
/// falls through to the dense path and fails with its own error rather than
/// half-resolving here.
pub(super) fn score_ffn(weights: &ModelWeights) -> Box<dyn FfnBackend + '_> {
    let arch = &*weights.arch;
    if !arch.is_moe() {
        return Box::new(WeightFfn { weights });
    }
    match arch.expert_format() {
        // Separate tensors per expert, as loaded.
        larql_models::ExpertFormat::PerExpert => Box::new(ExpertWeightFfn { weights }),
        // Stacked BF16 operands, read in place.
        larql_models::ExpertFormat::PackedBF16 => Box::new(PackedExpertWeightFfn { weights }),
        // Packed in the *checkpoint* only: `load_mxfp4_expert_tensors`
        // dequantises blocks+scales into per-expert tensors at load, so the
        // operand this tier sees is `PerExpert`. The declaration describes the
        // source; the loader owns the transform. If that transform ever stops
        // running eagerly, this arm is where it shows.
        larql_models::ExpertFormat::PackedMxfp4 => Box::new(ExpertWeightFfn { weights }),
    }
}

pub(super) fn forward_hidden(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    if token_ids.is_empty() {
        return Err("empty token window".into());
    }
    let ffn = score_ffn(weights);
    let mut h = larql_inference::forward::embed_tokens_pub(weights, token_ids);
    let ple_inputs =
        larql_inference::forward::ple::precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: std::collections::HashMap<usize, SharedKV> = std::collections::HashMap::new();
    for layer in 0..weights.num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _, kv_out)) = larql_inference::forward::run_layer_with_ffn(
            larql_inference::WeightsView::dense(weights),
            &h,
            layer,
            &*ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }
    }
    Ok(h)
}

pub(crate) fn forward_hidden_all_layers(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Result<Vec<Array2<f32>>, Box<dyn std::error::Error>> {
    if token_ids.is_empty() {
        return Err("empty token window".into());
    }
    let ffn = score_ffn(weights);
    let h0 = larql_inference::forward::embed_tokens_pub(weights, token_ids);
    let ple_inputs =
        larql_inference::forward::ple::precompute_per_layer_inputs(weights, &h0, token_ids);
    let mut captures: Vec<Array2<f32>> = Vec::with_capacity(weights.num_layers + 1);
    captures.push(h0.clone());
    let mut h = h0;
    let mut kv_cache: std::collections::HashMap<usize, SharedKV> = std::collections::HashMap::new();
    for layer in 0..weights.num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _, kv_out)) = larql_inference::forward::run_layer_with_ffn(
            larql_inference::WeightsView::dense(weights),
            &h,
            layer,
            &*ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }
        captures.push(h.clone());
    }
    Ok(captures)
}

pub(super) fn final_norm(weights: &ModelWeights, h: &Array2<f32>) -> Array2<f32> {
    apply_norm(
        weights,
        h,
        weights.arch.final_norm_key(),
        weights.arch.norm_weight_offset(),
    )
}

pub(super) fn logits_for_last_token(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let hidden = forward_hidden(weights, token_ids)?;
    let hidden = final_norm(weights, &hidden);
    logits_for_row(weights, &hidden, hidden.shape()[0] - 1)
}

pub(super) fn logits_for_row(
    weights: &ModelWeights,
    final_hidden: &Array2<f32>,
    row_idx: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if row_idx >= final_hidden.shape()[0] {
        return Err("logit row out of range".into());
    }
    let row = final_hidden.slice(s![row_idx..row_idx + 1, ..]);
    let raw = dot_proj(&row, &weights.lm_head);
    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    Ok(raw
        .row(0)
        .iter()
        .map(|&v| {
            let mut logit = v * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            logit
        })
        .collect())
}

pub(super) fn bits_for_target(
    logits: &[f32],
    target: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    let target = target as usize;
    if target >= logits.len() {
        return Err(format!("target token {target} out of vocab").into());
    }
    let max_logit = finite_max(logits)?;
    let exp_sum: f64 = logits
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| ((v - max_logit) as f64).exp())
        .sum();
    let logsumexp = max_logit as f64 + exp_sum.ln();
    Ok((logsumexp - logits[target] as f64) / LN_2)
}

pub(super) fn bits_for_raw_row(
    weights: &ModelWeights,
    row: ndarray::ArrayView1<'_, f32>,
    target: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    let target = target as usize;
    if target >= row.len() {
        return Err(format!("target token {target} out of vocab").into());
    }

    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let transform = |v: f32| {
        let mut logit = v * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logit
    };

    let max_logit = row
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(transform)
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |m| m.max(v)))
        })
        .ok_or_else(|| "all logits were non-finite".to_string())?;

    let exp_sum: f64 = row
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| ((transform(v) - max_logit) as f64).exp())
        .sum();
    let target_logit = transform(row[target]);
    let logsumexp = max_logit as f64 + exp_sum.ln();
    Ok((logsumexp - target_logit as f64) / LN_2)
}

pub(super) fn prob_for_target(
    logits: &[f32],
    target: u32,
) -> Result<f64, Box<dyn std::error::Error>> {
    Ok(2.0_f64.powf(-bits_for_target(logits, target)?))
}

/// Apply per-arch logit scaling/softcap and return natural-log probabilities
/// over the full vocabulary for one position. Length matches the input row.
pub(super) fn compute_log_probs_row(
    weights: &ModelWeights,
    row: ndarray::ArrayView1<'_, f32>,
) -> Vec<f32> {
    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let transform = |v: f32| {
        if !v.is_finite() {
            return v;
        }
        let mut logit = v * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logit
    };
    let scaled: Vec<f32> = row.iter().copied().map(transform).collect();
    let max_logit = scaled
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| ((v - max_logit) as f64).exp())
        .sum();
    let logsumexp = (max_logit as f64) + exp_sum.ln();
    scaled
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v as f64) - logsumexp) as f32
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}
