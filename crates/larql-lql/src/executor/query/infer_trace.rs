//! `EXPLAIN INFER` — full forward pass with optional attention capture
//! and logit lens, rendered per layer.

use crate::ast::LayerBand;
use crate::error::LqlError;
use crate::executor::{Backend, Session};

use super::resolve_bands;

impl Session {
    pub(crate) fn exec_infer_trace(
        &self,
        prompt: &str,
        top: Option<u32>,
        band: Option<LayerBand>,
        relations_only: bool,
        with_attention: bool,
    ) -> Result<Vec<String>, LqlError> {
        let top_k = top.unwrap_or(5) as usize;
        let per_layer = top.unwrap_or(3) as usize;

        // Weight backend has no feature labels — short-circuit to a
        // dense-only summary.
        if let Backend::Weight {
            weights, tokenizer, ..
        } = &self.backend
        {
            return self.exec_infer_trace_dense(weights, tokenizer, prompt, top_k);
        }

        // ── Phase 1: load model weights and tokenise ──
        let (path, config, patched) = self.require_vindex()?;
        if !config.has_model_weights {
            return Err(LqlError::Execution(
                "EXPLAIN INFER requires model weights. Rebuild with WITH INFERENCE.".into(),
            ));
        }
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(path, &mut cb)
            .map_err(|e| LqlError::exec("failed to load model weights", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let token_strs: Vec<Option<String>> = if with_attention {
            token_ids
                .iter()
                .map(|&id| larql_inference::decode_token(&tokenizer, id))
                .collect()
        } else {
            Vec::new()
        };

        // ── Phase 2: forward pass (with optional attention capture) ──
        //
        // Unlimited top_k: EXPLAIN INFER shares the activation-sum config
        // with `exec_infer` so running INFER then EXPLAIN INFER on the
        // same prompt gives the same baseline. The attention-capture path
        // is an optional second-channel for logit lens display; the
        // KNN override path below uses WalkFfn residuals either way,
        // matching the canonical `infer_patched` pipeline (ADR 0001).
        let walk_ffn =
            larql_inference::vindex::WalkFfn::new_unlimited_with_trace(&weights, patched);
        let start = std::time::Instant::now();
        let (predictions_raw, attention_captures, lens_residuals) = if with_attention {
            let r = larql_inference::predict_with_ffn_attention(
                &weights, &tokenizer, &token_ids, top_k, &walk_ffn,
            );
            (r.predictions, r.attention, r.residuals)
        } else {
            let r = larql_inference::predict_with_ffn(
                &weights, &tokenizer, &token_ids, top_k, &walk_ffn,
            );
            (r.predictions, Vec::new(), Vec::new())
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let residuals = walk_ffn.take_residuals();
        let (predictions, knn_override) = larql_inference::apply_knn_override(
            predictions_raw,
            &residuals,
            Some(&patched.knn_store),
            top_k,
        );

        // ── Phase 3: side-tables for the rendering loop ──
        let attention_map = build_attention_map(&attention_captures, &token_strs, with_attention);
        let lens_map = build_lens_map(&lens_residuals, &weights, &tokenizer, with_attention);

        let trace_layers = larql_inference::walk_trace_from_residuals(&residuals, patched);
        let classifier = self.relation_classifier();
        let bands = resolve_bands(config);
        let layer_range = band_to_layer_range(band, &bands);

        // ── Phase 4: format header ──
        let band_label = match band {
            Some(LayerBand::Syntax) => " (syntax)",
            Some(LayerBand::Knowledge) => " (knowledge)",
            Some(LayerBand::Output) => " (output)",
            _ => "",
        };

        let mut out = Vec::new();
        out.push(format!("Inference trace for {:?}{}:", prompt, band_label));
        if let Some(ovr) = &knn_override {
            out.push(format!(
                "Prediction: {} (KNN override, cos={:.2}, L{}) in {:.0}ms",
                ovr.token, ovr.cosine, ovr.layer, elapsed_ms
            ));
        } else {
            out.push(format!(
                "Prediction: {} ({:.2}%) in {:.0}ms",
                predictions.first().map(|(t, _)| t.as_str()).unwrap_or("?"),
                predictions.first().map(|(_, p)| p * 100.0).unwrap_or(0.0),
                elapsed_ms
            ));
        }
        out.push(String::new());

        // ── Phase 5: per-layer rendering ──
        for (layer, hits) in &trace_layers {
            if hits.is_empty() {
                continue;
            }
            if let Some((lo, hi)) = layer_range {
                if *layer < lo || *layer > hi {
                    continue;
                }
            }
            render_trace_layer(
                &mut out,
                *layer,
                hits,
                classifier,
                relations_only,
                per_layer,
                with_attention,
                &attention_map,
                &lens_map,
            );
        }

        Ok(out)
    }

    /// EXPLAIN INFER on a `Backend::Weight` (no vindex): produces a dense
    /// inference summary with no feature trace, since there are no
    /// gate vectors / down meta to attribute.
    fn exec_infer_trace_dense(
        &self,
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        prompt: &str,
        top_k: usize,
    ) -> Result<Vec<String>, LqlError> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let start = std::time::Instant::now();
        let result = larql_inference::predict(weights, tokenizer, &token_ids, top_k);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut out = Vec::new();
        out.push(format!(
            "Inference trace for {:?} (dense — no vindex):",
            prompt
        ));
        out.push(format!(
            "Prediction: {} ({:.2}%) in {:.0}ms",
            result
                .predictions
                .first()
                .map(|(t, _)| t.as_str())
                .unwrap_or("?"),
            result
                .predictions
                .first()
                .map(|(_, p)| p * 100.0)
                .unwrap_or(0.0),
            elapsed_ms,
        ));
        out.push(String::new());
        out.push("Note: no per-feature trace without a vindex. EXTRACT for full trace.".into());
        Ok(out)
    }
}

// ── EXPLAIN INFER helpers ────────────────────────────────────────────────
//
// `exec_infer_trace` is a five-phase pipeline (load → forward → side
// tables → header → render). The helpers below split the side-table
// builders and the per-layer rendering loop out of the main function.
// The cross-surface trace reconstruction lives in
// `larql_inference::walk_trace_from_residuals`.

/// Build a `layer → top-3 attended (token, weight)` map from the
/// captured attention weights. Returns an empty map when
/// `with_attention` is false. Averages across all heads, drops special
/// tokens (BOS/EOS) by skipping `None` entries from `decode_token`, and
/// truncates to the top 3 by weight.
fn build_attention_map(
    captures: &[larql_inference::LayerAttentionCapture],
    token_strs: &[Option<String>],
    with_attention: bool,
) -> std::collections::HashMap<usize, Vec<(String, f32)>> {
    if !with_attention {
        return std::collections::HashMap::new();
    }
    let mut map = std::collections::HashMap::new();
    for cap in captures {
        let n_heads = cap.weights.heads.len();
        if n_heads == 0 || token_strs.is_empty() {
            continue;
        }
        let seq_len = cap.weights.heads[0].len();
        let mut avg = vec![0.0f32; seq_len];
        for head in &cap.weights.heads {
            for (j, &w) in head.iter().enumerate() {
                avg[j] += w;
            }
        }
        for v in avg.iter_mut() {
            *v /= n_heads as f32;
        }
        let mut pairs: Vec<(String, f32)> = avg
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(j, w)| {
                let tok = token_strs.get(j)?.as_ref()?;
                Some((tok.trim().to_string(), w))
            })
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(3);
        map.insert(cap.layer, pairs);
    }
    map
}

/// Build a `layer → (top_token, probability)` map by running the logit
/// lens on each captured residual. Returns empty when `with_attention`
/// is false (only the attention path captures intermediate residuals).
fn build_lens_map(
    lens_residuals: &[(usize, Vec<f32>)],
    weights: &larql_inference::ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    with_attention: bool,
) -> std::collections::HashMap<usize, (String, f64)> {
    if !with_attention {
        return std::collections::HashMap::new();
    }
    lens_residuals
        .iter()
        .filter_map(|(layer, residual)| {
            let pred = larql_inference::logit_lens_top1(weights, tokenizer, residual.as_slice())?;
            Some((*layer, pred))
        })
        .collect()
}

/// Resolve a `LayerBand` to a `(lo, hi)` filter on the trace layers.
/// Returns `None` for `All` / no band — the caller treats that as
/// "include every layer".
fn band_to_layer_range(
    band: Option<LayerBand>,
    bands: &larql_vindex::LayerBands,
) -> Option<(usize, usize)> {
    match band {
        Some(LayerBand::Syntax) => Some(bands.syntax),
        Some(LayerBand::Knowledge) => Some(bands.knowledge),
        Some(LayerBand::Output) => Some(bands.output),
        Some(LayerBand::All) | None => None,
    }
}

/// Render one layer's worth of trace hits, in either the compact
/// `with_attention` single-line format (top hit + attention + lens) or
/// the standard multi-line format (top-N hits with relation labels).
#[allow(clippy::too_many_arguments)]
fn render_trace_layer(
    out: &mut Vec<String>,
    layer: usize,
    hits: &[larql_vindex::WalkHit],
    classifier: Option<&crate::relations::RelationClassifier>,
    relations_only: bool,
    per_layer: usize,
    with_attention: bool,
    attention_map: &std::collections::HashMap<usize, Vec<(String, f32)>>,
    lens_map: &std::collections::HashMap<usize, (String, f64)>,
) {
    // When filtering to relations only, re-sort so positive gates rank
    // above negative gates of equal magnitude (positive gates correlate
    // with the prediction; negative gates with the opposite).
    let labelled_hits: Vec<&larql_vindex::WalkHit> = if relations_only {
        let mut lh: Vec<_> = hits
            .iter()
            .filter(|hit| {
                classifier
                    .and_then(|rc| rc.label_for_feature(layer, hit.feature))
                    .map(|l| !l.is_empty())
                    .unwrap_or(false)
            })
            .collect();
        lh.sort_by(|a, b| {
            let a_pos = a.gate_score > 0.0;
            let b_pos = b.gate_score > 0.0;
            match (a_pos, b_pos) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => b
                    .gate_score
                    .abs()
                    .partial_cmp(&a.gate_score.abs())
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        lh
    } else {
        hits.iter().collect()
    };

    if with_attention {
        // Compact single-line format: feature + attention + logit lens.
        let hit = labelled_hits.first();
        let feature_part = if let Some(hit) = hit {
            let label = classifier
                .and_then(|rc| rc.label_for_feature(layer, hit.feature))
                .unwrap_or("");
            if relations_only && label.is_empty() {
                None
            } else {
                let top_token = hit.meta.top_token.trim();
                let name = if !label.is_empty() { label } else { top_token };
                Some(format!("{:<14} {:+.1}", name, hit.gate_score))
            }
        } else {
            None
        };
        let empty = format!("{:19}", "");
        let feature_str = feature_part.as_deref().unwrap_or(&empty);

        let attn_part = attention_map
            .get(&layer)
            .and_then(|attn| attn.first())
            .map(|(tok, w)| format!("{}({:.0}%)", tok, w * 100.0))
            .unwrap_or_default();

        let lens_part = lens_map
            .get(&layer)
            .map(|(tok, prob)| format!("{} ({:.1}%)", tok, prob * 100.0))
            .unwrap_or_default();

        if feature_part.is_some() || !lens_part.is_empty() {
            out.push(format!(
                "  L{:2}  {:<19}  {:<16} → {}",
                layer, feature_str, attn_part, lens_part,
            ));
        }
    } else {
        // Standard multi-line format without attention.
        let mut shown = 0;
        for hit in &labelled_hits {
            if shown >= per_layer {
                break;
            }
            let label = classifier
                .and_then(|rc| rc.label_for_feature(layer, hit.feature))
                .unwrap_or("");
            if relations_only && label.is_empty() {
                continue;
            }
            shown += 1;
            let label_str = if label.is_empty() {
                format!("{:14}", "")
            } else {
                format!("{:<14}", label)
            };
            let top_token = hit.meta.top_token.trim();
            let down_top: String = hit
                .meta
                .top_k
                .iter()
                .take(3)
                .map(|t| t.token.clone())
                .collect::<Vec<_>>()
                .join(", ");
            out.push(format!(
                "  L{:2}: {} F{:<5} gate={:+.1}  → {:15} [{}]",
                layer, label_str, hit.feature, hit.gate_score, top_token, down_top,
            ));
        }
    }
}
