//! `INFER` — full forward pass with attention. Requires model weights.

use crate::error::LqlError;
use crate::executor::{Backend, Session};

impl Session {
    pub(crate) fn exec_infer(
        &mut self,
        prompt: &str,
        top: Option<u32>,
        compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let top_k = top.unwrap_or(5) as usize;

        // Weight backend: dense inference (no vindex needed)
        if let Backend::Weight {
            weights, tokenizer, ..
        } = &self.backend
        {
            let encoding = tokenizer
                .encode(prompt, true)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let start = std::time::Instant::now();
            let result = larql_inference::predict(weights, tokenizer, &token_ids, top_k);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let mut out = Vec::new();
            out.push("Predictions (dense — no vindex):".into());
            for (i, (tok, prob)) in result.predictions.iter().enumerate() {
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
            }
            out.push(format!("  {:.0}ms", elapsed_ms));
            if !compare {
                out.push(String::new());
                out.push(
                    "Tip: EXTRACT into a vindex for walk FFN (sparse, faster, editable).".into(),
                );
            }
            return Ok(out);
        }

        // Vindex backend: walk FFN with optional dense comparison
        let (path, config, patched) = self.require_vindex()?;

        if !config.has_model_weights {
            return Err(LqlError::Execution(format!(
                "INFER requires model weights. This vindex was built without --include-weights.\n\
                 Rebuild: EXTRACT MODEL \"{}\" INTO \"{}\" WITH INFERENCE",
                config.model,
                path.display(),
            )));
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

        // Unlimited top_k: use every feature at each layer, matching
        // the dense FFN path exactly. The 8092 default dropped half
        // of Gemma's 16384 features from the activation sum, which is
        // fine for a clean model (the discarded features have very
        // small activations) but becomes catastrophic once an INSERT
        // lands a strong (×30 gate scale) slot. The slot's activation
        // then dominates a half-weakened baseline, producing
        // whichever installed target has the largest lm_head alignment
        // on every prompt. Matching Python's dense forward pass by
        // using every feature preserves the baseline and keeps the
        // installed slot proportional.
        let walk_ffn =
            larql_inference::vindex::WalkFfn::new_unlimited_with_trace(&weights, patched);
        let start = std::time::Instant::now();
        let result =
            larql_inference::predict_with_ffn(&weights, &tokenizer, &token_ids, top_k, &walk_ffn);
        let walk_ms = start.elapsed().as_secs_f64() * 1000.0;

        // DUAL-MODE: compose-mode inserts participate in the walk above
        // via the FFN overlay (their features fire during the normal
        // logit pathway). KNN-mode inserts are a side-channel override
        // — we check the per-layer KnnStore against captured residuals
        // and, if any stored key matches at cos > 0.75, emit the stored
        // target token as a top-1 override. KNN entries don't participate
        // in the forward pass; they intercept the output at inference.
        let residuals = walk_ffn.take_residuals();

        const KNN_COSINE_THRESHOLD: f32 = 0.75;
        let knn_layers = patched.knn_store.layers();
        let mut knn_override: Option<(String, f32, usize)> = None;
        if !knn_layers.is_empty() {
            for (layer, residual) in &residuals {
                if !knn_layers.contains(layer) {
                    continue;
                }
                if let Some((entry, cosine)) = patched.knn_store.query_top1(*layer, residual) {
                    if cosine > KNN_COSINE_THRESHOLD {
                        knn_override = Some((entry.target_token.clone(), cosine, *layer));
                        break;
                    }
                }
            }
        }

        // Build trace from residuals (same logic as take_trace but inline)
        let mut trace_layers = Vec::with_capacity(residuals.len());
        for (layer, residual) in &residuals {
            let r = larql_vindex::ndarray::Array1::from_vec(residual.clone());
            let hits = patched.gate_knn(*layer, &r, 20);
            let walk_hits: Vec<larql_vindex::WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = patched.feature_meta(*layer, feature)?;
                    Some(larql_vindex::WalkHit {
                        layer: *layer,
                        feature,
                        gate_score,
                        meta,
                    })
                })
                .collect();
            trace_layers.push((*layer, walk_hits));
        }

        let mut out = Vec::new();
        out.push("Predictions (walk FFN):".into());
        if let Some((ref token, cosine, knn_layer)) = knn_override {
            out.push(format!(
                "   1. {:20} (KNN override, cos={:.2}, L{})",
                token, cosine, knn_layer,
            ));
            for (i, (tok, prob)) in result.predictions.iter().enumerate() {
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 2, tok, prob * 100.0));
            }
        } else {
            for (i, (tok, prob)) in result.predictions.iter().enumerate() {
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
            }
        }
        out.push(format!("  {:.0}ms", walk_ms));

        out.push(String::new());
        out.push("Inference trace (features that fired with attention):".into());
        let classifier = self.relation_classifier();
        for (layer, hits) in &trace_layers {
            if hits.is_empty() {
                continue;
            }
            for hit in hits.iter().take(3) {
                let label = classifier
                    .and_then(|rc| rc.label_for_feature(*layer, hit.feature))
                    .unwrap_or("");
                let label_str = if label.is_empty() {
                    String::new()
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

        if compare {
            let start = std::time::Instant::now();
            let dense = larql_inference::predict(&weights, &tokenizer, &token_ids, top_k);
            let dense_ms = start.elapsed().as_secs_f64() * 1000.0;

            out.push(String::new());
            out.push("Predictions (dense):".into());
            for (i, (tok, prob)) in dense.predictions.iter().enumerate() {
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
            }
            out.push(format!("  {:.0}ms", dense_ms));
        }

        Ok(out)
    }
}
