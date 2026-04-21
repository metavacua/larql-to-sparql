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

        // Shared INFER pipeline — walk FFN (unlimited features) plus KnnStore
        // side-channel override. Same code path as `PyVindex::infer`; see ADR
        // 0001 (docs/adr/0001-python-lql-infer-parity.md).
        let infer = larql_inference::infer_patched(
            &weights,
            &tokenizer,
            patched,
            Some(&patched.knn_store),
            &token_ids,
            top_k,
        );

        let trace_layers = larql_inference::walk_trace_from_residuals(&infer.residuals, patched);

        let mut out = Vec::new();
        out.push("Predictions (walk FFN):".into());
        if let Some(ovr) = &infer.knn_override {
            out.push(format!(
                "   1. {:20} (KNN override, cos={:.2}, L{})",
                ovr.token, ovr.cosine, ovr.layer,
            ));
            for (i, (tok, prob)) in infer.predictions.iter().skip(1).enumerate() {
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 2, tok, prob * 100.0));
            }
        } else {
            for (i, (tok, prob)) in infer.predictions.iter().enumerate() {
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
            }
        }
        out.push(format!("  {:.0}ms", infer.walk_ms));

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
