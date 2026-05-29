//! Executor for TRACE statements.
//!
//! Runs a decomposed forward pass, captures attn/FFN deltas at every layer,
//! and formats the results. Optionally tracks a specific answer token,
//! shows the attn/FFN decomposition, or saves to a trace file.

use crate::ast::{Range, TracePositionMode};
use crate::error::LqlError;
use crate::executor::helpers::format_knn_override_summary;

#[derive(Debug)]
struct PendingRetrievalOverride {
    override_: larql_inference::KnnOverride,
    model_top1: Option<(String, f64)>,
}

impl super::Session {
    pub(crate) fn exec_trace(
        &self,
        prompt: &str,
        answer: Option<&str>,
        decompose: bool,
        layers: Option<&Range>,
        positions: Option<TracePositionMode>,
        save: Option<&str>,
    ) -> Result<Vec<String>, LqlError> {
        // Weight backend: dense inference (no vindex)
        if let super::Backend::Weight {
            weights, tokenizer, ..
        } = &self.backend
        {
            let ffn = larql_inference::WeightFfn { weights };
            return self.exec_trace_with_ffn(
                weights, tokenizer, &ffn, None, None, prompt, answer, decompose, layers, positions,
                save,
            );
        }

        // Vindex backend: load weights, use walk FFN (editable — INSERT/DELETE affects trace)
        let (path, config, patched) = self.require_vindex()?;

        if !config.has_model_weights {
            return Err(LqlError::Execution(format!(
                "TRACE requires model weights. Rebuild: EXTRACT MODEL \"{}\" INTO \"{}\" WITH ALL",
                config.model,
                path.display(),
            )));
        }

        if config.quant != larql_vindex::QuantFormat::None {
            return Err(LqlError::Execution(
                "TRACE does not yet support quantised (q4k) vindexes — the decomposed forward \
                 pass requires f32 attention tensors that are not present in q4k format. \
                 Tracked as T2 in the ROADMAP."
                    .into(),
            ));
        }

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(path, &mut cb)
            .map_err(|e| LqlError::exec("failed to load model weights", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        // WalkFfn uses vindex gate KNN — same as INFER, mutations are reflected.
        // Unlimited top_k to match `exec_infer`'s full-power baseline so a
        // TRACE of a prompt sees the same residuals / predictions the
        // production INFER path produces.
        let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited(&weights, patched);

        self.exec_trace_with_ffn(
            &weights,
            &tokenizer,
            &walk_ffn,
            Some(patched as &dyn larql_vindex::GateIndex),
            Some(&patched.knn_store),
            prompt,
            answer,
            decompose,
            layers,
            positions,
            save,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_trace_with_ffn(
        &self,
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &dyn larql_inference::FfnBackend,
        gate_index: Option<&dyn larql_vindex::GateIndex>,
        knn_store: Option<&larql_vindex::KnnStore>,
        prompt: &str,
        answer: Option<&str>,
        decompose: bool,
        layers: Option<&Range>,
        positions: Option<TracePositionMode>,
        save: Option<&str>,
    ) -> Result<Vec<String>, LqlError> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let pending_retrieval_override = match (gate_index, knn_store) {
            (Some(gate_index), Some(store)) if !store.is_empty() => {
                let infer = larql_inference::infer_patched(
                    weights,
                    tokenizer,
                    gate_index,
                    Some(store),
                    &token_ids,
                    1,
                );
                infer
                    .knn_override
                    .map(|override_| PendingRetrievalOverride {
                        override_,
                        model_top1: infer.model_top1,
                    })
            }
            _ => None,
        };

        let pos = match positions {
            Some(TracePositionMode::All) => larql_inference::TracePositions::All,
            _ => larql_inference::TracePositions::Last,
        };
        if save.is_some() && !matches!(positions, Some(TracePositionMode::All)) {
            return Err(LqlError::Execution(
                "TRACE SAVE requires POSITIONS ALL so the mmap trace contains complete token chains"
                    .into(),
            ));
        }

        let start = std::time::Instant::now();
        let mut trace = larql_inference::trace_residuals(weights, &token_ids, pos, false, ffn);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Fill in token strings
        trace.prompt = prompt.to_string();
        trace.tokens = token_ids
            .iter()
            .map(|&id| {
                tokenizer
                    .decode(&[id], true)
                    .unwrap_or_else(|_| format!("t{}", id))
            })
            .collect();

        let mut out = Vec::new();
        let n_layers = trace.n_layers;
        out.push(format!(
            "Trace: \"{}\" ({} tokens, {} layers, {:.0}ms)",
            prompt,
            trace.tokens.len(),
            n_layers,
            elapsed_ms,
        ));

        // Determine layer range to display
        let (l_start, l_end) = match layers {
            Some(r) => (r.start as i32, r.end as i32),
            None => (-1, n_layers as i32 - 1),
        };

        // If ANSWER specified: show answer trajectory
        if let Some(answer_str) = answer {
            let answer_tok = tokenizer
                .encode(format!(" {}", answer_str), true)
                .map_err(|e| LqlError::exec("tokenize answer", e))?;
            let answer_id = *answer_tok.get_ids().last().unwrap_or(&0);

            let traj = trace.answer_trajectory(weights, answer_id);

            out.push(String::new());
            out.push(format!("Answer trajectory for '{}':", answer_str));
            out.push(format!(
                "  {:>5} {:>6} {:>8} {:>9} {:>9} {:>8}",
                "Layer", "Rank", "Prob", "Attn", "FFN", "Who"
            ));

            for w in &traj {
                if w.layer < l_start || w.layer > l_end {
                    continue;
                }

                let who = if w.layer == -1 {
                    "embed"
                } else if w.attn_logit > 0.5 && w.ffn_logit > 0.5 {
                    "BOTH ↑"
                } else if w.attn_logit > 0.5 {
                    "ATTN ↑"
                } else if w.ffn_logit > 0.5 {
                    "FFN ↑"
                } else if w.attn_logit < -0.5 && w.ffn_logit < -0.5 {
                    "both ↓"
                } else if w.attn_logit < -0.5 {
                    "attn ↓"
                } else if w.ffn_logit < -0.5 {
                    "ffn ↓"
                } else {
                    "~"
                };

                let layer_str = if w.layer == -1 {
                    "emb".to_string()
                } else {
                    format!("L{}", w.layer)
                };
                out.push(format!(
                    "  {:>5} {:>6} {:>8.4} {:>+9.1} {:>+9.1} {:>8}",
                    layer_str, w.rank, w.prob, w.attn_logit, w.ffn_logit, who,
                ));
            }
            append_pending_retrieval_override(&mut out, pending_retrieval_override.as_ref());
            return self.maybe_save_and_return(out, &trace, weights, save);
        }

        // If DECOMPOSE: show attn vs FFN norms at each layer
        if decompose {
            out.push(String::new());
            out.push(format!(
                "  {:>5} {:>10} {:>10} {:>10} {:>10}",
                "Layer", "|attn_Δ|", "|ffn_Δ|", "|residual|", "attn%"
            ));

            for layer in l_start..=l_end {
                let last_pos = trace.tokens.len() - 1;
                let node = match trace.node(layer, last_pos) {
                    Some(n) => n,
                    None => continue,
                };
                let attn_norm = vec_norm(&node.attn_delta);
                let ffn_norm = vec_norm(&node.ffn_delta);
                let res_norm = vec_norm(&node.residual);
                let ratio = if attn_norm + ffn_norm > 0.0 {
                    attn_norm / (attn_norm + ffn_norm) * 100.0
                } else {
                    0.0
                };

                let layer_str = if layer == -1 {
                    "emb".to_string()
                } else {
                    format!("L{}", layer)
                };
                out.push(format!(
                    "  {:>5} {:>10.0} {:>10.0} {:>10.0} {:>9.0}%",
                    layer_str, attn_norm, ffn_norm, res_norm, ratio,
                ));
            }
            append_pending_retrieval_override(&mut out, pending_retrieval_override.as_ref());
            return self.maybe_save_and_return(out, &trace, weights, save);
        }

        // Default: per-layer summary (top-1 prediction at each layer)
        let summaries = trace.layer_summaries(weights, tokenizer);
        out.push(String::new());
        out.push(format!(
            "  {:>5} {:>12} {:>8} {:>10} {:>10}",
            "Layer", "Top-1", "Prob", "|attn_Δ|", "|ffn_Δ|"
        ));

        for s in &summaries {
            if s.layer < l_start || s.layer > l_end {
                continue;
            }
            let layer_str = if s.layer == -1 {
                "emb".to_string()
            } else {
                format!("L{}", s.layer)
            };
            out.push(format!(
                "  {:>5} {:>12} {:>8.3} {:>10.0} {:>10.0}",
                layer_str, s.top1_token, s.top1_prob, s.attn_delta_norm, s.ffn_delta_norm,
            ));
        }

        append_pending_retrieval_override(&mut out, pending_retrieval_override.as_ref());
        self.maybe_save_and_return(out, &trace, weights, save)
    }

    fn maybe_save_and_return(
        &self,
        mut out: Vec<String>,
        trace: &larql_inference::ResidualTrace,
        _weights: &larql_inference::ModelWeights,
        save: Option<&str>,
    ) -> Result<Vec<String>, LqlError> {
        if let Some(path) = save {
            let mut writer = larql_inference::TraceWriter::create(
                std::path::Path::new(path),
                trace.hidden_size,
                trace.n_layers,
            )
            .map_err(|e| LqlError::exec("save trace", e))?;

            let written = writer
                .write_trace(trace)
                .map_err(|e| LqlError::exec("write trace", e))?;
            writer
                .finish()
                .map_err(|e| LqlError::exec("finish trace", e))?;

            out.push(String::new());
            out.push(format!("Saved {} token chains to {}", written, path));
        }
        Ok(out)
    }
}

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn append_pending_retrieval_override(
    out: &mut Vec<String>,
    pending: Option<&PendingRetrievalOverride>,
) {
    let Some(pending) = pending else {
        return;
    };
    out.push(String::new());
    out.push("Pending retrieval override:".into());
    out.push(format!(
        "  {} ({})",
        pending.override_.token,
        format_knn_override_summary(&pending.override_, pending.model_top1.as_ref())
    ));
    out.push(
        "  note: KNN sidecar is applied after logits; it is not part of this residual/FFN DAG."
            .into(),
    );
}
