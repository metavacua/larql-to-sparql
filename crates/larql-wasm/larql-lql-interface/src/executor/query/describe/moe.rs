//! MoE-router-based DESCRIBE: when the model has an MoE router,
//! pick experts via the router and surface co-routed tokens
//! (entities that share experts with the queried entity).
//!
//! Returns `Ok(None)` for non-MoE backends so the dense walk-based
//! `exec_describe` handles them.

use std::collections::HashMap;

use crate::ast::LayerBand;
use crate::error::LqlError;
use crate::executor::helpers::is_content_token;
use crate::executor::tuning::{
    DESCRIBE_COROUTED_SAMPLE_SIZE, DESCRIBE_COROUTED_TOKENS_PER_EXPERT, DESCRIBE_MAX_EXPERTS_BRIEF,
    DESCRIBE_MAX_EXPERTS_VERBOSE, DESCRIBE_TOP_EXPERTS_FOR_COROUTED,
};
use crate::executor::{Backend, Session};

impl Session {
    /// MoE-router DESCRIBE. Returns `Ok(None)` when this backend has
    /// no MoE router (i.e. dense model — fall through to the standard
    /// gate-KNN path).
    pub(super) fn try_moe_describe(
        &self,
        entity: &str,
        _band: Option<LayerBand>,
        _layer: Option<u32>,
        verbose: bool,
    ) -> Result<Option<Vec<String>>, LqlError> {
        let router = match &self.backend {
            Backend::Vindex {
                router: Some(r),
                config,
                ..
            } => {
                if config
                    .model_config
                    .as_ref()
                    .and_then(|mc| mc.moe.as_ref())
                    .is_none()
                {
                    return Ok(None);
                }
                r
            }
            _ => return Ok(None),
        };

        let (path, config, _) = self.require_vindex()?;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let Some(query) =
            crate::executor::helpers::entity_query_vec(&tokenizer, &embed, embed_scale, entity)?
        else {
            return Ok(Some(vec![format!("{entity}\n  (not found)")]));
        };

        let bands = super::super::resolve_bands(config);

        let start = std::time::Instant::now();

        let mut out = vec![entity.to_string()];

        let knowledge_range = bands.knowledge.0..=bands.knowledge.1;
        let expert_summary = router.route_all_layers(&query, knowledge_range.clone());

        if verbose {
            out.push(format!(
                "  Routing (L{}-{}):",
                bands.knowledge.0, bands.knowledge.1
            ));
            for l in knowledge_range.clone() {
                if let Some(result) = router.route(l, &query) {
                    let experts_str: String = result
                        .experts
                        .iter()
                        .enumerate()
                        .map(|(i, e)| format!("E{} ({:.0}%)", e, result.probs[i] * 100.0))
                        .collect::<Vec<_>>()
                        .join(", ");
                    out.push(format!("    L{:2}: {}", l, experts_str));
                }
            }
            out.push(String::new());
        }

        let layers_total = bands.knowledge.1 - bands.knowledge.0 + 1;
        out.push(format!(
            "  Experts (L{}-{}):",
            bands.knowledge.0, bands.knowledge.1
        ));
        let max_experts = if verbose {
            DESCRIBE_MAX_EXPERTS_VERBOSE
        } else {
            DESCRIBE_MAX_EXPERTS_BRIEF
        };
        for (eid, count, avg_prob) in expert_summary.iter().take(max_experts) {
            out.push(format!(
                "    E{:<4} {}/{} layers  ({:.0}% avg)",
                eid,
                count,
                layers_total,
                avg_prob * 100.0,
            ));
        }

        let top_experts: Vec<usize> = expert_summary
            .iter()
            .take(DESCRIBE_TOP_EXPERTS_FOR_COROUTED)
            .map(|(e, _, _)| *e)
            .collect();

        if !top_experts.is_empty() {
            out.push(String::new());
            out.push("  Similar (shares experts):".into());

            let mid_layer = (bands.knowledge.0 + bands.knowledge.1) / 2;
            let sample_step = (embed.shape()[0] / DESCRIBE_COROUTED_SAMPLE_SIZE).max(1);
            let mut corouted_all: HashMap<usize, Vec<(String, f32)>> = HashMap::new();

            for tid in (0..embed.shape()[0]).step_by(sample_step) {
                let tok_emb = embed.row(tid).mapv(|v| v * embed_scale);
                if let Some(result) = router.route(mid_layer, &tok_emb) {
                    for (i, &eid) in result.experts.iter().enumerate() {
                        if !top_experts.contains(&eid) {
                            continue;
                        }
                        let tok_str = tokenizer
                            .decode(&[tid as u32], true)
                            .unwrap_or_default()
                            .trim()
                            .to_string();
                        if is_content_token(&tok_str)
                            && tok_str.len() > 1
                            && tok_str.to_lowercase() != entity.to_lowercase()
                        {
                            corouted_all
                                .entry(eid)
                                .or_default()
                                .push((tok_str, result.probs[i]));
                        }
                    }
                }
            }

            for &eid in &top_experts {
                if let Some(tokens) = corouted_all.get_mut(&eid) {
                    tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    tokens.dedup_by(|a, b| a.0.to_lowercase() == b.0.to_lowercase());
                    let display: String = tokens
                        .iter()
                        .take(DESCRIBE_COROUTED_TOKENS_PER_EXPERT)
                        .map(|(t, _)| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    out.push(format!("    E{}: {}", eid, display));
                }
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        out.push(format!("\n  {:.0}ms", elapsed_ms));

        Ok(Some(out))
    }
}
