//! `SELECT NEAREST TO "entity" AT LAYER N` — KNN at a fixed layer.
//!
//! Builds the entity's averaged embedding, runs `gate_knn` at the
//! requested layer, and renders one row per hit. Self-contained:
//! takes the patched vindex via `&Session` and resolves
//! embeddings/tokenizer from disk on each call.

use crate::ast::NearestClause;
use crate::error::LqlError;
use crate::executor::Session;

use super::format::{also_display, banner, format_also, NEAREST_DEFAULT_LIMIT};

impl Session {
    pub(super) fn exec_select_nearest(
        &self,
        index: &larql_vindex::PatchedVindex,
        path: &std::path::Path,
        nc: &NearestClause,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let limit = limit.unwrap_or(NEAREST_DEFAULT_LIMIT) as usize;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let Some(query) = crate::executor::helpers::entity_query_vec(
            &tokenizer,
            &embed,
            embed_scale,
            nc.entity.as_str(),
        )?
        else {
            return Ok(vec!["  (entity not found)".into()]);
        };

        let hits = index.gate_knn(nc.layer as usize, &query, limit);

        let classifier = self.relation_classifier();

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
            "Layer", "Feature", "Token", "Also", "Relation", "Score"
        ));
        out.push(banner(86));

        for (feat, score) in &hits {
            let meta = index.feature_meta(nc.layer as usize, *feat);
            let tok = meta
                .as_ref()
                .map(|m| m.top_token.clone())
                .unwrap_or_else(|| "-".into());
            let also = meta
                .as_ref()
                .map(|m| also_display(&format_also(&m.top_k)))
                .unwrap_or_default();
            let rel = classifier
                .and_then(|rc| rc.label_for_feature(nc.layer as usize, *feat))
                .unwrap_or("");
            out.push(format!(
                "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                nc.layer, feat, tok, also, rel, score
            ));
        }

        if hits.is_empty() {
            out.push("  (no matching features)".into());
        }

        Ok(out)
    }
}
