//! Phases 1–3 of `DESCRIBE <entity>`: build the query embedding,
//! pick scan layers, walk + collect edges into per-target buckets.

use std::collections::HashMap;

use crate::ast::LayerBand;
use crate::error::LqlError;
use crate::executor::helpers::{is_content_token, is_readable_token};
use crate::executor::tuning::{
    DESCRIBE_ALSO_CONTENT_TAKE, DESCRIBE_ALSO_READABLE_TAKE, DESCRIBE_COHERENCE_FLOOR,
    DESCRIBE_GATE_THRESHOLD,
};

/// Tokenise `entity` and build a query vector by averaging its token
/// embeddings. Returns `Ok(None)` when the entity tokenises to nothing
/// — the caller emits the "(not found)" line.
pub(super) fn describe_build_query(
    entity: &str,
    path: &std::path::Path,
) -> Result<Option<larql_vindex::ndarray::Array1<f32>>, LqlError> {
    let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
        .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)
        .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;
    crate::executor::helpers::entity_query_vec(&tokenizer, &embed, embed_scale, entity)
}

/// Filter `all_layers` down to those covered by the requested band /
/// explicit layer filter. An explicit `LAYER N` short-circuits — the
/// layer is returned regardless of band membership.
pub(super) fn describe_scan_layers(
    bands: &larql_vindex::LayerBands,
    all_layers: &[usize],
    band: Option<LayerBand>,
    layer: Option<u32>,
) -> Vec<usize> {
    if let Some(l) = layer {
        return vec![l as usize];
    }
    let in_range = |range: (usize, usize)| {
        all_layers
            .iter()
            .copied()
            .filter(move |l| *l >= range.0 && *l <= range.1)
            .collect::<Vec<_>>()
    };
    match band {
        Some(LayerBand::Syntax) => in_range(bands.syntax),
        Some(LayerBand::Knowledge) => in_range(bands.knowledge),
        Some(LayerBand::Output) => in_range(bands.output),
        Some(LayerBand::All) | None => all_layers.to_vec(),
    }
}

/// Per-target accumulator built up while walking the trace. One
/// instance per unique (lowercased) target token; multiple feature
/// hits across layers fold into a single row.
pub(super) struct DescribeEdge {
    pub gate: f32,
    pub layers: Vec<usize>,
    pub count: usize,
    pub original: String,
    pub also: Vec<String>,
    pub best_layer: usize,
    pub best_feature: usize,
}

/// Walk the trace, deduplicate by lowercased target token, and apply
/// content / coherence filters. Output is sorted descending by gate.
pub(super) fn describe_collect_edges(
    trace: &larql_vindex::WalkTrace,
    entity: &str,
) -> Vec<DescribeEdge> {
    let entity_lower = entity.to_lowercase();
    let mut edges: HashMap<String, DescribeEdge> = HashMap::new();

    for (layer_idx, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score < DESCRIBE_GATE_THRESHOLD {
                continue;
            }
            let tok = &hit.meta.top_token;
            if !is_content_token(tok) {
                continue;
            }
            if tok.to_lowercase() == entity_lower {
                continue;
            }

            let also_readable: Vec<String> = hit
                .meta
                .top_k
                .iter()
                .filter(|t| {
                    t.token.to_lowercase() != tok.to_lowercase()
                        && t.token.to_lowercase() != entity_lower
                        && is_readable_token(&t.token)
                        && t.logit > 0.0
                })
                .take(DESCRIBE_ALSO_READABLE_TAKE)
                .map(|t| t.token.clone())
                .collect();

            let also: Vec<String> = also_readable
                .iter()
                .filter(|t| is_content_token(t))
                .take(DESCRIBE_ALSO_CONTENT_TAKE)
                .cloned()
                .collect();

            // Coherence filter: weak edge with a noisy also-list is
            // probably tokenisation shadow rather than a real concept;
            // drop unless gate is comfortably above the coherence floor.
            if also.is_empty()
                && !also_readable.is_empty()
                && hit.gate_score < DESCRIBE_COHERENCE_FLOOR
            {
                continue;
            }

            let key = tok.to_lowercase();
            let entry = edges.entry(key).or_insert_with(|| DescribeEdge {
                gate: 0.0,
                layers: Vec::new(),
                count: 0,
                original: tok.to_string(),
                also,
                best_layer: *layer_idx,
                best_feature: hit.feature,
            });

            if hit.gate_score > entry.gate {
                entry.gate = hit.gate_score;
                entry.best_layer = *layer_idx;
                entry.best_feature = hit.feature;
            }
            if !entry.layers.contains(layer_idx) {
                entry.layers.push(*layer_idx);
            }
            entry.count += 1;
        }
    }

    let mut ranked: Vec<DescribeEdge> = edges.into_values().collect();
    ranked.sort_by(|a, b| {
        b.gate
            .partial_cmp(&a.gate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bands() -> larql_vindex::LayerBands {
        larql_vindex::LayerBands {
            syntax: (0, 4),
            knowledge: (5, 9),
            output: (10, 12),
        }
    }

    #[test]
    fn scan_layers_explicit_layer_short_circuits_band() {
        let layers: Vec<usize> = (0..13).collect();
        let out = describe_scan_layers(&bands(), &layers, Some(LayerBand::Syntax), Some(11));
        assert_eq!(out, vec![11]);
    }

    #[test]
    fn scan_layers_syntax_band_filters_to_range() {
        let layers: Vec<usize> = (0..13).collect();
        let out = describe_scan_layers(&bands(), &layers, Some(LayerBand::Syntax), None);
        assert_eq!(out, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn scan_layers_knowledge_band_filters_to_range() {
        let layers: Vec<usize> = (0..13).collect();
        let out = describe_scan_layers(&bands(), &layers, Some(LayerBand::Knowledge), None);
        assert_eq!(out, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn scan_layers_output_band_filters_to_range() {
        let layers: Vec<usize> = (0..13).collect();
        let out = describe_scan_layers(&bands(), &layers, Some(LayerBand::Output), None);
        assert_eq!(out, vec![10, 11, 12]);
    }

    #[test]
    fn scan_layers_all_band_returns_input_unchanged() {
        let layers: Vec<usize> = vec![0, 5, 12];
        let out = describe_scan_layers(&bands(), &layers, Some(LayerBand::All), None);
        assert_eq!(out, layers);
    }

    #[test]
    fn scan_layers_no_band_no_layer_returns_all() {
        let layers: Vec<usize> = vec![0, 5, 12];
        let out = describe_scan_layers(&bands(), &layers, None, None);
        assert_eq!(out, layers);
    }
}
