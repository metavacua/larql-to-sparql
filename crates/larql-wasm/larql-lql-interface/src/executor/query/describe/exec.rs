//! `exec_describe` orchestrator: build query, walk, format, render.

use crate::ast::{DescribeMode, LayerBand};
use crate::error::LqlError;
use crate::executor::tuning::{
    DESCRIBE_KNN_GATE_SCALE, DESCRIBE_MAX_EDGES_BRIEF, DESCRIBE_MAX_EDGES_VERBOSE,
    DESCRIBE_MAX_OUTPUT_BRIEF, DESCRIBE_SIGNAL_CLEAN, DESCRIBE_SIGNAL_MODERATE,
    DESCRIBE_WALK_TOP_K,
};
use crate::executor::Session;

use super::super::resolve_bands;
use super::collect::{
    describe_build_query, describe_collect_edges, describe_scan_layers, DescribeEdge,
};
use super::format::{describe_format_and_split, format_describe_edge};

/// One row of DESCRIBE banner output: "signal: clean (4 edges, max gate 22.0)".
fn signal_label(max_gate: f32) -> &'static str {
    if max_gate >= DESCRIBE_SIGNAL_CLEAN {
        "clean"
    } else if max_gate >= DESCRIBE_SIGNAL_MODERATE {
        "moderate"
    } else {
        "diffuse"
    }
}

impl Session {
    pub(crate) fn exec_describe(
        &self,
        entity: &str,
        band: Option<LayerBand>,
        layer: Option<u32>,
        relations_only: bool,
        mode: DescribeMode,
    ) -> Result<Vec<String>, LqlError> {
        let verbose = mode != DescribeMode::Brief;

        // MoE router-based DESCRIBE if available.
        if let Some(router_result) = self.try_moe_describe(entity, band, layer, verbose)? {
            return Ok(router_result);
        }

        // ── Phase 1: load embeddings + tokenizer, build query vector ──
        let (path, config, patched) = self.require_vindex()?;
        let Some(query) = describe_build_query(entity, path)? else {
            return Ok(vec![format!("{entity}\n  (not found)")]);
        };

        // ── Phase 2: pick scan layers from band/layer filter ──
        let bands = resolve_bands(config);
        let scan_layers = describe_scan_layers(&bands, &patched.loaded_layers(), band, layer);

        // ── Phase 3: walk + collect edges ──
        let trace = patched.walk(&query, &scan_layers, DESCRIBE_WALK_TOP_K);
        let mut edges = describe_collect_edges(&trace, entity);

        // ── Phase 3b: append KNN store entries for this entity ──
        let knn_hits = patched.knn_store.entries_for_entity(entity);
        for (knn_layer, entry) in knn_hits {
            edges.push(DescribeEdge {
                gate: entry.confidence * DESCRIBE_KNN_GATE_SCALE,
                layers: vec![knn_layer],
                count: 1,
                original: entry.target_token.clone(),
                also: vec![format!("[knn:{}]", entry.relation)],
                best_layer: knn_layer,
                best_feature: 0,
            });
        }

        // ── Phase 4: format ──
        let mut out = vec![entity.to_string()];
        if edges.is_empty() {
            out.push("  (no edges found)".into());
            return Ok(out);
        }

        let max_gate = edges.iter().map(|e| e.gate).fold(0.0_f32, f32::max);
        let edge_count = edges.len();
        out.push(format!(
            "  signal: {} ({} edges, max gate {:.1})",
            signal_label(max_gate),
            edge_count,
            max_gate,
        ));

        let formatted =
            describe_format_and_split(&edges, self.relation_classifier(), relations_only, &bands);

        let max_edges = if mode == DescribeMode::Brief {
            DESCRIBE_MAX_EDGES_BRIEF
        } else {
            DESCRIBE_MAX_EDGES_VERBOSE
        };

        if !formatted.syntax.is_empty() {
            out.push(format!(
                "  Syntax (L{}-{}):",
                bands.syntax.0, bands.syntax.1
            ));
            for edge in formatted.syntax.iter().take(max_edges) {
                out.push(format_describe_edge(edge, mode));
            }
        }
        if !formatted.knowledge.is_empty() {
            out.push(format!(
                "  Edges (L{}-{}):",
                bands.knowledge.0, bands.knowledge.1
            ));
            for edge in formatted.knowledge.iter().take(max_edges) {
                out.push(format_describe_edge(edge, mode));
            }
        }
        if !formatted.output_band.is_empty() {
            out.push(format!(
                "  Output (L{}-{}):",
                bands.output.0, bands.output.1
            ));
            let cap = if mode == DescribeMode::Brief {
                DESCRIBE_MAX_OUTPUT_BRIEF
            } else {
                max_edges
            };
            for edge in formatted.output_band.iter().take(cap) {
                out.push(format_describe_edge(edge, mode));
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_label_clean_at_threshold() {
        assert_eq!(signal_label(DESCRIBE_SIGNAL_CLEAN), "clean");
        assert_eq!(signal_label(50.0), "clean");
    }

    #[test]
    fn signal_label_moderate_band() {
        assert_eq!(signal_label(DESCRIBE_SIGNAL_MODERATE), "moderate");
        assert_eq!(signal_label(15.0), "moderate");
    }

    #[test]
    fn signal_label_diffuse_below_moderate() {
        assert_eq!(signal_label(0.0), "diffuse");
        assert_eq!(signal_label(DESCRIBE_SIGNAL_MODERATE - 0.001), "diffuse");
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn signal_thresholds_are_ordered() {
        // Pinned: callers depend on `clean ≥ moderate`. clippy flags
        // these as compile-time-known but the runtime assert documents
        // the constraint and acts as a regression guard if a future
        // edit reorders the constants.
        assert!(DESCRIBE_SIGNAL_CLEAN > DESCRIBE_SIGNAL_MODERATE);
        assert!(DESCRIBE_SIGNAL_MODERATE > 0.0);
    }
}
