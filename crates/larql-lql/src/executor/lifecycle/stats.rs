//! `STATS` — vindex / model summary, knowledge-graph coverage, layer bands.

use crate::error::LqlError;
use crate::executor::helpers::{dir_size, format_bytes, format_number};
use crate::executor::{Backend, Session};

impl Session {
    pub(crate) fn exec_stats(&self, _vindex_path: Option<&str>) -> Result<Vec<String>, LqlError> {
        match &self.backend {
            Backend::Vindex {
                path,
                config,
                patched,
                relation_classifier,
                ..
            } => {
                let index = patched.base();
                let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
                let file_size = dir_size(path);

                let mut out = Vec::new();
                out.push(format!("Model:           {}", config.model));
                out.push(String::new());
                out.push(format!(
                    "Features:        {} ({} x {} layers)",
                    format_number(total_features),
                    format_number(config.intermediate_size),
                    config.num_layers,
                ));

                // Knowledge graph coverage
                out.push(String::new());
                out.push("Knowledge Graph:".into());

                if let Some(rc) = relation_classifier {
                    let num_clusters = rc.num_clusters();
                    let num_probes = rc.num_probe_labels();

                    // Count mapped vs unmapped clusters
                    let mut mapped_clusters = 0;
                    for cluster_id in 0..num_clusters {
                        if let Some((label, _, _)) = rc.cluster_info(cluster_id) {
                            if !label.is_empty() {
                                mapped_clusters += 1;
                            }
                        }
                    }
                    let unmapped_clusters = num_clusters.saturating_sub(mapped_clusters);

                    // Count probe-confirmed relation types
                    // (unique labels among probe labels)
                    let probe_type_count = if num_probes > 0 {
                        let mut types = std::collections::HashSet::new();
                        // We can approximate by scanning loaded layers
                        let layers = index.loaded_layers();
                        for layer in &layers {
                            let n = index.num_features(*layer);
                            for feat in 0..n {
                                if rc.is_probe_label(*layer, feat) {
                                    if let Some(label) = rc.label_for_feature(*layer, feat) {
                                        types.insert(label.to_string());
                                    }
                                }
                            }
                        }
                        types.len()
                    } else {
                        0
                    };

                    out.push(format!("  Clusters:          {num_clusters}"));
                    if num_probes > 0 {
                        out.push(format!(
                            "  Mapped relations:  {num_probes} features ({probe_type_count} types, probe-confirmed)"
                        ));
                    }
                    if mapped_clusters > 0 {
                        out.push(format!(
                            "  Partially mapped:  {mapped_clusters} clusters (Wikidata/WordNet matched)"
                        ));
                    }
                    out.push(format!(
                        "  Unmapped:          {unmapped_clusters} clusters (model knows, we haven't identified yet)"
                    ));
                } else {
                    out.push("  (no relation clusters found)".into());
                }

                // Layer band breakdown
                let layers = index.loaded_layers();
                let syntax_features: usize = layers
                    .iter()
                    .filter(|l| **l <= 13)
                    .map(|l| index.num_features(*l))
                    .sum();
                let knowledge_features: usize = layers
                    .iter()
                    .filter(|l| **l >= 14 && **l <= 27)
                    .map(|l| index.num_features(*l))
                    .sum();
                let output_features: usize = layers
                    .iter()
                    .filter(|l| **l >= 28)
                    .map(|l| index.num_features(*l))
                    .sum();

                out.push(String::new());
                out.push("  By layer band:".into());
                let syntax_fmt = format_number(syntax_features);
                out.push(format!(
                    "    Syntax (L0-13):     {syntax_fmt} features"
                ));
                let knowledge_fmt = format_number(knowledge_features);
                out.push(format!(
                    "    Knowledge (L14-27): {knowledge_fmt} features"
                ));
                let output_fmt = format_number(output_features);
                out.push(format!(
                    "    Output (L28-33):    {output_fmt} features"
                ));

                // Coverage summary
                if let Some(rc) = relation_classifier {
                    let num_probes = rc.num_probe_labels();
                    let num_clusters = rc.num_clusters();

                    if num_clusters > 0 {
                        let mut mapped_clusters = 0;
                        for cluster_id in 0..num_clusters {
                            if let Some((label, _, _)) = rc.cluster_info(cluster_id) {
                                if !label.is_empty() {
                                    mapped_clusters += 1;
                                }
                            }
                        }

                        let probe_pct = if total_features > 0 {
                            (num_probes as f64 / total_features as f64) * 100.0
                        } else {
                            0.0
                        };
                        let cluster_pct = (mapped_clusters as f64 / num_clusters as f64) * 100.0;
                        let total_mapped_pct =
                            ((mapped_clusters as f64 / num_clusters as f64) * 100.0).min(100.0);
                        let unmapped_pct = 100.0 - total_mapped_pct;

                        out.push(String::new());
                        out.push("  Coverage:".into());
                        let total_fmt = format_number(total_features);
                        out.push(format!(
                            "    Probe-confirmed:   {probe_pct:.2}% of features ({num_probes} / {total_fmt})"
                        ));
                        out.push(format!(
                            "    Cluster-labelled:  {cluster_pct:.0}% of clusters ({mapped_clusters} / {num_clusters})"
                        ));
                        out.push(format!(
                            "    Unmapped:          ~{unmapped_pct:.0}% — the model knows more than we've labelled"
                        ));
                    }
                }

                out.push(String::new());
                let size_bytes = format_bytes(file_size);
                out.push(format!("Index size:      {size_bytes}"));
                let path_str = path.display();
                out.push(format!("Path:            {path_str}"));
                Ok(out)
            }
            Backend::Weight {
                model_id, weights, ..
            } => {
                let mut out = Vec::new();
                out.push(format!("Model:           {model_id}"));
                out.push("Backend:         live weights (no vindex)".to_string());
                out.push(String::new());
                out.push(format!("Layers:          {}", weights.num_layers));
                out.push(format!("Hidden size:     {}", weights.hidden_size));
                out.push(format!("Intermediate:    {}", weights.intermediate_size));
                let vocab_fmt = format_number(weights.vocab_size);
                out.push(format!(
                    "Vocab size:      {vocab_fmt}"
                ));
                out.push(String::new());
                out.push("Supported:       INFER, EXPLAIN INFER, STATS".into());
                out.push("For WALK/DESCRIBE/SELECT/INSERT: EXTRACT into a vindex first.".into());
                Ok(out)
            }
            Backend::Remote { .. } => self.remote_stats(),
            Backend::None => Err(LqlError::NoBackend),
        }
    }
}
