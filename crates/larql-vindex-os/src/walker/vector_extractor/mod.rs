//! Extract full vectors from model weight matrices to intermediate NDJSON files.
//!
//! Same safetensors loading and BLAS matmuls as weight_walker, but captures the
//! full weight vector (hidden-dim) alongside top-k token metadata. Output is one
//! `.vectors.jsonl` file per component type (ffn_down, ffn_gate, etc.).
//!
//! The stored vector is the raw weight direction (dim = hidden_size), NOT the
//! vocab-projected logits. The vocab projection is computed only to derive
//! top-k token metadata (same matmul as weight_walker).
//!
//! Zero forward passes. Pure matrix multiplication.
//!
//! Module layout (round-6 split, 2026-05-10):
//!   - `types`       — `ExtractConfig`, `ExtractCallbacks`, summaries.
//!   - `writer`      — `VectorWriter`, `scan_completed_layers` (resume helper).
//!   - `loader`      — `VectorExtractor` struct + `load()` / getters.
//!   - `ffn`         — `extract_ffn_down` / `extract_ffn_gate` / `extract_ffn_up`.
//!   - `attn`        — `extract_attn_ov` / `extract_attn_qk`.
//!   - `embeddings`  — `extract_embeddings`.
//!   - `mod` (here) — `extract_all` orchestrator + public re-exports.

mod attn;
mod embeddings;
mod ffn;
mod loader;
mod types;
mod writer;

use std::collections::HashSet;
use std::path::Path;

use crate::error::VindexError;
use crate::walker::utils::current_date;

// ── Public re-exports preserving the pre-split path
// (`larql_vindex::walker::vector_extractor::*`).

pub use larql_models::{
    TopKEntry, VectorFileHeader, VectorRecord, ALL_COMPONENTS, COMPONENT_ATTN_OV,
    COMPONENT_ATTN_QK, COMPONENT_EMBEDDINGS, COMPONENT_FFN_DOWN, COMPONENT_FFN_GATE,
    COMPONENT_FFN_UP,
};
pub use loader::VectorExtractor;
pub use types::{
    ComponentSummary, ExtractCallbacks, ExtractConfig, ExtractSummary, SilentExtractCallbacks,
};
pub use writer::{scan_completed_layers, VectorWriter};

impl VectorExtractor {
    /// Orchestrate extraction of all requested components across requested layers.
    ///
    /// Returns `None` for unimplemented components so the caller can decide
    /// how to report them (keeps eprintln out of core, same as weight_walker).
    pub fn extract_all(
        &self,
        config: &ExtractConfig,
        output_dir: &Path,
        resume: bool,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<ExtractSummary, VindexError> {
        std::fs::create_dir_all(output_dir)?;
        let overall_start = std::time::Instant::now();
        let mut summaries = Vec::new();

        let layers: Vec<usize> = match &config.layers {
            Some(ls) => ls.clone(),
            None => (0..self.num_layers()).collect(),
        };

        for component in &config.components {
            // Embeddings are layer-independent — handle separately
            if component == COMPONENT_EMBEDDINGS {
                let file_path = output_dir.join(format!("{component}.vectors.jsonl"));
                if resume && file_path.exists() {
                    summaries.push(ComponentSummary {
                        component: component.clone(),
                        vectors_written: 0,
                        output_path: file_path,
                        elapsed_secs: 0.0,
                    });
                    continue;
                }
                let comp_start = std::time::Instant::now();
                callbacks.on_component_start(component, 1);
                let mut w = VectorWriter::create(&file_path)?;
                w.write_header(&VectorFileHeader {
                    _header: true,
                    component: component.clone(),
                    model: self.model_name().to_string(),
                    dimension: self.hidden_size(),
                    extraction_date: current_date(),
                })?;
                let count = self.extract_embeddings(config, &mut w, callbacks)?;
                let elapsed_ms = comp_start.elapsed().as_secs_f64() * 1000.0;
                callbacks.on_layer_done(component, 0, count, elapsed_ms);
                callbacks.on_component_done(component, count);
                summaries.push(ComponentSummary {
                    component: component.clone(),
                    vectors_written: count,
                    output_path: file_path,
                    elapsed_secs: comp_start.elapsed().as_secs_f64(),
                });
                continue;
            }

            let file_path = output_dir.join(format!("{component}.vectors.jsonl"));
            let comp_start = std::time::Instant::now();

            let completed = if resume {
                scan_completed_layers(&file_path)?
            } else {
                HashSet::new()
            };

            let pending: Vec<usize> = layers
                .iter()
                .filter(|l| !completed.contains(l))
                .copied()
                .collect();

            if pending.is_empty() {
                summaries.push(ComponentSummary {
                    component: component.clone(),
                    vectors_written: 0,
                    output_path: file_path,
                    elapsed_secs: 0.0,
                });
                continue;
            }

            callbacks.on_component_start(component, pending.len());

            let (mut writer, _existing) = if resume && file_path.exists() {
                VectorWriter::append(&file_path)?
            } else {
                let mut w = VectorWriter::create(&file_path)?;
                w.write_header(&VectorFileHeader {
                    _header: true,
                    component: component.clone(),
                    model: self.model_name().to_string(),
                    dimension: self.hidden_size(),
                    extraction_date: current_date(),
                })?;
                (w, 0)
            };

            let mut total_written = 0;

            for &layer in &pending {
                let layer_start = std::time::Instant::now();

                let count = match component.as_str() {
                    COMPONENT_FFN_DOWN => {
                        self.extract_ffn_down(layer, config, &mut writer, callbacks)?
                    }
                    COMPONENT_FFN_GATE => {
                        self.extract_ffn_gate(layer, config, &mut writer, callbacks)?
                    }
                    COMPONENT_FFN_UP => {
                        self.extract_ffn_up(layer, config, &mut writer, callbacks)?
                    }
                    COMPONENT_ATTN_OV => {
                        self.extract_attn_ov(layer, config, &mut writer, callbacks)?
                    }
                    COMPONENT_ATTN_QK => {
                        self.extract_attn_qk(layer, config, &mut writer, callbacks)?
                    }
                    _ => 0,
                };

                let elapsed_ms = layer_start.elapsed().as_secs_f64() * 1000.0;
                callbacks.on_layer_done(component, layer, count, elapsed_ms);
                total_written += count;
            }

            callbacks.on_component_done(component, total_written);

            summaries.push(ComponentSummary {
                component: component.clone(),
                vectors_written: total_written,
                output_path: file_path,
                elapsed_secs: comp_start.elapsed().as_secs_f64(),
            });
        }

        let total_vectors = summaries.iter().map(|s| s.vectors_written).sum();
        Ok(ExtractSummary {
            components: summaries,
            total_vectors,
            elapsed_secs: overall_start.elapsed().as_secs_f64(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::walker::test_fixture::create_mock_model;

    fn fixture(slug: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("larql_vex_orch_{slug}"));
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);
        dir
    }

    fn cleanup(dir: &std::path::Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn extract_all_returns_summary_per_component() {
        let dir = fixture("extract_all");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        let out = dir.join("output");
        std::fs::create_dir_all(&out).unwrap();

        let cfg = ExtractConfig {
            components: vec![COMPONENT_FFN_DOWN.into(), COMPONENT_EMBEDDINGS.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let mut cb = SilentExtractCallbacks;
        let summary = extractor.extract_all(&cfg, &out, false, &mut cb).unwrap();

        assert_eq!(summary.components.len(), 2);
        assert!(summary.total_vectors >= 4 + 16);
        assert!(out.join("ffn_down.vectors.jsonl").exists());
        assert!(out.join("embeddings.vectors.jsonl").exists());
        cleanup(&dir);
    }

    #[test]
    fn extract_all_resume_skips_completed_layers() {
        let dir = fixture("extract_resume");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        let out = dir.join("output");
        std::fs::create_dir_all(&out).unwrap();

        let cfg = ExtractConfig {
            components: vec![COMPONENT_FFN_DOWN.into()],
            layers: None,
            top_k: 2,
        };
        let mut cb = SilentExtractCallbacks;
        let first = extractor.extract_all(&cfg, &out, false, &mut cb).unwrap();
        assert_eq!(first.total_vectors, 4 * 2);

        let resumed = extractor.extract_all(&cfg, &out, true, &mut cb).unwrap();
        assert_eq!(resumed.total_vectors, 0);
        cleanup(&dir);
    }

    #[test]
    fn extract_all_resume_skips_existing_embeddings() {
        let dir = fixture("extract_resume_emb");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        let out = dir.join("output");
        std::fs::create_dir_all(&out).unwrap();

        let cfg = ExtractConfig {
            components: vec![COMPONENT_EMBEDDINGS.into()],
            layers: None,
            top_k: 2,
        };
        let mut cb = SilentExtractCallbacks;
        let first = extractor.extract_all(&cfg, &out, false, &mut cb).unwrap();
        assert_eq!(first.total_vectors, 16);

        let resumed = extractor.extract_all(&cfg, &out, true, &mut cb).unwrap();
        assert_eq!(resumed.total_vectors, 0);
        cleanup(&dir);
    }

    #[test]
    fn extract_all_unknown_component_records_zero_writes() {
        let dir = fixture("extract_unknown");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        let out = dir.join("output");
        std::fs::create_dir_all(&out).unwrap();

        let cfg = ExtractConfig {
            components: vec!["never_a_real_component".into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let mut cb = SilentExtractCallbacks;
        let summary = extractor.extract_all(&cfg, &out, false, &mut cb).unwrap();
        assert_eq!(summary.components.len(), 1);
        assert_eq!(summary.total_vectors, 0);
        cleanup(&dir);
    }
}
