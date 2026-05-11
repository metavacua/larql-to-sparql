//! Attention OV / QK per-layer extraction methods.

use larql_models::{TopKEntry, VectorRecord, COMPONENT_ATTN_OV, COMPONENT_ATTN_QK};

use super::loader::VectorExtractor;
use super::types::{ExtractCallbacks, ExtractConfig};
use super::writer::VectorWriter;
use crate::error::VindexError;
use crate::walker::utils::{decode_token, partial_top_k};

impl VectorExtractor {
    /// Extract attention OV circuit vectors for a single layer.
    ///
    /// For each KV head, computes OV = O_h @ V_h and stores the mean output
    /// direction (hidden-dim) — the average column of the OV matrix, which
    /// represents the head's typical write direction. Same dimensionality as
    /// FFN vectors, so HNSW indexes work uniformly.
    pub fn extract_attn_ov(
        &self,
        layer: usize,
        config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, VindexError> {
        let prefix = format!("layers.{layer}.self_attn.");
        let w_v = self
            .weights
            .tensors
            .get(&format!("{prefix}v_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}v_proj.weight")))?;
        let w_o = self
            .weights
            .tensors
            .get(&format!("{prefix}o_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}o_proj.weight")))?;

        let head_dim = self.weights.arch.head_dim_for_layer(layer);
        let hidden = self.weights.hidden_size;
        let num_kv_heads = w_v.shape()[0] / head_dim;
        callbacks.on_layer_start(COMPONENT_ATTN_OV, layer, num_kv_heads);

        let mut count = 0;

        for h in 0..num_kv_heads {
            callbacks.on_progress(COMPONENT_ATTN_OV, layer, h, num_kv_heads);

            let v_h = w_v.slice(ndarray::s![h * head_dim..(h + 1) * head_dim, ..]);
            let o_h = w_o.slice(ndarray::s![.., h * head_dim..(h + 1) * head_dim]);

            // OV circuit: O_h @ V_h → (hidden, hidden)
            let ov = o_h.dot(&v_h);

            // Mean output direction: average column of OV → (hidden,)
            let mut vector = vec![0.0f32; hidden];
            for col in 0..hidden {
                let mut sum = 0.0f32;
                for row in 0..hidden {
                    sum += ov[[row, col]];
                }
                vector[col] = sum / hidden as f32;
            }

            // Top-k: project vocab through OV, find most amplified
            let transformed = self.weights.embed.dot(&ov.t());
            let norms: Vec<f32> = (0..self.weights.vocab_size)
                .map(|i| {
                    let row = transformed.row(i);
                    row.iter().map(|x| x * x).sum::<f32>().sqrt()
                })
                .collect();

            let top_k_pairs = partial_top_k(&norms, config.top_k);
            let top_k: Vec<TopKEntry> = top_k_pairs
                .iter()
                .filter_map(|&(idx, logit)| {
                    decode_token(&self.tokenizer, idx as u32).map(|token| TopKEntry {
                        token,
                        token_id: idx as u32,
                        logit,
                    })
                })
                .collect();

            let (top_token, top_token_id, c_score) = if let Some(first) = top_k.first() {
                (first.token.clone(), first.token_id, first.logit)
            } else {
                (String::new(), 0, 0.0)
            };

            writer.write_record(&VectorRecord {
                id: format!("L{layer}_H{h}"),
                layer,
                feature: h,
                dim: vector.len(),
                vector,
                top_token,
                top_token_id,
                c_score,
                top_k,
            })?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Extract attention Q/K mean direction vectors per head for a single layer.
    ///
    /// Each Q/K head's projection is (head_dim, hidden). We store the mean row
    /// as a hidden-dim vector — the average direction this head queries/keys in
    /// hidden space. Same dimensionality as FFN vectors for uniform HNSW indexing.
    pub fn extract_attn_qk(
        &self,
        layer: usize,
        _config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, VindexError> {
        let prefix = format!("layers.{layer}.self_attn.");
        let w_q = self
            .weights
            .tensors
            .get(&format!("{prefix}q_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}q_proj.weight")))?;
        let w_k = self
            .weights
            .tensors
            .get(&format!("{prefix}k_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}k_proj.weight")))?;

        let head_dim = self.weights.arch.head_dim_for_layer(layer);
        let hidden = self.weights.hidden_size;
        let num_q_heads = w_q.shape()[0] / head_dim;
        let num_kv_heads = w_k.shape()[0] / head_dim;
        let total = num_q_heads + num_kv_heads;
        callbacks.on_layer_start(COMPONENT_ATTN_QK, layer, total);

        let mut count = 0;

        for h in 0..num_q_heads {
            callbacks.on_progress(COMPONENT_ATTN_QK, layer, h, total);
            let head_slice = w_q.slice(ndarray::s![h * head_dim..(h + 1) * head_dim, ..]);
            let mut vector = vec![0.0f32; hidden];
            for col in 0..hidden {
                let mut sum = 0.0f32;
                for row in 0..head_dim {
                    sum += head_slice[[row, col]];
                }
                vector[col] = sum / head_dim as f32;
            }

            writer.write_record(&VectorRecord {
                id: format!("L{layer}_Q{h}"),
                layer,
                feature: h,
                dim: vector.len(),
                vector,
                top_token: String::new(),
                top_token_id: 0,
                c_score: 0.0,
                top_k: vec![],
            })?;
            count += 1;
        }

        for h in 0..num_kv_heads {
            callbacks.on_progress(COMPONENT_ATTN_QK, layer, num_q_heads + h, total);
            let head_slice = w_k.slice(ndarray::s![h * head_dim..(h + 1) * head_dim, ..]);
            let mut vector = vec![0.0f32; hidden];
            for col in 0..hidden {
                let mut sum = 0.0f32;
                for row in 0..head_dim {
                    sum += head_slice[[row, col]];
                }
                vector[col] = sum / head_dim as f32;
            }

            writer.write_record(&VectorRecord {
                id: format!("L{layer}_K{h}"),
                layer,
                feature: h,
                dim: vector.len(),
                vector,
                top_token: String::new(),
                top_token_id: 0,
                c_score: 0.0,
                top_k: vec![],
            })?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::SilentExtractCallbacks;
    use super::*;
    use crate::walker::test_fixture::create_mock_model;
    use larql_models::VectorFileHeader;

    fn fixture(slug: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("larql_vex_attn_{slug}"));
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);
        dir
    }

    fn cleanup(dir: &std::path::Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    fn make_writer(path: &std::path::Path) -> VectorWriter {
        let mut w = VectorWriter::create(path).unwrap();
        w.write_header(&VectorFileHeader {
            _header: true,
            component: "attn".into(),
            model: "test/mock".into(),
            dimension: 4,
            extraction_date: "2026-05-09".into(),
        })
        .unwrap();
        w
    }

    #[test]
    fn extract_attn_ov_writes_per_head_records() {
        let dir = fixture("extract_ov");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        let cfg = ExtractConfig {
            components: vec![COMPONENT_ATTN_OV.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let path = dir.join("ov.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        let n = extractor.extract_attn_ov(0, &cfg, &mut w, &mut cb).unwrap();
        assert_eq!(n, 2);
        cleanup(&dir);
    }

    #[test]
    fn extract_attn_qk_writes_per_head_records() {
        let dir = fixture("extract_qk");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        let cfg = ExtractConfig {
            components: vec![COMPONENT_ATTN_QK.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let path = dir.join("qk.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        let n = extractor.extract_attn_qk(0, &cfg, &mut w, &mut cb).unwrap();
        assert!(n > 0);
        cleanup(&dir);
    }

    #[test]
    fn extract_attn_ov_missing_tensor_errors() {
        let dir = fixture("missing_ov");
        let mut extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        extractor
            .weights
            .tensors
            .remove("layers.0.self_attn.v_proj.weight");
        let cfg = ExtractConfig {
            components: vec![COMPONENT_ATTN_OV.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let path = dir.join("ov.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        match extractor.extract_attn_ov(0, &cfg, &mut w, &mut cb) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("v_proj.weight"), "msg: {msg}");
            }
            other => panic!("expected MissingTensor; got {other:?}"),
        }
        cleanup(&dir);
    }

    #[test]
    fn extract_attn_qk_missing_tensor_errors() {
        let dir = fixture("missing_qk");
        let mut extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        extractor
            .weights
            .tensors
            .remove("layers.0.self_attn.q_proj.weight");
        let cfg = ExtractConfig {
            components: vec![COMPONENT_ATTN_QK.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let path = dir.join("qk.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        match extractor.extract_attn_qk(0, &cfg, &mut w, &mut cb) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("q_proj.weight"), "msg: {msg}");
            }
            other => panic!("expected MissingTensor; got {other:?}"),
        }
        cleanup(&dir);
    }
}
