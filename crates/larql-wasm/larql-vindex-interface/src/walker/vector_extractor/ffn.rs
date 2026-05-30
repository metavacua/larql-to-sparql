//! FFN (gate / up / down) per-layer extraction methods.

use larql_models::{
    TopKEntry, VectorRecord, COMPONENT_FFN_DOWN, COMPONENT_FFN_GATE, COMPONENT_FFN_UP,
};

use super::loader::VectorExtractor;
use super::types::{ExtractCallbacks, ExtractConfig};
use super::writer::VectorWriter;
use crate::error::VindexError;
use crate::walker::utils::{decode_token, partial_top_k_column};

impl VectorExtractor {
    /// Extract FFN down vectors for a single layer.
    ///
    /// The stored vector is `w_down.column(feat)` — the raw weight direction
    /// in hidden space (dim = hidden_size). The vocab projection
    /// (`embed @ w_down`) is computed only to derive top-k token metadata.
    pub fn extract_ffn_down(
        &self,
        layer: usize,
        config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, VindexError> {
        let prefix = format!("layers.{layer}.mlp.");
        let w_down = self
            .weights
            .tensors
            .get(&format!("{prefix}down_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}down_proj.weight")))?;

        let n_features = w_down.shape()[1];
        callbacks.on_layer_start(COMPONENT_FFN_DOWN, layer, n_features);

        let logits = self.weights.embed.dot(w_down);

        let progress_interval = (n_features / 20).max(1);
        let mut count = 0;

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(COMPONENT_FFN_DOWN, layer, feat_idx, n_features);
            }

            let vector: Vec<f32> = w_down.column(feat_idx).to_vec();
            let top_k_pairs = partial_top_k_column(&logits, feat_idx, config.top_k);
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
                id: format!("L{layer}_F{feat_idx}"),
                layer,
                feature: feat_idx,
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

    /// Extract FFN gate vectors for a single layer.
    pub fn extract_ffn_gate(
        &self,
        layer: usize,
        config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, VindexError> {
        let prefix = format!("layers.{layer}.mlp.");
        let w_gate = self
            .weights
            .tensors
            .get(&format!("{prefix}gate_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}gate_proj.weight")))?;

        let n_features = w_gate.shape()[0];
        callbacks.on_layer_start(COMPONENT_FFN_GATE, layer, n_features);

        let logits = self.weights.embed.dot(&w_gate.t());

        let progress_interval = (n_features / 20).max(1);
        let mut count = 0;

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(COMPONENT_FFN_GATE, layer, feat_idx, n_features);
            }

            let vector: Vec<f32> = w_gate.row(feat_idx).to_vec();
            let top_k_pairs = partial_top_k_column(&logits, feat_idx, config.top_k);
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
                id: format!("L{layer}_F{feat_idx}"),
                layer,
                feature: feat_idx,
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

    /// Extract FFN up vectors for a single layer.
    pub fn extract_ffn_up(
        &self,
        layer: usize,
        config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, VindexError> {
        let prefix = format!("layers.{layer}.mlp.");
        let w_up = self
            .weights
            .tensors
            .get(&format!("{prefix}up_proj.weight"))
            .ok_or_else(|| VindexError::MissingTensor(format!("{prefix}up_proj.weight")))?;

        let n_features = w_up.shape()[0];
        callbacks.on_layer_start(COMPONENT_FFN_UP, layer, n_features);

        let logits = self.weights.embed.dot(&w_up.t());
        let progress_interval = (n_features / 20).max(1);
        let mut count = 0;

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(COMPONENT_FFN_UP, layer, feat_idx, n_features);
            }

            let vector: Vec<f32> = w_up.row(feat_idx).to_vec();
            let top_k_pairs = partial_top_k_column(&logits, feat_idx, config.top_k);
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
                id: format!("L{layer}_F{feat_idx}"),
                layer,
                feature: feat_idx,
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
}

#[cfg(test)]
mod tests {
    use super::super::types::SilentExtractCallbacks;
    use super::*;
    use crate::walker::test_fixture::create_mock_model;
    use larql_models::VectorFileHeader;

    fn fixture(slug: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("larql_vex_ffn_{slug}"));
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
            component: "ffn_down".into(),
            model: "test/mock".into(),
            dimension: 4,
            extraction_date: "2026-05-09".into(),
        })
        .unwrap();
        w
    }

    #[test]
    fn extract_ffn_down_writes_intermediate_records() {
        let dir = fixture("extract_down");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        let cfg = ExtractConfig {
            components: vec![COMPONENT_FFN_DOWN.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let path = dir.join("down.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        let n = extractor
            .extract_ffn_down(0, &cfg, &mut w, &mut cb)
            .unwrap();
        assert_eq!(n, 4);
        cleanup(&dir);
    }

    #[test]
    fn extract_ffn_gate_writes_intermediate_records() {
        let dir = fixture("extract_gate");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        let cfg = ExtractConfig {
            components: vec![COMPONENT_FFN_GATE.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let path = dir.join("gate.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        let n = extractor
            .extract_ffn_gate(0, &cfg, &mut w, &mut cb)
            .unwrap();
        assert_eq!(n, 4);
        cleanup(&dir);
    }

    #[test]
    fn extract_ffn_up_writes_intermediate_records() {
        let dir = fixture("extract_up");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        let cfg = ExtractConfig {
            components: vec![COMPONENT_FFN_UP.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let path = dir.join("up.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        let n = extractor.extract_ffn_up(0, &cfg, &mut w, &mut cb).unwrap();
        assert_eq!(n, 4);
        cleanup(&dir);
    }

    #[test]
    fn extract_ffn_down_missing_tensor_errors() {
        let dir = fixture("missing_down");
        let mut extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        extractor
            .weights
            .tensors
            .remove("layers.0.mlp.down_proj.weight");
        let cfg = ExtractConfig {
            components: vec![COMPONENT_FFN_DOWN.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let path = dir.join("down.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        match extractor.extract_ffn_down(0, &cfg, &mut w, &mut cb) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("down_proj.weight"), "msg: {msg}");
            }
            other => panic!("expected MissingTensor; got {other:?}"),
        }
        cleanup(&dir);
    }

    #[test]
    fn extract_ffn_gate_missing_tensor_errors() {
        let dir = fixture("missing_gate");
        let mut extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        extractor
            .weights
            .tensors
            .remove("layers.0.mlp.gate_proj.weight");
        let cfg = ExtractConfig {
            components: vec![COMPONENT_FFN_GATE.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let path = dir.join("gate.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        match extractor.extract_ffn_gate(0, &cfg, &mut w, &mut cb) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("gate_proj.weight"), "msg: {msg}");
            }
            other => panic!("expected MissingTensor; got {other:?}"),
        }
        cleanup(&dir);
    }

    #[test]
    fn extract_ffn_up_missing_tensor_errors() {
        let dir = fixture("missing_up");
        let mut extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        extractor
            .weights
            .tensors
            .remove("layers.0.mlp.up_proj.weight");
        let cfg = ExtractConfig {
            components: vec![COMPONENT_FFN_UP.into()],
            layers: Some(vec![0]),
            top_k: 2,
        };
        let path = dir.join("up.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        match extractor.extract_ffn_up(0, &cfg, &mut w, &mut cb) {
            Err(VindexError::MissingTensor(msg)) => {
                assert!(msg.contains("up_proj.weight"), "msg: {msg}");
            }
            other => panic!("expected MissingTensor; got {other:?}"),
        }
        cleanup(&dir);
    }
}
