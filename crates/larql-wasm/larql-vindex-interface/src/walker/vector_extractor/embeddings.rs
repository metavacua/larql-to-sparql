//! Embeddings extraction — one record per vocab token.

use larql_models::{VectorRecord, COMPONENT_EMBEDDINGS};

use super::loader::VectorExtractor;
use super::types::{ExtractCallbacks, ExtractConfig};
use super::writer::VectorWriter;
use crate::error::VindexError;
use crate::walker::utils::decode_token;

impl VectorExtractor {
    /// Extract embedding vectors — one per vocab token.
    pub fn extract_embeddings(
        &self,
        _config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, VindexError> {
        let vocab_size = self.weights.vocab_size;
        callbacks.on_layer_start(COMPONENT_EMBEDDINGS, 0, vocab_size);

        let progress_interval = (vocab_size / 20).max(1);
        let mut count = 0;

        for tok_id in 0..vocab_size {
            if tok_id % progress_interval == 0 {
                callbacks.on_progress(COMPONENT_EMBEDDINGS, 0, tok_id, vocab_size);
            }

            let vector: Vec<f32> = self.weights.embed.row(tok_id).to_vec();
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

            let token = decode_token(&self.tokenizer, tok_id as u32).unwrap_or_default();

            writer.write_record(&VectorRecord {
                id: format!("T{tok_id}"),
                layer: 0,
                feature: tok_id,
                dim: vector.len(),
                vector,
                top_token: token,
                top_token_id: tok_id as u32,
                c_score: norm,
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
        let dir = std::env::temp_dir().join(format!("larql_vex_emb_{slug}"));
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
            component: "embeddings".into(),
            model: "test/mock".into(),
            dimension: 4,
            extraction_date: "2026-05-09".into(),
        })
        .unwrap();
        w
    }

    #[test]
    fn extract_embeddings_writes_vocab_records() {
        let dir = fixture("extract_emb");
        let extractor = VectorExtractor::load(dir.to_str().unwrap()).unwrap();
        let cfg = ExtractConfig {
            components: vec![COMPONENT_EMBEDDINGS.into()],
            layers: None,
            top_k: 2,
        };
        let path = dir.join("emb.jsonl");
        let mut w = make_writer(&path);
        let mut cb = SilentExtractCallbacks;
        let n = extractor.extract_embeddings(&cfg, &mut w, &mut cb).unwrap();
        assert_eq!(n, 16);
        cleanup(&dir);
    }
}
