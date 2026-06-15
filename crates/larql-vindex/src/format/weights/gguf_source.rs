//! `WeightSource` impl backed by a streaming GGUF tensor source.
//! Each accessor dequantizes one tensor at a time (bounded RAM).

use crate::extract::streaming::tensor_io::GgufTensorSource;
use crate::format::weights::write_f32::WeightSource;

pub struct GgufWeightSource<'a> {
    pub(crate) gguf: &'a GgufTensorSource,
    pub(crate) arch: &'a dyn larql_models::ModelArchitecture,
    pub(crate) num_layers: usize,
}

impl<'a> WeightSource for GgufWeightSource<'a> {
    fn get_tensor(&self, key: &str) -> Option<(Vec<f32>, usize, usize)> {
        let arr = self.gguf.get_tensor_f32(key).ok()??;
        let rows = arr.shape()[0];
        let cols = arr.shape()[1];
        let data = arr.as_standard_layout().to_owned().into_raw_vec_and_offset().0;
        Some((data, rows, cols))
    }

    fn get_vector(&self, key: &str) -> Option<Vec<f32>> {
        self.gguf.get_vector_f32(key).ok().flatten()
    }

    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        self.arch
    }

    fn num_layers(&self) -> usize {
        self.num_layers
    }

    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        self.get_tensor("lm_head.weight")
    }

    fn vector_names(&self) -> Vec<String> {
        self.gguf.vector_names()
    }

    fn get_packed_bf16(&self, _key: &str) -> Option<Vec<u8>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::loading::gguf::{GgufFile, GgufTensor, GgufValue, GgufWriter};
    use crate::extract::streaming::tensor_io::GgufTensorSource;

    fn make_test_gguf_source() -> (tempfile::TempDir, GgufTensorSource) {
        // 2-D tensor: logical shape (2 rows, 3 cols).
        // GGUF stores dims innermost-first: dims[0]=cols=3, dims[1]=rows=2.
        // Values row-major: row0=[1,2,3], row1=[4,5,6].
        let vals_2d = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data_2d = Vec::new();
        for v in vals_2d {
            data_2d.extend_from_slice(&v.to_le_bytes());
        }

        // 1-D tensor: 3 elements.
        let vals_1d = [7.0f32, 8.0, 9.0];
        let mut data_1d = Vec::new();
        for v in vals_1d {
            data_1d.extend_from_slice(&v.to_le_bytes());
        }

        let mut w = GgufWriter::new();
        w.meta("general.architecture", GgufValue::String("llama".into()));
        // 2-D tensor — dims[3,2] means cols=3, rows=2
        w.tensor(GgufTensor {
            name: "blk.0.ffn_up.weight".into(),
            dims: vec![3, 2],
            ggml_type: 0, // F32
            data: data_2d,
        });
        // 1-D tensor
        w.tensor(GgufTensor {
            name: "blk.0.attn_norm.weight".into(),
            dims: vec![3],
            ggml_type: 0, // F32
            data: data_1d,
        });

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        w.write_to_file(&path).unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let mut src = GgufTensorSource::from_gguf(gguf, 0, 0).unwrap();
        // Override index to map our chosen normalized keys → tensor indices.
        // Tensor 0 = 2-D (blk.0.ffn_up.weight → index 0)
        // Tensor 1 = 1-D (blk.0.attn_norm.weight → index 1)
        src.index = std::collections::HashMap::from([
            ("w2d".to_string(), 0usize),
            ("n1d".to_string(), 1usize),
        ]);

        (dir, src)
    }

    #[test]
    fn gguf_weight_source_get_tensor_returns_correct_shape_and_data() {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
        }));
        let (_dir, src) = make_test_gguf_source();
        let ws = GgufWeightSource {
            gguf: &src,
            arch: &*arch,
            num_layers: 1,
        };

        // 2-D tensor: GGUF dims=[3,2] → rows=2, cols=3
        let result = ws.get_tensor("w2d");
        assert!(result.is_some(), "get_tensor('w2d') must return Some");
        let (data, rows, cols) = result.unwrap();
        assert_eq!(rows, 2, "rows must be 2");
        assert_eq!(cols, 3, "cols must be 3");
        assert_eq!(data, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], "data must be row-major");

        // Missing key → None
        assert!(ws.get_tensor("missing").is_none());
    }

    #[test]
    fn gguf_weight_source_get_vector_returns_1d_data() {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
        }));
        let (_dir, src) = make_test_gguf_source();
        let ws = GgufWeightSource {
            gguf: &src,
            arch: &*arch,
            num_layers: 1,
        };

        let result = ws.get_vector("n1d");
        assert_eq!(result, Some(vec![7.0f32, 8.0, 9.0]), "get_vector must return 1-D data");

        assert!(ws.get_vector("missing").is_none());
    }

    #[test]
    fn gguf_weight_source_get_packed_bf16_returns_none() {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
        }));
        let (_dir, src) = make_test_gguf_source();
        let ws = GgufWeightSource {
            gguf: &src,
            arch: &*arch,
            num_layers: 1,
        };

        assert!(ws.get_packed_bf16("anything").is_none());
        assert!(ws.get_packed_bf16("w2d").is_none());
    }

    #[test]
    fn gguf_weight_source_vector_names_contains_1d_key() {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
        }));
        let (_dir, src) = make_test_gguf_source();
        let ws = GgufWeightSource {
            gguf: &src,
            arch: &*arch,
            num_layers: 1,
        };

        let names = ws.vector_names();
        assert!(names.contains(&"n1d".to_string()), "vector_names must contain 'n1d'; got: {names:?}");
        assert!(!names.contains(&"w2d".to_string()), "vector_names must NOT contain 2-D key 'w2d'");
    }

    /// Fix D: verify that `get_tensor` correctly row-major-flattens a transposed FFN tensor.
    ///
    /// `blk.0.ffn_up.weight` normalises to `layers.0.mlp.up_proj.weight`.
    /// `expected_shape` returns `(intermediate_size=2, hidden_size=3)` = `(rows=2, cols=3)`.
    /// The GGUF file stores dims=[2, 3] meaning cols=2, rows=3, so the raw array
    /// has shape (3, 2) — the transpose of the HF convention — and `orient` must
    /// swap it to (2, 3) before `GgufWeightSource::get_tensor` flattens row-major.
    #[test]
    fn gguf_weight_source_get_tensor_orient_transpose_flatten() {
        // hidden_size=3, intermediate_size=2 so expected up_proj shape = (2, 3).
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 3,
            "num_hidden_layers": 1,
            "intermediate_size": 2,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 32,
        }));

        // Build GGUF with blk.0.ffn_up.weight stored as dims=[2, 3]
        // (cols=2=intermediate_size, rows=3=hidden_size) — the PyTorch-transposed layout.
        // Raw bytes are contiguous along dims[0], so the values fill
        //   row0: [1.0, 2.0], row1: [3.0, 4.0], row2: [5.0, 6.0]
        // (shape (3,2) array).  After orient-transpose to (2,3):
        //   row0: [1.0, 3.0, 5.0], row1: [2.0, 4.0, 6.0]
        // Row-major flatten: [1.0, 3.0, 5.0, 2.0, 4.0, 6.0].
        let vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut data = Vec::new();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let mut w = GgufWriter::new();
        w.meta("general.architecture", GgufValue::String("llama".into()));
        // dims=[2,3]: dims[0]=2 (cols=intermediate), dims[1]=3 (rows=hidden)
        w.tensor(GgufTensor {
            name: "blk.0.ffn_up.weight".into(),
            dims: vec![2, 3],
            ggml_type: 0, // F32
            data,
        });

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("orient.gguf");
        w.write_to_file(&path).unwrap();
        let gguf = larql_models::loading::gguf::GgufFile::open(&path).unwrap();
        // hidden_size=3, intermediate_size=2 drive expected_shape / orient.
        let src = GgufTensorSource::from_gguf(gguf, 3, 2).unwrap();

        let ws = GgufWeightSource {
            gguf: &src,
            arch: &*arch,
            num_layers: 1,
        };

        // The normalized key for blk.0.ffn_up.weight is layers.0.mlp.up_proj.weight.
        let result = ws.get_tensor("layers.0.mlp.up_proj.weight");
        assert!(result.is_some(), "get_tensor must return Some for the FFN key");
        let (data_out, rows, cols) = result.unwrap();
        assert_eq!(rows, 2, "rows must equal intermediate_size after orient");
        assert_eq!(cols, 3, "cols must equal hidden_size after orient");
        // Row-major flatten of transposed (2,3) array.
        assert_eq!(
            data_out,
            vec![1.0f32, 3.0, 5.0, 2.0, 4.0, 6.0],
            "data must be the row-major flatten of the oriented (2,3) array"
        );
    }
}
