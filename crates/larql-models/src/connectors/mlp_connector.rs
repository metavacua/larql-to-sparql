//! 2-layer MLP GELU connector — weights + safetensors loader.
//!
//! Used by Granite Vision (SigLIP2 → LM). Forward pass lives in
//! `larql-compute::connectors::mlp_connector`.
//!
//! Tensor key convention (provisional — verify against a real
//! `ibm-granite/granite-3.2-*-vision` checkpoint):
//!
//! ```text
//! multi_modal_projector.linear_1.weight   (intermediate, vision_hidden)
//! multi_modal_projector.linear_1.bias     (intermediate,)
//! multi_modal_projector.linear_2.weight   (text_hidden, intermediate)
//! multi_modal_projector.linear_2.bias     (text_hidden,)
//! ```

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use ndarray::Array2;

use crate::detect::ModelError;
use crate::loading::safetensors::tensor_to_f32;

const PROJECTOR_PREFIX: &str = "multi_modal_projector.";

/// 2-layer MLP GELU connector weights.
#[derive(Debug)]
pub struct MlpConnectorWeights {
    /// First linear: (intermediate, vision_hidden). Applied as x @ W.T + b.
    pub fc1_weight: Array2<f32>,
    pub fc1_bias: Vec<f32>,
    /// Second linear: (text_hidden, intermediate). Applied as x @ W.T + b.
    pub fc2_weight: Array2<f32>,
    pub fc2_bias: Vec<f32>,
}

impl MlpConnectorWeights {
    pub fn vision_hidden(&self) -> usize {
        self.fc1_weight.ncols()
    }
    pub fn intermediate_size(&self) -> usize {
        self.fc1_weight.nrows()
    }
    pub fn text_hidden(&self) -> usize {
        self.fc2_weight.nrows()
    }
}

/// Load `MlpConnectorWeights` from a directory of safetensors files.
pub fn load_mlp_connector_from_safetensors(
    dir: impl AsRef<Path>,
) -> Result<MlpConnectorWeights, ModelError> {
    let dir = dir.as_ref();
    let mut tensors: HashMap<String, Array2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let entries = std::fs::read_dir(dir).map_err(|e| ModelError::Parse(e.to_string()))?;
    for entry in entries {
        let entry = entry.map_err(|e| ModelError::Parse(e.to_string()))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            continue;
        }
        load_one_file(&path, &mut tensors, &mut vectors)?;
    }

    let fc1_weight = tensors.remove("linear_1.weight").ok_or_else(|| {
        ModelError::Parse("missing multi_modal_projector.linear_1.weight".to_string())
    })?;
    let fc1_bias = vectors.remove("linear_1.bias").ok_or_else(|| {
        ModelError::Parse("missing multi_modal_projector.linear_1.bias".to_string())
    })?;
    let fc2_weight = tensors.remove("linear_2.weight").ok_or_else(|| {
        ModelError::Parse("missing multi_modal_projector.linear_2.weight".to_string())
    })?;
    let fc2_bias = vectors.remove("linear_2.bias").ok_or_else(|| {
        ModelError::Parse("missing multi_modal_projector.linear_2.bias".to_string())
    })?;

    Ok(MlpConnectorWeights {
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    })
}

fn load_one_file(
    path: &Path,
    tensors: &mut HashMap<String, Array2<f32>>,
    vectors: &mut HashMap<String, Vec<f32>>,
) -> Result<(), ModelError> {
    let file = std::fs::File::open(path).map_err(|e| ModelError::Parse(e.to_string()))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| ModelError::Parse(e.to_string()))?;
    let st = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| ModelError::Parse(e.to_string()))?;
    for (name, view) in st.tensors() {
        let key = match name.strip_prefix(PROJECTOR_PREFIX) {
            Some(rest) => rest.to_string(),
            None => continue,
        };
        let shape = view.shape().to_vec();
        let data = tensor_to_f32(&view)?;
        match shape.len() {
            2 => {
                let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                    .map_err(|e| ModelError::Parse(e.to_string()))?;
                tensors.insert(key, arr);
            }
            1 => {
                vectors.insert(key, data);
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: Vec<f32>) -> Vec<u8> {
        values.into_iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn write_synth_mlp_safetensors(
        dir: &std::path::Path,
        vision_hidden: usize,
        intermediate: usize,
        text_hidden: usize,
    ) {
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;

        let fc1w_bytes = f32_bytes(vec![0.1; intermediate * vision_hidden]);
        let fc1b_bytes = f32_bytes(vec![0.0; intermediate]);
        let fc2w_bytes = f32_bytes(vec![0.1; text_hidden * intermediate]);
        let fc2b_bytes = f32_bytes(vec![0.0; text_hidden]);

        let fc1w =
            TensorView::new(Dtype::F32, vec![intermediate, vision_hidden], &fc1w_bytes).unwrap();
        let fc1b = TensorView::new(Dtype::F32, vec![intermediate], &fc1b_bytes).unwrap();
        let fc2w =
            TensorView::new(Dtype::F32, vec![text_hidden, intermediate], &fc2w_bytes).unwrap();
        let fc2b = TensorView::new(Dtype::F32, vec![text_hidden], &fc2b_bytes).unwrap();

        let pairs: Vec<(&str, &TensorView<'_>)> = vec![
            ("multi_modal_projector.linear_1.weight", &fc1w),
            ("multi_modal_projector.linear_1.bias", &fc1b),
            ("multi_modal_projector.linear_2.weight", &fc2w),
            ("multi_modal_projector.linear_2.bias", &fc2b),
        ];
        serialize_to_file(pairs, None, &dir.join("model.safetensors")).expect("write synth mlp");
    }

    #[test]
    fn load_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        write_synth_mlp_safetensors(tmp.path(), 8, 16, 12);
        let w = load_mlp_connector_from_safetensors(tmp.path()).expect("load");
        assert_eq!(w.fc1_weight.shape(), &[16, 8]);
        assert_eq!(w.fc1_bias.len(), 16);
        assert_eq!(w.fc2_weight.shape(), &[12, 16]);
        assert_eq!(w.fc2_bias.len(), 12);
        assert_eq!(w.vision_hidden(), 8);
        assert_eq!(w.intermediate_size(), 16);
        assert_eq!(w.text_hidden(), 12);
    }

    #[test]
    fn errors_on_missing_dir() {
        let err = load_mlp_connector_from_safetensors("/nonexistent/xyz").expect_err("should fail");
        assert!(!format!("{err:?}").is_empty());
    }

    #[test]
    fn errors_on_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let err =
            load_mlp_connector_from_safetensors(tmp.path()).expect_err("should fail on empty dir");
        let msg = format!("{err:?}");
        assert!(msg.contains("linear_1") || msg.contains("missing"));
    }

    #[test]
    fn errors_when_fc1_weight_missing() {
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;
        let tmp = tempfile::tempdir().unwrap();
        let b = f32_bytes(vec![0.0; 16]);
        let view = TensorView::new(Dtype::F32, vec![16], &b).unwrap();
        let pair = ("multi_modal_projector.linear_1.bias", &view);
        serialize_to_file([pair], None, &tmp.path().join("model.safetensors")).expect("write");
        let err = load_mlp_connector_from_safetensors(tmp.path()).expect_err("missing fc1 weight");
        assert!(format!("{err:?}").contains("linear_1.weight"));
    }

    #[test]
    fn errors_when_fc2_bias_missing() {
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;
        let tmp = tempfile::tempdir().unwrap();
        let fc1w_b = f32_bytes(vec![0.0; 16 * 8]);
        let fc1b_b = f32_bytes(vec![0.0; 16]);
        let fc2w_b = f32_bytes(vec![0.0; 12 * 16]);
        let fc1w = TensorView::new(Dtype::F32, vec![16, 8], &fc1w_b).unwrap();
        let fc1b = TensorView::new(Dtype::F32, vec![16], &fc1b_b).unwrap();
        let fc2w = TensorView::new(Dtype::F32, vec![12, 16], &fc2w_b).unwrap();
        let pairs: Vec<(&str, &TensorView<'_>)> = vec![
            ("multi_modal_projector.linear_1.weight", &fc1w),
            ("multi_modal_projector.linear_1.bias", &fc1b),
            ("multi_modal_projector.linear_2.weight", &fc2w),
        ];
        serialize_to_file(pairs, None, &tmp.path().join("model.safetensors")).expect("write");
        let err = load_mlp_connector_from_safetensors(tmp.path()).expect_err("missing fc2 bias");
        assert!(format!("{err:?}").contains("linear_2.bias"));
    }

    #[test]
    fn skips_non_projector_tensors() {
        let tmp = tempfile::tempdir().unwrap();
        write_synth_mlp_safetensors(tmp.path(), 8, 16, 12);
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;
        let extra_b = f32_bytes(vec![0.0; 24]);
        let extra = TensorView::new(Dtype::F32, vec![4, 6], &extra_b).unwrap();
        let pairs: Vec<(&str, &TensorView<'_>)> = vec![("language_model.embed.weight", &extra)];
        serialize_to_file(pairs, None, &tmp.path().join("extra.safetensors")).unwrap();
        let w = load_mlp_connector_from_safetensors(tmp.path()).expect("load");
        assert_eq!(w.fc1_weight.shape(), &[16, 8]);
    }
}
