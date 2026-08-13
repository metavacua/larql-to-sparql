//! Gemma 3 vision connector — weights + safetensors loader.
//!
//! The connector ("multi_modal_projector" in HF nomenclature) maps the
//! SigLIP encoder's `(num_patches, vision_hidden) = (4096, 1152)` output
//! into the Gemma 3 LM's `(num_soft_tokens, text_hidden) = (256, 2560)`
//! input. See HuggingFace's `Gemma3MultiModalProjector` in
//! `transformers/models/gemma3/modular_gemma3.py` for the reference impl.
//!
//! Two tensors total:
//!   - `multi_modal_projector.mm_input_projection_weight` — (vision_hidden,
//!     text_hidden) = (1152, 2560). The matmul convention is `x @ W`
//!     (not `x @ W.T`); don't transpose.
//!   - `multi_modal_projector.mm_soft_emb_norm.weight` — (vision_hidden,)
//!     = (1152,). Gemma RMSNorm scale; runtime weight is `(1.0 + saved)`,
//!     consistent with other Gemma 3 norms. No bias (RMSNorm).
//!
//! Phase 1c scope: weights + loader. Forward pass lives in
//! `larql-compute::connectors::gemma3`.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use ndarray::Array2;

use crate::detect::ModelError;
use crate::loading::safetensors::tensor_to_f32;

const PROJECTOR_PREFIX: &str = "multi_modal_projector.";

/// Gemma 3 vision-to-LM projector weights.
#[derive(Debug)]
pub struct ProjectorWeights {
    /// Linear projection: `(vision_hidden, text_hidden) = (1152, 2560)`
    /// for Gemma 3 4B. Applied as `x @ W`, not `x @ W.T`.
    pub input_projection: Array2<f32>,
    /// RMSNorm scale on the vision_hidden axis. Runtime weight is
    /// `1.0 + soft_emb_norm[j]`, matching Gemma's other norms.
    pub soft_emb_norm: Vec<f32>,
}

impl ProjectorWeights {
    pub fn vision_hidden(&self) -> usize {
        self.input_projection.nrows()
    }
    pub fn text_hidden(&self) -> usize {
        self.input_projection.ncols()
    }
}

/// Load `ProjectorWeights` from a directory of safetensors files.
///
/// Scans every `*.safetensors` in `dir`, picks tensors whose key starts
/// with `multi_modal_projector.`, and assembles the projector. Errors
/// if either required tensor is missing — there is no fall-through.
pub fn load_projector_from_safetensors(
    dir: impl AsRef<Path>,
) -> Result<ProjectorWeights, ModelError> {
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

    let input_projection = tensors
        .remove("mm_input_projection_weight")
        .ok_or_else(|| {
            ModelError::Parse(
                "missing multi_modal_projector.mm_input_projection_weight".to_string(),
            )
        })?;
    let soft_emb_norm = vectors.remove("mm_soft_emb_norm.weight").ok_or_else(|| {
        ModelError::Parse("missing multi_modal_projector.mm_soft_emb_norm.weight".to_string())
    })?;

    Ok(ProjectorWeights {
        input_projection,
        soft_emb_norm,
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

    #[test]
    fn errors_on_missing_dir() {
        let err = load_projector_from_safetensors("/nonexistent/xyz").expect_err("should fail");
        assert!(!format!("{err:?}").is_empty());
    }

    #[test]
    fn errors_on_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let err =
            load_projector_from_safetensors(tmp.path()).expect_err("should fail on empty dir");
        let msg = format!("{err:?}");
        assert!(msg.contains("multi_modal_projector") || msg.contains("missing"));
    }

    #[test]
    fn weight_struct_geometry_helpers() {
        let w = ProjectorWeights {
            input_projection: Array2::<f32>::zeros((1152, 2560)),
            soft_emb_norm: vec![0.0; 1152],
        };
        assert_eq!(w.vision_hidden(), 1152);
        assert_eq!(w.text_hidden(), 2560);
    }

    // ── Synthetic-fixture loader tests ─────────────────────────────────────
    //
    // Same pattern as encoders/siglip.rs: write a tiny safetensors with
    // the projector's two tensors via safetensors::serialize_to_file,
    // then load. Covers load_one_file + assemble paths without needing
    // the real Gemma 3 checkpoint.

    fn f32_bytes(values: Vec<f32>) -> Vec<u8> {
        values.into_iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn write_synth_projector_safetensors(
        dir: &std::path::Path,
        vision_hidden: usize,
        text_hidden: usize,
    ) {
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;
        let proj_bytes = f32_bytes(vec![0.0; vision_hidden * text_hidden]);
        let norm_bytes = f32_bytes(vec![0.0; vision_hidden]);
        let pv =
            TensorView::new(Dtype::F32, vec![vision_hidden, text_hidden], &proj_bytes).unwrap();
        let nv = TensorView::new(Dtype::F32, vec![vision_hidden], &norm_bytes).unwrap();
        let pairs: Vec<(&str, &TensorView<'_>)> = vec![
            ("multi_modal_projector.mm_input_projection_weight", &pv),
            ("multi_modal_projector.mm_soft_emb_norm.weight", &nv),
        ];
        serialize_to_file(pairs, None, &dir.join("model.safetensors"))
            .expect("write synth projector");
    }

    #[test]
    fn load_projector_round_trip_against_synthetic_safetensors() {
        let tmp = tempfile::tempdir().unwrap();
        write_synth_projector_safetensors(tmp.path(), 8, 12);
        let w = load_projector_from_safetensors(tmp.path())
            .expect("synthetic projector should load cleanly");
        assert_eq!(w.input_projection.shape(), &[8, 12]);
        assert_eq!(w.soft_emb_norm.len(), 8);
        assert_eq!(w.vision_hidden(), 8);
        assert_eq!(w.text_hidden(), 12);
    }

    #[test]
    fn load_projector_errors_when_input_projection_missing() {
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;
        let tmp = tempfile::tempdir().unwrap();
        let norm_bytes = f32_bytes(vec![0.0; 8]);
        let view = TensorView::new(Dtype::F32, vec![8], &norm_bytes).unwrap();
        let pair = ("multi_modal_projector.mm_soft_emb_norm.weight", &view);
        serialize_to_file([pair], None, &tmp.path().join("model.safetensors"))
            .expect("write partial");
        let err = load_projector_from_safetensors(tmp.path())
            .expect_err("missing input_projection should error");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("mm_input_projection_weight"),
            "error must name the missing tensor: {msg}"
        );
    }

    #[test]
    fn load_projector_errors_when_soft_emb_norm_missing() {
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;
        let tmp = tempfile::tempdir().unwrap();
        let proj_bytes = f32_bytes(vec![0.0; 8 * 12]);
        let view = TensorView::new(Dtype::F32, vec![8, 12], &proj_bytes).unwrap();
        let pair = ("multi_modal_projector.mm_input_projection_weight", &view);
        serialize_to_file([pair], None, &tmp.path().join("model.safetensors"))
            .expect("write partial");
        let err = load_projector_from_safetensors(tmp.path())
            .expect_err("missing soft_emb_norm should error");
        let msg = format!("{err:?}");
        assert!(
            msg.contains("mm_soft_emb_norm"),
            "error must name the missing tensor: {msg}"
        );
    }

    #[test]
    fn load_projector_skips_non_safetensors_files() {
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("README.txt"), b"hello").unwrap();
        let proj_bytes = f32_bytes(vec![0.0; 8 * 12]);
        let norm_bytes = f32_bytes(vec![0.0; 8]);
        let pv = TensorView::new(Dtype::F32, vec![8, 12], &proj_bytes).unwrap();
        let nv = TensorView::new(Dtype::F32, vec![8], &norm_bytes).unwrap();
        let pairs: Vec<(&str, &TensorView<'_>)> = vec![
            ("multi_modal_projector.mm_input_projection_weight", &pv),
            ("multi_modal_projector.mm_soft_emb_norm.weight", &nv),
        ];
        serialize_to_file(pairs, None, &tmp.path().join("model.safetensors")).unwrap();
        let w = load_projector_from_safetensors(tmp.path()).expect("load");
        assert_eq!(w.input_projection.shape(), &[8, 12]);
    }

    #[test]
    fn loader_ignores_non_projector_tensors_and_unknown_ranks() {
        // Exercises the `None => continue` (non-prefixed tensor) and
        // `_ => {}` (rank-3 tensor) branches in load_one_file, which
        // real checkpoints trigger when language_model.* tensors sit
        // in the same safetensors shard as multi_modal_projector.* ones.
        use safetensors::tensor::{serialize_to_file, TensorView};
        use safetensors::Dtype;
        let tmp = tempfile::tempdir().unwrap();
        // Write a valid projector fixture.
        write_synth_projector_safetensors(tmp.path(), 8, 12);
        // Append a second shard with non-projector + rank-3 tensors.
        let extra_bytes = f32_bytes(vec![0.0; 24]);
        let r3_bytes = f32_bytes(vec![0.0; 2 * 3 * 4]);
        let extra = TensorView::new(Dtype::F32, vec![4, 6], &extra_bytes).unwrap();
        let r3 = TensorView::new(Dtype::F32, vec![2, 3, 4], &r3_bytes).unwrap();
        let pairs: Vec<(&str, &TensorView<'_>)> = vec![
            ("language_model.embed.weight", &extra),
            ("multi_modal_projector.mystery_3d_tensor", &r3),
        ];
        serialize_to_file(pairs, None, &tmp.path().join("extra.safetensors")).unwrap();
        // Load should still succeed with only the valid projector tensors.
        let w = load_projector_from_safetensors(tmp.path()).expect("load");
        assert_eq!(w.input_projection.shape(), &[8, 12]);
        assert_eq!(w.soft_emb_norm.len(), 8);
    }

    #[test]
    #[ignore = "requires google/gemma-3-4b-it in the local HF cache; NOT FOR CI"]
    fn load_real_gemma3_4b_it_projector() {
        let snap = "/Users/christopherhay/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767";
        if !std::path::Path::new(snap).exists() {
            eprintln!("snapshot not present, skipping: {snap}");
            return;
        }
        let w = load_projector_from_safetensors(snap).expect("load");
        assert_eq!(
            w.input_projection.shape(),
            &[1152, 2560],
            "(vision_hidden, text_hidden)"
        );
        assert_eq!(w.soft_emb_norm.len(), 1152);
        assert!(w.input_projection.iter().all(|v| v.is_finite()));
        assert!(w.soft_emb_norm.iter().all(|v| v.is_finite()));
    }
}
