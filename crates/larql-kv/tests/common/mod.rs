//! Shared fixtures for the MOSS parity tests (`docs/tts-funnel.md`):
//! readers for the step-0 reference dump's flat-binary export
//! (`moss_parity_dump.py --export-bin`) and comparison helpers.

#![allow(dead_code)] // each test binary uses a subset

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;

pub const ENV_MODEL_DIR: &str = "MOSS_TTS_REALTIME_DIR";
pub const ENV_BIN_DIR: &str = "MOSS_PARITY_BIN_DIR";

pub struct BinFixtures {
    shapes: HashMap<String, Vec<usize>>,
    dir: PathBuf,
}

impl BinFixtures {
    pub fn open_from_env() -> Self {
        let bin_dir = std::env::var(ENV_BIN_DIR)
            .unwrap_or_else(|_| panic!("set {ENV_BIN_DIR} to the parity-dump bin directory"));
        Self::open(Path::new(&bin_dir))
    }

    pub fn open(dir: &Path) -> Self {
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("manifest.json")).expect("bin manifest"),
        )
        .expect("bin manifest json");
        let shapes = manifest
            .as_object()
            .expect("manifest object")
            .iter()
            .map(|(key, meta)| {
                let shape = meta["shape"]
                    .as_array()
                    .expect("shape array")
                    .iter()
                    .map(|v| v.as_u64().expect("dim") as usize)
                    .collect();
                (key.clone(), shape)
            })
            .collect();
        Self {
            shapes,
            dir: dir.to_path_buf(),
        }
    }

    pub fn shape(&self, key: &str) -> &[usize] {
        self.shapes
            .get(key)
            .unwrap_or_else(|| panic!("no fixture {key}"))
    }

    fn f32_values(&self, key: &str) -> Vec<f32> {
        let bytes = std::fs::read(self.dir.join(format!("{key}.bin"))).expect(key);
        bytes
            .as_chunks::<{ std::mem::size_of::<f32>() }>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    pub fn f32_matrix(&self, key: &str) -> Array2<f32> {
        let shape = self.shape(key).to_vec();
        assert_eq!(shape.len(), 2, "{key} is not rank-2");
        Array2::from_shape_vec((shape[0], shape[1]), self.f32_values(key)).expect(key)
    }

    /// Rank-3 fixture as one matrix per outer index.
    pub fn f32_rank3(&self, key: &str) -> Vec<Array2<f32>> {
        let shape = self.shape(key).to_vec();
        assert_eq!(shape.len(), 3, "{key} is not rank-3");
        let (outer, rows, cols) = (shape[0], shape[1], shape[2]);
        let values = self.f32_values(key);
        (0..outer)
            .map(|index| {
                let start = index * rows * cols;
                Array2::from_shape_vec((rows, cols), values[start..start + rows * cols].to_vec())
                    .expect(key)
            })
            .collect()
    }

    pub fn i64_matrix_as_u32(&self, key: &str) -> Array2<u32> {
        let shape = self.shape(key).to_vec();
        assert_eq!(shape.len(), 2, "{key} is not rank-2");
        let bytes = std::fs::read(self.dir.join(format!("{key}.bin"))).expect(key);
        let values: Vec<u32> = bytes
            .chunks_exact(std::mem::size_of::<i64>())
            .map(|c| {
                let v = i64::from_le_bytes(c.try_into().unwrap());
                u32::try_from(v).expect("id fits in u32")
            })
            .collect();
        Array2::from_shape_vec((shape[0], shape[1]), values).expect(key)
    }
}

pub fn model_dir_from_env() -> String {
    std::env::var(ENV_MODEL_DIR)
        .unwrap_or_else(|_| panic!("set {ENV_MODEL_DIR} to the checkpoint snapshot"))
}

pub fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

pub fn row_cosine(a: &Array2<f32>, b: &Array2<f32>, row: usize) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.row(row).iter().zip(b.row(row).iter()) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}
