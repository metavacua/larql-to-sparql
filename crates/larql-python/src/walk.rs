//! WalkModel — mmap-backed model weights + vindex for walk FFN inference.
//!
//! Weight files are mmap'd, not loaded to heap. Array2 views point directly
//! at mmap'd memory. Only the pages touched during inference are paged in.
//! Peak RSS: ~one layer of weights at a time (OS manages page eviction).

use std::collections::HashMap;
use std::path::Path;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use ndarray::Array2;

use larql_vindex::{
    VectorIndex, SilentLoadCallbacks,
    load_vindex_config, load_vindex_tokenizer, tokenizers,
};
use larql_inference::{ModelWeights, WalkFfn, predict_with_ffn};
use larql_inference::ffn::FfnBackend;

use crate::trace_py;

/// Mmap'd weight file — kept alive so Array2 views remain valid.
struct WeightMmap {
    _file: std::fs::File,
    mmap: memmap2::Mmap,
}

/// Create ModelWeights backed by mmap'd files.
/// Tensors are Array2<f32> created from mmap'd memory.
/// For f32 vindexes: zero-copy (pointer into mmap).
/// For f16 vindexes: decoded to f32 (one allocation per tensor).
fn load_mmap_weights(dir: &Path) -> Result<(ModelWeights, Vec<WeightMmap>), String> {
    let config = load_vindex_config(dir).map_err(|e| e.to_string())?;

    if !config.has_model_weights {
        return Err("No model weights. Extract with --level all".into());
    }

    let model_cfg = config.model_config.as_ref()
        .ok_or("Missing model_config in index.json")?;

    let arch_json = serde_json::json!({
        "model_type": model_cfg.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_layers,
        "intermediate_size": config.intermediate_size,
        "head_dim": model_cfg.head_dim,
        "num_attention_heads": model_cfg.num_q_heads,
        "num_key_value_heads": model_cfg.num_kv_heads,
        "rope_theta": model_cfg.rope_base,
        "sliding_window": model_cfg.sliding_window,
        "vocab_size": config.vocab_size,
    });
    let arch = larql_models::detect_from_json(&arch_json);

    // Mmap weight files
    let mut mmaps: Vec<WeightMmap> = Vec::new();
    let mut mmap_index: HashMap<String, usize> = HashMap::new();

    let weight_files = ["attn_weights.bin", "up_weights.bin", "down_weights.bin", "norms.bin", "lm_head.bin"];
    for fname in &weight_files {
        let path = dir.join(fname);
        if path.exists() {
            let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
            mmap_index.insert(fname.to_string(), mmaps.len());
            mmaps.push(WeightMmap { _file: file, mmap });
        }
    }

    // Mmap embeddings
    let embed_file = std::fs::File::open(dir.join("embeddings.bin")).map_err(|e| e.to_string())?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file) }.map_err(|e| e.to_string())?;
    let embed_idx = mmaps.len();
    mmaps.push(WeightMmap { _file: embed_file, mmap: embed_mmap });

    // Mmap gate_vectors
    let gate_file = std::fs::File::open(dir.join("gate_vectors.bin")).map_err(|e| e.to_string())?;
    let gate_mmap = unsafe { memmap2::Mmap::map(&gate_file) }.map_err(|e| e.to_string())?;
    let gate_idx = mmaps.len();
    mmaps.push(WeightMmap { _file: gate_file, mmap: gate_mmap });

    // Read manifest
    let manifest_text = std::fs::read_to_string(dir.join("weight_manifest.json"))
        .map_err(|e| e.to_string())?;

    #[derive(serde::Deserialize)]
    struct Entry { key: String, kind: String, shape: Vec<usize>, offset: u64, length: u64, #[serde(default)] file: String }
    let entries: Vec<Entry> = serde_json::from_str(&manifest_text).map_err(|e| e.to_string())?;

    let is_f32 = config.dtype == larql_vindex::StorageDtype::F32;

    // Build tensors from mmap'd memory
    let mut tensors: HashMap<String, larql_models::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut lm_head_arr: Option<larql_models::WeightArray> = None;

    for entry in &entries {
        let fname = if entry.file.is_empty() { "model_weights.bin" } else { &entry.file };
        let mmap_idx = match mmap_index.get(fname) {
            Some(idx) => *idx,
            None => continue,
        };
        let mmap_data = &mmaps[mmap_idx].mmap;

        let offset = entry.offset as usize;
        let length = entry.length as usize;
        if offset + length > mmap_data.len() { continue; }

        let raw = &mmap_data[offset..offset + length];

        match entry.kind.as_str() {
            "tensor" if entry.shape.len() == 2 => {
                let (rows, cols) = (entry.shape[0], entry.shape[1]);
                let arr = if is_f32 {
                    // Zero-copy: create Array2 from a Vec that points at mmap memory.
                    //
                    // SAFETY:
                    // - The mmap is held alive in `_mmaps` for the lifetime of PyWalkModel
                    // - The pointer is aligned (f32, mmap page-aligned)
                    // - len and capacity are correct (checked by offset+length bounds above)
                    // - We use a leaked Vec (mem::forget prevents dealloc) so ndarray
                    //   won't free the mmap'd memory when the Array2 is dropped
                    // - The PyWalkModel drops _mmaps AFTER weights, so the memory is valid
                    //   for the entire lifetime of the arrays
                    let count = rows * cols;
                    let ptr = raw.as_ptr() as *mut f32;
                    let vec = unsafe { Vec::from_raw_parts(ptr, count, count) };
                    let arr = Array2::from_shape_vec((rows, cols), vec)
                        .map_err(|e| e.to_string())?.into_shared();
                    // Leak an extra Arc ref to prevent the Vec from being freed
                    // when the ArcArray2 drops — the mmap owns this memory
                    std::mem::forget(arr.clone());
                    arr
                } else {
                    let floats = larql_vindex::config::dtype::decode_floats(raw, config.dtype);
                    Array2::from_shape_vec((rows, cols), floats)
                        .map_err(|e| e.to_string())?.into_shared()
                };

                if entry.key == "lm_head.weight" {
                    lm_head_arr = Some(arr);
                } else {
                    tensors.insert(entry.key.clone(), arr);
                }
            }
            "vector" => {
                // Vectors are small (norms) — copy is fine
                let floats = if is_f32 {
                    unsafe {
                        std::slice::from_raw_parts(raw.as_ptr() as *const f32, entry.shape[0])
                    }.to_vec()
                } else {
                    larql_vindex::config::dtype::decode_floats(raw, config.dtype)
                };
                vectors.insert(entry.key.clone(), floats);
            }
            _ => {}
        }
    }

    // Embeddings from mmap — zero-copy for f32
    let embed_data = &mmaps[embed_idx].mmap;
    let embed = if is_f32 {
        let count = config.vocab_size * config.hidden_size;
        let ptr = embed_data.as_ptr() as *mut f32;
        let vec = unsafe { Vec::from_raw_parts(ptr, count, count) };
        let arr = Array2::from_shape_vec((config.vocab_size, config.hidden_size), vec)
            .map_err(|e| e.to_string())?.into_shared();
        std::mem::forget(arr.clone());
        arr
    } else {
        let floats = larql_vindex::config::dtype::decode_floats(embed_data, config.dtype);
        Array2::from_shape_vec((config.vocab_size, config.hidden_size), floats)
            .map_err(|e| e.to_string())?.into_shared()
    };

    // Gate vectors from mmap — zero-copy for f32
    let gate_data = &mmaps[gate_idx].mmap;
    let bpf = larql_vindex::config::dtype::bytes_per_float(config.dtype);
    for info in &config.layers {
        let float_offset = info.offset as usize / bpf;
        let float_count = info.num_features * config.hidden_size;

        let gate_arr = if is_f32 {
            let ptr = unsafe { (gate_data.as_ptr() as *const f32).add(float_offset) as *mut f32 };
            if float_offset + float_count > gate_data.len() / 4 { continue; }
            let vec = unsafe { Vec::from_raw_parts(ptr, float_count, float_count) };
            let arr = Array2::from_shape_vec((info.num_features, config.hidden_size), vec)
                .map_err(|e| e.to_string())?.into_shared();
            std::mem::forget(arr.clone());
            arr
        } else {
            let byte_offset = info.offset as usize;
            let byte_length = info.length as usize;
            if byte_offset + byte_length > gate_data.len() { continue; }
            let floats = larql_vindex::config::dtype::decode_floats(
                &gate_data[byte_offset..byte_offset + byte_length], config.dtype);
            Array2::from_shape_vec((info.num_features, config.hidden_size), floats)
                .map_err(|e| e.to_string())?.into_shared()
        };
        tensors.insert(arch.ffn_gate_key(info.layer), gate_arr);
    }

    let lm_head = lm_head_arr.unwrap_or_else(|| embed.clone());

    let weights = ModelWeights {
        tensors, vectors, raw_bytes: std::collections::HashMap::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed, lm_head,
        num_layers: config.num_layers,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        head_dim: model_cfg.head_dim,
        num_q_heads: model_cfg.num_q_heads,
        num_kv_heads: model_cfg.num_kv_heads,
        rope_base: model_cfg.rope_base,
        arch,
    };

    Ok((weights, mmaps))
}

// ── InferState: lazy-loaded mmap'd weights for vindex.infer() ──

/// Mmap'd model weights, reusable across infer() calls.
/// Created lazily on first infer(), held by PyVindex.
pub struct InferState {
    pub weights: ModelWeights,
    _mmaps: Vec<WeightMmap>,
}

impl InferState {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let (weights, mmaps) = load_mmap_weights(dir)?;
        Ok(Self { weights, _mmaps: mmaps })
    }
}

// ── Python class ──

#[pyclass(name = "WalkModel", unsendable)]
pub struct PyWalkModel {
    weights: ModelWeights,
    index: VectorIndex,
    tokenizer: tokenizers::Tokenizer,
    top_k: usize,
    path: String,
    // Hold mmaps alive — weight arrays reference this memory
    _mmaps: Vec<WeightMmap>,
}

#[pymethods]
impl PyWalkModel {
    /// Load a walk model from a vindex directory.
    ///
    /// Weight files are mmap'd, not loaded to heap. Only the pages
    /// touched during inference are paged into RSS by the OS.
    #[new]
    #[pyo3(signature = (path, top_k=8192))]
    fn new(path: &str, top_k: usize) -> PyResult<Self> {
        let dir = std::path::Path::new(path);

        let mut load_cb = SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir, &mut load_cb)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let (weights, mmaps) = load_mmap_weights(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        let tokenizer = load_vindex_tokenizer(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(Self { weights, index, tokenizer, top_k, path: path.to_string(), _mmaps: mmaps })
    }

    /// Run full forward pass with walk FFN. Returns [(token, probability)].
    #[pyo3(signature = (prompt, top_k_predictions=5))]
    fn predict(&self, prompt: &str, top_k_predictions: usize) -> PyResult<Vec<(String, f64)>> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let result = predict_with_ffn(
            &self.weights, &self.tokenizer, &token_ids, top_k_predictions, &walk_ffn
        );

        Ok(result.predictions)
    }

    /// Run walk FFN for a single layer.
    ///
    /// Accepts raw f32 bytes (from MLX memoryview), returns raw f32 bytes.
    /// No numpy: MLX → bytes → Rust → bytes → MLX.
    fn ffn_layer<'py>(
        &self, py: Python<'py>, layer: usize, x_bytes: &[u8], seq_len: usize
    ) -> PyResult<Bound<'py, PyBytes>> {
        let hidden = self.weights.hidden_size;
        let expected = seq_len * hidden * 4;
        if x_bytes.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected {} bytes ({}x{}xf32), got {}", expected, seq_len, hidden, x_bytes.len())
            ));
        }

        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(x_bytes.as_ptr() as *const f32, seq_len * hidden)
        };
        let x_arr = ndarray::ArrayView2::from_shape((seq_len, hidden), floats)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let output = walk_ffn.forward(layer, &x_arr.to_owned());

        let out_slice = output.as_slice().unwrap();
        let out_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(out_slice.as_ptr() as *const u8, out_slice.len() * 4)
        };
        Ok(PyBytes::new(py, out_bytes))
    }

    /// Feature selection only — returns indices for MLX sparse matmul.
    ///
    /// Runs gate KNN on the vindex for each sequence position, returns the
    /// union of top-K feature indices. MLX uses these to gather rows and
    /// do the matmul on Metal GPU.
    ///
    /// Args:
    ///     layer: layer index
    ///     x_bytes: raw f32 bytes (seq_len × hidden) from MLX
    ///     seq_len: number of sequence positions
    ///     top_k: features to select per position (default: self.top_k)
    ///
    /// Returns:
    ///     List of feature indices (sorted, deduplicated union across positions)
    #[pyo3(signature = (layer, x_bytes, seq_len, top_k=None))]
    fn gate_select(
        &self, layer: usize, x_bytes: &[u8], seq_len: usize, top_k: Option<usize>,
    ) -> PyResult<Vec<usize>> {
        let hidden = self.weights.hidden_size;
        let expected = seq_len * hidden * 4;
        if x_bytes.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected {} bytes ({}x{}xf32), got {}", expected, seq_len, hidden, x_bytes.len())
            ));
        }

        let k = top_k.unwrap_or(self.top_k);
        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(x_bytes.as_ptr() as *const f32, seq_len * hidden)
        };

        // Collect features across all positions
        let mut seen = std::collections::HashSet::new();
        for s in 0..seq_len {
            let row = &floats[s * hidden..(s + 1) * hidden];
            let arr = ndarray::Array1::from_vec(row.to_vec());
            let hits = self.index.gate_knn(layer, &arr, k);
            for (idx, _score) in hits {
                seen.insert(idx);
            }
        }

        let mut indices: Vec<usize> = seen.into_iter().collect();
        indices.sort_unstable();
        Ok(indices)
    }

    /// Feature selection returning indices and gate scores.
    ///
    /// Like gate_select but also returns the max gate score per feature
    /// (useful for debugging / weighted sparse FFN).
    #[pyo3(signature = (layer, x_bytes, seq_len, top_k=None))]
    fn gate_select_scored(
        &self, layer: usize, x_bytes: &[u8], seq_len: usize, top_k: Option<usize>,
    ) -> PyResult<(Vec<usize>, Vec<f32>)> {
        let hidden = self.weights.hidden_size;
        let expected = seq_len * hidden * 4;
        if x_bytes.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Expected {} bytes ({}x{}xf32), got {}", expected, seq_len, hidden, x_bytes.len())
            ));
        }

        let k = top_k.unwrap_or(self.top_k);
        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(x_bytes.as_ptr() as *const f32, seq_len * hidden)
        };

        let mut best: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        for s in 0..seq_len {
            let row = &floats[s * hidden..(s + 1) * hidden];
            let arr = ndarray::Array1::from_vec(row.to_vec());
            let hits = self.index.gate_knn(layer, &arr, k);
            for (idx, score) in hits {
                let entry = best.entry(idx).or_insert(0.0f32);
                if score.abs() > entry.abs() {
                    *entry = score;
                }
            }
        }

        let mut pairs: Vec<(usize, f32)> = best.into_iter().collect();
        pairs.sort_unstable_by_key(|(idx, _)| *idx);
        let indices = pairs.iter().map(|(i, _)| *i).collect();
        let scores = pairs.iter().map(|(_, s)| *s).collect();
        Ok((indices, scores))
    }

    #[getter]
    fn num_layers(&self) -> usize { self.weights.num_layers }

    #[getter]
    fn hidden_size(&self) -> usize { self.weights.hidden_size }

    #[getter]
    fn intermediate_size(&self) -> usize { self.weights.intermediate_size }

    #[getter]
    fn top_k(&self) -> usize { self.top_k }

    /// Capture a complete residual stream trace.
    ///
    /// Runs a full forward pass, recording the residual, attn_delta, and ffn_delta
    /// at every layer. Returns a ResidualTrace object.
    ///
    /// Args:
    ///     prompt: Input text
    ///     positions: "last" (default) or "all"
    ///
    /// Example:
    ///     t = walk_model.trace("The capital of France is")
    ///     t.answer_trajectory("Paris")
    #[pyo3(signature = (prompt, positions="last"))]
    fn trace(&self, prompt: &str, positions: &str) -> PyResult<trace_py::PyResidualTrace> {
        trace_py::capture_trace(&self.weights, &self.tokenizer, prompt, positions)
    }

    fn __repr__(&self) -> String {
        format!(
            "WalkModel(path='{}', layers={}, hidden={}, top_k={})",
            self.path, self.weights.num_layers, self.weights.hidden_size, self.top_k
        )
    }
}
