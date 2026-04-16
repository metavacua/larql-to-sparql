//! LM-head loaders + KNN.
//!
//! Loads the output projection (vocab × hidden) in either f32 or Q4
//! format. `lm_head_knn` projects a query through the LM head and
//! returns the top-K vocab tokens — the production token-prediction
//! path. Sibling to `super::walk` (FFN) and `super::attn` (attention).

use std::sync::Arc;

use crate::error::VindexError;
use crate::mmap_util::mmap_optimized;

use super::core::VectorIndex;

impl VectorIndex {
    /// Load Q4 lm_head for GPU logits (replaces CPU f32 lm_head KNN).
    pub fn load_lm_head_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("lm_head_q4.bin");
        if !path.exists() {
            return Err(VindexError::Parse("lm_head_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.lm_head_q4_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether Q4 lm_head is loaded.
    pub fn has_lm_head_q4(&self) -> bool {
        self.lm_head_q4_mmap.is_some()
    }

    // ── LM head (output projection) for vindex logits ──

    /// Load lm_head from lm_head.bin for KNN logit lookup.
    pub fn load_lm_head(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("lm_head.bin");
        if !path.exists() {
            return Err(VindexError::Parse("lm_head.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        // Detect vocab size from file size: vocab = file_bytes / (hidden_size * 4)
        let vocab = mmap.len() / (self.hidden_size * 4);
        self.vocab_size = vocab;
        self.lm_head_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether lm_head is loaded for vindex logits.
    pub fn has_lm_head(&self) -> bool {
        self.lm_head_mmap.is_some() && self.vocab_size > 0
    }

    /// KNN against lm_head via a ComputeBackend — GPU Q4 or CPU BLAS.
    ///
    /// If Q4 lm_head data and a Q4-capable backend are provided, uses Q4 matvec (~1ms).
    /// Otherwise falls back to CPU BLAS f32 (~10ms).
    pub fn lm_head_knn_backend(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(u32, f32)> {
        // Try Q4 path first
        if backend.has_q4() {
            if let Some(ref q4_mmap) = self.lm_head_q4_mmap {
                let vocab = self.vocab_size;
                let hidden = self.hidden_size;
                if vocab > 0 {
                    let x = query.as_slice().unwrap();
                    let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x);
                    if let Some(scores_vec) = backend.q4_matvec(
                        q4_mmap.as_ref(), &q8_x, &q8_scales, vocab, hidden,
                    ) {
                        let mut indexed: Vec<(u32, f32)> = scores_vec.iter().copied().enumerate()
                            .map(|(i, s)| (i as u32, s))
                            .collect();
                        let k = top_k.min(indexed.len());
                        if k > 0 && k < indexed.len() {
                            indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
                            indexed.truncate(k);
                        }
                        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        return indexed;
                    }
                }
            }
        }
        // Fallback to f32 BLAS
        self.lm_head_knn(query, top_k)
    }

    /// KNN against lm_head: find top-K tokens by dot product with query vector.
    /// Single BLAS gemv: query[1, hidden] @ lm_head[vocab, hidden]^T → [1, vocab].
    /// Then top-K selection. Returns (token_id, score) sorted by score descending.
    pub fn lm_head_knn(&self, query: &ndarray::Array1<f32>, top_k: usize) -> Vec<(u32, f32)> {
        let mmap = match self.lm_head_mmap.as_ref() {
            Some(m) => m,
            None => return vec![],
        };
        let vocab = self.vocab_size;
        let hidden = self.hidden_size;
        if vocab == 0 { return vec![]; }

        let expected = vocab * hidden * 4;
        if mmap.len() < expected { return vec![]; }

        // Zero-copy: reinterpret mmap as [vocab, hidden] f32 matrix
        let data = unsafe {
            let ptr = mmap.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, vocab * hidden)
        };
        let lm_view = ndarray::ArrayView2::from_shape((vocab, hidden), data).unwrap();

        // gemv via larql-compute: scores = query @ lm_head^T → [1, vocab]
        let hidden = self.hidden_size;
        let x = query.view().into_shape_with_order((1, hidden)).unwrap();
        let cpu = larql_compute::CpuBackend;
        use larql_compute::ComputeBackend;
        let result = cpu.matmul_transb(x, lm_view); // [1, hidden] @ [vocab, hidden]^T → [1, vocab]
        let scores = ndarray::Array1::from_vec(result.into_raw_vec_and_offset().0);

        // Top-K selection
        let mut indexed: Vec<(u32, f32)> = scores.iter().copied().enumerate()
            .map(|(i, s)| (i as u32, s))
            .collect();
        let k = top_k.min(indexed.len());
        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed
    }
}
