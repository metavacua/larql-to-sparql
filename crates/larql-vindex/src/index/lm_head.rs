//! LM-head loaders + KNN.
//!
//! Loads the output projection (vocab × hidden) in one of three formats:
//!
//! - **Q4_K** (`lm_head_q4.bin`): GPU Q4 matvec, ~1 ms on Metal.
//! - **f16**: adopted from the vindex's `embeddings.bin` when that file
//!   is IEEE-half (tied-embedding Gemma / Llama). Drives Metal's
//!   `f16_gemv` shader — half the memory-bandwidth of f32 without the
//!   5.6 GB heap clone that a dequantised lm_head would need on 31B.
//! - **f32** (`lm_head.bin` or cloned from `embed`): CPU BLAS fallback.
//!
//! `lm_head_knn_backend` dispatches in the order above, using the
//! cheapest available backend path for the loaded lm_head representation.
//! Sibling to `super::walk` (FFN) and `super::attn` (attention).

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

    /// Whether Q4 lm_head is loaded (from file or synthesized from f16 embeddings).
    pub fn has_lm_head_q4(&self) -> bool {
        self.lm_head_q4_mmap.is_some() || self.lm_head_q4_synth.is_some()
    }

    /// Synthesize Q4_0 lm_head in RAM from the f16 embeddings mmap.
    /// No-op if a Q4 source already exists or preconditions are not met.
    pub fn synthesize_lm_head_q4(&mut self) {
        if self.lm_head_q4_mmap.is_some() || self.lm_head_q4_synth.is_some() { return; }
        let vocab = self.vocab_size;
        let hidden = self.hidden_size;
        if vocab == 0 || hidden == 0 || hidden % 32 != 0 { return; }
        let f16_mmap = match self.lm_head_f16_mmap.as_ref() {
            Some(m) => m.clone(),
            None => return,
        };
        let expected = vocab * hidden * 2;
        if f16_mmap.len() < expected { return; }
        let blocks_per_row = hidden / 32;
        let bytes_per_row = blocks_per_row * 18;
        let mut out = Vec::with_capacity(vocab * bytes_per_row);
        let mut row_f32 = vec![0.0f32; hidden];
        for row in 0..vocab {
            let base = row * hidden * 2;
            for i in 0..hidden {
                let off = base + i * 2;
                let bits = u16::from_le_bytes([f16_mmap[off], f16_mmap[off + 1]]);
                row_f32[i] = larql_models::quant::half::f16_to_f32(bits);
            }
            let q4 = larql_compute::cpu::q4::quantize_q4_0(&row_f32);
            out.extend_from_slice(&q4);
        }
        self.lm_head_q4_synth = Some(Arc::new(out));
    }

    /// Adopt the vindex's f16 `embeddings.bin` mmap as an f16 view of the
    /// LM head. Safe only for tied-embedding models (Gemma 2/3/4, Llama
    /// when `tie_word_embeddings=true`) — the loader is responsible for
    /// gating. Caller must have already populated `vocab_size`.
    ///
    /// When set, `lm_head_knn_backend` prefers `ComputeBackend::f16_gemv`
    /// on the mmap'd bytes, avoiding the 5.6 GB f32 clone on Gemma 4 31B.
    pub fn set_lm_head_f16_mmap(&mut self, mmap: Arc<memmap2::Mmap>) {
        self.lm_head_f16_mmap = Some(mmap);
    }

    /// Whether an f16 mmap view of the LM head is available.
    pub fn has_lm_head_f16(&self) -> bool {
        self.lm_head_f16_mmap.is_some() && self.vocab_size > 0
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

    /// KNN against lm_head via a ComputeBackend. Tries paths in order:
    ///   1. Q4 matvec on `lm_head_q4.bin` (when present and backend has q4).
    ///   2. f16 gemv on the mmap'd embeddings (tied-embed models only).
    ///   3. f32 BLAS fallback via `lm_head_knn`.
    pub fn lm_head_knn_backend(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(u32, f32)> {
        // 1. Q4 path — ~1 ms on Metal (mmap file or synthesized from f16 embeddings).
        if backend.has_q4() {
            let q4_bytes: Option<&[u8]> = self.lm_head_q4_mmap
                .as_ref().map(|m| m.as_ref() as &[u8])
                .or_else(|| self.lm_head_q4_synth.as_ref().map(|v| v.as_slice()));
            if let Some(q4_data) = q4_bytes {
                let vocab = self.vocab_size;
                let hidden = self.hidden_size;
                if vocab > 0 {
                    let x = query.as_slice().unwrap();
                    let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x);
                    if let Some(scores_vec) = backend.q4_matvec(
                        q4_data, &q8_x, &q8_scales, vocab, hidden,
                    ) {
                        return Self::top_k_sorted(scores_vec, top_k);
                    }
                }
            }
        }
        // 2. f16 path — tied-embed Gemma, ~2× the bandwidth of Q4 but still
        //    half of f32 and avoids a 5.6 GB heap allocation on 31B.
        if let Some(ref f16_mmap) = self.lm_head_f16_mmap {
            let vocab = self.vocab_size;
            let hidden = self.hidden_size;
            if vocab > 0 {
                let expected = vocab * hidden * 2;
                if f16_mmap.len() >= expected {
                    if let Some(x) = query.as_slice() {
                        if let Some(scores_vec) = backend.f16_gemv(
                            &f16_mmap[..expected], x, vocab, hidden,
                        ) {
                            return Self::top_k_sorted(scores_vec, top_k);
                        }
                    }
                }
            }
        }
        // 3. f32 BLAS fallback.
        self.lm_head_knn(query, top_k)
    }

    /// Sort `scores` by descending value and keep the top `top_k`. Shared
    /// by the Q4 / f16 / f32 paths above.
    fn top_k_sorted(scores: Vec<f32>, top_k: usize) -> Vec<(u32, f32)> {
        let mut indexed: Vec<(u32, f32)> = scores.into_iter().enumerate()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `top_k_sorted` is the shared reduce used by Q4 / f16 / f32 paths.
    /// Pin the contract: descending by score, capped at `top_k`.
    #[test]
    fn top_k_sorted_descending_and_capped() {
        let scores = vec![0.5f32, 0.1, 0.9, 0.3, 0.7];
        let top3 = VectorIndex::top_k_sorted(scores.clone(), 3);
        let tokens: Vec<u32> = top3.iter().map(|(t, _)| *t).collect();
        let probs: Vec<f32> = top3.iter().map(|(_, s)| *s).collect();
        assert_eq!(tokens, vec![2, 4, 0], "expect descending-by-score token order");
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);

        // top_k larger than input → no truncation, but still sorted.
        let all = VectorIndex::top_k_sorted(scores, 99);
        assert_eq!(all.len(), 5);
        let probs: Vec<f32> = all.iter().map(|(_, s)| *s).collect();
        assert!(probs.windows(2).all(|w| w[0] >= w[1]));
    }
}
