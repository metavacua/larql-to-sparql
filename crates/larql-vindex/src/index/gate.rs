//! Gate KNN search — brute-force, batched, and HNSW.
//!
//! All gate KNN methods for VectorIndex: single-query, batched, expert-scoped,
//! score computation, HNSW integration, and top-K selection.

use ndarray::{Array1, Array2, ArrayView2};
use larql_compute::ComputeBackend;

use super::core::VectorIndex;
use super::types::*;

/// Matrix-vector multiply: view[N, hidden] × vec[hidden] → scores[N].
/// All compute goes through larql-compute.
fn gemv(view: &ArrayView2<f32>, vec: &Array1<f32>) -> Array1<f32> {
    let hidden = vec.len();
    let x = vec.view().into_shape_with_order((1, hidden)).unwrap();
    let cpu = larql_compute::CpuBackend;
    // x[1, hidden] @ view[N, hidden]^T → [1, N]
    let result = cpu.matmul_transb(x, *view);
    Array1::from_vec(result.into_raw_vec_and_offset().0)
}

/// Gate scores batch: gate[N, hidden] × x[seq, hidden]^T → [N, seq].
/// Equivalent to original gate.dot(&x.t()).
fn gate_matmul(gate: &ArrayView2<f32>, x: &ArrayView2<f32>) -> Array2<f32> {
    let cpu = larql_compute::CpuBackend;
    // gate[N, hidden] @ x[seq, hidden]^T = matmul_transb(gate, x) → [N, seq]
    cpu.matmul_transb(*gate, *x)
}

/// GPU-accelerated gate matmul for the single-position decode case.
///
/// When `x` is a single row (seq_len == 1) and the caller passes a Metal
/// backend, route the gate gemv through `f32_gemv` — the dedicated
/// row-per-simdgroup kernel that closed lm_head on the 4B. Returns
/// `None` if the gemv threshold isn't met or seq_len > 1; caller falls
/// back to `gate_matmul` (CPU BLAS).
///
/// Shape note: returns the [N, 1] column vector laid out as [N]; caller
/// wraps it into Array2 shape (N, 1) at the seam.
fn gate_gemv_gpu(
    gate: &ArrayView2<f32>,
    x: &ArrayView2<f32>,
    backend: &dyn larql_compute::ComputeBackend,
) -> Option<Array2<f32>> {
    if x.shape()[0] != 1 { return None; }
    let x_row = x.row(0);
    let x_slice = x_row.as_slice()?;
    // Force GPU dispatch regardless of the backend's flop_threshold —
    // per-layer gate gemvs are ~50–200 M FLOPs, below the default 500 M
    // threshold that protects tiny one-off gemvs. At 34/60 layers × every
    // decode token the aggregated saving is real even if each call alone
    // would be dispatch-bound.
    let scores = backend.f32_gemv_force(*gate, x_slice)?;
    Array2::from_shape_vec((gate.shape()[0], 1), scores).ok()
}

/// Resolved gate matrix data — owned f32 with feature count.
struct GateData {
    data: Vec<f32>,
    num_features: usize,
}

impl GateData {
    fn view(&self, hidden_size: usize) -> ArrayView2<'_, f32> {
        ArrayView2::from_shape((self.num_features, hidden_size), &self.data).unwrap()
    }
}

/// Gate KNN methods for VectorIndex.
impl VectorIndex {
    /// Cap the number of decoded f16 gate layers held in
    /// `f16_decode_cache`. Call with 0 for unlimited (default); non-zero
    /// enables LRU eviction on the next insert that would exceed the cap.
    ///
    /// Typical use: `larql serve --max-gate-cache-layers N` to bound a
    /// long-running server's RSS. A 31B f16 gate table decodes to ~433 MB
    /// per layer, so `--max-gate-cache-layers 4` caps decoded gates at
    /// ~1.7 GB (at the cost of repeated decode on evicted layers).
    pub fn set_gate_cache_max_layers(&self, max_layers: usize) {
        self.gate_cache_max_layers
            .store(max_layers, std::sync::atomic::Ordering::Relaxed);
        // Shrink eagerly if the new cap is below the current cache size.
        if max_layers > 0 {
            let mut cache = self.f16_decode_cache.lock().unwrap();
            let mut lru = self.gate_cache_lru.lock().unwrap();
            while lru.len() > max_layers {
                if let Some(evict) = lru.pop_back() {
                    if evict < cache.len() {
                        cache[evict] = None;
                    }
                }
            }
        }
    }

    /// Record a cache hit/miss on `layer`, evicting LRU entries if the
    /// cap is reached. Must be called with `cache` already locked by the
    /// caller; `just_inserted` is true when the caller *just* decoded and
    /// wrote `cache[layer]`.
    fn touch_gate_cache_lru(&self, layer: usize, just_inserted: bool, cache: &mut Vec<Option<Vec<f32>>>) {
        let max = self.gate_cache_max_layers.load(std::sync::atomic::Ordering::Relaxed);
        if max == 0 {
            return;
        }
        let mut lru = self.gate_cache_lru.lock().unwrap();
        // Move `layer` to the front (newest). If it's not in the queue
        // yet, push it; otherwise rotate.
        if let Some(pos) = lru.iter().position(|&l| l == layer) {
            lru.remove(pos);
        }
        lru.push_front(layer);
        if just_inserted {
            while lru.len() > max {
                if let Some(evict) = lru.pop_back() {
                    if evict < cache.len() && evict != layer {
                        cache[evict] = None;
                    }
                }
            }
        }
    }

    /// Resolve the gate matrix for a layer as contiguous f32.
    /// Handles all storage paths: warmed → heap → mmap f32 → mmap f16.
    /// Returns owned data (zero-copy from mmap via to_vec on the hot path).
    fn resolve_gate(&self, layer: usize) -> Option<GateData> {
        // 1. Warmed cache
        {
            let warmed = self.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self.gate_mmap_slices.get(layer).map(|s| s.num_features).unwrap_or(0);
                if nf > 0 {
                    return Some(GateData { data: data.clone(), num_features: nf });
                }
            }
        }

        // 2. Heap
        if let Some(Some(ref matrix)) = self.gate_vectors.get(layer) {
            return Some(GateData {
                data: matrix.as_slice().unwrap().to_vec(),
                num_features: matrix.shape()[0],
            });
        }

        // 3. Mmap
        if let Some(ref mmap) = self.gate_mmap_bytes {
            if let Some(slice) = self.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 { return None; }
                let bpf = crate::config::dtype::bytes_per_float(self.gate_mmap_dtype);
                let byte_offset = slice.float_offset * bpf;
                let byte_count = slice.num_features * self.hidden_size * bpf;
                let byte_end = byte_offset + byte_count;
                if byte_end > mmap.len() { return None; }

                let data = match self.gate_mmap_dtype {
                    crate::config::dtype::StorageDtype::F32 => {
                        let float_count = slice.num_features * self.hidden_size;
                        unsafe {
                            let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                            std::slice::from_raw_parts(ptr, float_count).to_vec()
                        }
                    }
                    crate::config::dtype::StorageDtype::F16 => {
                        let mut cache = self.f16_decode_cache.lock().unwrap();
                        if cache.len() <= layer { cache.resize(layer + 1, None); }
                        let miss = cache[layer].is_none();
                        if miss {
                            let raw = &mmap[byte_offset..byte_end];
                            cache[layer] = Some(larql_models::quant::half::decode_f16(raw));
                        }
                        self.touch_gate_cache_lru(layer, miss, &mut cache);
                        cache[layer].as_ref().unwrap().clone()
                    }
                };
                return Some(GateData { data, num_features: slice.num_features });
            }
        }

        None
    }

    /// Gate KNN: find the top-K features at a layer whose gate vectors have
    /// the highest dot product with the input residual. Uses BLAS matmul.
    ///
    /// In mmap mode, slices directly from the mmap'd file — zero heap allocation.
    /// Returns (feature_index, dot_product) sorted by absolute magnitude descending.
    pub fn gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // HNSW path
        if self.hnsw_enabled.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(results) = self.gate_knn_hnsw(layer, residual, top_k) {
                return results;
            }
        }

        // Fast path: f32 mmap zero-copy (no allocation, no clone)
        if let Some(scores) = self.gate_knn_mmap_fast(layer, residual) {
            return Self::top_k_from_scores(&scores, top_k);
        }

        // Fallback: resolve_gate (copies data for heap/f16 paths)
        let gate = match self.resolve_gate(layer) {
            Some(g) => g,
            None => return vec![],
        };
        let view = gate.view(self.hidden_size);
        let scores = gemv(&view, residual);
        Self::top_k_from_scores(&scores, top_k)
    }

    /// Zero-copy gate KNN for f32 mmap — no allocation, no clone.
    /// Returns None if not on the f32 mmap path (falls back to resolve_gate).
    fn gate_knn_mmap_fast(&self, layer: usize, residual: &Array1<f32>) -> Option<Array1<f32>> {
        // Warmed cache (RwLock read — lock-free when no writers)
        {
            let warmed = self.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self.gate_mmap_slices.get(layer).map(|s| s.num_features).unwrap_or(0);
                if nf > 0 {
                    let view = ArrayView2::from_shape((nf, self.hidden_size), data.as_slice()).unwrap();
                    return Some(gemv(&view, residual));
                }
            }
        }

        // f32 mmap zero-copy
        if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            if let Some(ref mmap) = self.gate_mmap_bytes {
                if let Some(slice) = self.gate_mmap_slices.get(layer) {
                    if slice.num_features == 0 { return None; }
                    let bpf = 4;
                    let byte_offset = slice.float_offset * bpf;
                    let byte_end = byte_offset + slice.num_features * self.hidden_size * bpf;
                    if byte_end > mmap.len() { return None; }
                    let data = unsafe {
                        let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                        std::slice::from_raw_parts(ptr, slice.num_features * self.hidden_size)
                    };
                    let view = ArrayView2::from_shape((slice.num_features, self.hidden_size), data).unwrap();
                    return Some(gemv(&view, residual));
                }
            }
        }

        None // Not on fast path — caller will use resolve_gate
    }

    /// Batched gate walk: scores all features via a single BLAS `gemv`, then
    /// extracts the top-K. Despite the name, this is batched matrix-vector —
    /// see [`Self::gate_walk_pure`] for a true per-feature implementation.
    pub fn gate_walk(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        let num_features = self.num_features(layer);
        if num_features == 0 { return None; }

        // Get gate data as contiguous f32 (from mmap or warmed cache)
        let gate_data: &[f32];
        let _owned: Vec<f32>;

        // Try zero-copy f32 mmap first
        let mmap_slice = if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            self.gate_mmap_bytes.as_ref().and_then(|mmap| {
                let slice = self.gate_mmap_slices.get(layer)?;
                if slice.num_features == 0 { return None; }
                let byte_offset = slice.float_offset * 4;
                let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
                if byte_end > mmap.len() { return None; }
                Some(unsafe {
                    std::slice::from_raw_parts(
                        mmap[byte_offset..byte_end].as_ptr() as *const f32,
                        slice.num_features * self.hidden_size,
                    )
                })
            })
        } else {
            None
        };

        if let Some(data) = mmap_slice {
            gate_data = data;
        } else {
            // Fallback: resolve gate (may clone)
            let gate = self.resolve_gate(layer)?;
            _owned = gate.data;
            gate_data = &_owned;
        }

        let hidden = self.hidden_size;

        // Single BLAS gemv: gate[N, hidden] × residual[hidden] → scores[N].
        let gate_view = ArrayView2::from_shape((num_features, hidden), gate_data).unwrap();
        let scores = gemv(&gate_view, residual);

        // Top-K selection
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        let k = top_k.min(indexed.len());
        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        Some(indexed)
    }

    /// Gate KNN within a specific feature range (for MoE expert-scoped queries).
    /// Only computes dot products for features [feat_start..feat_end].
    /// Returns (global_feature_index, score) pairs.
    pub fn gate_knn_expert(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        feat_start: usize,
        feat_end: usize,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // If promoted to heap, use heap path
        if let Some(Some(ref matrix)) = self.gate_vectors.get(layer) {
            let end = feat_end.min(matrix.shape()[0]);
            if feat_start >= end { return vec![]; }
            let slice = matrix.slice(ndarray::s![feat_start..end, ..]);
            let scores = gemv(&slice, residual);
            let mut hits = Self::top_k_from_scores(&scores, top_k);
            for hit in &mut hits { hit.0 += feat_start; }
            return hits;
        }

        if let Some(ref mmap) = self.gate_mmap_bytes {
            if let Some(slice) = self.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 || feat_start >= slice.num_features { return vec![]; }
                let end = feat_end.min(slice.num_features);
                let bpf = crate::config::dtype::bytes_per_float(self.gate_mmap_dtype);

                // Compute byte range for just this expert's features
                let layer_byte_start = slice.float_offset * bpf;
                let expert_byte_start = layer_byte_start + feat_start * self.hidden_size * bpf;
                let expert_byte_end = layer_byte_start + end * self.hidden_size * bpf;
                let n_features = end - feat_start;

                if expert_byte_end > mmap.len() { return vec![]; }

                match self.gate_mmap_dtype {
                    crate::config::dtype::StorageDtype::F32 => {
                        let data = unsafe {
                            let ptr = mmap[expert_byte_start..expert_byte_end].as_ptr() as *const f32;
                            std::slice::from_raw_parts(ptr, n_features * self.hidden_size)
                        };
                        let view = ndarray::ArrayView2::from_shape(
                            (n_features, self.hidden_size), data
                        ).unwrap();
                        let scores = gemv(&view, residual);
                        let mut hits = Self::top_k_from_scores(&scores, top_k);
                        // Offset indices to global feature space
                        for hit in &mut hits { hit.0 += feat_start; }
                        return hits;
                    }
                    crate::config::dtype::StorageDtype::F16 => {
                        let raw = &mmap[expert_byte_start..expert_byte_end];
                        let floats = larql_models::quant::half::decode_f16(raw);
                        let view = ndarray::ArrayView2::from_shape(
                            (n_features, self.hidden_size), &floats
                        ).unwrap();
                        let scores = gemv(&view, residual);
                        let mut hits = Self::top_k_from_scores(&scores, top_k);
                        for hit in &mut hits { hit.0 += feat_start; }
                        return hits;
                    }
                }
            }
        }
        // Fallback: full KNN filtered (slower)
        self.gate_knn(layer, residual, top_k * 10)
            .into_iter()
            .filter(|(f, _)| *f >= feat_start && *f < feat_end)
            .take(top_k)
            .collect()
    }

    fn top_k_from_scores(scores: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        let k = top_k.min(indexed.len());
        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed
    }

    /// Full walk: gate KNN at each layer, annotated with down token metadata.
    pub fn walk(
        &self,
        residual: &Array1<f32>,
        layers: &[usize],
        top_k: usize,
    ) -> WalkTrace {
        let mut trace_layers = Vec::with_capacity(layers.len());

        for &layer in layers {
            let hits = self.gate_knn(layer, residual, top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.feature_meta(layer, feature)?;
                    Some(WalkHit {
                        layer,
                        feature,
                        gate_score,
                        meta,
                    })
                })
                .collect();
            trace_layers.push((layer, walk_hits));
        }

        WalkTrace {
            layers: trace_layers,
        }
    }

    /// Batched gate KNN: compute scores for ALL sequence positions in one BLAS gemm.
    ///
    /// Input: x is [seq_len, hidden]. Computes gate_vectors @ x^T = [features, seq_len].
    /// Returns the union of per-position top-K feature indices (sorted).
    /// One gemm replaces seq_len separate gemv calls.
    pub fn gate_knn_batch(
        &self,
        layer: usize,
        x: &Array2<f32>,
        top_k: usize,
    ) -> Vec<usize> {
        let seq_len = x.shape()[0];
        if seq_len == 0 { return vec![]; }

        // Fast path: zero-copy f32 mmap/warmed
        let scores_2d = if let Some(s) = self.gate_scores_2d_fast(layer, x) {
            s
        } else if let Some(gate) = self.resolve_gate(layer) {
            gate_matmul(&gate.view(self.hidden_size), &x.view())
        } else {
            return vec![];
        };

        // scores_2d is [num_features, seq_len]
        // For each position, take top-K features and union them
        let num_features = scores_2d.shape()[0];
        let mut feature_set = std::collections::BTreeSet::new();

        for s in 0..seq_len {
            let col = scores_2d.column(s);
            let mut indexed: Vec<(usize, f32)> = col.iter().copied().enumerate().collect();
            let k = top_k.min(num_features);
            if k > 0 && k < indexed.len() {
                indexed.select_nth_unstable_by(k, |a, b| {
                    b.1.abs().partial_cmp(&a.1.abs()).unwrap()
                });
                indexed.truncate(k);
            }
            feature_set.extend(indexed.iter().map(|(idx, _)| *idx));
        }

        feature_set.into_iter().collect()
    }

    // Feature store methods (load_down/up_features, down/up_layer_matrix, warmup)
    // are in feature_store.rs

    /// Compute gate scores for all features × all positions in one BLAS gemm.
    /// Returns [seq_len, intermediate] matrix = x @ gate_vectors^T.
    /// These scores are the gate projections — the same as x @ W_gate.T.
    pub fn gate_scores_batch(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<Array2<f32>> {
        self.gate_scores_batch_backend(layer, x, None)
    }

    /// Backend-aware gate scores. When `backend` is present and `x` is
    /// a single row (seq_len == 1), route through `f32_gemv` — the
    /// same row-per-simdgroup path that closed lm_head. On Gemma 4 31B
    /// decode (hidden = 5376, ~18 K features, 60 layers) the CPU-BLAS
    /// path clocks ~4.3 ms/layer × 60 = 258 ms/token = 60 % of decode.
    /// Metal f32_gemv was measured at ~1 ms/layer on the lm_head of
    /// similar shape, so the upside is ~200 ms/token.
    pub fn gate_scores_batch_backend(
        &self,
        layer: usize,
        x: &Array2<f32>,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Array2<f32>> {
        if x.shape()[0] == 0 { return None; }

        // Metal gemv fast path (decode / single-row prefill).
        if let Some(be) = backend {
            if x.shape()[0] == 1 {
                if let Some(scores_2d) = self.gate_scores_2d_gpu(layer, x, be) {
                    return Some(scores_2d.t().to_owned());
                }
            }
        }

        // BLAS paths — warmed f32 / mmap f32 / lazy-decoded f16.
        let scores_2d = if let Some(s) = self.gate_scores_2d_fast(layer, x) {
            s
        } else {
            let gate = self.resolve_gate(layer)?;
            gate_matmul(&gate.view(self.hidden_size), &x.view())
        };
        Some(scores_2d.t().to_owned())
    }

    /// Zero-copy GPU gate scores for f32 mmap/warmed, single-row `x`.
    /// Matches `gate_scores_2d_fast` shape contract: returns [N, 1].
    fn gate_scores_2d_gpu(
        &self,
        layer: usize,
        x: &Array2<f32>,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Array2<f32>> {
        // Warmed cache (f32 heap).
        {
            let warmed = self.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self.gate_mmap_slices.get(layer).map(|s| s.num_features).unwrap_or(0);
                if nf > 0 {
                    let view = ArrayView2::from_shape((nf, self.hidden_size), data.as_slice()).unwrap();
                    if let Some(scores) = gate_gemv_gpu(&view, &x.view(), backend) {
                        return Some(scores);
                    }
                }
            }
        }
        // f32 mmap (zero-copy, the production path for f32 gate vectors).
        if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            if let Some(ref mmap) = self.gate_mmap_bytes {
                if let Some(slice) = self.gate_mmap_slices.get(layer) {
                    if slice.num_features == 0 { return None; }
                    let byte_offset = slice.float_offset * 4;
                    let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
                    if byte_end > mmap.len() { return None; }
                    let data = unsafe {
                        let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                        std::slice::from_raw_parts(ptr, slice.num_features * self.hidden_size)
                    };
                    let view = ArrayView2::from_shape((slice.num_features, self.hidden_size), data).unwrap();
                    if let Some(scores) = gate_gemv_gpu(&view, &x.view(), backend) {
                        return Some(scores);
                    }
                }
            }
        }
        // f16 mmap: zero-copy pass of raw f16 bytes to Metal's f16_gemv
        // shader, skipping the f16→f32 decode cache entirely. On 31B with
        // an ~18 K × 5376 gate matrix (387 MB f32, 194 MB f16) halving
        // the memory bandwidth is the difference between hitting the
        // CPU-BLAS ceiling and going faster on Metal.
        if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F16 {
            if x.shape()[0] == 1 {
                let slice = self.gate_mmap_slices.get(layer)?;
                if slice.num_features == 0 { return None; }
                let mmap = self.gate_mmap_bytes.as_ref()?;
                let byte_offset = slice.float_offset * 2;
                let byte_end = byte_offset + slice.num_features * self.hidden_size * 2;
                if byte_end <= mmap.len() {
                    let raw = &mmap[byte_offset..byte_end];
                    let x_row = x.row(0);
                    if let Some(x_slice) = x_row.as_slice() {
                        if let Some(scores) = backend.f16_gemv_force(
                            raw, x_slice, slice.num_features, self.hidden_size,
                        ) {
                            return Array2::from_shape_vec((slice.num_features, 1), scores).ok();
                        }
                    }
                }
            }
        }
        None
    }

    /// Zero-copy batch gate scores for f32 mmap/warmed — returns [features, seq].
    fn gate_scores_2d_fast(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        // Warmed cache
        {
            let warmed = self.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self.gate_mmap_slices.get(layer).map(|s| s.num_features).unwrap_or(0);
                if nf > 0 {
                    let view = ArrayView2::from_shape((nf, self.hidden_size), data.as_slice()).unwrap();
                    return Some(gate_matmul(&view, &x.view()));
                }
            }
        }
        // f32 mmap
        if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            if let Some(ref mmap) = self.gate_mmap_bytes {
                if let Some(slice) = self.gate_mmap_slices.get(layer) {
                    if slice.num_features == 0 { return None; }
                    let byte_offset = slice.float_offset * 4;
                    let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
                    if byte_end > mmap.len() { return None; }
                    let data = unsafe {
                        let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                        std::slice::from_raw_parts(ptr, slice.num_features * self.hidden_size)
                    };
                    let view = ArrayView2::from_shape((slice.num_features, self.hidden_size), data).unwrap();
                    return Some(gate_matmul(&view, &x.view()));
                }
            }
        }
        // f16 mmap — lazy decode into cache, then borrow (no per-call clone).
        // Holding the Mutex for the matmul is fine: forward passes are serial
        // per-layer, and this replaces a 462MB clone with a direct view.
        if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F16 {
            let slice = self.gate_mmap_slices.get(layer)?;
            if slice.num_features == 0 { return None; }
            let mmap = self.gate_mmap_bytes.as_ref()?;
            let mut cache = self.f16_decode_cache.lock().unwrap();
            if cache.len() <= layer { cache.resize(layer + 1, None); }
            let miss = cache[layer].is_none();
            if miss {
                let byte_offset = slice.float_offset * 2;
                let byte_end = byte_offset + slice.num_features * self.hidden_size * 2;
                if byte_end > mmap.len() { return None; }
                let raw = &mmap[byte_offset..byte_end];
                cache[layer] = Some(larql_models::quant::half::decode_f16(raw));
            }
            self.touch_gate_cache_lru(layer, miss, &mut cache);
            let data = cache[layer].as_ref().unwrap();
            let view = ArrayView2::from_shape((slice.num_features, self.hidden_size), data.as_slice()).unwrap();
            return Some(gate_matmul(&view, &x.view()));
        }
        None
    }

    /// Enable HNSW search. Indexes are built lazily on first query per layer.
    ///
    /// `ef_search`: beam width for search (50-200). Higher = better recall, slower.
    pub fn enable_hnsw(&self, ef_search: usize) {
        self.hnsw_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
        self.hnsw_ef_search.store(ef_search, std::sync::atomic::Ordering::Relaxed);
    }

    /// Disable HNSW, revert to brute-force matmul.
    pub fn disable_hnsw(&self) {
        self.hnsw_enabled.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether HNSW is currently enabled.
    pub fn is_hnsw_enabled(&self) -> bool {
        self.hnsw_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get the gate vector matrix for a layer as owned contiguous f32.
    /// Used by HNSW build which needs owned data.
    fn gate_matrix_f32(&self, layer: usize) -> Option<(Vec<f32>, usize)> {
        let gate = self.resolve_gate(layer)?;
        Some((gate.data, gate.num_features))
    }

    /// Get or build the HNSW index for a layer (lazy).
    fn get_or_build_hnsw(&self, layer: usize) -> bool {
        let mut cache = self.hnsw_cache.lock().unwrap();
        if cache.len() <= layer { cache.resize_with(layer + 1, || None); }
        if cache[layer].is_some() { return true; }

        // Build from gate vectors
        if let Some((data, num_features)) = self.gate_matrix_f32(layer) {
            let view = ArrayView2::from_shape(
                (num_features, self.hidden_size), &data
            ).unwrap();
            let hnsw = super::hnsw::HnswLayer::build(&view, 8, 32);
            cache[layer] = Some(hnsw);
            true
        } else {
            false
        }
    }

    /// Gate KNN via HNSW: graph search instead of brute-force matmul.
    fn gate_knn_hnsw(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        if !self.get_or_build_hnsw(layer) { return None; }

        let ef = self.hnsw_ef_search.load(std::sync::atomic::Ordering::Relaxed);

        // We need both the HNSW index and the vectors for search
        let cache = self.hnsw_cache.lock().unwrap();
        let hnsw = cache[layer].as_ref()?;

        // Get gate matrix for dot product computation during search
        let (data, num_features) = self.gate_matrix_f32(layer)?;
        let view = ArrayView2::from_shape(
            (num_features, self.hidden_size), &data
        ).unwrap();

        let results = hnsw.search(&view, residual, top_k, ef);
        Some(results)
    }

    /// Adaptive gate KNN — automatically picks the fastest path per layer.
    ///
    /// Dispatch order:
    /// 1. Pinned Q4 → backend.q4_matvec (pre-loaded, no page faults)
    /// 2. Mmap Q4 → backend.q4_matvec (paged on demand)
    /// 3. f32 mmap/heap → BLAS brute-force (fallback)
    ///
    /// The residency manager tracks which layers are pinned.
    /// More memory budget → more pinned layers → faster walk.
    pub fn gate_knn_adaptive(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        residency: &mut super::residency::ResidencyManager,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(usize, f32)> {
        residency.record_access(layer);

        // 1. Pinned Q4 (fastest — data already in RAM)
        if let Some(q4_data) = residency.pinned_q4(layer) {
            if backend.has_q4() {
                let x = residual.as_slice().unwrap();
                let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x);
                let num_features = self.num_features(layer);
                if let Some(scores_vec) = backend.q4_matvec(
                    q4_data, &q8_x, &q8_scales, num_features, self.hidden_size,
                ) {
                    return Self::top_k_from_scores(&Array1::from_vec(scores_vec), top_k);
                }
            }
        }

        // 2. Mmap Q4 (Q4 file loaded but not pinned — OS pages on demand)
        if let Some(hits) = self.gate_knn_q4(layer, residual, top_k, backend) {
            return hits;
        }

        // 3. f32 brute-force (fallback)
        self.gate_knn(layer, residual, top_k)
    }

    /// Gate KNN via Q4 matvec — scored by a ComputeBackend.
    ///
    /// The vindex provides the raw Q4 data. The backend scores it.
    /// Works with any backend: CPU C kernel, Metal GPU, CUDA, WASM.
    ///
    /// Returns None if Q4 gate data isn't loaded or backend doesn't support Q4.
    pub fn gate_knn_q4(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(usize, f32)>> {
        if !backend.has_q4() { return None; }
        let q4_data = self.gate_q4_data(layer)?;
        let slice = self.gate_q4_slices.get(layer)?;
        if slice.num_features == 0 { return None; }

        let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(residual.as_slice().unwrap());
        let scores_vec = backend.q4_matvec(
            q4_data, &q8_x, &q8_scales,
            slice.num_features, self.hidden_size,
        )?;

        let scores = Array1::from_vec(scores_vec);
        Some(Self::top_k_from_scores(&scores, top_k))
    }

}
