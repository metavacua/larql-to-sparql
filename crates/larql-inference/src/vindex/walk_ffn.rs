//! WalkFfn — FFN backend that replaces dense matmul with vindex lookups.
//!
//! Sparse walk path (preferred):
//!   gate_knn (HNSW or brute) → K up dot products → GEGLU → K down accumulations
//!   No dense matmuls. Reads only K feature vectors from mmap.
//!
//! Fallback paths:
//!   exact: gate/up from model weights + down from mmap (3 dense matmuls)
//!   full_mmap: all three from mmap (3 dense matmuls)
//!   sparse_model: gate KNN + sparse gather from model weights

use ndarray::Array2;
use rayon::prelude::*;

use larql_compute::ComputeBackend;
use crate::ffn::FfnBackend;
use crate::ffn::sparse_compute::sparse_ffn_forward;
use crate::model::ModelWeights;
use crate::vindex::l1_cache::FfnL1Cache;
use crate::vindex::walk_config::WalkFfnConfig;

use larql_vindex::{GateIndex, WalkHit, WalkTrace};

/// Helper enums for the K=full gemv path. Keep the backing storage alive
/// (Arc<Vec<f32>> or native mmap view) so the ArrayView2 borrows are valid.
#[allow(dead_code)]
enum UpMatrix<'a> {
    View(ndarray::ArrayView2<'a, f32>),
    Arc(std::sync::Arc<Vec<f32>>),
}
#[allow(dead_code)]
enum DownMatrix<'a> {
    View(ndarray::ArrayView2<'a, f32>),
    Arc(std::sync::Arc<Vec<f32>>),
}

/// True when the user asked for full-K (K ≥ feature count) — the signal
/// that we should route the walk through batched gemm rather than a
/// per-feature loop. Treats `usize::MAX` (set by `::dense` / `--k full`)
/// as full-K; also caches the check when top-K happens to exceed the
/// layer's feature count.
#[inline]
fn hits_len_ge_intermediate(config: &WalkFfnConfig, layer: usize, intermediate: usize) -> bool {
    match config.k_for(layer) {
        Some(k) => k >= (intermediate * 8) / 10,
        None => true,
    }
}

pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn GateIndex,
    pub config: WalkFfnConfig,
    pub backend: Option<&'a dyn ComputeBackend>,
    trace_residuals: std::cell::RefCell<Vec<(usize, Vec<f32>)>>,
    record_trace: bool,
    l1_cache: Option<FfnL1Cache>,
}

impl<'a> WalkFfn<'a> {
    /// Primary constructor. All other `::new*` constructors build a
    /// `WalkFfnConfig` and delegate here.
    pub fn from_config(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        config: WalkFfnConfig,
    ) -> Self {
        Self {
            weights, index, config, backend: None,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: false,
            l1_cache: None,
        }
    }

    /// Attach a compute backend (Metal / BLAS routing for dense-path gemms).
    pub fn with_backend(mut self, backend: &'a dyn ComputeBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Capture per-layer residuals for deferred WalkTrace reconstruction.
    pub fn with_trace(mut self) -> Self {
        self.record_trace = true;
        self
    }

    /// Enable the L1 in-process FFN output cache for this instance.
    /// Cache persists for the lifetime of this WalkFfn (one generation session).
    pub fn with_l1_cache(mut self, num_layers: usize) -> Self {
        self.l1_cache = Some(FfnL1Cache::new(num_layers));
        self
    }

    /// Return L1 cache hit/miss stats, if cache was enabled.
    pub fn l1_cache_stats(&self) -> Option<(u64, u64)> {
        self.l1_cache.as_ref().map(|c| (c.hits(), c.misses()))
    }

    /// Effective top-K for a layer. None (dense walk) maps to usize::MAX
    /// for the handful of call sites that still expect a numeric K.
    fn top_k_for(&self, layer: usize) -> usize {
        self.config.k_for(layer).unwrap_or(usize::MAX)
    }

    // ── Legacy constructors (maintained for caller compatibility) ──

    /// Create a WalkFfn with a uniform per-layer top-K.
    /// `top_k == usize::MAX` picks the dense walk path for every layer.
    pub fn new(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        let config = if top_k == usize::MAX {
            WalkFfnConfig::dense(weights.num_layers)
        } else {
            WalkFfnConfig::sparse(weights.num_layers, top_k)
        };
        Self::from_config(weights, index, config)
    }

    /// Create with unlimited K — no artificial cap on feature count.
    pub fn new_unlimited(weights: &'a ModelWeights, index: &'a dyn GateIndex) -> Self {
        Self::from_config(weights, index, WalkFfnConfig::dense(weights.num_layers))
    }

    pub fn new_with_backend(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        top_k: usize,
        backend: &'a dyn ComputeBackend,
    ) -> Self {
        Self::new(weights, index, top_k).with_backend(backend)
    }

    /// Create with backend and unlimited K.
    pub fn new_unlimited_with_backend(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        backend: &'a dyn ComputeBackend,
    ) -> Self {
        Self::new_unlimited(weights, index).with_backend(backend)
    }

    pub fn new_with_trace(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        Self::new(weights, index, top_k).with_trace()
    }

    /// Unlimited top_k plus residual tracing. Used by `exec_infer`
    /// whenever a patched session has installed slots — bounded
    /// top_k drops features from the activation sum, which is
    /// harmless on a clean model (dropped features have tiny
    /// activations) but catastrophic once a strong (×30 gate scale)
    /// INSERT slot is in the mix: the slot's activation then
    /// dominates a half-weakened baseline and hijacks every prompt
    /// to whichever installed target has the largest lm_head
    /// alignment. Matching the dense FFN by processing every
    /// feature keeps the baseline intact and the installed slot
    /// proportional.
    pub fn new_unlimited_with_trace(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
    ) -> Self {
        Self::new_unlimited(weights, index).with_trace()
    }

    /// Take raw per-layer residuals (the exact vectors gate_knn sees during inference).
    /// These are the normalized post-attention hidden states at the last token position.
    pub fn take_residuals(&self) -> Vec<(usize, Vec<f32>)> {
        self.trace_residuals.borrow_mut().drain(..).collect()
    }

    pub fn take_trace(&self) -> WalkTrace {
        let residuals = self.trace_residuals.borrow_mut().drain(..).collect::<Vec<_>>();
        let mut layers = Vec::with_capacity(residuals.len());
        for (layer, residual) in residuals {
            let r = ndarray::Array1::from_vec(residual);
            let hits = self.index.gate_knn(layer, &r, self.top_k_for(layer));
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.index.feature_meta(layer, feature)?.clone();
                    Some(WalkHit { layer, feature, gate_score, meta })
                })
                .collect();
            layers.push((layer, walk_hits));
        }
        WalkTrace { layers }
    }

    /// Sparse walk FFN: zero matrix multiplications.
    ///
    /// Per position:
    ///   1. gate_knn → top-K features with gate scores (HNSW graph search, no matmul)
    ///   2. For each feature: up_score = up_mmap[feat] · x  (dot product)
    ///   3. activation = silu(gate_score) * up_score          (GEGLU)
    ///   4. out += activation * down_mmap[feat]               (scaled vector add)
    ///
    /// Operations: K dot products + K scaled adds per position. No matmuls.
    fn walk_ffn_sparse(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];
        let intermediate = self.index.num_features(layer);

        // Prefer native f32 mmap (zero-copy). When the vindex is Q4K-only
        // (e.g. Gemma 4 31B) we decode one row at a time into scratch
        // buffers — no full-layer dequant cache, so memory stays flat
        // regardless of model size. The row-decode cost is ~60μs on 31B
        // and only fires K times per layer, so at the sparse K users
        // actually run (100–500) the overhead is bounded.
        let up_native = self.index.up_layer_matrix(layer);
        let down_native = self.index.down_layer_matrix(layer);
        let q4k_row_fallback = up_native.is_none() || down_native.is_none();
        // Sanity-check Q4K data is present so we fail early rather than
        // surfacing confusing per-row decode misses.
        if q4k_row_fallback && self.index.interleaved_q4k_layer_data(layer).is_none() {
            return None;
        }

        // No scratch buffers needed — Q4K fused kernels decode + math in one pass.
        let _ = q4k_row_fallback;

        let arch = &*self.weights.arch;
        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        // Hoist layer-level state: the HashMap lookups inside the feature
        // loop fire ~15M times per forward on 31B K=full. When no INSERT
        // has touched this layer we can skip them entirely.
        let layer_has_overrides = self.index.has_overrides_at(layer);
        let up_bias_for_layer = if !is_gated {
            arch.ffn_up_bias_key(layer).and_then(|bk| self.weights.vectors.get(&bk).cloned())
        } else { None };

        // K=full gemv fast path. When every feature is active (top-K > N),
        // the per-feature loop is mathematically equivalent to three dense
        // matmuls: gate_scores = x @ W_gate.T, up_scores = x @ W_up.T,
        // out = silu(gate)*up @ W_down.T. Routing through BLAS gemm is
        // 10–30× faster than iterating 10k+ features serially because
        // BLAS cache-blocks the work and keeps FMA pipelines saturated.
        //
        // Requires the up matrix cached as f32 [intermediate, hidden]. For
        // Q4K-only vindexes we call q4k_ffn_layer to build the cache on
        // first access (same mechanism as down_cache above). Memory cost:
        // ~3.4 GB on 4B per-model, ~27 GB on 31B — feasible on 4B laptops,
        // tight on 31B/64 GB machines (future work: per-layer streaming).
        // K=full fast path. Three variants, chosen by what the vindex exposes:
        //
        //  (A) native f32 mmap for up/down → route through BLAS sgemm
        //      (same as walk_ffn_interleaved); zero extra memory.
        //  (B) Q4K vindex, on-the-fly matmul_transb (direct-Q4K gemm)
        //      → decode + FMA fused per feature, parallel over W rows;
        //      zero extra memory (no f32 cache). Enables K=full on 31B
        //      within a 64 GB RAM budget.
        //  (C) Q4K vindex with cached f32 decode → fallback when direct
        //      matmul isn't available. Fastest on small models where
        //      memory is plentiful.
        //
        // Each variant terminates with the same silu/gelu * up → activation
        // → activation @ down → out sequence.
        let k_is_full = hits_len_ge_intermediate(&self.config, layer, intermediate);
        if !layer_has_overrides && is_gated && k_is_full {
            let x_slice_for_matmul: Option<&[f32]> = x.as_slice();
            if let (Some(gate_scores), Some(x_flat)) =
                (self.index.gate_scores_batch_backend(layer, x, self.backend), x_slice_for_matmul)
            {
                // Up leg — native f32 mmap if present, else direct Q4K matmul.
                let up_scores: Option<ndarray::Array2<f32>> = if let Some(v) = up_native {
                    Some(larql_compute::dot_proj_gpu(x, &v, self.backend))
                } else if let Some(y) = self.index.q4k_matmul_transb(layer, 1, x_flat, seq_len, self.backend) {
                    ndarray::Array2::from_shape_vec((seq_len, intermediate), y).ok()
                } else { None };

                if let Some(up_scores) = up_scores {
                    let activation = if use_gelu {
                        crate::ffn::gelu_tanh_gate_up(&gate_scores, &up_scores)
                    } else {
                        crate::ffn::silu_gate_up(&gate_scores, &up_scores)
                    };
                    // Down leg.
                    let act_slice: Option<&[f32]> = activation.as_slice();
                    let out_matmul: Option<ndarray::Array2<f32>> = if let Some(v) = down_native {
                        Some(larql_compute::matmul_gpu(&activation, &v, self.backend))
                    } else if let Some(act_flat) = act_slice {
                        self.index
                            .q4k_matmul_transb(layer, 2, act_flat, seq_len, self.backend)
                            .and_then(|y| ndarray::Array2::from_shape_vec((seq_len, hidden), y).ok())
                    } else { None };
                    if let Some(out_matmul) = out_matmul {
                        out.assign(&out_matmul);
                        full_activation.assign(&activation);
                        return Some((out, full_activation));
                    }
                }
            }
        }

        for s in 0..seq_len {
            let x_row = x.row(s);
            let x_owned = x_row.to_owned();
            // Used by q4k_ffn_row_dot (up fast path); constant per seq pos.
            let x_slice_owned: Vec<f32>;
            let x_slice: &[f32] = if let Some(sl) = x_row.as_slice() {
                sl
            } else {
                x_slice_owned = x_owned.as_slice().unwrap().to_vec();
                &x_slice_owned
            };

            // Gate: try fastest path available
            //   1. gate_walk (per-feature dot, no matmul) if available
            //   2. Q4 gate KNN via compute backend (0.5ms Metal, 1ms CPU Q4)
            //   3. f32 brute-force BLAS (1.1ms) as fallback
            let top_k = self.top_k_for(layer);
            let hits = self.index.gate_walk(layer, &x_owned, top_k)
                    .or_else(|| self.backend.and_then(|be| self.index.gate_knn_q4(layer, &x_owned, top_k, be)))
                    .unwrap_or_else(|| self.index.gate_knn(layer, &x_owned, top_k));

            let mut out_row = out.row_mut(s);

            // Parallel fast path — see comment above for trigger conditions.
            // Resolves the Q4K up slice once per layer, then the hot loop
            // calls `larql_models::quant::ggml::q4k_row_dot` directly (no
            // dyn dispatch per feature). On M3 Max this takes 31B K=full
            // from ~15 s to ~2 s per forward.
            let parallelisable = !layer_has_overrides
                && is_gated
                && hits.len() >= 512
                && down_native.is_none();
            // Populate the down cache here — only when the parallel path
            // will actually use it. At K=full the gemv fast path already
            // returned, so this pays for itself only on sparse K layers.
            let down_cache_local: Option<std::sync::Arc<Vec<f32>>> =
                if parallelisable { self.index.q4k_ffn_layer(layer, 2) } else { None };
            if parallelisable && down_cache_local.is_some() {
                let down_arc = down_cache_local.as_ref().unwrap();
                let down_data: &[f32] = down_arc.as_slice();
                // Hoist up-side Q4K slice out of the hot loop — one dyn call
                // here, then the closure uses `&[u8]` directly.
                let up_slices = self.index.interleaved_q4k_layer_data(layer);
                let up_q4k_bytes: Option<&[u8]> = match (up_native.as_ref(), up_slices) {
                    (Some(_), _) => None,
                    (None, Some(s)) if s[1].1 == "Q4_K" => Some(s[1].0),
                    _ => None,
                };
                let n_threads = rayon::current_num_threads().max(1);
                let chunk_size = hits.len().div_ceil(n_threads);
                let up_native_ref = up_native.as_ref();

                let partials: Vec<Vec<f32>> = hits
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut partial = vec![0.0f32; hidden];
                        for &(feat, gate_score) in chunk {
                            let up_score = if let Some(up_view) = up_native_ref {
                                up_view.row(feat).dot(&x_row)
                            } else if let Some(up_bytes) = up_q4k_bytes {
                                // Q4_K row stride: blocks_per_row * 144 bytes.
                                let bytes_per_row = (hidden / 256) * 144;
                                let start = feat * bytes_per_row;
                                let end = start + bytes_per_row;
                                match larql_models::quant::ggml::q4k_row_dot(
                                    &up_bytes[start..end], x_slice,
                                ) {
                                    Ok(v) => v,
                                    Err(_) => 0.0,
                                }
                            } else {
                                // Unknown up format — cheapest is to skip this
                                // feature. Accuracy at K=full may suffer but the
                                // parallelisable check gates this tightly.
                                0.0
                            };
                            let activated_gate = if use_gelu {
                                crate::ffn::gelu_tanh(gate_score)
                            } else {
                                gate_score * crate::ffn::sigmoid(gate_score)
                            };
                            let act = activated_gate * up_score;
                            if act.abs() > 1e-10 {
                                let row_start = feat * hidden;
                                let down_row = &down_data[row_start..row_start + hidden];
                                // Route through ndarray → BLAS saxpy rather
                                // than a hand-rolled loop; LLVM doesn't
                                // reliably auto-vectorise the scalar version.
                                let mut pv = ndarray::ArrayViewMut1::from(partial.as_mut_slice());
                                let dv = ndarray::ArrayView1::from(down_row);
                                pv.scaled_add(act, &dv);
                            }
                        }
                        partial
                    })
                    .collect();

                let out_slice = out_row.as_slice_mut().unwrap();
                for p in &partials {
                    for i in 0..hidden {
                        out_slice[i] += p[i];
                    }
                }
                // full_activation intentionally left zero in the fast path —
                // callers needing it drop to the serial loop.
                continue;
            }

            for (feat, gate_score) in hits {
                let act = if is_gated {
                    // Up source: INSERT override (rare) > native mmap row >
                    // Q4K per-row NEON decode. The `layer_has_overrides`
                    // early-out skips the HashMap lookup on clean layers.
                    let up_ov = if layer_has_overrides {
                        self.index.up_override(layer, feat)
                    } else { None };
                    let up_score = if let Some(up_ov) = up_ov {
                        if up_ov.len() == hidden {
                            ndarray::ArrayView1::from(up_ov).dot(&x_row)
                        } else if let Some(ref up_view) = up_native {
                            up_view.row(feat).dot(&x_row)
                        } else {
                            match self.index.q4k_ffn_row_dot(layer, 1, feat, x_slice) {
                                Some(v) => v, None => return None,
                            }
                        }
                    } else if let Some(ref up_view) = up_native {
                        up_view.row(feat).dot(&x_row)
                    } else {
                        match self.index.q4k_ffn_row_dot(layer, 1, feat, x_slice) {
                            Some(v) => v, None => return None,
                        }
                    };
                    let activated_gate = if use_gelu {
                        crate::ffn::gelu_tanh(gate_score)
                    } else {
                        gate_score * crate::ffn::sigmoid(gate_score)
                    };
                    activated_gate * up_score
                } else {
                    let mut v = gate_score;
                    if let Some(ref bias) = up_bias_for_layer {
                        if feat < bias.len() { v += bias[feat]; }
                    }
                    if use_gelu { crate::ffn::gelu_tanh(v) } else { v * crate::ffn::sigmoid(v) }
                };

                full_activation[[s, feat]] = act;

                if act.abs() > 1e-10 {
                    // Down: INSERT override (rare) > native mmap > Q4K cache.
                    let down_ov = if layer_has_overrides {
                        self.index.down_override(layer, feat)
                    } else { None };
                    if let Some(override_down) = down_ov {
                        if override_down.len() == hidden {
                            out_row.scaled_add(act, &ndarray::ArrayView1::from(override_down));
                            continue;
                        }
                    }
                    if let Some(ref down_view) = down_native {
                        out_row.scaled_add(act, &down_view.row(feat));
                    } else {
                        // Serial sparse fallback hits Q4K row-scaled-add
                        // against the transposed cache — populates it on
                        // demand; sized ~intermediate×hidden per layer.
                        let out_slice = out_row.as_slice_mut().unwrap();
                        if !self.index.q4k_ffn_row_scaled_add_via_cache(
                            layer, 2, feat, act, out_slice,
                        ) {
                            return None;
                        }
                    }
                }
            }
        }

        // Down bias
        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, full_activation))
    }

    /// Q4 interleaved walk: C kernel with vdotq_s32 for gate/up, scalar for down.
    /// Reads 44MB per layer instead of 315MB. Matches BLAS f32 speed on warm,
    /// faster on cold cache (7x less data to page in).
    fn walk_ffn_q4_interleaved(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        use larql_compute::cpu::ops::{q4_matvec, q4_vecmat};

        let q4_mmap = self.index.interleaved_q4_mmap_ref()?;
        let intermediate = self.index.num_features(layer);
        if intermediate == 0 { return None; }
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];

        let q4_bytes_per_matrix = intermediate * hidden / 32 * 18;
        let q4_bytes_per_layer = q4_bytes_per_matrix * 3;
        let layer_start = layer * q4_bytes_per_layer;

        let gate_q4 = &q4_mmap[layer_start..layer_start + q4_bytes_per_matrix];
        let up_q4 = &q4_mmap[layer_start + q4_bytes_per_matrix..layer_start + 2 * q4_bytes_per_matrix];
        let down_q4 = &q4_mmap[layer_start + 2 * q4_bytes_per_matrix..layer_start + 3 * q4_bytes_per_matrix];

        // Prefetch next layer
        self.index.prefetch_interleaved_q4_layer(layer + 1);

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        // Check for Metal Q4 backend
        let metal_q4 = self.backend.and_then(|be| if be.has_q4() { Some(be) } else { None });

        if let Some(be) = metal_q4 {
            // Metal: ONE GPU submission for all gate+up across ALL seq positions
            let x_flat = x.as_slice().unwrap();
            let (all_gate, all_up) = be.q4_matvec_pair_batch(
                gate_q4, up_q4, x_flat, seq_len, intermediate, hidden,
            ).unwrap();

            // GEGLU on CPU (element-wise, all positions)
            let mut all_activation: Vec<Vec<f32>> = Vec::with_capacity(seq_len);
            for s in 0..seq_len {
                let mut activation = vec![0.0f32; intermediate];
                for i in 0..intermediate {
                    let g = all_gate[s][i];
                    let u = all_up[s][i];
                    activation[i] = if use_gelu {
                        crate::ffn::gelu_tanh(g) * u
                    } else {
                        g * crate::ffn::sigmoid(g) * u
                    };
                    full_activation[[s, i]] = activation[i];
                }
                all_activation.push(activation);
            }

            // Down: one submission per position (GPU vecmat)
            for (s, activation_row) in all_activation.iter().enumerate().take(seq_len) {
                let down_result = be.q4_vecmat(activation_row, down_q4, intermediate, hidden).unwrap();
                let mut out_row = out.row_mut(s);
                for j in 0..hidden { out_row[j] = down_result[j]; }
            }
        } else {
            // C kernel path: vdotq for gate/up, scalar for down
            for s in 0..seq_len {
                let x_row = x.row(s);
                let x_slice = x_row.as_slice().unwrap();

                let gate_scores = q4_matvec::dispatch(gate_q4, x_slice, intermediate, hidden);
                let up_scores = q4_matvec::dispatch(up_q4, x_slice, intermediate, hidden);

                let mut activation = vec![0.0f32; intermediate];
                for i in 0..intermediate {
                    let g = gate_scores[i];
                    let u = up_scores[i];
                    activation[i] = if use_gelu {
                        crate::ffn::gelu_tanh(g) * u
                    } else {
                        g * crate::ffn::sigmoid(g) * u
                    };
                    full_activation[[s, i]] = activation[i];
                }

                let down_result = q4_vecmat::dispatch(&activation, down_q4, intermediate, hidden);
                let mut out_row = out.row_mut(s);
                for j in 0..hidden { out_row[j] = down_result[j]; }
            }
        }

        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, full_activation))
    }

    /// Interleaved walk: gate + up + down from one contiguous mmap per layer.
    /// Eliminates TLB thrash from 3 separate files. Prefetches next layer.
    fn walk_ffn_interleaved(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        // All three matrices from one contiguous region
        let gate_view = self.index.interleaved_gate(layer)?;
        let up_view = self.index.interleaved_up(layer)?;
        let down_view = self.index.interleaved_down(layer)?;

        // Prefetch next layer while we compute this one
        self.index.prefetch_interleaved_layer(layer + 1);

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // gate_scores = gate_vectors @ x^T (one BLAS gemv from contiguous region)
        let gate_scores = larql_compute::dot_proj_gpu(x, &gate_view, self.backend);

        // up_scores = x @ up_vectors^T (contiguous, right after gate in memory)
        let up_scores = larql_compute::dot_proj_gpu(x, &up_view, self.backend);

        // GEGLU
        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate_scores, &up_scores)
        } else {
            crate::ffn::silu_gate_up(&gate_scores, &up_scores)
        };

        // down: activation @ down_matrix (contiguous, right after up in memory)
        let mut out = larql_compute::matmul_gpu(&activation, &down_view, self.backend);

        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, activation))
    }

    /// Full mmap walk: gate + up + down all from mmap. Zero safetensor reads.
    ///
    /// gate_scores = gate_vectors @ x^T     (mmap, one BLAS gemm)
    /// up_scores   = up_vectors @ x^T       (mmap, one BLAS gemm)
    /// activation  = silu(gate) * up         (exact GEGLU)
    /// output      = activation @ down       (mmap, one BLAS gemm)
    ///
    /// Three mmap gemms. Same computation as dense. Zero model weight reads.
    fn walk_ffn_full_mmap(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let gate_scores = self.index.gate_scores_batch(layer, x)?;
        let up_view = self.index.up_layer_matrix(layer)?;
        let down_view = self.index.down_layer_matrix(layer)?;

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // up_scores = x @ up_vectors^T = [seq, intermediate]
        let up_scores = larql_compute::dot_proj_gpu(x, &up_view, self.backend);

        // GEGLU: silu(gate) * up  (exact, same as dense)
        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate_scores, &up_scores)
        } else {
            crate::ffn::silu_gate_up(&gate_scores, &up_scores)
        };

        // Down: activation @ down_matrix (mmap)
        let mut out = larql_compute::matmul_gpu(&activation, &down_view, self.backend);

        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        Some((out, activation))
    }

    /// CPU dequant path for Q4K streaming vindexes.
    ///
    /// Dequantises gate, up, and down matrices from the interleaved_q4k mmap for
    /// the given layer, then runs the standard dense GEGLU forward. Used by the
    /// INFER pipeline on q4k vindexes without a GPU backend.
    fn walk_ffn_q4k_dequant(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let ffn = self.index.interleaved_q4k_layer_data(layer)?;
        let arch = &*self.weights.arch;
        let intermediate = self.index.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        let hidden = x.shape()[1];

        let dequant = |bytes: &[u8], fmt: &str, rows: usize, cols: usize| -> Array2<f32> {
            let padded = rows * cols;
            let flat = match fmt {
                "Q6_K" => larql_models::quant::ggml::dequantize_q6_k(bytes, padded)
                    .expect("q6k dequant"),
                _ => larql_models::quant::ggml::dequantize_q4_k(bytes, padded)
                    .expect("q4k dequant"),
            };
            Array2::from_shape_vec((rows, cols), flat[..rows * cols].to_vec())
                .expect("dequant shape mismatch")
        };

        let w_gate = dequant(ffn[0].0, ffn[0].1, intermediate, hidden);
        let w_up = dequant(ffn[1].0, ffn[1].1, intermediate, hidden);
        let w_down = dequant(ffn[2].0, ffn[2].1, hidden, intermediate);

        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );
        let gate = crate::forward::dot_proj(x, &w_gate);
        let up = crate::forward::dot_proj(x, &w_up);
        let activation = if use_gelu {
            crate::ffn::gelu_tanh_gate_up(&gate, &up)
        } else {
            crate::ffn::silu_gate_up(&gate, &up)
        };
        let out = crate::forward::dot_proj(&activation, &w_down);
        Some((out, activation))
    }

    /// Walk FFN: gate/up from model weights + down from mmap.
    ///
    /// Uses dense gate/up matmul (exact, sequential reads) and reads the down
    /// matrix directly from the feature-major mmap (zero-copy BLAS gemm).
    /// Total: gate(105MB) + up(105MB) + down_mmap(105MB) = 315MB.
    /// Same bandwidth as dense but down read is from mmap (potentially cached).
    fn walk_ffn_exact(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;

        // If FFN weights were dropped (walk-only mode), fall through to full mmap
        let w_up = match self.weights.tensors.get(&arch.ffn_up_key(layer)) {
            Some(w) => w,
            None => {
                // No model FFN weights — use full mmap path
                if let Some(result) = self.walk_ffn_full_mmap(layer, x) {
                    return result;
                }
                panic!("walk_ffn_exact: no FFN weights and no mmap data for layer {layer}");
            }
        };

        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // Gate + up + GEGLU: exact computation from model weights
        let activation = if is_gated {
            let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
            let gate = crate::forward::dot_proj(x, w_gate);
            let up = crate::forward::dot_proj(x, w_up);
            if use_gelu {
                crate::ffn::gelu_tanh_gate_up(&gate, &up)
            } else {
                crate::ffn::silu_gate_up(&gate, &up)
            }
        } else {
            let mut proj = crate::forward::dot_proj(x, w_up);
            if let Some(bias) = arch.ffn_up_bias_key(layer)
                .and_then(|bk| self.weights.vectors.get(&bk))
            {
                crate::forward::add_bias(&mut proj, bias);
            }
            if use_gelu {
                proj.mapv(crate::ffn::gelu_tanh)
            } else {
                proj.mapv(|v| v * crate::ffn::sigmoid(v))
            }
        };

        // Down: zero-copy BLAS gemm against mmap'd feature-major matrix
        let out = if let Some(down_view) = self.index.down_layer_matrix(layer) {
            // Zero-copy: mmap reinterpreted as ArrayView2, routed through compute backend
            larql_compute::matmul_gpu(&activation, &down_view, self.backend)
        } else {
            // Fallback: read W_down from model weights via compute backend
            let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
            larql_compute::dot_proj_gpu(&activation, w_down, self.backend)
        };

        let mut out = out;
        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        (out, activation)
    }
}

impl<'a> FfnBackend for WalkFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let num_features = self.index.num_features(layer);
        if num_features == 0 {
            let dense_ffn = crate::ffn::WeightFfn { weights: self.weights };
            return dense_ffn.forward_with_activation(layer, x);
        }

        // Record for deferred trace
        if self.record_trace {
            let seq_len = x.shape()[0];
            let last_row = x.row(seq_len - 1).to_vec();
            self.trace_residuals.borrow_mut().push((layer, last_row));
        }

        // Override-aware routing: patched layers bypass the cache and go straight
        // to walk_ffn_sparse, which checks all three override slots per feature.
        // The BLAS/interleaved paths below operate on whole-layer matrices and
        // would silently produce wrong activations for overridden features.
        if self.index.has_overrides_at(layer) {
            if let Some(result) = self.walk_ffn_sparse(layer, x) {
                return result;
            }
        }

        // L1 cache: single-position only (autoregressive token, not prefill).
        // Placed after the override bypass so patched layers never hit here.
        // Uses residual_key (i16-quantised hash of x) which is path-independent —
        // the same input always produces the same FFN output regardless of which
        // walk_ variant executes below.
        let seq_len = x.shape()[0];
        let l1_key: Option<u64> = if seq_len == 1 && self.l1_cache.is_some() {
            let x_row = x.row(0);
            let owned;
            let slice: &[f32] = if let Some(s) = x_row.as_slice() {
                s
            } else {
                owned = x_row.to_vec();
                &owned
            };
            Some(FfnL1Cache::residual_key(slice))
        } else {
            None
        };

        if let Some(key) = l1_key {
            if let Some(cache) = &self.l1_cache {
                if let Some(cached) = cache.get(layer, key) {
                    let hidden = x.shape()[1];
                    let mut out = Array2::<f32>::zeros((1, hidden));
                    out.row_mut(0).assign(&ndarray::ArrayView1::from(cached.as_slice()));
                    return (out, Array2::zeros((1, num_features)));
                }
            }
        }

        // Routing: config.k_for(layer) decides the path.
        //   Some(k) → sparse walk (gate KNN + per-feature saxpy, no dense matmul).
        //   None    → dense walk (prefer mmap'd interleaved/q4; fall back to exact/weights).
        // Dense paths are attempted in perf-preference order.
        let result: (Array2<f32>, Array2<f32>) = 'routing: {
            // Sparse path: taken whenever the user specified a per-layer K.
            if self.config.is_sparse(layer) {
                if let Some(r) = self.walk_ffn_sparse(layer, x) {
                    break 'routing r;
                }
                // Sparse path requires up/down mmap — if unavailable, fall through
                // to the dense ladder below rather than silently dropping features.
            }

            // Q4 interleaved: preferred when GPU Q4 is available (Metal shader faster than BLAS).
            // CPU Q4 C kernel is slower than CPU BLAS at these dimensions — only use with GPU.
            if self.index.has_interleaved_q4() && self.backend.is_some_and(|be| be.has_q4()) {
                if let Some(r) = self.walk_ffn_q4_interleaved(layer, x) {
                    break 'routing r;
                }
            }

            // f32 interleaved: gate+up+down contiguous per layer.
            if self.index.has_interleaved() {
                if let Some(r) = self.walk_ffn_interleaved(layer, x) {
                    break 'routing r;
                }
            }

            // Full mmap walk: gate + up + down from 3 separate mmap files.
            if self.index.has_full_mmap_ffn() {
                if let Some(r) = self.walk_ffn_full_mmap(layer, x) {
                    break 'routing r;
                }
            }

            // Q4K interleaved CPU path: dequantise gate/up/down per layer from
            // the streaming Q4K mmap. Used by INFER on q4k vindexes without GPU.
            if self.index.has_interleaved_q4k() {
                if let Some(r) = self.walk_ffn_q4k_dequant(layer, x) {
                    break 'routing r;
                }
            }

            // Fallback: partial mmap (gate/up from model weights + down from mmap)
            if self.index.has_down_features() {
                break 'routing self.walk_ffn_exact(layer, x);
            }

            // Last resort: sparse matmul against model weights.
            let top_k = self.top_k_for(layer);
            let features = self.index.gate_knn_batch(layer, x, top_k);
            let has_any_override = features.iter().any(|&f| {
                self.index.down_override(layer, f).is_some()
                    || self.index.up_override(layer, f).is_some()
            }) || self.index.has_overrides_at(layer);

            if has_any_override {
                let slot_overrides: Vec<crate::ffn::FeatureSlotOverride<'_>> = features
                    .iter()
                    .map(|&f| crate::ffn::FeatureSlotOverride {
                        feature: f,
                        gate: self.index.gate_override(layer, f),
                        up: self.index.up_override(layer, f),
                        down: self.index.down_override(layer, f),
                    })
                    .filter(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some())
                    .collect();
                break 'routing crate::ffn::sparse_ffn_forward_with_full_overrides(
                    self.weights, layer, x, &features, &slot_overrides,
                );
            }
            break 'routing sparse_ffn_forward(self.weights, layer, x, &features);
        };

        // L1 cache insert: single position, key was computed above on miss.
        if let Some(key) = l1_key {
            if let Some(cache) = &self.l1_cache {
                cache.insert(layer, key, result.0.row(0).to_vec());
            }
        }

        result
    }

    fn name(&self) -> &str {
        "walk"
    }
}
