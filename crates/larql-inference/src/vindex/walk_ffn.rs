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

use larql_compute::ComputeBackend;
use crate::ffn::FfnBackend;
use crate::ffn::sparse_compute::{sparse_ffn_forward, sparse_ffn_forward_with_overrides};
use crate::model::ModelWeights;

use larql_vindex::{GateIndex, WalkHit, WalkTrace};

pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn GateIndex,
    pub top_k: usize,
    pub backend: Option<&'a dyn ComputeBackend>,
    trace_residuals: std::cell::RefCell<Vec<(usize, Vec<f32>)>>,
    record_trace: bool,
}

impl<'a> WalkFfn<'a> {
    pub fn new(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        Self {
            weights, index, top_k, backend: None,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: false,
        }
    }

    pub fn new_with_backend(
        weights: &'a ModelWeights,
        index: &'a dyn GateIndex,
        top_k: usize,
        backend: &'a dyn ComputeBackend,
    ) -> Self {
        Self {
            weights, index, top_k, backend: Some(backend),
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: false,
        }
    }

    pub fn new_with_trace(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        Self {
            weights, index, top_k, backend: None,
            trace_residuals: std::cell::RefCell::new(Vec::new()),
            record_trace: true,
        }
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
            let hits = self.index.gate_knn(layer, &r, self.top_k);
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
        let up_view = self.index.up_layer_matrix(layer)?;
        let down_view = self.index.down_layer_matrix(layer)?;

        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];
        let intermediate = self.index.num_features(layer);

        let arch = &*self.weights.arch;
        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        for s in 0..seq_len {
            let x_row = x.row(s);
            let x_owned = x_row.to_owned();

            // Gate walk: per-feature dot products (no matmul)
            // Falls back to gate_knn (HNSW or brute) if gate_walk not available
            let hits = self.index.gate_walk(layer, &x_owned, self.top_k)
                .unwrap_or_else(|| self.index.gate_knn(layer, &x_owned, self.top_k));

            let mut out_row = out.row_mut(s);

            for (feat, gate_score) in hits {
                let act = if is_gated {
                    // Up: single dot product from mmap (not a matmul)
                    let up_score = up_view.row(feat).dot(&x_row);
                    let activated_gate = if use_gelu {
                        crate::ffn::gelu_tanh(gate_score)
                    } else {
                        gate_score * crate::ffn::sigmoid(gate_score)
                    };
                    activated_gate * up_score
                } else {
                    let mut v = gate_score;
                    if let Some(bias) = arch.ffn_up_bias_key(layer)
                        .and_then(|bk| self.weights.vectors.get(&bk))
                    {
                        if feat < bias.len() { v += bias[feat]; }
                    }
                    if use_gelu { crate::ffn::gelu_tanh(v) } else { v * crate::ffn::sigmoid(v) }
                };

                full_activation[[s, feat]] = act;

                if act.abs() > 1e-10 {
                    // Down: scaled vector add from mmap (not a matmul)
                    if let Some(override_down) = self.index.down_override(layer, feat) {
                        if override_down.len() == hidden {
                            let ov = ndarray::ArrayView1::from(override_down);
                            out_row.scaled_add(act, &ov);
                            continue;
                        }
                    }
                    let down_row = down_view.row(feat);
                    out_row.scaled_add(act, &down_row);
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
    /// Currently slower than exact path due to 3 separate mmap file reads.
    #[allow(dead_code)]
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

    /// KNN-direct walk: gate scores as activations + down from mmap.
    /// NOTE: Produces wrong answer without up projection (tested: Jack instead of Paris).
    /// Kept for future research when combined gate+up vectors are available.
    #[allow(dead_code)]
    ///
    /// Gate KNN scores = x @ gate_vectors^T = the gate projection.
    /// Apply SiLU activation. Multiply by down matrix. Done.
    /// No gate matmul from model weights. No up matmul. No GEGLU.
    /// Two BLAS gemms: gate_knn + down. Reads 205MB instead of 315MB.
    fn walk_ffn_knn_direct(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let down_view = self.index.down_layer_matrix(layer)?;
        let gate_scores = self.index.gate_scores_batch(layer, x)?;

        let arch = &*self.weights.arch;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // Gate scores → SiLU/GELU activation (no up projection)
        let activation = if use_gelu {
            gate_scores.mapv(crate::ffn::gelu_tanh)
        } else {
            gate_scores.mapv(|v| v * crate::ffn::sigmoid(v))
        };

        // activation[seq, intermediate] @ down[intermediate, hidden] → [seq, hidden]
        let mut out = larql_compute::matmul_gpu(&activation, &down_view, self.backend);

        if let Some(bias) = arch.ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

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

        // Q4 interleaved: preferred when GPU Q4 is available (Metal shader faster than BLAS).
        // CPU Q4 C kernel is slower than CPU BLAS at these dimensions — only use with GPU.
        if self.index.has_interleaved_q4() && self.backend.is_some_and(|be| be.has_q4()) {
            if let Some(result) = self.walk_ffn_q4_interleaved(layer, x) {
                return result;
            }
        }

        // f32 interleaved: gate+up+down contiguous per layer.
        if self.index.has_interleaved() {
            if let Some(result) = self.walk_ffn_interleaved(layer, x) {
                return result;
            }
        }

        // Full mmap walk: gate + up + down from 3 separate mmap files.
        // At high K (>50% intermediate), uses full mmap matmuls.
        // At low K (<50%), uses per-feature sparse walk.
        //
        if self.index.has_full_mmap_ffn() {
            let intermediate = self.index.num_features(layer);
            if intermediate > 0 && self.top_k * 2 < intermediate {
                // Low K: per-feature sparse (no matmul, graph walk)
                if let Some(result) = self.walk_ffn_sparse(layer, x) {
                    return result;
                }
            } else {
                // High K: full mmap matmuls (production path)
                if let Some(mut result) = self.walk_ffn_full_mmap(layer, x) {
                    // Apply down overrides from INSERT as post-hoc corrections.
                    // For each overridden feature, subtract the model's down contribution
                    // and add the override's down contribution using the same activation.
                    if self.index.has_overrides_at(layer) {
                        let hidden = x.shape()[1];
                        let seq_len = x.shape()[0];
                        let (ref mut out, ref activation) = result;
                        if let Some(down_view) = self.index.down_layer_matrix(layer) {
                            for s in 0..seq_len {
                                let mut out_row = out.row_mut(s);
                                // Check each overridden feature
                                for feat in 0..intermediate {
                                    if let Some(override_down) = self.index.down_override(layer, feat) {
                                        if override_down.len() != hidden { continue; }
                                        let act = activation[[s, feat]];
                                        if act.abs() <= 1e-10 { continue; }
                                        // Subtract original down contribution
                                        let orig_down = down_view.row(feat);
                                        out_row.scaled_add(-act, &orig_down);
                                        // Add override down contribution
                                        let ov = ndarray::ArrayView1::from(override_down);
                                        out_row.scaled_add(act, &ov);
                                    }
                                }
                            }
                        }
                    }
                    return result;
                }
            }
        }

        // Fallback: partial mmap (gate/up from model weights + down from mmap)
        if self.index.has_down_features() {
            return self.walk_ffn_exact(layer, x);
        }

        // Gate KNN needed only for sparse fallback (no mmap down)
        let features = self.index.gate_knn_batch(layer, x, self.top_k);

        // Fallback: sparse matmul against model weights
        let has_overrides = features.iter().any(|&f| self.index.down_override(layer, f).is_some());
        if has_overrides {
            let overrides: Vec<(usize, &[f32])> = features.iter()
                .filter_map(|&f| self.index.down_override(layer, f).map(|v| (f, v)))
                .collect();
            sparse_ffn_forward_with_overrides(self.weights, layer, x, &features, &overrides)
        } else {
            sparse_ffn_forward(self.weights, layer, x, &features)
        }
    }

    fn name(&self) -> &str {
        "walk"
    }
}
