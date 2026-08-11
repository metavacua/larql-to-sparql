//! TurboQuantEngine — WHT + Lloyd-Max K/V cache compression.
//!
//! Algorithm (ICLR 2026 style):
//!   1. Normalize vector → unit norm (store scalar)
//!   2. Walsh-Hadamard rotation (spreads coordinates to Beta distribution)
//!   3. Lloyd-Max scalar quantization (3 or 4 bits per coordinate)
//!   4. Bit-pack indices
//!   5. Decode: unpack → centroids → inverse WHT → rescale
//!
//! The `TurboQuantEngine` wraps this codec around the K/V cache.
//! Every decode path (W1-GPU dispatch, CPU walk, executor-routed,
//! legacy `decode_step`) is **append-only**: each step decompresses
//! the prior cache for attention (O(N)), then encodes ONLY the newly
//! produced K/V row head-by-head and appends its packed bytes
//! (`CompressedLayer::append_row`). Old rows' bytes are never
//! re-encoded — the codec's reconstruction norm is slightly below the
//! stored norm, so a decompress→re-encode cycle would multiply every
//! stored norm by that ratio and compound it across steps.
//!
//! Codec contract is the same in all paths: WHT + Lloyd-Max 3/4-bit
//! per scalar, bit-pack indices; per-row round-trip cos ≈ 0.9954 at
//! 4-bit / ≈ 0.980 at 3-bit on isotropic unit vectors (Gaussian
//! simulation, 2026-07-30).

// `TurboQuant` (stateless codec struct) is portable. Everything else
// here -- `CompressedLayer` and all its methods, `detect_head_dim`,
// `resolve_block_dim`, `compress_matrix`, `last_row`, `TurboQuantEngine`
// (`impl KvEngine`, upstream-gated) -- is native-only: despite not
// touching VectorIndex/Instant/spin_pool themselves, their only callers
// are the four `impl TurboQuantEngine` blocks below, all
// `#[cfg(not(target_arch = "wasm32"))]`-gated (CI-confirmed via wasm32
// clippy dead-code, not left "portable but uncalled").
#[cfg(not(target_arch = "wasm32"))]
use larql_compute::ComputeBackend;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::{cpu_engine_backend, EngineBackend};
#[cfg(not(target_arch = "wasm32"))]
use larql_vindex::VectorIndex;
#[cfg(not(target_arch = "wasm32"))]
use ndarray::{s, Array2};

use super::{codebooks, lloyd_max, packing, rotation};
#[cfg(not(target_arch = "wasm32"))]
use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
#[cfg(not(target_arch = "wasm32"))]
use crate::{EngineInfo, KvEngine};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::attention::SharedKV;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend,
};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::ffn::{BackendFfn, FfnBackend};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::forward::ple::precompute_per_layer_inputs;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::forward::{embed_tokens_pub, run_ffn};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::kv_engine::EngineError;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::model::ModelWeights;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

// ─── TurboQuant codec ────────────────────────────────────────────────────────

/// WHT + Lloyd-Max codec. Stateless — all operations are deterministic
/// functions of the input vector and the pre-computed codebook.
#[derive(Clone)]
pub struct TurboQuant {
    pub bits: u8, // 3 or 4
}

impl TurboQuant {
    pub fn new(bits: u8) -> Self {
        assert!(bits == 3 || bits == 4, "TurboQuant: bits must be 3 or 4");
        Self { bits }
    }

    /// Encode a single vector: normalize → WHT → quantize → pack.
    /// Returns a freshly-allocated `Vec<u8>` — kept for ergonomic API
    /// stability. Hot-path callers use [`encode_vector_into`] with
    /// reusable scratch buffers.
    pub fn encode_vector(&self, x: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bytes_per_vector(x.len()));
        let mut scratch_f32 = vec![0.0f32; x.len()];
        let mut scratch_u8 = Vec::with_capacity(x.len());
        self.encode_vector_into(x, &mut out, &mut scratch_f32, &mut scratch_u8);
        out
    }

    /// Encode into a caller-provided byte buffer using caller-provided
    /// scratch. `scratch_f32` and `scratch_u8` are resized as needed
    /// and may be reused across calls to amortise allocation.
    ///
    /// 2026-05-19 codec hot-path optimisation: hoists the per-call
    /// allocations from [`encode_vector`] (x_hat, WHT output, indices)
    /// into a scratch pair the caller can keep alive across the
    /// compress_matrix loop. Together with [`rotation::wht_inplace`]'s
    /// NEON path this is the recompute_hot win.
    pub fn encode_vector_into(
        &self,
        x: &[f32],
        out: &mut Vec<u8>,
        scratch_f32: &mut Vec<f32>,
        scratch_u8: &mut Vec<u8>,
    ) {
        let d = x.len();
        scratch_f32.resize(d, 0.0);
        scratch_u8.clear();
        scratch_u8.reserve(d);

        let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-12 {
            let inv = 1.0 / norm;
            for (i, &v) in x.iter().enumerate() {
                scratch_f32[i] = v * inv;
            }
        } else {
            for v in scratch_f32.iter_mut() {
                *v = 0.0;
            }
        }
        rotation::wht_inplace(scratch_f32);
        // Coordinates of the rotated unit vector have std exactly 1/√d;
        // dividing by that sigma maps them into the N(0, 1) space the
        // unit codebook is trained on.
        let codebook = codebooks::unit_codebook(self.bits);
        let inv_sigma = 1.0 / codebooks::wht_coordinate_sigma(d);
        for &val in scratch_f32.iter() {
            scratch_u8.push(lloyd_max::quantize_scalar(val * inv_sigma, &codebook));
        }
        out.extend_from_slice(&norm.to_le_bytes());
        packing::pack_indices(scratch_u8, self.bits, out);
    }

    /// Decode a single vector: unpack → centroids → inverse WHT → rescale.
    /// Returns a freshly-allocated `Vec<f32>` — kept for ergonomic API
    /// stability. Hot-path callers use [`decode_vector_into`].
    pub fn decode_vector(&self, encoded: &[u8], dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; dim];
        let mut scratch_u8 = Vec::with_capacity(dim);
        self.decode_vector_into(encoded, dim, &mut out, &mut scratch_u8);
        out
    }

    /// Decode into a caller-provided f32 buffer using caller-provided
    /// scratch. `out` is resized to `dim`; `scratch_u8` is reused for
    /// the unpacked-index intermediate.
    pub fn decode_vector_into(
        &self,
        encoded: &[u8],
        dim: usize,
        out: &mut Vec<f32>,
        scratch_u8: &mut Vec<u8>,
    ) {
        let norm = f32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        scratch_u8.clear();
        packing::unpack_indices_into(&encoded[4..], dim, self.bits, scratch_u8);
        let codebook = codebooks::unit_codebook(self.bits);
        out.resize(dim, 0.0);
        for (i, &idx) in scratch_u8.iter().enumerate() {
            out[i] = codebook.centroids[idx as usize];
        }
        rotation::wht_inplace(out);
        // The WHT is linear, so the sigma scaling (undoing encode's
        // ×√d) commutes with it and folds into the norm restore.
        let scale = norm * codebooks::wht_coordinate_sigma(dim);
        for v in out.iter_mut() {
            *v *= scale;
        }
    }

    pub fn bytes_per_vector(&self, dim: usize) -> usize {
        4 + packing::packed_size(dim, self.bits)
    }
}

// ─── Compressed K/V layer ────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
pub(super) struct CompressedLayer {
    pub compressed_k: Vec<u8>,
    pub compressed_v: Vec<u8>,
    pub num_vecs: usize,
    pub kv_dim: usize,
    /// Largest power-of-two head dimension detected from kv_dim.
    pub head_dim: usize,
}

#[cfg(not(target_arch = "wasm32"))]
impl CompressedLayer {
    pub(super) fn compress(kv: &SharedKV, tq: &TurboQuant) -> Self {
        let (k, v) = kv;
        let num_vecs = k.shape()[0];
        let kv_dim = k.shape()[1];
        let head_dim = detect_head_dim(kv_dim);
        Self {
            compressed_k: compress_matrix(k, tq, head_dim),
            compressed_v: compress_matrix(v, tq, head_dim),
            num_vecs,
            kv_dim,
            head_dim,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn decompress(&self, tq: &TurboQuant) -> SharedKV {
        let k = decompress_matrix(
            &self.compressed_k,
            self.num_vecs,
            self.kv_dim,
            self.head_dim,
            tq,
        );
        let v = decompress_matrix(
            &self.compressed_v,
            self.num_vecs,
            self.kv_dim,
            self.head_dim,
            tq,
        );
        (k, v)
    }

    /// Append-only ingest of one new K/V row: encode `k_row` / `v_row`
    /// head-by-head and push the packed bytes onto the existing
    /// compressed buffers. Invariant: existing rows' bytes are never
    /// re-encoded — a decompress→re-encode cycle multiplies each stored
    /// norm by the codec's reconstruction-norm ratio (< 1) and
    /// compounds it across decode steps.
    pub(super) fn append_row(
        &mut self,
        k_row: &[f32],
        v_row: &[f32],
        tq: &TurboQuant,
        scratch_f32: &mut Vec<f32>,
        scratch_u8: &mut Vec<u8>,
    ) {
        debug_assert_eq!(k_row.len(), self.kv_dim, "K row width != layer kv_dim");
        debug_assert_eq!(v_row.len(), self.kv_dim, "V row width != layer kv_dim");
        for chunk in k_row.chunks(self.head_dim) {
            tq.encode_vector_into(chunk, &mut self.compressed_k, scratch_f32, scratch_u8);
        }
        for chunk in v_row.chunks(self.head_dim) {
            tq.encode_vector_into(chunk, &mut self.compressed_v, scratch_f32, scratch_u8);
        }
        self.num_vecs += 1;
    }

    /// Drop rows until the layer holds `rows` of them again.
    ///
    /// Byte-exact, and that is not an accident of the codec being good: rows
    /// are appended as whole head-chunks at fixed byte offsets and existing
    /// bytes are never re-encoded (see [`Self::append_row`]), so removing the
    /// tail restores precisely the buffer that preceded it. The compression is
    /// lossy against its *input*, not against what was stored — which is what
    /// lets a K/V-canonical engine rewind at all.
    ///
    /// No-op when `rows` is not smaller than the current count, so a caller
    /// rewinding a layer the failure never reached costs nothing.
    pub(super) fn truncate_rows(&mut self, rows: usize, tq: &TurboQuant) {
        if rows >= self.num_vecs {
            return;
        }
        let heads = self.kv_dim / self.head_dim.max(1);
        let bytes = rows * heads * tq.bytes_per_vector(self.head_dim);
        self.compressed_k.truncate(bytes);
        self.compressed_v.truncate(bytes);
        self.num_vecs = rows;
    }

    pub(super) fn memory_bytes(&self) -> usize {
        self.compressed_k.len() + self.compressed_v.len()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn detect_head_dim(kv_dim: usize) -> usize {
    for &hd in &[256usize, 128, 64, 32] {
        if kv_dim.is_multiple_of(hd) {
            return hd;
        }
    }
    kv_dim // fallback: treat whole row as one head
}

/// Fallible companion to [`detect_head_dim`]: the WHT butterfly requires a
/// power-of-two block dim, so a kv_dim with no supported head split (e.g.
/// 80) must surface as a typed error at prefill entry instead of the WHT
/// assert firing mid-prefill.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn resolve_block_dim(kv_dim: usize) -> Result<usize, EngineError> {
    let head_dim = detect_head_dim(kv_dim);
    if head_dim.is_power_of_two() {
        Ok(head_dim)
    } else {
        Err(EngineError::InvariantViolation {
            what: format!(
                "turbo-quant codec requires a power-of-two K/V block dim \
                 (WHT constraint); kv_dim={kv_dim} has no supported head split"
            ),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn compress_matrix(m: &Array2<f32>, tq: &TurboQuant, head_dim: usize) -> Vec<u8> {
    let rows = m.shape()[0];
    let cols = m.shape()[1];
    let heads_per_row = cols / head_dim;
    let mut buf = Vec::with_capacity(rows * heads_per_row * tq.bytes_per_vector(head_dim));
    // Hot-path scratch reused across every chunk. Eliminates the
    // per-call Vec churn that 2026-05-19 diagnostics flagged as the
    // codec's second-biggest cost (after the WHT butterfly itself).
    let mut scratch_f32 = Vec::with_capacity(head_dim);
    let mut scratch_u8 = Vec::with_capacity(head_dim);
    for row in m.rows() {
        let row_slice = row.as_slice().expect("non-contiguous row");
        for chunk in row_slice.chunks(head_dim) {
            tq.encode_vector_into(chunk, &mut buf, &mut scratch_f32, &mut scratch_u8);
        }
    }
    buf
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn decompress_matrix(
    bytes: &[u8],
    num_vecs: usize,
    kv_dim: usize,
    head_dim: usize,
    tq: &TurboQuant,
) -> Array2<f32> {
    let heads_per_vec = kv_dim / head_dim;
    let bytes_per_head = tq.bytes_per_vector(head_dim);
    let mut data = vec![0.0f32; num_vecs * kv_dim];
    // The per-vector WHT/codebook decode (`decode_vector_into`) is the per-step
    // bottleneck (a `/usr/bin/sample` profile put ~35% of the decode driver in
    // here, serial). Each vector writes a disjoint `kv_dim`-wide row, so fan it
    // across the spin pool — this keeps the cache COMPRESSED (the engine's
    // point: still decoded every step) but makes the decode parallel instead of
    // single-threaded. Per-chunk scratch (decode needs mutable scratch),
    // amortised over `CHUNK_VECS` vectors so it isn't reallocated per (vec,head).
    const CHUNK_VECS: usize = 8;
    larql_compute::cpu::spin_pool::par_chunks_mut(&mut data, kv_dim * CHUNK_VECS, |ci, chunk| {
        let mut decoded = Vec::with_capacity(head_dim);
        let mut scratch_u8 = Vec::with_capacity(head_dim);
        let base_vec = ci * CHUNK_VECS;
        let vecs_in_chunk = chunk.len() / kv_dim;
        for v in 0..vecs_in_chunk {
            let i = base_vec + v;
            for h in 0..heads_per_vec {
                let offset = (i * heads_per_vec + h) * bytes_per_head;
                tq.decode_vector_into(
                    &bytes[offset..offset + bytes_per_head],
                    head_dim,
                    &mut decoded,
                    &mut scratch_u8,
                );
                let row_start = v * kv_dim + h * head_dim;
                chunk[row_start..row_start + head_dim].copy_from_slice(&decoded);
            }
        }
    });
    Array2::from_shape_vec((num_vecs, kv_dim), data).expect("shape mismatch")
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

// ─── Engine ──────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
pub struct TurboQuantEngine {
    pub(super) tq: TurboQuant,
    pub(super) backend: Box<dyn EngineBackend>,
    pub(super) layers: Vec<CompressedLayer>,
    pub(super) abs_position: usize,
    pub(super) profiling: bool,
    pub(super) profile: crate::profiler::EngineProfiler,
    /// W1-GPU: handle into the backend's internal K/V cache, populated
    /// when prefill routes through `coarse_prefill_with_state`. `None`
    /// means the engine took the legacy per-layer walk path.
    pub(super) kv_handle: Option<larql_inference::KvHandle>,
    /// Engine-owned f32 dequant scratch for the per-layer fallback (see
    /// `MarkovResidualEngine::dequant_scratch`). Keeps `weights` immutable.
    pub(super) dequant_scratch: larql_inference::DequantScratch,
}

#[cfg(not(target_arch = "wasm32"))]
impl TurboQuantEngine {
    pub fn new(bits: u8) -> Self {
        Self::with_backend(bits, cpu_engine_backend())
    }

    pub fn with_backend(bits: u8, backend: Box<dyn EngineBackend>) -> Self {
        Self {
            tq: TurboQuant::new(bits),
            backend,
            layers: Vec::new(),
            abs_position: 0,
            profiling: false,
            profile: crate::profiler::EngineProfiler::default(),
            kv_handle: None,
            dequant_scratch: larql_inference::DequantScratch::new(),
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    /// First-use validation of the codec's power-of-two block-dim
    /// requirement across all layers. Called from every prefill entry
    /// point; decode paths inherit the guarantee from a successful
    /// prefill.
    fn validate_block_dims(&self, weights: &ModelWeights) -> Result<(), EngineError> {
        let arch = &*weights.arch;
        for layer in 0..weights.num_layers {
            let kv_dim = arch.num_kv_heads_for_layer(layer) * arch.head_dim_for_layer(layer);
            resolve_block_dim(kv_dim)?;
        }
        Ok(())
    }
}

// W1-GPU dispatch methods (`try_prefill_via_dispatch` /
// `decode_step_via_dispatch`) live in [`super::dispatch`] as an
// additional `impl TurboQuantEngine` block. They mutate the
// `pub(super)` fields above.

#[cfg(not(target_arch = "wasm32"))]
impl TurboQuantEngine {
    /// Shared body for `decode_step` / `decode_step_resident`.
    ///
    /// **Transactional.** Unlike the residual-canonical engines, this one's
    /// canonical state *is* the K/V: each layer's compressed cache grows
    /// before the FFN gets its chance to refuse, so a step that does not
    /// finish must undo those appends rather than leave a cache holding a
    /// token that produced no output. [`CompressedLayer::truncate_rows`] does
    /// that byte-exactly, and `abs_position` advances only on success — so a
    /// caller who fixes the cause can drive the same token again.
    fn decode_step_impl(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        // Recorded per layer rather than assumed uniform: nothing promises
        // every layer caches the same number of rows, and a wrong assumption
        // would rewind to a length no layer ever had.
        let entry_rows: Vec<usize> = self.layers.iter().map(|l| l.num_vecs).collect();
        match self.decode_step_appending(weights, ffn, token_id, index) {
            Ok(hidden) => Ok(hidden),
            Err(failure) => {
                for (layer, &rows) in self.layers.iter_mut().zip(&entry_rows) {
                    layer.truncate_rows(rows, &self.tq);
                }
                Err(failure)
            }
        }
    }

    /// The body of a decode step, which appends to `self.layers` as it goes.
    ///
    /// Split out so the rewind above can wrap every exit rather than every
    /// `?` having to remember it.
    fn decode_step_appending(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        let num_layers = weights.num_layers;
        let abs_position = self.abs_position;
        let mut h = embed_tokens_pub(weights, &[token_id]);
        // PLE inputs are per-token — recompute for this single-token decode
        // step, matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);
        // Codec scratch reused across layers.
        let mut scratch_f32: Vec<f32> = Vec::new();
        let mut scratch_u8: Vec<u8> = Vec::new();

        for layer in 0..num_layers {
            // Decompress full prior K/V for attention.
            let prior_kv = self.layers[layer].decompress(&self.tq);

            // Decode step returns updated K/V (prior + new token).
            let (h_post_attn, updated_kv) =
                larql_inference::attention::run_attention_block_decode_step_auto(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    &h,
                    layer,
                    Some(&prior_kv),
                    abs_position,
                    Some(self.backend.as_ref()),
                    index.map(|v| v as &dyn larql_compute::KvIndex),
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "run_attention_block_decode_step_backend returned None".into(),
                })?;

            // Append-only codec path: encode just the new row head-by-
            // head and push onto the existing compressed buffer.
            let layer_slot = &mut self.layers[layer];
            let new_rows = updated_kv.0.shape()[0];
            debug_assert_eq!(new_rows, layer_slot.num_vecs + 1, "decode adds one row");
            let k_last = updated_kv.0.row(new_rows - 1).to_owned();
            let v_last = updated_kv.1.row(new_rows - 1).to_owned();
            layer_slot.append_row(
                k_last.as_slice().expect("k row contig"),
                v_last.as_slice().expect("v row contig"),
                &self.tq,
                &mut scratch_f32,
                &mut scratch_u8,
            );

            let bffn = BackendFfn {
                weights,
                backend: self.backend.as_ref(),
            };
            h = crate::engines::layer_ffn_or_moe(
                weights,
                &h_post_attn,
                layer,
                &bffn,
                Some(ffn),
                ple_inputs.get(layer),
            )
            .map_err(EngineError::Execution)?;
        }

        self.abs_position += 1;
        Ok(last_row(&h))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl KvEngine for TurboQuantEngine {
    fn name(&self) -> &str {
        "turbo-quant"
    }

    fn info(&self) -> EngineInfo {
        let mem: usize = self.layers.iter().map(|l| l.memory_bytes()).sum();
        EngineInfo {
            name: "turbo-quant".into(),
            description: format!(
                "{}-bit WHT+Lloyd-Max K/V compression (mem={:.1}MB)",
                self.tq.bits,
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config: format!("bits={}", self.tq.bits),
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.validate_block_dims(weights)?;
        let num_layers = weights.num_layers;
        let be = Some(self.backend.as_compute());
        let mut h = embed_tokens_pub(weights, token_ids);
        // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
        // Built into a local, not into `self.layers`: a prefill that refuses
        // partway must leave the engine holding whatever cache it already had
        // rather than a truncated one for a prompt it never finished.
        let mut layers: Vec<CompressedLayer> = Vec::with_capacity(num_layers);

        for layer in 0..num_layers {
            let (h_post_attn, k, v) = run_attention_with_kv_backend(
                larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                &h,
                layer,
                be,
                None,
            )
            .ok_or_else(|| EngineError::BackendFailure {
                details: "run_attention_with_kv_backend returned None".into(),
            })?;
            layers.push(CompressedLayer::compress(&(k, v), &self.tq));

            let bffn = BackendFfn {
                weights,
                backend: self.backend.as_ref(),
            };
            h = crate::engines::layer_ffn_or_moe(
                weights,
                &h_post_attn,
                layer,
                &bffn,
                Some(ffn),
                ple_inputs.get(layer),
            )
            .map_err(EngineError::Execution)?;
        }

        self.layers = layers;
        self.abs_position = token_ids.len();
        Ok(last_row(&h))
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_impl(weights, ffn, token_id, None)
    }

    /// Resident-path decode: threads `index` to the attention step's
    /// Q4K-direct route (the non-standard-engine structural-gap fix).
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.decode_step_impl(weights, ffn, token_id, Some(index))
    }

    fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    fn stage_summary(&self) -> Option<crate::DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(self.profile.summary("turbo-quant", self.backend.name()))
    }

    /// Quant path: always run the per-layer compression cycle (capture
    /// K/V per layer, WHT+Lloyd-Max encode, decompress prior, etc.).
    /// W1-GPU: when the engine's backend supports `coarse_prefill_with_state`,
    /// route through the dispatch path — backend computes K/V on GPU,
    /// engine compresses the per-layer captured state into
    /// `CompressedLayer` entries. Falls back to the legacy CPU walk
    /// (`prefill_quant_cpu`) for backends without state-capture support.
    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.validate_block_dims(weights)?;
        if let Some(hidden) = self.try_prefill_via_dispatch(weights, index, token_ids) {
            return Ok(hidden);
        }
        self.kv_handle = None;
        let out = self
            .prefill_quant_cpu(weights, index, token_ids, backend)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "prefill_quant_cpu returned None".into(),
            })?;
        self.abs_position = token_ids.len();
        Ok(out)
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        if self.kv_handle.is_some() {
            return self
                .decode_step_via_dispatch(weights, index, token_id)
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "decode_step_via_dispatch returned None".into(),
                });
        }
        self.decode_step_quant_cpu(weights, index, token_id, backend)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "decode_step_quant_cpu returned None".into(),
            })
    }

    // ── Executor-aware migration (Phase 2 of engine-state-vs-execution spec) ──
    //
    // The legacy `prefill_quant_cpu` / `decode_step_quant_cpu` paths construct
    // their own `WalkFfn` and ignore the FFN parameter. The methods below
    // drive the per-layer loop through a caller-supplied `LayerExecutor` and
    // honor the FFN dispatcher — required for `larql bench --ffn
    // http://shard:8080` to route through the remote shard.
    //
    // Compression policy (WHT + Lloyd-Max per layer) is engine state and
    // stays here; only the per-layer compute is delegated.
    fn prefill_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.validate_block_dims(weights)?;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let num_layers = weights.num_layers;
        let mut h = embed_tokens_pub(weights, token_ids);
        // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
        self.layers.clear();

        for layer in 0..num_layers {
            let (h_out, kv) = executor
                .run_prefill_layer(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    layer,
                    &h,
                    ffn,
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "executor.run_prefill_layer returned None".into(),
                })?;
            self.layers.push(CompressedLayer::compress(&kv, &self.tq));
            // `LayerExecutor::run_*_layer` returns attention + bare FFN only
            // (`LocalWalkExecutor`, the sole production impl, ends at
            // `run_ffn`); the PLE + layer_scalar tail is the driving loop's
            // responsibility, mirroring the legacy `kv_prefill_run` sequence.
            h = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_out,
                layer,
                ple_inputs.get(layer),
            );
        }

        self.abs_position = token_ids.len();
        Ok(last_row(&h))
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let num_layers = weights.num_layers;
        let abs_position = self.abs_position;
        let mut h = embed_tokens_pub(weights, &[token_id]);
        // PLE inputs are per-token — recompute for this single-token decode
        // step, matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);
        // Codec scratch reused across layers.
        let mut scratch_f32: Vec<f32> = Vec::new();
        let mut scratch_u8: Vec<u8> = Vec::new();

        for layer in 0..num_layers {
            let prior_kv = self.layers[layer].decompress(&self.tq);
            let (h_out, updated_kv) = executor
                .run_decode_layer(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    layer,
                    &h,
                    &prior_kv,
                    abs_position,
                    ffn,
                )
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "executor.run_decode_layer returned None".into(),
                })?;
            // Append-only codec path (same structure as `decode_step_impl`
            // and `decode_step_quant_cpu`): only the LAST row of the
            // executor's updated K/V is new — the prior rows came from
            // decompressing our own cache, and re-encoding decompressed
            // rows compounds the codec's norm shrink every step.
            let layer_slot = &mut self.layers[layer];
            let new_rows = updated_kv.0.shape()[0];
            debug_assert_eq!(new_rows, layer_slot.num_vecs + 1, "decode adds one row");
            let k_last = updated_kv.0.row(new_rows - 1).to_owned();
            let v_last = updated_kv.1.row(new_rows - 1).to_owned();
            layer_slot.append_row(
                k_last.as_slice().expect("k row contig"),
                v_last.as_slice().expect("v row contig"),
                &self.tq,
                &mut scratch_f32,
                &mut scratch_u8,
            );
            // Executor returns bare post-FFN hidden; PLE + layer_scalar tail
            // is the driving loop's responsibility (see prefill loop above).
            h = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_out,
                layer,
                ple_inputs.get(layer),
            );
        }

        self.abs_position += 1;
        Ok(last_row(&h))
    }
}

// ── CPU quant-path helper methods (not part of the KvEngine trait) ───────────

#[cfg(not(target_arch = "wasm32"))]
impl TurboQuantEngine {
    fn prefill_quant_cpu(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let num_layers = weights.num_layers;
        let be = Some(backend);
        let mut h = embed_tokens_pub(weights, token_ids);
        // Empty on non-PLE archs — `ple_inputs.get(layer)` then yields `None`.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
        self.layers.clear();

        // Hoist WalkFfn — was rebuilt 34× per prefill.
        let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
            .with_backend(backend);

        for layer in 0..num_layers {
            let (h_post_attn, k, v) = run_attention_with_kv_backend(
                larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                &h,
                layer,
                be,
                None,
            )?;
            self.layers
                .push(CompressedLayer::compress(&(k, v), &self.tq));

            // Native-quantised FFN; falls back to WalkFfn → dense f32. Both
            // branches return the bare post-FFN hidden —
            // `ffn_decode_step_native` is also `moe_ffn_block_cpu`'s pre-PLE
            // dense slab — so the PLE + layer_scalar tail applies to either.
            let h_post_ffn = larql_inference::vindex::ffn_decode_step_native(
                weights,
                index,
                backend,
                &h_post_attn,
                layer,
            )
            .unwrap_or_else(|| {
                let (h, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
                h
            });
            h = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_post_ffn,
                layer,
                ple_inputs.get(layer),
            );
        }

        self.abs_position = token_ids.len();
        Some(last_row(&h))
    }

    fn decode_step_quant_cpu(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Option<Array2<f32>> {
        use std::time::Instant;
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        let num_layers = weights.num_layers;
        let abs_position = self.abs_position;
        let timing = self.profiling;
        let t_step = if timing { Some(Instant::now()) } else { None };

        let t_embed = if timing { Some(Instant::now()) } else { None };
        let mut h = embed_tokens_pub(weights, &[token_id]);
        let embed_us = t_embed
            .map(|t| t.elapsed().as_secs_f64() * 1e6)
            .unwrap_or(0.0);
        // PLE inputs are per-token — recompute for this single-token decode
        // step, matching the legacy `kv_decode_step_run` recipe exactly.
        let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);

        // Hoist WalkFfn — was rebuilt 34× per decode step.
        let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
            .with_backend(backend);
        // Codec scratch reused across layers.
        let mut scratch_f32: Vec<f32> = Vec::new();
        let mut scratch_u8: Vec<u8> = Vec::new();

        // Per-stage accumulators. For turbo_quant we reuse the existing
        // EngineProfiler slots:
        //   `recompute_hot`  ← codec **decode** (decompress prior K/V)
        //   `recompute_cold` ← codec **encode** (re-encode updated K/V)
        // Semantically these are the per-step codec work that the
        // engine's contract requires; print labels them "recompute_kv
        // (hot/cold)" but for this engine the meaning is decode/encode.
        let mut codec_decode_us = 0.0f64;
        let mut codec_encode_us = 0.0f64;
        let mut attention_us = 0.0f64;
        let mut ffn_us = 0.0f64;

        for layer in 0..num_layers {
            let t_dec = if timing { Some(Instant::now()) } else { None };
            let prior_kv = self.layers[layer].decompress(&self.tq);
            if let Some(t) = t_dec {
                codec_decode_us += t.elapsed().as_secs_f64() * 1e6;
            }

            let t_attn = if timing { Some(Instant::now()) } else { None };
            let (h_post_attn, updated_kv) = larql_inference::vindex::attention_decode_step_native(
                weights,
                index,
                backend,
                &h,
                layer,
                Some(&prior_kv),
                abs_position,
            )
            .or_else(|| {
                run_attention_block_decode_step_backend(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    &h,
                    layer,
                    Some(&prior_kv),
                    abs_position,
                    Some(backend),
                )
            })?;
            if let Some(t) = t_attn {
                attention_us += t.elapsed().as_secs_f64() * 1e6;
            }

            let t_enc = if timing { Some(Instant::now()) } else { None };
            // Append-only codec path (mirrors `dispatch.rs`'s 2026-05-19
            // fix). The attention call returns the full updated K/V
            // (prior + new); only the LAST row is new, the rest already
            // live in `self.layers[layer].compressed_{k,v}`. Encode just
            // the new row head-by-head and push onto the existing
            // compressed buffer. Per-step compress drops from O(N) to
            // O(head_dim · heads_per_row).
            let layer_slot = &mut self.layers[layer];
            let new_rows = updated_kv.0.shape()[0];
            debug_assert_eq!(new_rows, layer_slot.num_vecs + 1, "decode adds one row");
            let k_last = updated_kv.0.row(new_rows - 1).to_owned();
            let v_last = updated_kv.1.row(new_rows - 1).to_owned();
            layer_slot.append_row(
                k_last.as_slice().expect("k row contig"),
                v_last.as_slice().expect("v row contig"),
                &self.tq,
                &mut scratch_f32,
                &mut scratch_u8,
            );
            if let Some(t) = t_enc {
                codec_encode_us += t.elapsed().as_secs_f64() * 1e6;
            }

            let t_ffn = if timing { Some(Instant::now()) } else { None };
            // Both branches return the bare post-FFN hidden —
            // `ffn_decode_step_native` is also `moe_ffn_block_cpu`'s pre-PLE
            // dense slab — so the PLE + layer_scalar tail applies to either.
            let h_post_ffn = larql_inference::vindex::ffn_decode_step_native(
                weights,
                index,
                backend,
                &h_post_attn,
                layer,
            )
            .unwrap_or_else(|| {
                let (h, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
                h
            });
            let h_out = crate::engines::apply_ple_and_layer_scalar(
                weights,
                &h_post_ffn,
                layer,
                ple_inputs.get(layer),
            );
            if let Some(t) = t_ffn {
                ffn_us += t.elapsed().as_secs_f64() * 1e6;
            }
            h = h_out;
        }

        if let Some(t_step) = t_step {
            let p = &mut self.profile;
            p.embed.total_us += embed_us;
            p.embed.count += 1;
            p.recompute_hot.total_us += codec_decode_us;
            p.recompute_hot.count += 1;
            p.attention.total_us += attention_us;
            p.attention.count += 1;
            p.recompute_cold.total_us += codec_encode_us;
            p.recompute_cold.count += 1;
            p.ffn.total_us += ffn_us;
            p.ffn.count += 1;
            p.decode_total.total_us += t_step.elapsed().as_secs_f64() * 1e6;
            p.decode_total.count += 1;
        }

        self.abs_position += 1;
        Some(last_row(&h))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accuracy::cosine_similarity;

    /// TurboQuant's codebooks are optimised for unit-norm vectors (the natural
    /// distribution of K/V heads after QK-norm), so unit-norm inputs measure
    /// the codec at its operating point (cos ≈ 0.9954 at 4-bit).
    /// Generate a unit-norm vector using a simple LCG (no external rand dep).
    /// Uses lower 32 bits of the state for uniform [0, 1) values.
    fn unit_norm_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let raw: Vec<f32> = (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0
            })
            .collect();
        let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-12 {
            raw.iter().map(|v| v / norm).collect()
        } else {
            raw
        }
    }

    // ── Measured codec quality floors (2026-07-30, unit-sigma codebooks) ─────
    //
    // Mean round-trip values over 500 Gaussian-direction unit vectors
    // (`codec_mean_cosine_and_norm_meet_floor`):
    //   4-bit: mean cos 0.9954–0.9960 across d ∈ {32, 64, 128, 256}
    //   3-bit: mean cos 0.9830–0.9844
    //   mean decoded norm ≥ 0.9836 at both bit-widths
    // The mis-scaled codebooks (sigma trained √2 too small + silent D256
    // fallback for d ∉ {128, 256}) gave 0.9908 at 4-bit d=256, 0.947 at
    // d=64, 0.919 at d=32, and decoded norms of 0.95 / 0.70 / 0.55.
    // Floors sit just under today's measured values so a codebook
    // regression trips the assert instead of hiding under slack.
    const MEAN_COS_FLOOR_4BIT: f64 = 0.995;
    const MEAN_COS_FLOOR_3BIT: f64 = 0.982;
    const MEAN_NORM_FLOOR: f64 = 0.98;
    /// Single-vector floors for the LCG fixtures below (measured
    /// 2026-07-30: 0.9955 / 0.9956 at 4-bit, 0.9839 at 3-bit). The LCG
    /// produces structured low-bit coordinates, so these are
    /// deterministic per (dim, seed).
    const MIN_COS_4BIT: f64 = 0.995;
    const MIN_COS_3BIT: f64 = 0.982;

    /// Gaussian-direction unit vectors at every supported block dim.
    /// Pins the accuracy floor AND the absence of the old
    /// d ∉ {128, 256} fallback cliff.
    #[test]
    fn codec_mean_cosine_and_norm_meet_floor() {
        use rand::prelude::*;
        use rand_distr::Normal;
        const N_VECTORS: usize = 500;
        for bits in [3u8, 4] {
            for d in [32usize, 64, 128, 256] {
                let tq = TurboQuant::new(bits);
                let mut rng = StdRng::seed_from_u64(777);
                let dist = Normal::new(0.0f32, 1.0).unwrap();
                let mut cos_sum = 0.0f64;
                let mut norm_sum = 0.0f64;
                for _ in 0..N_VECTORS {
                    let raw: Vec<f32> = (0..d).map(|_| rng.sample(dist)).collect();
                    let nrm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let x: Vec<f32> = raw.iter().map(|v| v / nrm).collect();
                    let dec = tq.decode_vector(&tq.encode_vector(&x), d);
                    cos_sum += cosine_similarity(&x, &dec);
                    norm_sum += dec.iter().map(|v| v * v).sum::<f32>().sqrt() as f64;
                }
                let mean_cos = cos_sum / N_VECTORS as f64;
                let mean_norm = norm_sum / N_VECTORS as f64;
                let cos_floor = match bits {
                    4 => MEAN_COS_FLOOR_4BIT,
                    _ => MEAN_COS_FLOOR_3BIT,
                };
                assert!(
                    mean_cos > cos_floor,
                    "bits={bits} d={d}: mean cos {mean_cos:.4} < floor {cos_floor}"
                );
                assert!(
                    mean_norm > MEAN_NORM_FLOOR,
                    "bits={bits} d={d}: mean decoded norm {mean_norm:.4} < floor {MEAN_NORM_FLOOR}"
                );
            }
        }
    }

    // ── Codec roundtrip quality ───────────────────────────────────────────────

    #[test]
    fn encode_decode_4bit_cosine_near_one() {
        let tq = TurboQuant::new(4);
        let x = unit_norm_vec(256, 42);
        let enc = tq.encode_vector(&x);
        let dec = tq.decode_vector(&enc, 256);
        let cos = cosine_similarity(&x, &dec);
        assert!(cos > MIN_COS_4BIT, "4-bit cosine {cos:.4} < {MIN_COS_4BIT}");
    }

    #[test]
    fn encode_decode_3bit_cosine_acceptable() {
        let tq = TurboQuant::new(3);
        let x = unit_norm_vec(256, 99);
        let enc = tq.encode_vector(&x);
        let dec = tq.decode_vector(&enc, 256);
        let cos = cosine_similarity(&x, &dec);
        assert!(cos > MIN_COS_3BIT, "3-bit cosine {cos:.4} < {MIN_COS_3BIT}");
    }

    #[test]
    fn encode_decode_dim128_roundtrip() {
        let tq = TurboQuant::new(4);
        let x = unit_norm_vec(128, 7);
        let enc = tq.encode_vector(&x);
        let dec = tq.decode_vector(&enc, 128);
        let cos = cosine_similarity(&x, &dec);
        assert!(
            cos > MIN_COS_4BIT,
            "4-bit d=128 cosine {cos:.4} < {MIN_COS_4BIT}"
        );
    }

    /// d = 64 and d = 32 used to fall back to the D256 table silently
    /// (cos 0.92–0.95); the sigma-scaled unit codebook must serve them
    /// at full quality. Per-vector cosine variance grows as d shrinks
    /// (a couple of tail coordinates dominate at d = 32), so the
    /// single-fixture floor sits below the d ≥ 128 one: measured
    /// 2026-07-30 at 0.9967 (d=64, seed 21) and 0.9762 (d=32, seed 5).
    const MIN_COS_4BIT_SMALL_DIM: f64 = 0.975;

    #[test]
    fn encode_decode_small_dims_full_quality() {
        let tq = TurboQuant::new(4);
        for (dim, seed) in [(64usize, 21u64), (32, 5)] {
            let x = unit_norm_vec(dim, seed);
            let enc = tq.encode_vector(&x);
            let dec = tq.decode_vector(&enc, dim);
            let cos = cosine_similarity(&x, &dec);
            assert!(
                cos > MIN_COS_4BIT_SMALL_DIM,
                "4-bit d={dim} cosine {cos:.4} < {MIN_COS_4BIT_SMALL_DIM}"
            );
        }
    }

    #[test]
    fn norm_approximately_preserved() {
        let tq = TurboQuant::new(4);
        let x = unit_norm_vec(256, 13);
        let norm_orig: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let enc = tq.encode_vector(&x);
        let dec = tq.decode_vector(&enc, 256);
        let norm_dec: f32 = dec.iter().map(|v| v * v).sum::<f32>().sqrt();
        let ratio = norm_dec / norm_orig;
        // The codec stores the norm explicitly; the residual shortfall is
        // quantisation error only (measured ratio ≈ 0.995 at 4-bit,
        // 2026-07-30 — the mis-scaled codebooks clipped it to ≈ 0.95).
        const NORM_RATIO_TOL: f32 = 0.02;
        assert!(
            (ratio - 1.0).abs() < NORM_RATIO_TOL,
            "norm ratio {ratio:.4} outside 1.0 ± {NORM_RATIO_TOL}"
        );
    }

    #[test]
    fn zero_vector_roundtrip_no_panic() {
        let tq = TurboQuant::new(4);
        let x = vec![0.0f32; 256];
        let enc = tq.encode_vector(&x);
        let dec = tq.decode_vector(&enc, 256);
        // Zero vector: all decoded values should be ~0 (codec stores norm=0).
        let max_abs = dec.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-6,
            "zero vector decoded to non-zero: max_abs={max_abs}"
        );
    }

    #[test]
    fn identical_vectors_same_encoding() {
        let tq = TurboQuant::new(4);
        let x = unit_norm_vec(256, 55);
        let enc1 = tq.encode_vector(&x);
        let enc2 = tq.encode_vector(&x);
        assert_eq!(enc1, enc2, "encoding is not deterministic");
    }

    // ── Encoded byte size ────────────────────────────────────────────────────

    #[test]
    fn bytes_per_vector_4bit_dim256() {
        let tq = TurboQuant::new(4);
        // norm (4 bytes) + 256 × 4 bits / 8 = 4 + 128 = 132
        assert_eq!(tq.bytes_per_vector(256), 132);
    }

    #[test]
    fn bytes_per_vector_3bit_dim256() {
        let tq = TurboQuant::new(3);
        // norm (4 bytes) + ceil(256 × 3 / 8) = 4 + 96 = 100
        assert_eq!(tq.bytes_per_vector(256), 100);
    }

    #[test]
    fn bytes_per_vector_4bit_dim128() {
        let tq = TurboQuant::new(4);
        // 4 + 128 × 4 / 8 = 4 + 64 = 68
        assert_eq!(tq.bytes_per_vector(128), 68);
    }

    #[test]
    fn compression_ratio_vs_fp16() {
        let tq = TurboQuant::new(4);
        // FP16 per dim=256 vector: 256 × 2 = 512 bytes
        // TurboQuant 4-bit: 132 bytes
        // Ratio: 512 / 132 ≈ 3.9×
        let fp16_bytes = 256 * 2;
        let tq_bytes = tq.bytes_per_vector(256);
        let ratio = fp16_bytes as f64 / tq_bytes as f64;
        assert!(ratio > 3.5, "compression ratio {ratio:.2} < 3.5");
    }

    // ── Engine construction and config ────────────────────────────────────────

    #[test]
    fn engine_name_and_config_4bit() {
        let eng = TurboQuantEngine::new(4);
        assert_eq!(eng.name(), "turbo-quant");
        let info = eng.info();
        assert_eq!(info.config, "bits=4");
        assert!(info.backend.starts_with("cpu"));
        assert!(info.description.contains("4-bit"));
    }

    #[test]
    fn engine_name_and_config_3bit() {
        let eng = TurboQuantEngine::new(3);
        assert_eq!(eng.info().config, "bits=3");
        assert!(eng.info().description.contains("3-bit"));
    }

    #[test]
    fn engine_memory_zero_before_prefill() {
        let eng = TurboQuantEngine::new(4);
        assert_eq!(eng.memory_bytes(), 0);
    }

    #[test]
    fn engine_summary_shows_bits_in_config() {
        let eng = TurboQuantEngine::new(4);
        let s = eng.info().summary();
        assert!(s.contains("turbo-quant"), "summary missing name: {s}");
        assert!(s.contains("bits=4"), "summary missing config: {s}");
    }

    // ── CompressedLayer memory accounting ────────────────────────────────────

    #[test]
    fn compressed_layer_memory_is_smaller_than_fp32() {
        use ndarray::Array2;
        let tq = TurboQuant::new(4);
        // Single K/V pair: 10 positions, kv_dim=1024 (Gemma 3 4B-like)
        let k = Array2::<f32>::from_elem((10, 1024), 0.1);
        let v = Array2::<f32>::from_elem((10, 1024), 0.2);
        let cl = CompressedLayer::compress(&(k, v), &tq);
        let fp32_bytes = 10 * 1024 * 4 * 2; // K+V, f32
        let compressed = cl.memory_bytes();
        assert!(
            compressed < fp32_bytes,
            "compressed {compressed}B should be < fp32 {fp32_bytes}B"
        );
        // Compression ratio should be ~4×
        let ratio = fp32_bytes as f64 / compressed as f64;
        assert!(ratio > 3.0, "ratio {ratio:.2} < 3.0");
    }

    #[test]
    fn compressed_layer_roundtrip_cosine() {
        use ndarray::Array2;
        let tq = TurboQuant::new(4);
        // Use unit-norm rows matching TurboQuant's codebook distribution.
        let k_data: Vec<f32> = (0..10)
            .flat_map(|i| unit_norm_vec(256, i * 7 + 17))
            .collect();
        let v_data: Vec<f32> = (0..10)
            .flat_map(|i| unit_norm_vec(256, i * 7 + 31))
            .collect();
        let k = Array2::from_shape_vec((10, 256), k_data.clone()).unwrap();
        let v = Array2::from_shape_vec((10, 256), v_data.clone()).unwrap();
        let cl = CompressedLayer::compress(&(k, v), &tq);
        let (k_dec, v_dec) = cl.decompress(&tq);
        // Check last row cosine (most relevant for decode) on both K and V.
        let k_orig_last: Vec<f32> = k_data[9 * 256..10 * 256].to_vec();
        let k_dec_last: Vec<f32> = k_dec.row(9).to_vec();
        let k_cos = cosine_similarity(&k_orig_last, &k_dec_last);
        assert!(
            k_cos > MIN_COS_4BIT,
            "K roundtrip cosine {k_cos:.4} < {MIN_COS_4BIT}"
        );
        let v_orig_last: Vec<f32> = v_data[9 * 256..10 * 256].to_vec();
        let v_dec_last: Vec<f32> = v_dec.row(9).to_vec();
        let v_cos = cosine_similarity(&v_orig_last, &v_dec_last);
        assert!(
            v_cos > MIN_COS_4BIT,
            "V roundtrip cosine {v_cos:.4} < {MIN_COS_4BIT}"
        );
    }

    // ── Block-dim validation ─────────────────────────────────────────────────

    #[test]
    fn resolve_block_dim_accepts_power_of_two_splits() {
        assert_eq!(resolve_block_dim(1024).unwrap(), 256);
        assert_eq!(resolve_block_dim(128).unwrap(), 128);
        assert_eq!(resolve_block_dim(96).unwrap(), 32);
        assert_eq!(resolve_block_dim(64).unwrap(), 64);
    }

    /// kv_dim = 80 has no power-of-two head split in the supported set;
    /// it used to fall through to the whole-row fallback and panic on
    /// the WHT's power-of-two assert mid-prefill.
    #[test]
    fn resolve_block_dim_rejects_non_power_of_two() {
        let err = resolve_block_dim(80).unwrap_err();
        assert!(
            matches!(err, EngineError::InvariantViolation { ref what } if what.contains("power-of-two")),
            "expected InvariantViolation naming the power-of-two constraint, got {err:?}"
        );
    }
}

// ─── Integration tests with synthetic weights ─────────────────────────────────

#[cfg(test)]
mod integration_tests {
    use super::*;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn prefill_compresses_kv_for_all_layers() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = TurboQuantEngine::new(4);
        assert_eq!(engine.memory_bytes(), 0);
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill failed");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(
            engine.layers.len(),
            weights.num_layers,
            "one CompressedLayer per model layer"
        );
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_grows_compressed_cache() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = TurboQuantEngine::new(4);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let mem_before = engine.memory_bytes();

        engine.decode_step(&weights, &ffn, 1).expect("decode_step");
        // After decode: K/V cache has one more entry per layer → more compressed bytes
        assert!(
            engine.memory_bytes() > mem_before,
            "compressed cache should grow after each decode step"
        );
    }

    #[test]
    fn logits_finite_after_prefill_and_decode() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = TurboQuantEngine::new(4);
        let h_pre = engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        assert!(hidden_to_raw_logits(&weights, &h_pre)
            .iter()
            .all(|v| v.is_finite()));
        let h_dec = engine.decode_step(&weights, &ffn, 2).expect("decode");
        assert!(hidden_to_raw_logits(&weights, &h_dec)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn three_bit_engine_also_works() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = TurboQuantEngine::new(3);
        let h = engine
            .prefill(&weights, &ffn, &[0u32])
            .expect("3-bit prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        // 3-bit uses fewer bytes per compressed vector
        let mem3 = engine.memory_bytes();
        let mut engine4 = TurboQuantEngine::new(4);
        engine4
            .prefill(&weights, &ffn, &[0u32])
            .expect("4-bit prefill");
        assert!(
            mem3 < engine4.memory_bytes(),
            "3-bit should use less memory than 4-bit"
        );
    }

    // ── Q4K paths via CPU fallback ────────────────────────────────────────
    //
    // `fused_prefill` / `fused_decode_step` return `None` on a CPU
    // backend, so the engine falls through to `prefill_quant_cpu` /
    // `decode_step_quant_cpu` against the synthetic VectorIndex. Exercises
    // the Q4K branches without needing a real Metal-quantised model.

    #[test]
    fn prefill_q4k_cpu_fallback_compresses_kv() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = TurboQuantEngine::new(4);
        let h = engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(
            engine.layers.len(),
            weights.num_layers,
            "one CompressedLayer per model layer after prefill_quant"
        );
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_cpu_fallback_grows_compressed_cache() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = TurboQuantEngine::new(4);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > mem_before,
            "compressed cache should grow after decode_step_quant"
        );
    }

    // ── Phase 2: executor-driven path ─────────────────────────────────────

    #[test]
    fn prefill_quant_via_executor_compresses_kv() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = TurboQuantEngine::new(4);
        let h = engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("executor prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.layers.len(), weights.num_layers);
        assert!(engine.memory_bytes() > 0);
    }

    /// Steps for the append-only regressions. The pre-fix executor path
    /// decompressed and re-encoded the whole cache every step, so each
    /// stored norm shrank by the codec's reconstruction ratio per step
    /// (compounding to ~0.90 of true norm by 20 steps with today's
    /// codebooks, ~0.44 with the mis-scaled ones).
    const APPEND_ONLY_DECODE_STEPS: usize = 20;
    /// Append-only decode never rewrites old rows' bytes, so the first
    /// row's decoded norm is bit-stable across steps; tolerance covers
    /// f64-accumulation slack only.
    const FIRST_ROW_NORM_DRIFT_TOL: f64 = 1e-6;

    fn layer0_row0_k_norm(engine: &TurboQuantEngine) -> f64 {
        let (k, _v) = engine.layers[0].decompress(&engine.tq);
        k.row(0)
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Packed bytes of layer 0's first K row (all heads).
    fn layer0_row0_k_bytes(engine: &TurboQuantEngine) -> Vec<u8> {
        let l = &engine.layers[0];
        let bytes_per_row = (l.kv_dim / l.head_dim) * engine.tq.bytes_per_vector(l.head_dim);
        l.compressed_k[..bytes_per_row].to_vec()
    }

    /// Regression for the executor decode path re-encoding the entire
    /// cache each step (norm decay ≈ ratio^N on the first cached row).
    #[test]
    fn executor_decode_is_append_only_first_row_stable() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = TurboQuantEngine::new(4);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill");
        let norm_after_prefill = layer0_row0_k_norm(&engine);
        let bytes_after_prefill = layer0_row0_k_bytes(&engine);
        assert!(norm_after_prefill > 0.0, "fixture row must be non-zero");

        for step in 0..APPEND_ONLY_DECODE_STEPS {
            engine
                .decode_step_quant_via_executor(
                    &weights,
                    &executor,
                    &ffn,
                    &index,
                    (step % 3) as u32,
                )
                .expect("decode step");
        }

        let norm_after = layer0_row0_k_norm(&engine);
        let drift = (norm_after / norm_after_prefill - 1.0).abs();
        assert!(
            drift < FIRST_ROW_NORM_DRIFT_TOL,
            "first cached row's decoded norm drifted by {drift:.6} over \
             {APPEND_ONLY_DECODE_STEPS} executor decode steps \
             ({norm_after_prefill:.6} -> {norm_after:.6}); the append-only \
             invariant is broken"
        );
        assert_eq!(
            layer0_row0_k_bytes(&engine),
            bytes_after_prefill,
            "row-0 packed bytes must never be rewritten by decode"
        );
    }

    /// Parity of the append-only invariant across the executor and CPU
    /// quant decode paths: identical prefill, then N steps down each
    /// path — both must leave the first row's packed bytes at the
    /// post-prefill snapshot (and therefore equal to each other).
    #[test]
    fn executor_and_cpu_decode_paths_append_only_parity() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;

        let mut via_executor = TurboQuantEngine::new(4);
        via_executor
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill (executor engine)");
        let mut via_cpu = TurboQuantEngine::new(4);
        via_cpu
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill (cpu engine)");

        let snapshot = layer0_row0_k_bytes(&via_executor);
        assert_eq!(
            snapshot,
            layer0_row0_k_bytes(&via_cpu),
            "identical prefill must produce identical row-0 bytes"
        );

        for step in 0..APPEND_ONLY_DECODE_STEPS {
            let token = (step % 3) as u32;
            via_executor
                .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, token)
                .expect("executor decode step");
            via_cpu
                .decode_step_quant(&weights, &ffn, &index, token, &*backend)
                .expect("cpu decode step");
        }

        assert_eq!(
            layer0_row0_k_bytes(&via_executor),
            snapshot,
            "executor path rewrote row-0 bytes"
        );
        assert_eq!(
            layer0_row0_k_bytes(&via_cpu),
            snapshot,
            "cpu path rewrote row-0 bytes"
        );
    }

    #[test]
    fn decode_step_quant_via_executor_grows_cache() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = TurboQuantEngine::new(4);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > mem_before);
    }

    /// Drive the profiling-on branch of `decode_step_quant_cpu` —
    /// covers the `if timing { ... }` arms and the profiler accumulate.
    #[test]
    fn decode_step_quant_cpu_with_profiling_populates_summary() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = TurboQuantEngine::new(4).with_profiling(true);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill");
        engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode");
        let summary = engine
            .stage_summary()
            .expect("turbo-quant profiler should populate summary");
        assert_eq!(summary.engine, "turbo-quant");
        assert!(summary.steps >= 1);
        // recompute_hot (codec decode) and recompute_cold (codec encode)
        // both fire per layer per step.
        assert!(summary.avg_recompute_hot_us > 0.0);
        assert!(summary.avg_recompute_cold_us > 0.0);
        assert!(summary.avg_attention_us > 0.0);
        assert!(summary.avg_ffn_us > 0.0);
    }

    /// Counting FFN — proves the executor path dispatches through the
    /// caller-supplied backend instead of constructing a local `WalkFfn`.
    struct CountingFfn {
        calls: std::sync::atomic::AtomicUsize,
        hidden: usize,
    }
    impl larql_inference::ffn::FfnBackend for CountingFfn {
        fn forward(&self, _layer: usize, x: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ndarray::Array2::zeros((x.shape()[0], self.hidden))
        }
        fn name(&self) -> &str {
            "counting"
        }
    }

    #[test]
    fn executor_path_honors_ffn_parameter() {
        use larql_inference::layer_executor::LocalWalkExecutor;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: weights.hidden_size,
        };
        let mut engine = TurboQuantEngine::new(4);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("prefill via executor");
        // Prefill runs FFN once per layer (single chunked sequence).
        let call_count = ffn.calls.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            call_count, weights.num_layers,
            "executor path should dispatch FFN through the supplied backend \
             once per layer; got {call_count} for {} layers",
            weights.num_layers
        );
    }

    /// Minimal `Fused`-kind executor — the engine's executor-routed
    /// entry points should detect `dispatch_kind == Fused` and short-
    /// circuit to the legacy `prefill_quant` / `decode_step_quant`
    /// paths, ignoring the supplied executor's per-layer methods.
    struct FusedStubExecutor {
        backend: larql_compute::CpuBackend,
    }
    impl larql_inference::layer_executor::LayerExecutor for FusedStubExecutor {
        fn backend(&self) -> &dyn larql_compute::ComputeBackend {
            &self.backend
        }
        fn dispatch_kind(&self) -> larql_inference::layer_executor::ExecutorDispatchKind {
            larql_inference::layer_executor::ExecutorDispatchKind::Fused
        }
        fn name(&self) -> &str {
            "fused-stub"
        }
    }

    #[test]
    fn fused_executor_short_circuits_prefill_to_legacy_path() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let executor = FusedStubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let ffn = NullFfn;
        let mut engine = TurboQuantEngine::new(4);
        let h = engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("fused-stub prefill should route through prefill_quant");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert_eq!(engine.layers.len(), weights.num_layers);
    }

    #[test]
    fn fused_executor_short_circuits_decode_to_legacy_path() {
        use larql_inference::ffn::NullFfn;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let executor = FusedStubExecutor {
            backend: larql_compute::CpuBackend,
        };
        let ffn = NullFfn;
        let mut engine = TurboQuantEngine::new(4);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("prefill");
        let h = engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 3)
            .expect("fused-stub decode should route through decode_step_quant");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn counting_ffn_forward_observed_runs_forward_and_reports_absent() {
        use larql_inference::ffn::FfnBackend;
        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: 8,
        };
        let x = ndarray::Array2::<f32>::zeros((3, 8));
        let (h, obs) = ffn.forward_observed(0, &x);
        assert_eq!(h.shape(), &[3, 8]);
        assert!(
            obs.is_absent(),
            "counting stub must not fabricate activations"
        );
        assert_eq!(ffn.name(), "counting");
        // The default `forward_observed` delegates to `forward`, so
        // exactly one call is recorded.
        assert_eq!(ffn.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
