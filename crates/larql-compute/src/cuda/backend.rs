//! `CudaBackend` — owns the [`Driver`] and dispatches to the per-kernel
//! wrappers in `cuda::matmul`. The kernel surface is filled in across
//! the [`cuda-and-rotorquant-kv`][parent] sub-changes; this module's
//! current state is `cuda-f32-baseline`.
//!
//! [parent]: ../../../../openspec/changes/cuda-and-rotorquant-kv/

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use cudarc::driver::CudaSlice;
use ndarray::{Array2, ArrayView2};

use crate::backend::{Capability, ComputeBackend, MatMul};

use super::decode::CudaKvCache;
use super::dequant;
use super::driver::Driver;
use super::error::CudaInitError;
use super::matmul as kernels;
use super::scratch::{DecodeScratch, SpecDecodeScratch, SpecScratchKey};
use cudarc::driver::CudaGraph;

/// `cuda-decode-cuda-graph`: thin Send-marker wrapper around cudarc's
/// `CudaGraph`. The graph's internal `CUgraph` / `CUgraphExec` are
/// raw pointers which aren't `Send`/`Sync` by default; CUDA's driver
/// API is thread-safe for graph launches as long as the owning
/// context is bound to the calling thread, and `CudaBackend` already
/// follows that contract via `bind_to_thread`. We therefore mark the
/// wrapper `Send + Sync` so it can live in `Mutex<Option<…>>`.
pub(crate) struct DecodeGraph(pub(crate) CudaGraph);
unsafe impl Send for DecodeGraph {}
unsafe impl Sync for DecodeGraph {}

pub struct CudaBackend {
    drv: Arc<Driver>,
    pub(crate) kv_cache: Mutex<Option<CudaKvCache>>,
    // `cuda-decode-cuda-graph`: caches store `Arc<CudaSlice>` so the
    // captured-graph decode path can hold cheap clones of the
    // cached device pointers across the capture region without
    // serialising on the cache locks. Cache values are immutable
    // for the lifetime of the model — Arc::clone is the right
    // primitive.
    q4k_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<u8>>>>,
    q5k_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<u8>>>>,
    /// F.6: byte-cache for Q8_0 weights — uploads the packed bytes
    /// once per tensor; the direct kernel reads scale + int8 in place
    /// (no f32 expansion). Replaces the F.4 f32 cache for the direct
    /// path; the f32 cache below stays for the cuBLAS fallback.
    q8_0_byte_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<u8>>>>,
    /// F.4: dequantised f32 cache for Q8_0 weights. Qwen3.6-35B-A3B's
    /// attention/shared-expert projections are Q8_0; without this
    /// cache each per-token matvec re-dequants 50+ MiB on the host
    /// and re-uploads to the device — the F.4 first-light measurement
    /// showed this drops decode from 1.20 → 0.24 t/s. Caching the
    /// dequantised weight once per tensor recovers the cost. F.6
    /// replaces the dispatch with the direct kernel (q8_0_byte_device_cache);
    /// this stays as the fallback when the direct kernel rejects the shape.
    q8_0_f32_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<f32>>>>,
    q6k_f32_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<f32>>>>,
    q6k_packed_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<u8>>>>,
    q4k_f32_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<f32>>>>,
    /// `cuda-prefill-tensor-cores`: dequantised + downcast f16
    /// weights for the f16 cuBLAS GEMM prefill path. Halves the
    /// device memory vs `q4k_f32_device_cache` and unlocks Tensor
    /// Cores in cuBLAS hgemm.
    q4k_f16_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<half::f16>>>>,
    q6k_f16_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<half::f16>>>>,
    /// Per-pointer cache of small f32 weights (norms etc.) so the
    /// per-layer norm-weight htod's collapse to a one-time upload per
    /// host buffer. Keyed by host pointer + length + content hash so
    /// distinct host buffers with identical bytes still collide
    /// safely.
    f32_norm_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<f32>>>>,
    /// Per-content-hash cache of f32 weight matrices used as the `B`
    /// operand in `matmul` / `matmul_transb`. The DSv4 GPU push
    /// (PRs #339-#367) threads every matmul through this backend; the
    /// 2026-05-25 bench showed `dot_proj_gpu` ran at 0.98× CPU speed
    /// because each call re-uploaded its weight matrix (KB to MB).
    /// This cache eliminates that re-upload: first call htods, later
    /// calls borrow the device buffer via `Arc::clone`. Same content
    /// hash semantics as [`f32_norm_device_cache`] — ptr/len/head/tail
    /// 64-bit summary; safe for read-only inference weights.
    f32_weight_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<f32>>>>,
    /// Per-pointer cache of f16 weight bytes (e.g. tied-embedding
    /// `embeddings.bin` reused as lm_head) converted to device-resident
    /// f32 for cuBLAS GEMV. Avoids re-uploading the 168 MB lm_head
    /// matrix on every drafted token. Used by `f16_gemv` on
    /// architectures where the Q4_K matvec kernel can't run (e.g.
    /// Gemma 3 270M's lm_head with hidden=640 — not a multiple of
    /// the 256-element Q4_K super-block).
    f16_f32_device_cache: Mutex<HashMap<DeviceBytesKey, Arc<CudaSlice<f32>>>>,
    /// Qwen3.6 DeltaNet recurrent and Conv1D state buffers keyed by
    /// host ndarray storage pointer. The host state is uploaded on
    /// first use and whenever sequence position resets to 0; after
    /// that the device buffer is authoritative for the active sequence.
    pub(crate) qwen35_f32_state_cache: Mutex<HashMap<DeviceStateKey, CudaSlice<f32>>>,
    /// `cuda-decode-cuda-graph`: pre-allocated per-decode scratch
    /// buffers. `None` until the first decode_token_device call sees
    /// a shape; reused on every subsequent same-shape call.
    pub(crate) decode_scratch: Mutex<Option<DecodeScratch>>,
    /// Captured decode graph + the warmup-call counter. The graph is
    /// captured on the call AFTER the scratch is first populated (so
    /// every weight/norm cache is already warm and the captured graph
    /// holds no htod nodes). Subsequent calls replay the graph.
    pub(crate) decode_graph: Mutex<Option<DecodeGraph>>,
    /// Number of decode_token_device calls since scratch was last
    /// (re)allocated. The capture happens at call #2.
    pub(crate) decode_warmup_count: Mutex<u32>,
    /// `cuda-spec-cuda-graph` Phase A: pre-allocated per-(seq_len,
    /// shape) scratch buffers for the spec batched seq forward path.
    /// Eliminates ~714 device_alloc calls per spec iter on Gemma 3 4B.
    pub(crate) spec_decode_scratch: Mutex<HashMap<SpecScratchKey, SpecDecodeScratch>>,
    /// `cuda-spec-cuda-graph` Phase C: captured CUDA Graph for the
    /// spec batched seq forward, one per (seq_len, shape). Captured
    /// on the call AFTER the scratch is first warmed; subsequent
    /// calls replay the graph after writing `base_pos` + input to
    /// the scratch slots.
    pub(crate) spec_decode_graph: Mutex<HashMap<SpecScratchKey, DecodeGraph>>,
    /// Warmup counter per (seq_len, shape) for the spec graph capture.
    /// Capture happens at count == 1 (the second call); call 0 warms
    /// every cache / scratch buffer so the captured graph has no
    /// allocations in it.
    pub(crate) spec_decode_warmup: Mutex<HashMap<SpecScratchKey, u32>>,
    /// Per-layer device-resident K/V slabs for the Qwen 3.6 hybrid
    /// full-attention layers. Keyed by `cache_id` (layer index passed
    /// by `qwen35_attention_block_step`). Each slab stores K and V
    /// in `half::f16` to match `fused_decode_attention_device_kv`'s
    /// kernel signature, plus the host-side seq_len it's currently
    /// caught up to so the trait method can detect a stale cache
    /// (e.g. when the host slab was reset without `qwen35_gqa_decode_reset`
    /// being called).
    pub(crate) qwen35_kv_slab_cache: Mutex<HashMap<u64, Qwen35KvSlabDev>>,
    /// `Step 4 Phase 2`: RotorQuant-compressed K/V slabs. Each
    /// `Qwen35KvSlabDevQuant` holds packed Iso3 codes
    /// (≈ 3.13 bits/element) instead of f16 — at max_seq=4096,
    /// kv_dim=512: 16 KiB × 2 (K+V) per layer × 16 layers ≈ 512 KiB
    /// vs the f16 path's 64 MiB. Selected by env var
    /// `LARQL_QWEN35_KV_FORMAT=iso3`; default unset uses the f16
    /// slab cache above.
    pub(crate) qwen35_kv_slab_cache_quant: Mutex<HashMap<u64, Qwen35KvSlabDevQuant>>,
    /// `Step 4 Phase 2`: shared dequant scratch buffers. The
    /// compressed-cache attention path dequants the full
    /// `seq_len * kv_dim` slab into f32 scratch, converts to f16,
    /// then feeds the existing f16 attention kernel. Allocated
    /// lazily and reused across all attention calls in a step
    /// (16 layers × 1 call each share these buffers). Sized to the
    /// first-seen `max_seq * max_kv_dim`; grown on demand if a
    /// larger shape comes through.
    pub(crate) qwen35_kv_dequant_f32_scratch: Mutex<Option<CudaSlice<f32>>>,
    pub(crate) qwen35_kv_dequant_f16_k_scratch: Mutex<Option<CudaSlice<half::f16>>>,
    pub(crate) qwen35_kv_dequant_f16_v_scratch: Mutex<Option<CudaSlice<half::f16>>>,
}

/// Device-resident K/V cache slab for one Qwen 3.6 full-attention layer.
///
/// Size: `max_seq * num_kv_heads * head_dim * 2 (f16) * 2 (K and V)`.
/// On Qwen3.6 35B-A3B (16 full-attn layers, 4 KV heads × 128 head_dim,
/// max_seq=4096): 16 × 4 × 128 × 4096 × 2 × 2 ≈ 64 MiB total across
/// all 16 layers.
pub(crate) struct Qwen35KvSlabDev {
    pub k: CudaSlice<half::f16>,
    pub v: CudaSlice<half::f16>,
    pub max_seq: usize,
    pub kv_dim: usize,
    /// seq_len the device cache is caught up to. If the next call
    /// arrives with `host_seq_len != cached_seq_len + 1` the slab is
    /// stale and gets re-populated from the host slab.
    pub cached_seq_len: usize,
}

/// `Step 4 Phase 2` — RotorQuant-compressed K/V slab.
///
/// Codes are stored row-major: packed Iso3 bytes for `max_seq` rows
/// of `kv_dim` elements each. The per-row packed byte length comes
/// from `larql_rotorquant::quantized_device_len_bytes(Iso3, kv_dim)`,
/// not the per-element 3.13 bits divided by 8 — packing is per-block.
///
/// Lossy: cosine ≥ ~0.98 vs the original f16. Production validation
/// is via end-to-end logit-difference bench (greedy decode coherence
/// match) not exact roundtrip.
pub(crate) struct Qwen35KvSlabDevQuant {
    pub codes_k: CudaSlice<u8>,
    pub codes_v: CudaSlice<u8>,
    pub max_seq: usize,
    pub kv_dim: usize,
    /// Bytes per row in `codes_k` / `codes_v`. Cached so we don't
    /// re-call `quantized_device_len_bytes` on each step.
    pub packed_row_bytes: usize,
    pub cached_seq_len: usize,
}

impl CudaBackend {
    pub fn new() -> Result<Self, CudaInitError> {
        Self::new_with_index(0)
    }

    pub fn new_with_index(ordinal: usize) -> Result<Self, CudaInitError> {
        let drv = Driver::new_with_index(ordinal)?;
        Ok(CudaBackend {
            drv,
            kv_cache: Mutex::new(None),
            q4k_device_cache: Mutex::new(HashMap::new()),
            q5k_device_cache: Mutex::new(HashMap::new()),
            q6k_f32_device_cache: Mutex::new(HashMap::new()),
            q8_0_byte_device_cache: Mutex::new(HashMap::new()),
            q8_0_f32_device_cache: Mutex::new(HashMap::new()),
            q6k_packed_device_cache: Mutex::new(HashMap::new()),
            q4k_f32_device_cache: Mutex::new(HashMap::new()),
            q4k_f16_device_cache: Mutex::new(HashMap::new()),
            q6k_f16_device_cache: Mutex::new(HashMap::new()),
            f32_norm_device_cache: Mutex::new(HashMap::new()),
            f32_weight_device_cache: Mutex::new(HashMap::new()),
            f16_f32_device_cache: Mutex::new(HashMap::new()),
            qwen35_f32_state_cache: Mutex::new(HashMap::new()),
            decode_scratch: Mutex::new(None),
            decode_graph: Mutex::new(None),
            decode_warmup_count: Mutex::new(0),
            spec_decode_scratch: Mutex::new(HashMap::new()),
            spec_decode_graph: Mutex::new(HashMap::new()),
            spec_decode_warmup: Mutex::new(HashMap::new()),
            qwen35_kv_slab_cache: Mutex::new(HashMap::new()),
            qwen35_kv_slab_cache_quant: Mutex::new(HashMap::new()),
            qwen35_kv_dequant_f32_scratch: Mutex::new(None),
            qwen35_kv_dequant_f16_k_scratch: Mutex::new(None),
            qwen35_kv_dequant_f16_v_scratch: Mutex::new(None),
        })
    }

    /// Module-internal accessor used by `cuda::attn` so the helper
    /// can borrow the driver without exposing it crate-wide.
    pub(crate) fn driver(&self) -> &Driver {
        &self.drv
    }

    pub(crate) fn with_q4k_device_buf<R>(
        &self,
        host: &[u8],
        f: impl FnOnce(&CudaSlice<u8>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let arc = self.arc_q4k_device_buf(host)?;
        f(&arc)
    }

    /// Same content-keyed cache as [`with_q4k_device_buf`] but for
    /// Q5_K weight bytes. The two formats have different bytes-per-
    /// block (144 vs 176) so they live in separate caches; the
    /// content hash overlap would still be safe (the lookup would
    /// only ever return a buffer derived from the same host slice)
    /// but a dedicated cache keeps the semantics obvious.
    pub(crate) fn with_q5k_device_buf<R>(
        &self,
        host: &[u8],
        f: impl FnOnce(&CudaSlice<u8>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q5k_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q5k device cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return f(arc);
            }
        }
        let arc = Arc::new(self.drv.device_u8_buf_from(host)?);
        let mut cache = self
            .q5k_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q5k device cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        let arc_clone = Arc::clone(entry);
        drop(cache);
        f(&arc_clone)
    }

    /// Same content-keyed byte cache as `with_q5k_device_buf` but for
    /// Q8_0 weight bytes (34-byte blocks). Used by the F.6 direct
    /// matvec kernel so each Q8_0 tensor uploads once instead of
    /// re-dequanting to f32 per call.
    pub(crate) fn with_q8_0_device_buf<R>(
        &self,
        host: &[u8],
        f: impl FnOnce(&CudaSlice<u8>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self.q8_0_byte_device_cache.lock().map_err(|_| {
                CudaInitError::DriverMissing("q8_0 byte device cache poisoned".into())
            })?;
            if let Some(arc) = cache.get(&key) {
                return f(arc);
            }
        }
        let arc = Arc::new(self.drv.device_u8_buf_from(host)?);
        let mut cache = self
            .q8_0_byte_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q8_0 byte device cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        let arc_clone = Arc::clone(entry);
        drop(cache);
        f(&arc_clone)
    }

    /// `cuda-decode-cuda-graph`: Arc-cloned device buffer for the
    /// graph-capture path. Holds no lock once it returns — the Arc
    /// keeps the cached buffer alive even if the cache is later
    /// re-locked.
    pub(crate) fn arc_q4k_device_buf(
        &self,
        host: &[u8],
    ) -> Result<Arc<CudaSlice<u8>>, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q4k_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q4k device cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return Ok(Arc::clone(arc));
            }
        }
        let arc = Arc::new(self.drv.device_u8_buf_from(host)?);
        let mut cache = self
            .q4k_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q4k device cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry))
    }

    /// Per-pointer cache of *packed* Q6_K weight bytes on the
    /// device. Parallel to `with_q4k_device_buf`. First call htod's
    /// the packed 210 B/super-block stream; later calls borrow.
    /// Used by the Q6_K mmvq path (`cuda-q6k-mmvq`).
    pub(crate) fn with_q6k_packed_device_buf<R>(
        &self,
        host: &[u8],
        f: impl FnOnce(&CudaSlice<u8>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let arc = self.arc_q6k_packed_device_buf(host)?;
        f(&arc)
    }

    pub(crate) fn arc_q6k_packed_device_buf(
        &self,
        host: &[u8],
    ) -> Result<Arc<CudaSlice<u8>>, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q6k_packed_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q6k packed cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return Ok(Arc::clone(arc));
            }
        }
        let arc = Arc::new(self.drv.device_u8_buf_from(host)?);
        let mut cache = self
            .q6k_packed_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q6k packed cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry))
    }

    /// Per-pointer cache of *dequantised* f32 Q4_K weights on the
    /// device. `cuda-prefill-batched-q4k` uses this for cuBLAS GEMM
    /// during prefill — re-reading 9.6 GB of f32 weights is faster
    /// than per-call dequant. Decode still uses the packed cache via
    /// `with_q4k_device_buf` + mmvq.
    /// `cuda-prefill-tensor-cores`: f16 cache for cuBLAS hgemm.
    /// Dequantises Q4_K bytes via `dequant_q4_k`, downcasts each
    /// element to f16 on the host (one-time per session), and uploads
    /// the f16 buffer. Halves the device memory of the equivalent
    /// f32 cache (`q4k_f32_device_cache`) and unlocks Tensor Cores
    /// in `matmul_transb_device_inout_f16`.
    pub(crate) fn with_q4k_f16_device_buf<R>(
        &self,
        host: &[u8],
        n_elements: usize,
        f: impl FnOnce(&CudaSlice<half::f16>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q4k_f16_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q4k f16 cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return f(arc);
            }
        }
        let w_f32 = dequant::dequant_q4_k(host, n_elements)
            .map_err(|e| CudaInitError::DriverMissing(format!("q4k dequant: {e:?}")))?;
        let w_f16: Vec<half::f16> = w_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let arc = Arc::new(
            self.drv
                .stream
                .clone_htod(&w_f16)
                .map_err(|e| CudaInitError::DriverMissing(format!("htod q4k f16: {e:?}")))?,
        );
        let mut cache = self
            .q4k_f16_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q4k f16 cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        let arc = Arc::clone(entry);
        drop(cache);
        f(&arc)
    }

    /// `cuda-prefill-tensor-cores`: f16 cache for Q6_K weights.
    pub(crate) fn with_q6k_f16_device_buf<R>(
        &self,
        host: &[u8],
        n_elements: usize,
        f: impl FnOnce(&CudaSlice<half::f16>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q6k_f16_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q6k f16 cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return f(arc);
            }
        }
        let w_f32 = dequant::dequant_q6_k(host, n_elements)
            .map_err(|e| CudaInitError::DriverMissing(format!("q6k dequant: {e:?}")))?;
        let w_f16: Vec<half::f16> = w_f32.iter().map(|&v| half::f16::from_f32(v)).collect();
        let arc = Arc::new(
            self.drv
                .stream
                .clone_htod(&w_f16)
                .map_err(|e| CudaInitError::DriverMissing(format!("htod q6k f16: {e:?}")))?,
        );
        let mut cache = self
            .q6k_f16_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q6k f16 cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        let arc = Arc::clone(entry);
        drop(cache);
        f(&arc)
    }

    pub(crate) fn with_q4k_f32_device_buf<R>(
        &self,
        host: &[u8],
        n_elements: usize,
        f: impl FnOnce(&CudaSlice<f32>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q4k_f32_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q4k f32 cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return f(arc);
            }
        }
        let w = dequant::dequant_q4_k(host, n_elements)
            .map_err(|e| CudaInitError::DriverMissing(format!("q4k dequant: {e:?}")))?;
        let arc = Arc::new(self.drv.device_buf_from(&w)?);
        let mut cache = self
            .q4k_f32_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q4k f32 cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        let arc = Arc::clone(entry);
        drop(cache);
        f(&arc)
    }

    /// Cache f16 weight bytes (host mmap or Vec) converted once to a
    /// device-resident f32 buffer. The cache is keyed by host pointer
    /// + length + content hash; subsequent calls with the SAME host
    /// buffer return the cached `CudaSlice<f32>` without re-uploading
    /// the 168 MB lm_head matrix. Used by `f16_gemv` for the lm_head
    /// path on architectures where the Q4_K direct kernel can't run.
    pub(crate) fn with_f16_f32_device_buf<R>(
        &self,
        host_f16_bytes: &[u8],
        n_elements: usize,
        f: impl FnOnce(&CudaSlice<f32>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        if host_f16_bytes.len() < n_elements * 2 {
            return Err(CudaInitError::DriverMissing(format!(
                "f16 source bytes too short: got {}, need {}",
                host_f16_bytes.len(),
                n_elements * 2
            )));
        }
        let key = DeviceBytesKey::from_slice(host_f16_bytes);
        {
            let cache = self
                .f16_f32_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("f16 f32 cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return f(arc);
            }
        }
        let f16_slice: &[half::f16] = unsafe {
            std::slice::from_raw_parts(host_f16_bytes.as_ptr() as *const half::f16, n_elements)
        };
        let mut w_f32 = Vec::with_capacity(n_elements);
        for &h in f16_slice {
            w_f32.push(h.to_f32());
        }
        let arc = Arc::new(self.drv.device_buf_from(&w_f32)?);
        let mut cache = self
            .f16_f32_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("f16 f32 cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        let arc = Arc::clone(entry);
        drop(cache);
        f(&arc)
    }

    pub(crate) fn with_q6k_f32_device_buf<R>(
        &self,
        host: &[u8],
        n_elements: usize,
        f: impl FnOnce(&CudaSlice<f32>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q6k_f32_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q6k device cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return f(arc);
            }
        }
        let w = dequant::dequant_q6_k(host, n_elements)
            .map_err(|e| CudaInitError::DriverMissing(format!("q6k dequant: {e:?}")))?;
        let arc = Arc::new(self.drv.device_buf_from(&w)?);
        let mut cache = self
            .q6k_f32_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q6k device cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        let arc = Arc::clone(entry);
        drop(cache);
        f(&arc)
    }

    /// Mirror of `with_q6k_f32_device_buf` for Q8_0 weights. See the
    /// note on `q8_0_f32_device_cache` for why the cache exists.
    pub(crate) fn with_q8_0_f32_device_buf<R>(
        &self,
        host: &[u8],
        n_elements: usize,
        f: impl FnOnce(&CudaSlice<f32>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceBytesKey::from_slice(host);
        {
            let cache = self
                .q8_0_f32_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("q8_0 device cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return f(arc);
            }
        }
        let w = dequant::dequant_q8_0(host, n_elements)
            .map_err(|e| CudaInitError::DriverMissing(format!("q8_0 dequant: {e:?}")))?;
        let arc = Arc::new(self.drv.device_buf_from(&w)?);
        let mut cache = self
            .q8_0_f32_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("q8_0 device cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        let arc = Arc::clone(entry);
        drop(cache);
        f(&arc)
    }

    #[doc(hidden)]
    pub fn q4k_device_cache_len(&self) -> usize {
        self.q4k_device_cache
            .lock()
            .map(|cache| cache.len())
            .unwrap_or(0)
    }

    #[doc(hidden)]
    pub fn q6k_f32_device_cache_len(&self) -> usize {
        self.q6k_f32_device_cache
            .lock()
            .map(|cache| cache.len())
            .unwrap_or(0)
    }

    // ── Device-resident projection helpers ────────────────────────────
    //
    // `cuda-decode-device-resident` Phase 1 — keep per-layer state on
    // the GPU through the projection chain. Each helper takes a
    // `CudaSlice<f32>` input and returns a `CudaSlice<f32>` output;
    // there is no implicit `htod` / `dtoh` and no `sync` between
    // launches.

    /// H2D copy of an f32 host slice. Thin wrapper around the
    /// crate-private `Driver::device_buf_from`.
    pub(crate) fn htod_f32(&self, host: &[f32]) -> Result<CudaSlice<f32>, CudaInitError> {
        self.drv.device_buf_from(host)
    }

    /// Cached H2D for small f32 weights (norm vectors etc.).
    /// First call htod's and stashes the device buffer; subsequent
    /// calls with the same host pointer/content return a borrow of
    /// the cached buffer via the closure. Used by the device-resident
    /// decode path to avoid re-uploading the same per-layer norm
    /// weights on every token.
    pub(crate) fn with_norm_device_buf<R>(
        &self,
        host: &[f32],
        f: impl FnOnce(&CudaSlice<f32>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let arc = self.arc_norm_device_buf(host)?;
        f(&arc)
    }

    pub(crate) fn arc_norm_device_buf(
        &self,
        host: &[f32],
    ) -> Result<Arc<CudaSlice<f32>>, CudaInitError> {
        // SAFETY: f32 has no padding; reinterpreting as bytes for a
        // hash key is well-defined.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, std::mem::size_of_val(host))
        };
        let key = DeviceBytesKey::from_slice(bytes);
        {
            let cache = self
                .f32_norm_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("f32 norm cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return Ok(Arc::clone(arc));
            }
        }
        let arc = Arc::new(self.drv.device_buf_from(host)?);
        let mut cache = self
            .f32_norm_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("f32 norm cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry))
    }

    /// Content-keyed device cache for f32 matmul weight matrices.
    ///
    /// First call uploads `host` to the device and inserts; later
    /// calls with the same `(ptr, len, head, tail)` key return the
    /// same `Arc<CudaSlice<f32>>` without re-uploading.
    ///
    /// Used by [`MatMul::matmul`] / [`MatMul::matmul_transb`] for the
    /// `B` (weight) operand. The 2026-05-25 RTX 4090 bench on
    /// DSv4-Flash showed that the GPU push was bottlenecked by
    /// re-uploading B on every call — eliminating that re-upload is
    /// the change that should move the speedup from 0.98× to >1×.
    ///
    /// The cache lives on the backend and grows monotonically across
    /// the process lifetime. For workloads where weights move
    /// (training, partial dequant) callers must not use this — but
    /// for read-only inference (DSv4 / qwen35 / etc.) it's safe.
    pub(crate) fn arc_f32_weight_device_buf(
        &self,
        host: &[f32],
    ) -> Result<Arc<CudaSlice<f32>>, CudaInitError> {
        // SAFETY: f32 has no padding; reinterpreting as bytes for a
        // hash key is well-defined.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(host.as_ptr() as *const u8, std::mem::size_of_val(host))
        };
        let key = DeviceBytesKey::from_slice(bytes);
        {
            let cache = self
                .f32_weight_device_cache
                .lock()
                .map_err(|_| CudaInitError::DriverMissing("f32 weight cache poisoned".into()))?;
            if let Some(arc) = cache.get(&key) {
                return Ok(Arc::clone(arc));
            }
        }
        let arc = Arc::new(self.drv.device_buf_from(host)?);
        let mut cache = self
            .f32_weight_device_cache
            .lock()
            .map_err(|_| CudaInitError::DriverMissing("f32 weight cache poisoned".into()))?;
        let entry = cache.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(entry))
    }

    /// D2H copy. Synchronises the stream first.
    pub(crate) fn dtoh_f32(&self, dev: &CudaSlice<f32>) -> Result<Vec<f32>, CudaInitError> {
        self.drv.sync()?;
        self.drv.to_host(dev)
    }

    /// Allocate an uninitialised device buffer.
    pub(crate) fn alloc_f32(&self, len: usize) -> Result<CudaSlice<f32>, CudaInitError> {
        self.drv.device_alloc(len)
    }

    /// H2D copy a host slice into an existing device buffer at the
    /// given element offset. `cuda-decode-device-resident` Phase 3
    /// uses this to write a single K/V row into the persistent
    /// device-resident KV cache slab without reallocating.
    pub(crate) fn htod_into_slice(
        &self,
        src: &[f32],
        dst: &mut CudaSlice<f32>,
        offset: usize,
    ) -> Result<(), CudaInitError> {
        if offset + src.len() > dst.len() {
            return Err(CudaInitError::DriverMissing(format!(
                "htod_into_slice OOB: offset {offset} + len {} > dst {}",
                src.len(),
                dst.len(),
            )));
        }
        let mut sub = dst.slice_mut(offset..offset + src.len());
        self.drv
            .stream
            .memcpy_htod(src, &mut sub)
            .map_err(|e| CudaInitError::DriverMissing(format!("memcpy_htod into slice: {e:?}")))
    }

    /// `cuda-attn-wmma-f16kv` Phase 1: host f32 → device f16, with
    /// element-wise round-to-nearest conversion on the host. Used by
    /// `populate_kv_layer` and the legacy host-fallback decode path
    /// to write into the new f16 K/V cache.
    pub(crate) fn htod_f32_as_f16_into_slice(
        &self,
        src: &[f32],
        dst: &mut CudaSlice<half::f16>,
        offset: usize,
    ) -> Result<(), CudaInitError> {
        if offset + src.len() > dst.len() {
            return Err(CudaInitError::DriverMissing(format!(
                "htod_f32_as_f16 OOB: offset {offset} + len {} > dst {}",
                src.len(),
                dst.len(),
            )));
        }
        let h: Vec<half::f16> = src.iter().map(|&v| half::f16::from_f32(v)).collect();
        let mut sub = dst.slice_mut(offset..offset + h.len());
        self.drv
            .stream
            .memcpy_htod(&h, &mut sub)
            .map_err(|e| CudaInitError::DriverMissing(format!("memcpy_htod f16: {e:?}")))
    }

    /// `cuda-attn-wmma-f16kv` Phase 1: device f16 → host f32 vec.
    /// Round-trips through the host conversion so the legacy
    /// `decode_token` path's `fused_decode_attention(&[f32], &[f32])`
    /// host wrapper still gets the f32 it expects. Slow on purpose;
    /// this is back-out / parity only.
    pub(crate) fn dtoh_f16_as_f32(
        &self,
        dev: &CudaSlice<half::f16>,
    ) -> Result<Vec<f32>, CudaInitError> {
        self.drv.sync()?;
        let h: Vec<half::f16> = self
            .drv
            .stream
            .clone_dtoh(dev)
            .map_err(|e| CudaInitError::DriverMissing(format!("dtoh f16: {e:?}")))?;
        Ok(h.iter().map(|v| v.to_f32()).collect())
    }

    /// Q4_K matvec, device input + device output.
    pub(crate) fn q4k_matvec_device(
        &self,
        q4k_data: &[u8],
        x_dev: &CudaSlice<f32>,
        rows: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>, CudaInitError> {
        super::q4k_direct::matvec_device(self, q4k_data, x_dev, rows, hidden)
    }

    /// Q6_K matvec, device input + device output. Goes through the
    /// dequantised-f32 device cache and a cuBLAS gemv.
    pub(crate) fn q6k_matvec_device(
        &self,
        q6k_data: &[u8],
        x_dev: &CudaSlice<f32>,
        rows: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>, CudaInitError> {
        if x_dev.len() != hidden {
            return Err(CudaInitError::DriverMissing(format!(
                "q6k_matvec_device input mismatch: x_dev.len={} hidden={hidden}",
                x_dev.len(),
            )));
        }
        self.with_q6k_f32_device_buf(q6k_data, rows * hidden, |w_dev| {
            kernels::gemv_device_inout(self.driver(), w_dev, x_dev, rows, hidden)
        })
    }

    /// Q4_KF matvec, device input + device output. Q4_KF is rare on
    /// production vindexes so this path keeps the host-dequant + cuBLAS
    /// shape; the dequant trip happens once per call.
    pub(crate) fn q4kf_matvec_device(
        &self,
        q4kf_data: &[u8],
        x_dev: &CudaSlice<f32>,
        rows: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>, CudaInitError> {
        if x_dev.len() != hidden {
            return Err(CudaInitError::DriverMissing(format!(
                "q4kf_matvec_device input mismatch: x_dev.len={} hidden={hidden}",
                x_dev.len(),
            )));
        }
        let w = dequant::dequant_q4_kf(q4kf_data, rows * hidden)
            .map_err(|e| CudaInitError::DriverMissing(format!("q4kf dequant: {e:?}")))?;
        let w_dev = self.drv.device_buf_from(&w)?;
        kernels::gemv_device_inout(self.driver(), &w_dev, x_dev, rows, hidden)
    }

    /// f32 GEMV, device input + device output.
    pub(crate) fn f32_gemv_device(
        &self,
        w_dev: &CudaSlice<f32>,
        x_dev: &CudaSlice<f32>,
        rows: usize,
        hidden: usize,
    ) -> Result<CudaSlice<f32>, CudaInitError> {
        kernels::gemv_device_inout(self.driver(), w_dev, x_dev, rows, hidden)
    }

    /// Internal: contiguous row-major view of an `ArrayView2`. The
    /// fast-path is when the view is already standard layout; we only
    /// allocate on the slow-path (transposed / strided views).
    fn as_contiguous<'a>(&self, m: ArrayView2<'a, f32>) -> Vec<f32> {
        if let Some(slice) = m.as_slice() {
            slice.to_vec()
        } else {
            // Strided view — collect through ndarray's iterator into a
            // fresh Vec. Cheap on the dimensions we care about.
            m.iter().copied().collect()
        }
    }

    pub(crate) fn with_qwen35_state_device_buf<R>(
        &self,
        host: &mut [f32],
        sequence_pos: usize,
        f: impl FnOnce(&mut CudaSlice<f32>) -> Result<R, CudaInitError>,
    ) -> Result<R, CudaInitError> {
        let key = DeviceStateKey::from_slice(host);
        let mut cache = self.qwen35_f32_state_cache.lock().map_err(|_| {
            CudaInitError::DriverMissing("qwen35 state device cache poisoned".into())
        })?;
        if sequence_pos == 0 || !cache.contains_key(&key) {
            let state_dev = self.drv.device_buf_from(host)?;
            cache.insert(key, state_dev);
        }
        let state_dev = cache
            .get_mut(&key)
            .ok_or_else(|| CudaInitError::DriverMissing("qwen35 state cache miss".into()))?;
        f(state_dev)
    }

    /// `Step 4 Phase 2` — RotorQuant-Iso3-compressed KV attention step.
    ///
    /// Storage: per-layer packed Iso3 codes (≈3.13 bits/element).
    /// Per call:
    ///   1. f32 new row → host f16 conversion → htod
    ///   2. compress(f16) → packed bytes at slab.codes_*[new_row_idx]
    ///   3. dequant(packed[..seq_len]) → f32 scratch
    ///   4. f32 → f16 scratch (existing `f32_to_f16_device_into`)
    ///   5. existing `fused_decode_attention_device_kv` reads f16 scratch
    ///   6. dtoh attn output
    ///
    /// The f32 + f16 scratch buffers are shared across all 16
    /// full-attn layers in a step (lazily allocated, sized to
    /// `max_seq * kv_dim`). Net VRAM cost per layer:
    ///   compressed codes: 2 × packed_row_bytes × max_seq (the K + V slabs)
    /// vs the f16 path's:
    ///   2 × kv_dim × max_seq × 2 bytes (the K + V f16 slabs)
    /// — ≈ 3.5× saving on the per-layer storage at the cost of one
    /// shared scratch pair sized to the worst-case layer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen35_gqa_decode_step_iso3(
        &self,
        cache_id: u64,
        max_seq: usize,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        use larql_rotorquant::{
            copy_f16_to_quantized_device, dequantize_to_f32_device, quantized_device_len_bytes,
            CudaStream as RqStream, KvFormat, CUDA_BLOCK_ELEMENTS,
        };
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // Iso3 packs in blocks of `CUDA_BLOCK_ELEMENTS` (= 128) elements.
        // kv_dim must be a multiple. Qwen3.6-35B-A3B: kv_dim = 4 × 128 = 512. OK.
        if !kv_dim.is_multiple_of(CUDA_BLOCK_ELEMENTS) {
            return None;
        }
        let drv = self.driver();
        // `LARQL_QWEN35_KV_PRESEED=N` debug knob — see the f16 path
        // in `qwen35_gqa_decode_step` for the rationale. On Iso3 the
        // preseed rows are all-zero packed bytes (from `alloc_zeros`).
        let preseed_n = std::env::var("LARQL_QWEN35_KV_PRESEED")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
            .min(max_seq.saturating_sub(seq_len));
        let new_row_idx = preseed_n + (seq_len - 1);

        // Get/allocate the per-layer compressed slab.
        let packed_row_bytes = quantized_device_len_bytes(KvFormat::Iso3, kv_dim).ok()?;
        let total_packed_bytes = packed_row_bytes.checked_mul(max_seq)?;
        let mut cache = self.qwen35_kv_slab_cache_quant.lock().ok()?;
        let slab = cache.entry(cache_id).or_insert_with_key(|_| {
            let codes_k = drv
                .stream
                .alloc_zeros::<u8>(total_packed_bytes)
                .expect("alloc Iso3 codes_k");
            let codes_v = drv
                .stream
                .alloc_zeros::<u8>(total_packed_bytes)
                .expect("alloc Iso3 codes_v");
            Qwen35KvSlabDevQuant {
                codes_k,
                codes_v,
                max_seq,
                kv_dim,
                packed_row_bytes,
                cached_seq_len: 0,
            }
        });

        if slab.max_seq != max_seq
            || slab.kv_dim != kv_dim
            || slab.packed_row_bytes != packed_row_bytes
        {
            slab.codes_k = drv.stream.alloc_zeros::<u8>(total_packed_bytes).ok()?;
            slab.codes_v = drv.stream.alloc_zeros::<u8>(total_packed_bytes).ok()?;
            slab.max_seq = max_seq;
            slab.kv_dim = kv_dim;
            slab.packed_row_bytes = packed_row_bytes;
            slab.cached_seq_len = 0;
        }

        // Preseed bookkeeping: if `preseed_n` is set and this is the
        // first call, mark the leading rows as cached (their packed
        // bytes are zeros from `alloc_zeros` — fine for bench).
        if slab.cached_seq_len == 0 && preseed_n > 0 {
            slab.cached_seq_len = preseed_n;
        }

        // Stale-cache repopulation: compress all prior REAL rows
        // (after the preseed). With preseed=0 this matches the
        // original logic exactly.
        if slab.cached_seq_len != new_row_idx {
            let host_prev = seq_len - 1;
            if host_prev > 0 {
                let prev_elements = host_prev * kv_dim;
                let k_prev_f16: Vec<half::f16> = k_cache[..prev_elements]
                    .iter()
                    .map(|&v| half::f16::from_f32(v))
                    .collect();
                let v_prev_f16: Vec<half::f16> = v_cache[..prev_elements]
                    .iter()
                    .map(|&v| half::f16::from_f32(v))
                    .collect();
                let k_prev_dev = drv.stream.clone_htod(&k_prev_f16).ok()?;
                let v_prev_dev = drv.stream.clone_htod(&v_prev_f16).ok()?;
                use cudarc::driver::{DevicePtr, DevicePtrMut};
                let stream = drv.stream.clone();
                let rq_stream =
                    unsafe { RqStream::from_raw(stream.cu_stream() as *mut std::ffi::c_void) };
                let dst_byte_offset = preseed_n * packed_row_bytes;
                let dst_byte_len = host_prev * packed_row_bytes;
                {
                    let (src_ptr, _g1) = k_prev_dev.device_ptr(&stream);
                    let mut k_window = slab
                        .codes_k
                        .slice_mut(dst_byte_offset..dst_byte_offset + dst_byte_len);
                    let (dst_ptr, _g2) = k_window.device_ptr_mut(&stream);
                    unsafe {
                        copy_f16_to_quantized_device(
                            KvFormat::Iso3,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            prev_elements,
                            rq_stream,
                        )
                    }
                    .ok()?;
                }
                {
                    let (src_ptr, _g1) = v_prev_dev.device_ptr(&stream);
                    let mut v_window = slab
                        .codes_v
                        .slice_mut(dst_byte_offset..dst_byte_offset + dst_byte_len);
                    let (dst_ptr, _g2) = v_window.device_ptr_mut(&stream);
                    unsafe {
                        copy_f16_to_quantized_device(
                            KvFormat::Iso3,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            prev_elements,
                            rq_stream,
                        )
                    }
                    .ok()?;
                }
            }
            slab.cached_seq_len = new_row_idx;
        }

        // Compress the new (single) row into the packed slab at
        // offset `new_row_idx * packed_row_bytes`.
        {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let host_new = seq_len - 1;
            let new_k_f16: Vec<half::f16> = k_cache[host_new * kv_dim..(host_new + 1) * kv_dim]
                .iter()
                .map(|&v| half::f16::from_f32(v))
                .collect();
            let new_v_f16: Vec<half::f16> = v_cache[host_new * kv_dim..(host_new + 1) * kv_dim]
                .iter()
                .map(|&v| half::f16::from_f32(v))
                .collect();
            let new_k_dev = drv.stream.clone_htod(&new_k_f16).ok()?;
            let new_v_dev = drv.stream.clone_htod(&new_v_f16).ok()?;
            let stream = drv.stream.clone();
            let rq_stream =
                unsafe { RqStream::from_raw(stream.cu_stream() as *mut std::ffi::c_void) };
            let row_byte_offset = new_row_idx * packed_row_bytes;
            {
                let (src_ptr, _g1) = new_k_dev.device_ptr(&stream);
                let mut codes_k_row = slab
                    .codes_k
                    .slice_mut(row_byte_offset..row_byte_offset + packed_row_bytes);
                let (dst_ptr, _g2) = codes_k_row.device_ptr_mut(&stream);
                unsafe {
                    copy_f16_to_quantized_device(
                        KvFormat::Iso3,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        kv_dim,
                        rq_stream,
                    )
                }
                .ok()?;
            }
            {
                let (src_ptr, _g1) = new_v_dev.device_ptr(&stream);
                let mut codes_v_row = slab
                    .codes_v
                    .slice_mut(row_byte_offset..row_byte_offset + packed_row_bytes);
                let (dst_ptr, _g2) = codes_v_row.device_ptr_mut(&stream);
                unsafe {
                    copy_f16_to_quantized_device(
                        KvFormat::Iso3,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        kv_dim,
                        rq_stream,
                    )
                }
                .ok()?;
            }
        }
        slab.cached_seq_len = new_row_idx + 1;

        // Dequant all `new_row_idx + 1` rows (= preseed_n + seq_len)
        // to f32 scratch, then convert to f16 K/V scratches for the
        // existing attention kernel. The f16 scratches are sized to
        // `max_seq * kv_dim` each — they're transient per-layer,
        // written-and-immediately-read by one kernel call.
        let scratch_elements = max_seq * kv_dim;
        let dequant_elements = (new_row_idx + 1) * kv_dim;
        let mut f32_scratch_lock = self.qwen35_kv_dequant_f32_scratch.lock().ok()?;
        let mut f16_k_lock = self.qwen35_kv_dequant_f16_k_scratch.lock().ok()?;
        let mut f16_v_lock = self.qwen35_kv_dequant_f16_v_scratch.lock().ok()?;
        if f32_scratch_lock
            .as_ref()
            .is_none_or(|s| s.len() < scratch_elements)
        {
            *f32_scratch_lock = Some(drv.stream.alloc_zeros::<f32>(scratch_elements).ok()?);
        }
        if f16_k_lock
            .as_ref()
            .is_none_or(|s| s.len() < scratch_elements)
        {
            *f16_k_lock = Some(drv.stream.alloc_zeros::<half::f16>(scratch_elements).ok()?);
        }
        if f16_v_lock
            .as_ref()
            .is_none_or(|s| s.len() < scratch_elements)
        {
            *f16_v_lock = Some(drv.stream.alloc_zeros::<half::f16>(scratch_elements).ok()?);
        }
        let f32_scratch = f32_scratch_lock.as_mut()?;
        let k_scratch = f16_k_lock.as_mut()?;
        let v_scratch = f16_v_lock.as_mut()?;

        use cudarc::driver::{DevicePtr, DevicePtrMut};
        let stream = drv.stream.clone();
        let rq_stream = unsafe { RqStream::from_raw(stream.cu_stream() as *mut std::ffi::c_void) };

        // Dequant K → f32 → f16 K-scratch.
        {
            let (src_ptr, _g1) = slab.codes_k.device_ptr(&stream);
            let (dst_ptr, _g2) = f32_scratch.device_ptr_mut(&stream);
            unsafe {
                dequantize_to_f32_device(
                    KvFormat::Iso3,
                    src_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    dequant_elements,
                    rq_stream,
                )
            }
            .ok()?;
        }
        // Convert. `f32_to_f16_device_into` reads the FULL length of
        // `f32_scratch` — which is `scratch_elements` (max_seq*kv_dim),
        // not `dequant_elements`. The trailing rows we don't care
        // about get written too, but the attention kernel only reads
        // `pos+1 = seq_len` rows so the tail is harmless. Output
        // length matches the input length; both scratches are sized
        // identically.
        super::elem::f32_to_f16_device_into(self, f32_scratch, k_scratch).ok()?;

        // Dequant V → f32 → f16 V-scratch.
        {
            let (src_ptr, _g1) = slab.codes_v.device_ptr(&stream);
            let (dst_ptr, _g2) = f32_scratch.device_ptr_mut(&stream);
            unsafe {
                dequantize_to_f32_device(
                    KvFormat::Iso3,
                    src_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    dequant_elements,
                    rq_stream,
                )
            }
            .ok()?;
        }
        super::elem::f32_to_f16_device_into(self, f32_scratch, v_scratch).ok()?;

        // Run the existing fused attention kernel against the f16
        // scratches. The kernel will *also* write the new K/V row
        // into the scratches at `pos` — harmless since the scratches
        // are transient.
        let host_new = seq_len - 1;
        let new_k = &k_cache[host_new * kv_dim..(host_new + 1) * kv_dim];
        let new_v = &v_cache[host_new * kv_dim..(host_new + 1) * kv_dim];
        let q_dev = drv.device_buf_from(q).ok()?;
        let k_new_dev = drv.device_buf_from(new_k).ok()?;
        let v_new_dev = drv.device_buf_from(new_v).ok()?;

        let scale = (head_dim as f32).powf(-0.5);
        let opts = super::attn::FusedDecodeAttentionOpts {
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos: new_row_idx,
            max_seq,
            rotary_dim: 0, // host pre-roped; kernel skips RoPE
            rope_base: 0.0,
            eps: 0.0,
            qk_norm_offset: 0.0,
            attn_scale: scale,
            softcap: 0.0,
        };
        let out_dev = super::attn::fused_decode_attention_device_kv(
            self, &q_dev, &k_new_dev, &v_new_dev, k_scratch, v_scratch, None, None, opts,
        )
        .ok()?;

        let mut out_host = vec![0.0_f32; q_dim];
        drv.stream.memcpy_dtoh(&out_dev, &mut out_host).ok()?;
        drv.stream.synchronize().ok()?;
        Some(out_host)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DeviceBytesKey {
    ptr: usize,
    len: usize,
    head: u64,
    tail: u64,
}

impl DeviceBytesKey {
    fn from_slice(bytes: &[u8]) -> Self {
        fn read_u64(bytes: &[u8]) -> u64 {
            let mut out = [0u8; 8];
            let n = bytes.len().min(out.len());
            out[..n].copy_from_slice(&bytes[..n]);
            u64::from_le_bytes(out)
        }

        let tail_start = bytes.len().saturating_sub(8);
        Self {
            ptr: bytes.as_ptr() as usize,
            len: bytes.len(),
            head: read_u64(bytes),
            tail: read_u64(&bytes[tail_start..]),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DeviceStateKey {
    ptr: usize,
    len: usize,
}

impl DeviceStateKey {
    fn from_slice(values: &[f32]) -> Self {
        Self {
            ptr: values.as_ptr() as usize,
            len: values.len(),
        }
    }
}

// ── MatMul: real cuBLAS calls ──────────────────────────────────────────

/// Weight size (in f32 elements) at-or-below which the device cache
/// is used for the matmul `B` operand. **Default 0 = cache disabled.**
///
/// ## Why off by default
///
/// The 2026-05-25 RTX 4090 bench (`dsv4_bench_cpu_vs_cuda`) showed
/// this cache delivers **no measurable win on the DSv4 streaming
/// forward** and an earlier unbounded version OOM'd the card. Two
/// reasons it doesn't help the default path:
///
/// 1. `dsv4_streaming_model_forward_cached` re-loads + dequantizes
///    each layer's weights per token and **drops** them between
///    layers, so the `as_ptr()` cache key changes every call — the
///    cache always misses. (See [[dsv4-gpu-push]].)
/// 2. Even where it would hit, the streaming dequant (~30 s/layer)
///    dominates wall time; matmul htod is noise next to it.
///
/// Where it *is* useful: a **non-streaming hybrid** where attention
/// weights stay resident in host RAM with stable pointers and are
/// pushed to the GPU once (FFN-on-CPU / attention-on-GPU design).
/// In that mode set `LARQL_CUDA_WEIGHT_CACHE_MAX_ELEMS` to e.g.
/// `4194304` (4 M f32 = 16 MB) — large enough for every attention /
/// mHC / router / shared-expert weight, small enough to skip the
/// multi-GB MoE expert tensors that would blow VRAM.
///
/// Override at runtime with `LARQL_CUDA_WEIGHT_CACHE_MAX_ELEMS=N`.
const DEFAULT_WEIGHT_CACHE_MAX_ELEMS: usize = 0;

fn weight_cache_max_elems() -> usize {
    std::env::var("LARQL_CUDA_WEIGHT_CACHE_MAX_ELEMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WEIGHT_CACHE_MAX_ELEMS)
}

/// Return a device-resident `B` for the matmul, using the weight
/// cache when (a) the view is contiguous (cache key matches) and
/// (b) the tensor fits the size threshold.
fn cached_or_fresh_b(backend: &CudaBackend, b: ArrayView2<f32>) -> Arc<CudaSlice<f32>> {
    let max_elems = weight_cache_max_elems();
    match b.as_slice() {
        Some(b_slice) if max_elems > 0 && b_slice.len() <= max_elems => backend
            .arc_f32_weight_device_buf(b_slice)
            .expect("CudaBackend matmul: htod cached b failed"),
        Some(_) | None => {
            // Either too large to cache, or non-contiguous (strided/
            // transposed view; cache key wouldn't match anyway).
            // Collect-then-upload, no insert into cache.
            let b_buf = backend.as_contiguous(b);
            let dev = backend
                .drv
                .device_buf_from(&b_buf)
                .expect("CudaBackend matmul: htod b failed");
            Arc::new(dev)
        }
    }
}

impl MatMul for CudaBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k, k2, "matmul shape mismatch: {a:?} × {b:?}");

        // A is the per-call input — htod fresh. B is the weight —
        // route through `arc_f32_weight_device_buf` so repeated calls
        // borrow the same device buffer (this is the bench finding's
        // critical optimisation: see [[dsv4-gpu-push]]).
        let a_buf = self.as_contiguous(a);
        let a_dev = self
            .drv
            .device_buf_from(&a_buf)
            .expect("CudaBackend::matmul: htod a failed");
        let b_dev = cached_or_fresh_b(self, b);

        let c_dev = kernels::matmul_device_inout(&self.drv, &a_dev, &b_dev, m, n, k)
            .expect("CudaBackend::matmul: cuBLAS failed");
        self.drv.sync().expect("CudaBackend::matmul: sync failed");
        let out = self
            .drv
            .to_host(&c_dev)
            .expect("CudaBackend::matmul: dtoh failed");

        Array2::from_shape_vec((m, n), out).expect("CudaBackend::matmul: shape mismatch on result")
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        // C = A * B^T  with A: m×k, B: n×k → C: m×n
        let (m, k) = a.dim();
        let (n, k2) = b.dim();
        assert_eq!(k, k2, "matmul_transb shape mismatch: {a:?} × {b:?}^T");

        // Same caching strategy as `matmul`: A is input (fresh htod),
        // B is weight (cached). See [[dsv4-gpu-push]] for the bench
        // result motivating this.
        let a_buf = self.as_contiguous(a);
        let a_dev = self
            .drv
            .device_buf_from(&a_buf)
            .expect("CudaBackend::matmul_transb: htod a failed");
        let b_dev = cached_or_fresh_b(self, b);

        let c_dev = kernels::matmul_transb_device_inout(&self.drv, &a_dev, &b_dev, m, n, k)
            .expect("CudaBackend::matmul_transb: cuBLAS failed");
        self.drv
            .sync()
            .expect("CudaBackend::matmul_transb: sync failed");
        let out = self
            .drv
            .to_host(&c_dev)
            .expect("CudaBackend::matmul_transb: dtoh failed");

        Array2::from_shape_vec((m, n), out)
            .expect("CudaBackend::matmul_transb: shape mismatch on result")
    }

    fn f32_gemv(&self, w: ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        let (n, k) = w.dim();
        if x.len() != k {
            return None;
        }
        let w_buf = self.as_contiguous(w);
        kernels::gemv(&self.drv, &w_buf, x, n, k).ok()
    }

    /// f16 GEMV via session-cached dequantized f32 weight + cuBLAS sgemv.
    /// Lets `lm_head_knn_backend` route through CUDA on tied-embedding
    /// models (e.g. Gemma 3 270M) where the Q4_K direct matvec rejects
    /// non-multiple-of-256 hidden dims and the f16 mmap'd embeddings
    /// are the lm_head. First call uploads the dequantized weight
    /// (~84 MB f32 for 270M lm_head); subsequent calls reuse the cache.
    fn f16_gemv(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        if x.len() != k || w_f16.len() < n * k * 2 {
            return None;
        }
        let n_elements = n * k;
        self.with_f16_f32_device_buf(&w_f16[..n_elements * 2], n_elements, |w_dev| {
            kernels::gemv_device_w(&self.drv, w_dev, x, n, k)
        })
        .ok()
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn device_info(&self) -> String {
        self.drv.device_info()
    }

    fn supports(&self, cap: Capability) -> bool {
        if cap == Capability::CudaOxide {
            return cfg!(feature = "cuda-oxide");
        }
        // Capability bits flip on as sub-changes land:
        //   cuda-f32-baseline    → Cuda, F32Gemv
        //   cuda-q4-matvec       → +QuantMatVec, +Q4VecMat
        //   cuda-fused-attention → +FlashAttentionV2 (this change)
        //   rotorquant-*         → +KvCompressionRotorQuant
        matches!(
            cap,
            Capability::Cuda
                | Capability::F32Gemv
                | Capability::QuantMatVec
                | Capability::Q4VecMat
                | Capability::FlashAttentionV2
                | Capability::KvCompressionRotorQuant
                | Capability::DecodeToken
                | Capability::PrefillQ4
        )
    }

    fn qwen35_causal_conv1d_step(
        &self,
        weight: &[f32],
        state: &mut [f32],
        new: &[f32],
        d_conv: usize,
        conv_dim: usize,
        sequence_pos: usize,
    ) -> Option<Vec<f32>> {
        super::deltanet::causal_conv1d_step_cached(
            self,
            weight,
            state,
            new,
            d_conv,
            conv_dim,
            sequence_pos,
        )
        .ok()
    }

    fn qwen35_deltanet_step(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        log_g: &[f32],
        beta: &[f32],
        state: &mut [f32],
        s: usize,
        h_k: usize,
        h_v: usize,
        sequence_pos: usize,
        block_gqa: bool,
    ) -> Option<Vec<f32>> {
        super::deltanet::deltanet_step_cached(
            self,
            q,
            k,
            v,
            log_g,
            beta,
            state,
            s,
            h_k,
            h_v,
            sequence_pos,
            block_gqa,
        )
        .ok()
    }

    fn qwen35_l2_normalize_per_head(
        &self,
        x: &[f32],
        head_dim: usize,
        n_heads: usize,
        eps: f32,
    ) -> Option<Vec<f32>> {
        super::deltanet::l2_normalize_per_head(self, x, head_dim, n_heads, eps).ok()
    }

    fn qwen35_rms_norm_heads(
        &self,
        x: &[f32],
        weight: &[f32],
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Option<Vec<f32>> {
        super::deltanet::rms_norm_heads(self, x, weight, num_heads, head_dim, eps).ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_deltanet_recurrence_block(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        log_g: &[f32],
        beta: &[f32],
        ssm_norm_weight: &[f32],
        recurrent_state: &mut [f32],
        head_v_dim: usize,
        n_v_heads: usize,
        n_k_heads: usize,
        eps: f32,
        sequence_pos: usize,
        block_gqa: bool,
    ) -> Option<Vec<f32>> {
        super::qwen35_block::deltanet_recurrence_block_cached(
            self,
            q,
            k,
            v,
            log_g,
            beta,
            ssm_norm_weight,
            recurrent_state,
            head_v_dim,
            n_v_heads,
            n_k_heads,
            eps,
            sequence_pos,
            block_gqa,
        )
        .ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_ffn_lazy_block(
        &self,
        x: &[f32],
        gate_data: &[u8],
        gate_format: crate::QuantFormat,
        gate_rows: usize,
        up_data: &[u8],
        up_format: crate::QuantFormat,
        up_rows: usize,
        down_data: &[u8],
        down_format: crate::QuantFormat,
        down_rows: usize,
        hidden: usize,
    ) -> Option<Vec<f32>> {
        use crate::QuantFormat;
        // Only handle gate/up sharing input dim = hidden and producing
        // rows=ffn_dim, then down going ffn_dim → hidden. Mismatched
        // dims fall back to caller.
        if x.len() != hidden || gate_rows != up_rows {
            return None;
        }
        let ffn_dim = gate_rows;
        let drv = self.driver();

        // Dispatch per-format matvec to its device-resident variant.
        // Returns CudaSlice<f32> for downstream chaining.
        let matvec_dev = |data: &[u8],
                          format: QuantFormat,
                          rows: usize,
                          cols: usize,
                          x_dev: &cudarc::driver::CudaSlice<f32>|
         -> Option<cudarc::driver::CudaSlice<f32>> {
            match format {
                QuantFormat::Q4_K => {
                    super::q4k_direct::matvec_device(self, data, x_dev, rows, cols).ok()
                }
                QuantFormat::Q5_K => {
                    super::q5k_direct::matvec_device(self, data, x_dev, rows, cols).ok()
                }
                _ => None,
            }
        };

        let x_dev = drv.device_buf_from(x).ok()?;
        let gate_dev = matvec_dev(gate_data, gate_format, ffn_dim, hidden, &x_dev)?;
        let up_dev = matvec_dev(up_data, up_format, ffn_dim, hidden, &x_dev)?;
        // silu(gate) * up element-wise on device.
        let inter_dev =
            super::elem::silu_gate_up_device(self, &gate_dev, &up_dev, ffn_dim, false).ok()?;
        let down_dev = matvec_dev(down_data, down_format, down_rows, ffn_dim, &inter_dev)?;
        drv.sync().ok()?;
        drv.to_host(&down_dev).ok()
    }

    fn qwen35_paired_q4k_matvec(
        &self,
        a_data: &[u8],
        a_rows: usize,
        b_data: &[u8],
        b_rows: usize,
        x: &[f32],
        hidden: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        super::q4k_direct::matvec_pair(self, a_data, a_rows, b_data, b_rows, x, hidden).ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen35_deltanet_postproj_step(
        &self,
        qkv_mixed: &[f32],
        ssm_conv1d_weight: &[f32],
        log_g: &[f32],
        beta: &[f32],
        z: &[f32],
        ssm_norm_weight: &[f32],
        conv_state: &mut [f32],
        recurrent_state: &mut [f32],
        head_v_dim: usize,
        n_v_heads: usize,
        n_k_heads: usize,
        d_conv: usize,
        eps: f32,
        sequence_pos: usize,
        block_gqa: bool,
    ) -> Option<Vec<f32>> {
        super::qwen35_block::deltanet_postproj_step_cached(
            self,
            qkv_mixed,
            ssm_conv1d_weight,
            log_g,
            beta,
            z,
            ssm_norm_weight,
            conv_state,
            recurrent_state,
            head_v_dim,
            n_v_heads,
            n_k_heads,
            d_conv,
            eps,
            sequence_pos,
            block_gqa,
        )
        .ok()
    }

    fn qwen35_gqa_decode_step(
        &self,
        cache_id: u64,
        max_seq: usize,
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        if q.len() != q_dim
            || k_cache.len() != seq_len * kv_dim
            || v_cache.len() != seq_len * kv_dim
            || seq_len == 0
            || seq_len > max_seq
        {
            return None;
        }

        // `Step 4 Phase 2` dispatch: when `LARQL_QWEN35_KV_FORMAT=iso3`
        // route through the RotorQuant-compressed KV path. Otherwise
        // use the f16 device-resident cache (current default).
        if std::env::var("LARQL_QWEN35_KV_FORMAT").ok().as_deref() == Some("iso3") {
            return self.qwen35_gqa_decode_step_iso3(
                cache_id,
                max_seq,
                q,
                k_cache,
                v_cache,
                num_q_heads,
                num_kv_heads,
                head_dim,
                seq_len,
            );
        }

        let drv = self.driver();
        let mut cache = self.qwen35_kv_slab_cache.lock().ok()?;
        let slab = cache.entry(cache_id).or_insert_with_key(|_| {
            let cache_len = max_seq * kv_dim;
            let k = drv
                .stream
                .alloc_zeros::<half::f16>(cache_len)
                .expect("alloc qwen35_kv_slab K");
            let v = drv
                .stream
                .alloc_zeros::<half::f16>(cache_len)
                .expect("alloc qwen35_kv_slab V");
            Qwen35KvSlabDev {
                k,
                v,
                max_seq,
                kv_dim,
                cached_seq_len: 0,
            }
        });

        if slab.max_seq != max_seq || slab.kv_dim != kv_dim {
            let cache_len = max_seq * kv_dim;
            slab.k = drv.stream.alloc_zeros::<half::f16>(cache_len).ok()?;
            slab.v = drv.stream.alloc_zeros::<half::f16>(cache_len).ok()?;
            slab.max_seq = max_seq;
            slab.kv_dim = kv_dim;
            slab.cached_seq_len = 0;
        }

        // `LARQL_QWEN35_KV_PRESEED=N` debug knob: on first allocation
        // of this layer's slab, treat positions [0, N) as if they
        // were populated by a prior prefill. The cache contents are
        // whatever `alloc_zeros` produced (all f16 zeros) — output
        // won't be coherent but VRAM peak + per-token wall time are
        // measured under the long-context pressure they'd see with
        // a real 32K/128K prefill. Lets the bench harness validate
        // the architectural VRAM savings without paying for the
        // ~37 min/data-point prefill at 32K (or ~2.5 hr at 128K).
        let preseed_n = std::env::var("LARQL_QWEN35_KV_PRESEED")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0)
            .min(max_seq.saturating_sub(seq_len));
        if slab.cached_seq_len == 0 && preseed_n > 0 {
            // Zeros already populated by `alloc_zeros`. Just mark the
            // device cache as containing `preseed_n` rows so the
            // attention kernel reads them.
            slab.cached_seq_len = preseed_n;
        }
        let effective_new_row_idx = preseed_n + (seq_len - 1);

        // Stale-cache repopulation: same idea as before, but
        // accounting for the preseed offset (rows 0..preseed_n are
        // already considered cached; only rows >= preseed_n need
        // host data uploaded).
        if slab.cached_seq_len != effective_new_row_idx {
            let host_prev = seq_len - 1;
            if host_prev > 0 {
                let k_prev_f16: Vec<half::f16> = k_cache[..host_prev * kv_dim]
                    .iter()
                    .map(|&v| half::f16::from_f32(v))
                    .collect();
                let v_prev_f16: Vec<half::f16> = v_cache[..host_prev * kv_dim]
                    .iter()
                    .map(|&v| half::f16::from_f32(v))
                    .collect();
                let off_elements = preseed_n * kv_dim;
                drv.stream
                    .memcpy_htod(
                        &k_prev_f16,
                        &mut slab
                            .k
                            .slice_mut(off_elements..off_elements + k_prev_f16.len()),
                    )
                    .ok()?;
                drv.stream
                    .memcpy_htod(
                        &v_prev_f16,
                        &mut slab
                            .v
                            .slice_mut(off_elements..off_elements + v_prev_f16.len()),
                    )
                    .ok()?;
            }
            slab.cached_seq_len = effective_new_row_idx;
        }

        let new_k = &k_cache[(seq_len - 1) * kv_dim..seq_len * kv_dim];
        let new_v = &v_cache[(seq_len - 1) * kv_dim..seq_len * kv_dim];
        let q_dev = drv.device_buf_from(q).ok()?;
        let k_new_dev = drv.device_buf_from(new_k).ok()?;
        let v_new_dev = drv.device_buf_from(new_v).ok()?;

        let scale = (head_dim as f32).powf(-0.5);
        let opts = super::attn::FusedDecodeAttentionOpts {
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos: effective_new_row_idx,
            max_seq,
            // qwen35_attention_block_step applies QK-norm + RoPE on
            // host before reaching here; passing `rotary_dim = 0`
            // tells the kernel to skip RoPE (post-attn.cu fix shipped
            // alongside this revival — see the FUSED_DECODE_ATTN_SRC
            // doc comment on `rdim_pre`).
            rotary_dim: 0,
            rope_base: 0.0,
            eps: 0.0,
            qk_norm_offset: 0.0,
            attn_scale: scale,
            softcap: 0.0,
        };
        let out_dev = super::attn::fused_decode_attention_device_kv(
            self,
            &q_dev,
            &k_new_dev,
            &v_new_dev,
            &mut slab.k,
            &mut slab.v,
            None, // no qk-norm — host already applied
            None,
            opts,
        )
        .ok()?;

        slab.cached_seq_len = effective_new_row_idx + 1;
        let mut out_host = vec![0.0_f32; q_dim];
        drv.stream.memcpy_dtoh(&out_dev, &mut out_host).ok()?;
        drv.stream.synchronize().ok()?;
        Some(out_host)
    }

    fn qwen35_gqa_decode_reset(&self, cache_id: u64) {
        if let Ok(mut cache) = self.qwen35_kv_slab_cache.lock() {
            cache.remove(&cache_id);
        }
        // Also drop the iso3 slab if present so the per-sequence
        // reset clears both code paths.
        if let Ok(mut cache) = self.qwen35_kv_slab_cache_quant.lock() {
            cache.remove(&cache_id);
        }
    }

    fn qwen35_attention_prefill_batch(
        &self,
        cache_id: u64,
        max_seq: usize,
        base_pos: usize,
        q_seq: &[f32],
        k_seq: &[f32],
        v_seq: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Option<Vec<f32>> {
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        if q_seq.len() != seq_len * q_dim
            || k_seq.len() != seq_len * kv_dim
            || v_seq.len() != seq_len * kv_dim
            || seq_len == 0
            || base_pos + seq_len > max_seq
        {
            return None;
        }
        let drv = self.driver();

        // Get/allocate the per-layer f16 KV slab — same one the
        // single-step decode method uses, so a subsequent decode
        // call finds the populated cache. (The iso3 path's slab
        // is separate; Phase 4b currently wires only the f16
        // route — iso3 prefill batching is a follow-on.)
        let mut cache = self.qwen35_kv_slab_cache.lock().ok()?;
        let slab = cache.entry(cache_id).or_insert_with_key(|_| {
            let cache_len = max_seq * kv_dim;
            let k = drv
                .stream
                .alloc_zeros::<half::f16>(cache_len)
                .expect("alloc prefill K");
            let v = drv
                .stream
                .alloc_zeros::<half::f16>(cache_len)
                .expect("alloc prefill V");
            Qwen35KvSlabDev {
                k,
                v,
                max_seq,
                kv_dim,
                cached_seq_len: 0,
            }
        });
        if slab.max_seq != max_seq || slab.kv_dim != kv_dim {
            let cache_len = max_seq * kv_dim;
            slab.k = drv.stream.alloc_zeros::<half::f16>(cache_len).ok()?;
            slab.v = drv.stream.alloc_zeros::<half::f16>(cache_len).ok()?;
            slab.max_seq = max_seq;
            slab.kv_dim = kv_dim;
            slab.cached_seq_len = 0;
        }

        // htod the per-position Q/K/V (f32 host-roped).
        let q_dev = drv.device_buf_from(q_seq).ok()?;
        let k_dev = drv.device_buf_from(k_seq).ok()?;
        let v_dev = drv.device_buf_from(v_seq).ok()?;
        let mut out_dev = drv.device_alloc_uninit(seq_len * q_dim).ok()?;

        let scale = (head_dim as f32).powf(-0.5);
        let opts = super::attn::FusedDecodeAttentionOpts {
            num_q_heads,
            num_kv_heads,
            head_dim,
            pos: 0, // unused by the prefill kernel (it reads base_pos+sp)
            max_seq,
            // host pre-roped; skip RoPE in kernel (post the PR #225
            // attn.cu fix, `rotary_dim = 0` means skip RoPE).
            rotary_dim: 0,
            rope_base: 0.0,
            eps: 0.0,
            qk_norm_offset: 0.0,
            attn_scale: scale,
            softcap: 0.0,
        };
        super::attn::fused_prefill_attention_seq_device_into(
            self,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut slab.k,
            &mut slab.v,
            None, // no qk-norm — host already applied
            None,
            &mut out_dev,
            base_pos,
            seq_len,
            opts,
        )
        .ok()?;

        // Advance the slab's cached_seq_len to reflect newly-written
        // rows so a follow-on single-step decode call lands at the
        // right offset.
        slab.cached_seq_len = base_pos + seq_len;

        let mut out_host = vec![0.0_f32; seq_len * q_dim];
        drv.stream.memcpy_dtoh(&out_dev, &mut out_host).ok()?;
        drv.stream.synchronize().ok()?;
        Some(out_host)
    }

    fn qwen35_moe_ffn_batch(
        &self,
        x: &[f32],
        hidden: usize,
        ffn_dim: usize,
        experts: &[crate::backend::MoeFfnExpert<'_>],
    ) -> Option<Vec<Vec<f32>>> {
        use crate::QuantFormat;
        if x.len() != hidden || experts.is_empty() {
            return None;
        }
        let drv = self.driver();
        if drv.worker_streams.is_empty() {
            return None;
        }
        // Every expert must be Q4_K or Q5_K (the only formats we have
        // on-stream device kernels for). Mixed formats are fine — each
        // matvec dispatches to its own kernel.
        let supported = |f: QuantFormat| matches!(f, QuantFormat::Q4_K | QuantFormat::Q5_K);
        if !experts
            .iter()
            .all(|e| supported(e.gate_format) && supported(e.up_format) && supported(e.down_format))
        {
            return None;
        }

        // 1. Upload x once on the default stream and pre-warm every
        //    expert's weight cache, then sync so every worker stream
        //    sees populated buffers. cudarc event tracking is
        //    disabled (driver.rs:48) so cross-stream reads of
        //    `x_dev` / weight buffers rely on this explicit sync for
        //    ordering — without the pre-warm a worker kernel could
        //    race with the cache-population htod on drv.stream and
        //    read uninitialised device memory.
        let x_dev = drv.device_buf_from(x).ok()?;
        let prewarm = |bytes: &[u8], fmt: QuantFormat| -> Option<()> {
            match fmt {
                QuantFormat::Q4_K => self.arc_q4k_device_buf(bytes).ok().map(|_| ()),
                QuantFormat::Q5_K => self.with_q5k_device_buf(bytes, |_| Ok(())).ok(),
                _ => None,
            }
        };
        for e in experts.iter() {
            prewarm(e.gate_data, e.gate_format)?;
            prewarm(e.up_data, e.up_format)?;
            prewarm(e.down_data, e.down_format)?;
        }
        drv.sync().ok()?;

        // 2. Issue each expert chain on its own worker stream. The
        //    Q4/Q5 weight uploads cache by `DeviceBytesKey` inside the
        //    backend's per-format mutex, so concurrent calls serialise
        //    on first-touch but hit on subsequent same-tensor reuse —
        //    typical MoE pattern reuses the same expert weights every
        //    layer revisit.
        struct WorkerResult {
            stream: std::sync::Arc<cudarc::driver::CudaStream>,
            down_dev: cudarc::driver::CudaSlice<f32>,
        }
        let dispatch = |i: usize| -> Option<WorkerResult> {
            let expert = &experts[i];
            let stream = drv.worker_streams[i % drv.worker_streams.len()].clone();
            let gate_dev = match expert.gate_format {
                QuantFormat::Q4_K => super::q4k_direct::matvec_device_on_stream(
                    self,
                    expert.gate_data,
                    &x_dev,
                    ffn_dim,
                    hidden,
                    &stream,
                ),
                QuantFormat::Q5_K => super::q5k_direct::matvec_device_on_stream(
                    self,
                    expert.gate_data,
                    &x_dev,
                    ffn_dim,
                    hidden,
                    &stream,
                ),
                _ => return None,
            }
            .ok()?;
            let up_dev = match expert.up_format {
                QuantFormat::Q4_K => super::q4k_direct::matvec_device_on_stream(
                    self,
                    expert.up_data,
                    &x_dev,
                    ffn_dim,
                    hidden,
                    &stream,
                ),
                QuantFormat::Q5_K => super::q5k_direct::matvec_device_on_stream(
                    self,
                    expert.up_data,
                    &x_dev,
                    ffn_dim,
                    hidden,
                    &stream,
                ),
                _ => return None,
            }
            .ok()?;
            let inter_dev = super::elem::silu_gate_up_device_on_stream(
                self, &gate_dev, &up_dev, ffn_dim, false, &stream,
            )
            .ok()?;
            let down_dev = match expert.down_format {
                QuantFormat::Q4_K => super::q4k_direct::matvec_device_on_stream(
                    self,
                    expert.down_data,
                    &inter_dev,
                    hidden,
                    ffn_dim,
                    &stream,
                ),
                QuantFormat::Q5_K => super::q5k_direct::matvec_device_on_stream(
                    self,
                    expert.down_data,
                    &inter_dev,
                    hidden,
                    ffn_dim,
                    &stream,
                ),
                _ => return None,
            }
            .ok()?;
            Some(WorkerResult { stream, down_dev })
        };

        let mut workers = Vec::with_capacity(experts.len());
        for i in 0..experts.len() {
            workers.push(dispatch(i)?);
        }

        // 3. Sync each worker stream + dtoh its output. The
        //    per-stream synchronize is what makes the result safe to
        //    return — without event tracking cudarc won't otherwise
        //    enforce completion before the dtoh.
        let mut outputs = Vec::with_capacity(workers.len());
        for w in workers {
            w.stream.synchronize().ok()?;
            let mut host = vec![0.0_f32; w.down_dev.len()];
            w.stream.memcpy_dtoh(&w.down_dev, &mut host).ok()?;
            // memcpy_dtoh on a stream is async; another sync to be safe.
            w.stream.synchronize().ok()?;
            outputs.push(host);
        }
        Some(outputs)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Base smoke test that runs anywhere — no GPU required when the
    /// driver is missing, the call returns Err and the test passes.
    #[test]
    fn driver_missing_returns_typed_error() {
        match CudaBackend::new() {
            Ok(b) => {
                assert_eq!(b.name(), "cuda");
            }
            Err(CudaInitError::DriverMissing(_))
            | Err(CudaInitError::NoDevices)
            | Err(CudaInitError::ToolkitMismatch { .. }) => {
                // Expected on a host without a working CUDA driver.
            }
            Err(CudaInitError::NotImplemented(_)) => {
                panic!("backend should no longer report NotImplemented after f32 baseline");
            }
        }
    }

    #[test]
    fn supports_f32_gemv_after_baseline() {
        // We only assert the capability set if init succeeded; on
        // hosts without CUDA the test no-ops.
        if let Ok(b) = CudaBackend::new() {
            assert!(b.supports(Capability::Cuda));
            assert!(
                b.supports(Capability::F32Gemv),
                "cuda-f32-baseline must advertise F32Gemv"
            );
            assert!(b.supports(Capability::DecodeToken));
        }
    }

    #[test]
    fn supports_q4_matvec_after_q4_baseline() {
        if let Ok(b) = CudaBackend::new() {
            // Capabilities flipped on by cuda-q4-matvec.
            assert!(b.supports(Capability::QuantMatVec));
            assert!(b.supports(Capability::Q4VecMat));
            assert!(b.supports(Capability::KvCompressionRotorQuant));
            assert!(b.supports(Capability::DecodeToken));
        }
    }

    #[test]
    fn supports_fa2_after_fused_attention() {
        if let Ok(b) = CudaBackend::new() {
            // Cumulative capability set after cuda-fused-attention.
            assert!(b.supports(Capability::Cuda));
            assert!(b.supports(Capability::F32Gemv));
            assert!(b.supports(Capability::QuantMatVec));
            assert!(b.supports(Capability::Q4VecMat));
            assert!(b.supports(Capability::FlashAttentionV2));
            assert!(b.supports(Capability::DecodeToken));
            assert!(b.supports(Capability::PrefillQ4));
            assert!(b.supports(Capability::KvCompressionRotorQuant));
        }
    }

    #[test]
    fn supports_decode_after_cuda_decode_backend() {
        if let Ok(b) = CudaBackend::new() {
            assert!(b.supports(Capability::DecodeToken));
            assert!(b.supports(Capability::PrefillQ4));
        }
    }

    #[test]
    fn supports_cuda_oxide_when_feature_enabled() {
        if let Ok(b) = CudaBackend::new() {
            assert_eq!(
                b.supports(Capability::CudaOxide),
                cfg!(feature = "cuda-oxide")
            );
        }
    }

    /// `cuda-decode-cuda-graph` viability probe: confirms that
    /// cudarc 0.19's stream capture / graph replay mechanism actually
    /// works for our usage pattern (NVRTC-compiled kernel launched via
    /// `launch_builder`). If this test fails, the broader change is
    /// dead in the water and we'd skip the multi-day refactor.
    #[test]
    fn cuda_graph_capture_replay_smoke_test() {
        use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use cudarc::nvrtc::compile_ptx;

        if std::env::var("LARQL_CUDA_AVAILABLE").ok().as_deref() != Some("1") {
            return;
        }
        let Ok(backend) = CudaBackend::new() else {
            return;
        };
        // `Driver::new_with_index` already disabled event tracking
        // and switched to a non-default stream — both prerequisites
        // for CUDA Graph capture (CUDA_ERROR_STREAM_CAPTURE_ISOLATION
        // otherwise). Capture on the same stream that owns the
        // buffers.
        let ctx = &backend.driver().ctx;
        let stream = backend.driver().stream.clone();

        let src = r#"
extern "C" __global__ void axpb(const float* x, float* y, int n, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + b;
}
"#;
        let ptx = compile_ptx(src).expect("compile axpb");
        let module = ctx.load_module(ptx).expect("load axpb module");
        let func = module.load_function("axpb").expect("load axpb function");

        let n = 1024_usize;
        let x_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x_dev = stream.clone_htod(&x_host).expect("htod");
        let mut y_dev = unsafe { stream.alloc::<f32>(n).expect("alloc y") };
        // Drain pending alloc/htod work BEFORE entering capture mode —
        // CUDA_ERROR_STREAM_CAPTURE_ISOLATION otherwise.
        stream.synchronize().expect("pre-capture sync");

        let cfg = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_i = n as i32;
        let a = 2.0_f32;
        let b = 1.0_f32;

        // Compute the reference on host (avoid any cross-stream dep
        // confusion from a prior on-device launch).
        let y_ref: Vec<f32> = x_host.iter().map(|&xi| a * xi + b).collect();

        // ── Capture ────────────────────────────────
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .expect("begin_capture");
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&x_dev)
                .arg(&mut y_dev)
                .arg(&n_i)
                .arg(&a)
                .arg(&b)
                .launch(cfg)
                .expect("launch in capture");
        }
        let graph = stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .expect("end_capture")
            .expect("graph should not be empty");

        // Trash the output buffer to make sure replay actually does work.
        stream.memset_zeros(&mut y_dev).expect("memset");
        stream.synchronize().expect("sync");

        // ── Replay ──────────────────────────────────
        graph.launch().expect("graph launch");
        stream.synchronize().expect("sync");
        let y_replay = stream.clone_dtoh(&y_dev).expect("dtoh replay");

        let max_diff = y_ref
            .iter()
            .zip(&y_replay)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "graph replay drift {max_diff} > 1e-6 — capture mechanism is broken"
        );
    }
}
