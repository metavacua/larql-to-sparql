//! Metal GPU compute backend — Apple Silicon.
//!
//! All operations go through the [`ComputeBackend`] trait. Metal-specific
//! optimisations: simdgroup reductions, uint4 vectorised loads, native half reads,
//! zero-copy mmap buffers, single command buffer pipeline.
//!
//! ## Modules
//!
//! - `shaders/`:  Metal Shading Language — one file per kernel (~30 shader files)
//! - `ops/`:      GPU dispatch — one file per operation (6 dispatchers)
//! - `buffers`:   GPU buffer cache (zero-copy mmap, transient allocation)
//! - `f32_ops`:   f32 tiled matmul dispatch with GPU/CPU routing
//! - `calibrate`: CPU vs GPU auto-calibration
//!
//! ## Performance (M3 Max, Gemma 3 4B, 34 layers)
//!
//! - Full decode: ~0.38ms/layer, ~77 tok/s (Q4_KF path)
//! - vs Ollama: ~1.0–1.25× (at parity)
//! - Q4_K matvec: uint4 loads, 8 rows/TG, multi-row (nr0=2)
//! - KV attention: simd_max/simd_sum reductions, float4 Q·K dot products

mod attention_kernels; // AttentionKernels registry (M3 incremental — third of four)
pub mod buffers;
pub mod calibrate;
mod decode;
mod decode_hybrid;
/// Diagnostic and profiling tools — kernel bandwidth, decode-stage timing,
/// layer-level residual dumps. See `diag/mod.rs` for the full index.
pub mod diag;
mod direct_ops;
pub mod f32_ops;
mod ffn_kernels; // FfnKernels registry (M3 incremental — fourth of four; M3 complete)
mod flags; // cached env-var-derived backend flags (DecodeFlags)
pub mod kernel; // KernelHandle: pipeline + dispatch geometry, bundled
mod moe_dispatch;
mod norm_kernels; // NormKernels registry (M3 incremental — first of four)
mod quant_kernels; // QuantKernels registry (M3 incremental — second of four)
pub use attention_kernels::AttentionKernels;
pub use decode::profile::take_last_split_timings;
pub use ffn_kernels::FfnKernels;
pub use flags::{BackendOptions, DecodeFlags};
pub use moe_dispatch::MoeScratch;
pub use norm_kernels::NormKernels;
pub use quant_kernels::QuantKernels;
pub mod ops; // modular: ops/mod.rs → one file per operation
mod pipeline;
pub mod shaders; // modular: shaders/mod.rs → one file per shader
pub mod stages; // modular: stages/mod.rs → one file per pipeline stage
mod trait_impl;

use metal::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use buffers::BufferCache;
use f32_ops::F32Ops;
use kernel::KernelHandle;
use ops::q4_common::Q4Pipelines;

/// Metal GPU compute backend.
///
/// ## Pipeline field convention
///
/// Fields fall into two camps:
///
/// - **`KernelHandle`** — simdgroup-tiled kernels with hard-coded row
///   maps (`row_idx = tg_id * ROWS_PER_TG + sg_id`). Geometry travels
///   with the pipeline; dispatchers read `kernel.rows_per_tg` /
///   `kernel.threads_per_tg` rather than importing constants from a
///   shader module. This is the bug class the q4_matvec_v4 75 %-row
///   drop introduced (see ROADMAP ship log).
///
/// - **`ComputePipelineState`** — flat `dispatch_threads` kernels
///   (one thread per output element / row) or attention-shape
///   kernels (per-head dispatch). No row-map drift risk because the
///   dispatcher already specifies the geometry per call.
///
/// Twelve simdgroup-tiled fields use `KernelHandle`. The rest stay
/// bare. Decision per remaining field:
/// - `geglu_*`, `silu`, `gelu_tanh`, `residual_add`, `scale_vector` →
///   element-wise, flat dispatch.
/// - `rms_norm*`, `layer_norm*`, `v_norm*`, `qk_norm`, `residual_norm*`
///   → per-row reduction, flat dispatch (one threadgroup per row).
/// - `causal_attn`, `fused_attn`, `kv_attend`, `kv_append` → attention
///   geometry (per-head/per-position), not row-tiled.
/// - `rope_*`, `q8_quant` → flat dispatch_threads.
pub struct MetalBackend {
    queue: CommandQueue,
    bufs: BufferCache,
    f32_ops: F32Ops,
    pub q4: Q4Pipelines,
    /// Norm + residual + scale-vector pipelines. See [`NormKernels`].
    pub norms: NormKernels,
    /// Format-primitive matvec / matmul / quantize pipelines (Q4_K
    /// 4sg/8sg/stride32 + matmul, Q6_K 4sg/8sg, Q8 matvec, Q8 quant).
    /// See [`QuantKernels`].
    pub quant: QuantKernels,
    /// Attention dispatch + RoPE + QKV-projection pipelines (KV
    /// attend / append, fused-attn opt-in, RoPE variants, Q4_K /
    /// Q4_KF / Q4_K-Q6K-V / Q8 QKV proj). See [`AttentionKernels`].
    pub attention: AttentionKernels,
    /// FFN dispatch pipelines: gate+up variants (Q4_K production +
    /// `f16acc`/`8sg`/`coop` opt-ins, Q4_KF), activation kernels
    /// (silu/gelu_tanh + their geglu twins), fused activation+down
    /// (Q4_K and Q6_K). See [`FfnKernels`].
    pub ffn: FfnKernels,
    // (LayerNorm / V-norm / QK-norm / qk-norm-rope / post-norm fusions
    //  / scale_vector — moved into `NormKernels` (the `norms` field).
    //  RoPE / KV-attend / fused-attn / QKV-projection — moved into
    //  `AttentionKernels` (the `attention` field).
    //  geglu / silu / gelu_tanh / q4k_ffn_gate_up* / q4kf_ffn_gate_up
    //  / q4k_geglu_*_down / q6k_geglu_*_down* — moved into
    //  `FfnKernels` (the `ffn` field).)
    /// KV cache for decode mode — initialized on first decode_token call.
    kv_cache: std::sync::Mutex<Option<ops::kv_cache::KVCache>>,
    /// Pre-allocated MoE scratch for `decode_token_q4k_moe` — keyed
    /// by `(top_k, hidden, intermediate_size)`. Reused across decode
    /// calls so the ~15 buffer allocations (~120ms on Gemma 4 26B-A4B,
    /// M3 Max) only happen at first use, not per token. Mirrors the
    /// shape cache `larql-server` keeps in `state.rs::moe_scratches`,
    /// pulled inside the backend so the local decode path benefits
    /// without each caller threading a cache through.
    moe_scratch: std::sync::Mutex<Option<moe_dispatch::MoeScratch>>,
    /// Per-Layer Embeddings precomputed input table (Gemma 4 E2B).
    ///
    /// Set by [`prepare_ple_inputs`](Self::prepare_ple_inputs) before each
    /// `decode_token*` / prefill call when the active arch needs PLE; the
    /// per-layer Metal dispatch reads this buffer + offset for the active
    /// (layer, position). `None` for non-PLE archs.
    ///
    /// Carried on the backend (rather than threaded through every decode
    /// call) so the Metal-side trait surface and the per-layer dispatch
    /// signatures don't grow an extra arg for a feature only Gemma 4 E2B
    /// uses today.
    ple_inputs: std::sync::Mutex<Option<PleInputBuffer>>,
    // (rms_norm_q8 / residual_norm{,_q8,_store} — moved into
    //  `NormKernels` (the `norms` field).)
    /// Dedicated row-per-simdgroup f32 gemv for the LM head. Used in
    /// autoregressive decode where `matmul_transb(query, lm_head)` shows
    /// up as the dominant per-token cost.
    pub f32_gemv_pipeline: KernelHandle,
    pub f32_argmax_partial_pipeline: ComputePipelineState,
    /// Per-TG top-K reduction over a scores buffer. Produces `K_TOPK = 8`
    /// (val, idx) pairs per TG; CPU final reduction merges into the caller's
    /// requested top-k. Used by the lm_head top_k=5 path on Gemma 3/4.
    pub f32_topk_partial_pipeline: ComputePipelineState,
    /// Same layout as [`Self::f32_gemv_pipeline`], but with a `half`
    /// weight matrix. Halves bandwidth for tied-embedding models whose
    /// lm_head would otherwise live as a 5.6 GB f32 clone on 31B.
    pub f16_gemv_pipeline: KernelHandle,
    flop_threshold: AtomicUsize,
    /// Decode-path flag snapshot copied from
    /// [`BackendOptions::decode_flags`] at construction. Captured once
    /// so the hot path (encode_attn / encode_qkv / encode_ffn /
    /// encode_post_ffn / decode/mod.rs) doesn't pay ~12 `getenv`
    /// syscalls per layer per token. Construct a fresh backend (via
    /// [`new`](Self::new) or [`with_options`](Self::with_options)) to
    /// pick up flag changes.
    pub decode_flags: DecodeFlags,
}

impl MetalBackend {
    /// Create a Metal backend with default options derived from the
    /// process environment. Returns `None` if no Metal device is
    /// available.
    ///
    /// The historical env-driven defaults (`LARQL_Q4K_MATVEC_8SG`,
    /// `LARQL_Q6K_8SG`, `LARQL_FUSED_*`, etc.) keep working through
    /// [`BackendOptions::from_env`]. Callers that want explicit,
    /// shell-independent control should use
    /// [`with_options`](Self::with_options) instead.
    pub fn new() -> Option<Self> {
        Self::with_options(BackendOptions::from_env())
    }

    /// Create a Metal backend with explicit options. Returns `None` if
    /// no Metal device is available.
    pub fn with_options(backend_options: BackendOptions) -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let all_src = shaders::all_shaders();
        let library = device
            .new_library_with_source(&all_src, &opts)
            .map_err(|e| eprintln!("[metal] shader compile error: {e}"))
            .ok()?;

        use kernel::get_shader_pipeline;

        let f32_ops = F32Ops {
            sgemm_pipeline: get_shader_pipeline::<shaders::sgemm::Kernel>(&device, &library)?,
            transb_pipeline: get_shader_pipeline::<shaders::sgemm_transb::Kernel>(
                &device, &library,
            )?,
        };

        // (causal_attn now lives inside `AttentionKernels`.)

        // Q4 family pipelines.
        //
        // `matvec` is simdgroup-tiled. Its kernel name + row map +
        // threads-per-TG live in `shaders/q4_matvec_v4.rs` via the
        // `TiledKernel` impl on the `Kernel` marker; binding it here
        // is one type-parameter line. To swap to a future v6, change
        // `q4_matvec_v4::Kernel` → `q4_matvec_v6::Kernel` here and
        // nothing else. See `metal::kernel` and the q4_matvec_v4
        // 75 %-row-drop ship-log entry.
        //
        // `vecmat` and `f32_matvec` use flat `dispatch_threads` — no
        // per-TG geometry, bare pipeline state is enough.
        let q4 = Q4Pipelines {
            matvec: KernelHandle::from_kernel::<shaders::q4_matvec_v4::Kernel>(&device, &library)?,
            vecmat: get_shader_pipeline::<shaders::q4_vecmat::Kernel>(&device, &library)?,
            f32_matvec: get_shader_pipeline::<shaders::q4_f32_matvec::Kernel>(&device, &library)?,
        };

        let bufs = BufferCache::new(&device);

        // Norm + residual + scale-vector pipelines, bundled.
        let norms = NormKernels::build(&device, &library)?;

        // Format-primitive matvec / matmul / Q8-quantize pipelines,
        // bundled. The production `q4k_matvec_pipeline` and
        // `q6k_matvec_pipeline` aliases are picked from
        // `backend_options` here (replaces the inline 4sg/8sg branches
        // that previously lived between the per-variant pipeline
        // constructors).
        let quant = QuantKernels::build(&device, &library, &backend_options)?;

        // FFN dispatch pipelines (gate+up variants, activations,
        // fused activation+down for Q4_K and Q6_K), bundled.
        let ffn = FfnKernels::build(&device, &library)?;

        // (Q8 QKV projection now lives inside `AttentionKernels`.)
        // (Norm + residual + Q8-norm fusion pipelines now live inside
        //  `NormKernels` — see the `norms` binding above.)
        // Attention dispatch + RoPE + QKV projection pipelines, bundled.
        let attention = AttentionKernels::build(&device, &library)?;

        // Dedicated f32 / f16 gemv for the LM head (KernelHandle).
        let f32_gemv_pipeline =
            KernelHandle::from_kernel::<shaders::f32_gemv::Kernel>(&device, &library)?;
        let f32_argmax_partial_pipeline =
            get_shader_pipeline::<shaders::f32_gemv::ArgmaxKernel>(&device, &library)?;
        let f32_topk_partial_pipeline =
            get_shader_pipeline::<shaders::f32_gemv::TopKKernel>(&device, &library)?;
        let f16_gemv_pipeline =
            KernelHandle::from_kernel::<shaders::f16_gemv::Kernel>(&device, &library)?;

        // (RoPE / QKV projection / fused-attn — moved into
        //  `AttentionKernels` (the `attention` binding above).
        //  geglu / silu / gelu_tanh / q4k_ffn_gate_up* /
        //  q4kf_ffn_gate_up / q4k_geglu_*_down / q6k_geglu_*_down* —
        //  moved into `FfnKernels` (the `ffn` binding above).)

        Some(Self {
            queue,
            bufs,
            f32_ops,
            q4,
            norms,
            quant,
            attention,
            ffn,
            kv_cache: std::sync::Mutex::new(None),
            moe_scratch: std::sync::Mutex::new(None),
            ple_inputs: std::sync::Mutex::new(None),
            f32_gemv_pipeline,
            f32_argmax_partial_pipeline,
            f32_topk_partial_pipeline,
            f16_gemv_pipeline,
            flop_threshold: AtomicUsize::new(calibrate::DEFAULT_FLOP_THRESHOLD),
            decode_flags: backend_options.decode_flags,
        })
    }

    /// Auto-calibrate CPU vs GPU threshold.
    pub fn calibrate(&self) {
        let threshold = calibrate::calibrate(&self.f32_ops, &self.queue, &self.bufs);
        self.flop_threshold.store(threshold, Ordering::Relaxed);
    }

    pub fn flop_threshold(&self) -> usize {
        self.flop_threshold.load(Ordering::Relaxed)
    }
    pub fn set_flop_threshold(&self, t: usize) {
        self.flop_threshold
            .store(t.max(calibrate::MIN_FLOP_FLOOR), Ordering::Relaxed);
    }
    pub fn cache_size(&self) -> usize {
        self.bufs.len()
    }
    pub fn bufs(&self) -> &BufferCache {
        &self.bufs
    }
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Access the KV cache for hybrid decode (GPU attention + CPU FFN).
    /// Creates the cache on first access.
    pub fn kv_cache_mut(
        &self,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> std::sync::MutexGuard<'_, Option<ops::kv_cache::KVCache>> {
        let mut guard = self.kv_cache.lock().unwrap();
        let shapes = vec![(num_kv_heads, head_dim); num_layers];
        self.ensure_kv_cache_for_shapes(&mut guard, &shapes, decode::DEFAULT_KV_CACHE_MAX_SEQ);
        guard
    }

    /// Access the KV cache using per-layer pipeline geometry.
    ///
    /// This is the preferred path for heterogeneous attention layouts; it
    /// avoids the legacy uniform `(num_kv_heads, head_dim)` fallback.
    pub fn kv_cache_mut_for_layers(
        &self,
        layers: &[crate::FullPipelineLayer<'_>],
    ) -> std::sync::MutexGuard<'_, Option<ops::kv_cache::KVCache>> {
        let mut guard = self.kv_cache.lock().unwrap();
        self.ensure_kv_cache_for_layers(&mut guard, layers, decode::DEFAULT_KV_CACHE_MAX_SEQ);
        guard
    }

    /// Access the KV cache using explicit per-layer geometry.
    ///
    /// Use this when call sites pass absolute layer indices and only hold a
    /// slice of pipeline layers locally.
    pub fn kv_cache_mut_for_shapes(
        &self,
        shapes: &[(usize, usize)],
    ) -> std::sync::MutexGuard<'_, Option<ops::kv_cache::KVCache>> {
        let mut guard = self.kv_cache.lock().unwrap();
        self.ensure_kv_cache_for_shapes(&mut guard, shapes, decode::DEFAULT_KV_CACHE_MAX_SEQ);
        guard
    }

    /// Upload the precomputed Per-Layer Embeddings table for the next
    /// decode / prefill call. `data` is `[positions × num_layers × ple_dim]`
    /// f32 in position-major order — for one-token decode `positions = 1`,
    /// for prefill `positions = seq_len`.
    ///
    /// Layout (offset for the (position, layer) row):
    /// `((position * num_layers) + layer) * ple_dim` f32 elements from
    /// the start of `data`. The decode loop computes the byte offset and
    /// passes it through to the per-layer PLE dispatch.
    ///
    /// Set once before generation begins (decode reuses the same `[1 × num_layers × ple_dim]`
    /// upload across positions if the inference layer is responsible for
    /// re-computing per-token).  Call [`clear_ple_inputs`](Self::clear_ple_inputs)
    /// when generation finishes.
    pub fn prepare_ple_inputs(&self, data: &[f32], num_layers: usize, ple_dim: usize) {
        debug_assert!(
            data.len() % (num_layers * ple_dim) == 0,
            "PLE input table size {} must be a multiple of num_layers * ple_dim ({} * {})",
            data.len(),
            num_layers,
            ple_dim,
        );
        let positions = data.len() / (num_layers * ple_dim);
        let buffer = self.bufs.transient_from_f32(data);
        *self.ple_inputs.lock().unwrap() = Some(PleInputBuffer {
            buffer,
            num_layers,
            ple_dim,
            positions,
        });
    }

    /// Drop the PLE input table.  No-op if none was set.
    pub fn clear_ple_inputs(&self) {
        *self.ple_inputs.lock().unwrap() = None;
    }

    /// Internal: snapshot the current PLE inputs (cloned `Buffer` handle —
    /// Metal `Buffer` is refcounted) so the per-layer decode loop can
    /// release the mutex while still holding a stable reference.
    pub(crate) fn ple_inputs_snapshot(&self) -> Option<PleInputBuffer> {
        self.ple_inputs.lock().unwrap().clone()
    }
}

/// Precomputed Per-Layer Embeddings input table held on the Metal
/// backend.  See [`MetalBackend::prepare_ple_inputs`].
#[derive(Clone)]
pub(crate) struct PleInputBuffer {
    pub buffer: metal::Buffer,
    pub num_layers: usize,
    pub ple_dim: usize,
    pub positions: usize,
}

impl PleInputBuffer {
    /// Byte offset into [`Self::buffer`] for the `[ple_dim]` row at
    /// `(position, layer)`. Position-major layout.
    pub fn row_offset_bytes(&self, position: usize, layer: usize) -> u64 {
        debug_assert!(position < self.positions);
        debug_assert!(layer < self.num_layers);
        ((position * self.num_layers + layer) * self.ple_dim * 4) as u64
    }
}
