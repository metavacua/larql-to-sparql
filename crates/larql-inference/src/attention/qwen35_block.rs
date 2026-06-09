//! Qwen3-Next full-attention block forward + hybrid layer router
//! (Phase C.4b of `inference-qwen35-deltanet`).
//!
//! The full-attention block here handles Qwen 3.6's every-4th layer
//! — the one with softmax attention rather than Gated DeltaNet. It
//! incorporates the three Qwen3-Next quirks documented in design.md
//! §6:
//!
//! 1. **Fused Q + per-head sigmoid gate** projection (the `attn_q`
//!    tensor's output dim is `2 * n_head * head_dim`). Split via
//!    `qwen35_attn::split_q_gate`.
//! 2. **Per-head Q/K RMSNorm** at `[head_dim]` shape (not `[hidden]`).
//!    Existing `residual::rms_norm_heads` already implements this.
//! 3. **Partial RoPE** on the first `mrope_total_rotary(sections)` head
//!    dims. For text-only inference this is equivalent to single-
//!    section RoPE; the existing `apply_rope_partial_at` does it.
//!
//! Plus the standard Gemma-2-style pre+post-norm sandwich the
//! `Qwen35Arch::has_post_norms() = true` predicate declares.
//!
//! The `DeltaNetHybridCache` bundles a `KvCache` (for the 16
//! full-attention layers) and a `DeltaNetStateCache` (for the 48
//! linear-attention layers) plus a per-layer kind mask, so a single
//! object carries all the per-sequence state the hybrid model needs.
//!
//! The `hybrid_layer_step` router demonstrates the dispatch shape:
//! per layer, look at `layer_kinds[layer]` and call the appropriate
//! block. The full glue that runs the embed → 64 layers → final
//! norm → lm_head pipeline lives in a follow-up (Phase C.4c).

use ndarray::{ArcArray2, Array1, Array2};
use std::sync::Arc;

use super::deltanet_block::{deltanet_block_step, DeltaNetDims, DeltaNetLayerWeights};
use super::deltanet_state::{sigmoid, DeltaNetLayerState, DeltaNetStateCache};
use super::qwen35_attn::{apply_q_gate, split_q_gate};

/// One full-attention layer's weight tensors.
///
/// Stored as `ArcArray2<f32>` / `Arc<[f32]>` so cloning the struct
/// (e.g. when building a Qwen35Weights view) only bumps Arc
/// refcounts — no data copies. Shape convention: every projection
/// has `[out_features, in_features]` (matvec is `y = W @ x`).
#[derive(Clone)]
pub struct Qwen35AttentionLayerWeights {
    /// Pre-attention RMSNorm weight `[hidden]`.
    pub attn_norm: Arc<[f32]>,
    /// Fused Q + per-head gate projection
    /// `[2 * n_head * head_dim, hidden]`.
    pub attn_q: ArcArray2<f32>,
    /// K projection `[n_head_kv * head_dim, hidden]`.
    pub attn_k: ArcArray2<f32>,
    /// V projection `[n_head_kv * head_dim, hidden]`.
    pub attn_v: ArcArray2<f32>,
    /// Per-head Q RMSNorm weight `[head_dim]`.
    pub attn_q_norm: Arc<[f32]>,
    /// Per-head K RMSNorm weight `[head_dim]`.
    pub attn_k_norm: Arc<[f32]>,
    /// Output projection `[hidden, n_head * head_dim]`.
    pub attn_output: ArcArray2<f32>,

    /// Optional lazy-quantised versions of the four full-attn
    /// projections. Same opt-in semantics as the DeltaNet quants:
    /// when `Some`, the dense field is a 0×0 placeholder and the
    /// matvec goes through `QuantTensor::matvec`.
    pub attn_q_quant: Option<larql_models::quant::lazy::QuantTensor>,
    pub attn_k_quant: Option<larql_models::quant::lazy::QuantTensor>,
    pub attn_v_quant: Option<larql_models::quant::lazy::QuantTensor>,
    pub attn_output_quant: Option<larql_models::quant::lazy::QuantTensor>,
}

/// Shape constants for the full-attention layers (uniform across
/// layers within a model).
#[derive(Clone, Copy, Debug)]
pub struct Qwen35AttentionDims {
    /// Residual stream width. Qwen 3.6 27B: 5120.
    pub hidden: usize,
    /// Number of Q heads. Qwen 3.6: 24.
    pub n_head: usize,
    /// Number of K/V heads (GQA, `n_head / n_head_kv` = repeat
    /// factor). Qwen 3.6: 4.
    pub n_head_kv: usize,
    /// Per-head dimension. Qwen 3.6: 256.
    pub head_dim: usize,
    /// Number of rotary head dimensions (= `sum(rope_sections)`).
    /// Qwen 3.6: 64 (so 192 of the 256 head dims pass through
    /// without rotation).
    pub rotary_dim: usize,
    /// RoPE base. Qwen 3.6: 10_000_000.0.
    pub rope_base: f64,
    /// RMSNorm epsilon.
    pub eps: f32,
}

impl Qwen35AttentionDims {
    /// Construct from an architecture handle. Reads the standard
    /// transformer dims from `arch.config()` plus the MRoPE section
    /// sum from `arch.rope_dimension_sections()` for `rotary_dim`.
    /// `eps` is provided by the caller (typically `1e-6`).
    ///
    /// Required by `qwen35_generate_with_sampling` (task 2c.a of
    /// `vindex-qwen35moe-reader`). Pure utility — no integration
    /// concerns beyond what the existing `ModelArchitecture` trait
    /// already exposes after PR #167's vindex loader fix.
    pub fn from_arch(arch: &dyn larql_models::ModelArchitecture, eps: f32) -> Self {
        let cfg = arch.config();
        // Multi-section RoPE (Qwen 3.6) sums the section sizes. Single-
        // dim partial RoPE (Qwen3-Next / Qwen3-Coder-Next) uses
        // `partial_rotary_factor * head_dim`. Fall back to full head_dim
        // for plain RoPE.
        // For Qwen3-Next / Qwen3.6 MoE both ship MRoPE with
        // `rope_dimension_sections`, but the section sum is in PAIR
        // units (number of (i, i+n_dims/2) pairs assigned to each
        // position channel). The total rotated dim count is
        // `rope.dimension_count` (== `partial_rotary_factor * head_dim`).
        // Prefer the explicit count: on 35B-A3B (sections [11,11,10,0]
        // sum=32, but rotated dim=64) the section-sum is half the
        // correct rotary_dim — using it as-is rotated only the first
        // 32 dims of each head and produced sentence-level garbage
        // ('Basic 86841ったった...' on greedy 'Hi'). Falls back to
        // sum(sections) only when no partial_rotary_factor is set.
        let rotary_dim = cfg
            .partial_rotary_factor
            .map(|f| ((cfg.head_dim as f64) * f).round() as usize)
            .filter(|d| *d > 0)
            .or_else(|| {
                arch.rope_dimension_sections()
                    .map(|secs| secs.iter().copied().sum())
            })
            .unwrap_or(cfg.head_dim);
        Self {
            hidden: cfg.hidden_size,
            n_head: cfg.num_q_heads,
            n_head_kv: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_dim,
            // `rope_base_for_layer` per the trait — Qwen 3.6 is
            // uniform across layers so layer 0 stands in.
            rope_base: arch.rope_base_for_layer(0),
            eps,
        }
    }

    /// `n_head * head_dim`.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// `n_head_kv * head_dim`.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// `2 * q_dim` — the dim of the fused Q+gate projection output.
    #[inline]
    pub fn fused_q_dim(&self) -> usize {
        2 * self.q_dim()
    }

    /// `q_dim / kv_dim` (GQA repeat factor). Qwen 3.6: 24/4 = 6.
    #[inline]
    pub fn gqa_reps(&self) -> usize {
        self.q_dim() / self.kv_dim()
    }

    /// Partial-rotary fraction (= `rotary_dim / head_dim`).
    #[inline]
    pub fn rope_fraction(&self) -> f64 {
        self.rotary_dim as f64 / self.head_dim as f64
    }
}

/// One-token forward through a Qwen 3.6 full-attention layer.
///
/// `kv_layer` is the cumulative `[seq_len, kv_dim]` K and V slabs
/// for this layer — the function appends the new token's K/V row to
/// them and runs GQA softmax attention over the full slab.
///
/// `position` is the new token's absolute 0-indexed position. RoPE
/// is applied to the new Q row and the new K row before append.
///
/// Returns the block's output `[hidden]`, ready to be added back
/// to the residual stream by the caller.
pub fn qwen35_attention_block_step(
    x: &Array1<f32>,
    weights: &Qwen35AttentionLayerWeights,
    dims: &Qwen35AttentionDims,
    kv_layer: &mut (Array2<f32>, Array2<f32>),
    position: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    layer: usize,
) -> Array1<f32> {
    debug_assert_eq!(x.len(), dims.hidden);
    debug_assert_eq!(weights.attn_norm.len(), dims.hidden);
    if weights.attn_q_quant.is_none() {
        debug_assert_eq!(weights.attn_q.shape(), [dims.fused_q_dim(), dims.hidden]);
    }
    if weights.attn_k_quant.is_none() {
        debug_assert_eq!(weights.attn_k.shape(), [dims.kv_dim(), dims.hidden]);
    }
    if weights.attn_v_quant.is_none() {
        debug_assert_eq!(weights.attn_v.shape(), [dims.kv_dim(), dims.hidden]);
    }
    debug_assert_eq!(weights.attn_q_norm.len(), dims.head_dim);
    debug_assert_eq!(weights.attn_k_norm.len(), dims.head_dim);
    if weights.attn_output_quant.is_none() {
        debug_assert_eq!(weights.attn_output.shape(), [dims.hidden, dims.q_dim()]);
    }

    // Layer-boundary dumps for elementwise bisection against
    // llama-eval-callback. Mirrors deltanet_block's pattern: writes
    // `<dir>/<token_tag>/<name>_l{layer:02}.bin` in the same
    // `[i64 ne[4]; f32 data]` format as `LLAMA_DUMP_BIN_DIR`.
    let dump_fa = |name: &str, data: &[f32]| {
        if let Ok(dir) = std::env::var("LARQL_QWEN35_DUMP_LAYER_BOUNDARY") {
            let token_tag =
                std::env::var("LARQL_QWEN35_DUMP_TOKEN_TAG").unwrap_or_else(|_| "tok".to_string());
            let layer_dir = format!("{dir}/{token_tag}");
            let _ = std::fs::create_dir_all(&layer_dir);
            let path = format!("{layer_dir}/{name}_l{layer:02}.bin");
            if let Ok(mut f) = std::fs::File::create(&path) {
                use std::io::Write;
                let ne: [i64; 4] = [data.len() as i64, 1, 1, 1];
                for n in ne {
                    let _ = f.write_all(&n.to_le_bytes());
                }
                for v in data {
                    let _ = f.write_all(&v.to_le_bytes());
                }
            }
        }
    };

    // 1. Pre-attention RMSNorm.
    let x_norm = super::deltanet_block::rms_norm_1d_pub(x, &weights.attn_norm, dims.eps);
    dump_fa("fa_x_norm", x_norm.as_slice().unwrap_or(&[]));

    // 2. Projections.
    use crate::attention::gpu_tier::{self, GpuClass};
    use crate::attention::quant_dispatch::matvec_with_backend;
    // Phase E.7: route full-attn q/k/v/o through the AttnProj class
    // so they can be independently pushed back to CPU via
    // `LARQL_QWEN35_GPU_NO_ATTN_PROJ=1`.
    let proj_backend = gpu_tier::backend_for(GpuClass::AttnProj, backend);
    let q_fused_1d = if let Some(q) = weights.attn_q_quant.as_ref() {
        matvec_with_backend(q, &x_norm, proj_backend)
    } else {
        weights.attn_q.dot(&x_norm)
    }; // [fused_q_dim]
    let k_1d = if let Some(q) = weights.attn_k_quant.as_ref() {
        matvec_with_backend(q, &x_norm, proj_backend)
    } else {
        weights.attn_k.dot(&x_norm)
    }; // [kv_dim]
    let v_1d = if let Some(q) = weights.attn_v_quant.as_ref() {
        matvec_with_backend(q, &x_norm, proj_backend)
    } else {
        weights.attn_v.dot(&x_norm)
    }; // [kv_dim]
    dump_fa("fa_qcur_full", q_fused_1d.as_slice().unwrap_or(&[]));
    dump_fa("fa_kcur", k_1d.as_slice().unwrap_or(&[]));
    dump_fa("fa_vcur", v_1d.as_slice().unwrap_or(&[]));

    // 3. Split Q+gate. split_q_gate takes [seq_len, fused_q_dim] →
    //    (q [seq_len, q_dim], gate [seq_len, q_dim]). Wrap our
    //    single token as a 1-row matrix.
    let q_fused_2d = q_fused_1d
        .into_shape_with_order((1, dims.fused_q_dim()))
        .expect("q_fused reshape");
    let (q_2d, gate_2d) = split_q_gate(&q_fused_2d, dims.n_head, dims.head_dim);
    dump_fa("fa_q_split", q_2d.as_slice().unwrap_or(&[]));
    dump_fa("fa_gate_sigmoid", gate_2d.as_slice().unwrap_or(&[]));

    // 4. Per-head RMSNorm for Q and K.
    let q_normed = crate::residual::rms_norm_heads(
        &q_2d,
        &weights.attn_q_norm,
        dims.n_head,
        dims.head_dim,
        0.0,
    );
    let k_2d_in = k_1d
        .into_shape_with_order((1, dims.kv_dim()))
        .expect("k reshape");
    let k_normed = crate::residual::rms_norm_heads(
        &k_2d_in,
        &weights.attn_k_norm,
        dims.n_head_kv,
        dims.head_dim,
        0.0,
    );
    dump_fa("fa_q_normed", q_normed.as_slice().unwrap_or(&[]));
    dump_fa("fa_k_normed", k_normed.as_slice().unwrap_or(&[]));

    // 5. Partial RoPE applied at `position`. RoPE is a 2D op so the
    //    1-row matrices work directly.
    let q_roped = super::rope::apply_rope_partial_at(
        &q_normed,
        dims.n_head,
        dims.head_dim,
        dims.rope_base,
        dims.rope_fraction(),
        position,
    );
    let k_roped = super::rope::apply_rope_partial_at(
        &k_normed,
        dims.n_head_kv,
        dims.head_dim,
        dims.rope_base,
        dims.rope_fraction(),
        position,
    );
    dump_fa("fa_q_roped", q_roped.as_slice().unwrap_or(&[]));
    dump_fa("fa_k_roped", k_roped.as_slice().unwrap_or(&[]));

    // 6. Append the new K/V row to the cumulative cache slabs.
    let v_2d_in = v_1d
        .into_shape_with_order((1, dims.kv_dim()))
        .expect("v reshape");
    append_row(&mut kv_layer.0, k_roped.row(0));
    append_row(&mut kv_layer.1, v_2d_in.row(0));

    // 7. GQA softmax attention — single-row Q, cumulative K/V.
    //    Routed through `qwen35_gqa_decode_step` when the backend
    //    implements it (CUDA: fused softmax + GEMV); falls back to the
    //    host-side single-row scan otherwise. The dispatch site is
    //    gated by `GpuClass::AttnDecode` so the env var
    //    `LARQL_QWEN35_GPU_NO_ATTN_DECODE=1` forces CPU even with a
    //    GPU backend attached — useful for bisecting which class
    //    actually moves tok/s vs VRAM.
    let q_row = q_roped.row(0).to_owned();
    let scale = (dims.head_dim as f64).powf(-0.5);
    let attn_decode_backend = gpu_tier::backend_for(GpuClass::AttnDecode, backend);
    let attn_out_1d = if let Some(b) = attn_decode_backend {
        let seq_len = kv_layer.0.shape()[0];
        let k_flat = kv_layer
            .0
            .as_slice()
            .expect("k_cache contiguous after append_row");
        let v_flat = kv_layer
            .1
            .as_slice()
            .expect("v_cache contiguous after append_row");
        // cache_id keys the backend's per-layer device-resident KV
        // slab. Layer index alone is sufficient: the chat route holds
        // an exclusive write guard on the qwen35 weights for the full
        // duration of a request (server.state::LoadedModel), so two
        // concurrent requests don't share device state. Per-sequence
        // resets are propagated via `DeltaNetHybridCache::reset` →
        // `qwen35_gqa_decode_reset`.
        // `LARQL_QWEN35_KV_MAX_SEQ` lets callers override the default
        // 4096-row device-cache cap without bumping
        // `DEFAULT_GPU_KV_CACHE_MAX_SEQ` (which the generic decode
        // path also keys off). Phase 3 of the long-context arc uses
        // this to bench at 16K / 32K / 128K context; production
        // users override per-request workload size.
        let max_seq = std::env::var("LARQL_QWEN35_KV_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(crate::layer_graph::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ);
        b.qwen35_gqa_decode_step(
            layer as u64,
            max_seq,
            q_row.as_slice().expect("q_row contiguous"),
            k_flat,
            v_flat,
            dims.n_head,
            dims.n_head_kv,
            dims.head_dim,
            seq_len,
        )
        .map(ndarray::Array1::from)
        .unwrap_or_else(|| {
            gqa_decode_step(
                q_row.view(),
                kv_layer.0.view(),
                kv_layer.1.view(),
                dims.n_head,
                dims.n_head_kv,
                dims.head_dim,
                scale,
            )
        })
    } else {
        gqa_decode_step(
            q_row.view(),
            kv_layer.0.view(),
            kv_layer.1.view(),
            dims.n_head,
            dims.n_head_kv,
            dims.head_dim,
            scale,
        )
    };
    dump_fa("fa_attn_pregate", attn_out_1d.as_slice().unwrap_or(&[]));
    // Wrap back into 1-row matrix to reuse the C.3 gate-apply helper.
    let mut attn_out = attn_out_1d
        .into_shape_with_order((1, dims.q_dim()))
        .expect("attn_out reshape");
    apply_q_gate(&mut attn_out, &gate_2d);
    dump_fa("fa_attn_gated", attn_out.as_slice().unwrap_or(&[]));

    // 9. Output projection: y = attn_output @ attn_out[0].
    let attn_out_1d = attn_out.row(0).to_owned();
    if let Some(q) = weights.attn_output_quant.as_ref() {
        matvec_with_backend(q, &attn_out_1d, proj_backend)
    } else {
        weights.attn_output.dot(&attn_out_1d)
    }
}

/// Phase 4b-integration — batched-prefill sibling of
/// [`qwen35_attention_block_step`].
///
/// Processes `seq_len = x.nrows()` token positions through one full-
/// attention block in a single call. Per-position helpers (RMSNorm,
/// Q/K/V projections, output projection) currently loop over rows
/// internally — the win in this PR is **the attention scan**, which
/// runs as one batched CUDA kernel call via
/// [`ComputeBackend::qwen35_attention_prefill_batch`] instead of
/// `seq_len` sequential per-token calls.
///
/// Future PRs in the batched-prefill arc replace the per-row matvec
/// loops with batched matmul (4c-projections), and the host append
/// loop with a single bulk write.
///
/// Not yet wired into `qwen35_forward_prefill` — the integration
/// needs the matching DeltaNet / MoE batched siblings (Phases
/// 4c/4d) so the entire per-layer flow can run multi-position. Until
/// then this function exists as a tested entry point that
/// `qwen35_forward_prefill` will swap in once those land.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_attention_block_prefill(
    x: &Array2<f32>,
    weights: &Qwen35AttentionLayerWeights,
    dims: &Qwen35AttentionDims,
    kv_layer: &mut (Array2<f32>, Array2<f32>),
    base_pos: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    layer: usize,
) -> Array2<f32> {
    use crate::attention::gpu_tier::{self, GpuClass};
    use crate::attention::quant_dispatch::matvec_with_backend;

    let seq_len = x.shape()[0];
    debug_assert_eq!(x.shape()[1], dims.hidden);
    debug_assert!(
        seq_len > 0,
        "qwen35_attention_block_prefill needs seq_len >= 1"
    );

    let proj_backend = gpu_tier::backend_for(GpuClass::AttnProj, backend);
    let attn_decode_backend = gpu_tier::backend_for(GpuClass::AttnDecode, backend);

    // 1. Pre-attention RMSNorm of each row — batched in-place to
    //    avoid per-row Array1 allocations (same pattern as the
    //    post-attention norm in `qwen35_forward_prefill`).
    let mut x_norm = Array2::<f32>::zeros((seq_len, dims.hidden));
    super::deltanet_block::rms_norm_2d_into(
        x.view(),
        &weights.attn_norm,
        dims.eps,
        x_norm.view_mut(),
    );

    // 2. Q/K/V projections — batched matmul instead of per-row
    //    matvec. Routes through `matmul_with_backend`: when a GPU
    //    backend is attached, Q4_K projections dispatch to
    //    `gemm_proj_seq` (cached f16 dequant + cuBLAS hgemm tensor
    //    cores); otherwise the CPU `QuantTensor::matmul` runs, which
    //    quantises all `seq_len` activations to Q8_K once and reads
    //    each weight row across each rayon worker's slice of
    //    activations. For seq_len >> num_cores (e.g. 32K prefill on
    //    24 cores → 1.3K rows per thread), each weight row stays hot
    //    in the worker's L2/L3 between activations — weight bandwidth
    //    drops from `seq_len × W_bytes` to roughly `cores × W_bytes`.
    use crate::attention::quant_dispatch::matmul_with_backend;
    let q_fused = if let Some(q) = weights.attn_q_quant.as_ref() {
        matmul_with_backend(q, &x_norm, proj_backend)
    } else {
        x_norm.dot(&weights.attn_q.t())
    };
    let k_full = if let Some(q) = weights.attn_k_quant.as_ref() {
        matmul_with_backend(q, &x_norm, proj_backend)
    } else {
        x_norm.dot(&weights.attn_k.t())
    };
    let v_full = if let Some(q) = weights.attn_v_quant.as_ref() {
        matmul_with_backend(q, &x_norm, proj_backend)
    } else {
        x_norm.dot(&weights.attn_v.t())
    };

    // 3. Split Q + gate (already batched-capable).
    let (q_2d, gate_2d) = split_q_gate(&q_fused, dims.n_head, dims.head_dim);

    // 4. Per-head RMSNorm for Q and K (batched-capable helper).
    let q_normed = crate::residual::rms_norm_heads(
        &q_2d,
        &weights.attn_q_norm,
        dims.n_head,
        dims.head_dim,
        0.0,
    );
    let k_normed = crate::residual::rms_norm_heads(
        &k_full,
        &weights.attn_k_norm,
        dims.n_head_kv,
        dims.head_dim,
        0.0,
    );

    // 5. Partial RoPE — per-row positions start at base_pos.
    let q_roped = super::rope::apply_rope_partial_at(
        &q_normed,
        dims.n_head,
        dims.head_dim,
        dims.rope_base,
        dims.rope_fraction(),
        base_pos,
    );
    let k_roped = super::rope::apply_rope_partial_at(
        &k_normed,
        dims.n_head_kv,
        dims.head_dim,
        dims.rope_base,
        dims.rope_fraction(),
        base_pos,
    );

    // 6. Append all `seq_len` new K/V rows to the host cache slabs
    //    in a single bulk allocation. The per-row `append_row`
    //    form is quadratic — each call reallocates and recopies
    //    the entire prior slab, so a 32K prefill spends ~268 GB
    //    of memory traffic on recopies alone. `append_rows` is
    //    O(seq_len * kv_dim) — for the same shape, ~16 MB.
    append_rows(&mut kv_layer.0, &k_roped);
    append_rows(&mut kv_layer.1, &v_full);

    // 7. Batched attention scan. Routes through
    //    `qwen35_attention_prefill_batch` when the backend offers it;
    //    falls back to the per-row `gqa_decode_step` loop otherwise.
    let scale = (dims.head_dim as f64).powf(-0.5);
    let mut attn_out_seq = Array2::<f32>::zeros((seq_len, dims.q_dim()));
    let mut handled_via_kernel = false;
    if let Some(b) = attn_decode_backend {
        let max_seq = std::env::var("LARQL_QWEN35_KV_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(crate::layer_graph::pipeline_layer::DEFAULT_GPU_KV_CACHE_MAX_SEQ);
        let q_flat = q_roped.as_slice().expect("q_roped contiguous");
        let k_flat = k_roped.as_slice().expect("k_roped contiguous");
        let v_flat = v_full.as_slice().expect("v_full contiguous");
        if let Some(out) = b.qwen35_attention_prefill_batch(
            layer as u64,
            max_seq,
            base_pos,
            q_flat,
            k_flat,
            v_flat,
            dims.n_head,
            dims.n_head_kv,
            dims.head_dim,
            seq_len,
        ) {
            let arr =
                ndarray::Array2::from_shape_vec((seq_len, dims.q_dim()), out).expect("attn shape");
            attn_out_seq.assign(&arr);
            handled_via_kernel = true;
        }
    }
    if !handled_via_kernel {
        // CPU / unsupported-backend fallback: per-row single-position
        // gqa_decode_step against the (now-populated) host slabs.
        // The slabs were appended row-by-row above; for row `r` the
        // cache spans positions [0, base_pos + r + 1).
        //
        // Rows are independent — parallelise across `r` with rayon.
        // The CPU fit at 2K+ tokens has the O(N²) attention term
        // already dominating wall time; even at smaller seq_len the
        // per-row cost is large enough that rayon overhead is
        // negligible. Views (zero-copy slices into the shared K/V
        // slab) replace the prior per-row `.to_owned()` — saves
        // O(seq_len × cache_size) transient allocations.
        use rayon::prelude::*;
        let out_rows: Vec<Array1<f32>> = (0..seq_len)
            .into_par_iter()
            .map(|r| {
                let cache_end = base_pos + r + 1;
                gqa_decode_step(
                    q_roped.row(r),
                    kv_layer.0.slice(ndarray::s![..cache_end, ..]),
                    kv_layer.1.slice(ndarray::s![..cache_end, ..]),
                    dims.n_head,
                    dims.n_head_kv,
                    dims.head_dim,
                    scale,
                )
            })
            .collect();
        for (r, out_row) in out_rows.into_iter().enumerate() {
            attn_out_seq.row_mut(r).assign(&out_row);
        }
    }

    // 8. Apply Q gate (batched-capable).
    apply_q_gate(&mut attn_out_seq, &gate_2d);

    // 9. Output projection — batched matmul (same shape rationale
    //    as step 2's projections; same L3-reuse / cuBLAS hgemm win).
    let out = if let Some(q) = weights.attn_output_quant.as_ref() {
        matmul_with_backend(q, &attn_out_seq, proj_backend)
    } else {
        attn_out_seq.dot(&weights.attn_output.t())
    };
    out
}

/// Autoregressive GQA softmax attention for a single new Q row over
/// cumulative K/V slabs.
///
/// - `q`: `[num_q * head_dim]` — the new token's Q (post-RMSNorm + RoPE).
/// - `k_cache`, `v_cache`: `[seq_len, num_kv * head_dim]` — cumulative
///   K/V (the new row already appended).
/// - `num_q`, `num_kv`, `head_dim`: layer shapes (GQA repeat factor
///   = `num_q / num_kv`).
/// - `scale`: typically `1 / sqrt(head_dim)`.
///
/// Returns `[num_q * head_dim]`.
///
/// Takes views so the prefill fallback caller (which slices the cumulative
/// cache per row) can pass zero-copy slices instead of paying for a
/// per-row `.to_owned()`. Existing decode-step callers pass `.view()`.
///
/// Per-head loop with stable softmax (subtract row max before exp).
/// The KV-head index for Q head `h` is `h / reps` (repeat-interleave).
fn gqa_decode_step(
    q: ndarray::ArrayView1<f32>,
    k_cache: ndarray::ArrayView2<f32>,
    v_cache: ndarray::ArrayView2<f32>,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
    scale: f64,
) -> Array1<f32> {
    let seq_len = k_cache.shape()[0];
    debug_assert!(seq_len > 0, "k_cache must have at least the new token");
    debug_assert_eq!(q.len(), num_q * head_dim);
    debug_assert_eq!(k_cache.shape()[1], num_kv * head_dim);
    debug_assert_eq!(v_cache.shape()[1], num_kv * head_dim);
    debug_assert!(
        num_q.is_multiple_of(num_kv) && num_kv > 0,
        "num_q ({num_q}) must be a multiple of num_kv ({num_kv})"
    );

    let reps = num_q / num_kv;
    let scale_f32 = scale as f32;
    let mut out = Array1::<f32>::zeros(num_q * head_dim);
    let mut scores = vec![0.0_f32; seq_len];

    for h in 0..num_q {
        let kv_h = h / reps;
        // 1. Scores[t] = (q_h · k_cache[t, kv_h]) * scale.
        for t in 0..seq_len {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[h * head_dim + d] * k_cache[[t, kv_h * head_dim + d]];
            }
            scores[t] = dot * scale_f32;
        }
        // 2. Stable softmax.
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0_f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            exp_sum += *s;
        }
        let inv_sum = 1.0 / exp_sum;
        // 3. Weighted sum with V.
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for t in 0..seq_len {
                acc += scores[t] * inv_sum * v_cache[[t, kv_h * head_dim + d]];
            }
            out[h * head_dim + d] = acc;
        }
    }

    out
}

/// Append a row to a `[N, dim]` matrix, growing it to `[N+1, dim]`.
fn append_row(slab: &mut Array2<f32>, new_row: ndarray::ArrayView1<f32>) {
    let dim = slab.shape()[1];
    debug_assert_eq!(new_row.len(), dim);
    let mut next = Array2::<f32>::zeros((slab.shape()[0] + 1, dim));
    for r in 0..slab.shape()[0] {
        for c in 0..dim {
            next[[r, c]] = slab[[r, c]];
        }
    }
    let last = next.shape()[0] - 1;
    for c in 0..dim {
        next[[last, c]] = new_row[c];
    }
    *slab = next;
}

/// Bulk-append `new_rows.nrows()` rows to `slab` in a single
/// reallocation + memcpy. Replaces a per-row `append_row` loop —
/// crucial for prefill where the per-row variant is quadratic
/// (each call reallocates and copies the *entire* prior slab,
/// so a prompt of length N does O(N²) memory traffic on the
/// existing rows alone).
///
/// For a 32K-token prefill with kv_dim=512 the per-row form does
/// ~268 GB of element copies just on the existing-row recopies;
/// this form does ~16 MB. The end-to-end parity test
/// `qwen35_forward_prefill_matches_per_token_loop` already
/// confirms KV slab contents are bit-identical after the swap.
pub(crate) fn append_rows(slab: &mut Array2<f32>, new_rows: &Array2<f32>) {
    let dim = slab.shape()[1];
    let n_new = new_rows.shape()[0];
    debug_assert_eq!(new_rows.shape()[1], dim);
    if n_new == 0 {
        return;
    }
    let old_rows = slab.shape()[0];
    let mut next = Array2::<f32>::zeros((old_rows + n_new, dim));
    if old_rows > 0 {
        next.slice_mut(ndarray::s![..old_rows, ..]).assign(slab);
    }
    next.slice_mut(ndarray::s![old_rows.., ..]).assign(new_rows);
    *slab = next;
}

/// Hybrid per-sequence state for a Qwen 3.6 / Qwen 3.6 MoE model:
/// `KvCache`-equivalent slabs for the full-attention layers and a
/// `DeltaNetStateCache` for the linear layers, paired with a
/// `layer_kinds` mask so the router knows which kind each layer is.
///
/// For Qwen 3.6 27B (`n_layer = 64`, `full_attention_interval = 4`):
/// 48 entries in `dn_state` (Some on linear layers); 16 layers
/// hold non-empty KV slabs.
pub struct DeltaNetHybridCache {
    /// Per-layer K/V slabs. `None` for linear-attention layers
    /// (they don't have softmax KV). Each `Some` slot starts as
    /// `(Array2::zeros((0, kv_dim)), Array2::zeros((0, kv_dim)))`
    /// and grows by one row per decoded token.
    pub kv_layers: Vec<Option<(Array2<f32>, Array2<f32>)>>,
    /// DeltaNet recurrent + conv state for linear layers. The mask
    /// pattern mirrors `kv_layers` inversely.
    pub dn_state: DeltaNetStateCache,
    /// `layer_kinds[layer] == true` iff this is a linear (DeltaNet)
    /// layer. Comes from `Qwen35Arch::is_linear_attention_layer`.
    pub layer_kinds: Vec<bool>,
    /// Absolute position of the NEXT token to be decoded. Used as
    /// the RoPE position for the new K/Q rows.
    pub next_position: usize,
}

impl DeltaNetHybridCache {
    /// Allocate the cache for a fresh sequence.
    ///
    /// - `layer_kinds`: `true` for linear-attention layers.
    /// - `kv_dim` (for full-attn layers): `n_head_kv * head_dim`.
    /// - `d_conv`, `dn_conv_dim`, `dn_head_v_dim`, `dn_n_v_heads`
    ///   (for linear layers).
    pub fn allocate(
        layer_kinds: &[bool],
        kv_dim: usize,
        d_conv: usize,
        dn_conv_dim: usize,
        dn_head_v_dim: usize,
        dn_n_v_heads: usize,
    ) -> Self {
        let kv_layers = layer_kinds
            .iter()
            .map(|&is_linear| {
                if is_linear {
                    None
                } else {
                    Some((Array2::zeros((0, kv_dim)), Array2::zeros((0, kv_dim))))
                }
            })
            .collect();
        let dn_state = DeltaNetStateCache::allocate(
            layer_kinds,
            d_conv,
            dn_conv_dim,
            dn_head_v_dim,
            dn_n_v_heads,
        );
        Self {
            kv_layers,
            dn_state,
            layer_kinds: layer_kinds.to_vec(),
            next_position: 0,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layer_kinds.len()
    }

    /// Reset all state to the empty initial condition. Use between
    /// sequences within the same process.
    pub fn reset(&mut self) {
        for slot in self.kv_layers.iter_mut().flatten() {
            slot.0 = Array2::zeros((0, slot.0.shape()[1]));
            slot.1 = Array2::zeros((0, slot.1.shape()[1]));
        }
        self.dn_state.reset();
        self.next_position = 0;
    }

    /// Drop the GPU backend's device-resident KV slabs for every layer
    /// in this cache. Mirrors [`Self::reset`] for the device side: the
    /// caller (e.g. chat route between requests) calls both so neither
    /// host nor device state leaks across sequences.
    ///
    /// Layer indices match the `cache_id` the backend was handed at
    /// each `qwen35_gqa_decode_step` call.
    pub fn reset_device_kv(&self, backend: Option<&dyn larql_compute::ComputeBackend>) {
        let Some(b) = backend else { return };
        for (layer, _) in self
            .kv_layers
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_some())
        {
            b.qwen35_gqa_decode_reset(layer as u64);
        }
    }
}

/// Weight bundle for one Qwen 3.6 layer, tagged with the layer
/// kind. The router consumes this to know which block path to run.
/// Cloning is cheap (Arc bumps only).
#[derive(Clone)]
pub enum Qwen35LayerWeights {
    /// Gated DeltaNet linear-attention layer (Qwen 3.6: 48 of 64).
    Linear(DeltaNetLayerWeights),
    /// Full softmax-attention layer (Qwen 3.6: 16 of 64).
    Attention(Qwen35AttentionLayerWeights),
}

/// Per-layer router: dispatches the layer kind to the right block
/// forward. Mutates the appropriate slot in `hybrid_cache`.
///
/// Returns the BLOCK output (NOT yet added to the residual; caller
/// adds it, applies the optional `attn_post_norm`, and runs FFN).
pub fn hybrid_layer_step(
    layer: usize,
    x: &Array1<f32>,
    weights: &Qwen35LayerWeights,
    dn_dims: &DeltaNetDims,
    attn_dims: &Qwen35AttentionDims,
    hybrid_cache: &mut DeltaNetHybridCache,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    sequence_pos: usize,
) -> Array1<f32> {
    debug_assert!(layer < hybrid_cache.num_layers());
    let is_linear = hybrid_cache.layer_kinds[layer];
    match (weights, is_linear) {
        (Qwen35LayerWeights::Linear(w), true) => {
            let state = hybrid_cache.dn_state.layers[layer]
                .as_mut()
                .expect("linear-layer state should be allocated");
            deltanet_block_step(x, w, dn_dims, state, backend, sequence_pos, layer)
        }
        (Qwen35LayerWeights::Attention(w), false) => {
            let kv_layer = hybrid_cache.kv_layers[layer]
                .as_mut()
                .expect("full-attn KV slabs should be allocated");
            qwen35_attention_block_step(
                x,
                w,
                attn_dims,
                kv_layer,
                hybrid_cache.next_position,
                backend,
                layer,
            )
        }
        _ => panic!(
            "layer {} kind mismatch: weights and layer_kinds disagree",
            layer
        ),
    }
}

/// Multi-position sibling of `hybrid_layer_step`. Dispatches to
/// `qwen35_attention_block_prefill` or `deltanet_block_prefill`
/// based on `hybrid_cache.layer_kinds[layer]`.
///
/// Phase 4-final wiring entry point: `qwen35_forward_prefill`
/// calls this once per layer instead of looping over per-token
/// `hybrid_layer_step` calls.
#[allow(clippy::too_many_arguments)]
pub fn hybrid_layer_prefill(
    layer: usize,
    x_seq: &Array2<f32>,
    weights: &Qwen35LayerWeights,
    dn_dims: &DeltaNetDims,
    attn_dims: &Qwen35AttentionDims,
    hybrid_cache: &mut crate::attention::deltanet_state::DeltaNetStateCache,
    kv_layers: &mut [Option<(Array2<f32>, Array2<f32>)>],
    base_pos: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    layer_kinds: &[bool],
) -> Array2<f32> {
    debug_assert!(layer < layer_kinds.len());
    let is_linear = layer_kinds[layer];
    match (weights, is_linear) {
        (Qwen35LayerWeights::Linear(w), true) => {
            let state = hybrid_cache.layers[layer]
                .as_mut()
                .expect("linear-layer state should be allocated");
            super::deltanet_block::deltanet_block_prefill(
                x_seq, w, dn_dims, state, backend, base_pos, layer,
            )
        }
        (Qwen35LayerWeights::Attention(w), false) => {
            let kv_layer = kv_layers[layer]
                .as_mut()
                .expect("full-attn KV slabs should be allocated");
            qwen35_attention_block_prefill(x_seq, w, attn_dims, kv_layer, base_pos, backend, layer)
        }
        _ => panic!(
            "layer {} kind mismatch: weights and layer_kinds disagree",
            layer
        ),
    }
}

#[inline]
fn _silu(x: f32) -> f32 {
    // Kept for parity with deltanet_block::silu; not currently
    // referenced (full-attn block uses the sigmoid'd gate from
    // split_q_gate directly). Reserved for FFN integration in
    // Phase C.4c.
    x * sigmoid(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::deltanet_state::{DeltaNetLayerState, DeltaNetStateCache};
    use ndarray::Array2;
    use std::sync::Arc as StdArc;

    #[allow(clippy::too_many_arguments)]
    fn make_dn_w(
        attn_norm: Vec<f32>,
        attn_qkv: Array2<f32>,
        attn_gate: Array2<f32>,
        ssm_conv1d: Array2<f32>,
        ssm_dt: Vec<f32>,
        ssm_a: Vec<f32>,
        ssm_beta: Array2<f32>,
        ssm_alpha: Array2<f32>,
        ssm_norm: Vec<f32>,
        ssm_out: Array2<f32>,
    ) -> DeltaNetLayerWeights {
        DeltaNetLayerWeights {
            attn_norm: StdArc::from(attn_norm.as_slice()),
            attn_qkv: attn_qkv.into_shared(),
            attn_gate: attn_gate.into_shared(),
            ssm_conv1d: ssm_conv1d.into_shared(),
            ssm_dt: StdArc::from(ssm_dt.as_slice()),
            ssm_a: StdArc::from(ssm_a.as_slice()),
            ssm_beta: ssm_beta.into_shared(),
            ssm_alpha: ssm_alpha.into_shared(),
            ssm_norm: StdArc::from(ssm_norm.as_slice()),
            ssm_out: ssm_out.into_shared(),
            attn_qkv_quant: None,
            attn_gate_quant: None,
            ssm_out_quant: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn make_attn_w(
        attn_norm: Vec<f32>,
        attn_q: Array2<f32>,
        attn_k: Array2<f32>,
        attn_v: Array2<f32>,
        attn_q_norm: Vec<f32>,
        attn_k_norm: Vec<f32>,
        attn_output: Array2<f32>,
    ) -> Qwen35AttentionLayerWeights {
        Qwen35AttentionLayerWeights {
            attn_norm: StdArc::from(attn_norm.as_slice()),
            attn_q: attn_q.into_shared(),
            attn_k: attn_k.into_shared(),
            attn_v: attn_v.into_shared(),
            attn_q_norm: StdArc::from(attn_q_norm.as_slice()),
            attn_k_norm: StdArc::from(attn_k_norm.as_slice()),
            attn_output: attn_output.into_shared(),
            attn_q_quant: None,
            attn_k_quant: None,
            attn_v_quant: None,
            attn_output_quant: None,
        }
    }

    fn tiny_attn_dims() -> Qwen35AttentionDims {
        Qwen35AttentionDims {
            hidden: 4,
            n_head: 2,
            n_head_kv: 1, // GQA reps = 2
            head_dim: 2,
            rotary_dim: 2, // full rotation on the tiny head
            rope_base: 10_000.0,
            eps: 1e-6,
        }
    }

    fn tiny_dn_dims() -> DeltaNetDims {
        DeltaNetDims {
            hidden: 4,
            head_v_dim: 2,
            n_v_heads: 1,
            n_k_heads: 1,
            d_conv: 2,
            eps: 1e-6,
            block_gqa: true,
        }
    }

    #[test]
    fn attention_dims_helpers() {
        let d = tiny_attn_dims();
        assert_eq!(d.q_dim(), 4);
        assert_eq!(d.kv_dim(), 2);
        assert_eq!(d.fused_q_dim(), 8);
        assert_eq!(d.gqa_reps(), 2);
        assert!((d.rope_fraction() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn attention_dims_qwen35_27b_values() {
        let d = Qwen35AttentionDims {
            hidden: 5120,
            n_head: 24,
            n_head_kv: 4,
            head_dim: 256,
            rotary_dim: 64,
            rope_base: 10_000_000.0,
            eps: 1e-6,
        };
        assert_eq!(d.q_dim(), 24 * 256); // 6144
        assert_eq!(d.kv_dim(), 4 * 256); // 1024
        assert_eq!(d.fused_q_dim(), 2 * 6144); // 12288
        assert_eq!(d.gqa_reps(), 6); // 24 / 4
        assert!((d.rope_fraction() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn append_row_grows_slab() {
        let mut slab = Array2::<f32>::zeros((2, 3));
        slab[[0, 0]] = 1.0;
        slab[[1, 1]] = 2.0;
        let new_row = ndarray::arr1(&[10.0_f32, 20.0, 30.0]);
        super::append_row(&mut slab, new_row.view());
        assert_eq!(slab.shape(), &[3, 3]);
        // Old rows preserved.
        assert_eq!(slab[[0, 0]], 1.0);
        assert_eq!(slab[[1, 1]], 2.0);
        // New row at the end.
        assert_eq!(slab.row(2).to_vec(), vec![10.0, 20.0, 30.0]);
    }

    /// Lock in PR #204's `rotary_dim` priority: `partial_rotary_factor`
    /// wins over `sum(rope_dimension_sections)` when both are set with
    /// inconsistent values. The Qwen 3.6 35B-A3B GGUF ships
    /// `rope.dimension_sections = [11, 11, 10, 0]` (sum 32, in PAIR units
    /// per llama's mrope_cache_init) together with
    /// `rope.dimension_count = 64` (the total rotated dim count).
    /// The previously-shipped code used `sum(sections)` and rotated
    /// only the first 32 dims per head → garbage output. Fixed by
    /// preferring `partial_rotary_factor * head_dim`.
    #[test]
    fn rotary_dim_prefers_partial_rotary_factor_over_sections() {
        use larql_models::architectures::qwen35::Qwen35MoeArch;
        use larql_models::config::ModelConfig;

        // Mock the 35B-A3B config:
        // - head_dim = 256
        // - partial_rotary_factor = 0.25 → 64 dims rotated
        // - rope_dimension_sections sum = 32 (the "wrong" answer)
        let cfg = ModelConfig {
            model_type: "qwen35moe".into(),
            num_layers: 4,
            hidden_size: 2048,
            intermediate_size: 0,
            head_dim: 256,
            num_q_heads: 16,
            num_kv_heads: 2,
            vocab_size: Some(248044),
            rope_base: 10_000_000.0,
            rope_local_base: None,
            sliding_window: None,
            num_experts: Some(256),
            num_experts_per_token: Some(8),
            num_shared_experts: Some(1),
            kv_lora_rank: None,
            q_lora_rank: None,
            rope_scaling: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            query_pre_attn_scalar: None,
            embedding_multiplier: None,
            residual_multiplier: None,
            attention_multiplier: None,
            logits_scaling: None,
            global_head_dim: None,
            num_global_kv_heads: None,
            partial_rotary_factor: Some(0.25),
            sliding_window_pattern: None,
            layer_types: None,
            attention_k_eq_v: false,
            per_layer_embed_dim: None,
            num_kv_shared_layers: None,
            enable_moe_block: true,
            top_k_experts: Some(8),
            moe_intermediate_size: Some(512),
            full_attention_interval: Some(4),
            ssm_state_size: Some(128),
            ssm_inner_size: Some(4096),
            ssm_dt_rank: Some(32),
            ssm_group_count: Some(16),
            ssm_conv_kernel: Some(4),
            rope_dimension_sections: Some(vec![11, 11, 10, 0]),
        };
        let arch = Qwen35MoeArch::from_config(cfg);
        let dims = Qwen35AttentionDims::from_arch(&arch, 1e-6);
        assert_eq!(
            dims.rotary_dim, 64,
            "partial_rotary_factor=0.25 * head_dim=256 should give rotary_dim=64; \
             pre-fix code would have given sum(sections)=32"
        );
    }

    #[test]
    fn append_row_from_empty_slab() {
        let mut slab = Array2::<f32>::zeros((0, 3));
        let new_row = ndarray::arr1(&[1.0_f32, 2.0, 3.0]);
        super::append_row(&mut slab, new_row.view());
        assert_eq!(slab.shape(), &[1, 3]);
        assert_eq!(slab.row(0).to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn hybrid_cache_allocates_correct_per_layer_kind() {
        // Mimics Qwen 3.6 27B's 4-layer slice: linear, linear, linear, attn.
        let mask = vec![true, true, true, false];
        let cache = DeltaNetHybridCache::allocate(&mask, 4 /*kv_dim*/, 2, 4, 2, 1);
        assert_eq!(cache.num_layers(), 4);
        // Linear layers: KV slot = None, DN state slot = Some.
        for &i in &[0, 1, 2] {
            assert!(cache.kv_layers[i].is_none());
            assert!(cache.dn_state.layers[i].is_some());
        }
        // Full-attn layer: KV slot = Some (empty slab), DN state = None.
        assert!(cache.kv_layers[3].is_some());
        let (k0, v0) = cache.kv_layers[3].as_ref().unwrap();
        assert_eq!(k0.shape(), &[0, 4]);
        assert_eq!(v0.shape(), &[0, 4]);
        assert!(cache.dn_state.layers[3].is_none());
        assert_eq!(cache.next_position, 0);
    }

    #[test]
    fn hybrid_cache_reset_clears_state_and_position() {
        let mask = vec![true, false];
        let mut cache = DeltaNetHybridCache::allocate(&mask, 4, 2, 4, 2, 1);
        // Dirty the linear-layer state.
        if let Some(state) = cache.dn_state.layers[0].as_mut() {
            state.conv_state.fill(7.0);
            state.recurrent_state.fill(11.0);
        }
        // Grow the attn-layer KV slabs.
        if let Some((k, v)) = cache.kv_layers[1].as_mut() {
            let nr = ndarray::arr1(&[1.0_f32, 2.0, 3.0, 4.0]);
            super::append_row(k, nr.view());
            super::append_row(v, nr.view());
        }
        cache.next_position = 5;
        cache.reset();
        // Reset semantics: DN state zero, KV slabs back to [0, kv_dim], pos = 0.
        assert!(cache.dn_state.layers[0]
            .as_ref()
            .unwrap()
            .conv_state
            .iter()
            .all(|&v| v == 0.0));
        let (k, v) = cache.kv_layers[1].as_ref().unwrap();
        assert_eq!(k.shape(), &[0, 4]);
        assert_eq!(v.shape(), &[0, 4]);
        assert_eq!(cache.next_position, 0);
    }

    #[test]
    fn hybrid_layer_step_routes_to_deltanet_on_linear() {
        // Build tiny weights for both branches; layer 0 is linear,
        // so the router SHALL call deltanet_block_step.
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let mask = vec![true, false];
        let mut cache = DeltaNetHybridCache::allocate(
            &mask,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        // DeltaNet weights (constant-filled for shape correctness).
        let dn_weights = make_dn_w(
            vec![1.0_f32; dn_dims.hidden],
            Array2::from_elem((dn_dims.conv_dim(), dn_dims.hidden), 0.1_f32),
            Array2::from_elem((dn_dims.value_dim(), dn_dims.hidden), 0.1_f32),
            Array2::from_elem((dn_dims.d_conv, dn_dims.conv_dim()), 0.5_f32),
            vec![0.0_f32; dn_dims.n_v_heads],
            vec![-1.0_f32; dn_dims.n_v_heads],
            Array2::from_elem((dn_dims.n_v_heads, dn_dims.hidden), 0.1_f32),
            Array2::from_elem((dn_dims.n_v_heads, dn_dims.hidden), 0.1_f32),
            vec![1.0_f32; dn_dims.head_v_dim],
            Array2::from_elem((dn_dims.hidden, dn_dims.value_dim()), 0.5_f32),
        );
        let layer_weights = Qwen35LayerWeights::Linear(dn_weights);

        let x = Array1::from_elem(dn_dims.hidden, 1.0_f32);
        let y = hybrid_layer_step(
            0,
            &x,
            &layer_weights,
            &dn_dims,
            &attn_dims,
            &mut cache,
            None,
            0,
        );
        assert_eq!(y.len(), dn_dims.hidden);
        // Linear-layer state should be non-zero after one step.
        let state = cache.dn_state.layers[0].as_ref().unwrap();
        assert!(state.conv_state.iter().any(|&v| v.abs() > 0.0));
        // Full-attn KV slab still empty.
        let (k1, _v1) = cache.kv_layers[1].as_ref().unwrap();
        assert_eq!(k1.shape()[0], 0);
    }

    /// `append_rows` parity test: bulk append produces the exact
    /// same slab as a per-row `append_row` loop. Locks the
    /// optimization so future refactors can't silently break the
    /// KV-cache layout the attention kernels expect.
    #[test]
    fn append_rows_matches_per_row_loop() {
        let dim = 7usize;
        let mut slab_loop = Array2::<f32>::zeros((0, dim));
        let mut slab_bulk = Array2::<f32>::zeros((0, dim));

        // Seed: 2 rows via the per-row form to both, then exercise
        // both forms appending 5 more identical rows so we cover
        // the "non-empty initial slab" case (not just cold-start).
        let r0: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let r1: Vec<f32> = (0..dim).map(|i| (i + dim) as f32).collect();
        append_row(&mut slab_loop, ndarray::ArrayView1::from(&r0));
        append_row(&mut slab_loop, ndarray::ArrayView1::from(&r1));
        append_row(&mut slab_bulk, ndarray::ArrayView1::from(&r0));
        append_row(&mut slab_bulk, ndarray::ArrayView1::from(&r1));

        let n_new = 5usize;
        let mut new_data = Vec::with_capacity(n_new * dim);
        for r in 0..n_new {
            for c in 0..dim {
                new_data.push((100 + r * dim + c) as f32);
            }
        }
        let new_rows = Array2::from_shape_vec((n_new, dim), new_data).unwrap();

        for r in 0..n_new {
            append_row(&mut slab_loop, new_rows.row(r));
        }
        append_rows(&mut slab_bulk, &new_rows);

        assert_eq!(slab_loop.shape(), slab_bulk.shape());
        for r in 0..slab_loop.shape()[0] {
            for c in 0..dim {
                assert_eq!(slab_loop[[r, c]], slab_bulk[[r, c]], "row {r} col {c}");
            }
        }
    }

    /// Empty `new_rows` is a no-op: slab unchanged.
    #[test]
    fn append_rows_empty_is_noop() {
        let dim = 4usize;
        let initial =
            Array2::from_shape_vec((2, dim), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let mut slab = initial.clone();
        let empty = Array2::<f32>::zeros((0, dim));
        append_rows(&mut slab, &empty);
        assert_eq!(slab, initial);
    }

    #[test]
    #[should_panic(expected = "kind mismatch")]
    fn hybrid_layer_step_panics_on_kind_mismatch() {
        // layer 0 mask says linear, but we pass Attention weights.
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let mask = vec![true]; // 1 layer, linear
        let mut cache = DeltaNetHybridCache::allocate(
            &mask,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let attn_weights = make_attn_w(
            vec![1.0_f32; attn_dims.hidden],
            Array2::zeros((attn_dims.fused_q_dim(), attn_dims.hidden)),
            Array2::zeros((attn_dims.kv_dim(), attn_dims.hidden)),
            Array2::zeros((attn_dims.kv_dim(), attn_dims.hidden)),
            vec![1.0_f32; attn_dims.head_dim],
            vec![1.0_f32; attn_dims.head_dim],
            Array2::zeros((attn_dims.hidden, attn_dims.q_dim())),
        );
        let layer_weights = Qwen35LayerWeights::Attention(attn_weights);

        let x = Array1::from_elem(attn_dims.hidden, 1.0_f32);
        let _ = hybrid_layer_step(
            0,
            &x,
            &layer_weights,
            &dn_dims,
            &attn_dims,
            &mut cache,
            None,
            0,
        );
    }

    /// Avoids the dead-code-warning silence on the reserved helper.
    #[test]
    fn silu_helper_is_reachable() {
        let _ = super::_silu(0.0);
        let _ = DeltaNetLayerState::allocate(2, 4, 2, 1);
    }

    /// Phase 4b-integration parity gate: the batched-prefill
    /// `qwen35_attention_block_prefill` must produce bit-similar
    /// outputs to running `qwen35_attention_block_step` once per
    /// position over the same input. CPU backend is `None` so the
    /// kernel-routed fast path is bypassed — this test exercises
    /// the host-side per-row fallback (which the `handled_via_kernel`
    /// branch falls through to) end-to-end.
    #[test]
    fn attention_block_prefill_matches_per_position_loop() {
        use ndarray::Array1;
        let attn_dims = tiny_attn_dims();
        let hidden = attn_dims.hidden;
        let kv_dim = attn_dims.kv_dim();
        let fused_q = attn_dims.fused_q_dim();
        let q_dim = attn_dims.q_dim();

        // Synthetic weights — deterministic, non-zero, not too large.
        let attn_norm: Vec<f32> = (0..hidden).map(|i| 0.5 + 0.1 * i as f32).collect();
        let attn_q_norm: Vec<f32> = (0..attn_dims.head_dim)
            .map(|i| 0.7 + 0.05 * i as f32)
            .collect();
        let attn_k_norm: Vec<f32> = (0..attn_dims.head_dim)
            .map(|i| 0.9 - 0.05 * i as f32)
            .collect();
        let attn_q = Array2::from_shape_fn((fused_q, hidden), |(r, c)| {
            0.01 * (r as f32) + 0.02 * (c as f32) - 0.1
        });
        let attn_k = Array2::from_shape_fn((kv_dim, hidden), |(r, c)| {
            0.03 * (r as f32) - 0.01 * (c as f32) + 0.05
        });
        let attn_v = Array2::from_shape_fn((kv_dim, hidden), |(r, c)| {
            -0.02 * (r as f32) + 0.04 * (c as f32) + 0.02
        });
        let attn_output = Array2::from_shape_fn((hidden, q_dim), |(r, c)| {
            0.015 * (r as f32) + 0.025 * (c as f32) - 0.07
        });

        // Build TWO copies of the weights — `into_shared` consumes
        // the Array2 so we make two so each path gets its own.
        let weights_seq = make_attn_w(
            attn_norm.clone(),
            attn_q.clone(),
            attn_k.clone(),
            attn_v.clone(),
            attn_q_norm.clone(),
            attn_k_norm.clone(),
            attn_output.clone(),
        );
        let weights_batch = make_attn_w(
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_q_norm,
            attn_k_norm,
            attn_output,
        );

        let seq_len = 3_usize;
        let x_seq = Array2::from_shape_fn((seq_len, hidden), |(r, c)| {
            0.1 + 0.05 * (r as f32) + 0.07 * (c as f32)
        });

        // Reference: per-position loop, base_pos=0.
        let mut kv_seq = (
            Array2::<f32>::zeros((0, kv_dim)),
            Array2::<f32>::zeros((0, kv_dim)),
        );
        let mut ref_outs: Vec<Array1<f32>> = Vec::with_capacity(seq_len);
        for r in 0..seq_len {
            let xr = x_seq.row(r).to_owned();
            let y =
                qwen35_attention_block_step(&xr, &weights_seq, &attn_dims, &mut kv_seq, r, None, 0);
            ref_outs.push(y);
        }

        // Subject under test: batched call with seq_len=3, base_pos=0.
        let mut kv_batch = (
            Array2::<f32>::zeros((0, kv_dim)),
            Array2::<f32>::zeros((0, kv_dim)),
        );
        let batch_out = qwen35_attention_block_prefill(
            &x_seq,
            &weights_batch,
            &attn_dims,
            &mut kv_batch,
            0,
            None,
            0,
        );

        assert_eq!(batch_out.shape(), &[seq_len, hidden]);
        for r in 0..seq_len {
            for c in 0..hidden {
                let a = batch_out[[r, c]];
                let b = ref_outs[r][c];
                let diff = (a - b).abs();
                let tol = 1e-4 + 1e-4 * a.abs().max(b.abs());
                assert!(
                    diff < tol,
                    "mismatch at row {r} col {c}: batched={a:.6} ref={b:.6} diff={diff:.2e}"
                );
            }
        }
        // KV cache slabs should also be bit-identical after both runs.
        assert_eq!(kv_seq.0.shape(), kv_batch.0.shape());
        for r in 0..kv_seq.0.shape()[0] {
            for c in 0..kv_seq.0.shape()[1] {
                let diff_k = (kv_seq.0[[r, c]] - kv_batch.0[[r, c]]).abs();
                let diff_v = (kv_seq.1[[r, c]] - kv_batch.1[[r, c]]).abs();
                assert!(diff_k < 1e-5, "K mismatch at [{r},{c}]");
                assert!(diff_v < 1e-5, "V mismatch at [{r},{c}]");
            }
        }
    }
}
