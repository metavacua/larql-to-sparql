//! Per-layer attention block — Steps 1.5 through 5 of the decode loop.
//!
//! Inputs (already populated by `encode_input_norm_and_qkv`):
//! - `q_out`, `k_out`, `v_out`: raw Q/K/V projections (pre-norm, pre-RoPE).
//! - `h_buf`: layer-input residual.
//!
//! Outputs:
//! - `ffn_norm_out`: RMS-normed `h_buf + o_out` (FFN gate/up input).
//! - `h_post_attn`: raw `h_buf + o_out` (post-FFN residual base).
//! - `kv_cache.layers[l].current_len += 1` (the new token's K/V row is appended).
//!
//! Path selection (env-gated, defaults preserve the proven-win May-2026 fusion wave):
//! - `LARQL_FUSED_ATTN=1` (opt-in) — single `attn_fused` kernel covers QK-norm +
//!   RoPE + KV append + attend. Currently regresses on Gemma 3 4B (parallelism
//!   collapse 12 TGs → 8); kept registered for the multi-TG-per-head retry.
//! - `LARQL_FUSED_QK_NORM_ROPE=0` — opt out of the fused QK-norm + RoPE path.
//! - `LARQL_FUSED_KV_APPEND_ATTEND=0` — opt out of the fused KV append + attend.
//! - `LARQL_FUSED_POST_ATTN_NORM=0` — opt out of the triple-fused
//!   `post_attn_norm + residual + ffn_norm + store`.
//!
//! No behaviour change vs. the prior inline code; pure code motion to make the
//! per-stage profiler boundary tractable (next step) and shrink the decode
//! loop body.

use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

use super::ops;
use crate::ops::kv_cache::{MAX_HEAD_DIM_DOUBLE_SG, MAX_HEAD_DIM_SINGLE_SG};
use crate::MetalBackend;
use larql_compute::FullPipelineLayer;
use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;

/// First of the two consecutive `attn_fused` slots carrying attention
/// sinks; the `has_sinks` flag follows in slot 19. See `stages::sinks`.
const ATTN_FUSED_SINKS_INDEX: u64 = 18;

/// `inv_freq` slot on `attn_fused` — took over the old `rope_base` index in
/// place, so no other binding moved.
const ATTN_FUSED_INV_FREQ_INDEX: u64 = 16;

/// `amplitude` slot on `attn_fused`, appended after the sinks pair.
const ATTN_FUSED_AMPLITUDE_INDEX: u64 = 20;

/// `softcap` slot on `attn_fused` (0.0 = disabled).
const ATTN_FUSED_SOFTCAP_INDEX: u64 = 22;

/// `has_qk_norm` slot on `attn_fused` — 0 lets no-QK-norm archs
/// (GPT-OSS) take the single-dispatch path with Q/K passed through raw.
const ATTN_FUSED_HAS_QK_NORM_INDEX: u64 = 23;

/// First of the four consecutive `attn_fused` slots carrying the Q/K/V
/// projection biases: q_bias(24), k_bias(25), v_bias(26), then the
/// `has_qkv_bias` flag (27). Sinks convention — stub buffers bound when
/// absent, flag gates the read.
const ATTN_FUSED_QKV_BIAS_INDEX: u64 = 24;

/// `softcap` slot on `kv_append_attend_fused` (0.0 = disabled).
const KV_APPEND_ATTEND_SOFTCAP_INDEX: u64 = 14;

/// Absolute stream position for RoPE on `attn_fused`. The kernel's cache
/// row stays occupancy-indexed; only the rotation angle uses this. The
/// two diverge once a sliding window compacts (audit F11).
const ATTN_FUSED_ABS_POS_INDEX: u64 = 21;

/// First of the two consecutive `kv_append_attend_fused` slots carrying
/// attention sinks; the `has_sinks` flag follows in slot 13.
const KV_APPEND_ATTEND_SINKS_INDEX: u64 = 12;

pub(super) struct AttnBufs<'a> {
    /// Layer-input residual (read).
    pub h_buf: &'a Buffer,
    pub q_out: &'a Buffer,
    pub k_out: &'a Buffer,
    pub v_out: &'a Buffer,
    pub attn_out_buf: &'a Buffer,
    pub o_out_buf: &'a Buffer,
    /// FFN gate/up input (written).
    pub ffn_norm_out: &'a Buffer,
    /// Post-FFN residual base (written).
    pub h_post_attn: &'a Buffer,
    /// Scratch for Q8 quantize on the legacy O-proj path.
    pub o_q8_scratch: &'a Buffer,
    pub o_q8s_scratch: &'a Buffer,
    /// Scratch for the Q8-input residual+norm path.
    pub ffn_q8: &'a Buffer,
    pub ffn_q8s: &'a Buffer,
    /// Scratch for the unfused post-attn norm chain.
    pub normed_scratch: &'a Buffer,
    pub wo: &'a Buffer,
    pub wo_scales: Option<&'a Buffer>,
    pub post_attn_norm: &'a Buffer,
}

pub(super) struct AttnDims {
    pub hidden: usize,
    pub layer_q_dim: usize,
    /// True iff the FFN side will run Q4_K family (selects the fused
    /// `residual_norm_store` path that mirrors the FFN's input dtype).
    pub ffn_uses_kquant: bool,
}

impl MetalBackend {
    /// Whether this layer's decode step will take the fully-fused
    /// `attn_fused` path (QK-norm/passthrough + RoPE + KV-append +
    /// attend + optional QKV biases, ONE dispatch).
    ///
    /// The SINGLE authority for that decision: `encode_input_norm_and_qkv`
    /// consults it to skip the separate Q/K/V `bias_add` dispatches (the
    /// fused kernel applies them on load), and `encode_attention_block`
    /// consults it to pick the dispatch. Two sites re-deriving this
    /// independently is the dispatch-geometry-disagreement defect class —
    /// a bias applied twice or not at all, silently.
    pub(super) fn attn_fused_will_fire(
        &self,
        layer: &FullPipelineLayer,
        kv_cache: &ops::kv_cache::KVCache,
        layer_idx: usize,
    ) -> bool {
        let attn_spec = layer.attention_spec();
        if layer.kv_shared_source.is_some()
            || !self.decode_flags.fused_attn
            || attn_spec.head_dim > MAX_HEAD_DIM_SINGLE_SG
            || attn_spec.has_v_norm
        {
            return false;
        }
        // QK-norm must be both-or-neither: the kernel norms Q and K under
        // one flag. A half-normed arch takes the unfused chain.
        if attn_spec.q_norm_enabled != attn_spec.k_norm_enabled {
            return false;
        }
        // Same all-or-nothing rule for the projection biases: the kernel
        // reads all three under one flag, and a partially-biased layer
        // bound with stub buffers would read out of bounds. The unfused
        // chain's per-site bias_add dispatches handle mixed presence.
        let n_bias = [layer.attn_q_bias, layer.attn_k_bias, layer.attn_v_bias]
            .iter()
            .filter(|b| b.is_some())
            .count();
        if n_bias != 0 && n_bias != 3 {
            return false;
        }
        let window_size = self.effective_window_for(attn_spec.sliding_window as u32);
        let t_val = (kv_cache.layers[layer_idx].current_len + 1) as u32;
        ops::kv_cache::attention_span(t_val, window_size) <= ops::kv_cache::SHORT_ATTENTION_SPAN
    }

    /// Encode the per-layer attention block (Steps 1.5–5). See the module
    /// doc-comment for the full input/output contract.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_attention_block(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        kv_cache: &mut ops::kv_cache::KVCache,
        layer_idx: usize,
        bufs: AttnBufs<'_>,
        dims: AttnDims,
    ) {
        let AttnDims {
            hidden,
            layer_q_dim,
            ffn_uses_kquant,
        } = dims;
        let hidden_val = hidden as u32;
        // M2: extract per-layer params via the structured views.
        // `attn_spec` carries the attention shape (head_dim, head counts,
        // RoPE, sliding_window, qk_norm flags); `norms` carries the eps
        // and offsets. The compiler optimises these getter methods to
        // plain field reads, so this is a clarity win — the function
        // signature can later narrow to `(AttentionSpec, LayerNorms,
        // ...)` once vindex/inference callers stop passing a flat layer.
        let attn_spec = layer.attention_spec();
        let norms_view = layer.norms();
        let norm_offset = norms_view.norm_offset;
        let eps = norms_view.eps;
        let scale = attn_spec.attn_scale;
        let layer_head_dim = attn_spec.head_dim;
        let layer_num_q_heads = attn_spec.num_q_heads;
        let layer_num_kv_heads = attn_spec.num_kv_heads;
        let layer_rope_plan = &layer.rope_freq;
        let layer_rotary_dim = if attn_spec.rotary_dim > 0 {
            attn_spec.rotary_dim
        } else {
            layer_head_dim
        };
        // Narrower of the architecture's per-layer SWA and any window the
        // engine imposed on this sequence. The kernel attends
        // `[T - window_size, T)`, so this both bounds attention and lets
        // the cache hold more rows than the window between compactions.
        let window_size = self.effective_window_for(attn_spec.sliding_window as u32);

        // Env flags governing kernel-level fusion. Cached at backend
        // startup (see `metal::flags::DecodeFlags`) so the decode hot
        // path doesn't pay 4× `getenv` per layer.  Defaults preserve
        // the proven-win May-2026 fusion wave; opts-out are diagnostic
        // only.
        let use_fused_attn = self.decode_flags.fused_attn;
        let use_fused_qkn_rope = self.decode_flags.fused_qk_norm_rope;
        // KV-cache sharing (Gemma 4 E2B): later layers reuse K/V from a
        // "source" layer that already ran this token's attention. Shared
        // layers compute their own Q (against their own residual + W_q)
        // but skip K/V append + read from the source's cache. The fused
        // attention paths write the layer's own cache, so they're disabled
        // for shared layers — the unfused encode_kv_attend branch below
        // handles the source-cache binding.
        //
        // Shared layers' position counter is pinned to the source's
        // (`source.current_len` is the count of stored positions
        // including the new token, since source has already finished its
        // block for this token by the time we reach the shared layer).
        let kv_shared_source = layer.kv_shared_source;
        let attend_cache_idx = kv_shared_source.unwrap_or(layer_idx);
        // `pos` is the ABSOLUTE stream position to RoPE at; `t_val` is how
        // many cached rows to attend. They are equal-and-offset only while
        // nothing has been evicted — once a window slides, occupancy falls
        // and position keeps climbing, so they must be read from different
        // fields. Deriving `t_val` from `pos` (as `pos + 1`) is what tied
        // them together before.
        let pos = if let Some(src) = kv_shared_source {
            // Source has already advanced past the row it just wrote;
            // RoPE at that row's position.
            (kv_cache.layers[src].abs_position.saturating_sub(1)) as u32
        } else {
            kv_cache.layers[layer_idx].abs_position as u32
        };
        let t_val = if kv_shared_source.is_some() {
            // Source's current_len already counts this token.
            kv_cache.layers[attend_cache_idx].current_len as u32
        } else {
            (kv_cache.layers[layer_idx].current_len + 1) as u32
        };
        let attn_span = ops::kv_cache::attention_span(t_val, window_size);

        // kv_append_attend_fused uses a fixed tg_scores[SHORT_ATTENTION_SPAN]
        // threadgroup array. Spans beyond that overflow it — global-attention
        // layers (window_size=0) grow unboundedly and must fall back to
        // encode_kv_attend, which auto-selects kv_attention_long past the threshold.
        //
        // Additionally, the kernel is designed for head_dim <= MAX_HEAD_DIM_SINGLE_SG
        // (it dispatches exactly head_dim threads per group and assumes head_dim fits
        // in a single simdgroup). Layers with larger head_dim must use the unfused
        // encode_kv_append + encode_kv_attend path which handles arbitrary head_dim.
        let use_fused_kv_aa = attn_span <= ops::kv_cache::SHORT_ATTENTION_SPAN
            && layer_head_dim <= MAX_HEAD_DIM_SINGLE_SG
            && self.decode_flags.fused_kv_append_attend
            // Shared layers don't append to their own cache; the fused
            // append+attend kernel writes its argument cache, so it can't
            // be reused on the source's cache without corrupting the
            // source. Fall through to the unfused attend-only branch.
            && kv_shared_source.is_none();
        let use_fused_post_attn = self.decode_flags.fused_post_attn_norm;

        // Path 1: full attention fusion. Skips the qk_norm_rope dispatch,
        // the kv_append_attend_fused dispatch, AND the three Q/K/V
        // bias_add dispatches (when the layer has biases) — all handled
        // inside `attn_fused`. The decision comes from the shared
        // `attn_fused_will_fire` authority, which `encode_input_norm_and_qkv`
        // also consulted to skip its bias dispatches — the two sites must
        // agree or a bias is applied twice / dropped.
        let did_fused_attn = self.attn_fused_will_fire(layer, kv_cache, layer_idx);
        let _ = use_fused_attn;

        // ── Step 1.5 + 2: QK-norm + RoPE ──
        if did_fused_attn {
            let cache = &kv_cache.layers[layer_idx];
            // No-QK-norm archs bind the shared empty-slice stub; the
            // has_qk_norm flag gates the read (sinks convention).
            let q_w_buf = self.bufs.get_f32(layer.q_norm_weight.unwrap_or(&[]));
            let k_w_buf = self.bufs.get_f32(layer.k_norm_weight.unwrap_or(&[]));
            let t_val = (cache.current_len + 1) as u32;
            let hd_val = layer_head_dim as u32;
            let nq_val = layer_num_q_heads as u32;
            let nkv_val = cache.num_kv_heads as u32;
            let qk_off = norms_view.qk_norm_offset;
            let rdim = layer_rotary_dim as u32;
            let mut tg_w: u64 = 1;
            while tg_w < layer_head_dim as u64 && tg_w < MAX_HEAD_DIM_SINGLE_SG as u64 {
                tg_w <<= 1;
            }
            enc.set_compute_pipeline_state(&self.attention.attn_fused_pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(bufs.k_out), 0);
            enc.set_buffer(2, Some(bufs.v_out), 0);
            enc.set_buffer(3, Some(&cache.k_cache), 0);
            enc.set_buffer(4, Some(&cache.v_cache), 0);
            enc.set_buffer(5, Some(bufs.attn_out_buf), 0);
            enc.set_buffer(6, Some(&q_w_buf), 0);
            enc.set_buffer(7, Some(&k_w_buf), 0);
            enc.set_bytes(8, 4, &t_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &hd_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &nq_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(11, 4, &nkv_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(12, 4, &scale as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(13, 4, &window_size as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(14, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(15, 4, &qk_off as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(17, 4, &rdim as *const u32 as *const std::ffi::c_void);
            // Attention sinks (GPT-OSS) — see `stages::sinks`.
            crate::stages::sinks::bind(
                enc,
                ATTN_FUSED_SINKS_INDEX,
                layer.attn_sinks,
                layer_num_q_heads,
            );
            crate::stages::rope_freq::bind(
                enc,
                ATTN_FUSED_INV_FREQ_INDEX,
                ATTN_FUSED_AMPLITUDE_INDEX,
                layer_rope_plan,
                layer_head_dim,
                layer.rotary_dim,
            );
            enc.set_bytes(
                ATTN_FUSED_ABS_POS_INDEX,
                4,
                &pos as *const u32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                ATTN_FUSED_SOFTCAP_INDEX,
                4,
                &layer.attn_softcap as *const f32 as *const std::ffi::c_void,
            );
            // QK-norm flag + Q/K/V projection biases (GPT-OSS). Presence
            // is all-or-nothing here by `attn_fused_will_fire`'s gate;
            // absent slots bind the empty-slice stub, unread under the
            // flags.
            let has_qk_norm: u32 = u32::from(attn_spec.q_norm_enabled);
            enc.set_bytes(
                ATTN_FUSED_HAS_QK_NORM_INDEX,
                4,
                &has_qk_norm as *const u32 as *const std::ffi::c_void,
            );
            let has_qkv_bias: u32 = u32::from(layer.attn_q_bias.is_some());
            let qb_buf = self.bufs.get_f32(layer.attn_q_bias.unwrap_or(&[]));
            let kb_buf = self.bufs.get_f32(layer.attn_k_bias.unwrap_or(&[]));
            let vb_buf = self.bufs.get_f32(layer.attn_v_bias.unwrap_or(&[]));
            enc.set_buffer(ATTN_FUSED_QKV_BIAS_INDEX, Some(&qb_buf), 0);
            enc.set_buffer(ATTN_FUSED_QKV_BIAS_INDEX + 1, Some(&kb_buf), 0);
            enc.set_buffer(ATTN_FUSED_QKV_BIAS_INDEX + 2, Some(&vb_buf), 0);
            enc.set_bytes(
                ATTN_FUSED_QKV_BIAS_INDEX + 3,
                4,
                &has_qkv_bias as *const u32 as *const std::ffi::c_void,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(layer_num_q_heads as u64, 1, 1),
                MTLSize::new(tg_w, 1, 1),
            );
            kv_cache.layers[layer_idx].advance_one();
        } else if use_fused_qkn_rope
            && layer.q_norm_weight.is_some()
            && layer.k_norm_weight.is_some()
        {
            let q_w = layer.q_norm_weight.unwrap();
            let k_w = layer.k_norm_weight.unwrap();
            let hd_val = layer_head_dim as u32;
            let nq_val = layer_num_q_heads as u32;
            let qk_off = norms_view.qk_norm_offset;
            let rdim = layer_rotary_dim as u32;
            let mut tg_w: usize = 1;
            while tg_w < layer_head_dim && tg_w < MAX_HEAD_DIM_DOUBLE_SG {
                tg_w <<= 1;
            }
            let q_w_buf = self.bufs.get_f32(q_w);
            let k_w_buf = self.bufs.get_f32(k_w);
            let total_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
            enc.set_compute_pipeline_state(&self.norms.qk_norm_rope_fused_pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(bufs.k_out), 0);
            enc.set_buffer(2, Some(&q_w_buf), 0);
            enc.set_buffer(3, Some(&k_w_buf), 0);
            enc.set_bytes(4, 4, &hd_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &nq_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &qk_off as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &pos as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &rdim as *const u32 as *const std::ffi::c_void);
            crate::stages::rope_freq::bind(
                enc,
                8,
                11,
                layer_rope_plan,
                layer_head_dim,
                layer.rotary_dim,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(total_heads, 1, 1),
                MTLSize::new(tg_w as u64, 1, 1),
            );
        } else {
            if let (Some(q_w), Some(k_w)) = (layer.q_norm_weight, layer.k_norm_weight) {
                let hd_val = layer_head_dim as u32;
                let nq_val = layer_num_q_heads as u32;
                let qk_off = norms_view.qk_norm_offset;
                let mut tg_w: usize = 1;
                while tg_w < layer_head_dim && tg_w < MAX_HEAD_DIM_DOUBLE_SG {
                    tg_w <<= 1;
                }
                let q_w_buf = self.bufs.get_f32(q_w);
                let k_w_buf = self.bufs.get_f32(k_w);
                let total_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
                enc.set_compute_pipeline_state(&self.norms.qk_norm_qk_pipeline);
                enc.set_buffer(0, Some(bufs.q_out), 0);
                enc.set_buffer(1, Some(bufs.k_out), 0);
                enc.set_buffer(2, Some(&q_w_buf), 0);
                enc.set_buffer(3, Some(&k_w_buf), 0);
                enc.set_bytes(4, 4, &hd_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(5, 4, &nq_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &qk_off as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(total_heads, 1, 1),
                    MTLSize::new(tg_w as u64, 1, 1),
                );
            }

            // ── Step 2: RoPE on Q and K heads (batched — one dispatch each) ──
            let hd = layer_head_dim as u32;
            let rdim = layer_rotary_dim as u32;
            let rope_pairs = (layer_rotary_dim / 2) as u64;
            let num_q = layer_num_q_heads as u32;
            let total_qk_heads = (layer_num_q_heads + layer_num_kv_heads) as u64;
            enc.set_compute_pipeline_state(&self.attention.rope_at_pos_batched_qk_pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(bufs.k_out), 0);
            enc.set_bytes(2, 4, &hd as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &pos as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &rdim as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
            crate::stages::rope_freq::bind(
                enc,
                3,
                7,
                layer_rope_plan,
                layer_head_dim,
                layer.rotary_dim,
            );
            enc.dispatch_threads(
                MTLSize::new(rope_pairs, total_qk_heads, 1),
                MTLSize::new(rope_pairs.min(256), 1, 1),
            );
        }

        // ── Step 3: V-norm batched (optional) ──
        if layer.has_v_norm {
            let hd_val = layer_head_dim as u32;
            let num_kv = layer_num_kv_heads as u32;
            let mut tg_w: u64 = 1;
            while tg_w < layer_head_dim as u64 && tg_w < MAX_HEAD_DIM_DOUBLE_SG as u64 {
                tg_w <<= 1;
            }
            enc.set_compute_pipeline_state(&self.norms.v_norm_batched_pipeline);
            enc.set_buffer(0, Some(bufs.v_out), 0);
            enc.set_buffer(1, Some(bufs.v_out), 0);
            enc.set_bytes(2, 4, &hd_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(4, 4, &num_kv as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(layer_num_kv_heads as u64, 1, 1),
                MTLSize::new(tg_w, 1, 1),
            );
        }

        // ── Step 4: KV-append + KV-attend ──
        // Skipped entirely when `did_fused_attn` is true (the unified
        // `attn_fused` kernel above already wrote both cache rows + the
        // attention output and bumped current_len). Shared layers
        // (`kv_shared_source = Some(_)`) skip the append entirely and
        // attend against the source's cache.
        if did_fused_attn {
            // Already done — attn_fused wrote attn_out_buf + bumped current_len.
        } else if use_fused_kv_aa {
            let cache = &kv_cache.layers[layer_idx];
            let t_val = (cache.current_len + 1) as u32;
            let hd = cache.head_dim as u32;
            let num_q_val = layer_num_q_heads as u32;
            let num_kv = cache.num_kv_heads as u32;
            enc.set_compute_pipeline_state(&self.attention.kv_append_attend_fused_pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(&cache.k_cache), 0);
            enc.set_buffer(2, Some(&cache.v_cache), 0);
            enc.set_buffer(3, Some(bufs.attn_out_buf), 0);
            enc.set_bytes(4, 4, &t_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hd as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &num_q_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &window_size as *const u32 as *const std::ffi::c_void);
            enc.set_buffer(10, Some(bufs.k_out), 0);
            enc.set_buffer(11, Some(bufs.v_out), 0);
            // Attention sinks (GPT-OSS) — see `stages::sinks`.
            crate::stages::sinks::bind(
                enc,
                KV_APPEND_ATTEND_SINKS_INDEX,
                layer.attn_sinks,
                layer_num_q_heads,
            );
            enc.set_bytes(
                KV_APPEND_ATTEND_SOFTCAP_INDEX,
                4,
                &layer.attn_softcap as *const f32 as *const std::ffi::c_void,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(layer_num_q_heads as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(layer_head_dim as u64),
                    1,
                    1,
                ),
            );
        } else if let Some(src) = kv_shared_source {
            // Shared layer: attend against source's cache. Skip append
            // (the source already appended its K/V row this token).
            // `t_val = source.current_len` because source has already
            // incremented it to count the new token.
            let cache = &kv_cache.layers[src];
            let t_val_u = cache.current_len as u32;
            let hd_u = cache.head_dim as u32;
            let num_q_u = layer_num_q_heads as u32;
            let num_kv_u = cache.num_kv_heads as u32;
            // Pick the same pipeline encode_kv_attend would (long-form for
            // spans past SHORT_ATTENTION_SPAN; standard otherwise).
            let span_u = ops::kv_cache::attention_span(t_val_u, window_size);
            let pipeline = if span_u > ops::kv_cache::SHORT_ATTENTION_SPAN {
                &self.attention.kv_attend_long_pipeline
            } else {
                &self.attention.kv_attend_pipeline
            };
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(bufs.q_out), 0);
            enc.set_buffer(1, Some(&cache.k_cache), 0);
            enc.set_buffer(2, Some(&cache.v_cache), 0);
            enc.set_buffer(3, Some(bufs.attn_out_buf), 0);
            enc.set_bytes(4, 4, &t_val_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &hd_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &num_q_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &num_kv_u as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &window_size as *const u32 as *const std::ffi::c_void);
            crate::stages::sinks::bind(enc, 10, layer.attn_sinks, layer_num_q_heads);
            enc.set_bytes(
                12,
                4,
                &layer.attn_softcap as *const f32 as *const std::ffi::c_void,
            );
            enc.dispatch_thread_groups(
                MTLSize::new(layer_num_q_heads as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(cache.head_dim as u64),
                    1,
                    1,
                ),
            );
        } else {
            ops::kv_cache::encode_kv_append(
                enc,
                &kv_cache.layers[layer_idx],
                &self.attention.kv_append_pipeline,
                bufs.k_out,
                bufs.v_out,
            );
            ops::kv_cache::encode_kv_attend(
                enc,
                &kv_cache.layers[layer_idx],
                &self.attention.kv_attend_pipeline,
                Some(&self.attention.kv_attend_long_pipeline),
                bufs.q_out,
                bufs.attn_out_buf,
                layer_num_q_heads,
                scale,
                window_size,
                layer.attn_sinks,
                layer.attn_softcap,
            );
        }
        // Only own-cache layers advance current_len; shared layers leave
        // their (unused) cache pointer at 0 forever.
        if !did_fused_attn && kv_shared_source.is_none() {
            kv_cache.layers[layer_idx].advance_one();
        }

        // ── Step 5a: O projection ──
        //
        // Dispatched on `wo`'s own format, not `wq`'s. Selecting an
        // operand's ABI from a *different* operand's representation is
        // what produced the defect this replaced: `uses_kquant` reads
        // `wq`, so a Q4_0 `wo` fell to the Q8-only branch below and was
        // handed to `q8_matvec` as though its 18-byte packed blocks were
        // int8 rows, alongside a weight-scale buffer Q4_0 has no source
        // for. That was survivable only because every caller fabricated
        // an empty scale buffer; removing the fabrication exposed it.
        match layer.wo.format() {
            // `o_proj::encode` understands all four packed formats — its
            // own doc covers "Q4_0 / Q8_0: quantise attn_in -> Q8 int8 +
            // per-32 f16 scale" as well as the k-quant families.
            larql_compute::QuantFormat::Q4_0
            | larql_compute::QuantFormat::Q4_K
            | larql_compute::QuantFormat::Q4_KF
            | larql_compute::QuantFormat::Q6_K => {
                use crate::stages::quant_matvec::Pipelines;
                let pipes = Pipelines {
                    q4kf_proj: Some(&self.attention.q4kf_proj_pipeline.state),
                    q4k_matvec_fallback: &self.quant.q4k_matvec_pipeline,
                    q6k_matvec: &self.quant.q6k_matvec_pipeline,
                    q4_matvec: &self.q4.matvec,
                    q4k_matmul: None,
                };
                // K from the store's own row width, not the logical
                // `layer_q_dim` — O rows may be padded to the quant block
                // (same audit-F17 hazard as the QKV kernels; the padded
                // columns dequantise to zero, so the stored width is
                // exact given the input buffer's zeroed slack).
                let o_k_store = layer.wo.stored_cols(hidden, layer_q_dim);
                crate::stages::o_proj::encode(
                    enc,
                    &pipes,
                    &self.quant.q8_quant_pipeline,
                    layer.wo.format(),
                    bufs.wo,
                    bufs.attn_out_buf,
                    0,
                    bufs.o_q8_scratch,
                    0,
                    bufs.o_q8s_scratch,
                    0,
                    bufs.o_out_buf,
                    0,
                    o_k_store,
                    hidden,
                );
            }
            // Q8 legacy path: decode-specific `q8_matvec` shader (not in
            // stages::quant_matvec, which uses `q4_matvec` with a
            // different buffer layout). Q8_0 only — it is the one format
            // that supplies the external weight scales this shader reads
            // at buffer index 2.
            larql_compute::QuantFormat::Q8_0 => {
                let dim_val = layer_q_dim as u32;
                let blocks = layer_q_dim.div_ceil(LEGACY_BLOCK_ELEMS) as u32;
                enc.set_compute_pipeline_state(&self.quant.q8_quant_pipeline);
                enc.set_buffer(0, Some(bufs.attn_out_buf), 0);
                enc.set_buffer(1, Some(bufs.o_q8_scratch), 0);
                enc.set_buffer(2, Some(bufs.o_q8s_scratch), 0);
                enc.set_bytes(3, 4, &dim_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_threads(
                    MTLSize::new(blocks as u64, 1, 1),
                    MTLSize::new(
                        crate::kernels::DISPATCH_TG_MAX_THREADS.min(blocks as u64),
                        1,
                        1,
                    ),
                );

                let o_rows = hidden as u32;
                let o_k = layer_q_dim as u32;
                assert!(
                    layer_q_dim <= crate::shaders::q8_matvec::MAX_K,
                    "q8_matvec stages its Q8 input in threadgroup memory capped at \
                     K = {}; layer_q_dim {layer_q_dim} would corrupt it (audit F13)",
                    crate::shaders::q8_matvec::MAX_K,
                );
                enc.set_compute_pipeline_state(&self.quant.q8_matvec_pipeline.state);
                enc.set_buffer(0, Some(bufs.wo), 0);
                enc.set_buffer(1, Some(bufs.o_q8_scratch), 0);
                enc.set_buffer(
                    2,
                    Some(
                        bufs.wo_scales
                            .expect("legacy scale path requires an external-scale format"),
                    ),
                    0,
                );
                enc.set_buffer(3, Some(bufs.o_q8s_scratch), 0);
                enc.set_buffer(4, Some(bufs.o_out_buf), 0);
                enc.set_bytes(5, 4, &o_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &o_k as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new((hidden as u64).div_ceil(8), 1, 1),
                    MTLSize::new(256, 1, 1),
                );
            }
            // Loud rather than silent. The previous predicate routed
            // anything non-k-quant here, so an unsupported format was
            // reinterpreted as Q8 and produced plausible numbers instead
            // of an error.
            format => panic!(
                "O projection has no dispatch for {format:?}; supported: \
                 Q4_0, Q4_K, Q4_KF, Q6_K (o_proj::encode) and Q8_0 (legacy q8_matvec)"
            ),
        }

        // O-projection bias (GPT-OSS) joins before the residual add —
        // the same point the CPU reference (`forward::add_bias`) applies
        // it. Dispatched only when the layer carries the bias.
        if let Some(b) = layer.attn_o_bias {
            assert_eq!(
                b.len(),
                hidden,
                "O projection bias has {} entries but hidden is {hidden} — \
                 the extracted tensor does not match this model",
                b.len()
            );
            let b_buf = self.bufs.get_f32(b);
            crate::stages::bias_add::encode(
                enc,
                &self.attention.bias_add_pipeline,
                bufs.o_out_buf,
                0,
                &b_buf,
                hidden,
            );
        }

        // ── Step 5b: Residual + post-attn norm + ffn-input norm ──
        let has_post_norms = layer.has_post_norms;
        if has_post_norms {
            let pre_ffn_buf = if let Some(pfn) = layer.pre_ffn_norm {
                self.bufs.get_f32(pfn)
            } else {
                bufs.post_attn_norm.clone()
            };
            // The triple-fused kernel has no `b_scale` slot, so a
            // Granite-style `residual_multiplier != 1.0` must take the
            // unfused chain below, which binds it. The equivalent guards
            // in `encode_post_ffn.rs` existed for exactly this reason;
            // this site lacked one (capability audit F18).
            let fusable_residual = layer.residual_multiplier == 1.0;
            if use_fused_post_attn && ffn_uses_kquant && fusable_residual {
                // Triple-fused: post_attn_norm + residual_norm + h_post_attn
                // store in ONE dispatch.
                enc.set_compute_pipeline_state(&self.norms.post_attn_residual_norm_store_pipeline);
                enc.set_buffer(0, Some(bufs.h_buf), 0);
                enc.set_buffer(1, Some(bufs.o_out_buf), 0);
                enc.set_buffer(2, Some(bufs.post_attn_norm), 0);
                enc.set_buffer(3, Some(&pre_ffn_buf), 0);
                enc.set_buffer(4, Some(bufs.ffn_norm_out), 0);
                enc.set_buffer(5, Some(bufs.h_post_attn), 0);
                enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(
                        crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                        1,
                        1,
                    ),
                );
            } else {
                use crate::ops::full_pipeline::encode_rms_norm;
                encode_rms_norm(
                    enc,
                    &self.norms.rms_norm_pipeline,
                    bufs.o_out_buf,
                    bufs.post_attn_norm,
                    bufs.normed_scratch,
                    hidden,
                    eps,
                    norm_offset,
                );
                // Granite residual_multiplier; 1.0 for every other model.
                // `residual_norm_store` reads it at buffer(8),
                // `residual_norm_q8` at buffer(9).
                let b_scale: f32 = layer.residual_multiplier;
                if ffn_uses_kquant {
                    enc.set_compute_pipeline_state(&self.norms.residual_norm_store_pipeline);
                    enc.set_buffer(0, Some(bufs.h_buf), 0);
                    enc.set_buffer(1, Some(bufs.normed_scratch), 0);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(bufs.ffn_norm_out), 0);
                    enc.set_buffer(4, Some(bufs.h_post_attn), 0);
                    enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &b_scale as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(
                            crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                            1,
                            1,
                        ),
                    );
                } else {
                    enc.set_compute_pipeline_state(&self.norms.residual_norm_q8_pipeline);
                    enc.set_buffer(0, Some(bufs.h_buf), 0);
                    enc.set_buffer(1, Some(bufs.normed_scratch), 0);
                    enc.set_buffer(2, Some(&pre_ffn_buf), 0);
                    enc.set_buffer(3, Some(bufs.ffn_q8), 0);
                    enc.set_buffer(4, Some(bufs.ffn_q8s), 0);
                    enc.set_buffer(5, Some(bufs.h_post_attn), 0);
                    enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
                    enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
                    enc.set_bytes(9, 4, &b_scale as *const f32 as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(1, 1, 1),
                        MTLSize::new(
                            crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                            1,
                            1,
                        ),
                    );
                }
            }
        } else if ffn_uses_kquant {
            let b_scale: f32 = layer.residual_multiplier;
            enc.set_compute_pipeline_state(&self.norms.residual_norm_store_pipeline);
            enc.set_buffer(0, Some(bufs.h_buf), 0);
            enc.set_buffer(1, Some(bufs.o_out_buf), 0);
            enc.set_buffer(2, Some(bufs.post_attn_norm), 0);
            enc.set_buffer(3, Some(bufs.ffn_norm_out), 0);
            enc.set_buffer(4, Some(bufs.h_post_attn), 0);
            enc.set_bytes(5, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(6, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &b_scale as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                    1,
                    1,
                ),
            );
        } else {
            let b_scale: f32 = layer.residual_multiplier;
            enc.set_compute_pipeline_state(&self.norms.residual_norm_q8_pipeline);
            enc.set_buffer(0, Some(bufs.h_buf), 0);
            enc.set_buffer(1, Some(bufs.o_out_buf), 0);
            enc.set_buffer(2, Some(bufs.post_attn_norm), 0);
            enc.set_buffer(3, Some(bufs.ffn_q8), 0);
            enc.set_buffer(4, Some(bufs.ffn_q8s), 0);
            enc.set_buffer(5, Some(bufs.h_post_attn), 0);
            enc.set_bytes(6, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(7, 4, &eps as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &b_scale as *const f32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                    1,
                    1,
                ),
            );
        }
    }
}
