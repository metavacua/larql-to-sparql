//! GPU expert dispatch for per-layer Q4_K MoE models (§5.12).
//!
//! Called when a MoE layer's expert weights are in `QuantFormat::Q4_K`
//! (per-layer files, not BF16 blob). The router runs on CPU (cheap: 2816×128
//! matmul), expert FFNs run on GPU using existing Q4_K shaders.
//!
//! Flow per MoE layer (after the standard GPU commit for `h_post_attn`):
//!
//! 1. CPU: model-declared expert input + router policy + softmax + top-K.
//! 2. CPU→GPU: write the K selected experts' gate / up / down byte slices
//!    DIRECTLY into pre-allocated Metal staging buffers (one memcpy each).
//! 3. GPU: `q4k_ffn_gate_up` over all K experts in one dispatch.
//! 4. GPU: K × `geglu_gelu_tanh` — one per expert at strided act_buf offset
//!    `e × inter_padded` so down's `K = inter_padded` reads see zero padding.
//! 5. GPU: K × `q4k_matvec` for expert down projections.
//! 6. Commit + wait (one GPU sync per MoE layer).
//! 7. CPU: read back K × hidden expert outputs, weighted sum → `moe_out`.
//!
//! Phase 2 (2026-04-26): all scratch is pre-allocated once per decode call
//! via `MoeScratch::new(...)` and reused across every MoE layer. Previously
//! each layer called `bufs.output(...)` ~10 times (~120ms allocation overhead
//! per token at 30 MoE layers on M3 Max). Buffer sizes are constant per model
//! — `(top_k, hidden, inter_padded)` — so the buffers can stay live for the
//! whole decode and serve every layer's expert routing.

use metal::*;
use std::ffi::c_void;

use super::buffers::{read_buffer_f32, BufferCache};
use super::MetalBackend;
use larql_compute::cpu::ops::moe::{
    moe_expert_input, moe_post_expert_output, moe_route_from_router_input, moe_router_input,
};
use larql_compute::MoeLayerWeights;

/// Pre-allocated scratch for the whole MoE decode loop.
///
/// All sizes are determined by `(top_k, hidden, intermediate_size)` of the
/// first MoE layer, which is constant across MoE layers in the architectures
/// we currently target (Gemma 4 26B A4B). Sizing assumes Q4_K weights with
/// 256-element super-blocks, 144 bytes per row-block.
///
/// `act_buf` is sized to `top_k × inter_padded` and zero-initialised so the
/// `inter_padded - inter` padding columns of every expert's strided slice
/// contribute nothing through the down projection — required when
/// `moe.intermediate_size` is not a multiple of 256 (e.g. Gemma 4 26B's 2112
/// → inter_padded 2304).
pub struct MoeScratch {
    pub(super) top_k: usize,
    pub(super) inter: usize,
    pub(super) inter_padded: usize,
    pub(super) hidden: usize,
    /// The expert store's weight format — sizes the row strides below and
    /// selects the matvec pipelines at dispatch. Q4_K and Q6_K only.
    pub(super) format: larql_compute::QuantFormat,
    /// Gate/up STORED row width in elements (block-padded by the writer;
    /// GPT-OSS 2880 → 3072, block-multiple hidden sizes unchanged).
    pub(super) weight_cols: usize,
    pub(super) row_bytes: usize,
    pub(super) down_row_bytes: usize,

    pub(super) gate_buf: Buffer,
    pub(super) up_buf: Buffer,
    pub(super) down_bufs: Vec<Buffer>,

    pub(super) x_buf: Buffer,
    pub(super) g_out: Buffer,
    pub(super) u_out: Buffer,
    pub(super) act_buf: Buffer,
    pub(super) expert_outs: Buffer,
    /// De-interleaved per-selected-expert gate/up bias rows, `top_k × inter`
    /// f32 each. Always allocated (small) so the activation kernel's bias
    /// slots bind a real buffer; `has_bias` gates the read.
    pub(super) gate_bias_buf: Buffer,
    pub(super) up_bias_buf: Buffer,
    /// Selected experts' down-bias rows for the GPU weighted combine,
    /// `top_k × hidden` f32, slot-aligned with `expert_outs`. Staged by
    /// the inline-combine path only; the CPU-combine paths never read it.
    pub(super) down_bias_staged: Buffer,
}

// `Buffer` is `Send + Sync` on its own; the Metal types we hold here mirror
// the rest of `MetalBackend` (single-process, single-device).  Stamping it so
// `larql-server` can stash a `MoeScratch` inside `Arc<AppState>` without
// fighting the borrow checker.
unsafe impl Send for MoeScratch {}
unsafe impl Sync for MoeScratch {}

impl MoeScratch {
    /// Public constructor — used by `larql-server`'s shard expert path so it
    /// can preallocate one scratch per (hidden, intermediate, top_k) shape on
    /// startup and reuse it for every incoming RPC.
    pub fn new_public(backend: &MetalBackend, top_k: usize, hidden: usize, inter: usize) -> Self {
        Self::new(
            &backend.bufs,
            top_k,
            hidden,
            inter,
            larql_compute::QuantFormat::Q4_K,
            hidden,
        )
    }

    /// Format-aware public constructor: `format` sizes the row strides and
    /// selects the matvec pipelines; `weight_cols` is the gate/up STORED
    /// row width (`MoeLayerWeights::gate_up_cols`).
    pub fn new_public_with_format(
        backend: &MetalBackend,
        top_k: usize,
        hidden: usize,
        inter: usize,
        format: larql_compute::QuantFormat,
        weight_cols: usize,
    ) -> Self {
        Self::new(&backend.bufs, top_k, hidden, inter, format, weight_cols)
    }

    pub(super) fn new(
        bufs: &BufferCache,
        top_k: usize,
        hidden: usize,
        inter: usize,
        format: larql_compute::QuantFormat,
        weight_cols: usize,
    ) -> Self {
        let (block, bytes_per_block) = format
            .packed_block_layout()
            .expect("MoE expert scratch requires a block format (Q4_K/Q6_K)");
        let inter_padded = inter.div_ceil(block) * block;
        // Row strides from the STORE's own geometry: gate/up rows at the
        // writer-padded `weight_cols` (a truncating `hidden / block` here
        // mis-strided every non-block-multiple model), down at
        // `inter_padded` — both in this format's bytes per super-block.
        debug_assert!(weight_cols.is_multiple_of(block));
        let row_bytes = (weight_cols / block) * bytes_per_block;
        let down_row_bytes = (inter_padded / block) * bytes_per_block;

        let gate_buf = bufs.output((top_k * inter * row_bytes) as u64);
        let up_buf = bufs.output((top_k * inter * row_bytes) as u64);
        let down_bufs: Vec<Buffer> = (0..top_k)
            .map(|_| bufs.output((hidden * down_row_bytes) as u64))
            .collect();

        let x_buf = bufs.output((weight_cols * 4) as u64);
        let g_out = bufs.output((top_k * inter * 4) as u64);
        let u_out = bufs.output((top_k * inter * 4) as u64);
        let act_buf = bufs.output((top_k * inter_padded * 4) as u64);
        let expert_outs = bufs.output((top_k * hidden * 4) as u64);

        let gate_bias_buf = bufs.output((top_k * inter * 4) as u64);
        let up_bias_buf = bufs.output((top_k * inter * 4) as u64);
        let down_bias_staged = bufs.output((top_k * hidden * 4) as u64);

        // Zero the padding tails once. GEGLU writes only the first `inter`
        // floats of each expert's `inter_padded`-strided slice, so the
        // remaining `inter_padded - inter` floats stay zero forever. The
        // x_buf tail (`weight_cols - hidden`) likewise stays zero so the
        // writer's row padding contributes nothing to any dot product.
        unsafe {
            let ptr = act_buf.contents() as *mut f32;
            std::ptr::write_bytes(ptr, 0, top_k * inter_padded);
            let xp = x_buf.contents() as *mut f32;
            std::ptr::write_bytes(xp, 0, weight_cols);
        }

        Self {
            top_k,
            inter,
            inter_padded,
            hidden,
            format,
            weight_cols,
            row_bytes,
            down_row_bytes,
            gate_buf,
            up_buf,
            down_bufs,
            x_buf,
            g_out,
            u_out,
            act_buf,
            expert_outs,
            gate_bias_buf,
            up_bias_buf,
            down_bias_staged,
        }
    }
}

impl MetalBackend {
    /// High-level decode step using GPU expert dispatch for Q4_K per-layer format.
    ///
    /// `get_expert(layer_idx, expert_idx)` returns the (gate+up, down) byte
    /// slices for the requested expert, borrowed from the model weights (mmap).
    /// The borrow only needs to outlive the closure call — `gpu_moe_dispatch`
    /// memcpys both slices into pre-allocated Metal buffers before returning.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_token_q4k_moe<'w, F>(
        &self,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_base: f32,
        norm_eps: f32,
        get_expert: F,
    ) -> Option<Vec<f32>>
    where
        F: Fn(usize, usize) -> Option<(&'w [u8], &'w [u8])>,
    {
        let mut kv_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut kv_guard,
            layers,
            super::decode::DEFAULT_KV_CACHE_MAX_SEQ,
        );

        // Cache scratch by `(top_k, hidden, intermediate_size)` on the
        // backend so the ~15 Metal buffer allocations (~120ms on Gemma 4
        // 26B-A4B, M3 Max) only happen at first use, not per token. The
        // shape stays constant across MoE layers in the architectures we
        // currently target (Gemma 4 26B A4B and similar) and across
        // decode calls for the same loaded model — when the cached
        // scratch's shape doesn't match the requested shape we evict and
        // reallocate, mirroring `larql-server`'s `moe_scratches`
        // HashMap-by-shape cache. Holding the lock for the whole decode
        // matches the `kv_cache` pattern above; concurrent decodes on
        // the same backend serialise here just as they do on KV.
        let mut scratch_guard = self.moe_scratch.lock().unwrap();
        if let Some(shape) = layers.iter().find_map(|l| l.moe.as_ref()).map(|m| {
            (
                m.top_k,
                hidden,
                m.intermediate_size,
                m.expert_data_format,
                m.gate_up_cols(hidden),
            )
        }) {
            let needs_alloc = match scratch_guard.as_ref() {
                Some(s) => (s.top_k, s.hidden, s.inter, s.format, s.weight_cols) != shape,
                None => true,
            };
            if needs_alloc {
                *scratch_guard = Some(MoeScratch::new(
                    &self.bufs, shape.0, shape.1, shape.2, shape.3, shape.4,
                ));
            }
        }
        let scratch_ref = scratch_guard.as_ref();

        let mut moe_fn = {
            let get_expert_ref = &get_expert;
            move |layer_idx: usize, h_post_attn: &[f32]| -> Vec<f32> {
                let moe = match layers[layer_idx].moe.as_ref() {
                    Some(m) => m,
                    None => return vec![0.0f32; hidden],
                };
                let scratch = scratch_ref
                    .expect("MoE layer present but no scratch allocated — model has MoE");
                self.gpu_moe_dispatch_with_scratch(
                    h_post_attn,
                    moe,
                    norm_eps,
                    scratch,
                    |expert_idx| get_expert_ref(layer_idx, expert_idx),
                )
            }
        };

        // Merged-CB context: per-layer preconditions permitting, the
        // interleave routes on CPU and encodes experts + combine into the
        // command buffer the next layer's attention rides — the moe_fn
        // closure above stays as the per-layer fallback.
        let inline_ctx = scratch_ref.map(|s| super::decode::InlineMoeCtx::new(s, norm_eps));

        Some(self.decode_token_with_moe_split_fn(
            kv,
            layers,
            x,
            hidden,
            inter,
            q_dim,
            kv_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            Some(&mut moe_fn),
            None,
            None,
            larql_compute::StateDumpMask::Full,
            inline_ctx.as_ref(),
        ))
    }

    /// Public single-layer MoE block entry — the parity/test surface for
    /// [`Self::gpu_moe_dispatch_with_scratch`], which stays `pub(super)`.
    /// Runs the layer's full expert block (CPU routing + GPU experts +
    /// CPU combine) exactly as the decode loop does.
    pub fn moe_block_for_layer<'w, F>(
        &self,
        h_post_attn: &[f32],
        moe: &MoeLayerWeights<'_>,
        eps: f32,
        scratch: &MoeScratch,
        get_expert_bytes: F,
    ) -> Vec<f32>
    where
        F: Fn(usize) -> Option<(&'w [u8], &'w [u8])>,
    {
        self.gpu_moe_dispatch_with_scratch(h_post_attn, moe, eps, scratch, get_expert_bytes)
    }

    /// GPU expert dispatch with pre-allocated scratch.
    ///
    /// Per call this does:
    ///   - 1 CPU pre-experts norm + router pass (~hidden² FLOPs, cheap).
    ///   - top_k × 2 host→shared-memory memcpys (one per gate+up + one per
    ///     down byte slice); no Metal allocations in the hot path.
    ///   - 1 fused gate+up dispatch + top_k activation dispatches +
    ///     top_k down dispatches → committed and waited on once.
    ///   - 1 readback of `top_k × hidden` f32 expert outputs + CPU weighted sum
    ///     and post-experts norm.
    ///
    /// Cache-backed shared Metal buffer for an arbitrary byte slice — the
    /// caller passes a stable byte slice (typically a Q4_K mmap region for
    /// one expert) and gets back a `Buffer` keyed on `(ptr, len)`.
    ///
    /// First call pays the copy / aliasing cost; subsequent calls with the
    /// same `bytes` slice hit the cache and return in O(1).  Used by the
    /// shard expert path so per-RPC dispatches reuse the previous call's
    /// staged buffer instead of memcpy'ing into scratch every time.
    ///
    /// When `bytes` is page-aligned in both address and size, the underlying
    /// `BufferCache` uses `new_buffer_with_bytes_no_copy` (zero-cost alias);
    /// otherwise it falls back to `new_buffer_with_data` (one-time copy at
    /// cache miss).  Either way, the *steady-state* (warmed) cost is zero.
    pub fn cached_buffer_for_bytes(&self, bytes: &[u8]) -> Buffer {
        self.bufs.get_bytes(bytes)
    }

    /// Pre-staged variant of `run_experts_preselected_metal`: takes per-expert
    /// `(gate_up_buf, down_buf)` Metal buffers (typically created once via
    /// `shared_buffer_no_copy` at server startup) instead of byte slices that
    /// would have to be memcpy'd into scratch on every call.
    ///
    /// Same wire output as `run_experts_preselected_metal` — only the staging
    /// path differs.  Because each expert's weights live in its own buffer we
    /// dispatch `q4k_ffn_gate_up` once per expert rather than once-for-all-K;
    /// the per-dispatch cost (~10–50µs on M3) is dwarfed by the eliminated
    /// memcpy (~1ms/layer at K=8).
    #[allow(clippy::too_many_arguments)]
    pub fn run_experts_prestaged_metal(
        &self,
        h_norm: &[f32],
        expert_bufs: &[(Buffer, Buffer)],
        expert_weights: &[f32],
        scratch: &MoeScratch,
    ) -> Vec<f32> {
        let hidden = h_norm.len();
        let inter = scratch.inter;
        let inter_padded = scratch.inter_padded;
        debug_assert_eq!(hidden, scratch.hidden);
        debug_assert_eq!(expert_bufs.len(), expert_weights.len());

        if expert_bufs.is_empty() || hidden == 0 || inter == 0 {
            return vec![0.0f32; hidden];
        }

        let timing_enabled =
            larql_compute::options::env_flag(larql_compute::options::ENV_METAL_MOE_TIMING);
        let t_start = std::time::Instant::now();

        let valid_count = expert_bufs.len().min(scratch.top_k);

        // Stage h_norm only (small — `hidden * 4` bytes).
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(h_norm.as_ptr(), x_ptr, hidden);
        }
        let t_stage = t_start.elapsed();

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // Per-expert gate+up dispatch.  Each expert's `gate_up_buf` holds
        // `[gate || up]`; the kernel takes them as separate buffers — pass
        // the same buffer twice with the up offset for the second binding.
        let row_bytes = scratch.row_bytes;
        let gate_half_bytes = (inter * row_bytes) as u64;
        let n_rows = inter as u32;
        let k_cols = hidden as u32;
        // Geometry travels with `q4k_ffn_gate_up_pipeline` — read it
        // off the `KernelHandle` rather than re-importing shader-module
        // constants, so a future bump of the field to a different
        // simdgroup variant doesn't silently drop rows. Same dispatch-
        // geometry-mismatch class as the q4_matvec_v4 ROADMAP entry.
        let gate_up_kh = &self.ffn.q4k_ffn_gate_up_pipeline;
        let tgs_per_mat = (inter as u64).div_ceil(gate_up_kh.rows_per_tg);

        for (e, (gate_up_buf, _)) in expert_bufs.iter().enumerate().take(valid_count) {
            enc.set_compute_pipeline_state(&gate_up_kh.state);
            // Wg = gate (offset 0), Wu = up (offset gate_half_bytes) within the
            // same per-expert mmap-backed buffer.
            enc.set_buffer(0, Some(gate_up_buf), 0);
            enc.set_buffer(1, Some(gate_up_buf), gate_half_bytes);
            enc.set_buffer(2, Some(&scratch.x_buf), 0);
            // Per-expert output offsets so K dispatches don't clobber each
            // other; same offsets the GELU/down dispatches read below.
            enc.set_buffer(3, Some(&scratch.g_out), (e * inter * 4) as u64);
            enc.set_buffer(4, Some(&scratch.u_out), (e * inter * 4) as u64);
            enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(tgs_per_mat * 2, 1, 1),
                MTLSize::new(gate_up_kh.threads_per_tg, 1, 1),
            );
        }

        // GELU-tanh activation per expert (strided to inter_padded).
        let inter_u32 = inter as u32;
        for e in 0..valid_count {
            let g_offset = (e * inter * 4) as u64;
            let u_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            enc.set_compute_pipeline_state(&self.ffn.geglu_gelu_tanh_pipeline);
            enc.set_buffer(0, Some(&scratch.g_out), g_offset);
            enc.set_buffer(1, Some(&scratch.u_out), u_offset);
            enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
            enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(inter as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(inter as u64),
                    1,
                    1,
                ),
            );
        }

        // Per-expert down projection — use each expert's pre-staged down buffer.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        // Pull dispatch geometry from the bound pipeline so this works for
        // both the 4sg and 8sg variants of `q4k_matvec` — hardcoding the
        // 4sg constants while dispatching the 8sg pipeline (the production
        // default since 2026-04-28) leaves simdgroups 4..7 unscheduled and
        // only writes rows 0..3 of each TG's 8-row range. See the matching
        // fix in `trait_impl/quant_matvec.rs::q4k_matvec`.
        let down_rows_per_tg = self.quant.q4k_matvec_pipeline.rows_per_tg;
        let down_threads_per_tg = self.quant.q4k_matvec_pipeline.threads_per_tg;
        let down_tgs = (hidden as u64).div_ceil(down_rows_per_tg);
        for (e, (_, down_buf)) in expert_bufs.iter().enumerate().take(valid_count) {
            let act_offset = (e * inter_padded * 4) as u64;
            let out_offset = (e * hidden * 4) as u64;
            enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
            enc.set_buffer(0, Some(down_buf), 0);
            enc.set_buffer(1, Some(&scratch.act_buf), act_offset);
            enc.set_buffer(2, Some(&scratch.expert_outs), out_offset);
            enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(down_tgs, 1, 1),
                MTLSize::new(down_threads_per_tg, 1, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let t_gpu = t_start.elapsed();

        let all_expert_outputs = read_buffer_f32(&scratch.expert_outs, valid_count * hidden);
        let mut moe_out = vec![0.0f32; hidden];
        for e in 0..valid_count {
            let w = expert_weights[e];
            let out_slice = &all_expert_outputs[e * hidden..(e + 1) * hidden];
            for (acc, &v) in moe_out.iter_mut().zip(out_slice) {
                *acc += v * w;
            }
        }
        let t_total = t_start.elapsed();
        if timing_enabled {
            eprintln!(
                "[run_experts_metal/prestaged] K={valid_count} stage={:.2}ms gpu={:.2}ms \
                 readback+sum={:.2}ms total={:.2}ms",
                t_stage.as_secs_f32() * 1000.0,
                (t_gpu - t_stage).as_secs_f32() * 1000.0,
                (t_total - t_gpu).as_secs_f32() * 1000.0,
                t_total.as_secs_f32() * 1000.0,
            );
        }
        moe_out
    }

    /// Run a pre-selected set of MoE experts on the GPU and return their
    /// weighted sum.  Public surface used by `larql-server`'s shard endpoint —
    /// the client picks experts via its router, the server only computes them.
    ///
    /// `h_norm` is the *already* `pre_experts_norm`-applied residual.
    /// `expert_ids` and `expert_weights` are paired (both length K).
    /// `get_expert_bytes(eid)` returns `(gate_up_bytes, down_bytes)` mmap
    /// slices for one expert; if the shard does not own the expert it should
    /// return `None` (that expert is skipped).
    ///
    /// Returns the weighted sum **without** post-experts norm — the client
    /// applies post-norm once after summing across shards, since
    /// `rms_norm(a) + rms_norm(b) ≠ rms_norm(a + b)`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_experts_preselected_metal<'w, F>(
        &self,
        h_norm: &[f32],
        expert_ids: &[usize],
        expert_weights: &[f32],
        scratch: &MoeScratch,
        get_expert_bytes: F,
    ) -> Vec<f32>
    where
        F: Fn(usize) -> Option<(&'w [u8], &'w [u8])>,
    {
        let hidden = h_norm.len();
        let inter = scratch.inter;
        let inter_padded = scratch.inter_padded;
        debug_assert_eq!(hidden, scratch.hidden, "h_norm hidden vs scratch.hidden");
        debug_assert!(
            expert_ids.len() == expert_weights.len(),
            "expert_ids and expert_weights must be same length"
        );

        if expert_ids.is_empty() || hidden == 0 || inter == 0 {
            return vec![0.0f32; hidden];
        }

        let timing_enabled =
            larql_compute::options::env_flag(larql_compute::options::ENV_METAL_MOE_TIMING);
        let t_start = std::time::Instant::now();

        // ── Stage expert weight bytes into pre-allocated Metal buffers ─────
        let row_bytes = scratch.row_bytes;
        let gate_half_bytes = inter * row_bytes;
        let up_half_bytes = inter * row_bytes;
        let down_expert_bytes = hidden * scratch.down_row_bytes;

        let gate_ptr = scratch.gate_buf.contents() as *mut u8;
        let up_ptr = scratch.up_buf.contents() as *mut u8;

        let mut valid_weights: Vec<f32> = Vec::with_capacity(expert_ids.len());
        let mut valid_count = 0usize;

        for (k, &ei) in expert_ids.iter().enumerate() {
            let Some((gu_bytes, dn_bytes)) = get_expert_bytes(ei) else {
                continue;
            };
            if gu_bytes.len() < 2 * gate_half_bytes {
                continue;
            }
            if valid_count >= scratch.top_k {
                // Caller passed more experts than scratch was sized for.
                // Truncate to fit; should not happen in practice (client's
                // top_k matches the architecture's top_k that scratch was
                // allocated for).
                break;
            }

            // Q4_K layout: gate || up, each `inter * row_bytes` bytes.
            // SAFETY: gate_ptr / up_ptr are StorageModeShared Metal buffer
            // contents; offsets are bounded by `top_k * gate_half_bytes`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    gu_bytes.as_ptr(),
                    gate_ptr.add(valid_count * gate_half_bytes),
                    gate_half_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    gu_bytes.as_ptr().add(gate_half_bytes),
                    up_ptr.add(valid_count * up_half_bytes),
                    up_half_bytes,
                );
            }

            let dn_dst = scratch.down_bufs[valid_count].contents() as *mut u8;
            let copy_len = dn_bytes.len().min(down_expert_bytes);
            unsafe {
                std::ptr::copy_nonoverlapping(dn_bytes.as_ptr(), dn_dst, copy_len);
                if copy_len < down_expert_bytes {
                    std::ptr::write_bytes(dn_dst.add(copy_len), 0, down_expert_bytes - copy_len);
                }
            }

            valid_weights.push(expert_weights[k]);
            valid_count += 1;
        }

        if valid_count == 0 {
            return vec![0.0f32; hidden];
        }

        // ── Stage h_norm into pre-allocated x_buf ─────────────────────────
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(h_norm.as_ptr(), x_ptr, hidden);
        }
        let t_stage = t_start.elapsed();

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // q4k_ffn_gate_up over all valid_count experts at once.
        // Geometry travels with the `KernelHandle` (see `decode_hybrid.rs`
        // ship-log entry on this bug class).
        let gate_up_kh = &self.ffn.q4k_ffn_gate_up_pipeline;
        let n_rows = (valid_count * inter) as u32;
        let k_cols = hidden as u32;
        let tgs = (valid_count as u64 * inter as u64).div_ceil(gate_up_kh.rows_per_tg);

        enc.set_compute_pipeline_state(&gate_up_kh.state);
        enc.set_buffer(0, Some(&scratch.gate_buf), 0);
        enc.set_buffer(1, Some(&scratch.up_buf), 0);
        enc.set_buffer(2, Some(&scratch.x_buf), 0);
        enc.set_buffer(3, Some(&scratch.g_out), 0);
        enc.set_buffer(4, Some(&scratch.u_out), 0);
        enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(tgs * 2, 1, 1),
            MTLSize::new(gate_up_kh.threads_per_tg, 1, 1),
        );

        // GELU-tanh activation per expert (strided to inter_padded).
        let inter_u32 = inter as u32;
        for e in 0..valid_count {
            let g_offset = (e * inter * 4) as u64;
            let u_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            enc.set_compute_pipeline_state(&self.ffn.geglu_gelu_tanh_pipeline);
            enc.set_buffer(0, Some(&scratch.g_out), g_offset);
            enc.set_buffer(1, Some(&scratch.u_out), u_offset);
            enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
            enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
            enc.dispatch_threads(
                MTLSize::new(inter as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(inter as u64),
                    1,
                    1,
                ),
            );
        }

        // Down projection per expert.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        // Pull dispatch geometry from the bound pipeline so this works for
        // both the 4sg and 8sg variants of `q4k_matvec` — hardcoding the
        // 4sg constants while dispatching the 8sg pipeline (the production
        // default since 2026-04-28) leaves simdgroups 4..7 unscheduled and
        // only writes rows 0..3 of each TG's 8-row range. See the matching
        // fix in `trait_impl/quant_matvec.rs::q4k_matvec`.
        let down_rows_per_tg = self.quant.q4k_matvec_pipeline.rows_per_tg;
        let down_threads_per_tg = self.quant.q4k_matvec_pipeline.threads_per_tg;
        let down_tgs = (hidden as u64).div_ceil(down_rows_per_tg);

        for e in 0..valid_count {
            let act_offset = (e * inter_padded * 4) as u64;
            let out_offset = (e * hidden * 4) as u64;
            enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
            enc.set_buffer(0, Some(&scratch.down_bufs[e]), 0);
            enc.set_buffer(1, Some(&scratch.act_buf), act_offset);
            enc.set_buffer(2, Some(&scratch.expert_outs), out_offset);
            enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(down_tgs, 1, 1),
                MTLSize::new(down_threads_per_tg, 1, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let t_gpu = t_start.elapsed();

        // CPU weighted sum (no post-experts norm — client does that across shards).
        let all_expert_outputs = read_buffer_f32(&scratch.expert_outs, valid_count * hidden);
        let mut moe_out = vec![0.0f32; hidden];
        for e in 0..valid_count {
            let w = valid_weights[e];
            let out_slice = &all_expert_outputs[e * hidden..(e + 1) * hidden];
            for (acc, &v) in moe_out.iter_mut().zip(out_slice) {
                *acc += v * w;
            }
        }
        let t_total = t_start.elapsed();
        if timing_enabled {
            eprintln!(
                "[run_experts_metal] K={valid_count} stage={:.2}ms gpu={:.2}ms readback+sum={:.2}ms total={:.2}ms",
                t_stage.as_secs_f32() * 1000.0,
                (t_gpu - t_stage).as_secs_f32() * 1000.0,
                (t_total - t_gpu).as_secs_f32() * 1000.0,
                t_total.as_secs_f32() * 1000.0,
            );
        }
        moe_out
    }

    /// Run one dense (non-MoE) FFN layer on GPU using pre-loaded Q4K weight buffers.
    ///
    /// `h_norm` is the f32 FFN-input norm output, length = `hidden`.
    /// Gate and up projections run via `q4k_ffn_gate_up_8sg_pipeline`;
    /// activation via `geglu_gelu_tanh_pipeline`; down via `q4k_matvec_pipeline`.
    ///
    /// All three weight buffers must be pre-created from the mmap byte slices via
    /// `BufferCache::get_bytes` (zero-copy for page-aligned mmap data).
    ///
    /// Returns `Vec<f32>` of length `hidden` — the FFN delta (no residual add).
    #[allow(clippy::too_many_arguments)]
    pub fn run_dense_ffn_q4k(
        &self,
        h_norm: &[f32],
        gate_buf: &Buffer,
        up_buf: &Buffer,
        down_buf: &Buffer,
        hidden: usize,
        inter: usize,
        inter_padded: usize,
    ) -> Vec<f32> {
        if hidden == 0 || inter == 0 {
            return vec![0.0f32; hidden];
        }

        // Stage h_norm into a transient f32 buffer.
        let x_buf = self.bufs.transient_from_f32(h_norm);

        // Allocate scratch buffers.
        let gate_out = self.bufs.output((inter * 4) as u64);
        let up_out = self.bufs.output((inter * 4) as u64);
        let act_buf = self.bufs.output((inter_padded * 4) as u64);
        let out_buf = self.bufs.output((hidden * 4) as u64);

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // 1. q4k_ffn_gate_up_8sg — gate and up projections.
        // Geometry pulled from the bound `KernelHandle` so a future
        // pipeline swap can't drift from the dispatched TG shape.
        let gate_up_kh = &self.ffn.q4k_ffn_gate_up_8sg_pipeline;
        let n_rows = inter as u32;
        let k_cols = hidden as u32;
        let n_tgs = (inter as u64).div_ceil(gate_up_kh.rows_per_tg);
        enc.set_compute_pipeline_state(&gate_up_kh.state);
        enc.set_buffer(0, Some(gate_buf), 0);
        enc.set_buffer(1, Some(up_buf), 0);
        enc.set_buffer(2, Some(&x_buf), 0);
        enc.set_buffer(3, Some(&gate_out), 0);
        enc.set_buffer(4, Some(&up_out), 0);
        enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(n_tgs * 2, 1, 1),
            MTLSize::new(gate_up_kh.threads_per_tg, 1, 1),
        );

        // 2. geglu_gelu_tanh activation.
        let inter_u32 = inter as u32;
        enc.set_compute_pipeline_state(&self.ffn.geglu_gelu_tanh_pipeline);
        enc.set_buffer(0, Some(&gate_out), 0);
        enc.set_buffer(1, Some(&up_out), 0);
        enc.set_buffer(2, Some(&act_buf), 0);
        enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new(inter as u64, 1, 1),
            MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(inter as u64),
                1,
                1,
            ),
        );

        // 3. q4k_matvec down projection.
        // Pull dispatch geometry from the bound pipeline (not hardcoded) to avoid
        // the 4sg-vs-8sg dispatch geometry mismatch bug documented in ROADMAP.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        let down_rows_per_tg = self.quant.q4k_matvec_pipeline.rows_per_tg;
        let down_threads_per_tg = self.quant.q4k_matvec_pipeline.threads_per_tg;
        let down_tgs = (hidden as u64).div_ceil(down_rows_per_tg);
        enc.set_compute_pipeline_state(&self.quant.q4k_matvec_pipeline.state);
        enc.set_buffer(0, Some(down_buf), 0);
        enc.set_buffer(1, Some(&act_buf), 0);
        enc.set_buffer(2, Some(&out_buf), 0);
        enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(down_tgs, 1, 1),
            MTLSize::new(down_threads_per_tg, 1, 1),
        );

        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let result = read_buffer_f32(&out_buf, hidden);

        // Recycle scratch buffers back to the pool.
        self.bufs.recycle(gate_out);
        self.bufs.recycle(up_out);
        self.bufs.recycle(act_buf);
        self.bufs.recycle(out_buf);

        result
    }

    pub(super) fn gpu_moe_dispatch_with_scratch<'w, F>(
        &self,
        h_post_attn: &[f32],
        moe: &MoeLayerWeights<'_>,
        eps: f32,
        scratch: &MoeScratch,
        get_expert_bytes: F,
    ) -> Vec<f32>
    where
        F: Fn(usize) -> Option<(&'w [u8], &'w [u8])>,
    {
        let hidden = h_post_attn.len();
        let inter = moe.intermediate_size;
        let inter_padded = scratch.inter_padded;
        let top_k = moe.top_k;
        // ClampedGlu (with its biases) has a dedicated activation kernel
        // below. What has NO kernel is a Gated layer with expert biases —
        // no current model ships that combination, and running it bias-less
        // would be a plausible-looking wrong forward. Refuse it loudly.
        assert!(
            matches!(moe.gate_rule, larql_compute::MoeGateRule::ClampedGlu { .. })
                || (moe.experts_gate_up_bias.is_empty() && moe.experts_down_bias.is_empty()),
            "Metal MoE dispatch has no biased-Gated expert kernel; this \
             layer's biased experts must run on the CPU path"
        );
        debug_assert_eq!(
            moe.gate_up_cols(hidden),
            scratch.weight_cols,
            "MoE scratch was sized for a different gate/up row width"
        );
        debug_assert_eq!(moe.expert_data_format, scratch.format);
        debug_assert_eq!(top_k, scratch.top_k, "MoE top_k drift across layers");
        debug_assert_eq!(
            inter, scratch.inter,
            "MoE intermediate_size drift across layers"
        );
        debug_assert_eq!(
            hidden, scratch.hidden,
            "MoE hidden_size drift across layers"
        );

        // ── 1. CPU expert input + router ─────────────────────────────────
        // The policy comes from `MoeLayerWeights`, so Gemma's
        // pre_experts_norm routing convention is no longer implicit here.
        let expert_input = moe_expert_input(h_post_attn, moe, 0.0, eps);
        let router_in = moe_router_input(h_post_attn, &expert_input, moe, 0.0, eps);
        let (expert_indices, expert_weights) = moe_route_from_router_input(&router_in, moe);

        // ── 2. Stage expert weight bytes into pre-allocated Metal buffers ─
        let row_bytes = scratch.row_bytes;
        let gate_half_bytes = inter * row_bytes;
        let up_half_bytes = inter * row_bytes;
        let down_expert_bytes = hidden * scratch.down_row_bytes;

        // ── 1.5 Zero-copy fast path ──────────────────────────────────────
        // When every selected expert's byte slices resolve inside a
        // registered mmap region (`register_weight_region` at engine
        // setup), bind them as region offsets and skip the staging
        // memcpys entirely — they are ~2 GB/token of CPU bandwidth on
        // GPT-OSS-class models. Any miss (unregistered region, short
        // slice) falls through to the staged path below unchanged.
        if let Some(resolved) =
            self.resolve_selected_experts(scratch, &expert_indices, &expert_weights, |ei| {
                get_expert_bytes(ei)
            })
        {
            return self.dispatch_experts_zero_copy(&expert_input, moe, eps, scratch, &resolved);
        }

        let gate_ptr = scratch.gate_buf.contents() as *mut u8;
        let up_ptr = scratch.up_buf.contents() as *mut u8;

        let mut valid_weights: Vec<f32> = Vec::with_capacity(top_k);
        let mut valid_ids: Vec<usize> = Vec::with_capacity(top_k);
        let mut valid_count = 0usize;
        let stage_biases = !moe.experts_gate_up_bias.is_empty();

        for (k, &ei) in expert_indices.iter().enumerate() {
            // Scratch is sized for `scratch.top_k` experts; without this
            // guard a caller passing more indices than the scratch was
            // built for writes past `gate_buf`/`up_buf` (and would only
            // panic later on the `down_bufs` index). Bound against the
            // ALLOCATION, not `moe.top_k` — the two are only
            // debug_assert-ed equal, so in release they can drift. The
            // sibling staging loop in `run_experts_preselected_metal`
            // has carried this bound all along. Capability audit F10.
            if valid_count >= scratch.top_k {
                break;
            }
            let Some((gu_bytes, dn_bytes)) = get_expert_bytes(ei) else {
                continue;
            };
            if gu_bytes.len() < 2 * gate_half_bytes {
                continue;
            }

            // Q4_K layout: gate || up, each `inter * row_bytes` bytes.
            // SAFETY: gate_ptr / up_ptr are StorageModeShared Metal buffer
            // contents; offsets are bounded by `top_k * gate_half_bytes`
            // allocated up front (see `MoeScratch::new`), and the
            // `valid_count >= top_k` guard above enforces that bound.
            // Writes complete before the encoder dispatches the matvec
            // that reads them.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    gu_bytes.as_ptr(),
                    gate_ptr.add(valid_count * gate_half_bytes),
                    gate_half_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    gu_bytes.as_ptr().add(gate_half_bytes),
                    up_ptr.add(valid_count * up_half_bytes),
                    up_half_bytes,
                );
            }

            let dn_dst = scratch.down_bufs[valid_count].contents() as *mut u8;
            let copy_len = dn_bytes.len().min(down_expert_bytes);
            unsafe {
                std::ptr::copy_nonoverlapping(dn_bytes.as_ptr(), dn_dst, copy_len);
                if copy_len < down_expert_bytes {
                    std::ptr::write_bytes(dn_dst.add(copy_len), 0, down_expert_bytes - copy_len);
                }
            }

            // De-interleave this expert's fused gate/up bias row into the
            // staging buffers at the same slot the weights landed in, so
            // the activation kernel reads slot-aligned bias vectors. The
            // interleave convention is owned by `ExpertMlp::{gate,up}_bias`
            // (FusedHalf), not re-derived here.
            if stage_biases {
                let mlp = moe.expert_mlp(ei);
                unsafe {
                    let gb =
                        (scratch.gate_bias_buf.contents() as *mut f32).add(valid_count * inter);
                    let ub = (scratch.up_bias_buf.contents() as *mut f32).add(valid_count * inter);
                    for j in 0..inter {
                        *gb.add(j) = mlp.gate_bias(j);
                        *ub.add(j) = mlp.up_bias(j);
                    }
                }
            }

            valid_weights.push(expert_weights[k]);
            valid_ids.push(ei);
            valid_count += 1;
        }

        if valid_count == 0 {
            return vec![0.0f32; hidden];
        }

        // ── 3. Stage router-normed input into pre-allocated x_buf ─────────
        // The buffer is `weight_cols` wide with a permanently-zero tail, so
        // the writer's row padding contributes nothing to any dot product.
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(expert_input.as_ptr(), x_ptr, hidden);
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        // ── 4. Gate + up matvecs over all valid_count experts at once ────
        // Both staging buffers are one uniform-stride matrix of
        // `valid_count × inter` rows at `weight_cols` columns, so a single
        // dispatch (per projection) covers every selected expert.
        let n_rows = (valid_count * inter) as u32;
        let k_cols = scratch.weight_cols as u32;
        match scratch.format {
            larql_compute::QuantFormat::Q6_K => {
                // Two plain q6k_matvec dispatches. A fused Q6_K gate+up twin
                // of `q4k_ffn_gate_up` does not exist yet; at these shapes
                // the second dispatch costs ~µs against ~44 MB of weight
                // reads, so fusion is a tuning item, not a correctness one.
                let kh = &self.quant.q6k_matvec_pipeline;
                let tgs = (valid_count as u64 * inter as u64).div_ceil(kh.rows_per_tg);
                for (w_buf, out_buf) in [
                    (&scratch.gate_buf, &scratch.g_out),
                    (&scratch.up_buf, &scratch.u_out),
                ] {
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(w_buf), 0);
                    enc.set_buffer(1, Some(&scratch.x_buf), 0);
                    enc.set_buffer(2, Some(out_buf), 0);
                    enc.set_bytes(3, 4, &n_rows as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &k_cols as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(tgs, 1, 1),
                        MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                }
            }
            _ => {
                // Q4_K: the fused gate+up kernel. Geometry travels with the
                // `KernelHandle`; no shader-module ROWS_PER_TG re-imports.
                let gate_up_kh = &self.ffn.q4k_ffn_gate_up_pipeline;
                let tgs = (valid_count as u64 * inter as u64).div_ceil(gate_up_kh.rows_per_tg);
                enc.set_compute_pipeline_state(&gate_up_kh.state);
                enc.set_buffer(0, Some(&scratch.gate_buf), 0);
                enc.set_buffer(1, Some(&scratch.up_buf), 0);
                enc.set_buffer(2, Some(&scratch.x_buf), 0);
                enc.set_buffer(3, Some(&scratch.g_out), 0);
                enc.set_buffer(4, Some(&scratch.u_out), 0);
                enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(tgs * 2, 1, 1),
                    MTLSize::new(gate_up_kh.threads_per_tg, 1, 1),
                );
            }
        }

        // ── 5. Gate-rule activation per expert (strided to inter_padded) ──
        // Gate/up output is packed at stride `inter`; activation must land at
        // stride `inter_padded` because down reads `K = inter_padded`. One
        // small dispatch per expert with the right offsets gets us strided
        // output without a new shader. valid_count × ~5µs ≪ allocation cost.
        //
        // The kernel comes from the layer's typed `MoeGateRule` — the same
        // authority the CPU paths combine under — so a Silu model can no
        // longer silently run the GELU kernel, and ClampedGlu (GPT-OSS)
        // gets its own kernel with the bias slots bound.
        let inter_u32 = inter as u32;
        for e in 0..valid_count {
            let g_offset = (e * inter * 4) as u64;
            let u_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            match moe.gate_rule {
                larql_compute::MoeGateRule::ClampedGlu { limit, alpha } => {
                    let has_bias: u32 = u32::from(stage_biases);
                    let b_offset = (e * inter * 4) as u64;
                    enc.set_compute_pipeline_state(&self.ffn.clamped_glu_bias_pipeline);
                    enc.set_buffer(0, Some(&scratch.g_out), g_offset);
                    enc.set_buffer(1, Some(&scratch.u_out), u_offset);
                    enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
                    enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
                    enc.set_buffer(4, Some(&scratch.gate_bias_buf), b_offset);
                    enc.set_buffer(5, Some(&scratch.up_bias_buf), b_offset);
                    enc.set_bytes(6, 4, &has_bias as *const u32 as *const c_void);
                    enc.set_bytes(7, 4, &limit as *const f32 as *const c_void);
                    enc.set_bytes(8, 4, &alpha as *const f32 as *const c_void);
                }
                larql_compute::MoeGateRule::Gated(activation) => {
                    let pipeline = if activation.gate_up_is_gelu_tanh() {
                        &self.ffn.geglu_gelu_tanh_pipeline
                    } else {
                        &self.ffn.geglu_pipeline
                    };
                    enc.set_compute_pipeline_state(pipeline);
                    enc.set_buffer(0, Some(&scratch.g_out), g_offset);
                    enc.set_buffer(1, Some(&scratch.u_out), u_offset);
                    enc.set_buffer(2, Some(&scratch.act_buf), a_offset);
                    enc.set_bytes(3, 4, &inter_u32 as *const u32 as *const c_void);
                }
            }
            enc.dispatch_threads(
                MTLSize::new(inter as u64, 1, 1),
                MTLSize::new(
                    crate::kernels::DISPATCH_TG_MAX_THREADS.min(inter as u64),
                    1,
                    1,
                ),
            );
        }

        // ── 6. Down projection per expert ────────────────────────────────
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        // Pull dispatch geometry from the bound pipeline so this works for
        // both the 4sg and 8sg variants of the matvec — hardcoding the
        // 4sg constants while dispatching the 8sg pipeline (the production
        // default since 2026-04-28) leaves simdgroups 4..7 unscheduled and
        // only writes rows 0..3 of each TG's 8-row range. See the matching
        // fix in `trait_impl/quant_matvec.rs::q4k_matvec`.
        let down_kh = match scratch.format {
            larql_compute::QuantFormat::Q6_K => &self.quant.q6k_matvec_pipeline,
            _ => &self.quant.q4k_matvec_pipeline,
        };
        let down_tgs = (hidden as u64).div_ceil(down_kh.rows_per_tg);

        for e in 0..valid_count {
            let act_offset = (e * inter_padded * 4) as u64;
            let out_offset = (e * hidden * 4) as u64;
            enc.set_compute_pipeline_state(&down_kh.state);
            enc.set_buffer(0, Some(&scratch.down_bufs[e]), 0);
            enc.set_buffer(1, Some(&scratch.act_buf), act_offset);
            enc.set_buffer(2, Some(&scratch.expert_outs), out_offset);
            enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(down_tgs, 1, 1),
                MTLSize::new(down_kh.threads_per_tg, 1, 1),
            );
        }
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // ── 7. CPU weighted sum + post-experts norm ──────────────────────
        // The down bias joins each expert's output BEFORE the routing
        // weight, matching the reference (`ExpertWeightFfn::run_expert`
        // adds it inside the expert, then the caller weights). Applied here
        // on readback rather than in a shader: it is `hidden` adds per
        // expert against ~22 MB of weight reads.
        let all_expert_outputs = read_buffer_f32(&scratch.expert_outs, valid_count * hidden);
        let mut moe_out = vec![0.0f32; hidden];
        for e in 0..valid_count {
            let w = valid_weights[e];
            let mlp = moe.expert_mlp(valid_ids[e]);
            let out_slice = &all_expert_outputs[e * hidden..(e + 1) * hidden];
            if mlp.down_bias.is_empty() {
                for (acc, &v) in moe_out.iter_mut().zip(out_slice) {
                    *acc += v * w;
                }
            } else {
                for ((acc, &v), &b) in moe_out.iter_mut().zip(out_slice).zip(mlp.down_bias) {
                    *acc += (v + b) * w;
                }
            }
        }

        moe_post_expert_output(&moe_out, moe, 0.0, eps)
    }
}
