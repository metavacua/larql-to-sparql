//! The Q4_K MoE decode entry point and its per-layer expert hook.
//!
//! Split out of `moe_dispatch.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use crate::MetalBackend;
use larql_compute::MoeLayerWeights;
use metal::*;

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
        head: Option<crate::decode::HeadRequest<'_, '_>>,
    ) -> Option<Vec<f32>>
    where
        F: Fn(usize, usize) -> Option<(&'w [u8], &'w [u8])>,
    {
        let mut kv_guard = self.kv_cache.lock().unwrap();
        let kv = self.ensure_kv_cache_for_layers(
            &mut kv_guard,
            layers,
            crate::decode::DEFAULT_KV_CACHE_MAX_SEQ,
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
                // The served MoE seam carries its own layer coordinate, so
                // this is where routing becomes attributable. Nested inside
                // any outer per-layer scope, and restored on drop.
                let _route_scope = larql_compute::moe_route_observe::LayerScope::new(layer_idx);
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
                    &|expert_idx| get_expert_ref(layer_idx, expert_idx),
                )
            }
        };

        // Merged-CB context: per-layer preconditions permitting, the
        // interleave routes on CPU and encodes experts + combine into the
        // command buffer the next layer's attention rides — the moe_fn
        // closure above stays as the per-layer fallback.
        let inline_ctx = scratch_ref.map(|s| crate::decode::InlineMoeCtx::new(s, norm_eps));

        // A step whose command buffer failed (GPU address fault, ignored
        // buffer after an earlier fault) returns from its wait exactly like
        // a healthy one, with stale or garbage bytes in `h_buf`. Count the
        // failures across the whole step — every wait on this path goes
        // through `cb_status::wait_checked` — and refuse to hand a poisoned
        // hidden state to the sampler. The caller treats `None` as
        // "GPU decode returned None; stopping generation" (#229).
        let failures_before = crate::cb_status::non_completed_count();
        let h = self.decode_token_with_moe_split_fn(
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
            head,
        );
        if crate::cb_status::non_completed_count() != failures_before {
            return None;
        }
        // The gather kernel clamps router ids past `num_experts` and
        // counts them; a step that needed the clamp computed with the wrong
        // expert and must not be sampled from either.
        let refused = self.route_guard.take_new_refusals();
        if refused != 0 {
            eprintln!(
                "[metal] router emitted {refused} expert id(s) >= num_experts in this decode step; refusing the step (#229)"
            );
            return None;
        }
        Some(h)
    }

    /// Public single-layer MoE block entry — the parity/test surface for
    /// [`Self::gpu_moe_dispatch_with_scratch`], which stays `pub(crate)`.
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
        self.gpu_moe_dispatch_with_scratch(h_post_attn, moe, eps, scratch, &get_expert_bytes)
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
}
