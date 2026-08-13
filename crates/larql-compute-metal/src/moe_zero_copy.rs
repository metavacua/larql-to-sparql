//! Zero-copy MoE expert dispatch — experts bound as offsets into
//! registered mmap regions.
//!
//! The staged path in `moe_dispatch.rs` memcpys every selected expert's
//! bytes into scratch each layer each token — GPT-OSS decode: top-4 ×
//! ~22 MB × 24 layers ≈ 2.1 GB of CPU memcpy per token, the dominant
//! decode cost once attention moved to the GPU. When every selected
//! expert's byte slices resolve inside a registered region
//! (`BufferCache::register_region` over each layer-weights mmap), this
//! path binds `(region_buffer, byte_offset)` per expert instead: no
//! staging, no duplication, the GPU reads the mmap pages directly
//! through unified memory.
//!
//! Dispatch shape follows `run_experts_prestaged_metal` (per-expert
//! dispatches — separate buffers preclude the staged path's single
//! all-K matvec) with the staged path's format awareness kept: per-format
//! matvec selection (Q4_K fused gate+up / Q6_K paired matvecs), the
//! layer's typed `MoeGateRule` activation, ClampedGlu gate/up biases, and
//! the down bias + routing weight + post-experts norm on readback.

use metal::*;
use std::ffi::c_void;

use super::buffers::read_buffer_f32;
use super::moe_dispatch::MoeScratch;
use super::MetalBackend;
use larql_compute::cpu::ops::moe::moe_post_expert_output;
use larql_compute::MoeLayerWeights;

/// One selected expert resolved to zero-copy bindings.
pub(super) struct ResolvedExpert {
    pub gate_up: (Buffer, u64),
    pub down: (Buffer, u64),
    pub expert_id: usize,
    pub weight: f32,
}

impl MetalBackend {
    /// Resolve the router's selected experts to zero-copy region bindings.
    ///
    /// Returns `Some(resolved)` only when EVERY selected expert's byte
    /// slices lie inside registered regions AND hold their full extents
    /// (a short down slice would make the GPU read the next expert's
    /// bytes where the staged path zero-pads); any miss returns `None`
    /// and the caller stays on its staged/callback path.
    pub(super) fn resolve_selected_experts<'w>(
        &self,
        scratch: &MoeScratch,
        expert_indices: &[usize],
        expert_weights: &[f32],
        get_expert_bytes: impl Fn(usize) -> Option<(&'w [u8], &'w [u8])>,
    ) -> Option<Vec<ResolvedExpert>> {
        let gate_half_bytes = scratch.inter * scratch.row_bytes;
        let down_expert_bytes = scratch.hidden * scratch.down_row_bytes;
        let mut resolved = Vec::with_capacity(scratch.top_k);
        for (k, &ei) in expert_indices.iter().enumerate() {
            if resolved.len() >= scratch.top_k {
                break;
            }
            let (gu_bytes, dn_bytes) = get_expert_bytes(ei)?;
            if gu_bytes.len() < 2 * gate_half_bytes || dn_bytes.len() < down_expert_bytes {
                return None;
            }
            let gate_up = self.bufs.resolve_region(gu_bytes)?;
            let down = self.bufs.resolve_region(dn_bytes)?;
            resolved.push(ResolvedExpert {
                gate_up,
                down,
                expert_id: ei,
                weight: expert_weights[k],
            });
        }
        if resolved.is_empty() {
            return None;
        }
        Some(resolved)
    }

    /// Encode the expert dispatches (bias/x staging + gate/up +
    /// activation + down) into `enc` — no command-buffer lifecycle, no
    /// readback. Shared by the commit-and-CPU-combine wrapper below and
    /// the merged-CB path (`encode_experts_and_combine_zero_copy`).
    ///
    /// Caller guarantees:
    /// - `resolved.len() <= scratch.top_k` (output offsets are bounded by
    ///   the scratch allocation),
    /// - every `gate_up` slice held ≥ `2 × inter × row_bytes` bytes and
    ///   every `down` slice ≥ `hidden × down_row_bytes` at resolution time
    ///   (offsets stay in-bounds of the registered region's buffer),
    /// - the biased-Gated refusal already ran (shared assert at the top of
    ///   `gpu_moe_dispatch_with_scratch`).
    pub(super) fn encode_experts_zero_copy(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        expert_input: &[f32],
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
        resolved: &[ResolvedExpert],
    ) {
        let hidden = scratch.hidden;
        let inter = scratch.inter;
        let inter_padded = scratch.inter_padded;
        let valid_count = resolved.len();
        debug_assert!(valid_count <= scratch.top_k);

        // ── ClampedGlu gate/up biases: slot-aligned staging (small — the
        // weights stay zero-copy; the bias rows are `inter` f32 each).
        let stage_biases = !moe.experts_gate_up_bias.is_empty();
        if stage_biases {
            for (slot, r) in resolved.iter().enumerate() {
                let mlp = moe.expert_mlp(r.expert_id);
                // SAFETY: shared-storage scratch buffers allocated at
                // `top_k × inter` f32; `slot < valid_count <= top_k`.
                unsafe {
                    let gb = (scratch.gate_bias_buf.contents() as *mut f32).add(slot * inter);
                    let ub = (scratch.up_bias_buf.contents() as *mut f32).add(slot * inter);
                    for j in 0..inter {
                        *gb.add(j) = mlp.gate_bias(j);
                        *ub.add(j) = mlp.up_bias(j);
                    }
                }
            }
        }

        // ── Router-policy input into the pre-allocated x_buf (its
        // `weight_cols` tail is permanently zero — writer row padding
        // contributes nothing to any dot product).
        // SAFETY: shared-storage buffer sized `weight_cols ≥ hidden` f32.
        unsafe {
            let x_ptr = scratch.x_buf.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(expert_input.as_ptr(), x_ptr, hidden);
        }

        // ── Gate + up per expert, at the expert's region offset.
        let gate_half_bytes = (inter * scratch.row_bytes) as u64;
        let n_rows = inter as u32;
        let k_cols = scratch.weight_cols as u32;
        // Grouped dispatch wants ONE base buffer + a u32 byte-offset table.
        // Zero-copy resolution guarantees it per layer in practice (every
        // expert of a layer lives in that layer's mmap → one region
        // buffer); cross-buffer or >4 GiB offsets fall back to per-expert
        // dispatches, which are exact but occupancy-poor (η ≈ 0.64 at the
        // expert shape — the K3a measurement the grouped kernel exists for).
        let single_base = |extract: fn(&ResolvedExpert) -> &(Buffer, u64)| -> bool {
            resolved
                .windows(2)
                .all(|w| extract(&w[0]).0.gpu_address() == extract(&w[1]).0.gpu_address())
                && resolved.iter().all(|r| u32::try_from(extract(r).1).is_ok())
        };
        match scratch.format {
            larql_compute::QuantFormat::Q6_K if single_base(|r| &r.gate_up) => {
                // K3a grouped kernel: all selected experts' gate rows in one
                // 2-D dispatch (row tiles × slots), then the same for up.
                // Reduction body is byte-identical to `q6k_matvec`, so
                // outputs match the per-expert form exactly.
                let kh = &self.quant.q6k_grouped_experts_pipeline;
                let base = &resolved[0].gate_up.0;
                let row_tiles = (inter as u64).div_ceil(kh.rows_per_tg);
                let xstride_shared: u32 = 0;
                for half in [0u64, 1] {
                    let offsets: Vec<u32> = resolved
                        .iter()
                        .map(|r| (r.gate_up.1 + half * gate_half_bytes) as u32)
                        .collect();
                    let out_buf = if half == 0 {
                        &scratch.g_out
                    } else {
                        &scratch.u_out
                    };
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(base), 0);
                    enc.set_bytes(
                        1,
                        (offsets.len() * 4) as u64,
                        offsets.as_ptr() as *const c_void,
                    );
                    enc.set_buffer(2, Some(&scratch.x_buf), 0);
                    enc.set_buffer(3, Some(out_buf), 0);
                    enc.set_bytes(4, 4, &n_rows as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &k_cols as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &xstride_shared as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(row_tiles, valid_count as u64, 1),
                        MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                }
            }
            larql_compute::QuantFormat::Q6_K => {
                let kh = &self.quant.q6k_matvec_pipeline;
                let tgs = (inter as u64).div_ceil(kh.rows_per_tg);
                for (e, r) in resolved.iter().enumerate() {
                    let (buf, off) = &r.gate_up;
                    for (half, out_buf) in [(0u64, &scratch.g_out), (1, &scratch.u_out)] {
                        enc.set_compute_pipeline_state(&kh.state);
                        enc.set_buffer(0, Some(buf), off + half * gate_half_bytes);
                        enc.set_buffer(1, Some(&scratch.x_buf), 0);
                        enc.set_buffer(2, Some(out_buf), (e * inter * 4) as u64);
                        enc.set_bytes(3, 4, &n_rows as *const u32 as *const c_void);
                        enc.set_bytes(4, 4, &k_cols as *const u32 as *const c_void);
                        enc.dispatch_thread_groups(
                            MTLSize::new(tgs, 1, 1),
                            MTLSize::new(kh.threads_per_tg, 1, 1),
                        );
                    }
                }
            }
            _ if single_base(|r| &r.gate_up) => {
                // Q4_K grouped sibling: same offset-table interface, gate
                // rows for every selected expert in one 2-D dispatch, then
                // up. Reduction body is byte-identical to `q4k_matvec`.
                let kh = &self.quant.q4k_grouped_experts_pipeline;
                let base = &resolved[0].gate_up.0;
                let row_tiles = (inter as u64).div_ceil(kh.rows_per_tg);
                let xstride_shared: u32 = 0;
                for half in [0u64, 1] {
                    let offsets: Vec<u32> = resolved
                        .iter()
                        .map(|r| (r.gate_up.1 + half * gate_half_bytes) as u32)
                        .collect();
                    let out_buf = if half == 0 {
                        &scratch.g_out
                    } else {
                        &scratch.u_out
                    };
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(base), 0);
                    enc.set_bytes(
                        1,
                        (offsets.len() * 4) as u64,
                        offsets.as_ptr() as *const c_void,
                    );
                    enc.set_buffer(2, Some(&scratch.x_buf), 0);
                    enc.set_buffer(3, Some(out_buf), 0);
                    enc.set_bytes(4, 4, &n_rows as *const u32 as *const c_void);
                    enc.set_bytes(5, 4, &k_cols as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &xstride_shared as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(row_tiles, valid_count as u64, 1),
                        MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                }
            }
            _ => {
                // Q4_K family: fused gate+up kernel, gate at the expert
                // offset, up one gate-half further into the same buffer.
                let kh = &self.ffn.q4k_ffn_gate_up_pipeline;
                let tgs = (inter as u64).div_ceil(kh.rows_per_tg);
                for (e, r) in resolved.iter().enumerate() {
                    let (buf, off) = &r.gate_up;
                    enc.set_compute_pipeline_state(&kh.state);
                    enc.set_buffer(0, Some(buf), *off);
                    enc.set_buffer(1, Some(buf), off + gate_half_bytes);
                    enc.set_buffer(2, Some(&scratch.x_buf), 0);
                    enc.set_buffer(3, Some(&scratch.g_out), (e * inter * 4) as u64);
                    enc.set_buffer(4, Some(&scratch.u_out), (e * inter * 4) as u64);
                    enc.set_bytes(5, 4, &n_rows as *const u32 as *const c_void);
                    enc.set_bytes(6, 4, &k_cols as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(
                        MTLSize::new(tgs * 2, 1, 1),
                        MTLSize::new(kh.threads_per_tg, 1, 1),
                    );
                }
            }
        }

        // ── Typed gate-rule activation per expert (strided to
        // inter_padded so down's `K = inter_padded` reads see zeros).
        let inter_u32 = inter as u32;
        for e in 0..valid_count {
            let g_offset = (e * inter * 4) as u64;
            let a_offset = (e * inter_padded * 4) as u64;
            match moe.gate_rule {
                larql_compute::MoeGateRule::ClampedGlu { limit, alpha } => {
                    let has_bias: u32 = u32::from(stage_biases);
                    let b_offset = (e * inter * 4) as u64;
                    enc.set_compute_pipeline_state(&self.ffn.clamped_glu_bias_pipeline);
                    enc.set_buffer(0, Some(&scratch.g_out), g_offset);
                    enc.set_buffer(1, Some(&scratch.u_out), g_offset);
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
                    enc.set_buffer(1, Some(&scratch.u_out), g_offset);
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

        // ── Down projection at the experts' region offsets.
        let n_out = hidden as u32;
        let k_in = inter_padded as u32;
        if single_base(|r| &r.down) {
            // Grouped down: each slot reads its OWN activation
            // (`XSTRIDE = inter_padded` — the strided act_buf layout the
            // activation stage wrote above).
            let kh = match scratch.format {
                larql_compute::QuantFormat::Q6_K => &self.quant.q6k_grouped_experts_pipeline,
                _ => &self.quant.q4k_grouped_experts_pipeline,
            };
            let row_tiles = (hidden as u64).div_ceil(kh.rows_per_tg);
            let base = &resolved[0].down.0;
            let offsets: Vec<u32> = resolved.iter().map(|r| r.down.1 as u32).collect();
            let xstride_own: u32 = inter_padded as u32;
            enc.set_compute_pipeline_state(&kh.state);
            enc.set_buffer(0, Some(base), 0);
            enc.set_bytes(
                1,
                (offsets.len() * 4) as u64,
                offsets.as_ptr() as *const c_void,
            );
            enc.set_buffer(2, Some(&scratch.act_buf), 0);
            enc.set_buffer(3, Some(&scratch.expert_outs), 0);
            enc.set_bytes(4, 4, &n_out as *const u32 as *const c_void);
            enc.set_bytes(5, 4, &k_in as *const u32 as *const c_void);
            enc.set_bytes(6, 4, &xstride_own as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(row_tiles, valid_count as u64, 1),
                MTLSize::new(kh.threads_per_tg, 1, 1),
            );
        } else {
            let down_kh = match scratch.format {
                larql_compute::QuantFormat::Q6_K => &self.quant.q6k_matvec_pipeline,
                _ => &self.quant.q4k_matvec_pipeline,
            };
            let down_tgs = (hidden as u64).div_ceil(down_kh.rows_per_tg);
            for (e, r) in resolved.iter().enumerate() {
                let (buf, off) = &r.down;
                let act_offset = (e * inter_padded * 4) as u64;
                let out_offset = (e * hidden * 4) as u64;
                enc.set_compute_pipeline_state(&down_kh.state);
                enc.set_buffer(0, Some(buf), *off);
                enc.set_buffer(1, Some(&scratch.act_buf), act_offset);
                enc.set_buffer(2, Some(&scratch.expert_outs), out_offset);
                enc.set_bytes(3, 4, &n_out as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &k_in as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(down_tgs, 1, 1),
                    MTLSize::new(down_kh.threads_per_tg, 1, 1),
                );
            }
        }
    }

    /// Run `resolved` experts and return the weighted, post-normed MoE
    /// output — the standalone form: own command buffer, one commit+wait,
    /// CPU combine. The merged-CB decode path uses
    /// [`Self::encode_experts_and_combine_zero_copy`] instead.
    pub(super) fn dispatch_experts_zero_copy(
        &self,
        expert_input: &[f32],
        moe: &MoeLayerWeights<'_>,
        eps: f32,
        scratch: &MoeScratch,
        resolved: &[ResolvedExpert],
    ) -> Vec<f32> {
        let hidden = scratch.hidden;
        let valid_count = resolved.len();
        let timing_enabled =
            larql_compute::options::env_flag(larql_compute::options::ENV_METAL_MOE_TIMING);
        let t_start = std::time::Instant::now();

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        self.encode_experts_zero_copy(enc, expert_input, moe, scratch, resolved);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        let t_gpu = t_start.elapsed();

        // ── Readback: down bias joins each expert's output BEFORE the
        // routing weight (reference order), then post-experts norm.
        let all_expert_outputs = read_buffer_f32(&scratch.expert_outs, valid_count * hidden);
        let mut moe_out = vec![0.0f32; hidden];
        for (e, r) in resolved.iter().enumerate() {
            let w = r.weight;
            let mlp = moe.expert_mlp(r.expert_id);
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
        if timing_enabled {
            let t_total = t_start.elapsed();
            eprintln!(
                "[run_experts_metal/zero-copy] K={valid_count} gpu={:.2}ms \
                 readback+sum={:.2}ms total={:.2}ms",
                t_gpu.as_secs_f32() * 1000.0,
                (t_total - t_gpu).as_secs_f32() * 1000.0,
                t_total.as_secs_f32() * 1000.0,
            );
        }

        moe_post_expert_output(&moe_out, moe, 0.0, eps)
    }

    /// Merged-CB form: encode the expert dispatches AND the weighted
    /// combine (`new_h = h_post_attn + Σ w·(out + down_bias)`) into `enc`,
    /// with NO command-buffer lifecycle — the caller lets the next
    /// layer's attention ride the same buffer, halving per-layer waits.
    ///
    /// Valid ONLY for the identity-combine policy class
    /// (`MoePostExpertNormPolicy::None`, no combined-output norm, no
    /// layer scalar, unit residual multiplier) — the caller gates on
    /// that; this function debug-asserts it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_experts_and_combine_zero_copy(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        expert_input: &[f32],
        moe: &MoeLayerWeights<'_>,
        scratch: &MoeScratch,
        resolved: &[ResolvedExpert],
        h_post_attn: &Buffer,
        new_h: &Buffer,
    ) {
        debug_assert!(matches!(
            moe.routing_policy.post_expert_norm,
            larql_compute::MoePostExpertNormPolicy::None
        ));
        let hidden = scratch.hidden;
        let valid_count = resolved.len();

        self.encode_experts_zero_copy(enc, expert_input, moe, scratch, resolved);

        // Stage the selected experts' down-bias rows slot-aligned with
        // `expert_outs` (small — k × hidden f32 against ~22 MB/expert of
        // weight reads). All-or-nothing per layer: `expert_mlp` yields an
        // empty bias exactly when the layer has none.
        let has_bias = resolved
            .first()
            .is_some_and(|r| !moe.expert_mlp(r.expert_id).down_bias.is_empty());
        if has_bias {
            for (slot, r) in resolved.iter().enumerate() {
                let bias = moe.expert_mlp(r.expert_id).down_bias;
                debug_assert_eq!(bias.len(), hidden);
                // SAFETY: shared-storage scratch sized `top_k × hidden`
                // f32; `slot < valid_count <= top_k`. CPU writes complete
                // before commit (the caller commits after encoding).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bias.as_ptr(),
                        (scratch.down_bias_staged.contents() as *mut f32).add(slot * hidden),
                        hidden,
                    );
                }
            }
        }

        let weights: Vec<f32> = resolved.iter().map(|r| r.weight).collect();
        let hidden_u = hidden as u32;
        let k_u = valid_count as u32;
        let has_bias_u: u32 = u32::from(has_bias);
        enc.set_compute_pipeline_state(&self.ffn.moe_weighted_combine_pipeline);
        enc.set_buffer(0, Some(&scratch.expert_outs), 0);
        enc.set_buffer(1, Some(h_post_attn), 0);
        enc.set_buffer(2, Some(new_h), 0);
        enc.set_bytes(3, 4, &hidden_u as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &k_u as *const u32 as *const c_void);
        enc.set_bytes(
            5,
            (weights.len() * 4) as u64,
            weights.as_ptr() as *const c_void,
        );
        enc.set_buffer(6, Some(&scratch.down_bias_staged), 0);
        enc.set_bytes(7, 4, &has_bias_u as *const u32 as *const c_void);
        enc.dispatch_threads(
            MTLSize::new(hidden as u64, 1, 1),
            MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                1,
                1,
            ),
        );
    }
}
