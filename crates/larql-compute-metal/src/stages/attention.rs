//! Fused causal attention — one dispatch for the whole layer's QKV → attn_out.
//!
//! Dispatches `fused_attention` which handles RoPE (optional), QK-norm
//! (optional), causal GQA softmax, and softcap in a single Metal kernel.
//! Grid is `(num_q_heads, seq_len, 1)` threadgroups of 256 threads.
//!
//! When the caller has already applied QK-norm separately (via
//! `stages::qk_norm::encode_qk_norm`), pass `use_qk_norm = false`.
//! When the caller has already applied RoPE via `stages::rope::encode`,
//! pass `skip_rope = true`.

use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

/// Flags for the fused attention dispatch. Keeps the parameter list
/// readable; every boolean has an obvious default.
#[derive(Clone, Copy)]
pub struct Flags {
    pub use_qk_norm: bool,
    pub skip_rope: bool,
    pub softcap: f32,
    pub rotary_dim: u32,
    /// This layer's sliding window, or `None` for full attention.
    ///
    /// ROADMAP M4: prefill had no window at all, so every sliding layer
    /// attended the whole prefix on GPU while CPU prefill windowed —
    /// the M1 asymmetry inverted. Resolve it through
    /// `forward_overrides::effective_attention_window_for_layer`, the
    /// same rule the CPU path and the Metal decode path read, so the
    /// three cannot disagree about which layers are windowed.
    pub window: Option<usize>,
}

/// First of the two consecutive `fused_attention` slots carrying
/// attention sinks; the `has_sinks` flag follows in slot 15.
const SINKS_BUFFER_INDEX: u64 = 14;

/// `amplitude` slot on `fused_attention`, appended after the sinks pair so no
/// existing buffer index had to move. The frequency table took over the old
/// `rope_base` slot (9) in place. See `shaders::fused_attention`.
const FUSED_ATTENTION_AMPLITUDE_INDEX: u64 = 16;

/// Threadgroup width for `fused_attention`. The kernel's threadgroup
/// reductions size their scratch arrays against this, so it is fixed by
/// the shader rather than tunable here.
const THREADS_PER_THREADGROUP: u64 = 256;

/// Dispatch `fused_attention` into the given encoder. Caller owns the
/// encoder lifecycle.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    enc: &ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    q_buf: &Buffer,
    k_buf: &Buffer,
    v_buf: &Buffer,
    attn_out: &Buffer,
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    plan: &larql_compute::attention::rope::RopeFreqPlan,
    flags: Flags,
    sinks: Option<&[f32]>,
) {
    // The shader's threadgroup scratch fixes two hard ceilings that
    // were previously silently assumed (capability audit F14):
    // `tg_q[512]` caps head_dim, and `tg_scores[4096]` is indexed by
    // ABSOLUTE key position (not `k - k_start`), so a sliding window
    // does not shrink the footprint and seq_len past 4096 writes out
    // of bounds. GQA divisibility is the kernel's head-mapping
    // assumption (`head / (num_q / num_kv)`), asserted nowhere else.
    assert!(
        head_dim <= 512,
        "fused_attention supports head_dim <= 512 (tg_q scratch); got {head_dim}"
    );
    assert!(
        seq_len <= crate::shaders::fused_attention::MAX_FUSED_ATTENTION_SEQ_LEN,
        "fused_attention supports seq_len <= {} (tg_scores is indexed by absolute \
         position, so a window does not reduce this); got {seq_len}",
        crate::shaders::fused_attention::MAX_FUSED_ATTENTION_SEQ_LEN,
    );
    assert!(
        num_kv_heads > 0 && num_q_heads.is_multiple_of(num_kv_heads),
        "fused_attention assumes num_q_heads ({num_q_heads}) divisible by \
         num_kv_heads ({num_kv_heads})"
    );
    let seq_val = seq_len as u32;
    let hd_val = head_dim as u32;
    let nq_val = num_q_heads as u32;
    let nkv_val = num_kv_heads as u32;
    let qknorm_val: u32 = if flags.use_qk_norm { 1 } else { 0 };
    let skip_rope_val: u32 = if flags.skip_rope { 1 } else { 0 };

    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(q_buf), 0);
    enc.set_buffer(1, Some(k_buf), 0);
    enc.set_buffer(2, Some(v_buf), 0);
    enc.set_buffer(3, Some(attn_out), 0);
    enc.set_bytes(4, 4, &seq_val as *const u32 as *const c_void);
    enc.set_bytes(5, 4, &hd_val as *const u32 as *const c_void);
    enc.set_bytes(6, 4, &nq_val as *const u32 as *const c_void);
    enc.set_bytes(7, 4, &nkv_val as *const u32 as *const c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
    enc.set_bytes(10, 4, &qknorm_val as *const u32 as *const c_void);
    enc.set_bytes(11, 4, &flags.softcap as *const f32 as *const c_void);
    enc.set_bytes(12, 4, &skip_rope_val as *const u32 as *const c_void);
    enc.set_bytes(13, 4, &flags.rotary_dim as *const u32 as *const c_void);
    // 0 = no window. `effective_attention_window_for_layer` normalises a
    // declared zero-width window to `None`, so `Some(0)` cannot arrive
    // here and mean "attend nothing".
    let window_val: u32 = flags.window.unwrap_or(0) as u32;
    enc.set_bytes(17, 4, &window_val as *const u32 as *const c_void);
    // Attention sinks (GPT-OSS): one learned logit per query head that
    // competes in the softmax and is then discarded.
    super::sinks::bind(enc, SINKS_BUFFER_INDEX, sinks, num_q_heads);
    super::rope_freq::bind(
        enc,
        9,
        FUSED_ATTENTION_AMPLITUDE_INDEX,
        plan,
        head_dim,
        flags.rotary_dim as usize,
    );
    enc.dispatch_thread_groups(
        MTLSize::new(num_q_heads as u64, seq_len as u64, 1),
        MTLSize::new(THREADS_PER_THREADGROUP, 1, 1),
    );
}
