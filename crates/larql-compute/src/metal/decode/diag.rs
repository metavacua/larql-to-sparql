//! Debug + diagnostic helpers for the decode pipeline.
//!
//! Two env-gated facilities:
//! - `DECODE_DEBUG=1` — one-line summary of the first few decode calls
//!   (input RMS, hidden/inter sizes, whether any layer is MoE).
//! - A per-layer NaN/Inf stat dump used by the caller when a `diag_stop_layer`
//!   is supplied; see [`dump_layer_buffers`].

use crate::FullPipelineLayer;

/// Print the one-line `DECODE_DEBUG` entry for the Nth decode call. No-op
/// unless the env var is set and `call_n < 3` (caller's contract).
pub(super) fn log_decode_entry(
    call_n: usize,
    x: &[f32],
    hidden: usize,
    inter: usize,
    layers: &[FullPipelineLayer],
) {
    if std::env::var("DECODE_DEBUG").is_err() || call_n >= 3 { return; }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let has_moe = layers.iter().any(|l| l.moe.is_some());
    let has_combined = layers.iter().any(|l| l.moe_combined_output_norm);
    let n = layers.len();
    let outer_loaded = layers.iter().filter(|l| l.moe_outer_post_norm.is_some()).count();
    let post1_loaded = layers.iter().filter(|l| l.post_ffn_norm.is_some()).count();
    eprintln!(
        "[decode_token call={call_n}] x_rms={rms:.4} hidden={hidden} inter={inter} has_moe={has_moe} moe_combined_norm={has_combined} outer_post_norm={outer_loaded}/{n} post_ffn_norm_1={post1_loaded}/{n}"
    );
}

/// Bundle of per-sub-stage Metal buffers that the diagnostic early-exit
/// dumps. Passed as a struct so the main decode function can assemble it
/// once after computing per-layer dimensions.
pub(super) struct LayerDiagBufs<'a> {
    pub norm_f32_buf: &'a metal::Buffer,
    pub q_out: &'a metal::Buffer,
    pub k_out: &'a metal::Buffer,
    pub v_out: &'a metal::Buffer,
    pub attn_out_buf: &'a metal::Buffer,
    pub o_out_buf: &'a metal::Buffer,
    pub h_post_attn: &'a metal::Buffer,
    pub ffn_norm_out: &'a metal::Buffer,
    pub gate_out_scratch: &'a metal::Buffer,
    pub up_out: &'a metal::Buffer,
    pub act_buf: &'a metal::Buffer,
    pub down_out: &'a metal::Buffer,
    pub new_h: &'a metal::Buffer,
    pub hidden: usize,
    pub inter: usize,
    pub layer_q_dim: usize,
    pub layer_kv_dim: usize,
}

/// Dump NaN/Inf counts and max-abs for every buffer in `bufs`, tagged with
/// the layer index. Called after the command buffer has been committed and
/// waited — the Metal contents are stable by the time this runs.
pub(super) fn dump_layer_buffers(l: usize, bufs: &LayerDiagBufs<'_>) {
    let stat = |name: &str, buf: &metal::Buffer, n: usize| {
        let ptr = buf.contents() as *const f32;
        if ptr.is_null() {
            eprintln!("[diag L{l}] {name}: null contents");
            return;
        }
        let s = unsafe { std::slice::from_raw_parts(ptr, n) };
        let nan = s.iter().filter(|v| v.is_nan()).count();
        let inf = s.iter().filter(|v| v.is_infinite()).count();
        let maxabs = s
            .iter()
            .map(|v| v.abs())
            .filter(|v| v.is_finite())
            .fold(0.0f32, f32::max);
        eprintln!("[diag L{l}] {name}: len={n} nan={nan} inf={inf} max_abs={maxabs:.3e}");
    };
    stat("norm_f32_buf", bufs.norm_f32_buf, bufs.hidden);
    stat("q_out", bufs.q_out, bufs.layer_q_dim);
    stat("k_out", bufs.k_out, bufs.layer_kv_dim);
    stat("v_out", bufs.v_out, bufs.layer_kv_dim);
    stat("attn_out_buf", bufs.attn_out_buf, bufs.layer_q_dim);
    stat("o_out_buf", bufs.o_out_buf, bufs.hidden);
    stat("h_post_attn", bufs.h_post_attn, bufs.hidden);
    stat("ffn_norm_out", bufs.ffn_norm_out, bufs.hidden);
    stat("gate_out_scratch", bufs.gate_out_scratch, bufs.inter);
    stat("up_out", bufs.up_out, bufs.inter);
    stat("act_buf", bufs.act_buf, bufs.inter);
    stat("down_out", bufs.down_out, bufs.hidden);
    stat("new_h (h_out)", bufs.new_h, bufs.hidden);
}
