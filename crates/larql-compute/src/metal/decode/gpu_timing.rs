//! GPU-side wall-clock timing for `MTLCommandBuffer`. Diagnostic only;
//! production code paths don't read these unless `LARQL_GPU_TIMING=1`
//! is set.
//!
//! Why this exists: the bench's per-stage breakdown reports
//! "GPU fwd = 11.9 ms/tok" by sampling wall time around the whole
//! `decode_token` call. That figure is **CPU + GPU** wall time.
//! `MTLCommandBuffer` exposes `gpuStartTime` / `gpuEndTime` (in
//! CFTimeInterval seconds, host monotonic) — the actual GPU compute
//! window for that buffer. Subtracting the two and summing across all
//! per-token cmd buffers gives **GPU-only time**. The delta vs wall
//! time is CPU encoding overhead.
//!
//! For the gemma3-4b-q4k-v2 / ollama gap diagnosis (78.7 vs 95 tok/s,
//! 2.2 ms/tok delta), this answers the directional question: if
//! `wall ≈ gpu_time`, the gap lives in kernel efficiency (need
//! different shaders or fusion). If `wall >> gpu_time`, the gap lives
//! in CPU dispatch overhead (close via fewer dispatches / batched
//! encoding).
//!
//! `metal-rs 0.29` doesn't expose these on `CommandBufferRef`; we call
//! the underlying Objective-C selectors via `msg_send!`.

use metal::CommandBufferRef;
use objc::{msg_send, sel, sel_impl};

use crate::options;

/// Returns `(gpu_start_time, gpu_end_time)` in seconds (CFTimeInterval).
/// Subtract for the GPU-side wall window. Caller MUST have already
/// called `wait_until_completed` on the buffer; values for an
/// in-flight buffer are undefined.
#[allow(unexpected_cfgs)]
pub fn gpu_window_seconds(cmd: &CommandBufferRef) -> (f64, f64) {
    unsafe {
        let start: f64 = msg_send![cmd, GPUStartTime];
        let end: f64 = msg_send![cmd, GPUEndTime];
        (start, end)
    }
}

/// Convenience: `gpu_end - gpu_start` in milliseconds.
pub fn gpu_elapsed_ms(cmd: &CommandBufferRef) -> f64 {
    let (start, end) = gpu_window_seconds(cmd);
    (end - start) * 1000.0
}

/// Stage labels for fine-grained per-token GPU profiling.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DecodeStage {
    /// Attention block: input norm → QKV → QK-norm → RoPE → V-norm → KV-attend → O.
    Attention,
    /// Dense FFN gate+up dispatch only (fused or separate). Recorded when
    /// `LARQL_PROFILE_SPLIT=1` is set; replaces `DenseFfn` for the fine split.
    GateUp,
    /// FFN activation (GEGLU/SiLU) + down matvec + post-FFN residual.
    /// Paired with `GateUp` in the fine-split path.
    Down,
    /// Coarse FFN bucket (gate+up+act+down+residual together). Only emitted
    /// when the fine split isn't active; kept for legacy callers.
    DenseFfn,
    /// Final norm + lm_head (only if recorded; many decode paths run it on CPU).
    #[allow(dead_code)]
    Final,
    /// Anything else / unlabeled.
    Other,
}

/// Token-scope GPU time accumulator. Threads ms across multiple cmd
/// buffers (e.g., per-MoE-layer commits in `decode_token_with_moe_fn`)
/// and reports total at end-of-token when `LARQL_GPU_TIMING=1`.
///
/// When the caller uses [`Self::record_stage`] (instead of bare
/// [`Self::record`]) and `LARQL_DECODE_STAGE_TIMING=1` is set, the
/// summary additionally breaks the GPU total down per stage —
/// answers questions like "of the 17ms client GPU, how much is
/// attention vs dense FFN?" without rebuilding the model.
#[derive(Default)]
pub struct TokenGpuTime {
    pub total_gpu_ms: f64,
    pub n_cmd_buffers: usize,
    /// Per-stage GPU time accumulators. Updated by `record_stage`.
    pub attn_ms: f64,
    /// Gate+up dispatch (fine split). Zero when coarse split is active.
    pub gate_up_ms: f64,
    /// Activation+down+residual (fine split). Zero when coarse split is active.
    pub down_ms: f64,
    pub dense_ffn_ms: f64,
    pub final_ms: f64,
    pub other_ms: f64,
}

impl TokenGpuTime {
    /// Add the GPU window for `cmd` to the running total. Called after
    /// `cmd.wait_until_completed()`.
    pub fn record(&mut self, cmd: &CommandBufferRef) {
        self.record_stage(cmd, DecodeStage::Other);
    }

    /// Like [`Self::record`] but also accumulates the elapsed time into
    /// the per-stage bucket for fine-grained profiling.
    pub fn record_stage(&mut self, cmd: &CommandBufferRef, stage: DecodeStage) {
        let elapsed = gpu_elapsed_ms(cmd);
        if elapsed.is_finite() && elapsed > 0.0 {
            self.total_gpu_ms += elapsed;
            self.n_cmd_buffers += 1;
            match stage {
                DecodeStage::Attention => self.attn_ms += elapsed,
                DecodeStage::GateUp => self.gate_up_ms += elapsed,
                DecodeStage::Down => self.down_ms += elapsed,
                DecodeStage::DenseFfn => self.dense_ffn_ms += elapsed,
                DecodeStage::Final => self.final_ms += elapsed,
                DecodeStage::Other => self.other_ms += elapsed,
            }
        }
    }

    /// Print a token-summary line if `LARQL_GPU_TIMING=1`. `wall_ms`
    /// is the caller's CPU+GPU wall measurement (whatever they timed
    /// around the whole token's work). Adds a per-stage breakdown when
    /// `LARQL_PROFILE_SPLIT=1` (or the legacy alias
    /// `LARQL_DECODE_STAGE_TIMING=1`) is set.
    pub fn print_if_enabled(&self, wall_ms: f64) {
        let gpu_timing = options::env_flag(options::ENV_GPU_TIMING);
        let stage_timing = options::split_profile_requested();
        if !gpu_timing && !stage_timing {
            return;
        }
        let cpu_ms = wall_ms - self.total_gpu_ms;
        let cpu_pct = if wall_ms > 0.0 {
            cpu_ms / wall_ms * 100.0
        } else {
            0.0
        };
        eprintln!(
            "[gpu-timing] wall={:.3}ms  gpu={:.3}ms  cpu={:.3}ms ({:.1}%)  cmd_bufs={}",
            wall_ms, self.total_gpu_ms, cpu_ms, cpu_pct, self.n_cmd_buffers
        );
        if stage_timing {
            let total = self.total_gpu_ms;
            let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };
            if self.gate_up_ms > 0.0 || self.down_ms > 0.0 {
                // Fine split: gate+up and act+down measured separately.
                eprintln!(
                    "[gpu-timing/stage] attn={:.2}ms ({:.0}%)  \
                     gate+up={:.2}ms ({:.0}%)  act+down={:.2}ms ({:.0}%)  \
                     other={:.2}ms ({:.0}%)",
                    self.attn_ms,
                    pct(self.attn_ms),
                    self.gate_up_ms,
                    pct(self.gate_up_ms),
                    self.down_ms,
                    pct(self.down_ms),
                    self.other_ms,
                    pct(self.other_ms),
                );
            } else {
                // Coarse split: whole FFN in one bucket.
                eprintln!(
                    "[gpu-timing/stage] attn={:.2}ms ({:.0}%)  dense_ffn={:.2}ms ({:.0}%)  \
                     final={:.2}ms ({:.0}%)  other={:.2}ms ({:.0}%)",
                    self.attn_ms,
                    pct(self.attn_ms),
                    self.dense_ffn_ms,
                    pct(self.dense_ffn_ms),
                    self.final_ms,
                    pct(self.final_ms),
                    self.other_ms,
                    pct(self.other_ms),
                );
            }
        }
    }
}
