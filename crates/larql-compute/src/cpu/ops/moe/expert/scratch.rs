use crate::cpu::ops::q4k_q8k_dot::Q8KActivation;

/// Per-call scratch for `run_single_expert_with_scratch` — preallocate once
/// per gRPC frame and reuse across all K active experts.  Keeps allocation
/// off the hot path: at Gemma 4 26B-A4B sizes the un-pooled version was
/// minting ~360 fresh ~11KB Vecs per token per shard.
///
/// Sized for one expert's worth of intermediate buffers.  Per-call cost on
/// reuse is O(0) — just zeros the activation buffer's padding columns.
pub struct ExpertScratch {
    /// `[inter]` — gate matvec output before activation.
    pub gate_out: Vec<f32>,
    /// `[inter]` — up matvec output.
    pub up_out: Vec<f32>,
    /// `[inter_padded]` — activation buffer fed into down.  Padding columns
    /// (`inter..inter_padded`) are zero-initialised once and re-used
    /// untouched across calls (down's matvec reads them as zero).
    pub act: Vec<f32>,
    /// Q8_K quantisation of `act` for the down matvec on the Q4_K-direct
    /// path.  Pre-allocated at construction so the per-expert quantise
    /// doesn't allocate — eliminates the 5% / 150 µs alloc spikes that
    /// previously dragged the par_iter wall up across rayon workers.
    pub act_q8k: Q8KActivation,
    /// `[hidden]` — final expert output.
    pub out: Vec<f32>,
}

impl ExpertScratch {
    /// Allocate scratch sized for `(hidden, inter, inter_padded)`.  Call
    /// once per gRPC frame; share `&mut` across the K experts.
    pub fn new(hidden: usize, inter: usize, inter_padded: usize) -> Self {
        Self {
            gate_out: vec![0.0f32; inter],
            up_out: vec![0.0f32; inter],
            act: vec![0.0f32; inter_padded],
            act_q8k: Q8KActivation::with_capacity(inter_padded),
            out: vec![0.0f32; hidden],
        }
    }
}
