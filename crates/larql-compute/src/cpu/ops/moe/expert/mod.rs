//! Per-expert gated-FFN execution (gate_proj, up_proj, activation, down_proj).
//!
//! Used by the in-process MoE forward pass (`cpu_moe_forward`) and by the
//! remote expert server endpoint when one expert's work is delegated to a
//! shard. The BF16 expert weights are dequantized on demand so only the
//! selected experts pay the conversion cost.

mod f32;
mod norm;
mod q4k;
mod scratch;

#[cfg(test)]
mod tests;

pub use self::f32::{run_single_expert, run_single_expert_into};
pub use self::norm::{pre_experts_norm, run_single_expert_with_norm};
pub use self::q4k::{
    quantize_h_norm_for_q4k, run_single_expert_kq_q8k_into, run_single_expert_kq_q8k_parallel_into,
    run_single_expert_q4k_q8k_into,
};
pub use self::scratch::ExpertScratch;
