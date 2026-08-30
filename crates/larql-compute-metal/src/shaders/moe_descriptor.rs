//! Expert-descriptor kernels — rung C of the GPU-dataflow routing ladder.
//!
//! An `MoeExpertDescriptor` is the immutable description of one STORED
//! expert: where its payloads, exponent streams and bias rows live. It
//! describes expert `e` independently of routing — a route result is a
//! separate, runtime thing (`selected_ids`, rung B's output) that merely
//! POINTS at descriptors. Layer-uniform facts (quant format, fused row
//! layout, dims) deliberately do not appear here: the storage contract
//! (`MoeLayerWeights`) states them once per layer, and duplicating them
//! per expert would invite drift.
//!
//! Two kernels:
//!
//! - `moe_descriptor_gather` — `out[slot] = descs[selected_ids[slot]]`.
//!   The single runtime indirection the architecture needs: after it,
//!   every downstream binding is a static base buffer plus an offset the
//!   GPU already holds. Replaces the CPU building per-slot offset tables
//!   with `set_bytes` at encode time (`moe_zero_copy.rs`), which is the
//!   exact mechanism that forces routing to be a host decision.
//!
//! - `moe_bias_stage` — de-interleaves the selected experts' gate/up
//!   bias rows from the layer's bias bank into the slot-aligned scratch
//!   the activation kernels already read. Replaces the per-route CPU
//!   `memcpy` loop through `contents()` pointers — the hidden host
//!   dependency that would otherwise survive to rung E. Row layout
//!   matches `larql_models::quant::mxfp4::FusedHalf`: gate row `j` at
//!   fused index `2j`, up at `2j + 1`, per expert.
//!
//! Struct layout is mirrored by `GpuExpertDescriptor` in
//! `moe_descriptor.rs` (`#[repr(C)]`, six `u32`s); the two must not drift.

pub const SHADER: &str = r#"
struct MoeExpertDescriptor {
    uint gate_up_payload_off;  // bytes into the gate+up payload base
    uint down_payload_off;     // bytes into the down payload base
    uint gate_up_scale_off;    // bytes into the gate+up e8m0 base (0 if inline)
    uint down_scale_off;       // bytes into the down e8m0 base (0 if inline)
    uint gate_up_bias_off;     // f32 elements into the gate+up bias bank
    uint down_bias_off;        // f32 elements into the down bias bank
};

kernel void moe_descriptor_gather(
    device const MoeExpertDescriptor* descs        [[buffer(0)]],  // [E]
    device const uint*                selected_ids [[buffer(1)]],  // [n_slots]
    device MoeExpertDescriptor*       out          [[buffer(2)]],  // [n_slots]
    // Expanded per-slot offset tables, shaped exactly like the tables the
    // grouped-expert kernels already consume at buffer(1) — so the proven
    // kernels bind these verbatim (set_bytes → set_buffer at the call
    // site, zero kernel change). `gate1 = gate0 + gate_half_bytes` states
    // the ContiguousHalves layer fact once, as a uniform.
    device uint*                      gate0_offs   [[buffer(3)]],  // [n_slots]
    device uint*                      gate1_offs   [[buffer(4)]],  // [n_slots]
    device uint*                      down_offs    [[buffer(5)]],  // [n_slots]
    // Split-scale (native MXFP4) expansions: the e8m0 exponent streams'
    // per-slot offsets, index-aligned with the payload tables. Zero-filled
    // facts under inline scales (never bound then).
    device uint*                      gu_scale_offs [[buffer(6)]], // [n_slots]
    device uint*                      dn_scale_offs [[buffer(7)]], // [n_slots]
    constant uint&                    n_slots      [[buffer(8)]],
    constant uint&                    gate_half_bytes [[buffer(9)]],
    // Bounds guard (#229): `descs` has exactly `num_experts` entries and
    // nothing upstream proves the router's ids stay inside it. An id past
    // the end is a GPU page fault — a failed command buffer whose stale
    // outputs read like a finished one. Instead: count it, and gather
    // expert 0 so the step completes and the host can refuse it.
    constant uint&                    num_experts  [[buffer(10)]],
    device atomic_uint*               bad_ids      [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n_slots) {
        uint id = selected_ids[tid];
        if (id >= num_experts) {
            atomic_fetch_add_explicit(bad_ids, 1u, memory_order_relaxed);
            id = 0u;
        }
        MoeExpertDescriptor d = descs[id];
        out[tid] = d;
        gate0_offs[tid] = d.gate_up_payload_off;
        gate1_offs[tid] = d.gate_up_payload_off + gate_half_bytes;
        down_offs[tid]  = d.down_payload_off;
        gu_scale_offs[tid] = d.gate_up_scale_off;
        dn_scale_offs[tid] = d.down_scale_off;
    }
}

kernel void moe_down_bias_stage(
    device const float*               bias_bank  [[buffer(0)]],  // whole layer bank
    device const MoeExpertDescriptor* slot_descs [[buffer(1)]],  // [n_slots]
    device float*                     out        [[buffer(2)]],  // [n_slots, hidden]
    constant uint&                    hidden     [[buffer(3)]],
    constant uint&                    n_slots    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])  // (j, slot)
{
    uint j = gid.x;
    uint slot = gid.y;
    if (j >= hidden || slot >= n_slots) return;
    out[slot * hidden + j] = bias_bank[slot_descs[slot].down_bias_off + j];
}

kernel void moe_bias_stage(
    device const float*               bias_bank  [[buffer(0)]],  // whole layer bank
    device const MoeExpertDescriptor* slot_descs [[buffer(1)]],  // [n_slots]
    device float*                     gate_out   [[buffer(2)]],  // [n_slots, inter]
    device float*                     up_out     [[buffer(3)]],  // [n_slots, inter]
    constant uint&                    inter      [[buffer(4)]],
    constant uint&                    n_slots    [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])  // (j, slot)
{
    uint j = gid.x;
    uint slot = gid.y;
    if (j >= inter || slot >= n_slots) return;
    uint base = slot_descs[slot].gate_up_bias_off;
    // FusedHalf row map: gate at 2j, up at 2j+1.
    gate_out[slot * inter + j] = bias_bank[base + 2 * j];
    up_out[slot * inter + j]   = bias_bank[base + 2 * j + 1];
}
"#;

/// Marker for `moe_descriptor_gather` pipeline construction.
pub struct GatherKernel;
impl crate::kernels::ShaderKernel for GatherKernel {
    const KERNEL_NAME: &'static str = "moe_descriptor_gather";
}

/// Marker for `moe_bias_stage` pipeline construction.
pub struct BiasStageKernel;
impl crate::kernels::ShaderKernel for BiasStageKernel {
    const KERNEL_NAME: &'static str = "moe_bias_stage";
}

/// Marker for `moe_down_bias_stage` pipeline construction.
pub struct DownBiasStageKernel;
impl crate::kernels::ShaderKernel for DownBiasStageKernel {
    const KERNEL_NAME: &'static str = "moe_down_bias_stage";
}
