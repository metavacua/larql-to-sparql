//! Q6_K grouped-expert matvec — all selected experts in ONE dispatch.
//!
//! **K3a of the kernel ladder, promoted ahead of K2 and K4 by measurement.**
//!
//! `diag_shape_census` found that the identical `q6k_matvec` kernel reaches
//! 0.87-0.90 of roofline at K3's large shapes but only **0.64** at the expert
//! branch `[3584, 3072]`. That gap is not the codec and not the reduction tree —
//! it is occupancy. One expert matrix at `ROWS_PER_TG = 4` launches
//! `3584 / 4 = 896` threadgroups, too little independent work to hide memory
//! latency behind.
//!
//! But a token selects **16 experts per layer**. Dispatched together they supply
//! `896 x 16 = 14,336` threadgroups — sixteen times the parallel work, from the
//! model's own structure. The bottleneck was the stack failing to express
//! parallelism K3 already has, not a property of the shape.
//!
//! ## What this kernel changes, and what it deliberately does not
//!
//! Only the addressing. The reduction body is copied verbatim from
//! `q6k_matvec`, so per-element arithmetic and accumulation order are identical
//! and outputs must agree **exactly** with 16 separate dispatches — not to a
//! tolerance. Anything else would confound an occupancy result with a numerics
//! change.
//!
//! Specifically NOT done here:
//!   - no concatenation or repacking of the artifact; experts stay where they
//!     are and an offset table addresses them;
//!   - no feature or row sparsity — every selected expert is computed in full;
//!   - no padding to inflate threadgroup count. Padding would raise apparent
//!     occupancy while adding useless work; the win must come from scheduling
//!     *real* independent experts together.
//!
//! ## Dispatch
//!
//! Grid is 2-D: `(row_tiles, n_selected)`. Threadgroup `(tx, ty)` handles row
//! tile `tx` of the expert in slot `ty`. Each slot reads its payload base from
//! `offsets[ty]` (in bytes) and writes to `out[ty * N + row]`, so the caller
//! gets per-expert outputs and reduces them afterward — a `16 x N` reduction is
//! negligible against reading `16 x 9 MB` of weights, and keeps this first
//! grouped kernel simple enough to verify.
//!
//! The offset table is `uint` byte offsets rather than pointers so the same
//! interface works whether experts live in one buffer or many, and so K3b can
//! extend it to ragged position assignments without a signature change.
//!
//! ## Two input regimes, and why the stride is explicit
//!
//! Reading the engine's MoE path shows the two projections differ in a way that
//! matters here:
//!
//!   - **gate/up** consumes one hidden state shared by every selected expert.
//!     That case is *already* grouped in `moe_dispatch` via `q4k_ffn_gate_up`.
//!   - **down** consumes each expert's OWN intermediate activation, staged at
//!     `act_buf + slot * inter_padded`. That is the loop still running one
//!     dispatch per expert, so it is the case this kernel has to serve.
//!
//! `XSTRIDE` distinguishes them: `0` shares one vector across slots, `K` gives
//! each slot its own. It is a parameter rather than an inferred mode because
//! getting it wrong computes a real number from the wrong expert's activation —
//! a silent wrong answer, not a crash.

/// Output rows per threadgroup — one simdgroup each, matching `q6k_matvec` so
/// the occupancy comparison is like-for-like.
pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

pub const SHADER: &str = r#"
constant uint Q6KG_ROWS_PER_TG = 4;
constant uint Q6KG_BLOCK_SIZE  = 210;

kernel void q6k_grouped_experts(
    device const uchar*  W6K     [[buffer(0)]],  // all expert payloads
    device const uint*   offsets [[buffer(1)]],  // [n_sel] byte offset per slot
    device const float*  X       [[buffer(2)]],  // shared [K], or [n_sel, K]
    device float*        out     [[buffer(3)]],  // [n_sel, N] per-expert outputs
    constant uint&       N       [[buffer(4)]],
    constant uint&       K       [[buffer(5)]],
    // 0 = every slot reads the same X (gate/up: one hidden state for all
    // experts). K = each slot reads its own X (down: each expert consumes its
    // OWN intermediate activation). Getting this wrong silently computes the
    // wrong expert's product, so it is an explicit parameter, not a mode flag.
    constant uint&       XSTRIDE [[buffer(6)]],
    uint2 tg_id    [[threadgroup_position_in_grid]],
    uint  lane     [[thread_index_in_simdgroup]],
    uint  sg_id    [[simdgroup_index_in_threadgroup]])
{
    // tg_id.y selects the expert slot; tg_id.x the row tile within it.
    const uint slot    = tg_id.y;
    const uint row_idx = tg_id.x * Q6KG_ROWS_PER_TG + sg_id;
    if (row_idx >= N) { return; }

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q6KG_BLOCK_SIZE;
    // Offset table indirection is the whole difference from q6k_matvec.
    device const uchar* row = W6K + offsets[slot] + row_idx * bytes_per_row;
    device const float* Xs  = X + (ulong)slot * XSTRIDE;

    // ---- body identical to q6k_matvec from here ----
    const uint ix  = lane & 1u;
    const uint tid = lane >> 1u;
    const uint base    = tid << 2u;
    const uint sc_base = tid >> 2u;

    float acc = 0.0f;

    for (uint i = ix; i < superblocks; i += 2u) {
        device const uchar* block = row + i * Q6KG_BLOCK_SIZE;
        device const uchar* ql   = block;
        device const uchar* qh   = block + 128u;
        device const char*  sc   = (device const char*)(block + 192u);
        ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
        float  d = decode_f16_metal(d_bits);

        const uint xb = i * 256u + base;
        float xl[16];
        xl[ 0] = Xs[xb      ]; xl[ 1] = Xs[xb +  1u];
        xl[ 2] = Xs[xb +  2u]; xl[ 3] = Xs[xb +  3u];
        xl[ 4] = Xs[xb + 64u]; xl[ 5] = Xs[xb + 65u];
        xl[ 6] = Xs[xb + 66u]; xl[ 7] = Xs[xb + 67u];
        xl[ 8] = Xs[xb +128u]; xl[ 9] = Xs[xb +129u];
        xl[10] = Xs[xb +130u]; xl[11] = Xs[xb +131u];
        xl[12] = Xs[xb +192u]; xl[13] = Xs[xb +193u];
        xl[14] = Xs[xb +194u]; xl[15] = Xs[xb +195u];

        {
            const uint b = base;
            uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * float(sc[sc_base + 0u]);
            acc += _sc * (
                float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 0] +
                float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 1] +
                float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 2] +
                float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 3]);
        }
        {
            const uint b = base + 64u;
            uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * float(sc[sc_base + 4u]);
            acc += _sc * (
                float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 4] +
                float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 5] +
                float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[ 6] +
                float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 7]);
        }
        {
            const uint b = base + 128u;
            uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * float(sc[sc_base + 8u]);
            acc += _sc * (
                float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 8] +
                float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 9] +
                float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[10] +
                float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[11]);
        }
        {
            const uint b = base + 192u;
            uchar la = ql[b >> 1u], lb = ql[(b >> 1u) + 1u], hi = qh[b >> 2u];
            float _sc = d * float(sc[sc_base + 12u]);
            acc += _sc * (
                float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[12] +
                float((char)(((la >> 4u) & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[13] +
                float((char)((lb & 0x0Fu) | ((hi & 0x30u))) - 32) * xl[14] +
                float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[15]);
        }
    }

    acc = simd_sum(acc);
    if (lane == 0u) { out[slot * N + row_idx] = acc; }
}
"#;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q6k_grouped_experts";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
