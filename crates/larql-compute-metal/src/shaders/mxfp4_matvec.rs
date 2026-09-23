//! MXFP4 matrix-vector multiply — direct compressed execution, no
//! materialisation.
//!
//! **K1 of the fused-MXFP4 kernel ladder.** The question this rung answers is
//! whether MXFP4 can be consumed as a *compute* format rather than a storage
//! one. Before it, the only MXFP4 path was bulk `dequantize_expert` into an f32
//! buffer followed by a matvec over that buffer, which moves per weight:
//!
//! ```text
//!   0.53125 B  packed + scale, read
//!   4 B        f32 dequant output, WRITTEN
//!   4 B x T    f32, re-read by the matvec
//! ```
//!
//! — 8.53 B of traffic for 0.53 B of stored weight at T=1, measured at 16.1x
//! amplification and rising to 61.2x at T=7 (DEC-8.8). This kernel moves the
//! 0.53 B and nothing else: nibbles are unpacked and scaled **in registers**,
//! accumulated straight into the dot product, and never written back.
//!
//! ## Format
//!
//! Per output row, `groups = K / 32` groups. Group `g` holds:
//!   - 16 packed bytes at `packed[(row * groups + g) * 16 ..][..16]`,
//!     each byte carrying two 4-bit codes: **lo nibble first**, then hi.
//!   - one e8m0 scale byte at `scales[row * groups + g]`.
//!
//! Element `g * 32 + 2 * b` takes byte `b`'s lo nibble, `+ 1` takes its hi
//! nibble. The 4-bit code indexes `MXFP4_LUT` = ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}.
//!
//! e8m0 is exponent-only: `value = bitcast<f32>(byte << 23)`. Two sentinels are
//! reproduced from the CPU reference deliberately rather than left to the
//! bitcast — `0 -> 0.0` and `255 -> NaN`. A raw bitcast would give `+inf` for
//! 255, so omitting the check would silently diverge from the reference on
//! exactly the adversarial input a parity test should catch. (A scan of every
//! scale tensor in the GPT-OSS lineage found zero occurrences of either value,
//! so this is a correctness contract, not a hot path.)
//!
//! ## Parallelism
//!
//! One simdgroup per output row, `ROWS_PER_TG` simdgroups per threadgroup.
//! Within a row, lane `l` walks groups `l, l+32, l+64, ...`, so each lane reads
//! one contiguous 16-byte group and adjacent lanes cover 512 contiguous bytes
//! per step — coalesced without any nibble-stride games. The K reduction closes
//! with `simd_sum`.
//!
//! Accumulation order therefore differs from the CPU reference (which sums
//! left-to-right), so parity is a bounded-error contract, not bit-equality.

/// Output rows per threadgroup — one simdgroup each.
pub const ROWS_PER_TG: u64 = 4;
/// 4 simdgroups x 32 lanes.
pub const THREADS_PER_TG: u64 = 128;

pub const SHADER: &str = r#"
constant uint MXFP4_ROWS_PER_TG = 4;
constant uint MXFP4_GROUP_ELEMS = 32;
constant uint MXFP4_GROUP_BYTES = 16;

// ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} — sign in bit 3, then exp(2) and mantissa(1).
constant float MXFP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// e8m0 -> f32, matching the CPU reference including both sentinels.
inline float mxfp4_e8m0(uchar b) {
    if (b == 0)   { return 0.0f; }
    if (b == 255) { return NAN;  }
    return as_type<float>(uint(b) << 23);
}

kernel void mxfp4_matvec(
    device const uchar*  Wp    [[buffer(0)]],   // packed [M, groups, 16]
    device const uchar*  Ws    [[buffer(1)]],   // scales [M, groups]
    device const float*  X     [[buffer(2)]],   // [K]
    device float*        out   [[buffer(3)]],   // [M]
    constant uint&       M     [[buffer(4)]],
    constant uint&       K     [[buffer(5)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row = tg_id * MXFP4_ROWS_PER_TG + sg_id;
    if (row >= M) { return; }

    const uint groups = K / MXFP4_GROUP_ELEMS;
    device const uchar* row_p = Wp + (ulong)row * (ulong)groups * MXFP4_GROUP_BYTES;
    device const uchar* row_s = Ws + (ulong)row * (ulong)groups;

    float acc = 0.0f;

    // Lane l walks groups l, l+32, ... — one contiguous 16-byte read each,
    // 512 contiguous bytes across the simdgroup per step.
    for (uint g = lane; g < groups; g += 32u) {
        const float scale = mxfp4_e8m0(row_s[g]);
        device const uchar* blk = row_p + (ulong)g * MXFP4_GROUP_BYTES;
        const uint base = g * MXFP4_GROUP_ELEMS;

        // Dequantise in registers and fold straight into the dot product.
        // Nothing is written back; there is no f32 weight buffer.
        float part = 0.0f;
        for (uint b = 0u; b < MXFP4_GROUP_BYTES; ++b) {
            const uchar byte = blk[b];
            part += MXFP4_LUT[byte & 0x0Fu]        * X[base + 2u * b];
            part += MXFP4_LUT[(byte >> 4u) & 0x0Fu] * X[base + 2u * b + 1u];
        }
        acc += scale * part;
    }

    acc = simd_sum(acc);
    if (lane == 0u) { out[row] = acc; }
}
"#;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "mxfp4_matvec";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
