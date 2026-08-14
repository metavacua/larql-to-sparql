//! Q4 matvec v4: wide integer loads + simdgroup.
//!
//! Key change: load Q4 data as uint4 (16 bytes in one load),
//! extract nibbles with bitwise ops on packed uint32,
//! multiply with Q8 using integer arithmetic throughout.
//! Avoids per-byte load + per-nibble branch.
//!
//! Geometry is exposed via the [`Kernel`] marker (see
//! `metal::kernel::TiledKernel`) so the binding site picks up name +
//! row map + threads-per-TG by *path*, not by hand-typed strings.

pub const SHADER: &str = r#"
constant uint ROWS_PER_TG_V4 = 8;

kernel void q4_matvec_v4(
    device const uchar* Q4    [[buffer(0)]],
    device const char*  Q8    [[buffer(1)]],
    device const float* Q8s   [[buffer(2)]],
    device float*       out   [[buffer(3)]],
    constant uint&      N     [[buffer(4)]],
    constant uint&      K     [[buffer(5)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint blocks = K / 32;
    uint bytes_per_row = blocks * 18;

    // Load Q8 into threadgroup memory (one load per TG)
    threadgroup int8_t tg_q8[8192];
    threadgroup float tg_q8s[256];
    for (uint i = tid_in_tg; i < K; i += 256)
        tg_q8[i] = ((device const int8_t*)Q8)[i];
    for (uint i = tid_in_tg; i < blocks; i += 256)
        tg_q8s[i] = Q8s[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row_idx = tg_id * ROWS_PER_TG_V4 + sg_id;
    if (row_idx >= N) return;

    device const uchar* row = Q4 + row_idx * bytes_per_row;

    float acc = 0.0f;
    for (uint b = lane; b < blocks; b += 32) {
        device const uchar* block = row + b * 18;

        // Scale
        ushort scale_bits = ushort(block[0]) | (ushort(block[1]) << 8);
        float combined_scale = decode_f16_metal(scale_bits) * tg_q8s[b];

        // Load 16 nibble bytes as 4 × uint32 via uchar4 (alignment-safe)
        device const uchar* qb = block + 2;
        uint w0 = uint(qb[0]) | (uint(qb[1]) << 8) | (uint(qb[2]) << 16) | (uint(qb[3]) << 24);
        uint w1 = uint(qb[4]) | (uint(qb[5]) << 8) | (uint(qb[6]) << 16) | (uint(qb[7]) << 24);
        uint w2 = uint(qb[8]) | (uint(qb[9]) << 8) | (uint(qb[10]) << 16) | (uint(qb[11]) << 24);
        uint w3 = uint(qb[12]) | (uint(qb[13]) << 8) | (uint(qb[14]) << 16) | (uint(qb[15]) << 24);

        threadgroup const int8_t* q8 = tg_q8 + b * 32;

        // Extract nibbles and compute dot product using integer arithmetic
        int isum = 0;

        // Extract nibbles from uint32, subtract 8 for signed range, multiply with Q8.
        // Cast to int BEFORE subtraction to avoid uint underflow.
        //
        // **ggml planar Q4_0 nibble layout** (`quantize_row_q4_0_ref`): byte j
        // packs element j in its LOW nibble and element j+16 in its HIGH
        // nibble — the two halves of the 32-element block, not an adjacent
        // pair. So a word covering bytes 4k..4k+3 feeds q8 slots
        // {4k..4k+3} and {4k+16..4k+19}, interleaved low/high.
        //
        // Until 2026-08-07 this decoded a private layout (byte j → elements
        // 2j, 2j+1), which matched the CPU side only because the CPU shared
        // that private convention. See `q6k_matvec` for the Q6_K half of the
        // same defect.
        #define NIBBLE(w, shift) (int((w >> shift) & 0xFu) - 8)
        #define PROCESS_WORD(w, base) \
            isum += NIBBLE(w,  0) * int(q8[base+0]);      \
            isum += NIBBLE(w,  4) * int(q8[base+0+16]);   \
            isum += NIBBLE(w,  8) * int(q8[base+1]);      \
            isum += NIBBLE(w, 12) * int(q8[base+1+16]);   \
            isum += NIBBLE(w, 16) * int(q8[base+2]);      \
            isum += NIBBLE(w, 20) * int(q8[base+2+16]);   \
            isum += NIBBLE(w, 24) * int(q8[base+3]);      \
            isum += NIBBLE(w, 28) * int(q8[base+3+16]);

        PROCESS_WORD(w0, 0);
        PROCESS_WORD(w1, 4);
        PROCESS_WORD(w2, 8);
        PROCESS_WORD(w3, 12);
        #undef PROCESS_WORD

        acc += float(isum) * combined_scale;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[row_idx] = acc;
}
"#;

/// Hard input-length ceiling from the threadgroup staging arrays
/// (`tg_x8[8192]` int8 + 256 block scales): the staging loops write
/// past both for larger K — threadgroup-memory corruption, not a
/// graceful failure. Host dispatch sites assert against this
/// (capability audit F13).
pub const MAX_K: usize = 8192;

pub const ROWS_PER_TG: u64 = 8;
pub const THREADS_PER_TG: u64 = 256;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q4_matvec_v4";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
