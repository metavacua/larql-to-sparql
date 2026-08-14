//! Q6_K matrix-vector multiply — ggml's planar Q6_K layout.
//!
//! Q6_K super-block layout (256 values = 210 bytes):
//!   [0..127]    128 bytes: ql — lo4 bits, 2 per byte
//!   [128..191]   64 bytes: qh — hi2 bits, 4 per byte
//!   [192..207]   16 bytes: int8 scales, one per 16-element group
//!   [208..209]    2 bytes: f16 super-block scale d
//!
//! **The nibbles are planar, not sequential.** A super-block is two halves of
//! 128 elements; within half `h` (element base `e = 128h`), for `l` in 0..32:
//!
//!   ql[64h + l]      low nibble → element e+l      high nibble → element e+l+64
//!   ql[64h + l + 32] low nibble → element e+l+32   high nibble → element e+l+96
//!   qh[32h + l]      bits 0/2/4/6 → elements e+l, e+l+32, e+l+64, e+l+96
//!
//! So one `l` yields four elements at *stride 32* from three bytes, and the
//! scale index is still `element/16`. This mirrors `quantize_row_q6_K_ref`,
//! and it is what `larql_compute::cpu::ops::q4_common::quantize_q6_k` emits.
//!
//! Weight: d * sc[i/16] * ((lo4 | hi2<<4) - 32)
//!
//! Until 2026-08-07 these kernels decoded a private *interleaved* layout
//! (`lo4 = ql[i/2] >> 4*(i&1)`, `hi2 = qh[i/4] >> 2*(i%4)`), which agreed with
//! the CPU side only because the CPU shared the same private convention. When
//! the CPU moved to ggml conformance the two halves silently disagreed — see
//! `test_kernel_vindex_integration::stage_quant_matvec_routes_format_to_correct_shader`.
//!
//! **Key optimisations vs the previous all-lanes-per-superblock approach:**
//!
//! 1. **Inter-superblock interleaving**: `ix = lane & 1` splits 32 lanes into
//!    two groups. ix=0 processes superblocks 0,2,4,...; ix=1 processes 1,3,5,...
//!    Adjacent lanes read from different 210-byte memory regions simultaneously,
//!    letting the DRAM controller serve two banks in parallel.
//!
//! 2. **X preloading**: 16 X reads (4 per pass × 4 passes) are issued
//!    before ANY weight byte reads, hiding L2 latency behind weight fetches.
//!
//! 3. **Deferred scaling**: accumulate one unscaled sum per 4-element group,
//!    then apply `d * sc[j]` once — 4× fewer scale multiplications vs
//!    the previous per-element approach.
//!
//! 4. **Reduced TG size** (ROWS_PER_TG=4, 128 threads): halves register
//!    pressure vs the previous 256-thread design, allowing 2× more concurrent
//!    TGs on M3 Max for better LPDDR5X latency hiding.
//!
//! Each tid (0..15) within an ix-group handles 4 units × 4 elements = 16
//! elements per superblock. A unit is one `(half, l)` pair with
//! `l = tid + 16j`, so the four units are `(h,j) = (0,0) (0,1) (1,0) (1,1)`
//! and all 16 tids together cover all 256 elements. ✓
//!
//! Deferred scaling (note 3) does **not** survive the planar layout: a unit's
//! four elements sit in four different 16-element scale groups
//! (`8h + j + 2·plane`), so each takes its own `sc[]` lookup. `d` still
//! factors out once per unit. A 16-element scale group is contiguous in `l`,
//! which means it spans *threads*, not the elements one thread holds.

pub const SHADER: &str = r#"
constant uint Q6K_ROWS_PER_TG = 4;
constant uint Q6K_BLOCK_SIZE  = 210;

kernel void q6k_matvec(
    device const uchar*  W6K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint row_idx = tg_id * Q6K_ROWS_PER_TG + sg_id;
    if (row_idx >= N) return;

    const uint superblocks   = K / 256u;
    const uint bytes_per_row = superblocks * Q6K_BLOCK_SIZE;
    device const uchar* row  = W6K + row_idx * bytes_per_row;

    // Lane decomposition: ix splits 32 lanes into two interleaved-superblock
    // groups; tid is the position within each 16-lane group.
    const uint ix  = lane & 1u;   // 0 or 1
    const uint tid = lane >> 1u;  // 0..15

    // This tid owns nibble-plane column `l` of each half, for l = tid and
    // l = tid+16. One (half, l) unit is 3 bytes → 4 elements at stride 32.
    const uint l0 = tid;         // j = 0
    const uint l1 = tid + 16u;   // j = 1

    float acc = 0.0f;

    // ix=0 processes superblocks 0,2,4,...; ix=1 processes 1,3,5,...
    // Adjacent lanes in the simdgroup read from different 210-byte regions.
    for (uint i = ix; i < superblocks; i += 2u) {
        device const uchar* block = row + i * Q6K_BLOCK_SIZE;
        device const uchar* ql   = block;
        device const uchar* qh   = block + 128u;
        device const char*  sc   = (device const char*)(block + 192u);
        ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8u);
        float  d = decode_f16_metal(d_bits);

        // Preload all 16 X values for the 4 units before reading any weight
        // bytes. Explicit preload lets the GPU pipeline X fetches in parallel
        // with the upcoming ql/qh/sc reads. Planar means these run at stride
        // 32 within a half, not contiguously.
        const uint xb = i * 256u;
        const uint x0 = xb + l0;          // half 0, j 0
        const uint x1 = xb + l1;          // half 0, j 1
        const uint x2 = xb + 128u + l0;   // half 1, j 0
        const uint x3 = xb + 128u + l1;   // half 1, j 1
        float xl[16];
        xl[ 0] = X[x0]; xl[ 1] = X[x0 + 32u]; xl[ 2] = X[x0 + 64u]; xl[ 3] = X[x0 + 96u];
        xl[ 4] = X[x1]; xl[ 5] = X[x1 + 32u]; xl[ 6] = X[x1 + 64u]; xl[ 7] = X[x1 + 96u];
        xl[ 8] = X[x2]; xl[ 9] = X[x2 + 32u]; xl[10] = X[x2 + 64u]; xl[11] = X[x2 + 96u];
        xl[12] = X[x3]; xl[13] = X[x3 + 32u]; xl[14] = X[x3 + 64u]; xl[15] = X[x3 + 96u];

        // 4 units, each 2 ql bytes + 1 qh byte → 4 elements at stride 32.
        // Scale index is 8*half + j + 2*plane; `d` factors out per unit.

        // Unit 0: half 0, l = tid       (elements l, l+32, l+64, l+96)
        {
            uchar la = ql[l0], lb = ql[l0 + 32u], hi = qh[l0];
            acc += d * (
                float(sc[0u]) * float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 0] +
                float(sc[2u]) * float((char)((lb & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 1] +
                float(sc[4u]) * float((char)(((la >> 4u) & 0x0Fu) | (hi & 0x30u)) - 32) * xl[ 2] +
                float(sc[6u]) * float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 3]);
        }

        // Unit 1: half 0, l = tid + 16
        {
            uchar la = ql[l1], lb = ql[l1 + 32u], hi = qh[l1];
            acc += d * (
                float(sc[1u]) * float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 4] +
                float(sc[3u]) * float((char)((lb & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 5] +
                float(sc[5u]) * float((char)(((la >> 4u) & 0x0Fu) | (hi & 0x30u)) - 32) * xl[ 6] +
                float(sc[7u]) * float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[ 7]);
        }

        // Unit 2: half 1, l = tid
        {
            uchar la = ql[64u + l0], lb = ql[96u + l0], hi = qh[32u + l0];
            acc += d * (
                float(sc[ 8u]) * float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[ 8] +
                float(sc[10u]) * float((char)((lb & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[ 9] +
                float(sc[12u]) * float((char)(((la >> 4u) & 0x0Fu) | (hi & 0x30u)) - 32) * xl[10] +
                float(sc[14u]) * float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[11]);
        }

        // Unit 3: half 1, l = tid + 16
        {
            uchar la = ql[64u + l1], lb = ql[96u + l1], hi = qh[32u + l1];
            acc += d * (
                float(sc[ 9u]) * float((char)((la & 0x0Fu) | ((hi & 0x03u) << 4u)) - 32) * xl[12] +
                float(sc[11u]) * float((char)((lb & 0x0Fu) | ((hi & 0x0Cu) << 2u)) - 32) * xl[13] +
                float(sc[13u]) * float((char)(((la >> 4u) & 0x0Fu) | (hi & 0x30u)) - 32) * xl[14] +
                float(sc[15u]) * float((char)(((lb >> 4u) & 0x0Fu) | ((hi & 0xC0u) >> 2u)) - 32) * xl[15]);
        }
    }

    acc = simd_sum(acc);
    if (lane == 0u) out[row_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128;

/// Marker for the kernel-handle binding. See `metal::kernel::TiledKernel`.
pub struct Kernel;
impl crate::kernels::TiledKernel for Kernel {
    const KERNEL_NAME: &'static str = "q6k_matvec";
    const ROWS_PER_TG: u64 = ROWS_PER_TG;
    const THREADS_PER_TG: u64 = THREADS_PER_TG;
}
