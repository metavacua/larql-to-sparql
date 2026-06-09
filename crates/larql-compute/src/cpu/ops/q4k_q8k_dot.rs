//! Q4_K weight × Q8_K activation matrix-vector product.
//!
//! The hot path for CPU MoE on Gemma 4 26B-A4B.  Reads 144-byte Q4_K
//! super-blocks straight from the mmapped vindex (no f32 dequant cache),
//! quantises the activation once per call to Q8_K, and accumulates an
//! integer dot product per sub-block.  Math is mathematically equivalent
//! to `q4_common::q4k_matvec_into` (within Q8 quantisation noise on the
//! activation side), but avoids walking ~5.7 GB of f32 weights per token
//! at Gemma 4 26B-A4B sizes — DRAM pressure drops ~4×.
//!
//! Per llama.cpp `ggml_vec_dot_q4_K_q8_K`:
//!
//! ```text
//! per super-block (256 elements, 8 sub-blocks of 32):
//!   d_w    = f16_to_f32(block.d)        (per super-block weight scale)
//!   dmin_w = f16_to_f32(block.dmin)     (per super-block weight min-scale)
//!   d_y    = q8k.d                      (per super-block activation scale)
//!   for sb in 0..8:
//!     sc[sb] (u8 [0..63]), mn[sb] (u8 [0..63])  unpacked from the 12-byte header
//!     dot_sb = Σ_{i in 0..32} q4_nibble[i] * y_q[i]            (i32)
//!     sum_sb = Σ_{i in 0..32} y_q[i]                            (i16, precomputed)
//!     sum1 += sc[sb] * dot_sb
//!     sum2 += mn[sb] * sum_sb
//!   acc += d_w * d_y * sum1 - dmin_w * d_y * sum2
//! out[r] = acc
//! ```
//!
//! Inner kernel uses NEON `sdot` (ARMv8.2-A SDOT instruction, available on
//! Apple M1+ and most modern aarch64 chips) when compiled for `aarch64`;
//! falls back to a scalar reference otherwise.  Both paths share the
//! Q8_K activation quantiser and the per-super-block aggregation math —
//! only the inner i8×i8 → i32 dot differs.

use crate::cpu::ops::q4_common::f16_to_f32;
use larql_models::quant::ggml::{Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS};

/// Q4_K super-block layout: 144 bytes per 256 values.
const BLOCK_BYTES: usize = Q4_K_BLOCK_BYTES;
/// Number of f32 / i8 elements per Q4_K (and Q8_K) super-block.
const ELEMS_PER_BLOCK: usize = Q4_K_BLOCK_ELEMS;
/// Number of 32-element sub-blocks per super-block.
const SUBBLOCKS_PER_BLOCK: usize = 8;
/// Sub-block size (matches Q4_K's per-32 nibble groups).
const SUBBLOCK_SIZE: usize = 32;

/// Quantised activation in Q8_K layout, one entry per super-block of `x`.
///
/// `qs` packs all super-blocks contiguously: `qs[sb * 256 .. (sb+1) * 256]`
/// is the i8 sub-block stream for super-block `sb`.  `d[sb]` is the f32
/// scale.  `sums[sb * 8 + s]` is the i32 sum of the 32 i8 values in
/// sub-block `s` of super-block `sb` — precomputed once because every
/// row of the matrix needs it for the `mins` term.
pub struct Q8KActivation {
    pub qs: Vec<i8>,
    pub d: Vec<f32>,
    pub sums: Vec<i16>,
}

impl Q8KActivation {
    pub fn n_blocks(&self) -> usize {
        self.d.len()
    }

    /// Allocate an empty Q8KActivation sized for at least `cols` floats.
    /// Used to pre-allocate a reusable buffer in `ExpertScratch` so the
    /// per-expert `quantize_x_to_q8k_into` call doesn't re-allocate at
    /// production sizes.  Rounds `cols` up to the next 256-multiple so
    /// callers don't need to know about Q8_K's super-block geometry —
    /// `quantize_x_to_q8k_into` will resize anyway if the actual input
    /// length differs.
    pub fn with_capacity(cols: usize) -> Self {
        let n_blocks = cols.div_ceil(ELEMS_PER_BLOCK);
        Self {
            qs: vec![0i8; n_blocks * ELEMS_PER_BLOCK],
            d: vec![0.0f32; n_blocks],
            sums: vec![0i16; n_blocks * SUBBLOCKS_PER_BLOCK],
        }
    }
}

/// In-place version of `quantize_x_to_q8k`.  Resizes the output's buffers
/// to match `x.len()` (no-op if already correct), then quantises into
/// them.  Use this from hot paths where the caller owns a long-lived
/// `Q8KActivation` (e.g., per-rayon-thread scratch) so the per-expert
/// activation quantisation doesn't pay an allocator round-trip.
pub fn quantize_x_to_q8k_into(out: &mut Q8KActivation, x: &[f32]) {
    debug_assert_eq!(x.len() % ELEMS_PER_BLOCK, 0);
    let n_blocks = x.len() / ELEMS_PER_BLOCK;
    if out.d.len() != n_blocks {
        out.qs.resize(n_blocks * ELEMS_PER_BLOCK, 0);
        out.d.resize(n_blocks, 0.0);
        out.sums.resize(n_blocks * SUBBLOCKS_PER_BLOCK, 0);
    }

    for sb in 0..n_blocks {
        let base = sb * ELEMS_PER_BLOCK;
        let block = &x[base..base + ELEMS_PER_BLOCK];
        let amax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        out.d[sb] = scale;

        for s in 0..SUBBLOCKS_PER_BLOCK {
            let off = base + s * SUBBLOCK_SIZE;
            let qoff = sb * ELEMS_PER_BLOCK + s * SUBBLOCK_SIZE;
            let mut acc: i32 = 0;
            for j in 0..SUBBLOCK_SIZE {
                let q = (x[off + j] * inv).round().clamp(-127.0, 127.0) as i8;
                out.qs[qoff + j] = q;
                acc += q as i32;
            }
            out.sums[sb * SUBBLOCKS_PER_BLOCK + s] = acc as i16;
        }
    }
}

/// Quantise an activation vector to Q8_K.  `x.len()` must be a multiple of
/// 256.  Per super-block: find absmax, scale by `127 / absmax` (the
/// llama.cpp convention for Q8_K — symmetric int8 with the full
/// `[-127, 127]` range), and store `d = absmax / 127` so reconstruction
/// is `x ≈ d * q`.  Per sub-block of 32: precompute the i32 sum of the
/// quantised values for the dmin term in the matvec.
pub fn quantize_x_to_q8k(x: &[f32]) -> Q8KActivation {
    debug_assert_eq!(x.len() % ELEMS_PER_BLOCK, 0);
    let n_blocks = x.len() / ELEMS_PER_BLOCK;
    let mut qs = vec![0i8; n_blocks * ELEMS_PER_BLOCK];
    let mut d = vec![0.0f32; n_blocks];
    let mut sums = vec![0i16; n_blocks * SUBBLOCKS_PER_BLOCK];

    for sb in 0..n_blocks {
        let base = sb * ELEMS_PER_BLOCK;
        let block = &x[base..base + ELEMS_PER_BLOCK];
        let amax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        d[sb] = scale;

        for s in 0..SUBBLOCKS_PER_BLOCK {
            let off = base + s * SUBBLOCK_SIZE;
            let qoff = sb * ELEMS_PER_BLOCK + s * SUBBLOCK_SIZE;
            let mut acc: i32 = 0;
            for j in 0..SUBBLOCK_SIZE {
                let q = (x[off + j] * inv).round().clamp(-127.0, 127.0) as i8;
                qs[qoff + j] = q;
                acc += q as i32;
            }
            sums[sb * SUBBLOCKS_PER_BLOCK + s] = acc as i16;
        }
    }

    Q8KActivation { qs, d, sums }
}

/// Unpack the 12 packed scale/min bytes at the start of a Q4_K super-block
/// into 8 6-bit scales + 8 6-bit mins.  Matches llama.cpp's
/// `get_scale_min_k4` (and `q4_common::dequantize_q4_k` / `q4k_matvec.rs`).
#[inline(always)]
fn unpack_scales_mins(p: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..4 {
        scales[j] = p[j] & 0x3F;
        mins[j] = p[j + 4] & 0x3F;
        scales[j + 4] = (p[j + 8] & 0x0F) | ((p[j] >> 6) << 4);
        mins[j + 4] = (p[j + 8] >> 4) | ((p[j + 4] >> 6) << 4);
    }
    (scales, mins)
}

/// Scalar reference: `out = W · x` where `W` is `rows × cols` Q4_K and `x`
/// has been pre-quantised to Q8_K.  Mathematically equivalent (within Q8
/// quantisation noise on `x`) to `q4_common::q4k_matvec_into`.
///
/// This is the correctness oracle for the NEON implementation below — both
/// must produce bit-identical output given the same `(W, q8k_x)`.
pub fn q4k_q8k_matvec_scalar(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(q8k_x.qs.len(), cols);
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    if rows == 0 || cols == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if w.len() < rows * row_bytes {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    for (r, out_slot) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
            let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin_w = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (scales, mins) = unpack_scales_mins(&block[4..16]);
            // 16 = 2 (d) + 2 (dmin) + 12 (packed scales/mins).
            // The remaining BLOCK_BYTES-16 = 128 bytes are nibble-packed quants.
            let quants = &block[16..BLOCK_BYTES];

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs = &q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK];
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            // sum1 = Σ_sb scales[sb] · dot_int(q4_nibbles, q8_y)
            // sum2 = Σ_sb mins[sb]   · sum(q8_y in this sb)
            let mut sum1: i32 = 0;
            let mut sum2: i32 = 0;
            for g in 0..4 {
                let sb_lo = 2 * g;
                let sb_hi = 2 * g + 1;
                let chunk = &quants[g * 32..(g + 1) * 32];
                let y_lo = &q8_qs[sb_lo * SUBBLOCK_SIZE..(sb_lo + 1) * SUBBLOCK_SIZE];
                let y_hi = &q8_qs[sb_hi * SUBBLOCK_SIZE..(sb_hi + 1) * SUBBLOCK_SIZE];

                let mut dot_lo: i32 = 0;
                let mut dot_hi: i32 = 0;
                for l in 0..32 {
                    let byte = chunk[l];
                    let q_lo = (byte & 0x0F) as i32;
                    let q_hi = ((byte >> 4) & 0x0F) as i32;
                    dot_lo += q_lo * y_lo[l] as i32;
                    dot_hi += q_hi * y_hi[l] as i32;
                }
                sum1 += scales[sb_lo] as i32 * dot_lo + scales[sb_hi] as i32 * dot_hi;
                sum2 += mins[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mins[sb_hi] as i32 * q8_sums[sb_hi] as i32;
            }
            acc += d_w * d_y * sum1 as f32 - dmin_w * d_y * sum2 as f32;
        }
        *out_slot = acc;
    }
}

/// SDOT (signed 8-bit dot-product, accumulate-into-i32x4) wrapper.
///
/// Computes `acc + Σ_{lane=0..16} a[lane] * b[lane]`, returning an `int32x4_t`
/// where each i32 lane holds the sum of 4 i8 × i8 products.  One ARMv8.2-A
/// `SDOT` instruction; M1+ supports it natively (the `dotprod` target
/// feature is enabled by default for `aarch64-apple-darwin`).
///
/// Implemented via inline asm because `core::arch::aarch64::vdotq_s32` is
/// still gated behind the unstable `stdarch_neon_dotprod` feature on Rust
/// 1.91 (issue rust-lang/rust#117224).  The asm form is stable today.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn sdot_acc(
    acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    let result: std::arch::aarch64::int32x4_t;
    unsafe {
        core::arch::asm!(
            "sdot {0:v}.4s, {1:v}.16b, {2:v}.16b",
            inlateout(vreg) acc => result,
            in(vreg) a,
            in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    result
}

/// Software prefetch hint — bring the cache line containing `ptr` into
/// L1 ahead of an upcoming read. Emits an aarch64 `PRFM PLDL1KEEP` so
/// the data is fetched but tagged as keep-in-L1 (good for hot loops
/// that revisit nearby addresses).
///
/// M3 Max's hardware prefetcher handles linear sequential reads
/// well, but the Q4_K matvec stride (144 bytes per super-block, then
/// jumps to the next row) isn't a simple stride pattern. Explicit
/// hints close ~5-15% of the per-core gap to llama.cpp on these
/// kernels (which has the same hints in its hand-asm path).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(dead_code)] // kept for future re-enablement on harder access patterns; see DIAGNOSIS-2026-05-16-thread-scaling.md
unsafe fn prefetch_l1_keep(ptr: *const u8) {
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{0}]",
            in(reg) ptr,
            options(nostack, readonly, preserves_flags),
        );
    }
}

/// NEON-accelerated `q4k_q8k_matvec` for `aarch64`.  Inner kernel uses
/// `SDOT` (16 i8 × i8 → 4 i32 lanes per instruction) for the integer dot
/// products against the Q8_K activation.  Per-row work per super-block:
/// load 32-byte nibble chunk, mask low / shift high, two SDOT calls per
/// half (16 lanes each), add into per-row f32 accumulator.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q4k_q8k_matvec_neon(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(q8k_x.qs.len(), cols);
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    if rows == 0 || cols == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if w.len() < rows * row_bytes {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    // Mask vector for low-nibble extraction (broadcast 0x0F across 16 lanes).
    let mask_lo = unsafe { vdupq_n_u8(0x0F) };

    // No software prefetch: tested 2026-05-16 with `prfm pldl1keep`
    // hints at per-row and per-super-block granularity. Both regressed
    // single-thread throughput on M3 Max (5.5 vs 5.7 tok/s baseline).
    // The hardware prefetcher handles both the in-row Q4_K stride and
    // the row-to-row jump well enough that software hints compete for
    // L1 fill bandwidth without delivering new data. Kept the
    // `prefetch_l1_keep` helper for future re-enablement on harder
    // access patterns.
    for (r, out_slot) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
            let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin_w = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (scales, mins) = unpack_scales_mins(&block[4..16]);
            let quants_ptr = block[16..].as_ptr();

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            // sum1 = Σ_sb scales[sb] · dot_int(q4_nibbles, q8_y) (i32)
            // sum2 = Σ_sb mins[sb]   ·  Σ q8_y in this sb        (i32)
            //
            // Vector-running accumulator: keep the i32x4 partial sums
            // across all 4 groups in `sum1_v`, only horizontal-reduce
            // once per super-block instead of once per group. Each
            // group's lo/hi partial dot is scaled (vmulq_n_s32) and
            // added into `sum1_v` via vector mla. Eliminates the
            // 4-per-super-block `vaddvq_s32` + scalar mul chain that
            // forced a forced retire of the prior group's SDOTs.
            //
            // Independent SDOT pairs: instead of chaining
            //   acc = sdot(prev, lo1, y_lo1)
            // (which serialises on `prev` at 4-cycle latency), issue
            // both SDOTs into separate destination registers and
            // combine via vaddq_s32. Drops per-half latency from
            // 8 cycles → ~5 cycles on M3's OoO scheduler.
            let zero_v = unsafe { vdupq_n_s32(0) };
            let mut sum1_v = unsafe { vdupq_n_s32(0) };
            let mut sum2_acc: i32 = 0;

            for g in 0..4 {
                let sb_lo = 2 * g;
                let sb_hi = 2 * g + 1;
                // Paired load: 32 nibble bytes in one `ld1.2d` instead
                // of two `ldr`. Same total bandwidth but a single
                // pipeline slot and a clearer hint to the memory
                // subsystem.
                let nibs_pair = unsafe { vld1q_u8_x2(quants_ptr.add(g * 32)) };
                let nib0 = nibs_pair.0;
                let nib1 = nibs_pair.1;

                // Low nibbles → sub-block 2g, high nibbles → sub-block 2g+1.
                let lo0 = unsafe { vreinterpretq_s8_u8(vandq_u8(nib0, mask_lo)) };
                let lo1 = unsafe { vreinterpretq_s8_u8(vandq_u8(nib1, mask_lo)) };
                let hi0 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(nib0, 4)) };
                let hi1 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(nib1, 4)) };

                // Paired loads of the activation halves: 32 bytes
                // for each sub-block (lo + hi). Two `ld1.2d` total.
                let y_lo_pair = unsafe { vld1q_s8_x2(q8_qs_ptr.add(sb_lo * SUBBLOCK_SIZE)) };
                let y_hi_pair = unsafe { vld1q_s8_x2(q8_qs_ptr.add(sb_hi * SUBBLOCK_SIZE)) };
                let y_lo0 = y_lo_pair.0;
                let y_lo1 = y_lo_pair.1;
                let y_hi0 = y_hi_pair.0;
                let y_hi1 = y_hi_pair.1;

                // Independent SDOT pairs: 4 SDOTs into 4 destination
                // registers (no inter-SDOT data dependency), then sum
                // pairs with vaddq.
                let dlo0 = unsafe { sdot_acc(zero_v, lo0, y_lo0) };
                let dlo1 = unsafe { sdot_acc(zero_v, lo1, y_lo1) };
                let dhi0 = unsafe { sdot_acc(zero_v, hi0, y_hi0) };
                let dhi1 = unsafe { sdot_acc(zero_v, hi1, y_hi1) };
                let dlo_acc = unsafe { vaddq_s32(dlo0, dlo1) };
                let dhi_acc = unsafe { vaddq_s32(dhi0, dhi1) };

                // Scale and accumulate into running i32x4. The two
                // vmulq_n_s32 + two vaddq_s32 per group adds ~3 cycles
                // but saves the forced `vaddvq + scalar mul + scalar
                // add` chain (which serialised group g+1 behind it).
                let scaled_lo = unsafe { vmulq_n_s32(dlo_acc, scales[sb_lo] as i32) };
                let scaled_hi = unsafe { vmulq_n_s32(dhi_acc, scales[sb_hi] as i32) };
                sum1_v = unsafe { vaddq_s32(sum1_v, vaddq_s32(scaled_lo, scaled_hi)) };

                // `sum2` stays scalar — the input here is the
                // precomputed Q8_K sums, so no SDOT involved.
                sum2_acc += mins[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mins[sb_hi] as i32 * q8_sums[sb_hi] as i32;
            }
            let sum1 = unsafe { vaddvq_s32(sum1_v) };
            acc += d_w * d_y * sum1 as f32 - dmin_w * d_y * sum2_acc as f32;
        }
        *out_slot = acc;
    }
}

/// Two-row variant of `q4k_q8k_matvec_neon`: processes a pair of output rows
/// per inner loop iteration, sharing the activation Q8_K loads.
///
/// Per super-block: load activation halves once, decode both rows' headers,
/// then emit 16 SDOTs (8 per row) instead of 8 sequential ones.  The doubled
/// in-flight SDOT pressure gives the OoO scheduler more independent work to
/// hide DRAM-load latency on the Q4_K weight stream — the bottleneck the
/// 2026-05-01 profile pinned as the remaining ~70% of per-call time.
///
/// The activation load amortisation is small in raw bytes (256 i8 per
/// super-block, hot in L1) but moves the inner-loop bottleneck from
/// "scheduler stall while waiting for the next nibble byte" toward "SDOT
/// throughput limited" — which is what we want, because SDOT pipes can
/// run two-wide on Apple Silicon.
///
/// Tail handling: if `rows` is odd, the final row falls back to the
/// single-row kernel.  Production matvec dims (`inter=704`, `hidden=2816`)
/// are even so this is a no-op there.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q4k_q8k_matvec_neon_2row(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(q8k_x.qs.len(), cols);
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    if rows == 0 || cols == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if w.len() < rows * row_bytes {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let mask_lo = unsafe { vdupq_n_u8(0x0F) };

    // Pair-of-rows loop: process rows (r, r+1) together.
    let pairs = rows / 2;
    for p in 0..pairs {
        let r0 = 2 * p;
        let r1 = 2 * p + 1;
        let r0_base = r0 * row_bytes;
        let r1_base = r1 * row_bytes;
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        for sb in 0..n_blocks {
            let b0 = &w[r0_base + sb * BLOCK_BYTES..r0_base + (sb + 1) * BLOCK_BYTES];
            let b1 = &w[r1_base + sb * BLOCK_BYTES..r1_base + (sb + 1) * BLOCK_BYTES];
            let d0 = f16_to_f32(u16::from_le_bytes([b0[0], b0[1]]));
            let dmin0 = f16_to_f32(u16::from_le_bytes([b0[2], b0[3]]));
            let d1 = f16_to_f32(u16::from_le_bytes([b1[0], b1[1]]));
            let dmin1 = f16_to_f32(u16::from_le_bytes([b1[2], b1[3]]));
            let (sc0, mn0) = unpack_scales_mins(&b0[4..16]);
            let (sc1, mn1) = unpack_scales_mins(&b1[4..16]);
            let q0 = b0[16..].as_ptr();
            let q1 = b1[16..].as_ptr();

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            let mut s1_0: i32 = 0;
            let mut s2_0: i32 = 0;
            let mut s1_1: i32 = 0;
            let mut s2_1: i32 = 0;

            for grp in 0..4 {
                let sb_lo = 2 * grp;
                let sb_hi = 2 * grp + 1;
                // Activation halves shared across both rows.
                let y_lo0 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_lo * SUBBLOCK_SIZE)) };
                let y_lo1 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_lo * SUBBLOCK_SIZE + 16)) };
                let y_hi0 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_hi * SUBBLOCK_SIZE)) };
                let y_hi1 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_hi * SUBBLOCK_SIZE + 16)) };

                // Row-0 nibble bytes for this 32-byte group.
                let n0a = unsafe { vld1q_u8(q0.add(grp * 32)) };
                let n0b = unsafe { vld1q_u8(q0.add(grp * 32 + 16)) };
                let lo0a = unsafe { vreinterpretq_s8_u8(vandq_u8(n0a, mask_lo)) };
                let lo0b = unsafe { vreinterpretq_s8_u8(vandq_u8(n0b, mask_lo)) };
                let hi0a = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(n0a, 4)) };
                let hi0b = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(n0b, 4)) };

                // Row-1 nibble bytes.
                let n1a = unsafe { vld1q_u8(q1.add(grp * 32)) };
                let n1b = unsafe { vld1q_u8(q1.add(grp * 32 + 16)) };
                let lo1a = unsafe { vreinterpretq_s8_u8(vandq_u8(n1a, mask_lo)) };
                let lo1b = unsafe { vreinterpretq_s8_u8(vandq_u8(n1b, mask_lo)) };
                let hi1a = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(n1a, 4)) };
                let hi1b = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(n1b, 4)) };

                // 16 SDOTs total: 8 per row.  Issue them with the two
                // rows interleaved at the inter-iteration level so the
                // OoO scheduler can dispatch from either stream when one
                // is stalled on a load.
                let zero = unsafe { vdupq_n_s32(0) };
                let dlo_0 = unsafe {
                    let a = sdot_acc(zero, lo0a, y_lo0);
                    sdot_acc(a, lo0b, y_lo1)
                };
                let dlo_1 = unsafe {
                    let a = sdot_acc(zero, lo1a, y_lo0);
                    sdot_acc(a, lo1b, y_lo1)
                };
                let dhi_0 = unsafe {
                    let a = sdot_acc(zero, hi0a, y_hi0);
                    sdot_acc(a, hi0b, y_hi1)
                };
                let dhi_1 = unsafe {
                    let a = sdot_acc(zero, hi1a, y_hi0);
                    sdot_acc(a, hi1b, y_hi1)
                };
                let dot_lo_0 = unsafe { vaddvq_s32(dlo_0) };
                let dot_hi_0 = unsafe { vaddvq_s32(dhi_0) };
                let dot_lo_1 = unsafe { vaddvq_s32(dlo_1) };
                let dot_hi_1 = unsafe { vaddvq_s32(dhi_1) };

                s1_0 += sc0[sb_lo] as i32 * dot_lo_0 + sc0[sb_hi] as i32 * dot_hi_0;
                s2_0 += mn0[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mn0[sb_hi] as i32 * q8_sums[sb_hi] as i32;
                s1_1 += sc1[sb_lo] as i32 * dot_lo_1 + sc1[sb_hi] as i32 * dot_hi_1;
                s2_1 += mn1[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mn1[sb_hi] as i32 * q8_sums[sb_hi] as i32;
            }
            acc0 += d0 * d_y * s1_0 as f32 - dmin0 * d_y * s2_0 as f32;
            acc1 += d1 * d_y * s1_1 as f32 - dmin1 * d_y * s2_1 as f32;
        }
        out[r0] = acc0;
        out[r1] = acc1;
    }

    // Tail: odd row count → process the last row via the single-row kernel.
    if rows % 2 == 1 {
        let r = rows - 1;
        let mut tail_out = [0.0f32; 1];
        let row_w = &w[r * row_bytes..(r + 1) * row_bytes];
        q4k_q8k_matvec_neon(&mut tail_out, q8k_x, row_w, 1, cols);
        out[r] = tail_out[0];
    }
}

/// Hand-asm inner loop (C12 Phase 1): the per-super-block scaled integer dot
/// `sum1 = Σ_sb scale[sb] · Σ_i nibble[sb][i]·y[sb][i]`, computed in one
/// `asm!` block so the schedule is ours, not LLVM's.
///
/// Returns the same i32 `sum1` as the intrinsic / scalar paths (integer math
/// is exact regardless of order), so the f32 epilogue and `sum2` stay in Rust
/// and bit-parity reduces to "does this produce the same `sum1`".
///
/// vs `q4k_q8k_matvec_neon`'s inner loop it kills the 8 scalar `ldrb` scale
/// loads + scalar→vector broadcast: the 8 6-bit scales arrive as two i32x4
/// vectors and the per-sub-block scale is applied with `mul (by element)`.
/// The roofline microbench (`benches/q4k_q8k_matvec.rs`) showed the kernel is
/// compute/issue-bound (~33 cyc/super-block), not DRAM-bound, so cutting
/// issue-port pressure is the lever — see `docs/q4k-decode-kernel.md`
/// §"2026-06-02 roofline measurement".
///
/// Layout (matches `q4k_q8k_matvec_neon` exactly): the 128 nibble bytes walk
/// in 4 groups of 32; group `g` low nibbles → sub-block `2g`, high nibbles →
/// `2g+1`. Activation walks in 4 groups of 64 i8 (two sub-blocks each). Both
/// pointers post-increment through the super-block.
///
/// SAFETY: `quants` must point to ≥128 readable bytes, `act` to ≥256, and
/// `scales` to an 8-element i32 array. Requires the `dotprod` extension (SDOT),
/// baseline on `aarch64-apple-darwin` — same assumption as `sdot_acc`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn q4k_sb_sum1_asm(quants: *const u8, act: *const i8, scales: *const i32) -> i32 {
    let sum1: i32;
    // One group of the unrolled body, parameterised by the two scale lanes
    // (`$sv` = scale vector, `$l0`/`$l1` = lane indices for sub-blocks 2g/2g+1).
    // Single running accumulator `v17`: a 4-private-accumulator variant was
    // tried 2026-06-02 to break the per-super-block RAW chain but showed no
    // reliable gain — the asm/neon ratio swings ±1.5% run-to-run (observed
    // +3.7%..+4.9% for THIS form), larger than any v1↔4-acc difference. The
    // row loop inlines this fn, so the OoO core already overlaps the next
    // super-block's compute with this accumulator chain; the chain isn't the
    // bottleneck. See `docs/q4k-decode-kernel.md` §"Finding — latency-hiding
    // has low headroom".
    macro_rules! grp {
        ($sv:literal, $l0:literal, $l1:literal) => {
            concat!(
                "ld1 {{v0.16b, v1.16b}}, [{q}], #32\n",
                "ld1 {{v20.16b, v21.16b, v22.16b, v23.16b}}, [{a}], #64\n",
                "and  v2.16b, v0.16b, v16.16b\n", // lo0 (sub-block 2g, lanes 0..16)
                "and  v3.16b, v1.16b, v16.16b\n", // lo1 (sub-block 2g, lanes 16..32)
                "ushr v4.16b, v0.16b, #4\n",      // hi0 (sub-block 2g+1)
                "ushr v5.16b, v1.16b, #4\n",      // hi1
                "movi v6.4s, #0\n",
                "movi v7.4s, #0\n",
                "sdot v6.4s, v2.16b, v20.16b\n", // dot[2g]   lanes += lo0·y
                "sdot v6.4s, v3.16b, v21.16b\n", //            += lo1·y
                "sdot v7.4s, v4.16b, v22.16b\n", // dot[2g+1] += hi0·y
                "sdot v7.4s, v5.16b, v23.16b\n", //            += hi1·y
                "mul  v6.4s, v6.4s, ",
                $sv,
                ".s[",
                $l0,
                "]\n", // × scale[2g]
                "mul  v7.4s, v7.4s, ",
                $sv,
                ".s[",
                $l1,
                "]\n", // × scale[2g+1]
                "add  v17.4s, v17.4s, v6.4s\n",
                "add  v17.4s, v17.4s, v7.4s\n",
            )
        };
    }
    unsafe {
        core::arch::asm!(
            "movi v16.16b, #0x0f",                  // nibble mask
            "movi v17.4s, #0",                      // sum1 accumulator (i32x4)
            "ld1 {{v18.4s, v19.4s}}, [{scales}]",   // scales[0..4], scales[4..8]
            grp!("v18", "0", "1"),                  // group 0 → sub-blocks 0,1
            grp!("v18", "2", "3"),                  // group 1 → sub-blocks 2,3
            grp!("v19", "0", "1"),                  // group 2 → sub-blocks 4,5
            grp!("v19", "2", "3"),                  // group 3 → sub-blocks 6,7
            "addv s17, v17.4s",                     // horizontal sum of the 4 lanes
            "fmov {sum1:w}, s17",
            q = inout(reg) quants => _,
            a = inout(reg) act => _,
            scales = in(reg) scales,
            sum1 = out(reg) sum1,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            options(nostack, readonly),
        );
    }
    sum1
}

/// Hand-asm Q4_K × Q8_K matvec (C12 Phase 1). Identical interface and output
/// to [`q4k_q8k_matvec_neon`] — `sum1` comes from [`q4k_sb_sum1_asm`], the
/// `sum2` term and f32 epilogue are the same Rust code, so it is bit-exact
/// with the scalar reference (`q8k_matvec_asm_matches_scalar_bit_exact`).
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q4k_q8k_matvec_asm(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(q8k_x.qs.len(), cols);
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    if rows == 0 || cols == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if w.len() < rows * row_bytes {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    for (r, out_slot) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
            let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin_w = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (scales, mins) = unpack_scales_mins(&block[4..16]);
            let quants_ptr = block[16..].as_ptr();

            // Scales as i32 for the `ld1 {v18,v19}` load inside the asm.
            let sc = [
                scales[0] as i32,
                scales[1] as i32,
                scales[2] as i32,
                scales[3] as i32,
                scales[4] as i32,
                scales[5] as i32,
                scales[6] as i32,
                scales[7] as i32,
            ];

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            // SAFETY: a Q4_K super-block is 144 bytes (16 header + 128 quants),
            // `q8_qs_ptr` spans a full 256-i8 super-block, `sc` is 8 i32.
            let sum1 = unsafe { q4k_sb_sum1_asm(quants_ptr, q8_qs_ptr, sc.as_ptr()) };

            // sum2 stays scalar (precomputed Q8_K sums; no SDOT) — identical
            // to the neon path so the f32 epilogue is bit-for-bit the same.
            let mut sum2_acc: i32 = 0;
            for s in 0..SUBBLOCKS_PER_BLOCK {
                sum2_acc += mins[s] as i32 * q8_sums[s] as i32;
            }
            acc += d_w * d_y * sum1 as f32 - dmin_w * d_y * sum2_acc as f32;
        }
        *out_slot = acc;
    }
}

/// C12 opt-in: route Q4_K × Q8_K matvecs through the hand-asm kernel
/// (`q4k_q8k_matvec_asm`) instead of the intrinsic path when `LARQL_Q4K_ASM`
/// is `1`/`true`. Read once and cached — the env lookup must not land in the
/// per-token hot loop. Default off; both paths are bit-exact.
/// Pure parse of the `LARQL_Q4K_ASM` opt-in value (`1`/`true` → on).
/// Split out so the truth table is unit-testable without touching the
/// process environment or the `OnceLock` cache below.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn q4k_asm_flag_enabled(val: Option<&str>) -> bool {
    matches!(val, Some(v) if v == "1" || v.eq_ignore_ascii_case("true"))
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn use_asm_kernel() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| q4k_asm_flag_enabled(std::env::var("LARQL_Q4K_ASM").ok().as_deref()))
}

/// Public entry point: dispatches to NEON on aarch64, scalar elsewhere.
/// Caller pre-quantises `x` once via `quantize_x_to_q8k` (cost is amortised
/// across all rows of the same matvec, and across all K active experts that
/// share `h_norm`).
pub fn q4k_q8k_matvec_into(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // 2-row variant tried 2026-05-01 — bit-exact (`q8k_matvec_2row_matches_single_row_bit_exact`)
        // but bench-neutral on M3 Max: per-thread is BW-bound on the
        // per-row Q4_K weight stream (1.1 MB at 82 µs ≈ 14 GB/s), and
        // sharing the small activation Q8K (256 B) across 2 rows didn't
        // free real DRAM bandwidth.  Kept as `q4k_q8k_matvec_neon_2row`
        // for future hardware where ILP may dominate over BW.
        // (NB: the "BW-bound" read was overturned 2026-06-02 — the kernel
        // is compute/issue-bound, see `docs/q4k-decode-kernel.md`.)
        //
        // C12: opt-in hand-asm kernel (`LARQL_Q4K_ASM=1`). Bit-exact with
        // the intrinsic path; ~+2.5% isolated. Default off until the
        // two-super-block-interleaved version closes more of the gap and
        // the gate_up fused path gets the same treatment.
        if use_asm_kernel() {
            q4k_q8k_matvec_asm(out, q8k_x, w, rows, cols);
        } else {
            q4k_q8k_matvec_neon(out, q8k_x, w, rows, cols);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: runtime check guarantees AVX2 availability.
        unsafe { q4k_q8k_matvec_avx2(out, q8k_x, w, rows, cols) };
        return;
    }
    #[allow(unreachable_code)]
    q4k_q8k_matvec_scalar(out, q8k_x, w, rows, cols);
}

/// Batched multi-row Q4_K × Q8_K matmul. Computes
/// `out[n * rows + r] = dot(q8k_xs[n], W_row[r])` for every
/// `n in 0..q8k_xs.len()` and `r in 0..rows`.
///
/// Equivalent to looping [`q4k_q8k_matvec_into`] once per input
/// row, but rayon parallelism moves to the **outer N axis** so the
/// per-row matvec dispatch overhead amortises across the whole
/// matmul. Critical for prefill: a 14-token prompt currently fans
/// out 14 separate rayon dispatches per attention projection; this
/// kernel collapses that to one dispatch with N=14 tasks.
///
/// `out.len()` must equal `q8k_xs.len() * rows`. Row-major output
/// (`out[n * rows + r]`).
///
/// Task 2c of `vindex-qwen35moe-reader` — wait, no, **Arc 1** from
/// RESUME_PROMPT's open-levers list. Closes the prefill speed gap
/// to llama.cpp by eliminating per-row rayon dispatch overhead.
pub fn q4k_q8k_matmul_into(
    out: &mut [f32],
    q8k_xs: &[Q8KActivation],
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    let n = q8k_xs.len();
    if n == 0 || rows == 0 || cols == 0 {
        out.fill(0.0);
        return;
    }
    assert_eq!(
        out.len(),
        n * rows,
        "q4k_q8k_matmul_into: out.len() ({}) must equal n_inputs ({n}) * rows ({rows})",
        out.len()
    );

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: runtime check guarantees AVX2 availability for
        // every rayon worker.
        unsafe { q4k_q8k_matmul_avx2(out, q8k_xs, w, rows, cols) };
        return;
    }

    // Portable fallback: serial outer loop, per-row matvec
    // (which itself may go parallel on the inner row axis).
    for (n_idx, q8k) in q8k_xs.iter().enumerate() {
        let out_row = &mut out[n_idx * rows..(n_idx + 1) * rows];
        q4k_q8k_matvec_into(out_row, q8k, w, rows, cols);
    }
}

/// Outer-N parallel matmul on x86_64 AVX2. Each rayon task handles
/// one input row against the full weight matrix, using the
/// **serial** per-row AVX2 path (no nested rayon dispatch). N
/// (input rows) is the parallelism level — typically prefill seq
/// length, 14-200 tokens.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_q8k_matmul_avx2(
    out: &mut [f32],
    q8k_xs: &[Q8KActivation],
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    let expected_w = rows * row_bytes;
    if w.len() < expected_w {
        out.fill(0.0);
        return;
    }

    use rayon::prelude::*;
    out.par_chunks_mut(rows)
        .enumerate()
        .for_each(|(n_idx, out_row)| {
            let q8k = &q8k_xs[n_idx];
            for (r, out_slot) in out_row.iter_mut().enumerate().take(rows) {
                // SAFETY: outer fn is `target_feature(avx2)`; rayon
                // workers inherit the same target features at
                // compile time; runtime check at the public entry
                // guarantees AVX2 availability.
                unsafe {
                    compute_row_q4k_avx2(out_slot, r, q8k, w, n_blocks, row_bytes);
                }
            }
        });
}

/// AVX2 Q4_K × Q8_K matvec for x86_64.
///
/// `vpmaddubsw` (unsigned×signed 8-bit → adjacent-pair-summed 16-bit) replaces
/// 32 scalar multiplies per 32-element group.  `vpmaddwd` widens to 32-bit.
/// On AMD EPYC / Intel Haswell+ this is ~12–16× faster than the scalar path.
///
/// Bit-equivalence with the scalar reference is verified in unit tests below.
/// Per-row Q4_K × Q8_K dot product (AVX2). Computes `out_slot = sum over
/// super-blocks` for row `r` of the weight matrix. Disjoint per-row state
/// makes this parallelisable — see [`q4k_q8k_matvec_avx2`] for the rayon
/// dispatch.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compute_row_q4k_avx2(
    out_slot: &mut f32,
    r: usize,
    q8k_x: &Q8KActivation,
    w: &[u8],
    n_blocks: usize,
    row_bytes: usize,
) {
    use std::arch::x86_64::*;
    let lo_mask = _mm256_set1_epi8(0x0F);
    let ones_epi16 = _mm256_set1_epi16(1);
    let row_base = r * row_bytes;
    let mut acc = 0.0f32;

    for sb in 0..n_blocks {
        let block = &w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
        let d_w = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin_w = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (scales, mins) = unpack_scales_mins(&block[4..16]);
        let quants = &block[16..BLOCK_BYTES];
        let q8_base = sb * ELEMS_PER_BLOCK;
        let q8_qs = &q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK];
        let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
        let d_y = q8k_x.d[sb];

        let mut sum1: i32 = 0;
        let mut sum2: i32 = 0;

        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;

            let q4 = _mm256_loadu_si256(quants.as_ptr().add(g * 32) as *const __m256i);
            let lo_nibbles = _mm256_and_si256(q4, lo_mask);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(q4, 4), lo_mask);

            let y_lo =
                _mm256_loadu_si256(q8_qs.as_ptr().add(sb_lo * SUBBLOCK_SIZE) as *const __m256i);
            let y_hi =
                _mm256_loadu_si256(q8_qs.as_ptr().add(sb_hi * SUBBLOCK_SIZE) as *const __m256i);

            let dot_lo = hsum_i32x8(_mm256_madd_epi16(
                _mm256_maddubs_epi16(lo_nibbles, y_lo),
                ones_epi16,
            ));
            let dot_hi = hsum_i32x8(_mm256_madd_epi16(
                _mm256_maddubs_epi16(hi_nibbles, y_hi),
                ones_epi16,
            ));

            sum1 += scales[sb_lo] as i32 * dot_lo + scales[sb_hi] as i32 * dot_hi;
            sum2 += mins[sb_lo] as i32 * q8_sums[sb_lo] as i32
                + mins[sb_hi] as i32 * q8_sums[sb_hi] as i32;
        }
        acc += d_w * d_y * sum1 as f32 - dmin_w * d_y * sum2 as f32;
    }
    *out_slot = acc;
}

/// Rayon-parallel row dispatch for AVX2 Q4_K × Q8_K matvec.
///
/// The row loop is embarrassingly parallel — each row reads its own slice of
/// `w` and writes a single `out` element, with `q8k_x` shared read-only. On a
/// 48-core host, splitting `rows` across cores converts the previously
/// single-threaded ~17 Gelem/s bottleneck into a near-linear scale-out at the
/// matvec sizes used in Gemma 3 4B decode (rows in 1024..10240).
///
/// Per-row reduction order is preserved — bit-exactness vs the scalar
/// reference still holds because the inner accumulators stay row-local.
///
/// Small matvecs (rows < `MIN_PAR_ROWS`) skip rayon to avoid the per-task
/// dispatch overhead. The threshold is picked so small-batch unit tests
/// (rows in 5..7) run sequentially — they're the existing bit-exact and
/// canonical-dequant correctness oracles.
const MIN_PAR_ROWS: usize = 16;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_q8k_matvec_avx2(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    if rows == 0 || cols == 0 || w.len() < rows * (cols / ELEMS_PER_BLOCK) * BLOCK_BYTES {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;

    if rows < MIN_PAR_ROWS {
        for (r, out_slot) in out.iter_mut().enumerate().take(rows) {
            compute_row_q4k_avx2(out_slot, r, q8k_x, w, n_blocks, row_bytes);
        }
        return;
    }

    use rayon::prelude::*;
    let chunk_rows = rows.div_ceil(rayon::current_num_threads().max(1)).max(4);
    out[..rows]
        .par_chunks_mut(chunk_rows)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start_row = chunk_idx * chunk_rows;
            for (i, out_slot) in chunk.iter_mut().enumerate() {
                let r = start_row + i;
                // SAFETY: outer fn requires AVX2 (target_feature); caller's
                // runtime detection precondition holds for every thread.
                unsafe {
                    compute_row_q4k_avx2(out_slot, r, q8k_x, w, n_blocks, row_bytes);
                }
            }
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32x8(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let v128 = _mm_add_epi32(lo, hi);
    let v64 = _mm_add_epi32(v128, _mm_srli_si128(v128, 8));
    let v32 = _mm_add_epi32(v64, _mm_srli_si128(v64, 4));
    _mm_cvtsi128_si32(v32)
}

// ── Q4_KF × Q8_K matvec ──────────────────────────────────────────────────────
//
// Q4_KF super-block: 160 bytes per 256 values, "pre-baked" Q4_K variant where
// the per-sub-block scales and mins have already been multiplied by the
// super-block's d and dmin:
//
//   [0..16]   16 bytes: 8 × f16  d*scale[j]
//   [16..32]  16 bytes: 8 × f16  dmin*min[j]
//   [32..160] 128 bytes nibbles (identical layout to Q4_K's [16..144])
//
// Nibble layout: four groups of 32 bytes, each group covers two sub-blocks
// 2g (low nibble) and 2g+1 (high nibble). Same as Q4_K, same as `q4k_q8k`
// inner loop — the only difference vs `q4k_q8k_matvec` is that scales are
// f32 (decoded from f16, pre-multiplied) rather than i8 scales paired with
// f16 d/dmin. Math:
//
//   acc += d_y * (Σ_g  d*scale[g] · dot_g  -  Σ_g  dmin*min[g] · q8_sum[g])
//
// where `dot_g = Σ_{i ∈ 32-elem sub-block g} nibble[i] · q8_qs[i]` (i32) and
// `q8_sum[g]` is the precomputed sum of q8 lanes for that sub-block.

const Q4KF_BLOCK_BYTES: usize = crate::pipeline::Q4_KF_BLOCK_BYTES;

/// Scalar reference: Q4_KF weights × Q8_K activation matvec. Correctness
/// oracle for the AVX2 implementation below.
pub fn q4kf_q8k_matvec_scalar(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q4KF_BLOCK_BYTES;
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if rows == 0 || cols == 0 || w.len() < rows * row_bytes {
        return;
    }
    for (r, out_r) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * Q4KF_BLOCK_BYTES..];
            // Pre-baked scales and mins (8 each, f16 → f32).
            let mut scales = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            for j in 0..8 {
                let s_bits = u16::from_le_bytes([block[j * 2], block[j * 2 + 1]]);
                let m_bits = u16::from_le_bytes([block[16 + j * 2], block[16 + j * 2 + 1]]);
                scales[j] = f16_to_f32(s_bits);
                mins[j] = f16_to_f32(m_bits);
            }
            let quants = &block[32..Q4KF_BLOCK_BYTES];
            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs = &q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK];
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            for g in 0..4 {
                let sb_lo = 2 * g;
                let sb_hi = 2 * g + 1;
                let chunk = &quants[g * 32..(g + 1) * 32];
                let y_lo = &q8_qs[sb_lo * SUBBLOCK_SIZE..(sb_lo + 1) * SUBBLOCK_SIZE];
                let y_hi = &q8_qs[sb_hi * SUBBLOCK_SIZE..(sb_hi + 1) * SUBBLOCK_SIZE];

                let mut dot_lo: i32 = 0;
                let mut dot_hi: i32 = 0;
                for l in 0..32 {
                    let byte = chunk[l];
                    let q_lo = (byte & 0x0F) as i32;
                    let q_hi = ((byte >> 4) & 0x0F) as i32;
                    dot_lo += q_lo * y_lo[l] as i32;
                    dot_hi += q_hi * y_hi[l] as i32;
                }
                acc += d_y
                    * (scales[sb_lo] * dot_lo as f32 + scales[sb_hi] * dot_hi as f32
                        - mins[sb_lo] * q8_sums[sb_lo] as f32
                        - mins[sb_hi] * q8_sums[sb_hi] as f32);
            }
        }
        *out_r = acc;
    }
}

/// AVX2 Q4_KF × Q8_K matvec for x86_64. Same `vpmaddubsw` / `vpmaddwd`
/// inner loop as `q4k_q8k_matvec_avx2`; differs only in scale handling
/// (f32 pre-baked vs Q4_K's i8 × f16 split). Bit-equivalent to the scalar
/// reference modulo f32 reduction order (verified in unit tests).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4kf_q8k_matvec_avx2(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::x86_64::*;

    if rows == 0 || cols == 0 || w.len() < rows * (cols / ELEMS_PER_BLOCK) * Q4KF_BLOCK_BYTES {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q4KF_BLOCK_BYTES;
    let lo_mask = _mm256_set1_epi8(0x0F);
    let ones_epi16 = _mm256_set1_epi16(1);

    for (r, out_slot) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;

        for sb in 0..n_blocks {
            let block =
                &w[row_base + sb * Q4KF_BLOCK_BYTES..row_base + (sb + 1) * Q4KF_BLOCK_BYTES];
            // Decode 8 f16 scales + 8 f16 mins (pre-baked d*scale, dmin*min).
            let mut scales = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            for j in 0..8 {
                let s_bits = u16::from_le_bytes([block[j * 2], block[j * 2 + 1]]);
                let m_bits = u16::from_le_bytes([block[16 + j * 2], block[16 + j * 2 + 1]]);
                scales[j] = f16_to_f32(s_bits);
                mins[j] = f16_to_f32(m_bits);
            }
            let quants = &block[32..Q4KF_BLOCK_BYTES];
            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs = &q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK];
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            for g in 0..4 {
                let sb_lo = 2 * g;
                let sb_hi = 2 * g + 1;

                let q4 = _mm256_loadu_si256(quants.as_ptr().add(g * 32) as *const __m256i);
                let lo_nibbles = _mm256_and_si256(q4, lo_mask);
                let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(q4, 4), lo_mask);

                let y_lo =
                    _mm256_loadu_si256(q8_qs.as_ptr().add(sb_lo * SUBBLOCK_SIZE) as *const __m256i);
                let y_hi =
                    _mm256_loadu_si256(q8_qs.as_ptr().add(sb_hi * SUBBLOCK_SIZE) as *const __m256i);

                let dot_lo = hsum_i32x8(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(lo_nibbles, y_lo),
                    ones_epi16,
                )) as f32;
                let dot_hi = hsum_i32x8(_mm256_madd_epi16(
                    _mm256_maddubs_epi16(hi_nibbles, y_hi),
                    ones_epi16,
                )) as f32;

                acc += d_y
                    * (scales[sb_lo] * dot_lo + scales[sb_hi] * dot_hi
                        - mins[sb_lo] * q8_sums[sb_lo] as f32
                        - mins[sb_hi] * q8_sums[sb_hi] as f32);
            }
        }
        *out_slot = acc;
    }
}

/// Public entry: AVX2 on x86_64 (when available), scalar otherwise. `w` is
/// a Q4_KF weight matrix of `rows × cols`; `q8k_x` is the pre-quantised
/// activation. Caller is responsible for `cols % 256 == 0`.
pub fn q4kf_q8k_matvec_into(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: avx2 detected; bounds validated inside.
        unsafe { q4kf_q8k_matvec_avx2(out, q8k_x, w, rows, cols) };
        return;
    }
    q4kf_q8k_matvec_scalar(out, q8k_x, w, rows, cols);
}

/// Fused gate+up matvec: produce two output vectors from two weight matrices
/// against the SAME pre-quantised Q8_K activation in one pass.  Each
/// super-block of `q8k_x` is loaded once and SDOT'd against both `gate_w`
/// and `up_w` per row — gate and up SDOTs interleave on the OoO engine,
/// hiding cross-instruction latency that the back-to-back independent
/// `q4k_q8k_matvec_into` calls couldn't.
///
/// Caller layouts: `gate_w.len() == up_w.len() == rows * (cols / 256) * 144`,
/// `gate_out.len() == up_out.len() == rows`.
pub fn q4k_q8k_gate_up_into(
    gate_out: &mut [f32],
    up_out: &mut [f32],
    q8k_x: &Q8KActivation,
    gate_w: &[u8],
    up_w: &[u8],
    rows: usize,
    cols: usize,
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        q4k_q8k_gate_up_neon(gate_out, up_out, q8k_x, gate_w, up_w, rows, cols);
        return;
    }
    #[allow(unreachable_code)]
    {
        // Fallback (covers x86_64 with AVX2, and any other target): two
        // independent matvecs through the AVX2-dispatched entry point.
        // The aarch64 NEON `q4k_q8k_gate_up_neon` above is preserved as a
        // bespoke interleaved kernel for completeness, but on x86_64 the
        // OoO engine extracts enough ILP from two sequential
        // `q4k_q8k_matvec_into` calls that a manually-interleaved fused
        // kernel doesn't pay (see `moe/expert.rs:466-471` for the same
        // empirical observation on M3 Max). The critical fix vs the prior
        // scalar fallback: this now reaches AVX2 — gate+up dropped from
        // ~57 ms (2× scalar) to ~3 ms (2× AVX2) at the
        // `prefill_10240` Q4_K shape.
        q4k_q8k_matvec_into(gate_out, q8k_x, gate_w, rows, cols);
        q4k_q8k_matvec_into(up_out, q8k_x, up_w, rows, cols);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn q4k_q8k_gate_up_neon(
    gate_out: &mut [f32],
    up_out: &mut [f32],
    q8k_x: &Q8KActivation,
    gate_w: &[u8],
    up_w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(gate_out.len(), rows);
    debug_assert_eq!(up_out.len(), rows);
    debug_assert_eq!(q8k_x.qs.len(), cols);
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    if rows == 0 || cols == 0 {
        for v in gate_out.iter_mut() {
            *v = 0.0;
        }
        for v in up_out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * BLOCK_BYTES;
    if gate_w.len() < rows * row_bytes || up_w.len() < rows * row_bytes {
        for v in gate_out.iter_mut() {
            *v = 0.0;
        }
        for v in up_out.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let mask_lo = unsafe { vdupq_n_u8(0x0F) };

    for r in 0..rows {
        let row_base = r * row_bytes;
        let mut acc_g = 0.0f32;
        let mut acc_u = 0.0f32;
        for sb in 0..n_blocks {
            let g_block = &gate_w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
            let u_block = &up_w[row_base + sb * BLOCK_BYTES..row_base + (sb + 1) * BLOCK_BYTES];
            let d_g = f16_to_f32(u16::from_le_bytes([g_block[0], g_block[1]]));
            let dmin_g = f16_to_f32(u16::from_le_bytes([g_block[2], g_block[3]]));
            let d_u = f16_to_f32(u16::from_le_bytes([u_block[0], u_block[1]]));
            let dmin_u = f16_to_f32(u16::from_le_bytes([u_block[2], u_block[3]]));
            let (sc_g, mn_g) = unpack_scales_mins(&g_block[4..16]);
            let (sc_u, mn_u) = unpack_scales_mins(&u_block[4..16]);
            let q_g = g_block[16..].as_ptr();
            let q_u = u_block[16..].as_ptr();

            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs_ptr = q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK].as_ptr();
            let q8_sums = &q8k_x.sums[sb * SUBBLOCKS_PER_BLOCK..(sb + 1) * SUBBLOCKS_PER_BLOCK];
            let d_y = q8k_x.d[sb];

            let mut s1_g: i32 = 0;
            let mut s2_g: i32 = 0;
            let mut s1_u: i32 = 0;
            let mut s2_u: i32 = 0;

            for grp in 0..4 {
                let sb_lo = 2 * grp;
                let sb_hi = 2 * grp + 1;
                // Activation halves shared between gate and up.
                let y_lo0 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_lo * SUBBLOCK_SIZE)) };
                let y_lo1 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_lo * SUBBLOCK_SIZE + 16)) };
                let y_hi0 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_hi * SUBBLOCK_SIZE)) };
                let y_hi1 = unsafe { vld1q_s8(q8_qs_ptr.add(sb_hi * SUBBLOCK_SIZE + 16)) };

                let gnib0 = unsafe { vld1q_u8(q_g.add(grp * 32)) };
                let gnib1 = unsafe { vld1q_u8(q_g.add(grp * 32 + 16)) };
                let glo0 = unsafe { vreinterpretq_s8_u8(vandq_u8(gnib0, mask_lo)) };
                let glo1 = unsafe { vreinterpretq_s8_u8(vandq_u8(gnib1, mask_lo)) };
                let ghi0 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(gnib0, 4)) };
                let ghi1 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(gnib1, 4)) };

                let unib0 = unsafe { vld1q_u8(q_u.add(grp * 32)) };
                let unib1 = unsafe { vld1q_u8(q_u.add(grp * 32 + 16)) };
                let ulo0 = unsafe { vreinterpretq_s8_u8(vandq_u8(unib0, mask_lo)) };
                let ulo1 = unsafe { vreinterpretq_s8_u8(vandq_u8(unib1, mask_lo)) };
                let uhi0 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(unib0, 4)) };
                let uhi1 = unsafe { vreinterpretq_s8_u8(vshrq_n_u8(unib1, 4)) };

                // 8 SDOTs per group, gate / up issued back-to-back so the
                // OoO engine can dispatch them on different ports.
                let zero = unsafe { vdupq_n_s32(0) };
                let g_dlo = unsafe {
                    let a = sdot_acc(zero, glo0, y_lo0);
                    sdot_acc(a, glo1, y_lo1)
                };
                let u_dlo = unsafe {
                    let a = sdot_acc(zero, ulo0, y_lo0);
                    sdot_acc(a, ulo1, y_lo1)
                };
                let g_dhi = unsafe {
                    let a = sdot_acc(zero, ghi0, y_hi0);
                    sdot_acc(a, ghi1, y_hi1)
                };
                let u_dhi = unsafe {
                    let a = sdot_acc(zero, uhi0, y_hi0);
                    sdot_acc(a, uhi1, y_hi1)
                };

                let g_dot_lo = unsafe { vaddvq_s32(g_dlo) };
                let g_dot_hi = unsafe { vaddvq_s32(g_dhi) };
                let u_dot_lo = unsafe { vaddvq_s32(u_dlo) };
                let u_dot_hi = unsafe { vaddvq_s32(u_dhi) };

                s1_g += sc_g[sb_lo] as i32 * g_dot_lo + sc_g[sb_hi] as i32 * g_dot_hi;
                s2_g += mn_g[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mn_g[sb_hi] as i32 * q8_sums[sb_hi] as i32;
                s1_u += sc_u[sb_lo] as i32 * u_dot_lo + sc_u[sb_hi] as i32 * u_dot_hi;
                s2_u += mn_u[sb_lo] as i32 * q8_sums[sb_lo] as i32
                    + mn_u[sb_hi] as i32 * q8_sums[sb_hi] as i32;
            }
            acc_g += d_g * d_y * s1_g as f32 - dmin_g * d_y * s2_g as f32;
            acc_u += d_u * d_y * s1_u as f32 - dmin_u * d_y * s2_u as f32;
        }
        gate_out[r] = acc_g;
        up_out[r] = acc_u;
    }
}

// ── Q6_K × Q8_K matvec ───────────────────────────────────────────────────────
//
// Q6_K super-block: 210 bytes per 256 values, llama.cpp wire format.
// Matches `quantize_q6_k` in this crate and the canonical
// `larql_models::quant::ggml::dequantize_q6_k`.
//
//   [0..128]   128 bytes: ql — lo4 bits, interleaved-stride layout
//   [128..192]  64 bytes: qh — hi2 bits, packed 4 per byte (2 bits each)
//   [192..208]  16 bytes: scales — one int8 per 16 elements
//   [208..210]   2 bytes: d — f16 super-block scale
//
// Layout: two halves of 128 elements each. Per half, for l in 0..32:
//   y[l + 0]  ← (ql[l]     & 0xF) | (((qh[l] >> 0) & 3) << 4) − 32   scale sc[is+0]
//   y[l + 32] ← (ql[l+32]  & 0xF) | (((qh[l] >> 2) & 3) << 4) − 32   scale sc[is+2]
//   y[l + 64] ← (ql[l]     >> 4 ) | (((qh[l] >> 4) & 3) << 4) − 32   scale sc[is+4]
//   y[l + 96] ← (ql[l+32]  >> 4 ) | (((qh[l] >> 6) & 3) << 4) − 32   scale sc[is+6]
// where is = l/16. Half h uses ql[h*64..], qh[h*32..], sc[h*8..].
//
// This is the SAME interleaved layout produced by `quantize_q6_k` and read by
// `dequantize_row_q6_K` in llama.cpp. The previous sequential layout produced
// garbled dot products on vindex-extracted Q6_K (off by ~7% on smooth input,
// much worse on real weights).

/// Q6_K super-block size in bytes (re-export of the wire-format constant).
const Q6K_BLOCK_BYTES: usize = larql_models::quant::ggml::Q6_K_BLOCK_BYTES;

/// Scalar reference: Q6_K weights × Q8_K activation matvec.
/// Reads the llama.cpp Q6_K wire format directly.
///
/// `>> 0` in the four `qh_byte >> {0,2,4,6}` extractions is kept for
/// regularity with the other 2-bit slots.
#[allow(clippy::identity_op)]
pub fn q6k_q8k_matvec_scalar(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if rows == 0 || cols == 0 || w.len() < rows * row_bytes {
        return;
    }
    for (r, out_r) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * Q6K_BLOCK_BYTES..];
            let ql = &block[0..128];
            let qh = &block[128..192];
            let sc = &block[192..208]; // 16 × int8
            let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let d_y = q8k_x.d[sb];
            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_qs = &q8k_x.qs[q8_base..q8_base + ELEMS_PER_BLOCK];

            let mut sum1: i32 = 0;
            for half in 0..2usize {
                let ql_off = half * 64;
                let qh_off = half * 32;
                let sc_off = half * 8;
                let y_off = half * 128;
                for l in 0..32usize {
                    let is = l / 16;
                    let qh_byte = qh[qh_off + l] as i32;
                    let q1 =
                        (((ql[ql_off + l] & 0x0F) as i32) | (((qh_byte >> 0) & 0x03) << 4)) - 32;
                    let q2 = (((ql[ql_off + l + 32] & 0x0F) as i32)
                        | (((qh_byte >> 2) & 0x03) << 4))
                        - 32;
                    let q3 = (((ql[ql_off + l] >> 4) as i32) | (((qh_byte >> 4) & 0x03) << 4)) - 32;
                    let q4 =
                        (((ql[ql_off + l + 32] >> 4) as i32) | (((qh_byte >> 6) & 0x03) << 4)) - 32;
                    let s0 = sc[sc_off + is] as i8 as i32;
                    let s1 = sc[sc_off + is + 2] as i8 as i32;
                    let s2 = sc[sc_off + is + 4] as i8 as i32;
                    let s3 = sc[sc_off + is + 6] as i8 as i32;
                    sum1 += s0 * q1 * q8_qs[y_off + l] as i32;
                    sum1 += s1 * q2 * q8_qs[y_off + l + 32] as i32;
                    sum1 += s2 * q3 * q8_qs[y_off + l + 64] as i32;
                    sum1 += s3 * q4 * q8_qs[y_off + l + 96] as i32;
                }
            }
            acc += d_w * d_y * sum1 as f32;
        }
        *out_r = acc;
    }
}

/// NEON-accelerated Q6_K × Q8_K matvec for `aarch64`.
///
/// WARNING: This implementation reads the legacy "sequential nibble"
/// Q6_K layout (scale group `g` uses ql[g*8..(g+1)*8] / qh[g*4..(g+1)*4]
/// for elements `g*16..(g+1)*16`). The on-disk Q6_K wire format produced
/// by `quantize_q6_k` and read by `larql_models::quant::ggml::dequantize_q6_k`
/// uses the llama.cpp interleaved-stride layout, which is incompatible.
///
/// `q6k_q8k_matvec_into` therefore dispatches to the scalar path on all
/// targets until this kernel is re-vectorised against the canonical
/// layout. See follow-up issue. Kept for reference; not in the production
/// dispatch graph.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[allow(dead_code)]
pub fn q6k_q8k_matvec_neon(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    use std::arch::aarch64::*;

    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if rows == 0 || cols == 0 || w.len() < rows * row_bytes {
        return;
    }

    // Shift-right pattern for hi2 extraction: 0, -2, -4, -6 repeated 4×.
    // vshlq_s8 with negative b shifts right: out[i] = a[i] >> (-b[i]).
    const SHIFT_RIGHT: [i8; 16] = [0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6];
    let shift_v = unsafe { vld1q_s8(SHIFT_RIGHT.as_ptr()) };
    let mask_0f = unsafe { vdupq_n_u8(0x0F) };
    let mask_03 = unsafe { vdupq_n_u8(0x03) };
    let sub32 = unsafe { vdupq_n_s8(32) };

    // No software prefetch — see q4k_q8k_matvec_neon for the rationale.
    for (r, out_r) in out.iter_mut().enumerate().take(rows) {
        let row_base = r * row_bytes;
        let mut acc = 0.0f32;
        for sb in 0..n_blocks {
            let block = &w[row_base + sb * Q6K_BLOCK_BYTES..];
            let ql_base = block.as_ptr();
            let qh_base = unsafe { block.as_ptr().add(128) };
            let sc_base = unsafe { block.as_ptr().add(192) as *const i8 };
            let d_w = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            let d_y = q8k_x.d[sb];
            let q8_base = sb * ELEMS_PER_BLOCK;
            let q8_ptr = q8k_x.qs.as_ptr();

            let mut sum1: i32 = 0;

            for g in 0..16usize {
                // Scale group g covers elements g*16..(g+1)*16.
                // ql bytes for group g: ql[g*8..(g+1)*8] (8 bytes → 16 nibbles).
                // qh bytes for group g: qh[g*4..(g+1)*4] (4 bytes → 16 × 2-bit).
                let ql_g = unsafe { ql_base.add(g * 8) };
                let qh_g = unsafe { qh_base.add(g * 4) };
                let q8_g = unsafe { q8_ptr.add(q8_base + g * 16) };
                let scale = unsafe { *sc_base.add(g) as i32 };

                // ── Lo4 extraction (8 ql bytes → 16 uint4 values, in element order) ──
                // ql_v[j] holds lo4 of element 2j (low nibble) and 2j+1 (high nibble).
                let ql_v = unsafe { vld1_u8(ql_g) };
                let lo4_even = unsafe { vand_u8(ql_v, vget_low_u8(mask_0f)) }; // elements 0,2,4,...,14
                let lo4_odd = unsafe { vshr_n_u8(ql_v, 4) }; // elements 1,3,5,...,15
                                                             // Interleave to restore element order: [e0,e1,e2,...,e15].
                let zip = unsafe { vzip_u8(lo4_even, lo4_odd) };
                let lo4_v = unsafe { vcombine_u8(zip.0, zip.1) }; // uint8x16_t

                // ── Hi2 extraction (4 qh bytes → 16 uint2 values) ──
                // Each qh byte j holds hi2 for elements 4j+0..4j+3 in bits 0-1,2-3,4-5,6-7.
                // Build a 16-byte vector with each qh byte replicated 4 times, then
                // shift right by [0,2,4,6, 0,2,4,6, ...] and mask to 2 bits.
                let (q0, q1, q2, q3) = unsafe {
                    (
                        (*qh_g) as u32 * 0x01010101u32,
                        (*qh_g.add(1)) as u32 * 0x01010101u32,
                        (*qh_g.add(2)) as u32 * 0x01010101u32,
                        (*qh_g.add(3)) as u32 * 0x01010101u32,
                    )
                };
                let qh_rep: uint8x16_t = unsafe {
                    vreinterpretq_u8_u32(vcombine_u32(
                        vreinterpret_u32_u64(vcreate_u64((q0 as u64) | ((q1 as u64) << 32))),
                        vreinterpret_u32_u64(vcreate_u64((q2 as u64) | ((q3 as u64) << 32))),
                    ))
                };
                // Variable right-shift then mask to 2 bits.
                let hi2_v = unsafe {
                    vandq_u8(
                        vreinterpretq_u8_s8(vshlq_s8(vreinterpretq_s8_u8(qh_rep), shift_v)),
                        mask_03,
                    )
                };

                // ── Combine → signed int8 weight values ──
                // raw6 = lo4 | (hi2 << 4) ∈ [0..63]; signed = raw6 - 32 ∈ [-32..31].
                let hi2_shifted = unsafe { vshlq_n_u8(hi2_v, 4) };
                let combined = unsafe { vorrq_u8(lo4_v, hi2_shifted) };
                let q6_raw: int8x16_t = unsafe { vsubq_s8(vreinterpretq_s8_u8(combined), sub32) };

                // ── SDOT: 16 × (q6_raw[i] * q8k[i]) → 4 partial i32 sums ──
                let q8_v = unsafe { vld1q_s8(q8_g) };
                let dot_v = unsafe { sdot_acc(vdupq_n_s32(0), q6_raw, q8_v) };
                let dot = unsafe { vaddvq_s32(dot_v) };

                sum1 += scale * dot;
            }

            acc += d_w * d_y * sum1 as f32;
        }
        *out_r = acc;
    }
}

/// AVX2 Q6_K × Q8_K matvec for x86_64.
///
/// Per super-block, per half (128 elements): load 64 ql bytes (two 32-byte
/// chunks) + 32 qh bytes, derive the four 32-element signed-i8 q6 strides
/// at positions {0, 32, 64, 96} within the half via nibble + hi-2-bit
/// extraction + `- 32` lift. Per stride: `_mm256_sign_epi8` flips q8's
/// sign to match q6's, `_mm256_maddubs_epi16` accumulates 16 i16 pair-
/// sums, `_mm256_madd_epi16(_, ones)` widens to 8 i32 lanes. The 8 i32
/// lanes split 4+4 across the two 16-element sub-blocks of the stride,
/// each scaled by its own int8 `sc` entry. Bit-equivalent to the scalar
/// reference (verified in tests).
/// Per-row Q6_K × Q8_K dot product (AVX2). Row-disjoint state — see
/// [`q6k_q8k_matvec_avx2`] for rayon dispatch over rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// The `for g in 0..4 { q6_stride[g] ... q8_ptr.add(... + g * 32) }` loop
// also uses `g` as a stride multiplier for sibling pointer arithmetic,
// so `iter().enumerate()` would force splitting the body across two
// bindings to no benefit. The fixed array makes the bounds trivially
// known to the compiler.
#[allow(clippy::needless_range_loop)]
unsafe fn compute_row_q6k_avx2(
    out_r: &mut f32,
    r: usize,
    q8k_x: &Q8KActivation,
    w: &[u8],
    n_blocks: usize,
    row_bytes: usize,
) {
    use std::arch::x86_64::*;
    let lo4_mask = _mm256_set1_epi8(0x0F);
    let hi2_mask = _mm256_set1_epi8(0x03);
    let neg32 = _mm256_set1_epi8(32);
    let ones_16 = _mm256_set1_epi16(1);
    let row_base = r * row_bytes;
    let mut acc = 0.0f32;

    for sb in 0..n_blocks {
        let block = w.as_ptr().add(row_base + sb * Q6K_BLOCK_BYTES);
        let ql_ptr = block;
        let qh_ptr = block.add(128);
        let sc_ptr = block.add(192) as *const i8;
        let d_w = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
        let d_y = q8k_x.d[sb];
        let q8_base = sb * ELEMS_PER_BLOCK;
        let q8_ptr = q8k_x.qs.as_ptr().add(q8_base);

        let mut sumi_total: i32 = 0;

        for half in 0..2usize {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;
            let x_half = half * 128;

            let ql0 = _mm256_loadu_si256(ql_ptr.add(ql_off) as *const __m256i);
            let ql32 = _mm256_loadu_si256(ql_ptr.add(ql_off + 32) as *const __m256i);
            let qh = _mm256_loadu_si256(qh_ptr.add(qh_off) as *const __m256i);

            let s0_lo = _mm256_and_si256(ql0, lo4_mask);
            let s0_hi = _mm256_slli_epi16::<4>(_mm256_and_si256(qh, hi2_mask));
            let q6_s0 = _mm256_sub_epi8(_mm256_or_si256(s0_lo, s0_hi), neg32);

            let s1_lo = _mm256_and_si256(ql32, lo4_mask);
            let s1_hi =
                _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<2>(qh), hi2_mask));
            let q6_s1 = _mm256_sub_epi8(_mm256_or_si256(s1_lo, s1_hi), neg32);

            let s2_lo = _mm256_and_si256(_mm256_srli_epi16::<4>(ql0), lo4_mask);
            let s2_hi =
                _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<4>(qh), hi2_mask));
            let q6_s2 = _mm256_sub_epi8(_mm256_or_si256(s2_lo, s2_hi), neg32);

            let s3_lo = _mm256_and_si256(_mm256_srli_epi16::<4>(ql32), lo4_mask);
            let s3_hi =
                _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<6>(qh), hi2_mask));
            let q6_s3 = _mm256_sub_epi8(_mm256_or_si256(s3_lo, s3_hi), neg32);

            let q6_stride = [q6_s0, q6_s1, q6_s2, q6_s3];

            for g in 0..4usize {
                let q8_v = _mm256_loadu_si256(q8_ptr.add(x_half + g * 32) as *const __m256i);
                let q8_signed_flipped = _mm256_sign_epi8(q8_v, q6_stride[g]);
                let q6_abs = _mm256_abs_epi8(q6_stride[g]);
                let prod_i16 = _mm256_maddubs_epi16(q6_abs, q8_signed_flipped);
                let sum_i32 = _mm256_madd_epi16(prod_i16, ones_16);

                let lo128 = _mm256_castsi256_si128(sum_i32);
                let hi128 = _mm256_extracti128_si256::<1>(sum_i32);
                let sumi_lo = horiz_sum_i32_128(lo128);
                let sumi_hi = horiz_sum_i32_128(hi128);

                let sc_lo = *sc_ptr.add(sc_off + 2 * g) as i32;
                let sc_hi = *sc_ptr.add(sc_off + 2 * g + 1) as i32;

                sumi_total += sc_lo * sumi_lo;
                sumi_total += sc_hi * sumi_hi;
            }
        }

        acc += d_w * d_y * sumi_total as f32;
    }
    *out_r = acc;
}

/// Rayon-parallel row dispatch for AVX2 Q6_K × Q8_K matvec. Same pattern as
/// [`q4k_q8k_matvec_avx2`] — see its doc for rationale. Q6_K's row size is
/// 210 bytes/256 vs Q4_K's 144 bytes/256, so it's more BW-heavy per row; the
/// parallel win is correspondingly larger when memory bandwidth scales with
/// thread count.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q6k_q8k_matvec_avx2(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(cols % ELEMS_PER_BLOCK, 0);
    let n_blocks = cols / ELEMS_PER_BLOCK;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    for v in out.iter_mut() {
        *v = 0.0;
    }
    if rows == 0 || cols == 0 || w.len() < rows * row_bytes {
        return;
    }

    if rows < MIN_PAR_ROWS {
        for (r, out_r) in out.iter_mut().enumerate().take(rows) {
            compute_row_q6k_avx2(out_r, r, q8k_x, w, n_blocks, row_bytes);
        }
        return;
    }

    use rayon::prelude::*;
    let chunk_rows = rows.div_ceil(rayon::current_num_threads().max(1)).max(4);
    out[..rows]
        .par_chunks_mut(chunk_rows)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start_row = chunk_idx * chunk_rows;
            for (i, out_r) in chunk.iter_mut().enumerate() {
                let r = start_row + i;
                // SAFETY: outer fn requires AVX2; runtime detection happened
                // in `q6k_q8k_matvec_into` before dispatch.
                unsafe {
                    compute_row_q6k_avx2(out_r, r, q8k_x, w, n_blocks, row_bytes);
                }
            }
        });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horiz_sum_i32_128(v: std::arch::x86_64::__m128i) -> i32 {
    use std::arch::x86_64::*;
    let s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0b00_00_11_10)); // 4 → 2
    let s2 = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01)); // 2 → 1
    _mm_cvtsi128_si32(s2)
}

/// Public entry point: AVX2 on x86_64 (when available), scalar otherwise.
/// `w` is a Q6_K weight matrix of `rows` rows × `cols` columns.
/// `q8k_x` is the pre-quantised activation vector (`cols` elements).
///
/// NEON intentionally not dispatched: the existing aarch64 SIMD reads a
/// legacy sequential layout incompatible with the on-disk Q6_K format.
/// See `q6k_q8k_matvec_neon` doc for the follow-up.
pub fn q6k_q8k_matvec_into(
    out: &mut [f32],
    q8k_x: &Q8KActivation,
    w: &[u8],
    rows: usize,
    cols: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: avx2 detected; lengths validated by the AVX2 body.
        unsafe { q6k_q8k_matvec_avx2(out, q8k_x, w, rows, cols) };
        return;
    }
    q6k_q8k_matvec_scalar(out, q8k_x, w, rows, cols);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::q4_common::{q4k_matvec_into, quantize_q4_k, quantize_q6_k};

    /// Q8_K round-trip should reconstruct within 0.5% of absmax (1 LSB on
    /// the 127-step scale).  Sums must equal the literal i32 sums of the
    /// quantised values per sub-block.
    #[test]
    fn q8k_quantize_round_trip_within_quant_step() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 / 128.0 - 1.0) * 5.0).collect();
        let q = quantize_x_to_q8k(&x);
        assert_eq!(q.qs.len(), 256);
        assert_eq!(q.d.len(), 1);
        assert_eq!(q.sums.len(), 8);

        let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let step = amax / 127.0;
        for (xv, qv) in x.iter().zip(q.qs.iter()) {
            let recon = q.d[0] * (*qv as f32);
            assert!(
                (xv - recon).abs() < step.max(1e-6),
                "x={xv} recon={recon} step={step}"
            );
        }
        // Sums match the literal sums per sub-block.
        for s in 0..8 {
            let actual: i32 = q.qs[s * 32..(s + 1) * 32].iter().map(|&v| v as i32).sum();
            assert_eq!(actual as i16, q.sums[s]);
        }
    }

    /// Q8_K of all-zeros should produce zero scale + all-zero sums.
    #[test]
    fn q8k_zero_input_clean() {
        let x = vec![0.0f32; 256];
        let q = quantize_x_to_q8k(&x);
        assert_eq!(q.d[0], 0.0);
        assert!(q.qs.iter().all(|&v| v == 0));
        assert!(q.sums.iter().all(|&v| v == 0));
    }

    /// Scalar Q4_K×Q8_K matches the f32-cached path within Q8 quant noise.
    /// Same Q4_K-quantised weights and same f32 activation; one path runs
    /// the f32 dot `q4_common::q4k_matvec_into`, the other quantises x to
    /// Q8_K and runs the integer-dot reference.  Difference should be on
    /// the order of `‖w‖ · ε_q8 · ‖x‖`, well below 1e-3 for typical inputs.
    #[test]
    fn q8k_matvec_matches_f32_cached_within_q8_noise() {
        // Single super-block, single row matrix.
        let cols = 256;
        let rows = 4;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.013).sin()).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.007).cos() * 0.5)
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        assert_eq!(w_q4.len(), rows * 144);

        let mut out_f32 = vec![0.0f32; rows];
        q4k_matvec_into(&mut out_f32, &x, &w_q4, rows, cols);

        let q8 = quantize_x_to_q8k(&x);
        let mut out_q8 = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_q8, &q8, &w_q4, rows, cols);

        // Q8 quantisation step on x is amax/127; downstream noise per
        // output element is ~‖w_row‖₁ · step.  For typical sin-ramp inputs
        // that comes out in the 1e-2 range; tolerate 5e-2 to leave headroom
        // for f16 scale conversion error in d/dmin.
        for r in 0..rows {
            let diff = (out_f32[r] - out_q8[r]).abs();
            assert!(
                diff < 5e-2,
                "row {r}: f32={} q8={} diff={diff}",
                out_f32[r],
                out_q8[r]
            );
        }
    }

    /// Multi-block matrix: hidden=512 = 2 super-blocks per row.  Stresses
    /// the per-super-block aggregation (`acc += ...` summed over 2+ blocks).
    #[test]
    fn q8k_matvec_multi_block_within_noise() {
        let cols = 512; // 2 super-blocks
        let rows = 16;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.011).cos() * 2.0).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.009).sin() * 0.3)
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);

        let mut out_f32 = vec![0.0f32; rows];
        q4k_matvec_into(&mut out_f32, &x, &w_q4, rows, cols);

        let q8 = quantize_x_to_q8k(&x);
        let mut out_q8 = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_q8, &q8, &w_q4, rows, cols);

        for r in 0..rows {
            let diff = (out_f32[r] - out_q8[r]).abs();
            assert!(
                diff < 8e-2,
                "row {r}: f32={} q8={} diff={diff}",
                out_f32[r],
                out_q8[r]
            );
        }
    }

    /// NEON kernel must be bit-identical to the scalar Q8_K reference on
    /// aarch64 — both implement the same i32 dot math.  Different inputs
    /// from the noise tests above to catch byte-ordering / lane-mapping
    /// bugs that happen to vanish on regular ramps.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q8k_matvec_neon_matches_scalar_bit_exact() {
        let cols = 1024; // 4 super-blocks — exercises sb-loop + g-loop
        let rows = 7; // odd row count — exercises tail handling
                      // Use a non-symmetric, non-monotonic input so any lane/byte-swap
                      // bug can't accidentally produce the right sum.
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_neon = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols);
        q4k_q8k_matvec_neon(&mut out_neon, &q8, &w_q4, rows, cols);

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_neon[r].to_bits(),
                "row {r}: scalar={} neon={} diff={}",
                out_scalar[r],
                out_neon[r],
                (out_scalar[r] - out_neon[r]).abs()
            );
        }
    }

    /// C12 hand-asm kernel must be bit-identical to the scalar reference —
    /// it computes the same i32 `sum1`, same `sum2`, same f32 epilogue.
    /// Exercises several shapes: odd rows (tail-free in this kernel), a
    /// production attention width (2560), and a multi-super-block FFN width.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q8k_matvec_asm_matches_scalar_bit_exact() {
        for &(rows, cols) in &[(7usize, 1024usize), (8, 2560), (3, 2560), (16, 512)] {
            // Non-symmetric, non-monotonic inputs so a lane/byte-swap bug
            // can't accidentally produce the right sum.
            let x: Vec<f32> = (0..cols)
                .map(|i| {
                    let f = i as f32;
                    ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
                })
                .collect();
            let w_f32: Vec<f32> = (0..rows * cols)
                .map(|i| {
                    let f = i as f32;
                    ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
                })
                .collect();
            let w_q4 = quantize_q4_k(&w_f32);
            let q8 = quantize_x_to_q8k(&x);

            let mut out_scalar = vec![0.0f32; rows];
            let mut out_asm = vec![0.0f32; rows];
            q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols);
            q4k_q8k_matvec_asm(&mut out_asm, &q8, &w_q4, rows, cols);

            for r in 0..rows {
                assert_eq!(
                    out_scalar[r].to_bits(),
                    out_asm[r].to_bits(),
                    "rows={rows} cols={cols} row {r}: scalar={} asm={} diff={}",
                    out_scalar[r],
                    out_asm[r],
                    (out_scalar[r] - out_asm[r]).abs()
                );
            }
        }
    }

    /// Asm kernel's early-return guards (zero dims, short weight buffer)
    /// must zero the output, same as the scalar/neon paths.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q8k_matvec_asm_zero_dims_and_short_weights_zero_output() {
        // cols == 0 → early return.
        let empty = Q8KActivation {
            qs: vec![],
            d: vec![],
            sums: vec![],
        };
        let mut out = vec![1.0f32; 4];
        q4k_q8k_matvec_asm(&mut out, &empty, &[], 4, 0);
        assert!(out.iter().all(|&v| v == 0.0), "zero-dims must zero output");

        // w shorter than rows * row_bytes → early return.
        let cols = 256;
        let rows = 2;
        let q = quantize_x_to_q8k(&vec![0.5f32; cols]);
        let w = vec![0u8; BLOCK_BYTES]; // one row's worth, but rows == 2
        let mut out = vec![1.0f32; rows];
        q4k_q8k_matvec_asm(&mut out, &q, &w, rows, cols);
        assert!(
            out.iter().all(|&v| v == 0.0),
            "short buffer must zero output"
        );
    }

    /// `LARQL_Q4K_ASM` opt-in truth table (the pure parse behind the
    /// `OnceLock`-cached `use_asm_kernel`).
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q4k_asm_flag_truth_table() {
        assert!(q4k_asm_flag_enabled(Some("1")));
        assert!(q4k_asm_flag_enabled(Some("true")));
        assert!(q4k_asm_flag_enabled(Some("TRUE")));
        assert!(q4k_asm_flag_enabled(Some("True")));
        assert!(!q4k_asm_flag_enabled(Some("0")));
        assert!(!q4k_asm_flag_enabled(Some("false")));
        assert!(!q4k_asm_flag_enabled(Some("yes")));
        assert!(!q4k_asm_flag_enabled(Some("")));
        assert!(!q4k_asm_flag_enabled(None));
    }

    /// `quantize_x_to_q8k_into` must produce the same `qs`, `d`, `sums` as
    /// the allocating `quantize_x_to_q8k` for any well-sized input — both
    /// also handle resize correctly when reused across different sizes.
    #[test]
    fn q8k_in_place_matches_alloc_version() {
        let x: Vec<f32> = (0..512).map(|i| (i as f32 * 0.013).sin() * 3.0).collect();
        let alloc_q = quantize_x_to_q8k(&x);

        let mut buf = Q8KActivation::with_capacity(512);
        quantize_x_to_q8k_into(&mut buf, &x);

        assert_eq!(buf.qs, alloc_q.qs);
        assert_eq!(buf.d, alloc_q.d);
        assert_eq!(buf.sums, alloc_q.sums);

        // Resize-on-reuse: quantise smaller input into the same buffer.
        let x2: Vec<f32> = (0..256).map(|i| (i as f32 * 0.021).cos()).collect();
        let alloc_q2 = quantize_x_to_q8k(&x2);
        quantize_x_to_q8k_into(&mut buf, &x2);
        assert_eq!(buf.qs.len(), 256);
        assert_eq!(buf.d.len(), 1);
        assert_eq!(buf.sums.len(), 8);
        assert_eq!(buf.qs, alloc_q2.qs);
        assert_eq!(buf.d, alloc_q2.d);
        assert_eq!(buf.sums, alloc_q2.sums);
    }

    /// 2-row matvec must produce bit-exact outputs equal to the single-row
    /// kernel for the same input — the dot math is identical, only the
    /// instruction scheduling differs.  Test on both even and odd row
    /// counts so the tail-handling path is exercised.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn q8k_matvec_2row_matches_single_row_bit_exact() {
        for &rows in &[2usize, 4, 7, 11, 16, 17] {
            let cols = 1024;
            let x: Vec<f32> = (0..cols)
                .map(|i| (i as f32 * 0.0173).sin() * 1.7 + (i as f32 * 0.041).cos() * 0.9)
                .collect();
            let w_f32: Vec<f32> = (0..rows * cols)
                .map(|i| (i as f32 * 0.013).cos() * 0.4 - (i as f32 * 0.027).sin() * 0.2)
                .collect();
            let w_q4 = quantize_q4_k(&w_f32);
            let q8 = quantize_x_to_q8k(&x);

            let mut out_single = vec![0.0f32; rows];
            let mut out_2row = vec![0.0f32; rows];
            q4k_q8k_matvec_neon(&mut out_single, &q8, &w_q4, rows, cols);
            q4k_q8k_matvec_neon_2row(&mut out_2row, &q8, &w_q4, rows, cols);

            for r in 0..rows {
                assert_eq!(
                    out_single[r].to_bits(),
                    out_2row[r].to_bits(),
                    "rows={rows} r={r}: single={} 2row={} diff={}",
                    out_single[r],
                    out_2row[r],
                    (out_single[r] - out_2row[r]).abs()
                );
            }
        }
    }

    /// Fused gate+up must produce bit-exact outputs equal to two separate
    /// matvec calls — both compile down to the same i32 dot math; only the
    /// instruction interleaving differs.
    #[test]
    fn q8k_gate_up_fused_matches_separate_matvecs() {
        let cols = 1024;
        let rows = 11;
        let x: Vec<f32> = (0..cols)
            .map(|i| (i as f32 * 0.0151).sin() * 1.4 + (i as f32 * 0.029).cos() * 0.7)
            .collect();
        let g_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.011).cos() * 0.4 - (i as f32 * 0.027).sin() * 0.2)
            .collect();
        let u_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.013).sin() * 0.3 + (i as f32 * 0.041).cos() * 0.5)
            .collect();
        let g_w = quantize_q4_k(&g_f32);
        let u_w = quantize_q4_k(&u_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut g_sep = vec![0.0f32; rows];
        let mut u_sep = vec![0.0f32; rows];
        q4k_q8k_matvec_into(&mut g_sep, &q8, &g_w, rows, cols);
        q4k_q8k_matvec_into(&mut u_sep, &q8, &u_w, rows, cols);

        let mut g_fused = vec![0.0f32; rows];
        let mut u_fused = vec![0.0f32; rows];
        q4k_q8k_gate_up_into(&mut g_fused, &mut u_fused, &q8, &g_w, &u_w, rows, cols);

        for r in 0..rows {
            assert_eq!(
                g_sep[r].to_bits(),
                g_fused[r].to_bits(),
                "gate row {r}: sep={} fused={}",
                g_sep[r],
                g_fused[r]
            );
            assert_eq!(
                u_sep[r].to_bits(),
                u_fused[r].to_bits(),
                "up row {r}: sep={} fused={}",
                u_sep[r],
                u_fused[r]
            );
        }
    }

    /// Empty / degenerate dims should produce zeros without panic.
    #[test]
    fn q8k_matvec_zero_dims_returns_zero() {
        let q = Q8KActivation {
            qs: vec![],
            d: vec![],
            sums: vec![],
        };
        let mut out = vec![1.0f32; 4];
        q4k_q8k_matvec_scalar(&mut out, &q, &[], 4, 0);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Misaligned col count (not a multiple of 256) should fail safely
    /// (leave caller-visible zeros, like the scalar `q4k_matvec_into`).
    #[test]
    fn q8k_matvec_short_weight_buffer_returns_zero() {
        let cols = 256;
        let rows = 2;
        let x = vec![0.5f32; cols];
        let q = quantize_x_to_q8k(&x);
        let w = vec![0u8; 144]; // only enough for 1 row, but rows=2
        let mut out = vec![1.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out, &q, &w, rows, cols);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    /// Q4_KF AVX2 must match the scalar reference within f32 round-off
    /// (both fuse the same dot + scale arithmetic but in slightly
    /// different orders, so bit-exact isn't quite achievable — 1e-5
    /// rel is the actual envelope).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q4kf_q8k_matvec_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::cpu::ops::q4_common::{q4k_to_q4kf, quantize_q4_k};
        let cols = 1024;
        let rows = 7;
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32 * 0.0173).sin() * 1.7 + (i as f32 * 0.041).cos() * 0.9) * 1.3)
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.013).cos() * 0.4 - (i as f32 * 0.027).sin() * 0.2) * 0.6)
            .collect();
        let w_q4k = quantize_q4_k(&w_f32);
        let w_q4kf = q4k_to_q4kf(&w_q4k, rows, cols);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_avx2 = vec![0.0f32; rows];
        q4kf_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4kf, rows, cols);
        unsafe { q4kf_q8k_matvec_avx2(&mut out_avx2, &q8, &w_q4kf, rows, cols) };

        for r in 0..rows {
            let rel = (out_scalar[r] - out_avx2[r]).abs() / out_scalar[r].abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "row {r}: scalar={} avx2={} rel={rel}",
                out_scalar[r],
                out_avx2[r],
            );
        }
    }

    /// Canonical oracle: dequant via `dequantize_q4_kf` × x ≈ AVX2 matvec
    /// within Q8_K activation noise.
    #[test]
    fn q4kf_q8k_matvec_matches_canonical_dequant() {
        use crate::cpu::ops::q4_common::{dequantize_q4_kf, q4k_to_q4kf, quantize_q4_k};
        let cols = 512;
        let rows = 5;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin() * 1.5).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.006).cos() * 0.7)
            .collect();
        let w_q4k = quantize_q4_k(&w_f32);
        let w_q4kf = q4k_to_q4kf(&w_q4k, rows, cols);
        let w_deq = dequantize_q4_kf(&w_q4kf, rows * cols).expect("dequant q4_kf");

        let mut f32_ref = vec![0.0f32; rows];
        for (r, slot) in f32_ref.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += w_deq[r * cols + c] * x[c];
            }
            *slot = acc;
        }

        let q8 = quantize_x_to_q8k(&x);
        let mut got = vec![0.0f32; rows];
        q4kf_q8k_matvec_into(&mut got, &q8, &w_q4kf, rows, cols);

        for r in 0..rows {
            let rel = (f32_ref[r] - got[r]).abs() / f32_ref[r].abs().max(1e-6);
            assert!(
                rel < 1.5e-2,
                "row {r}: ref={} got={} rel={rel}",
                f32_ref[r],
                got[r]
            );
        }
    }

    /// AVX2 Q6_K matvec must be bit-equivalent to the scalar reference
    /// modulo i32 reduction-order independence (both produce the same
    /// `sumi_total` per super-block, then the same `d_w * d_y * sumi` f32
    /// product — only the order of i32 additions within a super-block
    /// differs, and both fit comfortably in i32 so there's no overflow).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q6k_q8k_matvec_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let cols = 1024; // 4 super-blocks
        let rows = 7;
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q6 = quantize_q6_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_avx2 = vec![0.0f32; rows];
        q6k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q6, rows, cols);
        unsafe { q6k_q8k_matvec_avx2(&mut out_avx2, &q8, &w_q6, rows, cols) };

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_avx2[r].to_bits(),
                "row {r}: scalar={} avx2={} diff={}",
                out_scalar[r],
                out_avx2[r],
                (out_scalar[r] - out_avx2[r]).abs()
            );
        }
    }

    /// Canonical oracle for Q4_K × Q8_K: dequantise via
    /// `larql_models::quant::ggml::dequantize_q4_k` (mirrors llama.cpp
    /// `dequantize_row_q4_K` wire format) and compute the f32 reference
    /// dot. Both the scalar and AVX2 paths must match within Q8_K
    /// activation noise. Defensive against the same class of layout bug
    /// that #102 found in Q6_K — the existing
    /// `q8k_matvec_matches_f32_cached_within_q8_noise` test compares
    /// against `q4k_matvec_into` (an internal reader), which would mask
    /// a parallel-wrong-the-same-way bug. This oracle uses the
    /// `larql-models` dequantiser as ground truth.
    #[test]
    fn q4k_q8k_matvec_matches_canonical_dequant() {
        use larql_models::quant::ggml::dequantize_q4_k as canonical_dequant_q4_k;
        let cols = 512;
        let rows = 5;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin() * 1.5).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.006).cos() * 0.7)
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        let w_deq = canonical_dequant_q4_k(&w_q4, rows * cols).expect("dequant q4_k");

        // f32 reference: dot(canonical_dequant(w_q4), x) row-wise.
        let mut f32_ref = vec![0.0f32; rows];
        for (r, out_r) in f32_ref.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += w_deq[r * cols + c] * x[c];
            }
            *out_r = acc;
        }

        let q8 = quantize_x_to_q8k(&x);
        let mut scalar_path = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut scalar_path, &q8, &w_q4, rows, cols);

        // Q8_K activation noise ≤ ~0.5 % per block; relative tolerance of
        // 1.5 % covers the noise with margin yet flags any layout-mismatch
        // (which would be O(1) error).
        for r in 0..rows {
            let ref_v = f32_ref[r];
            let got = scalar_path[r];
            let rel = (ref_v - got).abs() / ref_v.abs().max(1e-6);
            assert!(
                rel < 1.5e-2,
                "scalar row {r}: ref={ref_v} got={got} rel={rel}"
            );
        }

        // Dispatched path (AVX2 on x86_64 with the feature, scalar
        // otherwise) must agree with the scalar bit-exactly. The
        // dispatched path is what production code reaches via
        // `q4k_q8k_matvec_into`.
        let mut into_path = vec![0.0f32; rows];
        q4k_q8k_matvec_into(&mut into_path, &q8, &w_q4, rows, cols);
        for r in 0..rows {
            assert_eq!(
                scalar_path[r].to_bits(),
                into_path[r].to_bits(),
                "dispatched != scalar at row {r}: scalar={} dispatched={}",
                scalar_path[r],
                into_path[r],
            );
        }
    }

    /// Canonical oracle: dequantise via `larql_models::quant::ggml::dequantize_q6_k`
    /// (mirrors llama.cpp `dequantize_row_q6_K` wire format) and compute the
    /// f32 reference dot. `q6k_q8k_matvec_scalar` must match — anything
    /// looser hides a layout bug. The previous comparison-to-`q6k_matvec::dispatch`
    /// was tautological because both readers used the same (buggy) sequential layout.
    #[test]
    fn q6k_q8k_matvec_matches_canonical_dequant() {
        use larql_models::quant::ggml::dequantize_q6_k as canonical_dequant_q6_k;
        let cols = 512;
        let rows = 5;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin() * 1.5).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.006).cos() * 0.7)
            .collect();
        let w_q6 = quantize_q6_k(&w_f32);
        let w_deq = canonical_dequant_q6_k(&w_q6, rows * cols).expect("dequant q6_k");

        // f32 reference: dot(canonical_dequant(w_q6), x) row-wise.
        let mut f32_ref = vec![0.0f32; rows];
        for (r, out_r) in f32_ref.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += w_deq[r * cols + c] * x[c];
            }
            *out_r = acc;
        }

        let q8 = quantize_x_to_q8k(&x);
        let mut q8_path = vec![0.0f32; rows];
        q6k_q8k_matvec_scalar(&mut q8_path, &q8, &w_q6, rows, cols);

        // Q8_K quantisation of `x` introduces at most ~0.5 % activation noise.
        // Tolerance: relative 1.5 % of the f32 reference magnitude (well above
        // Q8_K noise, well below any layout-mismatch error which would be O(1)).
        for r in 0..rows {
            let ref_v = f32_ref[r];
            let got = q8_path[r];
            let abs = (ref_v - got).abs();
            let rel = abs / ref_v.abs().max(1e-6);
            assert!(
                rel < 1.5e-2,
                "row {r}: ref={ref_v} got={got} abs={abs} rel={rel}"
            );
        }
    }

    /// Cross-path parity: the two production Q6_K matvec entry points
    /// must agree on identical weights. `q6k_matvec::dispatch` (trait-
    /// dispatched f32-input scalar; called via `CpuBackend::q6k_matvec`
    /// from attention V-projection, lm-head KNN, speculative wiring,
    /// CUDA fallback decode) and `q6k_q8k_matvec_into` (Q8_K-input
    /// AVX2-on-x86_64; called from `walk_ffn_q8k`'s Q6_K branch added in
    /// #103) both consume the same llama.cpp Q6_K wire format. They
    /// differ only in whether the activation `x` is f32 or pre-quantised
    /// to Q8_K, so they should agree within Q8_K activation noise (~0.5
    /// % per block — dot product averages this down further).
    #[test]
    fn q6k_two_production_paths_agree_within_q8k_noise() {
        let cols = 512; // 2 super-blocks per row
        let rows = 5;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.017).sin() * 1.5).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.006).cos() * 0.7)
            .collect();
        let w_q6 = quantize_q6_k(&w_f32);

        // f32-input trait path (production attention V proj / lm_head).
        let f32_path = crate::cpu::ops::q6k_matvec::dispatch(&w_q6, &x, rows, cols);

        // Q8K-input AVX2 path (production FFN_DOWN via walk-ffn-q8k).
        let q8 = quantize_x_to_q8k(&x);
        let mut q8k_path = vec![0.0f32; rows];
        q6k_q8k_matvec_into(&mut q8k_path, &q8, &w_q6, rows, cols);

        for r in 0..rows {
            let f = f32_path[r];
            let q = q8k_path[r];
            let rel = (f - q).abs() / f.abs().max(1e-6);
            assert!(rel < 1.5e-2, "row {r}: f32_path={f} q8k_path={q} rel={rel}");
        }
    }

    #[test]
    fn q6k_q8k_public_entrypoint_matches_scalar() {
        let cols = 256;
        let rows = 3;
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.031).cos()).collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 * 0.011).sin() * 0.4)
            .collect();
        let w_q6 = quantize_q6_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);
        let mut scalar = vec![0.0f32; rows];
        let mut dispatched = vec![0.0f32; rows];

        q6k_q8k_matvec_scalar(&mut scalar, &q8, &w_q6, rows, cols);
        q6k_q8k_matvec_into(&mut dispatched, &q8, &w_q6, rows, cols);

        for (a, b) in scalar.iter().zip(dispatched.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn q6k_q8k_zero_dims_and_short_weights_zero_output() {
        let q = Q8KActivation::with_capacity(0);
        let mut out = vec![1.0f32; 4];
        q6k_q8k_matvec_scalar(&mut out, &q, &[], 4, 0);
        assert_eq!(out, vec![0.0f32; 4]);

        let x = vec![1.0f32; 256];
        let q = quantize_x_to_q8k(&x);
        let mut out = vec![1.0f32; 2];
        q6k_q8k_matvec_scalar(&mut out, &q, &vec![0u8; 210], 2, 256);
        assert_eq!(out, vec![0.0f32; 2]);
    }

    /// AVX2 must produce bit-identical output to the scalar reference.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn q8k_matvec_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return; // Skip on hardware without AVX2.
        }
        let cols = 1024;
        let rows = 7;
        let x: Vec<f32> = (0..cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0173).sin() * 1.7 + (f * 0.041).cos() * 0.9) * 1.3
            })
            .collect();
        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let f = i as f32;
                ((f * 0.013).cos() * 0.4 - (f * 0.027).sin() * 0.2) * 0.6
            })
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);
        let q8 = quantize_x_to_q8k(&x);

        let mut out_scalar = vec![0.0f32; rows];
        let mut out_avx2 = vec![0.0f32; rows];
        q4k_q8k_matvec_scalar(&mut out_scalar, &q8, &w_q4, rows, cols);
        unsafe { q4k_q8k_matvec_avx2(&mut out_avx2, &q8, &w_q4, rows, cols) };

        for r in 0..rows {
            assert_eq!(
                out_scalar[r].to_bits(),
                out_avx2[r].to_bits(),
                "row {r}: scalar={} avx2={} diff={}",
                out_scalar[r],
                out_avx2[r],
                (out_scalar[r] - out_avx2[r]).abs()
            );
        }
    }

    /// `q4k_q8k_matmul_into` must produce **bit-exact** output to a loop
    /// of `q4k_q8k_matvec_into` calls. This is the load-bearing
    /// correctness guarantee — prefill calls swap from the per-row loop
    /// to the batched kernel; any divergence would silently break
    /// every chat completion that goes through prefill.
    #[test]
    fn q4k_q8k_matmul_into_matches_per_row_matvec_loop_bit_exact() {
        // Realistic prefill dims: hidden=2048 (= 8 super-blocks),
        // rows=8192 (qwen3.6 attn_qkv), n=14 (typical chat prompt seq).
        // Trim to a tractable test size while keeping multi-block + N>1.
        let cols = 512; // 2 super-blocks
        let rows = 64; // > MIN_PAR_ROWS to exercise the parallel branch
        let n_inputs = 5usize;

        let w_f32: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32) * 0.0017).sin() * 0.4)
            .collect();
        let w_q4 = quantize_q4_k(&w_f32);

        let q8k_xs: Vec<Q8KActivation> = (0..n_inputs)
            .map(|n| {
                let x: Vec<f32> = (0..cols)
                    .map(|i| ((n as f32) * 0.31 + (i as f32) * 0.011).cos() * 1.5)
                    .collect();
                quantize_x_to_q8k(&x)
            })
            .collect();

        // Reference: per-row loop.
        let mut out_loop = vec![0.0f32; n_inputs * rows];
        for (n_idx, q8k) in q8k_xs.iter().enumerate() {
            let slot = &mut out_loop[n_idx * rows..(n_idx + 1) * rows];
            q4k_q8k_matvec_into(slot, q8k, &w_q4, rows, cols);
        }

        // Batched kernel.
        let mut out_batched = vec![0.0f32; n_inputs * rows];
        q4k_q8k_matmul_into(&mut out_batched, &q8k_xs, &w_q4, rows, cols);

        for n in 0..n_inputs {
            for r in 0..rows {
                let i = n * rows + r;
                assert_eq!(
                    out_loop[i].to_bits(),
                    out_batched[i].to_bits(),
                    "[n={n} r={r}]: loop={} batched={} diff={}",
                    out_loop[i],
                    out_batched[i],
                    (out_loop[i] - out_batched[i]).abs()
                );
            }
        }
    }

    /// Zero inputs / zero rows / zero cols must produce a zero-filled
    /// output without panicking — the fast-path guard at the top of
    /// the matmul.
    #[test]
    fn q4k_q8k_matmul_into_zero_inputs_zeroes_output() {
        let mut out = vec![1.0f32; 0]; // n=0 => out is empty
        q4k_q8k_matmul_into(&mut out, &[], &[], 64, 256);
        assert!(out.is_empty());

        // n=2 but rows=0 should still zero-fill the (zero-sized) output.
        let q8: Vec<Q8KActivation> = (0..2).map(|_| quantize_x_to_q8k(&vec![0.0; 256])).collect();
        let mut out = vec![1.0f32; 0];
        q4k_q8k_matmul_into(&mut out, &q8, &[], 0, 256);
        assert!(out.is_empty());
    }
}
