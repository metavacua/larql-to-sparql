//! Q6_K — 256-element super-block, 210 bytes/block. Highest precision
//! K-quant; typical for the down projection in Ollama-shaped Q4_K_M
//! mixes. NEON row dot + scaled-add with scalar fallbacks.

use crate::ModelError;

use super::check_block_input;
use crate::quant::half::f16_to_f32;

pub fn q6k_row_dot(data: &[u8], x: &[f32]) -> Result<f32, ModelError> {
    const BLOCK: usize = 210;
    const SUPER: usize = 256;
    let n = x.len();
    if !n.is_multiple_of(SUPER) {
        return Err(ModelError::Parse(format!(
            "q6k_row_dot: row length {n} not a multiple of {SUPER}"
        )));
    }
    let n_blocks = n / SUPER;
    if data.len() < n_blocks * BLOCK {
        return Err(ModelError::Parse(format!(
            "q6k_row_dot: data short: {} < {}",
            data.len(),
            n_blocks * BLOCK,
        )));
    }

    // C.5j: the NEON path was built around the OLD (buggy) Q6_K element
    // layout — sequential per-scale-subblock — but the correct llama.cpp
    // layout is interleaved (`y[l]`, `y[l+32]`, `y[l+64]`, `y[l+96]`).
    // Forcing scalar until the NEON path is rewritten to match. Loses
    // ~3-4× on Apple Silicon Q6_K matvec but correctness > perf.
    // TODO: port the interleaved layout to `q6k_row_dot_neon`.
    Ok(q6k_row_dot_scalar(data, x, n_blocks))
}

/// Scalar reference used on non-aarch64 and by tests.
#[inline]
#[allow(dead_code)]
pub(super) fn q6k_row_dot_scalar(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    // Mirror of llama.cpp `dequantize_row_q6_K` element layout — see the
    // `dequantize_q6_k` body above for the per-half interleaved scheme.
    let mut acc = 0.0f32;
    for sb in 0..n_blocks {
        let block = &data[sb * 210..(sb + 1) * 210];
        let ql_full = &block[0..128];
        let qh_full = &block[128..192];
        let sc_full = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let x_base = sb * 256;
        for half in 0..2 {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;
            let x_half = x_base + half * 128;
            for l in 0..32_usize {
                let is = l / 16;
                let qh_byte = qh_full[qh_off + l];
                let q1 = ((ql_full[ql_off + l] & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) as i8
                    as i32
                    - 32;
                let q2 = ((ql_full[ql_off + l + 32] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i8
                    as i32
                    - 32;
                let q3 =
                    ((ql_full[ql_off + l] >> 4) | (((qh_byte >> 4) & 0x03) << 4)) as i8 as i32 - 32;
                let q4 = ((ql_full[ql_off + l + 32] >> 4) | (((qh_byte >> 6) & 0x03) << 4)) as i8
                    as i32
                    - 32;
                let s0 = d * (sc_full[sc_off + is] as i8) as f32;
                let s1 = d * (sc_full[sc_off + is + 2] as i8) as f32;
                let s2 = d * (sc_full[sc_off + is + 4] as i8) as f32;
                let s3 = d * (sc_full[sc_off + is + 6] as i8) as f32;
                acc += s0 * q1 as f32 * x[x_half + l];
                acc += s1 * q2 as f32 * x[x_half + l + 32];
                acc += s2 * q3 as f32 * x[x_half + l + 64];
                acc += s3 * q4 as f32 * x[x_half + l + 96];
            }
        }
    }
    acc
}

/// NEON-SIMD Q6K dequant + dot. Decodes 16 signed 6-bit values per scale
/// subblock into four f32x4 lanes, uses four parallel accumulators for ILP.
/// Cuts per-layer Q6_K down-projection from ~42ms to ~10-12ms on M-series.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn q6k_row_dot_neon(data: &[u8], x: &[f32], n_blocks: usize) -> f32 {
    use std::arch::aarch64::*;
    const BLOCK: usize = 210;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    let x_ptr = x.as_ptr();
    for sb in 0..n_blocks {
        let block = data.as_ptr().add(sb * BLOCK);
        let ql = block;
        let qh = block.add(128);
        let scales = block.add(192);
        let d = f16_to_f32(u16::from_le_bytes([*block.add(208), *block.add(209)]));
        let sb_base = x_ptr.add(sb * 256);
        // 16 scale subblocks × 16 elements = 256 super-block elements.
        // Each subblock j covers ql[j*8..(j+1)*8] (8 bytes → 16 nibbles) and
        // qh[j*4..(j+1)*4] (4 bytes → 16 two-bit pairs).
        for j in 0..16 {
            let sc = d * (*(scales.add(j) as *const i8)) as f32;
            let ql_j = ql.add(j * 8);
            let qh_j = qh.add(j * 4);
            // Decode 16 signed 6-bit vals via scalar extract → i8 stack array.
            // Widening i8 → i32 → f32 then SIMDs.
            let mut vals = [0i8; 16];
            for chunk in 0..4 {
                let ql_b0 = *ql_j.add(chunk * 2);
                let ql_b1 = *ql_j.add(chunk * 2 + 1);
                let qh_b = *qh_j.add(chunk);
                let base = chunk * 4;
                // Even idx: low nibble; odd idx: high nibble. hi2 = (qh >> (k*2)) & 3.
                let lo0 = (ql_b0 & 0x0F) as u16 | (((qh_b & 0x03) as u16) << 4);
                let lo1 = ((ql_b0 >> 4) & 0x0F) as u16 | ((((qh_b >> 2) & 0x03) as u16) << 4);
                let lo2 = (ql_b1 & 0x0F) as u16 | ((((qh_b >> 4) & 0x03) as u16) << 4);
                let lo3 = ((ql_b1 >> 4) & 0x0F) as u16 | ((((qh_b >> 6) & 0x03) as u16) << 4);
                vals[base] = (lo0 as i16 - 32) as i8;
                vals[base + 1] = (lo1 as i16 - 32) as i8;
                vals[base + 2] = (lo2 as i16 - 32) as i8;
                vals[base + 3] = (lo3 as i16 - 32) as i8;
            }
            // Widen i8×16 → i16×8 × 2 → i32×4 × 4 → f32×4 × 4.
            let vals_i8 = vld1q_s8(vals.as_ptr());
            let lo_i16 = vmovl_s8(vget_low_s8(vals_i8));
            let hi_i16 = vmovl_s8(vget_high_s8(vals_i8));
            let v0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_i16)));
            let v1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_i16)));
            let v2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_i16)));
            let v3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_i16)));
            let sc_v = vdupq_n_f32(sc);
            let x_j = sb_base.add(j * 16);
            let x0 = vld1q_f32(x_j);
            let x1 = vld1q_f32(x_j.add(4));
            let x2 = vld1q_f32(x_j.add(8));
            let x3 = vld1q_f32(x_j.add(12));
            // acc += (v * sc) * x — pre-scale then FMA.
            acc0 = vfmaq_f32(acc0, vmulq_f32(v0, sc_v), x0);
            acc1 = vfmaq_f32(acc1, vmulq_f32(v1, sc_v), x1);
            acc2 = vfmaq_f32(acc2, vmulq_f32(v2, sc_v), x2);
            acc3 = vfmaq_f32(acc3, vmulq_f32(v3, sc_v), x3);
        }
    }
    let acc01 = vaddq_f32(acc0, acc1);
    let acc23 = vaddq_f32(acc2, acc3);
    vaddvq_f32(vaddq_f32(acc01, acc23))
}

/// Fused Q6_K decode + scaled add.
#[inline]
pub fn q6k_row_scaled_add(data: &[u8], alpha: f32, out: &mut [f32]) -> Result<(), ModelError> {
    let block_size = 210;
    let super_block = 256;
    let n = out.len();
    if !n.is_multiple_of(super_block) {
        return Err(ModelError::Parse(format!(
            "q6k_row_scaled_add: row length {n} not a multiple of {super_block}"
        )));
    }
    let n_blocks = n / super_block;
    if data.len() < n_blocks * block_size {
        return Err(ModelError::Parse(format!(
            "q6k_row_scaled_add: data short: {} < {}",
            data.len(),
            n_blocks * block_size,
        )));
    }
    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let ql_full = &block[0..128];
        let qh_full = &block[128..192];
        let sc_full = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let out_base = sb * super_block;
        for half in 0..2 {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;
            let out_half = out_base + half * 128;
            for l in 0..32_usize {
                let is = l / 16;
                let qh_byte = qh_full[qh_off + l];
                let q1 = ((ql_full[ql_off + l] & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) as i8
                    as i32
                    - 32;
                let q2 = ((ql_full[ql_off + l + 32] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i8
                    as i32
                    - 32;
                let q3 =
                    ((ql_full[ql_off + l] >> 4) | (((qh_byte >> 4) & 0x03) << 4)) as i8 as i32 - 32;
                let q4 = ((ql_full[ql_off + l + 32] >> 4) | (((qh_byte >> 6) & 0x03) << 4)) as i8
                    as i32
                    - 32;
                let s0 = d * (sc_full[sc_off + is] as i8) as f32;
                let s1 = d * (sc_full[sc_off + is + 2] as i8) as f32;
                let s2 = d * (sc_full[sc_off + is + 4] as i8) as f32;
                let s3 = d * (sc_full[sc_off + is + 6] as i8) as f32;
                out[out_half + l] += alpha * s0 * q1 as f32;
                out[out_half + l + 32] += alpha * s1 * q2 as f32;
                out[out_half + l + 64] += alpha * s2 * q3 as f32;
                out[out_half + l + 96] += alpha * s3 * q4 as f32;
            }
        }
    }
    Ok(())
}

/// Q6_K: super-block of 256 values = 210 bytes.
/// [0..127] lower 4 bits, [128..191] upper 2 bits, [192..207] 16 int8 scales, [208..209] f16 d.
pub fn dequantize_q6_k(data: &[u8], n_elements: usize) -> Result<Vec<f32>, ModelError> {
    let block_size = 210;
    let super_block = 256;
    let n_blocks = check_block_input("Q6_K", data, n_elements, super_block, block_size)?;
    let mut out = vec![0.0_f32; n_elements];

    // Exact mirror of llama.cpp `dequantize_row_q6_K` (ggml/src/ggml-quants.c).
    // Q6_K super-block: 256 elements stored as TWO halves of 128 each.
    // Per half: 64 bytes ql (low 4 bits), 32 bytes qh (high 2 bits), 8 i8 scales.
    // Within each half, l in 0..32 fills 4 interleaved output positions:
    //   y[l + 0]  ← ql[l]      low4 | qh[l] bits 0..1
    //   y[l + 32] ← ql[l + 32] low4 | qh[l] bits 2..3
    //   y[l + 64] ← ql[l]      high4 | qh[l] bits 4..5
    //   y[l + 96] ← ql[l + 32] high4 | qh[l] bits 6..7
    // Scales applied per-32-element sub-block: sc[is+0], sc[is+2], sc[is+4], sc[is+6]
    // where is = l/16 (0 for first 16 ls, 1 for second 16). All values offset by -32.
    for sb in 0..n_blocks {
        let block = &data[sb * block_size..(sb + 1) * block_size];
        let ql_full = &block[0..128];
        let qh_full = &block[128..192];
        let sc_full = &block[192..208];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let y_base = sb * super_block;

        for half in 0..2 {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;
            let y_half = y_base + half * 128;
            for l in 0..32_usize {
                let is = l / 16;
                let qh_byte = qh_full[qh_off + l];
                let q1 = ((ql_full[ql_off + l] & 0x0F) | (((qh_byte >> 0) & 0x03) << 4)) as i8
                    as i32
                    - 32;
                let q2 = ((ql_full[ql_off + l + 32] & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i8
                    as i32
                    - 32;
                let q3 =
                    ((ql_full[ql_off + l] >> 4) | (((qh_byte >> 4) & 0x03) << 4)) as i8 as i32 - 32;
                let q4 = ((ql_full[ql_off + l + 32] >> 4) | (((qh_byte >> 6) & 0x03) << 4)) as i8
                    as i32
                    - 32;
                let s0 = d * (sc_full[sc_off + is] as i8) as f32;
                let s1 = d * (sc_full[sc_off + is + 2] as i8) as f32;
                let s2 = d * (sc_full[sc_off + is + 4] as i8) as f32;
                let s3 = d * (sc_full[sc_off + is + 6] as i8) as f32;
                out[y_half + l] = s0 * q1 as f32;
                out[y_half + l + 32] = s1 * q2 as f32;
                out[y_half + l + 64] = s2 * q3 as f32;
                out[y_half + l + 96] = s3 * q4 as f32;
            }
        }
    }
    Ok(out)
}
