//! Shared Q4 utilities for CPU backend.
//!
//! C FFI declarations for the vdotq_s32 kernel (csrc/q4_dot.c)
//! and Q8 quantization helper.

extern "C" {
    /// C kernel: Q4_0 × Q8_0 matrix-vector multiply with ARM vdotq_s32.
    pub fn q4_0_matvec_c(
        q4_data: *const u8,
        q8_x: *const i8,
        q8_scales: *const f32,
        scores: *mut f32,
        num_rows: usize,
        hidden: usize,
    );

    /// C kernel: Q4_0 vector-matrix multiply (scatter-accumulate).
    pub fn q4_0_vecmat_c(
        activation: *const f32,
        q4_data: *const u8,
        out: *mut f32,
        intermediate: usize,
        hidden: usize,
    );
}

/// Pre-quantize f32 vector to Q8_0 (int8 + per-block f32 scale).
pub fn quantize_to_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n_blocks = x.len() / 32;
    let mut q8 = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; n_blocks];
    for (b, scale_out) in scales.iter_mut().enumerate().take(n_blocks) {
        let off = b * 32;
        let block = &x[off..off + 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        *scale_out = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for j in 0..32 {
            q8[off + j] = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (q8, scales)
}

/// Quantize f32 data to Q4_0 format (4-bit, block size 32).
///
/// Each block of 32 floats becomes 18 bytes: 2 bytes f16 scale + 16 bytes packed nibbles.
/// Used for weight quantization in benchmarks, tests, and tooling.
pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(32), "data length must be a multiple of 32");
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 18);
    for i in 0..n_blocks {
        let block = &data[i * 32..(i + 1) * 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 7.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        // f32 → f16 conversion
        let bits = scale.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;
        let f16 = if exp == 0 { sign as u16 }
            else if exp == 255 { (sign | 0x7C00 | (mant >> 13)) as u16 }
            else {
                let new_exp = exp - 127 + 15;
                if new_exp >= 31 { (sign | 0x7C00) as u16 }
                else if new_exp <= 0 { sign as u16 }
                else { (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16 }
            };
        out.extend_from_slice(&f16.to_le_bytes());
        for j in 0..16 {
            let lo = ((block[j * 2] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((block[j * 2 + 1] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// Encode f32 to f16 bits (for quantize helpers).
///
/// Handles subnormals. When `new_exp <= 0` the value is small enough that f16
/// can only represent it as a subnormal (implicit leading 0 instead of 1). We
/// construct that subnormal mantissa by shifting the implicit-one back in and
/// right-shifting — previously this branch just emitted signed zero, which
/// meant Q-quant scales for small weight sub-blocks silently collapsed to
/// zero and the whole super-block decoded as zero. Real-world NN weights have
/// sub-block ranges ~10⁻² and scales ~10⁻⁵, exactly in f16 subnormal range.
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 { return sign as u16; }
    if exp == 255 { return (sign | 0x7C00 | (mant >> 13)) as u16; }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 { return (sign | 0x7C00) as u16; }
    if new_exp <= 0 {
        // Subnormal: value = (1 + mant/2^23) * 2^(exp-127), we need to express
        // it as (subnormal_mant/2^10) * 2^-14 where subnormal_mant ∈ [0, 1023].
        // Include the implicit leading 1, shift right to align with f16's
        // subnormal scale.
        let shift = 1 - new_exp; // number of extra right-shifts past the normal encoding
        let with_implicit = (mant | 0x800000) as u32;
        let sub_mant = with_implicit >> (13 + shift as u32);
        return (sign | sub_mant as u32) as u16;
    }
    (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
}

/// Quantize f32 data to Q4_K format (4-bit with sub-block scales, Ollama-compatible).
///
/// Each super-block of 256 floats becomes 148 bytes:
///   [0..1]    f16 d (delta)
///   [2..3]    f16 dmin (minimum)
///   [4..15]   12 bytes: 8 × 6-bit sub-block scales (packed)
///   [16..19]  4 bytes: 8 × 4-bit sub-block mins (packed)
///   [20..147] 128 bytes: 256 × 4-bit values (packed nibbles)
pub fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(256), "data length must be a multiple of 256");
    let n_superblocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_superblocks * 148);

    for sb in 0..n_superblocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Compute per-sub-block (32 values each) min and max.
        // Follow llama.cpp's convention of forcing min ≤ 0 when all values
        // in a sub-block happen to be positive. Q4_K encodes `y = sc*nibble
        // - mn` with nibble ∈ [0, 15] — if the effective min were e.g. 0.05
        // instead of 0, all reconstructed values would be shifted down by
        // 0.05 and lose positive-mean sub-blocks entirely.
        let mut sub_mins = [0.0f32; 8];
        let mut sub_maxs = [0.0f32; 8];
        for j in 0..8 {
            let sub = &block[j * 32..(j + 1) * 32];
            let mn = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            sub_mins[j] = mn.min(0.0);
            sub_maxs[j] = mx.max(0.0);
        }

        // Global delta and min.
        //
        // Decode is `x = (d * q_scale) * nibble - (dmin * q_min)` with
        // nibble ∈ [0, 15], q_scale ∈ [0, 63] (6-bit), q_min ∈ [0, 15]
        // (4-bit). To span a sub-block's range with the 15 nibble levels we
        // need `15 * d * q_scale ≈ sub_range`, so when `q_scale = 63`
        // (maximum) and `sub_range = global_max_range`:
        //     d = global_max_range / (15 * 63)
        // Without the factor of 15 in the denominator the effective
        // per-nibble step is too coarse, most values collapse onto nibble=0
        // or 1, and reconstruction loses almost all weight-space detail.
        let global_max_range = sub_maxs.iter().zip(&sub_mins).map(|(a, b)| a - b)
            .fold(0.0f32, f32::max);
        let global_min = sub_mins.iter().copied().fold(f32::INFINITY, f32::min);

        let d = if global_max_range > 0.0 { global_max_range / (15.0 * 63.0) } else { 0.0 };
        let dmin = if global_min < 0.0 { -global_min / 15.0 } else { 0.0 };

        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());

        // Compute 8 sub-block scales and mins.
        // `q_scales[j] = sub_range / (15 * d)` so that `d * q_scales[j] * 15
        // ≈ sub_range` (full dynamic range of 15 nibble levels).
        // `q_mins[j] = |sub_min| / dmin` with q_mins ∈ [0, 15].
        let mut q_scales = [0u8; 8];
        let mut q_mins = [0u8; 8];
        for j in 0..8 {
            let range = sub_maxs[j] - sub_mins[j];
            q_scales[j] = if d > 0.0 {
                (range / (15.0 * d)).round().clamp(0.0, 63.0) as u8
            } else { 0 };
            q_mins[j] = if dmin > 0.0 {
                (-sub_mins[j] / dmin).round().clamp(0.0, 15.0) as u8
            } else { 0 };
        }

        // Pack 6-bit scales into 12 bytes (simplified: only using lower 6 bits of 8 bytes)
        let mut sc_packed = [0u8; 12];
        for j in 0..8 {
            sc_packed[j] = q_scales[j] & 0x3F;
        }
        out.extend_from_slice(&sc_packed);

        // Pack 4-bit mins into 4 bytes
        let mut min_packed = [0u8; 4];
        for j in 0..4 {
            min_packed[j] = (q_mins[j] & 0x0F) | ((q_mins[j + 4] & 0x0F) << 4);
        }
        out.extend_from_slice(&min_packed);

        // Quantize 256 values to 4-bit nibbles
        for j in 0..8 {
            let sc = d * q_scales[j] as f32;
            let mn = dmin * q_mins[j] as f32;
            let inv_sc = if sc > 0.0 { 1.0 / sc } else { 0.0 };
            let sub = &block[j * 32..(j + 1) * 32];

            for i in 0..16 {
                let v0 = ((sub[i * 2] + mn) * inv_sc).round().clamp(0.0, 15.0) as u8;
                let v1 = ((sub[i * 2 + 1] + mn) * inv_sc).round().clamp(0.0, 15.0) as u8;
                out.push(v0 | (v1 << 4));
            }
        }
    }
    out
}

/// Quantize f32 data to Q6_K format (6-bit with sub-block scales, Ollama-compatible).
///
/// Each super-block of 256 floats becomes 210 bytes:
///   [0..127]    128 bytes: lower 4 bits of each value (packed nibbles)
///   [128..191]   64 bytes: upper 2 bits (packed, 4 per byte)
///   [192..207]   16 bytes: 16 × int8 scales (one per 16-value sub-block)
///   [208..209]    2 bytes: f16 super-block scale (d)
pub fn quantize_q6_k(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(256), "data length must be a multiple of 256");
    let n_superblocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_superblocks * 210);

    for sb in 0..n_superblocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Q6_K decode is `x = d * sub_scale * q` with q ∈ [-32, 31] (6-bit
        // signed). To span the sub-block's amax with 31 levels on the
        // positive side: `d * sub_scale * 31 ≈ sub_max`. Picking d so the
        // largest sub-block's sub_scale hits the i8 cap:
        //   d = amax / (31 * 127)         # generous headroom
        // and `sub_scale = round(sub_max / (31 * d))`.
        // The previous `d = amax/32` / `sub_scale = sub_max/d` collapsed
        // most values onto q ∈ {-1, 0, 1} because the scale per level was
        // 32× too coarse.
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let d = amax / (31.0 * 127.0);
        let _inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

        // Compute per-sub-block (16 values) int8 scales.
        let mut sub_scales = [0i8; 16];
        for (j, sub_scale) in sub_scales.iter_mut().enumerate() {
            let sub = &block[j * 16..(j + 1) * 16];
            let sub_max = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let sc = if d > 0.0 { sub_max / (31.0 * d) } else { 0.0 };
            *sub_scale = sc.round().clamp(-128.0, 127.0) as i8;
        }

        // Quantize all 256 values to 6-bit
        let mut q6_vals = [0u8; 256];
        for (j, &sub_scale) in sub_scales.iter().enumerate() {
            let sc = d * sub_scale as f32;
            let inv_sc = if sc.abs() > 1e-10 { 1.0 / sc } else { 0.0 };
            for i in 0..16 {
                let idx = j * 16 + i;
                let q = (block[idx] * inv_sc).round().clamp(-32.0, 31.0) as i8;
                q6_vals[idx] = (q + 32) as u8; // bias to unsigned
            }
        }

        // Pack lower 4 bits: 128 bytes (2 nibbles per byte)
        let mut ql = [0u8; 128];
        for i in 0..128 {
            ql[i] = (q6_vals[i * 2] & 0x0F) | ((q6_vals[i * 2 + 1] & 0x0F) << 4);
        }
        out.extend_from_slice(&ql);

        // Pack upper 2 bits: 64 bytes (4 × 2 bits per byte)
        let mut qh = [0u8; 64];
        for (i, &q6_val) in q6_vals.iter().enumerate() {
            let hi2 = (q6_val >> 4) & 0x03;
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            qh[byte_idx] |= hi2 << bit_offset;
        }
        out.extend_from_slice(&qh);

        // 16 × int8 scales
        for &s in &sub_scales {
            out.push(s as u8);
        }

        // f16 super-block scale
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    out
}

/// Quantize f32 to GGUF Q4_K format (144 bytes per 256 values).
///
/// GGUF layout: half d, half dmin, scales[12] (packed 6-bit scales+mins), qs[128].
/// Scales and mins are packed into the SAME 12-byte array:
///   bytes 0-3: lower 6 bits of scales 0-3
///   bytes 4-7: lower 6 bits of scales 4-7
///   bytes 8-11: upper 2 bits of scales + lower 4 bits of mins
pub fn quantize_q4_k_gguf(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(256));
    let n_superblocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_superblocks * 144);

    for sb in 0..n_superblocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Per-sub-block min/max
        let mut sub_mins = [0.0f32; 8];
        let mut sub_maxs = [0.0f32; 8];
        for j in 0..8 {
            let sub = &block[j * 32..(j + 1) * 32];
            sub_mins[j] = sub.iter().copied().fold(f32::INFINITY, f32::min);
            sub_maxs[j] = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }

        let global_max_range = sub_maxs.iter().zip(&sub_mins).map(|(a, b)| a - b).fold(0.0f32, f32::max);
        let global_min = sub_mins.iter().copied().fold(f32::INFINITY, f32::min);

        let d = if global_max_range > 0.0 { global_max_range / 63.0 } else { 0.0 };
        let dmin = if global_min < 0.0 { -global_min / 63.0 } else { 0.0 };

        // Quantize scales and mins to 6-bit each
        let mut q_scales = [0u8; 8];
        let mut q_mins = [0u8; 8];
        for j in 0..8 {
            let range = sub_maxs[j] - sub_mins[j];
            q_scales[j] = if d > 0.0 { (range / d).round().clamp(0.0, 63.0) as u8 } else { 0 };
            q_mins[j] = if dmin > 0.0 { (-sub_mins[j] / dmin).round().clamp(0.0, 63.0) as u8 } else { 0 };
        }

        // Write d, dmin as f16
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());

        // Pack scales[12]: GGUF format
        // bytes 0-3: (scales[0..4] & 0x3F) | (mins[0..4] << 6)  — lower 6 of scale + lower 2 of min
        // bytes 4-7: (scales[4..8] & 0x3F) | (mins[4..8] << 6)
        // bytes 8-11: upper 4 bits of mins packed
        let mut packed = [0u8; 12];
        for j in 0..4 {
            packed[j] = (q_scales[j] & 0x3F) | ((q_mins[j] & 0x03) << 6);
            packed[j + 4] = (q_scales[j + 4] & 0x3F) | ((q_mins[j + 4] & 0x03) << 6);
        }
        // bytes 8-11: pack upper bits of mins
        packed[8] = ((q_mins[0] >> 2) & 0x0F) | (((q_mins[1] >> 2) & 0x0F) << 4);
        packed[9] = ((q_mins[2] >> 2) & 0x0F) | (((q_mins[3] >> 2) & 0x0F) << 4);
        packed[10] = ((q_mins[4] >> 2) & 0x0F) | (((q_mins[5] >> 2) & 0x0F) << 4);
        packed[11] = ((q_mins[6] >> 2) & 0x0F) | (((q_mins[7] >> 2) & 0x0F) << 4);
        out.extend_from_slice(&packed);

        // Quantize 256 values to 4-bit nibbles
        for j in 0..8 {
            let sc = d * q_scales[j] as f32;
            let mn = dmin * q_mins[j] as f32;
            let inv_sc = if sc > 0.0 { 1.0 / sc } else { 0.0 };
            let sub = &block[j * 32..(j + 1) * 32];
            for i in 0..16 {
                let v0 = ((sub[i * 2] + mn) * inv_sc).round().clamp(0.0, 15.0) as u8;
                let v1 = ((sub[i * 2 + 1] + mn) * inv_sc).round().clamp(0.0, 15.0) as u8;
                out.push(v0 | (v1 << 4));
            }
        }
    }
    out
}

/// Convert Q4_K (148 bytes/block) to GGUF Q4_K (144 bytes/block) for fast GPU inference.
///
/// Processes a flat byte array of Q4_K superblocks. Each 148-byte block becomes 144 bytes.
/// Repacks scale/min headers from separate arrays into GGUF's interleaved 12-byte format.
/// Our 4-bit mins (0-15) fit within GGUF's 6-bit min range (0-63).
pub fn q4k_to_gguf(q4k_data: &[u8]) -> Vec<u8> {
    assert!(q4k_data.len().is_multiple_of(148), "Q4_K data must be a multiple of 148 bytes");
    let n_blocks = q4k_data.len() / 148;
    let mut out = Vec::with_capacity(n_blocks * 144);

    for i in 0..n_blocks {
        let block = &q4k_data[i * 148..];

        // Copy d, dmin (4 bytes — same in both formats)
        out.extend_from_slice(&block[0..4]);

        // Unpack our scales[12] + mins[4] into GGUF packed[12]
        let sc = &block[4..16];
        let mn = &block[16..20];

        let mut q_scales = [0u8; 8];
        let mut q_mins = [0u8; 8];
        for j in 0..4 {
            q_scales[j] = sc[j] & 0x3F;
            q_scales[j + 4] = sc[j + 4] & 0x3F;
            q_mins[j] = mn[j] & 0x0F;
            q_mins[j + 4] = (mn[j] >> 4) & 0x0F;
        }

        // Pack into GGUF format: 12 bytes
        let mut packed = [0u8; 12];
        for j in 0..4 {
            packed[j] = (q_scales[j] & 0x3F) | ((q_mins[j] & 0x03) << 6);
            packed[j + 4] = (q_scales[j + 4] & 0x3F) | ((q_mins[j + 4] & 0x03) << 6);
        }
        packed[8] = ((q_mins[0] >> 2) & 0x0F) | (((q_mins[1] >> 2) & 0x0F) << 4);
        packed[9] = ((q_mins[2] >> 2) & 0x0F) | (((q_mins[3] >> 2) & 0x0F) << 4);
        packed[10] = ((q_mins[4] >> 2) & 0x0F) | (((q_mins[5] >> 2) & 0x0F) << 4);
        packed[11] = ((q_mins[6] >> 2) & 0x0F) | (((q_mins[7] >> 2) & 0x0F) << 4);
        out.extend_from_slice(&packed);

        // Copy nibbles unchanged (128 bytes)
        out.extend_from_slice(&block[20..148]);
    }
    out
}

/// Convert Q4_K data to Q4_KF (pre-baked half scales) for fast GPU inference.
///
/// Q4_KF eliminates ALL header decode + scale unpack from the inference hot loop.
/// Each 148-byte Q4_K superblock becomes 160 bytes:
///   [0..15]    8 × f16 pre-computed d*scale_j (16 bytes)
///   [16..31]   8 × f16 pre-computed dmin*min_j (16 bytes)
///   [32..159]  128 bytes nibbles (unchanged)
pub fn q4k_to_q4kf(q4k_data: &[u8], num_rows: usize, hidden: usize) -> Vec<u8> {
    let superblocks_per_row = hidden / 256;
    let q4k_bytes_per_row = superblocks_per_row * 148;
    let q4kf_bytes_per_row = superblocks_per_row * 160;
    let mut out = Vec::with_capacity(num_rows * q4kf_bytes_per_row);

    for row in 0..num_rows {
        for sb in 0..superblocks_per_row {
            let offset = row * q4k_bytes_per_row + sb * 148;
            let block = &q4k_data[offset..];

            // Decode Q4_K header
            let d_bits = u16::from_le_bytes([block[0], block[1]]);
            let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
            let d = f16_to_f32(d_bits);
            let dmin = f16_to_f32(dmin_bits);

            // Unpack 8 scales and mins, pre-bake products
            let sc_bytes = &block[4..16];
            let min_bytes = &block[16..20];

            let mut scales = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            for j in 0..4 {
                scales[j] = d * (sc_bytes[j] & 0x3F) as f32;
                scales[j + 4] = d * (sc_bytes[j + 4] & 0x3F) as f32;
                mins[j] = dmin * (min_bytes[j] & 0x0F) as f32;
                mins[j + 4] = dmin * ((min_bytes[j] >> 4) & 0x0F) as f32;
            }

            // Write pre-baked scales as f16
            for scale in &scales {
                out.extend_from_slice(&f32_to_f16(*scale).to_le_bytes());
            }
            // Write pre-baked mins as f16
            for min in &mins {
                out.extend_from_slice(&f32_to_f16(*min).to_le_bytes());
            }
            // Copy nibbles unchanged
            out.extend_from_slice(&block[20..148]);
        }
    }
    out
}

/// Quantize f32 data directly to Q4_KF format (pre-baked half scales).
pub fn quantize_q4_kf(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(256), "data length must be a multiple of 256");
    // First quantize to Q4_K, then convert
    let q4k = quantize_q4_k(data);
    let num_rows = 1; // treat as single row
    let hidden = data.len();
    q4k_to_q4kf(&q4k, num_rows, hidden)
}

/// Decode f16 bits to f32 (shared helper).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else { f32::NAN };
    }
    let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
    if sign == 1 { -val } else { val }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_quantize_round_trip() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let (q8, scales) = quantize_to_q8(&x);
        assert_eq!(q8.len(), 64);
        assert_eq!(scales.len(), 2); // 64 / 32
        assert!(scales.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn q8_zero_input() {
        let x = vec![0.0f32; 32];
        let (q8, scales) = quantize_to_q8(&x);
        assert!(q8.iter().all(|&v| v == 0));
        assert!(scales[0] == 0.0);
    }

    // ── quantize_q4_0 tests ──

    #[test]
    fn q4_output_size() {
        // 64 floats = 2 blocks of 32, each block → 18 bytes (2 f16 scale + 16 nibbles)
        let data = vec![1.0f32; 64];
        let q4 = quantize_q4_0(&data);
        assert_eq!(q4.len(), 2 * 18);

        let data = vec![1.0f32; 256];
        let q4 = quantize_q4_0(&data);
        assert_eq!(q4.len(), 8 * 18);
    }

    #[test]
    fn q4_zero_input() {
        let data = vec![0.0f32; 32];
        let q4 = quantize_q4_0(&data);
        assert_eq!(q4.len(), 18);
        // Scale should be zero (f16 zero = 0x0000)
        assert_eq!(q4[0], 0);
        assert_eq!(q4[1], 0);
        // All nibbles should encode 8 (zero quantized = 0 + bias 8)
        for &b in &q4[2..18] {
            assert_eq!(b, 0x88, "zero input should quantize to bias value 0x88");
        }
    }

    #[test]
    fn q4_round_trip_accuracy() {
        // Quantize then dequantize, check values are close
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let q4 = quantize_q4_0(&data);

        // Dequantize: read f16 scale, unpack nibbles, multiply
        let scale_bits = u16::from_le_bytes([q4[0], q4[1]]);
        let scale = f16_to_f32(scale_bits);

        let mut decoded = Vec::with_capacity(32);
        for j in 0..16 {
            let byte = q4[2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            decoded.push(lo as f32 * scale);
            decoded.push(hi as f32 * scale);
        }

        // Check approximate reconstruction (Q4 is lossy, but should be close)
        let max_err: f32 = data.iter().zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 2.0, "Q4 round-trip max error {max_err} exceeds 2.0");
    }

    #[test]
    #[should_panic(expected = "multiple of 32")]
    fn q4_rejects_non_aligned() {
        let data = vec![1.0f32; 33];
        let _ = quantize_q4_0(&data);
    }

    #[test]
    fn q4_matvec_uses_quantized_data() {
        // End-to-end: quantize a matrix, run matvec, verify nonzero output
        let hidden = 256;
        let rows = 64;
        let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4 = quantize_q4_0(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let (q8_x, q8_scales) = quantize_to_q8(&x);

        let mut scores = vec![0.0f32; rows];
        unsafe {
            q4_0_matvec_c(
                q4.as_ptr(), q8_x.as_ptr(), q8_scales.as_ptr(),
                scores.as_mut_ptr(), rows, hidden,
            );
        }
        assert!(scores.iter().any(|&v| v.abs() > 0.01), "Q4 matvec should produce nonzero");
    }

    /// Decode f16 bits to f32 (for test verification).
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as i32;
        let mant = (bits & 0x3FF) as u32;
        if exp == 0 {
            if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
            // Subnormal
            let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
            return if sign == 1 { -val } else { val };
        }
        if exp == 31 {
            return if mant == 0 {
                if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else { f32::NAN };
        }
        let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
        if sign == 1 { -val } else { val }
    }
}
