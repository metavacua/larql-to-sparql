use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;

use super::{f16_to_f32, f32_to_f16};

/// Pre-quantize f32 vector to Q8_0 (int8 + per-block f32 scale).
pub fn quantize_to_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n_blocks = x.len() / LEGACY_BLOCK_ELEMS;
    let mut q8 = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; n_blocks];
    for (b, scale_out) in scales.iter_mut().enumerate().take(n_blocks) {
        let off = b * LEGACY_BLOCK_ELEMS;
        let block = &x[off..off + LEGACY_BLOCK_ELEMS];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        *scale_out = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for j in 0..LEGACY_BLOCK_ELEMS {
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
    assert!(
        data.len().is_multiple_of(LEGACY_BLOCK_ELEMS),
        "data length must be a multiple of 32"
    );
    let n_blocks = data.len() / LEGACY_BLOCK_ELEMS;
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
        let f16 = if exp == 0 {
            sign as u16
        } else if exp == 255 {
            (sign | 0x7C00 | (mant >> 13)) as u16
        } else {
            let new_exp = exp - 127 + 15;
            if new_exp >= 31 {
                (sign | 0x7C00) as u16
            } else if new_exp <= 0 {
                sign as u16
            } else {
                (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
            }
        };
        out.extend_from_slice(&f16.to_le_bytes());
        // ggml planar nibble layout (`quantize_row_q4_0_ref`): byte j packs
        // element j (low nibble) and element j+16 (high nibble).
        for j in 0..16 {
            let lo = ((block[j] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((block[j + 16] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// Quantize f32 data to Q4_K format — the canonical llama.cpp / GGUF
/// layout (Ollama-compatible, 144 bytes per 256-element super-block).
///
/// Block layout (matches `kernel_mul_mv_q4_K_f32` in llama.cpp and the
/// `q4kf_proj` / `q4kf_qkv_proj` Metal shaders):
///   [0..1]    f16 d (super-block scale)
///   [2..3]    f16 dmin (super-block min)
///   [4..15]   12 bytes packing 8 × 6-bit `q_scales` + 8 × 6-bit `q_mins`
///             via `get_scale_min_k4`.
///   [16..143] 128 bytes of 4-bit nibbles arranged as FOUR 32-byte groups.
///             Each group holds TWO adjacent sub-blocks — low nibbles go
///             to sub-block `2g`, high nibbles go to sub-block `2g+1`.
///             `scales[2g]` / `mins[2g]` scale the low nibbles,
///             `scales[2g+1]` / `mins[2g+1]` scale the high nibbles.
///
/// Round-trips exactly through `dequantize_q4_k` in this crate and
/// `larql_models::quant::ggml::dequantize_q4_k`, and decodes identically
/// via the Metal shaders and llama.cpp's reference `dequantize_row_q4_K`.
pub fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    assert!(
        data.len().is_multiple_of(256),
        "data length must be a multiple of 256"
    );
    let n_superblocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_superblocks * 144);

    for sb in 0..n_superblocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Per-sub-block min/max — force min ≤ 0 so purely-positive
        // sub-blocks don't get shifted down by their own baseline.
        let mut sub_mins = [0.0f32; 8];
        let mut sub_maxs = [0.0f32; 8];
        for j in 0..8 {
            let sub = &block[j * 32..(j + 1) * 32];
            let mn = sub.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            sub_mins[j] = mn.min(0.0);
            sub_maxs[j] = mx.max(0.0);
        }

        let global_max_range = sub_maxs
            .iter()
            .zip(&sub_mins)
            .map(|(a, b)| a - b)
            .fold(0.0f32, f32::max);
        let global_min = sub_mins.iter().copied().fold(f32::INFINITY, f32::min);

        // Q4_K decode is `x = (d * q_scale) * nibble - (dmin * q_min)`
        // with nibble ∈ [0, 15], q_scale ∈ [0, 63], q_min ∈ [0, 63].
        let d = if global_max_range > 0.0 {
            global_max_range / (15.0 * 63.0)
        } else {
            0.0
        };
        let dmin = if global_min < 0.0 {
            -global_min / 63.0
        } else {
            0.0
        };

        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());

        let mut q_scales = [0u8; 8];
        let mut q_mins = [0u8; 8];
        for j in 0..8 {
            let range = sub_maxs[j] - sub_mins[j];
            q_scales[j] = if d > 0.0 {
                (range / (15.0 * d)).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
            q_mins[j] = if dmin > 0.0 {
                (-sub_mins[j] / dmin).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
        }

        // 12-byte scales + mins packing, `get_scale_min_k4` reference:
        //   j < 4: scales[j] = packed[j]     & 0x3F
        //          mins[j]   = packed[j+4]   & 0x3F
        //   j ≥ 4: scales[j] = (packed[j+4] & 0x0F) | ((packed[j-4] >> 6) << 4)
        //          mins[j]   = (packed[j+4] >> 4)   | ((packed[j]   >> 6) << 4)
        let mut packed = [0u8; 12];
        for j in 0..4 {
            packed[j] = (q_scales[j] & 0x3F) | (((q_scales[j + 4] >> 4) & 0x03) << 6);
            packed[j + 4] = (q_mins[j] & 0x3F) | (((q_mins[j + 4] >> 4) & 0x03) << 6);
            packed[j + 8] = (q_scales[j + 4] & 0x0F) | ((q_mins[j + 4] & 0x0F) << 4);
        }
        out.extend_from_slice(&packed);

        // Nibble packing: llama.cpp groups two adjacent sub-blocks into
        // one 32-byte span. For group `g` ∈ [0,4):
        //   byte[g*32 + l].low_nibble  = encoded sub-block `2g`   value `l`
        //   byte[g*32 + l].high_nibble = encoded sub-block `2g+1` value `l`
        // Encoding uses that sub-block's own scale/min:
        //   enc = round((v + dmin*q_min) / (d*q_scale)) clamped to [0, 15]
        for g in 0..4 {
            let sb_lo = 2 * g;
            let sb_hi = 2 * g + 1;
            let sc_lo = d * q_scales[sb_lo] as f32;
            let sc_hi = d * q_scales[sb_hi] as f32;
            let mn_lo = dmin * q_mins[sb_lo] as f32;
            let mn_hi = dmin * q_mins[sb_hi] as f32;
            let inv_lo = if sc_lo > 0.0 { 1.0 / sc_lo } else { 0.0 };
            let inv_hi = if sc_hi > 0.0 { 1.0 / sc_hi } else { 0.0 };
            let lo_sub = &block[sb_lo * 32..(sb_lo + 1) * 32];
            let hi_sub = &block[sb_hi * 32..(sb_hi + 1) * 32];
            for l in 0..32 {
                let lo = ((lo_sub[l] + mn_lo) * inv_lo).round().clamp(0.0, 15.0) as u8;
                let hi = ((hi_sub[l] + mn_hi) * inv_hi).round().clamp(0.0, 15.0) as u8;
                out.push(lo | (hi << 4));
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
    assert!(
        data.len().is_multiple_of(256),
        "data length must be a multiple of 256"
    );
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

        // Pack per ggml's planar Q6_K layout (`quantize_row_q6_K_ref`):
        // within each 128-element half, ql[l] holds element l in its low
        // nibble and element l+64 in its high nibble; ql[l+32] holds
        // elements l+32 / l+96. qh[l] packs the two high bits of elements
        // l, l+32, l+64, l+96 at shifts 0/2/4/6.
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for half in 0..2 {
            let e = half * 128; // element base for this half
            for l in 0..32 {
                let q1 = q6_vals[e + l];
                let q2 = q6_vals[e + l + 32];
                let q3 = q6_vals[e + l + 64];
                let q4 = q6_vals[e + l + 96];
                ql[half * 64 + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
                ql[half * 64 + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);
                qh[half * 32 + l] = ((q1 >> 4) & 3)
                    | (((q2 >> 4) & 3) << 2)
                    | (((q3 >> 4) & 3) << 4)
                    | (((q4 >> 4) & 3) << 6);
            }
        }
        out.extend_from_slice(&ql);
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

/// Convert Q4_K data (144-byte GGUF layout) to Q4_KF (pre-baked half
/// scales) for fast GPU inference.
///
/// Q4_KF eliminates all header decode + scale unpack from the inference
/// hot loop. Each 144-byte Q4_K superblock becomes 160 bytes:
///   [0..15]    8 × f16 pre-computed d*scale_j (16 bytes)
///   [16..31]   8 × f16 pre-computed dmin*min_j (16 bytes)
///   [32..159]  128 bytes nibbles (unchanged)
/// **No Metal kernel consumes this 160-byte layout** (capability audit
/// F15): the live `Q4_KF`-tagged shaders read standard 144-byte GGUF
/// Q4_K blocks. This conversion survives as the CPU-side pre-baked
/// experiment only — do not feed its output to the Metal Q4_KF path.
pub fn q4k_to_q4kf(q4k_data: &[u8], num_rows: usize, hidden: usize) -> Vec<u8> {
    let superblocks_per_row = hidden / 256;
    let q4k_bytes_per_row = superblocks_per_row * 144;
    let q4kf_bytes_per_row = superblocks_per_row * 160;
    let mut out = Vec::with_capacity(num_rows * q4kf_bytes_per_row);

    for row in 0..num_rows {
        for sb in 0..superblocks_per_row {
            let offset = row * q4k_bytes_per_row + sb * 144;
            let block = &q4k_data[offset..offset + 144];

            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

            // Unpack scales + mins per llama.cpp's `get_scale_min_k4`.
            let p = &block[4..16];
            let mut q_scales = [0u8; 8];
            let mut q_mins = [0u8; 8];
            for j in 0..4 {
                q_scales[j] = p[j] & 0x3F;
                q_mins[j] = p[j + 4] & 0x3F;
                q_scales[j + 4] = (p[j + 8] & 0x0F) | ((p[j] >> 6) << 4);
                q_mins[j + 4] = (p[j + 8] >> 4) | ((p[j + 4] >> 6) << 4);
            }

            // Pre-bake d·scale and dmin·min, write as f16.
            for &qs in &q_scales {
                let s = d * qs as f32;
                out.extend_from_slice(&f32_to_f16(s).to_le_bytes());
            }
            for &qm in &q_mins {
                let m = dmin * qm as f32;
                out.extend_from_slice(&f32_to_f16(m).to_le_bytes());
            }
            // Copy 128 nibble bytes unchanged.
            out.extend_from_slice(&block[16..144]);
        }
    }
    out
}

/// Quantize f32 data directly to Q4_KF format (pre-baked half scales).
pub fn quantize_q4_kf(data: &[f32]) -> Vec<u8> {
    assert!(
        data.len().is_multiple_of(256),
        "data length must be a multiple of 256"
    );
    // First quantize to Q4_K, then convert
    let q4k = quantize_q4_k(data);
    let num_rows = 1; // treat as single row
    let hidden = data.len();
    q4k_to_q4kf(&q4k, num_rows, hidden)
}
