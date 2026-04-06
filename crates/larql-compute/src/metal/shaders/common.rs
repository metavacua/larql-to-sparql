//! Shared Metal shader utilities — f16 decode, constants.

/// Common Metal header included by all shaders.
pub const HEADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float decode_f16_metal(ushort bits) {
    uint sign = uint(bits & 0x8000) << 16;
    uint exp = (bits >> 10) & 0x1F;
    uint mant = bits & 0x3FF;
    if (exp == 0) return as_type<float>(sign);
    exp = exp + 127 - 15;
    return as_type<float>(sign | (exp << 23) | (mant << 13));
}

// Q4_K super-block: 256 values in 148 bytes.
// Struct layout matches the serialized format exactly.
struct block_q4_K {
    ushort d;           // f16 delta (2 bytes)
    ushort dmin;        // f16 minimum (2 bytes)
    uchar  scales[12];  // 8 × 6-bit sub-block scales packed (12 bytes)
    uchar  mins[4];     // 8 × 4-bit sub-block mins packed (4 bytes)
    uchar  qs[128];     // 256 × 4-bit values (128 bytes)
};                      // Total: 148 bytes

// Q4_KF super-block: 256 values in 160 bytes.
// Pre-baked scales: d*scale_j and dmin*min_j pre-computed as half.
// Eliminates ALL header decode + scale unpack from the inference hot loop.
struct block_q4_kf {
    half   scales[8];   // pre-computed d * scale_j (16 bytes)
    half   mins[8];     // pre-computed dmin * min_j (16 bytes)
    uchar  qs[128];     // 256 × 4-bit values (128 bytes)
};                      // Total: 160 bytes
"#;
