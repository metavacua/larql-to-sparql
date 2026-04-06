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
"#;
