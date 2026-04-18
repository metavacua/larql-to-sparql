//! Shared Metal shader utilities — f16 decode, constants.

/// Common Metal header included by all shaders.
pub const HEADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Decode an f16 bit-pattern to f32, preserving subnormals.
//
// The previous hand-rolled unpack flushed subnormals to ±0 (the `exp == 0`
// branch returned `sign` for any mantissa). Q4_K and Q6_K super-block scales
// use `d = amax / (31 * 127)` which lands in f16 subnormal range whenever
// the row's amax < ~0.24 — every such row previously decoded as zero on GPU
// while CPU read the correct value, causing silent all-zero rows in V/FFN
// projections.
//
// Using Metal's native `half` cast delegates subnormal handling to the
// hardware's IEEE-754 f16 implementation, which Apple Silicon supports.
static inline float decode_f16_metal(ushort bits) {
    return float(as_type<half>(bits));
}

// Q4_K super-block: 256 values in 148 bytes (larql format).
struct block_q4_K {
    ushort d;           // f16 delta (2 bytes)
    ushort dmin;        // f16 minimum (2 bytes)
    uchar  scales[12];  // 8 × 6-bit sub-block scales packed (12 bytes)
    uchar  mins[4];     // 8 × 4-bit sub-block mins packed (4 bytes)
    uchar  qs[128];     // 256 × 4-bit values (128 bytes)
};                      // Total: 148 bytes

// GGUF Q4_K super-block: 256 values in 144 bytes.
// Scales AND mins packed into 12 bytes (6 bits each).
// This matches llama.cpp/Ollama's exact format.
struct block_q4_K_gguf {
    half   d;           // super-block scale (2 bytes)
    half   dmin;        // super-block min scale (2 bytes)
    uchar  scales[12];  // 8 scales + 8 mins packed in 6 bits each
    uchar  qs[128];     // 256 × 4-bit values (128 bytes)
};                      // Total: 144 bytes

// Q4_KF super-block: 256 values in 160 bytes.
// Pre-baked scales: d*scale_j and dmin*min_j pre-computed as half.
// Eliminates ALL header decode + scale unpack from the inference hot loop.
struct block_q4_kf {
    half   scales[8];   // pre-computed d * scale_j (16 bytes)
    half   mins[8];     // pre-computed dmin * min_j (16 bytes)
    uchar  qs[128];     // 256 × 4-bit values (128 bytes)
};                      // Total: 160 bytes
"#;
