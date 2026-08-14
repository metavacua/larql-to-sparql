//! Q8 quantization: f32 ��� int8 + per-block scale.
//! Used for chaining layers: layer N output (f32) → Q8 → layer N+1 input.
//! One thread per block of 32 elements.

pub const SHADER: &str = r#"
kernel void quantize_q8(
    device const float* input  [[buffer(0)]],
    device char*        q8_out [[buffer(1)]],
    device float*       scales [[buffer(2)]],
    constant uint&      K      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint block = tid;
    // Round UP: several hosts dispatch div_ceil(K, 32) threads, and the
    // previous truncating K / 32 left the final partial block's scale
    // slot unwritten — downstream Q8 matvecs then read an uninitialised
    // scale (capability audit F6). The partial block quantises its
    // K % 32 real elements only.
    uint num_blocks = (K + 31u) / 32u;
    if (block >= num_blocks) return;
    uint off = block * 32;
    uint n = min(32u, K - off);
    float amax = 0.0f;
    for (uint j = 0; j < n; j++) {
        float v = abs(input[off + j]);
        if (v > amax) amax = v;
    }
    float scale = amax / 127.0f;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    scales[block] = scale;
    for (uint j = 0; j < n; j++) {
        float v = input[off + j] * inv;
        v = clamp(v, -128.0f, 127.0f);
        q8_out[off + j] = char(int(round(v)));
    }
}
"#;

pub struct Kernel;
impl crate::kernels::ShaderKernel for Kernel {
    const KERNEL_NAME: &'static str = "quantize_q8";
}
