//! Rotary Position Embedding (RoPE) — applies position-dependent rotation to Q/K vectors.
//!
//! Split-half pairing: rotates (x[i], x[i + half_dim]) pairs.
//! Matches HuggingFace default and MLX traditional=False.

pub const SHADER: &str = r#"
// Apply RoPE to a [seq_len, dim] matrix in-place.
// Each thread handles one (position, dimension_pair).
// Grid: (dim/2, seq_len, 1).
kernel void rope_apply(
    device float* x         [[buffer(0)]],   // [seq_len, dim] — modified in-place
    constant uint&  dim     [[buffer(1)]],
    constant float& base    [[buffer(2)]],   // rope_theta (e.g., 10000.0 or 1000000.0)
    uint2 tid [[thread_position_in_grid]])
{
    uint d = tid.x;           // dimension pair index [0, dim/2)
    uint pos = tid.y;         // sequence position
    uint hdim = dim / 2;
    if (d >= hdim) return;

    float freq = 1.0f / pow(base, float(2 * d) / float(dim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint idx_re = pos * dim + d;
    uint idx_im = pos * dim + d + hdim;

    float re = x[idx_re];
    float im = x[idx_im];

    x[idx_re] = re * cos_a - im * sin_a;
    x[idx_im] = re * sin_a + im * cos_a;
}
"#;
