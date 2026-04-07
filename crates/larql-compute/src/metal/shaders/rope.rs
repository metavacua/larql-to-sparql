//! Rotary Position Embedding (RoPE) — applies position-dependent rotation to Q/K vectors.
//!
//! Split-half pairing: rotates (x[i], x[i + half_dim]) pairs.
//! Matches HuggingFace default and MLX traditional=False.

pub const SHADER: &str = r#"
// Apply RoPE to a single vector [dim] in-place at a given absolute position.
// Used by KV-cached decode: apply to Q and K at the correct sequence position.
// Grid: (dim/2, 1, 1).
kernel void rope_at_pos(
    device float* x         [[buffer(0)]],   // [dim] — modified in-place (one head)
    constant uint&  dim     [[buffer(1)]],   // head_dim
    constant float& base    [[buffer(2)]],   // rope_theta
    constant uint&  pos     [[buffer(3)]],   // absolute position in sequence
    uint tid [[thread_position_in_grid]])
{
    uint hdim = dim / 2;
    if (tid >= hdim) return;

    float freq = 1.0f / pow(base, float(2 * tid) / float(dim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float re = x[tid];
    float im = x[tid + hdim];

    x[tid]        = re * cos_a - im * sin_a;
    x[tid + hdim] = re * sin_a + im * cos_a;
}

// Apply RoPE to a [seq_len, dim] matrix in-place.
// Supports partial rotation: only the first `rotary_dim` dimensions are rotated,
// the rest pass through unchanged.
// Each thread handles one (position, dimension_pair).
// Grid: (rotary_dim/2, seq_len, 1).
kernel void rope_apply(
    device float* x           [[buffer(0)]],   // [seq_len, dim] — modified in-place
    constant uint&  dim       [[buffer(1)]],
    constant float& base      [[buffer(2)]],   // rope_theta (e.g., 10000.0 or 1000000.0)
    constant uint&  rotary_dim[[buffer(3)]],   // dimensions to rotate (≤ dim). 0 = use dim.
    uint2 tid [[thread_position_in_grid]])
{
    uint rdim = (rotary_dim == 0) ? dim : min(rotary_dim, dim);
    uint d = tid.x;           // dimension pair index [0, rdim/2)
    uint pos = tid.y;         // sequence position
    uint hdim = rdim / 2;
    if (d >= hdim) return;

    float freq = 1.0f / pow(base, float(2 * d) / float(rdim));
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
