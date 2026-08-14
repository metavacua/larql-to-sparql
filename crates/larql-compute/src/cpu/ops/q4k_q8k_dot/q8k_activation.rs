use super::common::{ELEMS_PER_BLOCK, SUBBLOCKS_PER_BLOCK, SUBBLOCK_SIZE};

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
