//! Fixtures and CPU oracles shared by `test_matmul_trait_arms.rs`.
//!
//! Every oracle here is a plain left-to-right loop over the *dequantised*
//! weights, so a GPU arm agreeing with it proves the kernel read the
//! format and the reduction — not that two GPU paths agree with each
//! other. Tolerances are named at the use site.

#![allow(dead_code)]

use larql_models::quant::mxfp4::{
    dequantize_expert, MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS, MXFP4_TABLE,
};

/// Multiplier for the xorshift32 seed mix; keeps fixtures deterministic
/// per seed while decorrelating neighbouring seeds.
const SEED_MIX: u32 = 2_654_435_761;

/// E8M0 byte encoding the multiplier `1.0` (`2^(127-127)`).
const E8M0_ONE: u8 = 127;
/// Spread of E8M0 exponents below `E8M0_ONE` that the MXFP4 fixture
/// cycles through, so the scale stream carries distinct bytes per group
/// and a kernel indexing it wrongly cannot agree with the oracle.
const E8M0_EXPONENT_SPREAD: u8 = 6;

/// Deterministic uniform values in `[-1, 1)` — an xorshift32 stream.
pub fn uniform_values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(SEED_MIX).wrapping_add(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// `out[r] = sum_c w[r, c] * x[c]` over a row-major `rows × k` matrix,
/// accumulated left to right in f32 — the oracle every GPU arm is
/// judged against.
pub fn cpu_gemv(w: &[f32], x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    assert_eq!(w.len(), rows * k, "oracle: weight length");
    assert_eq!(x.len(), k, "oracle: input length");
    (0..rows)
        .map(|r| {
            w[r * k..(r + 1) * k]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

/// Largest `|ref - got|` divided by `max |ref|`, so rows whose dot
/// product is naturally small are not judged against an absolute
/// epsilon. Panics on a length mismatch — a wrong output length is a
/// failure, not a small error.
pub fn rel_error(reference: &[f32], got: &[f32]) -> f32 {
    assert_eq!(reference.len(), got.len(), "output length mismatch");
    let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    reference
        .iter()
        .zip(got)
        .map(|(r, g)| (r - g).abs())
        .fold(0.0f32, f32::max)
        / scale
}

/// Index of the largest finite score, ties to the lowest index.
pub fn cpu_argmax(scores: &[f32]) -> (u32, f32) {
    let (mut best_i, mut best_v) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in scores.iter().enumerate() {
        if v.is_finite() && v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    (best_i as u32, best_v)
}

/// `(index, score)` of the `top_k` largest scores, sorted descending.
pub fn cpu_topk(scores: &[f32], top_k: usize) -> Vec<(u32, f32)> {
    let mut ranked: Vec<(u32, f32)> = scores
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(top_k);
    ranked
}

/// Smallest gap between consecutive entries of a descending-sorted
/// score list. A fixture whose top scores are closer than the GPU
/// reduction tolerance cannot pin an argmax, so tests assert this is
/// comfortably larger than the tolerance before trusting the index.
pub fn min_gap_between_top(scores: &[f32], top_k: usize) -> f32 {
    let ranked = cpu_topk(scores, top_k + 1);
    ranked
        .windows(2)
        .map(|w| w[0].1 - w[1].1)
        .fold(f32::INFINITY, f32::min)
}

/// A packed MXFP4 matrix plus its scale stream and the dequantised
/// weights it denotes.
pub struct Mxfp4Fixture {
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub dequantised: Vec<f32>,
}

/// Build an MXFP4 matrix whose nibble stream cycles through every code
/// of the table and whose E8M0 scales differ per group. Derived from
/// `seed` so the two fixtures in a `_multi` batch are different
/// matrices. `k` must be a multiple of `MXFP4_GROUP_ELEMS`.
pub fn mxfp4_fixture(rows: usize, k: usize, seed: u32) -> Mxfp4Fixture {
    assert!(
        k.is_multiple_of(MXFP4_GROUP_ELEMS),
        "fixture: k must be group-aligned"
    );
    let groups = k / MXFP4_GROUP_ELEMS;
    let mut state = seed.wrapping_mul(SEED_MIX).wrapping_add(1);
    let mut next_byte = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state >> 24) as u8
    };
    let packed: Vec<u8> = (0..rows * groups * MXFP4_GROUP_BYTES)
        .map(|_| next_byte())
        .collect();
    let scales: Vec<u8> = (0..rows * groups)
        .map(|i| E8M0_ONE - ((i as u8).wrapping_add(next_byte()) % E8M0_EXPONENT_SPREAD))
        .collect();
    let dequantised = dequantize_expert(&packed, &scales, rows, groups).expect("dequantise");
    // Fixture sanity: the codes actually span the table, so a kernel
    // with a wrong LUT cannot agree with the oracle by accident.
    let mut seen = [false; MXFP4_TABLE.len()];
    for &b in &packed {
        seen[(b & 0x0F) as usize] = true;
        seen[(b >> 4) as usize] = true;
    }
    assert!(
        seen.iter().all(|&s| s),
        "fixture: nibble stream must exercise every MXFP4 code"
    );
    Mxfp4Fixture {
        packed,
        scales,
        dequantised,
    }
}
