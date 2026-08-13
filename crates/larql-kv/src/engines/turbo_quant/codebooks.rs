//! Lloyd-Max codebooks for the turbo-quant codec.
//!
//! After the orthonormal WHT of a unit-norm vector in d dimensions, each
//! coordinate follows Beta(d/2, d/2) (rescaled to [-1, 1]) with mean 0 and
//! standard deviation **exactly 1/√d** — the rotation preserves the unit
//! norm, and the squared coordinates sum to 1 over d slots. For d ≥ 32 this
//! is close enough to N(0, 1/d) that a Gaussian Lloyd-Max quantizer is the
//! right scalar quantizer.
//!
//! Lloyd-Max quantizers for Gaussians scale linearly with sigma, so ONE
//! table per bit-width trained on N(0, 1) serves every block dim the WHT
//! accepts: encode multiplies coordinates by √d into unit-sigma space,
//! decode multiplies centroids back by 1/√d.
//! `scaled_unit_table_matches_per_sigma_training` pins the equivalence
//! numerically; `unit_codebook_matches_regenerated` pins the tables to the
//! in-repo generator.

// `unit_codebook` is one of the round's rare genuine REIMPLEMENTATIONS,
// not an exclusion: it is pure arithmetic over the already-const
// centroid arrays, so wasm32v1-none CAN express it -- it just can't
// cache the result behind `std::sync::LazyLock` (no core/alloc
// equivalent). Native keeps its zero-copy `LazyLock`-cached path
// unchanged; wasm32 rebuilds the (tiny, 8- or 16-entry) codebook per
// call. Both arms return byte-identical values -- `Cow<'static,
// Codebook>` lets both share one return type and one set of call
// sites in `engine.rs`.
#[cfg(target_arch = "wasm32")]
use alloc::borrow::Cow;
#[cfg(not(target_arch = "wasm32"))]
use std::borrow::Cow;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::LazyLock;
// `wht_coordinate_sigma` below calls `f32::sqrt`, which needs
// `num_traits::Float` in scope under `#![no_std]` (core's f32 has no
// sqrt/exp/tanh/etc without a libm backend).
#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

use super::lloyd_max::Codebook;

/// Number of Lloyd-Max samples / iterations used to generate the unit
/// tables below (and to regenerate them in the pinning test).
#[cfg(test)]
const GEN_SAMPLES: usize = 100_000;
#[cfg(test)]
const GEN_MAX_ITERS: usize = 200;
#[cfg(test)]
const GEN_SEED: u64 = 12345;

/// 3-bit (8-level) Lloyd-Max centroids for N(0, 1).
///
/// Generated 2026-07-30 with `lloyd_max::compute_codebook` over 100_000
/// samples of N(0, 1) (StdRng seed 12345, 200 Lloyd iterations), then
/// symmetrized as c[i] := (c[i] − c[n−1−i]) / 2 — the target density is
/// even, so the optimal codebook is symmetric and mirroring halves the
/// sampling noise.
pub const UNIT_CENTROIDS_3BIT: [f32; 8] = [
    -2.1522968,
    -1.3450537,
    -0.7562692,
    -0.24459186,
    0.24459186,
    0.7562692,
    1.3450537,
    2.1522968,
];

/// 4-bit (16-level) Lloyd-Max centroids for N(0, 1).
///
/// Same generation provenance as [`UNIT_CENTROIDS_3BIT`].
pub const UNIT_CENTROIDS_4BIT: [f32; 16] = [
    -2.6791892,
    -2.0297308,
    -1.5931331,
    -1.2384968,
    -0.9306258,
    -0.6500872,
    -0.38392428,
    -0.12723812,
    0.12723812,
    0.38392428,
    0.6500872,
    0.9306258,
    1.2384968,
    1.5931331,
    2.0297308,
    2.6791892,
];

/// Coordinate standard deviation of a WHT-rotated unit-norm vector in
/// `dim` dimensions: exactly 1/√dim (norm preservation + isotropy).
/// Encode divides by this (scales into the unit-sigma codebook space);
/// decode multiplies by it.
pub fn wht_coordinate_sigma(dim: usize) -> f32 {
    1.0 / (dim as f32).sqrt()
}

#[cfg(not(target_arch = "wasm32"))]
static UNIT_CODEBOOK_3BIT: LazyLock<Codebook> =
    LazyLock::new(|| codebook_from_centroids(&UNIT_CENTROIDS_3BIT));

#[cfg(not(target_arch = "wasm32"))]
static UNIT_CODEBOOK_4BIT: LazyLock<Codebook> =
    LazyLock::new(|| codebook_from_centroids(&UNIT_CENTROIDS_4BIT));

/// Boundaries are the Lloyd-Max fixed point's midpoints between adjacent
/// centroids — derived, not stored.
fn codebook_from_centroids(centroids: &[f32]) -> Codebook {
    let boundaries = centroids.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
    Codebook {
        boundaries,
        centroids: centroids.to_vec(),
    }
}

/// Unit-sigma (N(0, 1)) codebook for a bit-width. Callers scale values by
/// [`wht_coordinate_sigma`] for the actual block dim; every power-of-two
/// dim the WHT accepts gets a correctly-scaled codebook (no silent
/// fallback to a mis-scaled table).
///
/// Native: `Cow::Borrowed` from the process-lifetime `LazyLock` cache
/// (computed once, zero-copy thereafter — unchanged behaviour). Wasm32:
/// `Cow::Owned`, rebuilt from the const centroid arrays on every call
/// (no LazyLock/OnceLock under core+alloc-only) — the one genuine
/// reimplementation in this round; see the module-level doc comment.
pub fn unit_codebook(bits: u8) -> Cow<'static, Codebook> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        match bits {
            3 => Cow::Borrowed(&*UNIT_CODEBOOK_3BIT),
            4 => Cow::Borrowed(&*UNIT_CODEBOOK_4BIT),
            _ => panic!("turbo-quant: unsupported bit width {bits} (expected 3 or 4)"),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        match bits {
            3 => Cow::Owned(codebook_from_centroids(&UNIT_CENTROIDS_3BIT)),
            4 => Cow::Owned(codebook_from_centroids(&UNIT_CENTROIDS_4BIT)),
            _ => panic!("turbo-quant: unsupported bit width {bits} (expected 3 or 4)"),
        }
    }
}

/// Build a Lloyd-Max codebook for N(0, sigma²) from samples. Test-only:
/// used to regenerate and pin the constant tables above.
#[cfg(test)]
fn make_gaussian_codebook(n_levels: usize, sigma: f32) -> Codebook {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(GEN_SEED);
    let dist = Normal::new(0.0f32, sigma).unwrap();
    let samples: Vec<f32> = (0..GEN_SAMPLES).map(|_| rng.sample(dist)).collect();

    super::lloyd_max::compute_codebook(&samples, n_levels, GEN_MAX_ITERS)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regeneration must reproduce the pasted constants to within f32
    /// noise — this is the provenance pin for the tables.
    const REGEN_TOL: f32 = 1e-6;

    /// Max acceptable per-centroid difference (in unit-sigma scale)
    /// between a table trained at sigma = 1/√d and the unit table
    /// scaled by the same sigma. Lloyd-Max is scale-equivariant, so
    /// this is f32 rounding only (measured 0.0 exactly).
    const SCALE_EQUIVALENCE_TOL: f32 = 1e-4;

    fn symmetrized(cb: &Codebook) -> Vec<f32> {
        let n = cb.centroids.len();
        (0..n)
            .map(|i| (cb.centroids[i] - cb.centroids[n - 1 - i]) / 2.0)
            .collect()
    }

    #[test]
    fn unit_codebook_matches_regenerated() {
        for (n_levels, table) in [
            (8usize, &UNIT_CENTROIDS_3BIT[..]),
            (16, &UNIT_CENTROIDS_4BIT[..]),
        ] {
            let regen = symmetrized(&make_gaussian_codebook(n_levels, 1.0));
            for (i, (a, b)) in regen.iter().zip(table.iter()).enumerate() {
                assert!(
                    (a - b).abs() < REGEN_TOL,
                    "n_levels={n_levels} centroid {i}: regenerated {a} vs constant {b}"
                );
            }
        }
    }

    /// A codebook trained directly at sigma = 1/√d must equal the unit
    /// table scaled by that sigma — this is what licenses serving every
    /// block dim from one table per bit-width.
    #[test]
    fn scaled_unit_table_matches_per_sigma_training() {
        for (d, n_levels, unit_table) in [
            (256usize, 16usize, &UNIT_CENTROIDS_4BIT[..]),
            (128, 8, &UNIT_CENTROIDS_3BIT[..]),
        ] {
            let sigma = wht_coordinate_sigma(d);
            let trained = symmetrized(&make_gaussian_codebook(n_levels, sigma));
            for (i, (t, u)) in trained.iter().zip(unit_table.iter()).enumerate() {
                let diff_unit_scale = (t / sigma - u).abs();
                assert!(
                    diff_unit_scale < SCALE_EQUIVALENCE_TOL,
                    "d={d} n_levels={n_levels} centroid {i}: trained/sigma = {} \
                     vs unit {u} (diff {diff_unit_scale})",
                    t / sigma
                );
            }
        }
    }

    #[test]
    fn unit_codebook_4bit_has_16_centroids() {
        let cb = unit_codebook(4);
        assert_eq!(cb.centroids.len(), 16);
        assert_eq!(cb.boundaries.len(), 15);
    }

    #[test]
    fn unit_codebook_3bit_has_8_centroids() {
        let cb = unit_codebook(3);
        assert_eq!(cb.centroids.len(), 8);
        assert_eq!(cb.boundaries.len(), 7);
    }

    #[test]
    fn unit_codebook_centroids_sorted() {
        for bits in [3u8, 4] {
            let cb = unit_codebook(bits);
            for w in cb.centroids.windows(2) {
                assert!(w[0] < w[1], "{bits}-bit: centroids not sorted");
            }
        }
    }

    #[test]
    fn unit_codebook_exactly_symmetric() {
        for bits in [3u8, 4] {
            let cb = unit_codebook(bits);
            let n = cb.centroids.len();
            for i in 0..n / 2 {
                assert_eq!(
                    cb.centroids[i],
                    -cb.centroids[n - 1 - i],
                    "{bits}-bit codebook not symmetric at index {i}"
                );
            }
        }
    }

    #[test]
    fn wht_coordinate_sigma_values() {
        assert_eq!(wht_coordinate_sigma(256), 1.0 / 16.0);
        assert_eq!(wht_coordinate_sigma(64), 1.0 / 8.0);
        assert_eq!(wht_coordinate_sigma(4), 0.5);
    }

    #[test]
    #[should_panic(expected = "unsupported bit width")]
    fn unit_codebook_rejects_unsupported_bits() {
        unit_codebook(5);
    }
}
