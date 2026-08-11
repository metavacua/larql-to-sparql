//! A conditioning context that survives the expert row permutation.
//!
//! Pure. **This module exists because of a confound that a coordinate-aligned
//! census cannot see and its synthetic tests cannot reproduce.**
//!
//! A routed expert computes `w2 · act(w1 x, w3 x)`. Permute the rows of `w1` and
//! `w3` and the matching columns of `w2` and the expert computes *exactly the
//! same function*. The intermediate index is therefore arbitrary per expert:
//! there is no reason for row `r` of expert 0 to correspond to row `r` of expert
//! 1, and the checkpoint carries no canonical ordering.
//!
//! ```text
//!   w1 : [moe_intermediate = 3072, latent = 3584]
//!          ^ rows: PERMUTATION-ARBITRARY per expert
//!                                    ^ columns: shared input channels
//! ```
//!
//! So a full-coordinate `(row, column)` alignment across experts compares one
//! arbitrary pairing out of `3072!`. A null result from it says "these experts
//! do not resemble each other *at the identity permutation*", which is very
//! nearly no claim at all — the identity is not privileged.
//!
//! The column axis has no such freedom. Both experts read the same latent
//! vector, so column `c` means the same input channel in every expert of the
//! layer. Conditioning on a **per-column** profile is invariant to the row
//! permutation and therefore measures redundancy that survives it.
//!
//! Two consequences worth stating plainly:
//!
//! - A positive column-profile result is a **real** transport lever: the decoder
//!   needs no row correspondence to use it.
//! - A negative one still does not refute inter-expert redundancy in general,
//!   because a permutation-aligned comparison (an assignment problem over 3072
//!   rows) remains unmeasured. It refutes only the two orderings actually tested.

use super::symbol_census::FP4_CODES;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Per-input-channel symbol histogram pooled over every row of every expert.
pub struct ColumnProfile {
    per_column: Vec<[u64; FP4_CODES]>,
    row_len: usize,
}

impl ColumnProfile {
    /// `None` if the streams are ragged, empty, or not a whole number of rows.
    pub fn build(streams: &[Vec<u8>], row_len: usize) -> Option<Self> {
        let len = streams.first()?.len();
        if row_len == 0 || len == 0 || len % row_len != 0 {
            return None;
        }
        if streams.iter().any(|s| s.len() != len) {
            return None;
        }
        let mut per_column = vec![[0u64; FP4_CODES]; row_len];
        for s in streams {
            accumulate(&mut per_column, s, row_len);
        }
        Some(Self {
            per_column,
            row_len,
        })
    }

    /// The leave-one-out column profile for one expert, expanded to the same
    /// length as its stream so it can be conditioned on coordinate by coordinate.
    ///
    /// Leave-one-out matters more here than for a row-aligned prototype: each
    /// expert contributes `rows` samples to every column, so a bank of 16 leaves
    /// one expert holding 1/16 of the tally it would then be scored against.
    pub fn profile_for(&self, own: &[u8]) -> Vec<u8> {
        let mut own_counts = vec![[0u64; FP4_CODES]; self.row_len];
        accumulate(&mut own_counts, own, self.row_len);

        let modes: Vec<u8> = (0..self.row_len)
            .map(|c| {
                let mut best = 0usize;
                let mut best_n = 0u64;
                for (code, (&total, &own)) in
                    self.per_column[c].iter().zip(&own_counts[c]).enumerate()
                {
                    let n = total.saturating_sub(own);
                    if n > best_n {
                        best_n = n;
                        best = code;
                    }
                }
                best as u8
            })
            .collect();

        (0..own.len()).map(|i| modes[i % self.row_len]).collect()
    }
}

fn accumulate(counts: &mut [[u64; FP4_CODES]], stream: &[u8], row_len: usize) {
    for (i, &sym) in stream.iter().enumerate() {
        if (sym as usize) < FP4_CODES {
            counts[i % row_len][sym as usize] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows are identical within an expert, so each column has one symbol.
    fn column_constant(experts: usize, rows: usize, row_len: usize) -> Vec<Vec<u8>> {
        (0..experts)
            .map(|_| {
                (0..rows * row_len)
                    .map(|i| ((i % row_len) % FP4_CODES) as u8)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_shared_column_signature_is_recovered_exactly() {
        let bank = column_constant(4, 8, 12);
        let p = ColumnProfile::build(&bank, 12).unwrap();
        let profile = p.profile_for(&bank[0]);
        assert_eq!(profile, bank[0], "the profile should reproduce the columns");
    }

    #[test]
    fn the_profile_leaves_its_own_expert_out() {
        // Three experts vote 5 on column 0, one votes 9 everywhere. The odd one
        // must be handed 5; a contaminated tally would still hand it 5 here, so
        // make it decisive: two vote 9, two vote 5.
        let row_len = 1;
        let bank = vec![vec![9u8; 4], vec![9u8; 4], vec![5u8; 4], vec![5u8; 4]];
        let p = ColumnProfile::build(&bank, row_len).unwrap();
        // Removing a 9-voter leaves 9:4 vs 5:8 -> 5.
        assert_eq!(p.profile_for(&bank[0]), vec![5, 5, 5, 5]);
        // Removing a 5-voter leaves 9:8 vs 5:4 -> 9.
        assert_eq!(p.profile_for(&bank[2]), vec![9, 9, 9, 9]);
    }

    #[test]
    fn the_profile_is_invariant_to_permuting_an_experts_rows() {
        // The whole point: shuffling one expert's intermediate units must not
        // change the column profile the OTHERS see, and must not change the
        // profile that expert is handed either.
        let row_len = 6;
        let rows = 5;
        let mut bank: Vec<Vec<u8>> = (0..3)
            .map(|e| {
                (0..rows * row_len)
                    .map(|i| (((i / row_len) + e + (i % row_len)) % FP4_CODES) as u8)
                    .collect()
            })
            .collect();

        let before = ColumnProfile::build(&bank, row_len)
            .unwrap()
            .profile_for(&bank[0]);

        // Reverse expert 2's row order — a legal expert-preserving permutation.
        let permuted: Vec<u8> = bank[2]
            .chunks(row_len)
            .rev()
            .flat_map(|r| r.to_vec())
            .collect();
        bank[2] = permuted;

        let after = ColumnProfile::build(&bank, row_len)
            .unwrap()
            .profile_for(&bank[0]);
        assert_eq!(before, after, "row order must not move a column profile");
    }

    #[test]
    fn ragged_or_misshapen_banks_are_refused() {
        assert!(ColumnProfile::build(&[], 4).is_none());
        assert!(ColumnProfile::build(&[vec![1u8; 8]], 0).is_none());
        assert!(ColumnProfile::build(&[vec![1u8; 8]], 3).is_none(), "8 % 3");
        assert!(ColumnProfile::build(&[vec![1u8; 8], vec![1u8; 4]], 4).is_none());
        assert!(ColumnProfile::build(&[vec![1u8; 8], vec![1u8; 8]], 4).is_some());
    }

    #[test]
    fn out_of_range_symbols_do_not_index_past_the_table() {
        let bank = vec![vec![0u8, 99, 2, 3], vec![0u8, 1, 2, 200]];
        let p = ColumnProfile::build(&bank, 4).unwrap();
        assert_eq!(p.profile_for(&bank[0]).len(), 4);
    }

    #[test]
    fn a_column_no_other_expert_voted_on_falls_back_to_code_zero() {
        // Single-expert bank: leave-one-out empties every tally.
        let bank = vec![vec![7u8; 4]];
        let p = ColumnProfile::build(&bank, 2).unwrap();
        assert_eq!(p.profile_for(&bank[0]), vec![0, 0, 0, 0]);
    }
}
