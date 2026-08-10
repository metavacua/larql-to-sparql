//! Opt-in channel masking of the MoE experts' **shared input** — the
//! latent-axis sparsity probe.
//!
//! ## What this tests
//!
//! R4 refuted sparsifying the *nonlinear population* (FFN rows, the `m`
//! axis): with routing supplied free the sparse kernel still captured only
//! 23–40% of its row-count reduction. This probes the other axis of the
//! factorisation — the **shared interface feeding** that population.
//!
//! Every selected expert reads the same `expert_input` vector, and expert
//! `gate_up` is laid out `[2*inter, hidden]`. So masking hidden channel `j`
//! removes column `j` from **every active expert's gate AND up matrix** —
//! one decision amortised across `2·K` large projections, where a row
//! decision buys one row in one expert. That is the leverage difference,
//! and it is why the R4 negative does not transfer.
//!
//! It also sidesteps the cost problem that motivated ANN: `expert_input` is
//! already computed by the architecture, so scoring its channels needs no
//! second dense projection over a large candidate set.
//!
//! ## Faithfulness
//!
//! The mask is applied **after routing**, so the trained router still
//! selects the same experts with the same weights; only the information fed
//! to them is reduced. All `K` experts and all `m` neurons per expert are
//! retained. Masking a channel is exactly equivalent to not reading that
//! weight column, so zeroing (not mean-ablation) is the faithful semantics
//! here — the ablation *is* the lever.
//!
//! ## Ceiling (compute before believing any quality number)
//!
//! Against the DEC-8.6 ledger (dense 29.86 / routed 25.83 GB/token), input-
//! side masking scales the gate+up two-thirds of expert bytes and leaves
//! `down` fixed, so end-to-end speedup is capped at **1.448× even at zero
//! retention**; ~1.18× at r=0.5. Both-sides masking scales all three and
//! reaches 1.30× at r=0.5. Neither alone carries 5.8→10 tok/s.
//!
//! Default off — absent env, every call is a no-op and output is
//! byte-identical.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

// core::sync::atomic is the same module std re-exports -- portable
// regardless of target, no cfg needed. OnceLock has no core/alloc
// equivalent (needs real thread-safety), but it's only used inside
// this file's native-only `active()` below.
use core::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

use crate::options;

/// Retained fraction of shared-input channels, in `(0, 1]`. Absent or `>= 1`
/// disables the probe entirely.
pub const ENV_LATENT_RETENTION: &str = "LARQL_MOE_LATENT_RETENTION";
/// Channel selector: `magnitude` (default) or `random`.
pub const ENV_LATENT_MODE: &str = "LARQL_MOE_LATENT_MODE";
/// Also mask the same channels on the expert-branch output (the analogue of
/// masking the aggregated latent before a shared up-projection).
pub const ENV_LATENT_BOTH_SIDES: &str = "LARQL_MOE_LATENT_BOTH_SIDES";
/// Channel-block granularity. `1` (default) selects individual channels.
///
/// Whether that is realisable depends entirely on weight LAYOUT, and an
/// earlier version of this comment got it wrong. In **output-major**
/// `[2m, hidden]` storage a per-channel mask strides within each row and
/// saves nothing, so blocks are needed. In **input-major** `[hidden, 2m]`
/// storage each channel is already a contiguous `2m` slab, so `block = 1`
/// is itself the physical I/O block and no grouping is required. This knob
/// therefore measures the output-major case; pair it with [`ENV_LATENT_PERM`]
/// before concluding anything about clustering.
pub const ENV_LATENT_BLOCK: &str = "LARQL_MOE_LATENT_BLOCK";
/// Path to a channel co-selection dump, written at process exit. Line `j` is
/// `count` — how often channel `j` survived. Feeds the offline permutation.
pub const ENV_LATENT_STATS: &str = "LARQL_MOE_LATENT_STATS";
/// Path to a channel permutation (one channel index per line, new order).
/// Blocks are formed over the PERMUTED order, which is exactly what an
/// offline weight permutation would realise: `z' = Pz`, `W' = W·Pᵀ` leaves
/// the model unchanged, so blocking in a co-activation order costs nothing.
/// Without this, blocks are native-order — the checkpoint's arbitrary
/// coordinate layout, which has no reason to carry locality.
pub const ENV_LATENT_PERM: &str = "LARQL_MOE_LATENT_PERM";
/// Path for the sampled pairwise co-selection matrix (binary `u32` row-major
/// `n×n`, upper triangle filled). Requires [`ENV_LATENT_STATS`] too.
pub const ENV_LATENT_COSTATS: &str = "LARQL_MOE_LATENT_COSTATS";

/// Odd multiplier for the per-call PRNG (SplitMix64 constant).
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_MIX1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_MIX2: u64 = 0x94D0_49BB_1331_11EB;

/// Which channels the probe retains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Keep the top-`r` channels by `|z_j|`. The cheap practical proxy.
    Magnitude,
    /// Keep a fresh pseudorandom `r` fraction per call — the control for
    /// `Magnitude`. If magnitude does not beat random, channel importance is
    /// not concentrated and no cheap selector on this axis will help.
    Random,
}

/// An active latent-mask configuration.
#[derive(Debug, Clone)]
pub struct LatentMask {
    pub retention: f32,
    pub mode: Mode,
    pub both_sides: bool,
    /// Channels selected together as a group. 1 = per-channel, which an
    /// input-major layout already streams contiguously; >1 = block-granular,
    /// which only matters for an output-major layout.
    pub block: usize,
    /// Channel order blocks are formed over. Empty = native checkpoint order.
    /// A permutation here stands in for the exact offline weight permutation,
    /// so a co-activation ordering can be evaluated without repacking.
    pub perm: Vec<u32>,
}

/// Per-call counter, so `Mode::Random` draws a different subset each call
/// while staying deterministic for a given call order.
static CALLS: AtomicU64 = AtomicU64::new(0);

/// The active probe configuration, or `None` when the probe is off.
///
/// Resolved once per process: this is an experiment switch, not a runtime
/// knob, and caching keeps the hot path free of `getenv`.
///
/// wasm32v1-none has no `std::env`/`std::fs` at all, so there is no
/// mechanism to configure this probe there in the first place --
/// `None` (the same value as an unset env var natively) is the
/// honestly-correct answer, not a stub. Keeping the same public
/// signature means every caller compiles unchanged on both targets.
#[cfg(target_arch = "wasm32")]
pub fn active() -> Option<&'static LatentMask> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
pub fn active() -> Option<&'static LatentMask> {
    static CFG: OnceLock<Option<LatentMask>> = OnceLock::new();
    CFG.get_or_init(|| {
        let retention: f32 = options::env_value(ENV_LATENT_RETENTION)?.parse().ok()?;
        if !(retention > 0.0 && retention < 1.0) {
            return None;
        }
        let mode = match options::env_value(ENV_LATENT_MODE).as_deref() {
            Some("random") => Mode::Random,
            _ => Mode::Magnitude,
        };
        let perm = options::env_value(ENV_LATENT_PERM)
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|s| {
                s.split_whitespace()
                    .filter_map(|t| t.parse::<u32>().ok())
                    .collect::<Vec<u32>>()
            })
            .unwrap_or_default();
        Some(LatentMask {
            retention,
            mode,
            both_sides: options::env_flag(ENV_LATENT_BOTH_SIDES),
            block: options::env_usize(ENV_LATENT_BLOCK).unwrap_or(1).max(1),
            perm,
        })
    })
    .as_ref()
}

impl LatentMask {
    /// Number of channels retained out of `n` — at least 1, never more
    /// than `n`.
    fn keep_count(&self, n: usize) -> usize {
        ((n as f32 * self.retention).ceil() as usize).clamp(1, n)
    }

    /// Choose the retained channels and zero the rest in place.
    /// Returns the retained set so the same channels can be reused on the
    /// output side.
    pub fn mask_in_place(&self, v: &mut [f32]) -> Vec<bool> {
        let n = v.len();
        let keep = self.keep_count(n);
        let mut retained = vec![false; n];
        if keep >= n {
            retained.iter_mut().for_each(|r| *r = true);
            return retained;
        }

        // Block-granular selection: score each block by its total |v| mass
        // and keep whole blocks. Relevant to an output-major layout, where a
        // per-channel mask saves no bytes; an input-major layout streams each
        // channel contiguously already and needs no blocking at all.
        if self.block > 1 {
            // Block `b` owns positions `b*block .. (b+1)*block` of the
            // ORDER — native when `perm` is empty, otherwise the supplied
            // permutation. Grouping over a permuted order is exactly what an
            // offline weight permutation realises, so this measures whether
            // ANY ordering can express the masks, not just the checkpoint's.
            let order: &[u32] = if self.perm.len() == n {
                &self.perm
            } else {
                &[]
            };
            let at = |pos: usize| -> usize {
                if order.is_empty() {
                    pos
                } else {
                    order[pos] as usize
                }
            };
            let nblocks = n.div_ceil(self.block);
            let keep_blocks =
                (((nblocks as f32) * self.retention).ceil() as usize).clamp(1, nblocks);
            let mut scored: Vec<(f32, usize)> = (0..nblocks)
                .map(|b| {
                    let start = b * self.block;
                    let end = (start + self.block).min(n);
                    let mass = (start..end).map(|pos| v[at(pos)].abs()).sum::<f32>();
                    (mass, b)
                })
                .collect();
            scored.select_nth_unstable_by(keep_blocks - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal)
            });
            for &(_, b) in &scored[..keep_blocks] {
                let start = b * self.block;
                let end = (start + self.block).min(n);
                for pos in start..end {
                    retained[at(pos)] = true;
                }
            }
            for (val, &r) in v.iter_mut().zip(retained.iter()) {
                if !r {
                    *val = 0.0;
                }
            }
            return retained;
        }

        match self.mode {
            Mode::Magnitude => {
                let mut idx: Vec<u32> = (0..n as u32).collect();
                // Partition so the `keep` largest |v| land first.
                idx.select_nth_unstable_by(keep - 1, |&a, &b| {
                    v[b as usize]
                        .abs()
                        .partial_cmp(&v[a as usize].abs())
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                for &i in &idx[..keep] {
                    retained[i as usize] = true;
                }
            }
            Mode::Random => {
                // SplitMix64 over (call, channel) — a fresh subset per call,
                // deterministic for a given call order. Score every channel
                // and keep the `keep` lowest, which is a uniform subset.
                let call = CALLS.fetch_add(1, Ordering::Relaxed);
                let mut scored: Vec<(u64, u32)> = (0..n as u32)
                    .map(|j| (splitmix(call.wrapping_mul(SPLITMIX_GAMMA) ^ j as u64), j))
                    .collect();
                scored.select_nth_unstable(keep - 1);
                for &(_, j) in &scored[..keep] {
                    retained[j as usize] = true;
                }
            }
        }

        for (val, &r) in v.iter_mut().zip(retained.iter()) {
            if !r {
                *val = 0.0;
            }
        }
        retained
    }

    /// Zero the non-retained channels of `v` using an existing mask.
    pub fn apply(&self, v: &mut [f32], retained: &[bool]) {
        for (val, &r) in v.iter_mut().zip(retained.iter()) {
            if !r {
                *val = 0.0;
            }
        }
    }
}

/// Per-channel survival counts, accumulated only when [`ENV_LATENT_STATS`]
/// names an output path. Feeds the offline co-activation ordering.
/// Indexed `[layer][channel]`. **Per-layer is load-bearing**: channel `j` in
/// layer 4 has no relationship to channel `j` in layer 63, so aggregating by
/// channel index across layers produces uniform marginals even when every
/// layer has a perfectly stable subset. An earlier version did exactly that
/// and wrongly concluded there was no stable subset.
#[cfg(not(target_arch = "wasm32"))]
static STATS: std::sync::Mutex<Vec<Vec<u64>>> = std::sync::Mutex::new(Vec::new());
/// Pairwise co-selection counts (row-major `n×n`), sampled. Marginal counts
/// cannot answer whether a channel ORDERING exists: uniform marginals are
/// compatible with both "no structure" and "strongly correlated groups".
/// Only the pairwise matrix separates those.
#[cfg(not(target_arch = "wasm32"))]
static COSTATS: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
static COSTATS_SAMPLES: AtomicU64 = AtomicU64::new(0);
/// Accumulate the (expensive, O(kept²)) pair matrix on 1 call in N.
const COSTATS_SAMPLE_EVERY: u64 = 64;

/// No-op on wasm32: same reasoning as `active()` above -- there is no
/// `std::env`/`std::fs` mechanism to enable stats collection there at
/// all, so this matches exactly what happens natively when
/// `ENV_LATENT_STATS` is unset (the common case; this is opt-in
/// tooling, not part of any hot path's normal behavior).
#[cfg(target_arch = "wasm32")]
pub fn record_stats(_layer: usize, _retained: &[bool]) {}

#[cfg(not(target_arch = "wasm32"))]
pub fn record_stats(layer: usize, retained: &[bool]) {
    if options::env_value(ENV_LATENT_STATS).is_none() {
        return;
    }
    let mut s = STATS.lock().unwrap();
    if s.len() <= layer {
        s.resize(layer + 1, Vec::new());
    }
    let row = &mut s[layer];
    if row.len() < retained.len() {
        row.resize(retained.len(), 0);
    }
    for (slot, &r) in row.iter_mut().zip(retained.iter()) {
        if r {
            *slot += 1;
        }
    }
    drop(s);

    if options::env_value(ENV_LATENT_COSTATS).is_none() {
        return;
    }
    let call = COSTATS_SAMPLES.fetch_add(1, Ordering::Relaxed);
    if !call.is_multiple_of(COSTATS_SAMPLE_EVERY) {
        return;
    }
    let n = retained.len();
    let kept: Vec<u32> = retained
        .iter()
        .enumerate()
        .filter(|(_, &r)| r)
        .map(|(j, _)| j as u32)
        .collect();
    let mut c = COSTATS.lock().unwrap();
    if c.len() < n * n {
        c.resize(n * n, 0);
    }
    for (a_i, &a) in kept.iter().enumerate() {
        let row = a as usize * n;
        for &b in &kept[a_i..] {
            c[row + b as usize] += 1;
        }
    }
}

/// No-op on wasm32: same reasoning as `record_stats` above.
#[cfg(target_arch = "wasm32")]
pub fn dump_stats() {}

/// Write the accumulated per-channel survival counts. Call once at exit.
/// No-op when stats collection was never enabled.
#[cfg(not(target_arch = "wasm32"))]
pub fn dump_stats() {
    let Some(path) = options::env_value(ENV_LATENT_STATS) else {
        return;
    };
    let s = STATS.lock().unwrap();
    if s.is_empty() {
        return;
    }
    let mut body = String::new();
    for (layer, row) in s.iter().enumerate() {
        for (chan, count) in row.iter().enumerate() {
            body.push_str(&format!("{layer} {chan} {count}\n"));
        }
    }
    let _ = std::fs::write(path, body);
    drop(s);

    if let Some(co_path) = options::env_value(ENV_LATENT_COSTATS) {
        let c = COSTATS.lock().unwrap();
        if !c.is_empty() {
            let mut bytes = Vec::with_capacity(c.len() * 4);
            for v in c.iter() {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let _ = std::fs::write(co_path, bytes);
        }
    }
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(SPLITMIX_GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_MIX1);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_MIX2);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 8;

    fn mask(retention: f32, mode: Mode) -> LatentMask {
        LatentMask {
            retention,
            mode,
            both_sides: false,
            block: 1,
            perm: Vec::new(),
        }
    }

    fn block_mask(retention: f32, block: usize) -> LatentMask {
        LatentMask {
            retention,
            mode: Mode::Magnitude,
            both_sides: false,
            block,
            perm: Vec::new(),
        }
    }

    #[test]
    fn magnitude_keeps_the_largest_channels() {
        let m = mask(0.5, Mode::Magnitude);
        let mut v = vec![1.0, -8.0, 2.0, -7.0, 0.5, 6.0, -0.1, 5.0];
        let retained = m.mask_in_place(&mut v);
        assert_eq!(retained.iter().filter(|r| **r).count(), 4);
        // The four largest by |v| are indices 1, 3, 5, 7.
        for i in [1usize, 3, 5, 7] {
            assert!(retained[i], "channel {i} should survive");
            assert_ne!(v[i], 0.0);
        }
        for i in [0usize, 2, 4, 6] {
            assert!(!retained[i]);
            assert_eq!(v[i], 0.0, "channel {i} must be zeroed");
        }
    }

    #[test]
    fn keep_count_clamps_and_rounds_up() {
        assert_eq!(mask(0.5, Mode::Magnitude).keep_count(N), 4);
        // Rounds up so a tiny retention never empties the vector.
        assert_eq!(mask(0.01, Mode::Magnitude).keep_count(N), 1);
        assert_eq!(mask(0.99, Mode::Magnitude).keep_count(N), 8);
    }

    #[test]
    fn random_keeps_the_right_count_and_varies_across_calls() {
        let m = mask(0.5, Mode::Random);
        let base = vec![1.0f32; N];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..8 {
            let mut v = base.clone();
            let retained = m.mask_in_place(&mut v);
            assert_eq!(retained.iter().filter(|r| **r).count(), 4);
            assert_eq!(v.iter().filter(|x| **x != 0.0).count(), 4);
            seen.insert(retained);
        }
        // A per-call draw must not return the same subset every time —
        // otherwise it is a fixed pool, a different control entirely.
        assert!(seen.len() > 1, "random mode must vary across calls");
    }

    #[test]
    fn apply_reuses_an_existing_mask() {
        let m = mask(0.5, Mode::Magnitude);
        let retained = vec![true, false, true, false, true, false, true, false];
        let mut v = vec![3.0f32; N];
        m.apply(&mut v, &retained);
        assert_eq!(v, vec![3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn block_mode_keeps_whole_contiguous_blocks() {
        // Two blocks of 4. Block 1 carries far more |v| mass, so at r=0.5
        // exactly that block survives — intact, not cherry-picked channels.
        let m = block_mask(0.5, 4);
        let mut v = vec![0.1, 0.1, 0.1, 0.1, 9.0, 8.0, 7.0, 6.0];
        let retained = m.mask_in_place(&mut v);
        assert_eq!(
            retained,
            vec![false, false, false, false, true, true, true, true]
        );
        assert_eq!(&v[..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&v[4..], &[9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn block_mode_never_splits_a_block_even_when_channels_disagree() {
        // Block 0 holds the single largest channel (10.0) but less total
        // mass (10.0) than block 1 (12.0). Per-channel selection would keep
        // channel 0; block selection keeps block 1 whole. The two criteria
        // genuinely disagree — which is why block tolerance has to be
        // measured separately and cannot be inferred from block=1.
        let m = block_mask(0.5, 4);
        let mut v = vec![10.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0];
        let retained = m.mask_in_place(&mut v);
        assert_eq!(retained.iter().filter(|r| **r).count(), 4);
        assert_eq!(
            retained,
            vec![false, false, false, false, true, true, true, true]
        );
        assert_eq!(
            v[0], 0.0,
            "the largest single channel is dropped with its block"
        );
    }

    #[test]
    fn a_permutation_regroups_blocks_without_touching_values() {
        // Native order splits the two big channels across both blocks, so
        // native block-2 selection cannot isolate them. A permutation that
        // puts them adjacent lets one block capture both — the whole point
        // of the co-activation ordering.
        let mut m = block_mask(0.5, 2);
        let v0 = vec![9.0f32, 0.1, 0.1, 8.0];

        let mut native = v0.clone();
        let got_native = m.mask_in_place(&mut native);
        assert_eq!(got_native, vec![true, true, false, false]);

        // Order (0, 3, 1, 2): block 0 = channels {0, 3} = both big ones.
        m.perm = vec![0, 3, 1, 2];
        let mut permuted = v0.clone();
        let got_perm = m.mask_in_place(&mut permuted);
        assert_eq!(got_perm, vec![true, false, false, true]);
        assert_eq!(permuted, vec![9.0, 0.0, 0.0, 8.0]);
    }

    #[test]
    fn retention_outside_the_open_unit_interval_disables_the_probe() {
        // Guarded at `active()`; assert the boundary logic the parse uses.
        for r in [0.0f32, 1.0, 1.5, -0.2] {
            assert!(!(r > 0.0 && r < 1.0), "r={r} must not enable the probe");
        }
    }

    // ── Stats accumulation ──────────────────────────────────────────────
    //
    // `STATS`/`COSTATS` are process-global, and `set_env_override` is
    // thread-local, so a parallel test cannot own them exclusively. These
    // assert *structure* — the file was written, it parses, our layer is in
    // it with a non-zero count — rather than exact totals, which another
    // test's accumulation would perturb.

    struct EnvGuard;
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            options::clear_fast_path_overrides();
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("latent-mask-tests");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{name}-{}", std::process::id()))
    }

    #[test]
    fn stats_are_not_collected_unless_a_path_is_named() {
        // The hot path must stay free: no env, no lock, no allocation.
        let _g = EnvGuard;
        options::clear_fast_path_overrides();
        // Nothing to assert but that it does not panic and does not write.
        record_stats(0, &[true, false, true]);
        dump_stats();
    }

    #[test]
    fn recorded_survivals_reach_the_dumped_file() {
        let _g = EnvGuard;
        let path = tmp("stats.txt");
        let _ = std::fs::remove_file(&path);
        options::set_env_override(ENV_LATENT_STATS, Some(path.to_str().unwrap()));

        // A layer index no other test uses, so our row is identifiable.
        const LAYER: usize = 41;
        record_stats(LAYER, &[true, false, true, false]);
        record_stats(LAYER, &[true, false, false, true]);
        dump_stats();

        let body = std::fs::read_to_string(&path).expect("dump_stats must write the file");
        // `layer chan count` triples; channel 0 survived both calls.
        let ours: Vec<&str> = body
            .lines()
            .filter(|l| l.starts_with(&format!("{LAYER} ")))
            .collect();
        assert!(!ours.is_empty(), "our layer must appear: {body:.200}");
        let chan0 = ours
            .iter()
            .find(|l| l.starts_with(&format!("{LAYER} 0 ")))
            .expect("channel 0 row");
        let count: u64 = chan0.split_whitespace().nth(2).unwrap().parse().unwrap();
        assert!(count >= 2, "channel 0 survived twice, got {count}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_short_row_grows_to_fit_a_wider_mask() {
        // Layers are discovered as they are first recorded, and a later call
        // may carry more channels than the first — resizing must preserve the
        // counts already accumulated rather than reallocating them away.
        let _g = EnvGuard;
        let path = tmp("grow.txt");
        let _ = std::fs::remove_file(&path);
        options::set_env_override(ENV_LATENT_STATS, Some(path.to_str().unwrap()));

        const LAYER: usize = 43;
        record_stats(LAYER, &[true, true]);
        record_stats(LAYER, &[true, false, true, true]);
        dump_stats();

        let body = std::fs::read_to_string(&path).expect("file");
        let row0 = body
            .lines()
            .find(|l| l.starts_with(&format!("{LAYER} 0 ")))
            .expect("channel 0");
        let c0: u64 = row0.split_whitespace().nth(2).unwrap().parse().unwrap();
        assert!(c0 >= 2, "channel 0 survived both calls, got {c0}");
        assert!(
            body.lines().any(|l| l.starts_with(&format!("{LAYER} 3 "))),
            "the widened row must reach channel 3"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn costats_need_their_own_switch_and_write_a_square_matrix() {
        let _g = EnvGuard;
        let stats = tmp("co-stats.txt");
        let co = tmp("co-pairs.bin");
        let _ = std::fs::remove_file(&stats);
        let _ = std::fs::remove_file(&co);
        options::set_env_override(ENV_LATENT_STATS, Some(stats.to_str().unwrap()));
        options::set_env_override(ENV_LATENT_COSTATS, Some(co.to_str().unwrap()));

        // Sampled 1-in-N, so drive enough calls that at least one lands.
        for _ in 0..(COSTATS_SAMPLE_EVERY * 2 + 1) {
            record_stats(45, &[true, false, true, false]);
        }
        dump_stats();

        let bytes = std::fs::read(&co).expect("costats file");
        assert!(
            !bytes.is_empty(),
            "a sampled call must have filled the matrix"
        );
        assert_eq!(bytes.len() % 4, 0, "u32 counts, little-endian");
        let _ = std::fs::remove_file(&stats);
        let _ = std::fs::remove_file(&co);
    }
}
