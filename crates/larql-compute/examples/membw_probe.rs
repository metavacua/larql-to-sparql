//! membw_probe — STREAM-class attainable memory bandwidth for the CPU cluster.
//!
//! **Why this exists.** Every bandwidth plan in this repo is currently made
//! against two numbers that are not the same kind of number:
//!
//!   - "larql extracts ~47 GB/s effective vs llama.cpp ~70" (C10/C12, 2026-06-12)
//!     — an *achieved* figure, and one measured **before** the spin-barrier pool
//!     landed +28%.
//!   - "the 400 GB/s GPU bandwidth ceiling" (larql-compute ROADMAP/CHANGELOG)
//!     — the SoC **spec sheet**, which the CPU cluster cannot reach on Apple
//!     silicon: the P-cluster's fabric ports cap it well below the GPU's.
//!
//! Achieved-vs-spec is not a roofline. This probe measures the third number —
//! what the CPU cluster can *actually* attain on this box — so "we are at X% of
//! possible" becomes answerable, and so the question "is there CPU bandwidth
//! headroom left, or does the bandwidth-bound half belong on the GPU?" has a
//! measurement behind it instead of a spec sheet.
//!
//! **The trap this probe is built to avoid.** C12's roofline finding (scalar 9.3
//! vs NEON 17.7 GiB/s on *identical data*, size-invariant) proved that a naive
//! streaming loop measures **issue rate, not bandwidth**. If both sizes report
//! the same GB/s you have measured your own loop. So every arm here is
//! vector-wide with 8 independent accumulators, and the probe **self-checks**:
//! it runs each arm at an L2-resident size as well as a DRAM-resident size and
//! reports the ratio. A cache-resident arm that fails to substantially outrun
//! the DRAM arm means the loop is issue-bound and the DRAM number is a **floor,
//! not a ceiling** — the probe says so in its own output rather than letting the
//! number get quoted as attainable.
//!
//! Arms:
//!   - `read`  — pure load stream. The decode-relevant one: weight streaming is
//!     read-only, and it is what the effective-GB/s figure should be compared to.
//!   - `copy`  — read + write (classic STREAM copy), for the activation traffic.
//!
//! Usage:
//!   cargo run --release -p larql-compute --example membw_probe
//!   cargo run --release -p larql-compute --example membw_probe -- --json out.json
//!
//! Run it on AC power on a quiet machine — see `feedback_thermal_perf_artifacts`;
//! a hot or contended box invents 1.5–3× phantom regressions. The probe prints
//! a spread across repeats so a contended run is visible rather than silent.

use std::sync::Barrier;
use std::time::Instant;

/// DRAM-resident working set per thread. Must be far larger than the 16 MB SLC
/// so that every load is a fabric transaction and no reuse is possible.
const DRAM_BYTES_PER_THREAD: usize = 256 << 20; // 256 MiB

/// Cache-resident working set per thread, sized to sit inside the P-core L2
/// (4 MB on M3 Max) with room to spare. This arm exists only to validate the
/// probe — see the module docs.
const CACHE_BYTES_PER_THREAD: usize = 1 << 20; // 1 MiB

/// Repeats per (arm, thread-count) cell. Reported as best + median: best is the
/// attainable estimate, the best/median spread is the contention tell.
const REPEATS: usize = 5;

/// Bytes each thread streams per timed region, independent of working-set size.
///
/// The cache-resident arm is 256× smaller than the DRAM arm, so a single pass
/// over it finishes in ~10 µs — far below the cost of spawning the thread that
/// runs it. Measured that way the "cache" arm reports *thread spawn*, comes out
/// **slower** than DRAM, and the self-check then fires against a probe artifact
/// instead of against a real issue-rate limit. Holding streamed bytes constant
/// makes both arms run for the same ~duration and amortise spawn identically.
const STREAMED_BYTES_PER_THREAD: usize = 512 << 20; // 512 MiB

/// Bytes consumed per unrolled iteration of the kernels below: 8 accumulators ×
/// one 16-byte vector each.
const BYTES_PER_ITER: usize = 8 * 16;

// ── kernels ─────────────────────────────────────────────────────────────────

/// Sum `buf` with 8 independent vector accumulators.
///
/// Eight accumulators is the point: a single accumulator serialises on the FP
/// add latency and turns the loop into a dependency chain, which is exactly the
/// issue-bound measurement C12 caught. Independent accumulators let the core
/// keep enough loads in flight that the limiter is the memory system.
#[cfg(target_arch = "aarch64")]
fn read_stream(buf: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32};
    debug_assert!((buf.len() * 4).is_multiple_of(BYTES_PER_ITER));
    unsafe {
        let mut a = [vdupq_n_f32(0.0); 8];
        let mut p = buf.as_ptr();
        let end = p.add(buf.len());
        while p < end {
            for (k, acc) in a.iter_mut().enumerate() {
                *acc = vaddq_f32(*acc, vld1q_f32(p.add(k * 4)));
            }
            p = p.add(32);
        }
        let mut s = vdupq_n_f32(0.0);
        for acc in a {
            s = vaddq_f32(s, acc);
        }
        vaddvq_f32(s)
    }
}

/// Portable fallback. Same 8-accumulator shape; autovectorisation dependent, so
/// the self-check matters even more on this path.
#[cfg(not(target_arch = "aarch64"))]
fn read_stream(buf: &[f32]) -> f32 {
    // Same invariant as the NEON path — and load-bearing here, because
    // `chunks_exact` silently drops a remainder, which would undercount the
    // bytes this arm claims to have streamed.
    debug_assert!((buf.len() * 4).is_multiple_of(BYTES_PER_ITER));
    let mut a = [0.0f32; 8];
    for chunk in buf.chunks_exact(8) {
        for (acc, v) in a.iter_mut().zip(chunk) {
            *acc += *v;
        }
    }
    a.iter().sum()
}

/// Copy `src` into `dst` with vector loads/stores. Counted as read+write bytes.
#[cfg(target_arch = "aarch64")]
fn copy_stream(src: &[f32], dst: &mut [f32]) {
    use std::arch::aarch64::{vld1q_f32, vst1q_f32};
    debug_assert_eq!(src.len(), dst.len());
    unsafe {
        let mut p = src.as_ptr();
        let mut q = dst.as_mut_ptr();
        let end = p.add(src.len());
        while p < end {
            for k in 0..8 {
                vst1q_f32(q.add(k * 4), vld1q_f32(p.add(k * 4)));
            }
            p = p.add(32);
            q = q.add(32);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn copy_stream(src: &[f32], dst: &mut [f32]) {
    dst.copy_from_slice(src);
}

// ── harness ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Read,
    Copy,
}

impl Arm {
    fn label(self) -> &'static str {
        match self {
            Arm::Read => "read",
            Arm::Copy => "copy",
        }
    }

    /// Bytes of DRAM traffic per byte of working set touched. `copy` reads the
    /// source and writes the destination, so it moves twice its footprint.
    ///
    /// Caveat kept honest: this counts the *write* only once. On a
    /// write-allocate cache the store also pulls the destination line in, so
    /// true `copy` traffic can be 3× footprint. We report the 2× convention
    /// (what STREAM does) and lean on `read` for the roofline number.
    fn traffic_multiplier(self) -> f64 {
        match self {
            Arm::Read => 1.0,
            Arm::Copy => 2.0,
        }
    }
}

/// One measured cell: arm × thread count × working-set class.
struct Cell {
    arm: Arm,
    threads: usize,
    resident: &'static str,
    best_gbs: f64,
    median_gbs: f64,
}

/// Run one (arm, threads, bytes-per-thread) cell.
///
/// Each thread owns a private, contiguous buffer that it first-touches before
/// the timed region, so page-fault cost and any NUMA-ish placement effect land
/// in setup rather than in the measurement. A `Barrier` releases all threads
/// together — without it, short-running threads finish before late ones start
/// and the aggregate reads high.
fn run_cell(arm: Arm, threads: usize, bytes_per_thread: usize, resident: &'static str) -> Cell {
    let elems = bytes_per_thread / 4;
    let passes = (STREAMED_BYTES_PER_THREAD / bytes_per_thread).max(1);
    let mut samples: Vec<f64> = Vec::with_capacity(REPEATS);

    // Allocate + first-touch outside the timed region.
    let mut srcs: Vec<Vec<f32>> = (0..threads)
        .map(|t| (0..elems).map(|i| ((i + t) % 251) as f32 * 0.001).collect())
        .collect();
    let mut dsts: Vec<Vec<f32>> = if arm == Arm::Copy {
        (0..threads).map(|_| vec![0.0f32; elems]).collect()
    } else {
        Vec::new()
    };

    for _ in 0..REPEATS {
        let barrier = Barrier::new(threads + 1);
        // Pair each source with its destination up front. Indexing `dsts[t]`
        // inside the spawn loop would hold one mutable borrow of `dsts` per
        // thread; draining a single iterator hands out disjoint `&mut`s instead.
        let mut dst_iter = dsts.iter_mut();
        let pairs: Vec<_> = srcs
            .iter_mut()
            .map(|src| {
                let dst = if arm == Arm::Copy {
                    dst_iter.next()
                } else {
                    None
                };
                (src, dst)
            })
            .collect();

        let elapsed = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads);
            for (src, dst) in pairs {
                let barrier = &barrier;
                handles.push(scope.spawn(move || {
                    barrier.wait();
                    match dst {
                        Some(d) => {
                            for _ in 0..passes {
                                copy_stream(src, d);
                                std::hint::black_box(d.as_ptr());
                            }
                        }
                        None => {
                            for _ in 0..passes {
                                std::hint::black_box(read_stream(src));
                            }
                        }
                    }
                }));
            }
            barrier.wait();
            let start = Instant::now();
            for h in handles {
                h.join().expect("worker panicked");
            }
            start.elapsed()
        });

        let bytes = (bytes_per_thread * threads * passes) as f64 * arm.traffic_multiplier();
        samples.push(bytes / elapsed.as_secs_f64() / 1e9);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    Cell {
        arm,
        threads,
        resident,
        best_gbs: *samples.last().expect("REPEATS > 0"),
        median_gbs: samples[samples.len() / 2],
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json_path = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let sweep: Vec<usize> = [1, 2, 4, 6, 8, 10, 12, max_threads]
        .into_iter()
        .filter(|t| *t <= max_threads)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    println!("membw_probe — attainable CPU memory bandwidth");
    println!(
        "  cores={max_threads}  dram_set={} MiB/thread  cache_set={} MiB/thread  repeats={REPEATS}",
        DRAM_BYTES_PER_THREAD >> 20,
        CACHE_BYTES_PER_THREAD >> 20,
    );
    println!();

    let mut cells: Vec<Cell> = Vec::new();
    for arm in [Arm::Read, Arm::Copy] {
        for &t in &sweep {
            cells.push(run_cell(arm, t, DRAM_BYTES_PER_THREAD, "dram"));
        }
        // Self-check arm: single- and full-width only; we need the ratio, not a curve.
        for &t in [1usize, max_threads].iter() {
            cells.push(run_cell(arm, t, CACHE_BYTES_PER_THREAD, "cache"));
        }
    }

    println!(
        "{:<6} {:<7} {:>8} {:>12} {:>12}  spread",
        "arm", "set", "threads", "best GB/s", "median GB/s"
    );
    for c in &cells {
        let spread = if c.median_gbs > 0.0 {
            (c.best_gbs / c.median_gbs - 1.0) * 100.0
        } else {
            0.0
        };
        let flag = if spread > 8.0 { "  <-- noisy" } else { "" };
        println!(
            "{:<6} {:<7} {:>8} {:>12.1} {:>12.1}  {:>+5.1}%{}",
            c.arm.label(),
            c.resident,
            c.threads,
            c.best_gbs,
            c.median_gbs,
            spread,
            flag
        );
    }
    println!();

    // ── verdict ─────────────────────────────────────────────────────────────
    for arm in [Arm::Read, Arm::Copy] {
        let peak_dram = cells
            .iter()
            .filter(|c| c.arm == arm && c.resident == "dram")
            .map(|c| c.best_gbs)
            .fold(0.0f64, f64::max);
        let peak_cache = cells
            .iter()
            .filter(|c| c.arm == arm && c.resident == "cache")
            .map(|c| c.best_gbs)
            .fold(0.0f64, f64::max);
        let at = cells
            .iter()
            .filter(|c| c.arm == arm && c.resident == "dram")
            .max_by(|a, b| {
                a.best_gbs
                    .partial_cmp(&b.best_gbs)
                    .expect("no NaN bandwidths")
            })
            .map(|c| c.threads)
            .unwrap_or(0);

        println!(
            "{}: peak DRAM {peak_dram:.1} GB/s at {at} threads; cache-resident {peak_cache:.1} GB/s",
            arm.label()
        );
        // The C12 guard. A probe that cannot go faster out of L2 than out of DRAM
        // is measuring itself, and its DRAM number is a lower bound on attainable.
        let ratio = if peak_dram > 0.0 {
            peak_cache / peak_dram
        } else {
            0.0
        };
        if ratio < 1.5 {
            println!(
                "  !! NOT CONFIRMED MEMORY-LIMITED (cache/dram = {ratio:.2}×, want >1.5×): \
                 treat {peak_dram:.1} GB/s as a FLOOR on attainable, not a ceiling."
            );
            if arm == Arm::Copy {
                // Expected for `copy`, and not a defect in the loop: shrinking the
                // working set does not make this arm cache-resident, because the
                // stores still have to reach memory. The control cannot isolate
                // issue rate here, so `copy` is reported as a floor by design and
                // `read` carries the roofline. Do not "fix" this by loosening the
                // threshold.
                println!(
                    "     (expected for copy — the store stream reaches memory in both arms, \
                     so this control cannot isolate issue rate; use the `read` roofline.)"
                );
            }
        } else {
            println!(
                "  self-check ok (cache/dram = {ratio:.2}×) — the DRAM arm is memory-limited."
            );
        }
    }

    if let Some(path) = json_path {
        let rows: Vec<String> = cells
            .iter()
            .map(|c| {
                format!(
                    r#"{{"arm":"{}","set":"{}","threads":{},"best_gbs":{:.3},"median_gbs":{:.3}}}"#,
                    c.arm.label(),
                    c.resident,
                    c.threads,
                    c.best_gbs,
                    c.median_gbs
                )
            })
            .collect();
        let json = format!(
            r#"{{"probe":"membw","cores":{max_threads},"dram_bytes_per_thread":{},"cache_bytes_per_thread":{},"repeats":{REPEATS},"cells":[{}]}}"#,
            DRAM_BYTES_PER_THREAD,
            CACHE_BYTES_PER_THREAD,
            rows.join(",")
        );
        std::fs::write(&path, json).expect("write json");
        println!("\nwrote {path}");
    }
}
