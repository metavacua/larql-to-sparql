//! Tests for [`super`].
//!
//! Split out of `spin_pool.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use super::*;
use std::sync::atomic::AtomicU32;

// ── Wrapper scheduling parity ───────────────────────────────────────
//
// `par_chunks_mut`/`par_chunks_mut2` each branch on `enabled()`, so a CI
// run with one ambient setting leaves the other arm dead — and CI sets
// `LARQL_SPIN_POOL=0` (see .github/workflows/larql-compute.yml) because
// a spin barrier has no free core to spin on there. These drive both
// arms explicitly, so coverage and the equivalence claim both hold
// regardless of the ambient flag.
//
// The override is thread-local rather than `std::env::set_var`, which
// segfaults under parallel `getenv` (see `options::ENV_OVERRIDES`).

/// Run `f` with the spin pool forced on, then forced off, returning
/// both results for comparison.
fn both_arms<T, F: Fn() -> T>(f: F) -> (T, T) {
    use crate::options::{clear_fast_path_overrides, set_fast_path_override, ENV_SPIN_POOL};
    set_fast_path_override(ENV_SPIN_POOL, true);
    let spin = f();
    set_fast_path_override(ENV_SPIN_POOL, false);
    let rayon = f();
    clear_fast_path_overrides();
    (spin, rayon)
}

#[test]
fn par_chunks_mut_agrees_across_both_schedulers() {
    let run = || {
        let mut out = vec![0u64; 1000];
        par_chunks_mut(&mut out, 7, |ci, c| {
            for (i, slot) in c.iter_mut().enumerate() {
                *slot = (ci as u64) * 1000 + i as u64;
            }
        });
        out
    };
    let (spin, rayon) = both_arms(run);
    assert_eq!(spin, rayon, "scheduler changed the result");
    // Ragged tail: 1000 is not a multiple of 7, so the last chunk is
    // short — the place an off-by-one would differ between arms.
    assert_eq!(spin.len(), 1000);
}

#[test]
fn par_chunks_mut2_agrees_across_both_schedulers() {
    let run = || {
        let (mut a, mut b) = (vec![0u64; 517], vec![0u64; 517]);
        par_chunks_mut2(&mut a, &mut b, 32, |ci, ca, cb| {
            for (i, (x, y)) in ca.iter_mut().zip(cb.iter_mut()).enumerate() {
                *x = (ci as u64) << 20 | i as u64;
                *y = !*x;
            }
        });
        (a, b)
    };
    let (spin, rayon) = both_arms(run);
    assert_eq!(spin, rayon, "scheduler changed the result");
    // The two outputs must stay index-aligned, which is the whole
    // reason this wrapper exists.
    assert!(spin.0.iter().zip(&spin.1).all(|(x, y)| *y == !*x));
}

#[test]
fn both_wrappers_no_op_on_empty_input_either_way() {
    let (spin, rayon) = both_arms(|| {
        let mut out: Vec<u64> = Vec::new();
        par_chunks_mut(&mut out, 8, |_, _| unreachable!("no chunks to run"));
        let (mut a, mut b): (Vec<u64>, Vec<u64>) = (Vec::new(), Vec::new());
        par_chunks_mut2(&mut a, &mut b, 8, |_, _, _| unreachable!("no chunks"));
        (out, a, b)
    });
    assert_eq!(spin, rayon);
}

#[test]
fn a_zero_chunk_size_is_a_no_op_either_way() {
    let (spin, rayon) = both_arms(|| {
        let mut out = vec![9u64; 4];
        par_chunks_mut(&mut out, 0, |_, _| unreachable!("chunk==0 must not run"));
        out
    });
    assert_eq!(spin, rayon);
    assert_eq!(spin, vec![9u64; 4], "input must be untouched");
}

#[test]
fn runs_every_chunk_exactly_once() {
    let pool = SpinPool::new(4);
    let hits: Vec<AtomicU32> = (0..1000).map(|_| AtomicU32::new(0)).collect();
    pool.for_each_chunk(hits.len(), |c| {
        hits[c].fetch_add(1, Ordering::Relaxed);
    });
    for (i, h) in hits.iter().enumerate() {
        assert_eq!(h.load(Ordering::Relaxed), 1, "chunk {i} ran != once");
    }
}

/// Each participant must get ONE unbroken run of chunk indices.
///
/// `runs_every_chunk_exactly_once` pins the correctness invariant the
/// barrier needs, and it passes under round-robin striding too — so it
/// cannot catch a revert to `c += n_participants`. Contiguity is the
/// *performance* invariant: chunks map to contiguous byte ranges of the
/// weight slab, and a strided owner defeats the sequential prefetcher
/// (measured 1.87× slower at the 415 MB lm_head-class shape). This test is
/// what makes that regression loud instead of silent.
///
/// Uneven counts are included deliberately: the `rem` participants take one
/// extra chunk, which is where an off-by-one would overlap or gap two
/// participants' ranges.
#[test]
fn each_participant_runs_a_contiguous_block() {
    for (n_threads, num_chunks) in [(4usize, 1000usize), (4, 1001), (4, 3), (8, 8), (3, 100)] {
        let pool = SpinPool::new(n_threads);
        let seen: std::sync::Mutex<Vec<(std::thread::ThreadId, usize)>> =
            std::sync::Mutex::new(Vec::new());
        pool.for_each_chunk(num_chunks, |c| {
            seen.lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((std::thread::current().id(), c));
        });

        let mut by_thread: std::collections::HashMap<std::thread::ThreadId, Vec<usize>> =
            std::collections::HashMap::new();
        for (tid, c) in seen.into_inner().unwrap_or_else(|e| e.into_inner()) {
            by_thread.entry(tid).or_default().push(c);
        }

        let mut covered = 0usize;
        for (tid, mut chunks) in by_thread {
            chunks.sort_unstable();
            let (first, last) = (chunks[0], chunks[chunks.len() - 1]);
            assert_eq!(
                last - first + 1,
                chunks.len(),
                "participant {tid:?} got a non-contiguous block {chunks:?} \
                 (n_threads={n_threads}, num_chunks={num_chunks}) — this is the \
                 strided-ownership regression; see run_chunks"
            );
            covered += chunks.len();
        }
        assert_eq!(
            covered, num_chunks,
            "blocks must cover 0..{num_chunks} exactly (n_threads={n_threads})"
        );
    }
}

/// The global pool must never include efficiency cores.
///
/// Static partitioning makes the barrier wait on the slowest participant,
/// so one E-core in the pool cost 3.5× end-to-end on the 26B. This asserts
/// the cap rather than the mechanism, because the mechanism only shows up
/// under a large-shape benchmark on a heterogeneous box — the exact
/// combination that let the regression ship unnoticed.
#[test]
fn global_pool_excludes_efficiency_cores() {
    let n = global().num_threads();
    assert!(n >= 1, "pool must have at least one participant");
    if let Some(p) = performance_cores() {
        assert!(
            n <= p,
            "global pool has {n} participants but only {p} performance cores — \
             an efficiency core in a statically-partitioned pool stalls the barrier \
             (see global() docs)"
        );
    }
    assert!(
        n <= rayon::current_num_threads().max(1),
        "pool must never exceed an explicitly configured thread count"
    );
}

#[test]
fn disjoint_mut_writes_match_serial() {
    // The production pattern: each chunk writes its disjoint row range of a
    // shared output buffer via a raw pointer (caller guarantees disjoint).
    let pool = SpinPool::new(4);
    let rows = 517usize;
    let chunk = 32usize;
    let n_chunks = rows.div_ceil(chunk);
    let mut out = vec![0u64; rows];
    let ptr = out.as_mut_ptr() as usize;
    pool.for_each_chunk(n_chunks, |ci| {
        let start = ci * chunk;
        let end = (start + chunk).min(rows);
        for r in start..end {
            // SAFETY: chunks are disjoint row ranges of `out`.
            unsafe { *(ptr as *mut u64).add(r) = (r as u64) * 3 + 1 };
        }
    });
    for (r, v) in out.iter().enumerate() {
        assert_eq!(*v, (r as u64) * 3 + 1);
    }
}

#[test]
fn parallel_sum_matches_serial() {
    let pool = SpinPool::new(8);
    let n = 100_000usize;
    let partials: Vec<AtomicU64> = (0..64).map(|_| AtomicU64::new(0)).collect();
    let chunk = n.div_ceil(64);
    pool.for_each_chunk(64, |ci| {
        let start = ci * chunk;
        let end = (start + chunk).min(n);
        let s: u64 = (start as u64..end as u64).sum();
        partials[ci].store(s, Ordering::Relaxed);
    });
    let got: u64 = partials.iter().map(|a| a.load(Ordering::Relaxed)).sum();
    let want: u64 = (0..n as u64).sum();
    assert_eq!(got, want);
}

#[test]
fn zero_chunks_is_noop() {
    let pool = SpinPool::new(4);
    pool.for_each_chunk(0, |_| panic!("must not run"));
}

#[test]
fn single_thread_runs_inline() {
    let pool = SpinPool::new(1);
    let hits: Vec<AtomicU32> = (0..50).map(|_| AtomicU32::new(0)).collect();
    pool.for_each_chunk(hits.len(), |c| {
        hits[c].fetch_add(1, Ordering::Relaxed);
    });
    assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
}

#[test]
fn chunk_panic_propagates_and_pool_stays_usable() {
    // A panicking body must (a) NOT hang the barrier (a dead worker would
    // never count its chunk → dispatcher spins forever) and (b) propagate
    // the panic to the dispatcher. Chunk 37 lands on a worker (37 % 4 != 0),
    // exercising the worker-side catch, not just the dispatcher's own.
    let pool = SpinPool::new(4);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.for_each_chunk(50, |c| {
            if c == 37 {
                panic!("boom at chunk {c}");
            }
        });
    }));
    assert!(
        result.is_err(),
        "a panicking chunk body must propagate to the dispatcher"
    );
    // The pool must still work after a panic (not poisoned / not hung).
    let hits: Vec<AtomicU32> = (0..20).map(|_| AtomicU32::new(0)).collect();
    pool.for_each_chunk(hits.len(), |c| {
        hits[c].fetch_add(1, Ordering::Relaxed);
    });
    assert!(
        hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
        "pool must stay usable after a chunk panic"
    );
}

#[test]
fn concurrent_dispatchers_stay_consistent() {
    // Multiple driver threads dispatching on one shared pool (the
    // `--concurrent N` / multi-threaded-test shape). The dispatch lock
    // serializes them; each dispatch must still complete correctly.
    let pool = SpinPool::new(4);
    std::thread::scope(|s| {
        for _ in 0..3 {
            s.spawn(|| {
                for round in 1..=50u64 {
                    let acc: Vec<AtomicU64> = (0..20).map(|_| AtomicU64::new(0)).collect();
                    pool.for_each_chunk(20, |c| {
                        acc[c].store(round * (c as u64 + 1), Ordering::Relaxed);
                    });
                    for (c, a) in acc.iter().enumerate() {
                        assert_eq!(a.load(Ordering::Relaxed), round * (c as u64 + 1));
                    }
                }
            });
        }
    });
}

/// Cross-dispatch read-after-write — the real decode pipeline shape
/// (dispatch A writes a buffer; the *next* dispatch B reads it and writes a
/// derived buffer). Exercises the visibility the disjoint-write tests don't:
/// workers running dispatch B must observe ALL of dispatch A's writes (the
/// `barrier_A.Acquire → epoch_B.Release → worker_B.Acquire` chain). The pool
/// is oversubscribed (more workers than cores) so the barrier routinely waits
/// on a descheduled worker. Kept fast (a few hundred rounds) — under EXTREME
/// oversubscription (2× burners, 4000 rounds) this and the disjoint-write
/// path stayed correct, so this is a regression guard, not a repro.
#[test]
fn stress_cross_dispatch_read_after_write() {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    // Oversubscribe the pool itself (more workers than cores) so the barrier
    // routinely waits on a descheduled worker.
    let pool = SpinPool::new((cores + 2).max(4));
    let n = 61usize; // chunks; not a multiple of the thread count
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    for round in 1..=400u64 {
        // Dispatch A: fill `a` with a round-derived pattern.
        let pa = a.as_mut_ptr() as usize;
        pool.for_each_chunk(n, |c| {
            // SAFETY: chunk c owns element c.
            unsafe { *(pa as *mut u64).add(c) = round.wrapping_mul(c as u64 + 1) | 1 };
        });
        // Dispatch B: read `a`, write `b = f(a)`. If B's workers don't see
        // all of A's writes, `b[c]` is wrong (or derived from a stale 0).
        let pa_r = a.as_ptr() as usize;
        let pb = b.as_mut_ptr() as usize;
        pool.for_each_chunk(n, |c| {
            // SAFETY: read element c (written by A's chunk c), write b[c].
            let av = unsafe { *(pa_r as *const u64).add(c) };
            unsafe { *(pb as *mut u64).add(c) = av.wrapping_mul(31).wrapping_add(7) };
        });
        for c in 0..n {
            let want_a = round.wrapping_mul(c as u64 + 1) | 1;
            assert_eq!(a[c], want_a, "round {round} chunk {c}: A wrong");
            assert_eq!(
                b[c],
                want_a.wrapping_mul(31).wrapping_add(7),
                "round {round} chunk {c}: B read a stale/partial A"
            );
        }
    }
}

#[test]
fn back_to_back_dispatches_reuse_workers() {
    // Exercises the epoch path: many tiny dispatches in a row (the decode
    // loop shape) must each complete fully.
    let pool = SpinPool::new(4);
    for round in 1..=200u64 {
        let acc: Vec<AtomicU64> = (0..16).map(|_| AtomicU64::new(0)).collect();
        pool.for_each_chunk(16, |c| {
            acc[c].store(round * (c as u64 + 1), Ordering::Relaxed);
        });
        for (c, a) in acc.iter().enumerate() {
            assert_eq!(a.load(Ordering::Relaxed), round * (c as u64 + 1));
        }
    }
}

// ── SIGSEGV reproduction attempt (three crash reports, all localizing
// inside `par_chunks_mut`/`run_chunks`, fault addresses matching real
// model dimensions or `0x1`) ────────────────────────────────────────────
//
// Gated on `heavy_tests`: these run a production-scale decode simulation
// (34 layers × hundreds of tokens, deliberately oversubscribed) and take
// minutes under coverage instrumentation on a 2-core CI runner — a spin
// barrier has no spare core to spin on there, so CI pays the worst case
// of exactly the contention these tests exist to provoke. Run locally via
// `make larql-compute-test-integration`.
#[cfg(feature = "heavy_tests")]
mod stress_decode_shape {
    use super::*;

    const RD_HIDDEN: usize = 2560;
    const RD_INTER: usize = 10240;
    const RD_Q_DIM: usize = 2048;
    const RD_KV_DIM: usize = 1024;
    const RD_ROWS_CHUNK: usize = 32;
    const RD_ELEM_CHUNK: usize = 256;

    fn rd_f32_encode(caller: u64, round: u64, tag: u64, idx: usize) -> f32 {
        // Cheap, collision-resistant-enough encoding of (caller, round, tag,
        // idx) into an f32 so a read of the wrong slot/epoch/caller's data is
        // caught as a mismatch rather than silently looking plausible.
        ((caller.wrapping_mul(998_244_353)
            ^ round.wrapping_mul(1_000_003)
            ^ tag.wrapping_mul(97)
            ^ idx as u64)
            % 1_000_000) as f32
    }
    fn rd_fill_chunk(chunk: &mut [f32], global_start: usize, caller: u64, round: u64, tag: u64) {
        for (local_i, v) in chunk.iter_mut().enumerate() {
            // Cheap but non-trivial busy-work per element (~tens of ns) so a
            // chunk's dispatched duration is closer to the real matvec
            // kernel's (many SDOTs per row) than a near-instant synthetic
            // write - in case the bug is timing/duration-dependent (spin ->
            // yield -> park transitions calibrated around real chunk cost).
            let mut acc = 0.0f32;
            for j in 0..64u32 {
                acc = acc * 1.0000001 + (j as f32).sin();
            }
            *v = rd_f32_encode(caller, round, tag, global_start + local_i) + acc * 0.0;
        }
    }
    fn rd_check(buf: &[f32], caller: u64, round: u64, tag: u64) {
        for (i, &v) in buf.iter().enumerate() {
            let want = rd_f32_encode(caller, round, tag, i);
            assert_eq!(
                v.to_bits(),
                want.to_bits(),
                "caller {caller} round {round} tag {tag} idx {i}: corrupted \
             (got {v}, want {want}) - buffer len {}",
                buf.len()
            );
        }
    }

    /// One simulated decode step's worth of dispatches against `par_chunks_mut`
    /// - the actual public entry point production uses, with the REAL
    ///   gemma-3-4b-it row counts (q_dim=2048, kv_dim=1024, hidden=2560,
    ///   intermediate=10240 - all exact multiples of their chunk sizes, same as
    ///   production, so this can't exercise the underflow this file already
    ///   hardened against; it's targeting a different bug). `caller` tags every
    ///   value so concurrent callers (simulating overlapping requests) can tell
    ///   their own data apart from a caller they got mixed up with.
    fn rd_run(caller: u64, rounds: u64, layers: u64) {
        let mut q = vec![0.0f32; RD_Q_DIM];
        let mut k = vec![0.0f32; RD_KV_DIM];
        let mut v = vec![0.0f32; RD_KV_DIM];
        let mut o = vec![0.0f32; RD_HIDDEN];
        let mut gate = vec![0.0f32; RD_INTER];
        let mut up = vec![0.0f32; RD_INTER];
        let mut activated = vec![0.0f32; RD_INTER];
        let mut down = vec![0.0f32; RD_HIDDEN];

        for round in 0..rounds {
            for layer in 0..layers {
                let tag = round * layers + layer;
                for (buf, sub) in [
                    (&mut q, 0u64),
                    (&mut k, 1),
                    (&mut v, 2),
                    (&mut o, 3),
                    (&mut gate, 4),
                    (&mut up, 5),
                    (&mut down, 7),
                ] {
                    let t = tag.wrapping_mul(8) + sub;
                    par_chunks_mut(buf, RD_ROWS_CHUNK, |ci, chunk| {
                        rd_fill_chunk(chunk, ci * RD_ROWS_CHUNK, caller, round, t)
                    });
                    rd_check(buf, caller, round, t);
                }
                // Elementwise activation: 256-chunked, INTER-sized - the
                // production shape at kquant_forward/cached.rs:869.
                let t = tag.wrapping_mul(8) + 6;
                par_chunks_mut(&mut activated, RD_ELEM_CHUNK, |ci, chunk| {
                    rd_fill_chunk(chunk, ci * RD_ELEM_CHUNK, caller, round, t)
                });
                rd_check(&activated, caller, round, t);
            }
        }
    }

    /// Single sequential caller, at production scale (34 layers x 400
    /// simulated tokens). `--test-threads` contention from other tests
    /// sharing `global()` is part of the reproduction attempt - do not run
    /// this test in isolation only.
    #[test]
    fn stress_realistic_decode_shape_no_corruption() {
        rd_run(0, 40, 34);
    }

    /// Genuinely concurrent callers (unlike `concurrent_dispatchers_stay_
    /// consistent` above, which uses one FIXED dispatch size for every
    /// thread) - each thread runs its own independent `rd_run`-shaped decode
    /// loop against the SAME shared `global()` pool at the SAME time, so
    /// `dispatch_lock` has to actually serialize dispatchers with DIFFERENT
    /// `(total, chunk)` shapes mid-flight, not just identical ones.
    #[test]
    fn stress_concurrent_realistic_decode_shape_no_corruption() {
        std::thread::scope(|s| {
            for caller in 0..6u64 {
                s.spawn(move || rd_run(caller, 8, 34));
            }
        });
    }

    /// Empty-block straggler pin: alternate a 1-chunk dispatch (most
    /// participants own an empty block, so the barrier passes without
    /// them) with a wide dispatch, back-to-back. Pre-`task_gen` a
    /// straggler from the 1-chunk dispatch could read the re-published
    /// slot, run the wide dispatch's chunks attributed to the old epoch,
    /// then run them AGAIN on re-observing the epoch — over-counting
    /// `completed`, releasing the barrier early, and letting the closure
    /// drop while another worker still executed it (SIGSEGV under
    /// coverage instrumentation, where the window is widest). Exactly-once
    /// per chunk is the observable invariant; the counters are re-created
    /// per dispatch so a late write from a released dispatch is also a
    /// use-after-free the harness can catch.
    #[test]
    fn alternating_block_sizes_run_each_chunk_exactly_once() {
        use std::sync::atomic::AtomicU8;
        const PARTICIPANTS: usize = 4;
        const WIDE_CHUNKS: usize = 7;
        const ITERATIONS: usize = 50_000;
        let pool = SpinPool::new(PARTICIPANTS);
        for iter in 0..ITERATIONS {
            let n = if iter % 2 == 0 { 1 } else { WIDE_CHUNKS };
            let counters: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
            pool.for_each_chunk(n, |c| {
                counters[c].fetch_add(1, Ordering::Relaxed);
            });
            for (c, counter) in counters.iter().enumerate() {
                assert_eq!(
                    counter.load(Ordering::Relaxed),
                    1,
                    "iter {iter}: chunk {c} of {n} not run exactly once"
                );
            }
        }
    }
}
