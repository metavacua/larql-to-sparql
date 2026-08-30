//! The three kernels must agree, and the executor's partitioning must
//! not change an answer.

use super::super::executor::CpuExecutor;
use super::super::kernels::{bf16_dot, bf16_dot_portable, BlasF32, FusedBf16, ScalarF32};
use super::super::projector::{CpuParallelism, DenseProjector, WeightRows};
use crate::format::vindex3::fixtures::lcg_values;

fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}
fn widen(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn metrics(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut num, mut den, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    for (p, q) in a.iter().zip(b) {
        num += (*p as f64 - *q as f64).powi(2);
        den += (*q as f64).powi(2);
        mx = mx.max((*p as f64 - *q as f64).abs());
    }
    ((num / den.max(f64::MIN_POSITIVE)).sqrt(), mx)
}

/// The BF16 widen is EXACT — it is a bit-shift, not a conversion.
///
/// This is what lets the fused kernel claim it changes representation and
/// mechanics and no numerical value: every stored code unit denotes
/// exactly the f32 the scalar path would have multiplied.
#[test]
fn the_bf16_widen_is_exact_not_a_conversion() {
    for v in [0.0f32, 1.0, -1.0, 1e-30, 1e30, 0.1, -12345.678] {
        let round = widen(narrow(v));
        assert_eq!(
            round.to_bits() & 0xffff_0000,
            v.to_bits() & 0xffff_0000,
            "widen(narrow({v})) lost the top half"
        );
        assert_eq!(round, widen(narrow(round)), "not idempotent at {v}");
    }
}

/// All three kernels compute the same projection, to summation order.
#[test]
fn the_kernels_agree_on_the_same_projection() {
    let (out_dim, in_dim) = (257usize, 320usize); // deliberately not round
    let f32w: Vec<f32> = lcg_values(out_dim * in_dim, 7)
        .iter()
        .map(|v| widen(narrow(*v)))
        .collect();
    let bf: Vec<u16> = f32w.iter().map(|v| narrow(*v)).collect();
    let x = lcg_values(in_dim, 9);

    let exec = CpuExecutor::new().unwrap();
    let scalar = exec.project(&ScalarF32, WeightRows::F32(&f32w), &x, out_dim);
    let blas = exec.project(&BlasF32, WeightRows::F32(&f32w), &x, out_dim);
    let fused = exec.project(&FusedBf16, WeightRows::Bf16(&bf), &x, out_dim);

    let (rel_b, _) = metrics(&blas, &scalar);
    let (rel_f, _) = metrics(&fused, &scalar);
    assert!(rel_b < 1e-5, "blas vs scalar rel_rms {rel_b:e}");
    assert!(
        rel_f < 1e-5,
        "fused bf16 vs scalar rel_rms {rel_f:e} — the widen should introduce nothing \
         beyond summation order"
    );
}

/// **Partitioning must not change an answer.**
///
/// The executor is free to cut rows however it likes; a kernel that read
/// outside its slab, or an executor that mis-sliced one, would show up
/// here and nowhere else. Row counts chosen so the last partition is
/// short.
#[test]
fn the_row_partition_does_not_change_the_result() {
    let (out_dim, in_dim) = (1001usize, 128usize);
    let f32w: Vec<f32> = lcg_values(out_dim * in_dim, 3)
        .iter()
        .map(|v| widen(narrow(*v)))
        .collect();
    let bf: Vec<u16> = f32w.iter().map(|v| narrow(*v)).collect();
    let x = lcg_values(in_dim, 4);

    let exec = CpuExecutor::new().unwrap();
    // One call over everything — the definition.
    let mut whole = vec![0.0f32; out_dim];
    FusedBf16.project_rows(WeightRows::Bf16(&bf), &x, &mut whole);

    for workers in [1usize, 2, 3, 5, 8, 13] {
        let rows = out_dim.div_ceil(workers);
        let mut split = vec![0.0f32; out_dim];
        for (i, slot) in split.chunks_mut(rows).enumerate() {
            let slab = WeightRows::Bf16(&bf).slice_rows(in_dim, i * rows, slot.len());
            FusedBf16.project_rows(slab, &x, slot);
        }
        assert_eq!(
            split, whole,
            "{workers}-way partition changed the result — row slabs are independent \
             and must be bit-identical however they are cut"
        );
    }
    // And through the executor's own policy.
    let via = exec.project(&FusedBf16, WeightRows::Bf16(&bf), &x, out_dim);
    assert_eq!(via, whole);
}

/// Each kernel states who threads it, and the executor honours it.
///
/// The rule this file exists to protect: at most one layer of parallelism
/// owns the machine. A `LibraryOwned` kernel that the executor also
/// partitioned would nest Accelerate's threads inside Rayon's.
#[test]
fn threading_ownership_is_declared_not_assumed() {
    assert_eq!(ScalarF32.parallelism(), CpuParallelism::Serial);
    assert_eq!(BlasF32.parallelism(), CpuParallelism::LibraryOwned);
    assert_eq!(FusedBf16.parallelism(), CpuParallelism::ExternalPool);

    let exec = CpuExecutor::new().unwrap();
    assert!(exec.workers() >= 1);
}

/// Every matrix class Qwen3.8-27B actually decodes through, with the
/// stored geometry taken from the container's own tensor table — not a
/// representative subset. CPU-1B benched three shapes and the policy has
/// to cover ten; the two that were never measured (`k_proj`/`v_proj` at
/// 1024 rows, and the `48`-row delta gates) are exactly the ones near a
/// crossover, which is where a threshold guessed from the large shapes
/// would be wrong.
const REAL_SHAPES: &[(&str, usize, usize)] = &[
    ("mlp gate/up_proj", 17408, 5120),
    ("mlp down_proj", 5120, 17408),
    ("delta in_proj_qkv", 10240, 5120),
    ("delta in_proj_z", 6144, 5120),
    ("delta out_proj", 5120, 6144),
    ("attn q_proj", 12288, 5120),
    ("attn o_proj", 5120, 6144),
    ("attn k/v_proj", 1024, 5120),
    ("delta in_proj_a/b", 48, 5120),
    ("output head", 248320, 5120),
];

/// Row counts swept at the model's own `in_dim` to LOCATE the crossover
/// rather than infer it. `MIN_COMPACT_BYTES` is read off this, so the
/// sweep has to bracket the answer on both sides — a sweep that only
/// went one way would confirm whatever it started from.
const CROSSOVER_ROWS: &[usize] = &[48, 192, 384, 512, 640, 768, 832, 896, 1024, 1280, 2048];

/// The `in_dim` every crossover point is measured at: Qwen3.8's hidden.
const CROSSOVER_IN_DIM: usize = 5120;

/// The executor's policy, measured on the real shapes.
///
/// Env-gated. Confirms the seam reproduces CPU-1B's hand-rolled numbers
/// rather than losing them to the abstraction — an executor whose
/// dispatch cost ate the win would be worse than no seam at all.
///
/// ```text
/// QW_CPU_EXEC_BENCH=1 cargo test --release exec::cpu -- --nocapture
/// ```
#[test]
fn executor_policy_bench() {
    if std::env::var("QW_CPU_EXEC_BENCH").is_err() {
        eprintln!("SKIP executor_policy_bench: set QW_CPU_EXEC_BENCH=1");
        return;
    }
    use std::time::Instant;
    let exec = CpuExecutor::new().unwrap();
    println!("\n  executor workers: {}\n", exec.workers());
    println!(
        "  {:22} {:>10} {:>10} {:>10}   bytes read",
        "shape", "blas f32", "fused bf16", "speedup"
    );
    for (label, out_dim, in_dim) in REAL_SHAPES.iter().copied() {
        let f32w: Vec<f32> = lcg_values(out_dim * in_dim, 11)
            .iter()
            .map(|v| widen(narrow(*v)))
            .collect();
        let bf: Vec<u16> = f32w.iter().map(|v| narrow(*v)).collect();
        let x = lcg_values(in_dim, 22);
        let iters = (1_000_000_000.0 / (out_dim * in_dim) as f64).clamp(3.0, 200.0) as usize;

        let mut sink = 0.0f32;
        let t = Instant::now();
        for _ in 0..iters {
            sink += exec.project(&BlasF32, WeightRows::F32(&f32w), &x, out_dim)[0];
        }
        let b = t.elapsed().as_secs_f64() / iters as f64;
        let t = Instant::now();
        for _ in 0..iters {
            sink += exec.project(&FusedBf16, WeightRows::Bf16(&bf), &x, out_dim)[0];
        }
        let f = t.elapsed().as_secs_f64() / iters as f64;
        std::hint::black_box(sink);
        println!(
            "  {label:22} {:8.2}ms {:8.2}ms {:9.2}x   f32 {:5.1} / bf16 {:5.1} GB/s",
            b * 1e3,
            f * 1e3,
            b / f,
            (out_dim * in_dim * 4) as f64 / b / 1e9,
            (out_dim * in_dim * 2) as f64 / f / 1e9,
        );
    }
    println!();
}

/// **The threshold probe.** Where does compact-to-registers start winning?
///
/// `MIN_COMPACT_BYTES` decides, per matrix, whether the checkpoint's bf16
/// bytes stay compact into a fused kernel or get widened for BLAS. CPU-1B
/// measured two points either side of that question (48 rows: BLAS by
/// 3.8x; 10240 rows: fused by 2.5x) and a threshold picked from two points
/// is a guess wearing a number. This sweeps the row count at the model's
/// own `in_dim` so the constant is READ OFF a curve.
///
/// Reports the ratio in both directions on purpose: the decision is not
/// "which is faster" but "how much is at stake", and a shallow crossover
/// means the constant's exact value barely matters — which is itself
/// worth knowing before defending one.
///
/// ```text
/// QW_CPU_CROSSOVER=1 cargo test --release exec::cpu::kernels_tests -- --nocapture
/// ```
#[test]
fn compact_crossover_probe() {
    if std::env::var("QW_CPU_CROSSOVER").is_err() {
        eprintln!("SKIP compact_crossover_probe: set QW_CPU_CROSSOVER=1");
        return;
    }
    let exec = CpuExecutor::new().unwrap();
    println!(
        "\n  crossover sweep at in_dim {CROSSOVER_IN_DIM}, {} workers\n",
        exec.workers()
    );
    println!(
        "  {:>6} {:>12} {:>12} {:>10} {:>10} {:>12}",
        "rows", "f32 bytes", "bf16 bytes", "blas ms", "fused ms", "fused/blas"
    );
    for &rows in CROSSOVER_ROWS {
        let (blas, fused) = time_both(&exec, rows, CROSSOVER_IN_DIM);
        println!(
            "  {rows:>6} {:>10.2} MB {:>10.2} MB {:>10.3} {:>10.3} {:>11.2}x",
            (rows * CROSSOVER_IN_DIM * 4) as f64 / 1e6,
            (rows * CROSSOVER_IN_DIM * 2) as f64 / 1e6,
            blas * 1e3,
            fused * 1e3,
            blas / fused,
        );
    }
    println!("\n  fused/blas > 1 means compact-to-registers WINS at that size.\n");
}

/// One shape, both kernels, through the executor's own policy.
///
/// Through the executor rather than the kernels directly: the threshold
/// this feeds governs a decision the executor makes, so measuring the
/// kernels bare would price something the model never runs.
fn time_both(exec: &CpuExecutor, out_dim: usize, in_dim: usize) -> (f64, f64) {
    use std::time::Instant;
    let f32w: Vec<f32> = lcg_values(out_dim * in_dim, 11)
        .iter()
        .map(|v| widen(narrow(*v)))
        .collect();
    let bf: Vec<u16> = f32w.iter().map(|v| narrow(*v)).collect();
    let x = lcg_values(in_dim, 22);
    let iters = (2_000_000_000.0 / (out_dim * in_dim) as f64).clamp(5.0, 400.0) as usize;

    let mut sink = 0.0f32;
    // One untimed pass each: the first touch of a 100 MB slab faults it
    // in, and charging that to whichever kernel ran first would invent a
    // crossover out of page faults.
    sink += exec.project(&BlasF32, WeightRows::F32(&f32w), &x, out_dim)[0];
    sink += exec.project(&FusedBf16, WeightRows::Bf16(&bf), &x, out_dim)[0];

    let t = Instant::now();
    for _ in 0..iters {
        sink += exec.project(&BlasF32, WeightRows::F32(&f32w), &x, out_dim)[0];
    }
    let blas = t.elapsed().as_secs_f64() / iters as f64;
    let t = Instant::now();
    for _ in 0..iters {
        sink += exec.project(&FusedBf16, WeightRows::Bf16(&bf), &x, out_dim)[0];
    }
    let fused = t.elapsed().as_secs_f64() / iters as f64;
    std::hint::black_box(sink);
    (blas, fused)
}

/// **At most one layer of parallelism owns the machine.**
///
/// The batched driver runs whole positions in parallel and calls the
/// backend from inside that loop, so a projection reached from there must
/// NOT fan out again. Without this the batch path would nest a
/// twelve-worker partition inside every position of an already-parallel
/// loop — the rule in [`super`] broken from the caller's end rather than
/// the kernel's.
///
/// Counts `project_rows` calls rather than timing anything: the claim is
/// about how the work was CUT, and a timing test would pass on a fast
/// machine whatever the answer.
#[test]
fn a_projection_inside_a_parallel_region_does_not_fan_out_again() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts the slabs it is handed. Declares `ExternalPool`, so an
    /// executor that ignored the caller would partition it.
    struct Counting(AtomicUsize);
    impl DenseProjector for Counting {
        fn parallelism(&self) -> CpuParallelism {
            CpuParallelism::ExternalPool
        }
        fn project_rows(&self, _w: WeightRows<'_>, _x: &[f32], out: &mut [f32]) {
            self.0.fetch_add(1, Ordering::Relaxed);
            out.fill(1.0);
        }
    }

    let exec = CpuExecutor::new().unwrap();
    // Big enough to clear the split threshold, or the executor would
    // decline for the wrong reason and the test would prove nothing.
    let in_dim = 512;
    let out_dim = 8192;
    let w = vec![0u16; out_dim * in_dim];
    let x = vec![1.0f32; in_dim];
    assert!(WeightRows::Bf16(&w).bytes() > 4 * 1024 * 1024);

    let outer = Counting(AtomicUsize::new(0));
    exec.project(&outer, WeightRows::Bf16(&w), &x, out_dim);
    let owned = outer.0.load(Ordering::Relaxed);
    assert!(
        owned > 1,
        "on its own thread the executor must partition, or the comparison below is vacuous"
    );

    let inner = Counting(AtomicUsize::new(0));
    rayon::scope(|s| {
        s.spawn(|_| {
            exec.project(&inner, WeightRows::Bf16(&w), &x, out_dim);
        });
    });
    assert_eq!(
        inner.0.load(Ordering::Relaxed),
        1,
        "a caller that already owns the machine must get ONE call, not {owned}"
    );
}

/// The pool is the machine's, not the backend's.
#[test]
fn the_shared_pool_is_one_pool() {
    let a = super::super::executor::shared().expect("the pool builds");
    let b = super::super::executor::shared().expect("and is the same one");
    assert!(std::ptr::eq(a, b));
    assert!(a.workers() >= 1);
}

/// The worker count falls back sanely where no performance-core split is
/// reported — every non-Apple target, and a machine reporting nonsense.
#[test]
fn the_worker_count_saturates_rather_than_fills_the_machine() {
    use super::super::executor::workers_from;
    // Half the cores, because a streaming kernel saturates memory long
    // before it runs out of cores — measured 488.7 ms of projection at
    // twelve workers against 440.3 at six, on the same token.
    assert_eq!(workers_from(Some(12)), 6);
    assert_eq!(workers_from(Some(16)), 8);
    // ...but never below the measured floor, where the memory system
    // stops being saturated at all (3 workers cost 589 ms/token).
    assert_eq!(workers_from(Some(8)), 4);
    assert_eq!(workers_from(Some(6)), 4);
    // ...and never more workers than the machine has cores.
    assert_eq!(workers_from(Some(2)), 2);
    assert_eq!(workers_from(Some(1)), 1);

    // Nonsense and silence both fall back to the total core count, then
    // through the same policy.
    let total = std::thread::available_parallelism().map_or(1, |n| n.get());
    let expect = (total / 2).clamp(1, total).max(4.min(total));
    assert_eq!(workers_from(None), expect);
    assert_eq!(
        workers_from(Some(0)),
        expect,
        "zero performance cores is nonsense, not a pool size"
    );
}

/// `WeightRows` reports geometry and slices the same way in both
/// representations.
///
/// Both arms, because the f32 one is the arm the model's own decode never
/// takes for a large matrix — so it is exactly the arm that could rot
/// unnoticed.
#[test]
fn row_geometry_is_the_same_in_both_representations() {
    let f = vec![1.0f32; 6 * 4];
    let b = vec![0x3f80u16; 6 * 4];
    for (rows, width) in [(WeightRows::F32(&f), 4usize), (WeightRows::Bf16(&b), 2)] {
        assert_eq!(rows.rows(4), 6);
        assert_eq!(rows.bytes(), 6 * 4 * width);
        let cut = rows.slice_rows(4, 2, 3);
        assert_eq!(cut.rows(4), 3);
        assert_eq!(cut.bytes(), 3 * 4 * width);
    }
}

/// The portable widen-and-accumulate and the NEON one agree.
///
/// `bf16_dot_portable` calls itself "the definition the NEON version must
/// agree with", and until now nothing checked that. On aarch64 it is dead
/// code — every call dispatches to NEON — so the claim was carried by its
/// own doc comment and would only be tested by shipping x86 a wrong
/// answer.
///
/// Not asserted bit-for-bit: the NEON version keeps four accumulators and
/// the portable one keeps a running sum, so they reassociate differently
/// on purpose. The claim is that they compute the same dot product, and
/// the tolerance is f32 summation over the length used here.
#[test]
fn the_portable_and_neon_dots_agree() {
    for len in [1usize, 7, 16, 17, 64, 129, 512] {
        let f: Vec<f32> = (0..len).map(|i| ((i * 37) as f32 * 0.011).sin()).collect();
        let w: Vec<u16> = f.iter().map(|v| narrow(*v)).collect();
        let x: Vec<f32> = (0..len).map(|i| ((i * 11) as f32 * 0.017).cos()).collect();

        let portable = bf16_dot_portable(&w, &x);
        let dispatched = bf16_dot(&w, &x);
        let magnitude: f32 = w.iter().zip(&x).map(|(b, v)| (widen(*b) * v).abs()).sum();
        assert!(
            (portable - dispatched).abs() <= 1e-6 * magnitude.max(1.0),
            "len {len}: portable {portable} vs dispatched {dispatched}"
        );
    }
}

/// Each kernel refuses a representation it cannot read, by name.
///
/// A panic and not an error on purpose: `PhysicalProjectionPlan` pairs
/// format to kernel in one value and the executor OBSERVES that pairing
/// rather than re-deriving it, so a mismatch here is not bad input — it is
/// that invariant broken, and it should stop rather than be handled.
#[test]
#[should_panic(expected = "f32 weights only")]
fn the_blas_kernel_refuses_compact_weights() {
    let w = [0x3f80u16; 8];
    let mut out = [0.0f32; 2];
    BlasF32.project_rows(WeightRows::Bf16(&w), &[1.0; 4], &mut out);
}

#[test]
#[should_panic(expected = "f32 weights only")]
fn the_scalar_oracle_refuses_compact_weights() {
    let w = [0x3f80u16; 8];
    let mut out = [0.0f32; 2];
    ScalarF32.project_rows(WeightRows::Bf16(&w), &[1.0; 4], &mut out);
}

#[test]
#[should_panic(expected = "bf16 weights only")]
fn the_fused_kernel_refuses_widened_weights() {
    let w = [1.0f32; 8];
    let mut out = [0.0f32; 2];
    FusedBf16.project_rows(WeightRows::F32(&w), &[1.0; 4], &mut out);
}

/// The pool size can be pinned, and only by a value that makes sense.
///
/// The parse is deliberately strict about the floor: a `0` reaching
/// `ThreadPoolBuilder` produces rayon's DEFAULT pool, not an empty one,
/// so a typo would silently change the topology under a measurement and
/// look like a result.
#[test]
fn a_pinned_pool_size_is_honoured_and_nonsense_is_not() {
    use super::super::executor::{parse_workers, WORKERS_ENV};
    assert_eq!(WORKERS_ENV, "LARQL_CPU_WORKERS");
    // The executor's OWN parse, not a restatement of it: a test that
    // rewrote the chain would agree with itself whatever the real one
    // did. Called directly rather than through the environment, which the
    // rest of the suite shares.
    assert_eq!(parse_workers(Some("10")), Some(10));
    assert_eq!(parse_workers(Some(" 8 ")), Some(8));
    assert_eq!(parse_workers(Some("0")), None);
    assert_eq!(parse_workers(Some("-4")), None);
    assert_eq!(parse_workers(Some("many")), None);
    assert_eq!(parse_workers(Some("")), None);
    assert_eq!(parse_workers(None), None);
}
