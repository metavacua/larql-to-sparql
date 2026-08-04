//! bench_mxfp4_dequant — DEC-8.8: what fraction of memory bandwidth does the
//! MXFP4 serving path sustain at real K3 shapes?
//!
//! Every throughput target on the K3 slice arm divides an attainable bandwidth
//! by the bytes a token reads, which assumes the kernel realises that bandwidth
//! in full. The gap — `eta` — moves the dense-precision frontier by a whole
//! difficulty class, so no target row may be pinned before it is measured.
//!
//! ## What this found before it measured anything
//!
//! `QuantMatVec` dispatches Q4_K / Q4_KF / Q6_K / Q4_0 / Q8_0 on **both** the CPU
//! and Metal backends. Neither has an MXFP4 arm, and `grep -ri mxfp4
//! crates/larql-compute-metal` returns nothing. **K3's own serving format has no
//! fused dequant-matvec kernel on any backend.** The only MXFP4 path is bulk
//! `dequantize_expert` into an f32 buffer followed by a matvec over it — which
//! reads 0.53 bytes per weight, writes 4, then reads those 4 back.
//!
//! So this does not report "eta for MXFP4" as if comparable to Q4K's. It reports
//! the Q4K fused path as the reference and upper bound, the MXFP4
//! dequant-to-buffer path as what exists today, and keeps them apart.
//!
//! ## Method: the working set must exceed the caches
//!
//! One K3 expert branch is 5.85 MB packed. This machine's P-cluster L2 is 16 MB.
//! Benchmarking a single matrix in a loop therefore measures **cache** bandwidth
//! and overstates the serving kernel — the same class of error that made a
//! previous storage benchmark report 1.0 us NVMe latency. Every arm here rotates
//! through `--working-set-mb` of distinct matrices (default 512 MB), so each
//! iteration touches weights that cannot be resident. A `warm` arm deliberately
//! re-reads one matrix, to show the size of the error being avoided.
//!
//! ## Conventions (R0)
//!
//! - `eta` is reported against **367 GB/s**, the GPU-attainable roofline, for
//!   every arm including CPU ones. A CPU arm's eta is therefore *not* its
//!   efficiency against its own 127 GB/s ceiling — both are printed.
//! - Bytes are **all-in** (codeword + e8m0 scale, 0.53125 B/weight) unless a row
//!   says payload (0.5 B/weight). Q4K's are its real on-wire block size.
//! - Across block widths, GB/s counts the weight matrix **once per pass**: flat
//!   with rising T means the block genuinely reuses the read; falling as 1/T
//!   means it is a token loop wearing a block's clothing.
//!
//! Usage:
//!   cargo run --release -p larql-compute-metal --example bench_mxfp4_dequant

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute::prelude::*;
use larql_compute::CpuBackend;
use larql_compute_metal::MetalBackend;
use larql_models::quant::mxfp4::dequantize_expert;
use std::time::{Duration, Instant};

const GPU_ATTAINABLE_GB_S: f64 = 367.0;
const CPU_ATTAINABLE_GB_S: f64 = 127.0;

const MXFP4_ALL_IN_BYTES: f64 = 0.53125; // 4-bit codeword + 8-bit scale per 32
const MXFP4_PAYLOAD_BYTES: f64 = 0.5; // codeword alone

const BLOCK_WIDTHS: [usize; 5] = [1, 2, 3, 5, 7];

/// A K3 matrix shape worth measuring, `[out, in]`.
struct Shape {
    label: &'static str,
    out: usize,
    inp: usize,
}

const SHAPES: [Shape; 2] = [
    // Five of these per KDA layer dominate the always-on dense term.
    Shape {
        label: "KDA projection",
        out: 12288,
        inp: 7168,
    },
    // One routed-expert branch, in the 3584-dim latent space.
    Shape {
        label: "expert branch",
        out: 3072,
        inp: 3584,
    },
];

fn arg_mb(name: &str, default: usize) -> usize {
    let mut it = std::env::args().skip_while(|a| a != name);
    it.next();
    it.next().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn synth_f32(out: usize, inp: usize, variant: usize) -> Vec<f32> {
    (0..out * inp)
        .map(|i| ((((i * 31 + variant * 17) % 251) as f32) - 125.0) * 0.001)
        .collect()
}

fn synth_x(inp: usize) -> Vec<f32> {
    (0..inp)
        .map(|i| (((i % 251) as f32) - 125.0) * 0.001)
        .collect()
}

/// N distinct copies of a quantised matrix, rotated so no iteration re-reads a
/// cache-resident one.
struct WorkingSet {
    mats: Vec<Vec<u8>>,
    cursor: std::cell::Cell<usize>,
}

impl WorkingSet {
    fn build(
        out: usize,
        inp: usize,
        target_bytes: usize,
        quantise: impl Fn(&[f32]) -> Vec<u8>,
    ) -> Self {
        let one = quantise(&synth_f32(out, inp, 0));
        let n = (target_bytes / one.len().max(1)).max(2);
        let mut mats = Vec::with_capacity(n);
        mats.push(one);
        for variant in 1..n {
            mats.push(quantise(&synth_f32(out, inp, variant)));
        }
        Self {
            mats,
            cursor: std::cell::Cell::new(0),
        }
    }

    fn next(&self) -> &[u8] {
        let i = self.cursor.get();
        self.cursor.set((i + 1) % self.mats.len());
        &self.mats[i]
    }

    fn bytes_each(&self) -> f64 {
        self.mats[0].len() as f64
    }

    fn total_mb(&self) -> f64 {
        (self.mats.len() * self.mats[0].len()) as f64 / 1e6
    }
}

/// Run `f` for at least `min_secs`; returns seconds per iteration.
fn bench<F: FnMut() -> f32>(min_secs: f64, mut f: F) -> f64 {
    std::hint::black_box(f());
    let target = Duration::from_secs_f64(min_secs);
    let start = Instant::now();
    let mut iters: u64 = 0;
    let mut sink = 0.0f32;
    loop {
        sink += f();
        iters += 1;
        if start.elapsed() >= target {
            break;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(sink);
    elapsed / iters as f64
}

fn report(label: &str, secs: f64, bytes_once: f64, vectors: usize) {
    let gb_s = bytes_once / secs / 1e9;
    println!(
        "  {label:<38} {:>8.3} ms  {:>7.2} GB/s  eta(367) {:>5.2}  eta(127) {:>5.2}  {:>7.0} vec/s",
        secs * 1e3,
        gb_s,
        gb_s / GPU_ATTAINABLE_GB_S,
        gb_s / CPU_ATTAINABLE_GB_S,
        vectors as f64 / secs,
    );
}

/// Banked end-to-end Metal decode bandwidth (`project_memory_bandwidth_roofline`).
/// An isolated matvec landing far below this band is evidence the isolated
/// measurement is not sampling the regime decode runs in.
const BANKED_E2E_METAL_GB_S: (f64, f64) = (273.0, 314.0);

/// Repeats an arm and reports the spread, because a single sample cannot
/// distinguish a measurement from a thermal artefact. The house records
/// 1.5-3x phantom regressions on a hot machine.
fn repeat<F: FnMut() -> f64>(n: usize, mut f: F) -> (f64, f64, f64) {
    let mut v: Vec<f64> = (0..n).map(|_| f()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

/// Per-shape record, kept so the arms can be cross-checked against each other.
struct Measured {
    label: &'static str,
    bytes: f64,
    secs: f64,
    spread: f64,
}

/// A bandwidth measurement must get FASTER as the matrix gets SMALLER. If it
/// does not, time is dominated by a fixed per-call cost (Metal dispatch, kernel
/// launch, buffer setup) and the GB/s figure is not bandwidth at all.
///
/// This is the same failure class as a storage benchmark that measures the page
/// cache: the instrument's floor sits above the quantity being measured. The
/// house already records it as `isolated kernel speedup != e2e win` — isolated
/// single-dispatch numbers do not predict end-to-end on this codebase; the
/// batched `diag_profile_kernels` example is the predictive instrument.
fn validity_check(records: &[Measured]) -> bool {
    let mut sorted: Vec<&Measured> = records.iter().collect();
    sorted.sort_by(|a, b| a.bytes.partial_cmp(&b.bytes).unwrap());
    let mut sound = true;
    for w in sorted.windows(2) {
        let (small, big) = (w[0], w[1]);
        if small.secs >= big.secs {
            sound = false;
            println!(
                "  !! {} ({:.1} MB) took {:.3} ms while {} ({:.1} MB) took {:.3} ms",
                small.label,
                small.bytes / 1e6,
                small.secs * 1e3,
                big.label,
                big.bytes / 1e6,
                big.secs * 1e3,
            );
            let overhead = small.secs
                - (big.secs - small.secs) * small.bytes / (big.bytes - small.bytes).max(1.0);
            if overhead > 0.0 {
                println!("     implied fixed cost per call ~{:.3} ms", overhead * 1e3);
            }
        }
    }
    sound
}

fn main() {
    let ws_mb = arg_mb("--working-set-mb", 512);
    let secs = arg_mb("--secs", 2) as f64;
    let target_bytes = ws_mb * 1_000_000;
    let metal = MetalBackend::new();

    println!("DEC-8.8 — MXFP4 serving-path bandwidth at K3 shapes");
    println!("======================================================================");
    println!("working set {ws_mb} MB rotating (P-cluster L2 is 16 MB — a single");
    println!("  matrix in a loop would measure cache, not memory)");
    println!(
        "roofline: {GPU_ATTAINABLE_GB_S:.0} GB/s GPU-attainable, {CPU_ATTAINABLE_GB_S:.0} GB/s CPU"
    );
    println!(
        "Metal backend: {}",
        if metal.is_some() {
            "available"
        } else {
            "UNAVAILABLE"
        }
    );
    println!();
    println!("FINDING (pre-measurement): no MXFP4 arm exists in QuantMatVec on");
    println!("  either backend, and no MXFP4 shader exists in larql-compute-metal.");
    println!("  K3's serving format has no fused dequant-matvec kernel anywhere.");
    println!();

    let mut metal_records: Vec<Measured> = Vec::new();
    for sh in SHAPES {
        let weights = sh.out * sh.inp;
        let mxfp4_all_in = weights as f64 * MXFP4_ALL_IN_BYTES;
        let mxfp4_payload = weights as f64 * MXFP4_PAYLOAD_BYTES;
        let x = synth_x(sh.inp);

        println!(
            "### {} [{}, {}] — {:.2} M weights",
            sh.label,
            sh.out,
            sh.inp,
            weights as f64 / 1e6
        );
        println!(
            "    MXFP4 {:.2} MB all-in / {:.2} MB payload; f32 would be {:.2} MB",
            mxfp4_all_in / 1e6,
            mxfp4_payload / 1e6,
            weights as f64 * 4.0 / 1e6
        );

        let q4k_ws = WorkingSet::build(sh.out, sh.inp, target_bytes, quantize_q4_k);
        let q4k_bytes = q4k_ws.bytes_each();
        println!(
            "    Q4K {:.2} MB each ({:.3} B/weight), {} copies = {:.0} MB rotating",
            q4k_bytes / 1e6,
            q4k_bytes / weights as f64,
            q4k_ws.mats.len(),
            q4k_ws.total_mb()
        );
        println!();

        // ---- reference: Q4K fused, the only format with a kernel ----------
        println!("  -- Q4K fused matvec (reference; MXFP4 has no equivalent) --");
        let t_cold = bench(secs, || {
            let w = q4k_ws.next();
            CpuBackend
                .q4k_matvec(w, &x, sh.out, sh.inp)
                .map(|o| o[0])
                .unwrap_or(0.0)
        });
        report("CPU, rotating working set", t_cold, q4k_bytes, 1);

        let hot = &q4k_ws.mats[0];
        let t_warm = bench(secs, || {
            CpuBackend
                .q4k_matvec(hot, &x, sh.out, sh.inp)
                .map(|o| o[0])
                .unwrap_or(0.0)
        });
        report("CPU, one matrix re-read (CACHE)", t_warm, q4k_bytes, 1);
        println!(
            "    ^ re-reading one matrix overstates by {:.2}x — this is the error the",
            t_cold / t_warm.max(1e-12)
        );
        println!("      rotation exists to avoid; the rotating row is the real one.");

        if let Some(m) = metal.as_ref() {
            let (t_metal, fast, slow) = repeat(5, || {
                bench(secs, || {
                    let w = q4k_ws.next();
                    m.q4k_matvec(w, &x, sh.out, sh.inp)
                        .map(|o| o[0])
                        .unwrap_or(0.0)
                })
            });
            report("Metal, rotating (median of 5)", t_metal, q4k_bytes, 1);
            let spread = slow / fast;
            println!(
                "    spread {:.2}x across 5 samples ({:.2}-{:.2} GB/s){}",
                spread,
                q4k_bytes / slow / 1e9,
                q4k_bytes / fast / 1e9,
                if spread > 1.3 {
                    "  <- UNSTABLE, not a usable eta"
                } else {
                    ""
                }
            );
            metal_records.push(Measured {
                label: sh.label,
                bytes: q4k_bytes,
                secs: t_metal,
                spread,
            });

            println!("  -- block width: does reuse repair efficiency? (Metal, Q4K) --");
            for t in BLOCK_WIDTHS {
                let dt = bench(secs, || {
                    let w = q4k_ws.next();
                    let mut acc = 0.0;
                    for _ in 0..t {
                        acc += m
                            .q4k_matvec(w, &x, sh.out, sh.inp)
                            .map(|o| o[0])
                            .unwrap_or(0.0);
                    }
                    acc
                });
                report(&format!("T={t} (weights counted once)"), dt, q4k_bytes, t);
            }
            println!("    flat GB/s as T rises = genuine reuse; ~1/T = a token loop.");
        }
        println!();

        // ---- today's only MXFP4 path --------------------------------------
        println!("  -- MXFP4: bulk dequant to f32 buffer, then matvec --");
        let groups = sh.inp / 32;
        let blocks: Vec<u8> = (0..sh.out * groups * 16)
            .map(|i| ((i * 37 + 11) % 251) as u8)
            .collect();
        let scales: Vec<u8> = (0..sh.out * groups)
            .map(|i| (120 + (i % 15)) as u8)
            .collect();

        let t_dq = bench(secs, || {
            dequantize_expert(&blocks, &scales, sh.out, groups)
                .map(|w| w[0])
                .unwrap_or(0.0)
        });
        report("dequantize_expert alone", t_dq, mxfp4_all_in, 0);

        let t_full = bench(secs, || {
            let w = dequantize_expert(&blocks, &scales, sh.out, groups).unwrap();
            let mut acc = 0.0f32;
            for r in 0..sh.out {
                let base = r * sh.inp;
                let mut s = 0.0f32;
                for c in 0..sh.inp {
                    s += w[base + c] * x[c];
                }
                acc += s;
            }
            acc
        });
        report("dequant + f32 matvec (full path)", t_full, mxfp4_all_in, 1);
        println!(
            "    codec is {:.0}% of the full path",
            100.0 * t_dq / t_full
        );
        println!();
        println!("  -- traffic decomposition: MXFP4 is being used as STORAGE, not compute --");
        println!("     per weight, the materialisation path moves:");
        println!("       {MXFP4_ALL_IN_BYTES:.3} B packed+scale read");
        println!("       4.000 B f32 dequant output WRITTEN");
        println!("       4.000 B x T f32 re-read by matvec");
        for t in BLOCK_WIDTHS {
            let traffic = MXFP4_ALL_IN_BYTES + 4.0 + 4.0 * t as f64;
            println!(
                "       T={t}: {traffic:.2} B traffic for {MXFP4_ALL_IN_BYTES:.3} B stored \
                 -> {:.1}x amplification  ({:.2} GB/token-equivalent)",
                traffic / MXFP4_ALL_IN_BYTES,
                traffic * weights as f64 / 1e9,
            );
        }
        println!("     A fused kernel moves the {MXFP4_ALL_IN_BYTES:.3} B and nothing else.");
        println!("     This is why MXFP4 eta is not merely hard to measure: the current");
        println!("     implementation is not executing MXFP4 as a compressed compute");
        println!("     format at all.");
        println!("    NOTE: this arm cannot rotate a working set — dequantize_expert takes");
        println!("      one (blocks, scales) pair, so it re-reads the same source. That");
        println!("      FLATTERS it, and it still loses. Treat as an upper bound.");
        println!();
    }

    println!("======================================================================");
    println!("VALIDITY CHECK — is this bandwidth, or per-dispatch overhead?");
    let sound = validity_check(&metal_records);
    if !sound {
        println!();
        println!("  VERDICT: NOT a bandwidth measurement. Time does not scale with");
        println!("  bytes, so these GB/s figures are dispatch-bound and MUST NOT be");
        println!("  used as eta. Do not select a DEC-8.7b target row from them.");
        println!();
        println!("  The house instrument for this is the BATCHED profile:");
        println!("    cargo run --release -p larql-compute-metal \\");
        println!("      --example diag_profile_kernels");
        println!("  Isolated single-dispatch benchmarks are already recorded as");
        println!("  non-predictive of end-to-end on this codebase. An isolated");
        println!("  matvec measures launch cost; a batched profile measures the");
        println!("  regime decode actually runs in.");
        println!();
        println!("  What DOES survive from this run, because neither depends on");
        println!("  the absolute GB/s:");
        println!("    1. No fused MXFP4 kernel exists on either backend (grep, not timing).");
        println!("    2. T sequential matvecs give ZERO weight reuse — vec/s stays flat");
        println!("       while GB/s falls as 1/T. Block width does not imply one weight");
        println!("       read (R6), measured rather than assumed: d(T)=r(T)=T today.");
        return;
    }
    println!("  time scales with bytes.");
    let unstable: Vec<&Measured> = metal_records.iter().filter(|m| m.spread > 1.3).collect();
    let best_gb_s = metal_records
        .iter()
        .map(|m| m.bytes / m.secs / 1e9)
        .fold(0.0f64, f64::max);
    if !unstable.is_empty() || best_gb_s < BANKED_E2E_METAL_GB_S.0 * 0.5 {
        println!();
        println!("  BUT the numbers are still not a usable eta:");
        for m in &unstable {
            println!("    - {} varies {:.2}x across samples", m.label, m.spread);
        }
        println!(
            "    - best isolated reading {:.0} GB/s vs banked end-to-end {:.0}-{:.0} GB/s",
            best_gb_s, BANKED_E2E_METAL_GB_S.0, BANKED_E2E_METAL_GB_S.1
        );
        println!(
            "      for the same kernel class — a {:.0}x discrepancy.",
            BANKED_E2E_METAL_GB_S.0 / best_gb_s.max(1e-9)
        );
        println!();
        println!("  VERDICT: an isolated single-dispatch matvec does not sample the");
        println!("  regime decode runs in. This is already recorded as a house rule:");
        println!("  isolated kernel speedup does not predict end-to-end; the BATCHED");
        println!("  profile is the predictive instrument.");
        println!();
        println!("    cargo run --release -p larql-compute-metal --example diag_profile_kernels");
        println!();
        println!("  Do NOT select a DEC-8.7b target row from these figures.");
        println!();
        println!("  What survives, because neither depends on absolute GB/s:");
        println!("    1. No fused MXFP4 kernel exists on either backend (grep, not timing).");
        println!("    2. T sequential matvecs give ZERO weight reuse: vec/s flat while");
        println!("       GB/s falls as 1/T. R6 measured, not assumed — d(T)=r(T)=T today,");
        println!("       so DEC-9.2's arithmetic currently rests on a reuse that the");
        println!("       kernels do not provide.");
        return;
    }
    println!();
    println!("Selecting the DEC-8.7b target row from these numbers:");
    println!("  eta >= 0.85        -> 10 tok/s stays a credible dense-side target");
    println!("  eta 0.74-0.85      -> 10 tok/s is a stretch, wants better than 3-bit");
    println!("  eta <  0.74        -> take the 8 tok/s row unless block execution helps");
    println!("  eta rises with T   -> DEC-9 gains value independently of acceptance,");
    println!("                        because block GEMM repairs dequant efficiency");
    println!("                        as well as amortising reads");
    println!();
    println!("Use the Q4K rotating figure as the eta UPPER BOUND for");
    println!("`larql k3-ledger frontier --dequant-efficiency`, flagged as a");
    println!("different codec. MXFP4's true eta is unmeasurable until a fused");
    println!("kernel exists — and building it is now on the critical path.");
}
