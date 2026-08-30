//! K2 — the exact MXFP4 layout/decode tournament.
//!
//! Four arms, crossed so no comparison confounds two changes:
//!
//! ```text
//!   B - A   interleaved adaptive scales vs the checkpoint's separate stream
//!   C - B   256-entry byte-pair LUT vs the 16-entry nibble LUT
//!   D - B   8-entry magnitude LUT + explicit sign
//!   C - D   full lookup vs smaller table with sign arithmetic
//! ```
//!
//! Arms B/C/D read a byte-identical artifact, so C-B and D-B are pure decode
//! effects. Only A changes the bytes, so B-A is the layout effect. A three-arm
//! tournament without B would have been uninterpretable.
//!
//! **Actual kernel milliseconds decide.** `bpw/eta` is explanatory only: the
//! arms deliberately read different amounts (4.25 vs 4.0625 bpw), so a GB/s
//! comparison across the A boundary flatters whichever arm reads more.
//!
//! Bit-width labels, kept distinct on purpose:
//!   - **4.0625 bpw** — the layout benchmarked here: a fixed 1-bit exponent
//!     delta per group, which covers the 97.12% of real K3 superblocks whose
//!     spread is <= 1. Common path only.
//!   - **4.06731 bpw** — the whole-checkpoint exact format, which additionally
//!     prices the 2-bit delta for the 2.88% wide tail plus a per-superblock mode
//!     bit. That tail, not any global scale table, is the entire difference.
//!
//! Usage:
//!   cargo run --release -p larql-compute-metal --example bench_mxfp4_layouts [N K TOP_K]
//!
//! Defaults are the K3 expert branch; `5760 2880 4` and `2880 2880 4` are
//! gpt-oss-20b's fused gate/up and down shapes.

extern crate blas_src;

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Requires macOS + Metal");
}

#[cfg(target_os = "macos")]
fn main() {
    use larql_compute_metal::diag::mxfp4_layout;

    let mut args = std::env::args().skip(1);
    let mut dim = |default: usize| {
        args.next()
            .map(|a| a.parse().expect("N K TOP_K must be integers"))
            .unwrap_or(default)
    };
    let n: usize = dim(3584); // K3 expert branch rows
    let k: usize = dim(3072); // latent reduction
    let top_k: usize = dim(16);
    #[allow(non_snake_case)]
    let (N, K, TOP_K) = (n, k, top_k);
    const ROOFLINE: f64 = 367.0;
    const BATCH: usize = 8;
    const WARMUP: usize = 3;
    const ITERS: usize = 20;

    println!("K2 — MXFP4 exact layout tournament, [{N}, {K}] x {TOP_K} grouped experts");
    println!("=========================================================================");
    println!("  grouped dispatch, batched {BATCH}/cmdbuf, cold (payloads >> 16 MB L2)");
    println!();

    let arms = mxfp4_layout::race(N, K, TOP_K, BATCH, WARMUP, ITERS);
    let base = &arms[0];

    println!(
        "{:<20} {:>7} {:>9} {:>10} {:>7} {:>9} {:>9} {:>9}",
        "arm", "bpw", "MB read", "ms", "eta", "vs A", "oracle", "vs-A diff"
    );
    for a in &arms {
        println!(
            "{:<20} {:>7.4} {:>9.1} {:>10.4} {:>7.3} {:>8.3}x {:>9.2e} {:>9.2e}",
            a.name,
            a.bpw,
            a.bytes / 1e6,
            a.ms,
            a.eta(ROOFLINE),
            base.ms / a.ms,
            a.max_abs_diff,
            a.cross_diff
        );
        if let Some(i) = a.first_bad {
            println!("      ^ first disagreement with A at output index {i}");
        }
    }
    println!();

    let get = |i: usize| &arms[i];
    let (a, a2, b, c, d, g, e, f) = (
        get(0),
        get(1),
        get(2),
        get(3),
        get(4),
        get(5),
        get(6),
        get(7),
    );
    println!("--- the pre-registered comparisons ---");
    {
        let pct = 100.0 * (a.ms / a2.ms - 1.0);
        println!(
            "  {:<28} {:>+7.2}%   ({:.4} -> {:.4} ms)",
            "A2 - A vectorised skeleton", pct, a.ms, a2.ms
        );
    }
    let cmp = |label: &str, from: &mxfp4_layout::ArmResult, to: &mxfp4_layout::ArmResult| {
        let pct = 100.0 * (from.ms / to.ms - 1.0);
        println!(
            "  {label:<28} {:>+7.2}%   ({:.4} -> {:.4} ms)",
            pct, from.ms, to.ms
        );
    };
    cmp("B - A  layout + scale", a, b);
    cmp("C - B  byte-pair LUT", b, c);
    cmp("D - B  magnitude LUT", b, d);
    cmp("C - D  full vs small table", d, c);
    cmp("G - D  table-free bit math", d, g);
    println!();

    println!("--- ceiling probes: is the gap decode ALU or the skeleton? ---");
    cmp("E - D  fp4 grid vs affine", d, e);
    cmp("F - E  X gather + FMA chain", e, f);
    println!();
    println!("  D is the real kernel. E keeps every byte read and every X read but");
    println!("  swaps the non-uniform fp4 grid for a trivial affine map, so E-D is");
    println!("  the price of the value set. F drops the X gather entirely, so F-E is");
    println!("  the price of the input path. Whatever remains in F is the skeleton:");
    println!("  weight streaming, loop and reduction.");
    let floor = f.ms;
    println!();
    println!(
        "  skeleton floor (F)      {:.4} ms  = {:.0}% of D",
        floor,
        100.0 * floor / d.ms
    );
    println!(
        "  + fp4 decode (D - E)    {:.4} ms  = {:.0}% of D",
        d.ms - e.ms,
        100.0 * (d.ms - e.ms) / d.ms
    );
    println!(
        "  + X gather   (E - F)    {:.4} ms  = {:.0}% of D",
        e.ms - floor,
        100.0 * (e.ms - floor) / d.ms
    );
    println!();

    // Correctness gate: the arms encode identical values, so any real
    // disagreement means an arm is not doing the same work.
    let worst = arms
        .iter()
        .filter(|a| a.checked)
        .fold(0.0f32, |m, a| m.max(a.max_abs_diff).max(a.cross_diff));
    println!("--- gates ---");
    if worst < 1e-3 {
        println!("  outputs agree across all arms (max abs diff {worst:.2e}) — same work");
    } else {
        println!("  !! arms DISAGREE (max abs diff {worst:.2e}) — timings are not comparable");
    }
    let best = arms
        .iter()
        .filter(|a| a.checked)
        .min_by(|x, y| x.ms.partial_cmp(&y.ms).unwrap())
        .unwrap();
    println!(
        "  fastest: {} at {:.4} ms (eta {:.3})",
        best.name,
        best.ms,
        best.eta(ROOFLINE)
    );
    if best.bpw > 4.0625 {
        println!(
            "  NOTE the fastest arm is also the largest — bytes are not the binding cost here."
        );
    }
    println!();
    println!("  Interleaving is available independently of any decode change, so");
    println!("  a C or D win only justifies the LUT if it beats B, not just A.");
}
