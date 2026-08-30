//! CPU-3A control: does a bench of the SHIPPED kernel reproduce what the
//! model actually spends?
//!
//! Nothing about a Q8 kernel can be believed until this passes. CPU-2D
//! spent a rung on a scratch recurrence that ran 294 us against the
//! model's 617 — a convincing-looking harness measuring a different
//! program — and its speedup ratios licensed nothing. Re-running the same
//! comparison inside the real binary reproduced 594 us against 617, and
//! only then did the result mean anything.
//!
//! So this bench runs `FusedBf16` through `CpuExecutor`, over Qwen3.8's
//! own projection shapes weighted by the call counts one token actually
//! issues, and compares the total against the projection class the leaf
//! ledger measures on the real model:
//!
//! ```text
//!   Projection   497 calls   436.60 ms   51.29 GB
//! ```
//!
//! A new kernel is compared against THIS number, not against a number
//! from a different harness.
//!
//! ```text
//! QW_PROJ_COST=1 cargo test --release exec::cpu::tests::projection_cost -- --nocapture
//! ```

use super::super::executor::CpuExecutor;
use super::super::kernels::{BlasF32, FusedBf16};
use super::super::projector::WeightRows;
use crate::format::vindex3::fixtures::lcg_values;

/// Every projection a Qwen3.8 token issues: `(name, out_dim, in_dim,
/// calls, compact)`.
///
/// Counts are the ledger's, post-CPU-2C: 401 compact calls over 51.20 GB
/// and 96 tiny f32 calls over 0.09 GB. `q_proj` is 12288 rows because it
/// is the FUSED query/gate projection and one call now serves both.
pub(super) const TOKEN: &[(&str, usize, usize, usize, bool)] = &[
    ("mlp.gate_proj", 17408, 5120, 64, true),
    ("mlp.up_proj", 17408, 5120, 64, true),
    ("mlp.down_proj", 5120, 17408, 64, true),
    ("delta.in_proj_qkv", 10240, 5120, 48, true),
    ("delta.in_proj_z", 6144, 5120, 48, true),
    ("delta.out_proj", 5120, 6144, 48, true),
    ("attn.q_proj (fused)", 12288, 5120, 16, true),
    ("attn.k_proj", 1024, 5120, 16, true),
    ("attn.v_proj", 1024, 5120, 16, true),
    ("attn.o_proj", 5120, 6144, 16, true),
    ("output_head", 248320, 5120, 1, true),
    ("delta.in_proj_a", 48, 5120, 48, false),
    ("delta.in_proj_b", 48, 5120, 48, false),
];

/// The compact projections alone: `(name, out_dim, in_dim, calls)`.
///
/// Shared with the Q8 bench so both describe the SAME token. Two tables
/// would drift, and the second one to drift would be comparing formats
/// over different work.
pub(super) const COMPACT: &[(&str, usize, usize, usize)] = &[
    ("mlp.gate_proj", 17408, 5120, 64),
    ("mlp.up_proj", 17408, 5120, 64),
    ("mlp.down_proj", 5120, 17408, 64),
    ("delta.in_proj_qkv", 10240, 5120, 48),
    ("delta.in_proj_z", 6144, 5120, 48),
    ("delta.out_proj", 5120, 6144, 48),
    ("attn.q_proj (fused)", 12288, 5120, 16),
    ("attn.k_proj", 1024, 5120, 16),
    ("attn.v_proj", 1024, 5120, 16),
    ("attn.o_proj", 5120, 6144, 16),
    ("output_head", 248320, 5120, 1),
];

/// What the leaf ledger measures for the projection class on the real
/// model, so the control has something to miss.
const MODEL_PROJECTION_MS: f64 = 436.60;
const MODEL_PROJECTION_CALLS: usize = 497;
const MODEL_PROJECTION_GB: f64 = 51.29;

fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

#[test]
fn the_bench_reproduces_the_models_projection_cost() {
    if std::env::var("QW_PROJ_COST").is_err() {
        eprintln!("SKIP projection_cost: set QW_PROJ_COST=1");
        return;
    }
    use std::time::Instant;
    let exec = CpuExecutor::new().unwrap();
    println!(
        "\n  Shipped BF16 kernel, {} workers, over one token's projections\n",
        exec.workers()
    );
    println!(
        "  {:<22} {:>6} {:>8} {:>7} {:>11} {:>11}",
        "projection", "calls", "MB each", "ms each", "ms/token", "GB/token"
    );

    let (mut total_ms, mut total_gb, mut total_calls) = (0.0f64, 0.0f64, 0usize);
    for (name, out_dim, in_dim, calls, compact) in TOKEN.iter().copied() {
        let f32w = lcg_values(out_dim * in_dim, 11);
        let bf: Vec<u16> = f32w.iter().map(|v| narrow(*v)).collect();
        let x = lcg_values(in_dim, 22);
        let bytes = (out_dim * in_dim * if compact { 2 } else { 4 }) as f64;
        // Enough passes to cross the allocation once at least, capped so
        // the 2.5 GB head does not dominate the run.
        let iters = (3_000_000_000.0 / (out_dim * in_dim) as f64).clamp(3.0, 100.0) as usize;

        let mut sink = 0.0f32;
        let run = |sink: &mut f32| {
            if compact {
                *sink += exec.project(&FusedBf16, WeightRows::Bf16(&bf), &x, out_dim)[0];
            } else {
                *sink += exec.project(&BlasF32, WeightRows::F32(&f32w), &x, out_dim)[0];
            }
        };
        run(&mut sink);
        let t = Instant::now();
        for _ in 0..iters {
            run(&mut sink);
        }
        let each = t.elapsed().as_secs_f64() / iters as f64;
        std::hint::black_box(sink);

        let ms = each * calls as f64 * 1e3;
        let gb = bytes * calls as f64 / 1e9;
        total_ms += ms;
        total_gb += gb;
        total_calls += calls;
        println!(
            "  {name:<22} {calls:>6} {:>8.1} {:>7.3} {:>11.2} {:>11.2}",
            bytes / 1e6,
            each * 1e3,
            ms,
            gb
        );
    }

    println!("  {:-<72}", "");
    println!(
        "  {:<22} {total_calls:>6} {:>17.2} {:>11.2}",
        "bench total", total_ms, total_gb
    );
    println!(
        "  {:<22} {MODEL_PROJECTION_CALLS:>6} {:>17.2} {:>11.2}   <- leaf ledger, real model",
        "model", MODEL_PROJECTION_MS, MODEL_PROJECTION_GB
    );
    let error = (total_ms - MODEL_PROJECTION_MS) / MODEL_PROJECTION_MS * 100.0;
    println!("\n  control error {error:+.1}%");

    assert_eq!(
        total_calls, MODEL_PROJECTION_CALLS,
        "the shape table does not describe the token the model decodes"
    );
    assert!(
        (total_gb - MODEL_PROJECTION_GB).abs() < 0.05,
        "bench streams {total_gb:.2} GB against the model's {MODEL_PROJECTION_GB:.2}"
    );
    // Deliberately loose: the bench loops one shape at a time, so a
    // 10.5 MB `k_proj` stays L2-resident here and does not in a decode
    // that streams 51 GB between two touches of it. The control is
    // asking whether this harness measures THE SAME PROGRAM, not whether
    // it reproduces the cache state — and a harness 2x out, as CPU-2D's
    // scratch transcription was, fails this by a mile.
    assert!(
        error.abs() < 25.0,
        "the bench does not reproduce the model's projection cost ({error:+.1}%), so no ratio \
         measured against it licenses a claim about LARQL"
    );
}

/// The two tables describe one token.
///
/// `COMPACT` exists so the Q8 bench compares formats over the same work
/// this control validated; if it ever stopped being the compact subset of
/// `TOKEN`, the two benches would be measuring different models.
#[test]
fn the_compact_table_is_the_compact_subset_of_the_token() {
    let from_token: Vec<_> = TOKEN
        .iter()
        .filter(|(_, _, _, _, compact)| *compact)
        .map(|(n, o, i, c, _)| (*n, *o, *i, *c))
        .collect();
    assert_eq!(from_token, COMPACT.to_vec());
}
